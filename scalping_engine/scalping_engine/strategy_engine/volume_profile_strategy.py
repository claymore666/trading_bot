import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class VolumeProfileStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf Volume Profile (Volumen-Profil).
    Diese Strategie identifiziert wichtige Preiszonen mit hohem Handelsvolumen und
    handelt auf Basis von Ausbrüchen aus oder Abprallern von diesen Zonen.
    """
    
    def __init__(
        self,
        name: str = "Volume Profile Strategy",
        description: str = "Handelt auf Basis von Volumen-Profil und Value Areas",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Volume-Profile-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'profile_period': 100,            # Anzahl der Kerzen für das Volumen-Profil
            'price_levels': 50,               # Anzahl der Preislevels für das Histogramm
            'value_area_volume_percent': 70,  # Prozentsatz des Volumens für die Value Area (typisch: 70%)
            'signal_type': 'value_area',      # Signaltyp: 'value_area', 'volume_node', 'poc'
            'poc_distance_filter': 0.5,       # Minimaler Abstand vom aktuellen Preis zum POC in Prozent
            'volume_node_significance': 2.0,  # Signifikanz eines Volume Node im Vergleich zum Durchschnitt
            'entry_condition': 'bounce',      # Einstiegsbedingung: 'bounce', 'breakout'
            'poc_lookback': 3,                # Anzahl der POCs aus vorherigen Perioden für Unterstützung/Widerstand
            'volume_filter': True,            # Volumen-Filter verwenden
            'trend_filter': True,             # Trend-Filter verwenden
            'trend_period': 50,               # Periode für Trend-Bestimmung
            'atr_period': 14,                 # ATR-Periode für Volatilitätsberechnung
            'atr_multiplier': 0.5,            # ATR-Multiplikator für Einstiegs- und Stop-Loss-Berechnung
            'confirmation_candles': 1,        # Anzahl der Bestätigungskerzen für Signale
            'max_vp_profiles': 3,             # Maximale Anzahl an gespeicherten VP-Profilen
            'refresh_interval': 20,           # Intervall in Perioden für die Neuberechnung des Volumen-Profils
            'volume_profile_method': 'fixed',  # Methode: 'fixed' (festes Zeitfenster) oder 'rolling' (gleitendes Fenster)
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
        
        # Interne Variablen für Volume Profile
        self.volume_profiles = []  # Liste von Volume Profiles
        self.last_profile_calc = 0  # Index der letzten Profilberechnung
    
    def _calculate_volume_profile(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Dict[str, Any]:
        """
        Berechnet das Volumen-Profil für einen bestimmten Zeitraum.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            start_idx: Startindex für die Berechnung
            end_idx: Endindex für die Berechnung
            
        Returns:
            Dict[str, Any]: Volumen-Profil mit POC, Value Area und Volumen-Nodes
        """
        # Bereich der Daten extrahieren
        data_range = data.iloc[start_idx:end_idx + 1]
        
        if data_range.empty:
            logger.warning(f"Leerer Datenbereich für Volumen-Profil: {start_idx}-{end_idx}")
            return {}
        
        # Preisbereich bestimmen
        price_high = data_range['high'].max()
        price_low = data_range['low'].min()
        
        # Sicherstellen, dass wir einen gültigen Preisbereich haben
        if price_high <= price_low:
            logger.warning(f"Ungültiger Preisbereich für Volumen-Profil: {price_low}-{price_high}")
            return {}
        
        # Preislevels erstellen
        num_levels = self.parameters['price_levels']
        price_delta = (price_high - price_low) / num_levels
        price_levels = [price_low + i * price_delta for i in range(num_levels + 1)]
        
        # Volume-Profil-Histogramm initialisieren
        volume_histogram = np.zeros(num_levels)
        
        # Für jede Kerze das Volumen auf die Preislevels verteilen
        for _, row in data_range.iterrows():
            candle_high = row['high']
            candle_low = row['low']
            candle_volume = row['volume']
            
            # Bestimmen, welche Preislevels von der Kerze abgedeckt werden
            for i in range(num_levels):
                level_low = price_levels[i]
                level_high = price_levels[i + 1]
                
                # Prüfen, ob die Kerze das Preislevel abdeckt
                if candle_low <= level_high and candle_high >= level_low:
                    # Überlappungsbereich berechnen
                    overlap_low = max(candle_low, level_low)
                    overlap_high = min(candle_high, level_high)
                    overlap_ratio = (overlap_high - overlap_low) / (candle_high - candle_low)
                    
                    # Volumen anteilig zuweisen
                    volume_histogram[i] += candle_volume * overlap_ratio
        
        # Point of Control (POC) - Preislevel mit dem höchsten Volumen
        poc_index = np.argmax(volume_histogram)
        poc_price = (price_levels[poc_index] + price_levels[poc_index + 1]) / 2
        
        # Value Area - Preisbereich, der x% des Volumens enthält
        value_area_percent = self.parameters['value_area_volume_percent'] / 100
        total_volume = np.sum(volume_histogram)
        target_volume = total_volume * value_area_percent
        
        # Value Area von POC aus berechnen (in beide Richtungen)
        volume_sum = volume_histogram[poc_index]
        va_upper_idx = poc_index
        va_lower_idx = poc_index
        
        while volume_sum < target_volume and (va_upper_idx < num_levels - 1 or va_lower_idx > 0):
            # Entscheiden, ob nach oben oder unten erweitern
            if va_upper_idx < num_levels - 1 and va_lower_idx > 0:
                if volume_histogram[va_upper_idx + 1] >= volume_histogram[va_lower_idx - 1]:
                    va_upper_idx += 1
                    volume_sum += volume_histogram[va_upper_idx]
                else:
                    va_lower_idx -= 1
                    volume_sum += volume_histogram[va_lower_idx]
            elif va_upper_idx < num_levels - 1:
                va_upper_idx += 1
                volume_sum += volume_histogram[va_upper_idx]
            elif va_lower_idx > 0:
                va_lower_idx -= 1
                volume_sum += volume_histogram[va_lower_idx]
            else:
                break
        
        value_area_high = price_levels[va_upper_idx + 1]
        value_area_low = price_levels[va_lower_idx]
        
        # Volumen-Nodes (signifikante Volumenspitzen) finden
        avg_volume = np.mean(volume_histogram)
        significance = self.parameters['volume_node_significance']
        volume_nodes = []
        
        for i in range(num_levels):
            if volume_histogram[i] > avg_volume * significance:
                node_price = (price_levels[i] + price_levels[i + 1]) / 2
                volume_nodes.append({
                    'price': node_price,
                    'volume': volume_histogram[i],
                    'relative_volume': volume_histogram[i] / avg_volume
                })
        
        # Volumen-Profil zusammenstellen
        volume_profile = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': data.index[start_idx],
            'end_time': data.index[end_idx],
            'price_high': price_high,
            'price_low': price_low,
            'price_levels': price_levels,
            'volume_histogram': volume_histogram.tolist(),
            'poc': {
                'price': poc_price,
                'index': poc_index,
                'volume': volume_histogram[poc_index]
            },
            'value_area': {
                'high': value_area_high,
                'low': value_area_low,
                'volume_percent': self.parameters['value_area_volume_percent']
            },
            'volume_nodes': volume_nodes
        }
        
        return volume_profile
    
    def _update_volume_profiles(self, data: pd.DataFrame, current_idx: int) -> None:
        """
        Aktualisiert die Volume Profiles basierend auf neuen Daten.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            current_idx: Aktueller Index im DataFrame
        """
        profile_period = self.parameters['profile_period']
        refresh_interval = self.parameters['refresh_interval']
        max_profiles = self.parameters['max_vp_profiles']
        method = self.parameters['volume_profile_method']
        
        # Prüfen, ob eine Aktualisierung erforderlich ist
        if current_idx < profile_period:
            return
        
        if method == 'fixed':
            # Festes Zeitfenster für jedes Profil
            # Aktualisieren wir nur, wenn genug neue Daten vorhanden sind oder das erste Profil
            periods_since_last_calc = current_idx - self.last_profile_calc
            
            if len(self.volume_profiles) == 0 or periods_since_last_calc >= refresh_interval:
                # Calculate a new profile
                start_idx = max(0, current_idx - profile_period + 1)
                end_idx = current_idx
                
                profile = self._calculate_volume_profile(data, start_idx, end_idx)
                
                if profile:
                    # Neues Profil hinzufügen
                    self.volume_profiles.append(profile)
                    
                    # Älteste Profile entfernen, wenn das Maximum erreicht ist
                    if len(self.volume_profiles) > max_profiles:
                        self.volume_profiles.pop(0)
                    
                    self.last_profile_calc = current_idx
                    logger.debug(f"Neues Volumen-Profil berechnet für Bereich {start_idx}-{end_idx}")
        
        elif method == 'rolling':
            # Gleitendes Fenster, aktualisieren wir bei jedem Aufruf
            if current_idx % refresh_interval == 0:
                start_idx = max(0, current_idx - profile_period + 1)
                end_idx = current_idx
                
                profile = self._calculate_volume_profile(data, start_idx, end_idx)
                
                if profile:
                    # Neues Profil hinzufügen
                    self.volume_profiles.append(profile)
                    
                    # Älteste Profile entfernen, wenn das Maximum erreicht ist
                    if len(self.volume_profiles) > max_profiles:
                        self.volume_profiles.pop(0)
                    
                    self.last_profile_calc = current_idx
                    logger.debug(f"Neues Volumen-Profil berechnet für Bereich {start_idx}-{end_idx}")
    
    def _get_volume_levels(self, data: pd.DataFrame, current_idx: int) -> Dict[str, List[float]]:
        """
        Gibt die wichtigen Volumen-Levels für den aktuellen Zeitpunkt zurück.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            current_idx: Aktueller Index im DataFrame
            
        Returns:
            Dict[str, List[float]]: Wichtige Volumen-Levels (POCs, Value Areas, Volumen-Nodes)
        """
        if not self.volume_profiles:
            return {'pocs': [], 'value_area_highs': [], 'value_area_lows': [], 'volume_nodes': []}
        
        # POCs extrahieren
        pocs = [profile['poc']['price'] for profile in self.volume_profiles]
        
        # Value Areas extrahieren
        value_area_highs = [profile['value_area']['high'] for profile in self.volume_profiles]
        value_area_lows = [profile['value_area']['low'] for profile in self.volume_profiles]
        
        # Volumen-Nodes extrahieren
        volume_nodes = []
        for profile in self.volume_profiles:
            for node in profile['volume_nodes']:
                volume_nodes.append(node['price'])
        
        return {
            'pocs': pocs,
            'value_area_highs': value_area_highs,
            'value_area_lows': value_area_lows,
            'volume_nodes': volume_nodes
        }
    
    def _is_price_near_level(self, price: float, level: float, atr: float) -> bool:
        """
        Prüft, ob ein Preis nahe an einem bestimmten Level ist.
        
        Args:
            price: Aktueller Preis
            level: Preislevel
            atr: Average True Range
            
        Returns:
            bool: True, wenn der Preis nahe am Level ist
        """
        distance = abs(price - level)
        atr_multiplier = self.parameters['atr_multiplier']
        
        return distance <= atr * atr_multiplier
    
    def _is_price_crossing_level(
        self, current_price: float, previous_price: float, level: float
    ) -> Tuple[bool, str]:
        """
        Prüft, ob ein Preis ein Level kreuzt.
        
        Args:
            current_price: Aktueller Preis
            previous_price: Vorheriger Preis
            level: Preislevel
            
        Returns:
            Tuple[bool, str]: (Kreuzung ja/nein, Richtung der Kreuzung)
        """
        if previous_price <= level and current_price > level:
            return True, 'up'
        elif previous_price >= level and current_price < level:
            return True, 'down'
        else:
            return False, ''
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf dem Volumen-Profil.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        signal_type = self.parameters['signal_type']
        entry_condition = self.parameters['entry_condition']
        trend_filter = self.parameters['trend_filter']
        trend_period = self.parameters['trend_period']
        volume_filter = self.parameters['volume_filter']
        confirmation_candles = self.parameters['confirmation_candles']
        
        # ATR berechnen, falls nicht vorhanden
        if 'atr' not in data.columns:
            atr_period = self.parameters['atr_period']
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Trend-Filter berechnen, falls aktiviert
        if trend_filter:
            data['trend_ma'] = data['close'].rolling(window=trend_period).mean()
            data['trend'] = 0
            data.loc[data['close'] > data['trend_ma'], 'trend'] = 1
            data.loc[data['close'] < data['trend_ma'], 'trend'] = -1
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Volume Profiles regelmäßig aktualisieren
        profile_period = self.parameters['profile_period']
        
        # Für jeden Datenpunkt prüfen, ob ein Signal generiert werden soll
        for i in range(profile_period, len(data)):
            # Volume Profiles aktualisieren
            self._update_volume_profiles(data, i)
            
            # Wenn keine Profile vorhanden sind, kein Signal generieren
            if not self.volume_profiles:
                continue
            
            # Wichtige Volumen-Levels abrufen
            volume_levels = self._get_volume_levels(data, i)
            current_price = data['close'].iloc[i]
            current_atr = data['atr'].iloc[i]
            
            # Signal basierend auf dem gewählten Typ generieren
            if signal_type == 'poc':
                # Point of Control (POC) - handel auf dem Niveau mit dem höchsten Volumen
                pocs = volume_levels['pocs']
                
                for poc in pocs:
                    if entry_condition == 'bounce':
                        # Einstieg bei Abpraller vom POC
                        if self._is_price_near_level(current_price, poc, current_atr):
                            # Abpraller nach oben (Long)
                            if (data['close'].iloc[i] > data['open'].iloc[i] and 
                                data['low'].iloc[i] < poc and data['close'].iloc[i] > poc):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = 1
                                    logger.debug(f"POC-Bounce-Long bei {data.index[i]}, Preis: {current_price}, POC: {poc}")
                            
                            # Abpraller nach unten (Short)
                            elif (data['close'].iloc[i] < data['open'].iloc[i] and 
                                  data['high'].iloc[i] > poc and data['close'].iloc[i] < poc):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = -1
                                    logger.debug(f"POC-Bounce-Short bei {data.index[i]}, Preis: {current_price}, POC: {poc}")
                    
                    elif entry_condition == 'breakout':
                        # Einstieg bei Ausbruch über/unter POC
                        if i > 0:
                            prev_price = data['close'].iloc[i-1]
                            crossing, direction = self._is_price_crossing_level(current_price, prev_price, poc)
                            
                            if crossing:
                                # Ausbruch nach oben (Long)
                                if direction == 'up':
                                    # Zusätzliche Filter anwenden
                                    if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                        data.loc[data.index[i], 'signal'] = 1
                                        logger.debug(f"POC-Breakout-Long bei {data.index[i]}, Preis: {current_price}, POC: {poc}")
                                
                                # Ausbruch nach unten (Short)
                                elif direction == 'down':
                                    # Zusätzliche Filter anwenden
                                    if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                        data.loc[data.index[i], 'signal'] = -1
                                        logger.debug(f"POC-Breakout-Short bei {data.index[i]}, Preis: {current_price}, POC: {poc}")
            
            elif signal_type == 'value_area':
                # Value Area - handel an den Grenzen der Value Area
                value_area_highs = volume_levels['value_area_highs']
                value_area_lows = volume_levels['value_area_lows']
                
                for va_high in value_area_highs:
                    if entry_condition == 'bounce':
                        # Abpraller von Value Area High nach unten (Short)
                        if self._is_price_near_level(current_price, va_high, current_atr):
                            if (data['close'].iloc[i] < data['open'].iloc[i] and 
                                data['high'].iloc[i] > va_high and data['close'].iloc[i] < va_high):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = -1
                                    logger.debug(f"VA-High-Bounce-Short bei {data.index[i]}, Preis: {current_price}, VA-High: {va_high}")
                    
                    elif entry_condition == 'breakout':
                        # Ausbruch über Value Area High (Long)
                        if i > 0:
                            prev_price = data['close'].iloc[i-1]
                            crossing, direction = self._is_price_crossing_level(current_price, prev_price, va_high)
                            
                            if crossing and direction == 'up':
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = 1
                                    logger.debug(f"VA-High-Breakout-Long bei {data.index[i]}, Preis: {current_price}, VA-High: {va_high}")
                
                for va_low in value_area_lows:
                    if entry_condition == 'bounce':
                        # Abpraller von Value Area Low nach oben (Long)
                        if self._is_price_near_level(current_price, va_low, current_atr):
                            if (data['close'].iloc[i] > data['open'].iloc[i] and 
                                data['low'].iloc[i] < va_low and data['close'].iloc[i] > va_low):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = 1
                                    logger.debug(f"VA-Low-Bounce-Long bei {data.index[i]}, Preis: {current_price}, VA-Low: {va_low}")
                    
                    elif entry_condition == 'breakout':
                        # Ausbruch unter Value Area Low (Short)
                        if i > 0:
                            prev_price = data['close'].iloc[i-1]
                            crossing, direction = self._is_price_crossing_level(current_price, prev_price, va_low)
                            
                            if crossing and direction == 'down':
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = -1
                                    logger.debug(f"VA-Low-Breakout-Short bei {data.index[i]}, Preis: {current_price}, VA-Low: {va_low}")
            
            elif signal_type == 'volume_node':
                # Volume Node - handel an signifikanten Volumenspitzen
                volume_nodes = volume_levels['volume_nodes']
                
                for node in volume_nodes:
                    if entry_condition == 'bounce':
                        # Abpraller von Volume Node
                        if self._is_price_near_level(current_price, node, current_atr):
                            # Abpraller nach oben (Long)
                            if (data['close'].iloc[i] > data['open'].iloc[i] and 
                                data['low'].iloc[i] < node and data['close'].iloc[i] > node):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = 1
                                    logger.debug(f"VolumeNode-Bounce-Long bei {data.index[i]}, Preis: {current_price}, Node: {node}")
                            
                            # Abpraller nach unten (Short)
                            elif (data['close'].iloc[i] < data['open'].iloc[i] and 
                                  data['high'].iloc[i] > node and data['close'].iloc[i] < node):
                                
                                # Zusätzliche Filter anwenden
                                if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                    data.loc[data.index[i], 'signal'] = -1
                                    logger.debug(f"VolumeNode-Bounce-Short bei {data.index[i]}, Preis: {current_price}, Node: {node}")
                    
                    elif entry_condition == 'breakout':
                        # Ausbruch über/unter Volume Node
                        if i > 0:
                            prev_price = data['close'].iloc[i-1]
                            crossing, direction = self._is_price_crossing_level(current_price, prev_price, node)
                            
                            if crossing:
                                # Ausbruch nach oben (Long)
                                if direction == 'up':
                                    # Zusätzliche Filter anwenden
                                    if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i]:
                                        data.loc[data.index[i], 'signal'] = 1
                                        logger.debug(f"VolumeNode-Breakout-Long bei {data.index[i]}, Preis: {current_price}, Node: {node}")
                                
                                # Ausbruch nach unten (Short)
                                elif direction == 'down':
                                    # Zusätzliche Filter anwenden
                                    if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i]:
                                        data.loc[data.index[i], 'signal'] = -1
                                        logger.debug(f"VolumeNode-Breakout-Short bei {data.index[i]}, Preis: {current_price}, Node: {node}")
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
