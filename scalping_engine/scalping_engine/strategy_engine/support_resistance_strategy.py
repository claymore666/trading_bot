import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class SupportResistanceStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf Support- und Resistance-Levels.
    """
    
    def __init__(
        self,
        name: str = "Support Resistance Strategy",
        description: str = "Handelt auf Basis von Support- und Resistance-Levels",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Support-Resistance-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'lookback_periods': 20,       # Anzahl der Perioden zur Identifikation von Swing-Hochs/Tiefs
            'threshold_pct': 0.5,         # Schwellenwert in Prozent für die Bestimmung von Swings
            'num_touches': 2,             # Mindestanzahl von "Berührungen" für einen validen Level
            'level_proximity_pct': 0.2,   # Gruppierung von nahen Levels innerhalb dieses Prozentsatzes
            'breakout_confirmation': 2,   # Anzahl der Kerzen für die Breakout-Bestätigung
            'invalidation_periods': 5,    # Anzahl der Perioden, bis ein Level ungültig wird
            'volume_filter': True,        # Volumen-Filter verwenden
            'atr_period': 14,             # Periode für ATR-Berechnung
            'atr_multiplier': 0.5,        # ATR-Multiplikator für Einstiegs- und Stop-Loss-Berechnung
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
        
        # Interne Variablen für Support-/Resistance-Levels
        self.support_levels = []
        self.resistance_levels = []
    
    def _find_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Identifiziert Swing-Hoch- und Swing-Tief-Punkte im Preischart.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            Tuple[List[int], List[int]]: Listen mit Indizes von Swing-Hochs und Swing-Tiefs
        """
        lookback = self.parameters['lookback_periods']
        threshold_pct = self.parameters['threshold_pct'] / 100
        
        # Listen für Swing-Hochs und Swing-Tiefs
        swing_highs = []
        swing_lows = []
        
        # Preisdaten
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Iteration über die Daten (wir ignorieren die ersten und letzten lookback Perioden)
        for i in range(lookback, len(data) - lookback):
            # Prüfen, ob es sich um ein Swing-Hoch handelt
            is_swing_high = True
            current_high = high_prices[i]
            
            # Vergleich mit vorherigen und nachfolgenden Kerzen
            for j in range(1, lookback + 1):
                if high_prices[i - j] > current_high or high_prices[i + j] > current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Prüfen, ob das Swing-Hoch signifikant ist (% über dem Durchschnitt)
                avg_price = np.mean(high_prices[i - lookback:i + lookback + 1])
                if (current_high - avg_price) / avg_price > threshold_pct:
                    swing_highs.append(i)
            
            # Prüfen, ob es sich um ein Swing-Tief handelt
            is_swing_low = True
            current_low = low_prices[i]
            
            # Vergleich mit vorherigen und nachfolgenden Kerzen
            for j in range(1, lookback + 1):
                if low_prices[i - j] < current_low or low_prices[i + j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Prüfen, ob das Swing-Tief signifikant ist (% unter dem Durchschnitt)
                avg_price = np.mean(low_prices[i - lookback:i + lookback + 1])
                if (avg_price - current_low) / avg_price > threshold_pct:
                    swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def _group_levels(self, price_levels: List[float]) -> List[float]:
        """
        Gruppiert ähnliche Preislevels, um Duplikate zu vermeiden.
        
        Args:
            price_levels: Liste von Preislevels
            
        Returns:
            List[float]: Gruppierte Preislevels
        """
        if not price_levels:
            return []
        
        proximity_pct = self.parameters['level_proximity_pct'] / 100
        grouped_levels = []
        
        # Sortiere Levels aufsteigend
        sorted_levels = sorted(price_levels)
        
        # Aktueller Cluster
        current_cluster = [sorted_levels[0]]
        
        # Durchlaufe alle Levels
        for level in sorted_levels[1:]:
            # Berechne den durchschnittlichen Preis des aktuellen Clusters
            avg_price = sum(current_cluster) / len(current_cluster)
            
            # Wenn der Level nahe genug am Cluster ist, füge ihn hinzu
            if abs(level - avg_price) / avg_price <= proximity_pct:
                current_cluster.append(level)
            else:
                # Andernfalls schließe den aktuellen Cluster ab und beginne einen neuen
                grouped_levels.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Füge den letzten Cluster hinzu
        if current_cluster:
            grouped_levels.append(sum(current_cluster) / len(current_cluster))
        
        return grouped_levels
    
    def _identify_support_resistance(self):
        """
        Identifiziert Support- und Resistance-Levels basierend auf Swing-Punkten.
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return
        
        # Swing-Punkte finden
        swing_highs, swing_lows = self._find_swing_points(self._data)
        
        # Preislisten für Support- und Resistance-Levels
        resistance_prices = [self._data['high'].iloc[idx] for idx in swing_highs]
        support_prices = [self._data['low'].iloc[idx] for idx in swing_lows]
        
        # Levels gruppieren
        self.resistance_levels = self._group_levels(resistance_prices)
        self.support_levels = self._group_levels(support_prices)
        
        logger.info(f"Identifizierte {len(self.support_levels)} Support-Levels und {len(self.resistance_levels)} Resistance-Levels")
    
    def _count_level_touches(self, data: pd.DataFrame, level: float, is_support: bool) -> int:
        """
        Zählt, wie oft ein Preisniveau "berührt" wurde.
        
        Args:
            data: DataFrame mit Preisdaten
            level: Preisniveau
            is_support: True für Support-Level, False für Resistance-Level
            
        Returns:
            int: Anzahl der Berührungen
        """
        tolerance = level * 0.002  # 0.2% Toleranz
        
        if is_support:
            # Für Support-Levels prüfen wir, wie oft der Preis nach unten in die Nähe kam
            touches = sum((data['low'] >= level - tolerance) & (data['low'] <= level + tolerance))
        else:
            # Für Resistance-Levels prüfen wir, wie oft der Preis nach oben in die Nähe kam
            touches = sum((data['high'] <= level + tolerance) & (data['high'] >= level - tolerance))
        
        return touches
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Support-/Resistance-Breakouts.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Support- und Resistance-Levels identifizieren
        self._identify_support_resistance()
        
        # Parameter extrahieren
        breakout_confirmation = self.parameters['breakout_confirmation']
        num_touches = self.parameters['num_touches']
        volume_filter = self.parameters['volume_filter']
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # ATR für Stop-Loss-Berechnung
        atr_period = self.parameters['atr_period']
        if 'atr' not in data.columns:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Nur valide Support- und Resistance-Levels verwenden
        valid_supports = []
        valid_resistances = []
        
        for level in self.support_levels:
            if self._count_level_touches(data, level, True) >= num_touches:
                valid_supports.append(level)
        
        for level in self.resistance_levels:
            if self._count_level_touches(data, level, False) >= num_touches:
                valid_resistances.append(level)
        
        # Für jeden Datenpunkt prüfen, ob es einen Breakout gibt
        for i in range(breakout_confirmation, len(data)):
            current_close = data['close'].iloc[i]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            
            # Prüfen auf Breakout über Resistance-Level (Long-Signal)
            for level in valid_resistances:
                # Prüfen, ob der Preis zuvor unter dem Level war
                was_below = all(data['close'].iloc[i-breakout_confirmation:i] < level)
                
                # Breakout über Resistance (Long)
                if was_below and current_close > level and volume_filter_condition.iloc[i]:
                    data.loc[data.index[i], 'signal'] = 1
                    logger.debug(f"Resistance-Breakout bei {data.index[i]}, Preis: {current_close}, Level: {level}")
                    break
            
            # Prüfen auf Breakout unter Support-Level (Short-Signal)
            for level in valid_supports:
                # Prüfen, ob der Preis zuvor über dem Level war
                was_above = all(data['close'].iloc[i-breakout_confirmation:i] > level)
                
                # Breakout unter Support (Short)
                if was_above and current_close < level and volume_filter_condition.iloc[i]:
                    data.loc[data.index[i], 'signal'] = -1
                    logger.debug(f"Support-Breakout bei {data.index[i]}, Preis: {current_close}, Level: {level}")
                    break
        
        return data
