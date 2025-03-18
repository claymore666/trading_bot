import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger
from datetime import datetime, timedelta

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class PivotPointStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf Pivot-Points.
    """
    
    def __init__(
        self,
        name: str = "Pivot Point Strategy",
        description: str = "Handelt auf Basis von Pivot-Points und ihren Support/Resistance-Levels",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Pivot-Point-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'pivot_type': 'standard',       # Typ der Pivot-Points: 'standard', 'camarilla', 'woodie', 'fibonacci'
            'pivot_period': 'daily',        # Zeitraum für Pivot-Points: 'daily', 'weekly', 'monthly'
            'price_buffer_pct': 0.1,        # Puffer-Prozentsatz für Pivot-Level-Durchbrüche
            'confirmation_periods': 1,      # Anzahl der Perioden für die Bestätigung eines Durchbruchs
            'pivot_levels_for_signal': 3,   # Anzahl der Pivot-Levels für Signalgenerierung (1=nur PP, 2=PP+S1/R1, 3=PP+S1/R1+S2/R2, ...)
            'signal_type': 'breakout',      # Signaltyp: 'breakout', 'reversal', 'bounce'
            'volume_filter': True,          # Volumen-Filter verwenden
            'trend_filter': True,           # Trend-Filter verwenden
            'trend_period': 50,             # Periode für Trend-Bestimmung
            'macd_filter': False,           # MACD als zusätzlichen Filter verwenden
            'macd_fast': 12,                # Schnelle MACD-Periode
            'macd_slow': 26,                # Langsame MACD-Periode
            'macd_signal': 9,               # MACD-Signal-Periode
            'rsi_filter': False,            # RSI als zusätzlichen Filter verwenden
            'rsi_period': 14,               # RSI-Periode
            'rsi_overbought': 70,           # RSI-Niveau für überkaufte Bedingung
            'rsi_oversold': 30,             # RSI-Niveau für überverkaufte Bedingung
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
        
        # Interne Variable für Pivot-Levels
        self.pivot_levels = {}
    
    def _get_period_data(self, data: pd.DataFrame, period: str = 'daily') -> List[Tuple[pd.Timestamp, pd.Series]]:
        """
        Gruppiert die Daten nach dem angegebenen Zeitraum.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            period: Zeitraum ('daily', 'weekly', 'monthly')
            
        Returns:
            List[Tuple[pd.Timestamp, pd.Series]]: Liste von (Datum, OHLCV-Serie) Tupeln
        """
        # Sicherstellen, dass der Index ein DatetimeIndex ist
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("Daten haben keinen DatetimeIndex")
            return []
        
        # Kopie der Daten erstellen
        df = data.copy()
        
        # Gruppierung nach dem angegebenen Zeitraum
        if period == 'daily':
            # Gruppierung nach Kalendertag
            grouped = df.groupby(df.index.date)
        elif period == 'weekly':
            # Gruppierung nach Kalenderwoche
            grouped = df.groupby([df.index.year, df.index.isocalendar().week])
        elif period == 'monthly':
            # Gruppierung nach Kalendermonat
            grouped = df.groupby([df.index.year, df.index.month])
        else:
            logger.error(f"Unbekannter Zeitraum: {period}")
            return []
        
        # Liste der (Datum, OHLCV-Serie) Tupel erstellen
        # Für jede Gruppe speichern wir Datum, Open, High, Low, Close, Volume
        period_data = []
        
        for group_key, group_df in grouped:
            if len(group_df) > 0:
                # Datum der ersten Kerze in der Gruppe
                date = group_df.index[0]
                
                # OHLCV-Werte für die Gruppe
                period_ohlcv = pd.Series({
                    'open': group_df.iloc[0]['open'],             # Erstes Open
                    'high': group_df['high'].max(),               # Höchstes High
                    'low': group_df['low'].min(),                 # Niedrigstes Low
                    'close': group_df.iloc[-1]['close'],          # Letztes Close
                    'volume': group_df['volume'].sum() if 'volume' in group_df.columns else 0  # Gesamtvolumen
                })
                
                period_data.append((date, period_ohlcv))
        
        return period_data
    
    def _calculate_standard_pivot_points(self, ohlc: pd.Series) -> Dict[str, float]:
        """
        Berechnet die Standard-Pivot-Points.
        
        Args:
            ohlc: Serie mit OHLC-Werten
            
        Returns:
            Dict[str, float]: Berechnete Pivot-Levels
        """
        pivot = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
        
        s1 = (2 * pivot) - ohlc['high']
        s2 = pivot - (ohlc['high'] - ohlc['low'])
        s3 = s2 - (ohlc['high'] - ohlc['low'])
        
        r1 = (2 * pivot) - ohlc['low']
        r2 = pivot + (ohlc['high'] - ohlc['low'])
        r3 = r2 + (ohlc['high'] - ohlc['low'])
        
        return {
            'pp': pivot,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        }
    
    def _calculate_camarilla_pivot_points(self, ohlc: pd.Series) -> Dict[str, float]:
        """
        Berechnet die Camarilla-Pivot-Points.
        
        Args:
            ohlc: Serie mit OHLC-Werten
            
        Returns:
            Dict[str, float]: Berechnete Pivot-Levels
        """
        pivot = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
        range_val = ohlc['high'] - ohlc['low']
        
        s1 = ohlc['close'] - (range_val * 1.1 / 12)
        s2 = ohlc['close'] - (range_val * 1.1 / 6)
        s3 = ohlc['close'] - (range_val * 1.1 / 4)
        s4 = ohlc['close'] - (range_val * 1.1 / 2)
        
        r1 = ohlc['close'] + (range_val * 1.1 / 12)
        r2 = ohlc['close'] + (range_val * 1.1 / 6)
        r3 = ohlc['close'] + (range_val * 1.1 / 4)
        r4 = ohlc['close'] + (range_val * 1.1 / 2)
        
        return {
            'pp': pivot,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'r4': r4
        }
    
    def _calculate_woodie_pivot_points(self, ohlc: pd.Series) -> Dict[str, float]:
        """
        Berechnet die Woodie-Pivot-Points.
        
        Args:
            ohlc: Serie mit OHLC-Werten
            
        Returns:
            Dict[str, float]: Berechnete Pivot-Levels
        """
        pivot = (ohlc['high'] + ohlc['low'] + 2 * ohlc['close']) / 4
        
        s1 = (2 * pivot) - ohlc['high']
        s2 = pivot - (ohlc['high'] - ohlc['low'])
        s3 = s2 - (ohlc['high'] - ohlc['low'])
        s4 = s3 - (ohlc['high'] - ohlc['low'])
        
        r1 = (2 * pivot) - ohlc['low']
        r2 = pivot + (ohlc['high'] - ohlc['low'])
        r3 = r2 + (ohlc['high'] - ohlc['low'])
        r4 = r3 + (ohlc['high'] - ohlc['low'])
        
        return {
            'pp': pivot,
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            'r4': r4
        }
    
    def _calculate_fibonacci_pivot_points(self, ohlc: pd.Series) -> Dict[str, float]:
        """
        Berechnet die Fibonacci-Pivot-Points.
        
        Args:
            ohlc: Serie mit OHLC-Werten
            
        Returns:
            Dict[str, float]: Berechnete Pivot-Levels
        """
        pivot = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
        range_val = ohlc['high'] - ohlc['low']
        
        s1 = pivot - 0.382 * range_val
        s2 = pivot - 0.618 * range_val
        s3 = pivot - 1.0 * range_val
        
        r1 = pivot + 0.382 * range_val
        r2 = pivot + 0.618 * range_val
        r3 = pivot + 1.0 * range_val
        
        return {
            'pp': pivot,
            's1': s1,
            's2': s2,
            's3': s3,
            'r1': r1,
            'r2': r2,
            'r3': r3
        }
    
    def _calculate_pivot_points(self, data: pd.DataFrame) -> Dict[pd.Timestamp, Dict[str, float]]:
        """
        Berechnet Pivot-Points für alle Perioden im DataFrame.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            
        Returns:
            Dict[pd.Timestamp, Dict[str, float]]: Dictionary mit Datum -> Pivot-Levels Mapping
        """
        pivot_type = self.parameters['pivot_type']
        pivot_period = self.parameters['pivot_period']
        
        # Daten nach Perioden gruppieren
        period_data = self._get_period_data(data, pivot_period)
        
        # Dictionary für Pivot-Levels
        pivot_levels = {}
        
        # Für jede Periode Pivot-Points berechnen
        for date, ohlc in period_data:
            # Je nach Pivot-Typ die entsprechende Berechnungsmethode aufrufen
            if pivot_type == 'standard':
                levels = self._calculate_standard_pivot_points(ohlc)
            elif pivot_type == 'camarilla':
                levels = self._calculate_camarilla_pivot_points(ohlc)
            elif pivot_type == 'woodie':
                levels = self._calculate_woodie_pivot_points(ohlc)
            elif pivot_type == 'fibonacci':
                levels = self._calculate_fibonacci_pivot_points(ohlc)
            else:
                logger.error(f"Unbekannter Pivot-Typ: {pivot_type}")
                continue
            
            # Pivot-Levels für diese Periode speichern
            pivot_levels[date] = levels
        
        return pivot_levels
    
    def _get_active_pivot_levels(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Gibt die aktiven Pivot-Levels für ein bestimmtes Datum zurück.
        
        Args:
            date: Datum, für das die aktiven Pivot-Levels gesucht werden
            
        Returns:
            Dict[str, float]: Aktive Pivot-Levels
        """
        # Pivot-Levels nach Datum sortieren (absteigend)
        sorted_dates = sorted(self.pivot_levels.keys(), reverse=True)
        
        # Die neueste Periode finden, die vor dem angegebenen Datum liegt
        for pivot_date in sorted_dates:
            if pivot_date < date:
                return self.pivot_levels[pivot_date]
        
        # Wenn keine passende Periode gefunden wurde, leeres Dictionary zurückgeben
        return {}
    
    def _is_breakout(self, price: float, pivot_level: float, buffer_pct: float, direction: str) -> bool:
        """
        Prüft, ob ein Breakout über/unter ein Pivot-Level vorliegt.
        
        Args:
            price: Aktueller Preis
            pivot_level: Pivot-Level
            buffer_pct: Puffer-Prozentsatz
            direction: Richtung des Breakouts ('up' oder 'down')
            
        Returns:
            bool: True wenn Breakout, sonst False
        """
        buffer = pivot_level * buffer_pct / 100
        
        if direction == 'up':
            return price > pivot_level + buffer
        else:  # direction == 'down'
            return price < pivot_level - buffer
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Pivot-Points.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        price_buffer_pct = self.parameters['price_buffer_pct']
        confirmation_periods = self.parameters['confirmation_periods']
        pivot_levels_for_signal = self.parameters['pivot_levels_for_signal']
        signal_type = self.parameters['signal_type']
        volume_filter = self.parameters['volume_filter']
        trend_filter = self.parameters['trend_filter']
        trend_period = self.parameters['trend_period']
        macd_filter = self.parameters['macd_filter']
        macd_fast = self.parameters['macd_fast']
        macd_slow = self.parameters['macd_slow']
        macd_signal = self.parameters['macd_signal']
        rsi_filter = self.parameters['rsi_filter']
        rsi_period = self.parameters['rsi_period']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        
        # Pivot-Points berechnen
        self.pivot_levels = self._calculate_pivot_points(data)
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Pivot-Levels in den Dataframe einfügen
        # Für jedes Datum die aktiven Pivot-Levels ermitteln
        for i, date in enumerate(data.index):
            active_levels = self._get_active_pivot_levels(date)
            
            if not active_levels:
                continue
            
            # Pivot-Levels in den Dataframe einfügen
            for level_name, level_value in active_levels.items():
                if f'pivot_{level_name}' not in data.columns:
                    data[f'pivot_{level_name}'] = np.nan
                
                data.loc[date, f'pivot_{level_name}'] = level_value
        
        # Filter anwenden
        # 1. Trend-Filter
        if trend_filter:
            data['trend_ma'] = data['close'].rolling(window=trend_period).mean()
            data['trend'] = 0
            data.loc[data['close'] > data['trend_ma'], 'trend'] = 1
            data.loc[data['close'] < data['trend_ma'], 'trend'] = -1
        
        # 2. MACD-Filter
        if macd_filter:
            data['macd_fast_ema'] = data['close'].ewm(span=macd_fast, adjust=False).mean()
            data['macd_slow_ema'] = data['close'].ewm(span=macd_slow, adjust=False).mean()
            data['macd'] = data['macd_fast_ema'] - data['macd_slow_ema']
            data['macd_signal_line'] = data['macd'].ewm(span=macd_signal, adjust=False).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal_line']
            
            # MACD-Signale
            data['macd_bullish'] = data['macd'] > data['macd_signal_line']
            data['macd_bearish'] = data['macd'] < data['macd_signal_line']
        else:
            data['macd_bullish'] = True
            data['macd_bearish'] = True
        
        # 3. RSI-Filter
        if rsi_filter:
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # RSI-Signale
            data['rsi_bullish'] = data['rsi'] < rsi_oversold
            data['rsi_bearish'] = data['rsi'] > rsi_overbought
        else:
            data['rsi_bullish'] = True
            data['rsi_bearish'] = True
        
        # 4. Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Signale generieren
        for i in range(confirmation_periods, len(data)):
            current_price = data['close'].iloc[i]
            pivot_columns = [col for col in data.columns if col.startswith('pivot_')]
            
            # Pivot-Levels für die aktuelle Kerze sortieren
            levels = {}
            for col in pivot_columns:
                if not pd.isna(data[col].iloc[i]):
                    level_name = col.replace('pivot_', '')
                    levels[level_name] = data[col].iloc[i]
            
            if not levels:
                continue
            
            # Support- und Resistance-Levels ermitteln
            pp = levels.get('pp', None)
            supports = sorted([v for k, v in levels.items() if k.startswith('s')], reverse=True)
            resistances = sorted([v for k, v in levels.items() if k.startswith('r')])
            
            # Signal-Generierung nach Typ
            if signal_type == 'breakout':
                # Breakout-Signale
                # Nur die angegebene Anzahl von Levels verwenden
                if resistances and len(resistances) >= pivot_levels_for_signal:
                    # Resistance-Breakout (Long)
                    key_resistance = resistances[pivot_levels_for_signal - 1]
                    
                    # Bestätigung prüfen
                    breakout_confirmed = True
                    for j in range(1, confirmation_periods + 1):
                        if i - j >= 0 and data['close'].iloc[i-j] > key_resistance:
                            breakout_confirmed = False
                            break
                    
                    if (breakout_confirmed and
                        self._is_breakout(current_price, key_resistance, price_buffer_pct, 'up') and
                        (not trend_filter or data['trend'].iloc[i] > 0) and
                        (not macd_filter or data['macd_bullish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bullish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = 1
                        logger.debug(f"Resistance-Breakout bei {data.index[i]}, Preis: {current_price}, Level: {key_resistance}")
                
                if supports and len(supports) >= pivot_levels_for_signal:
                    # Support-Breakout (Short)
                    key_support = supports[pivot_levels_for_signal - 1]
                    
                    # Bestätigung prüfen
                    breakout_confirmed = True
                    for j in range(1, confirmation_periods + 1):
                        if i - j >= 0 and data['close'].iloc[i-j] < key_support:
                            breakout_confirmed = False
                            break
                    
                    if (breakout_confirmed and
                        self._is_breakout(current_price, key_support, price_buffer_pct, 'down') and
                        (not trend_filter or data['trend'].iloc[i] < 0) and
                        (not macd_filter or data['macd_bearish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bearish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = -1
                        logger.debug(f"Support-Breakout bei {data.index[i]}, Preis: {current_price}, Level: {key_support}")
            
            elif signal_type == 'reversal':
                # Reversal-Signale
                if supports and len(supports) >= pivot_levels_for_signal:
                    # Support-Reversal (Long)
                    key_support = supports[pivot_levels_for_signal - 1]
                    
                    # Prüfen, ob der Preis sich dem Support nähert und dann dreht
                    near_support = abs(data['low'].iloc[i] - key_support) / key_support <= price_buffer_pct / 100
                    price_rising = data['close'].iloc[i] > data['open'].iloc[i]
                    
                    if (near_support and price_rising and
                        (not trend_filter or data['trend'].iloc[i] > 0) and
                        (not macd_filter or data['macd_bullish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bullish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = 1
                        logger.debug(f"Support-Reversal bei {data.index[i]}, Preis: {current_price}, Level: {key_support}")
                
                if resistances and len(resistances) >= pivot_levels_for_signal:
                    # Resistance-Reversal (Short)
                    key_resistance = resistances[pivot_levels_for_signal - 1]
                    
                    # Prüfen, ob der Preis sich dem Resistance nähert und dann dreht
                    near_resistance = abs(data['high'].iloc[i] - key_resistance) / key_resistance <= price_buffer_pct / 100
                    price_falling = data['close'].iloc[i] < data['open'].iloc[i]
                    
                    if (near_resistance and price_falling and
                        (not trend_filter or data['trend'].iloc[i] < 0) and
                        (not macd_filter or data['macd_bearish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bearish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = -1
                        logger.debug(f"Resistance-Reversal bei {data.index[i]}, Preis: {current_price}, Level: {key_resistance}")
            
            elif signal_type == 'bounce':
                # Bounce-Signale (ähnlich wie Reversal, aber ohne Kerzenfarbe)
                if supports and len(supports) >= pivot_levels_for_signal:
                    # Support-Bounce (Long)
                    key_support = supports[pivot_levels_for_signal - 1]
                    
                    # Prüfen, ob der Preis nahe am Support ist
                    near_support = abs(data['low'].iloc[i] - key_support) / key_support <= price_buffer_pct / 100
                    
                    if (near_support and
                        (not trend_filter or data['trend'].iloc[i] > 0) and
                        (not macd_filter or data['macd_bullish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bullish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = 1
                        logger.debug(f"Support-Bounce bei {data.index[i]}, Preis: {current_price}, Level: {key_support}")
                
                if resistances and len(resistances) >= pivot_levels_for_signal:
                    # Resistance-Bounce (Short)
                    key_resistance = resistances[pivot_levels_for_signal - 1]
                    
                    # Prüfen, ob der Preis nahe am Resistance ist
                    near_resistance = abs(data['high'].iloc[i] - key_resistance) / key_resistance <= price_buffer_pct / 100
                    
                    if (near_resistance and
                        (not trend_filter or data['trend'].iloc[i] < 0) and
                        (not macd_filter or data['macd_bearish'].iloc[i]) and
                        (not rsi_filter or data['rsi_bearish'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = -1
                        logger.debug(f"Resistance-Bounce bei {data.index[i]}, Preis: {current_price}, Level: {key_resistance}")
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
