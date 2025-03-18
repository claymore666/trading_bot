import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class CandlestickPatternStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf Candlestick-Mustern.
    """
    
    def __init__(
        self,
        name: str = "Candlestick Pattern Strategy",
        description: str = "Handelt auf Basis von Candlestick-Mustern",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Candlestick-Pattern-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'body_doji_ratio': 0.05,      # Verhältnis von Body zu Range für Doji
            'hammer_ratio': 0.3,          # Verhältnis für Hammer/Shooting Star
            'engulfing_factor': 1.1,      # Faktor für Engulfing-Muster
            'trend_period': 20,           # Periode für Trendbestimmung
            'confirmation_period': 1,     # Anzahl der Bestätigungskerzen
            'pattern_types': ['doji', 'hammer', 'engulfing', 'star'],  # Arten von zu erkennenden Mustern
            'volume_filter': True,        # Volumen-Filter verwenden
            'trend_filter': True,         # Trend-Filter verwenden
            'use_ema': True,              # EMA statt SMA für Trend-Filterung verwenden
            'active_patterns': {          # Aktive Muster und ihre Gewichtung
                'doji': 1.0,
                'hammer': 1.0,
                'shooting_star': 1.0, 
                'bullish_engulfing': 1.0,
                'bearish_engulfing': 1.0,
                'morning_star': 1.0,
                'evening_star': 1.0
            }
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def _is_doji(self, open_price: float, close_price: float, high_price: float, low_price: float) -> bool:
        """
        Prüft, ob eine Kerze ein Doji ist (sehr kleiner Body im Verhältnis zum Gesamtbereich).
        
        Args:
            open_price: Eröffnungspreis
            close_price: Schlusskurs
            high_price: Tageshöchstkurs
            low_price: Tagestiefstkurs
            
        Returns:
            bool: True, wenn die Kerze ein Doji ist
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0:  # Schutz vor Division durch Null
            return False
        
        body_to_range_ratio = body_size / range_size
        return body_to_range_ratio <= self.parameters['body_doji_ratio']
    
    def _is_hammer(self, open_price: float, close_price: float, high_price: float, low_price: float) -> bool:
        """
        Prüft, ob eine Kerze ein Hammer ist (kleiner Body am oberen Ende, langer unterer Schatten).
        
        Args:
            open_price: Eröffnungspreis
            close_price: Schlusskurs
            high_price: Tageshöchstkurs
            low_price: Tagestiefstkurs
            
        Returns:
            bool: True, wenn die Kerze ein Hammer ist
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0:  # Schutz vor Division durch Null
            return False
        
        # Der Body muss im oberen Teil der Kerze sein
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Ein Hammer hat einen kleinen oberen Schatten
        if upper_shadow > body_size * 0.5:
            return False
        
        # Der untere Schatten muss mindestens X-mal so groß sein wie der Body
        hammer_ratio = self.parameters['hammer_ratio']
        return lower_shadow >= body_size * (1 / hammer_ratio) and body_size <= range_size * hammer_ratio
    
    def _is_shooting_star(self, open_price: float, close_price: float, high_price: float, low_price: float) -> bool:
        """
        Prüft, ob eine Kerze ein Shooting Star ist (kleiner Body am unteren Ende, langer oberer Schatten).
        
        Args:
            open_price: Eröffnungspreis
            close_price: Schlusskurs
            high_price: Tageshöchstkurs
            low_price: Tagestiefstkurs
            
        Returns:
            bool: True, wenn die Kerze ein Shooting Star ist
        """
        body_size = abs(close_price - open_price)
        range_size = high_price - low_price
        
        if range_size == 0:  # Schutz vor Division durch Null
            return False
        
        # Der Body muss im unteren Teil der Kerze sein
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Ein Shooting Star hat einen kleinen unteren Schatten
        if lower_shadow > body_size * 0.5:
            return False
        
        # Der obere Schatten muss mindestens X-mal so groß sein wie der Body
        hammer_ratio = self.parameters['hammer_ratio']
        return upper_shadow >= body_size * (1 / hammer_ratio) and body_size <= range_size * hammer_ratio
    
    def _is_bullish_engulfing(self, current_open: float, current_close: float, 
                           prev_open: float, prev_close: float) -> bool:
        """
        Prüft, ob ein bullisches Engulfing-Muster vorliegt.
        
        Args:
            current_open: Eröffnungspreis der aktuellen Kerze
            current_close: Schlusskurs der aktuellen Kerze
            prev_open: Eröffnungspreis der vorherigen Kerze
            prev_close: Schlusskurs der vorherigen Kerze
            
        Returns:
            bool: True, wenn ein bullisches Engulfing-Muster vorliegt
        """
        # Vorherige Kerze muss ein roter Körper sein (close < open)
        if prev_close >= prev_open:
            return False
        
        # Aktuelle Kerze muss ein grüner Körper sein (close > open)
        if current_close <= current_open:
            return False
        
        # Der aktuelle Körper muss den vorherigen Körper umschließen
        engulfing_factor = self.parameters['engulfing_factor']
        return (current_open <= prev_close * engulfing_factor and 
                current_close >= prev_open * engulfing_factor)
    
    def _is_bearish_engulfing(self, current_open: float, current_close: float, 
                           prev_open: float, prev_close: float) -> bool:
        """
        Prüft, ob ein bärisches Engulfing-Muster vorliegt.
        
        Args:
            current_open: Eröffnungspreis der aktuellen Kerze
            current_close: Schlusskurs der aktuellen Kerze
            prev_open: Eröffnungspreis der vorherigen Kerze
            prev_close: Schlusskurs der vorherigen Kerze
            
        Returns:
            bool: True, wenn ein bärisches Engulfing-Muster vorliegt
        """
        # Vorherige Kerze muss ein grüner Körper sein (close > open)
        if prev_close <= prev_open:
            return False
        
        # Aktuelle Kerze muss ein roter Körper sein (close < open)
        if current_close >= current_open:
            return False
        
        # Der aktuelle Körper muss den vorherigen Körper umschließen
        engulfing_factor = self.parameters['engulfing_factor']
        return (current_open >= prev_close * engulfing_factor and 
                current_close <= prev_open * engulfing_factor)
    
    def _is_morning_star(self, data: pd.DataFrame, index: int) -> bool:
        """
        Prüft, ob ein Morning-Star-Muster vorliegt (Umkehrmuster am Tiefpunkt).
        
        Args:
            data: DataFrame mit Preisdaten
            index: Index der aktuellen Kerze
            
        Returns:
            bool: True, wenn ein Morning-Star-Muster vorliegt
        """
        if index < 2:
            return False
        
        # Erste Kerze: Bearish
        first_open = data['open'].iloc[index-2]
        first_close = data['close'].iloc[index-2]
        if first_close >= first_open:
            return False
        
        # Zweite Kerze: Doji oder kleiner Körper
        second_open = data['open'].iloc[index-1]
        second_close = data['close'].iloc[index-1]
        second_high = data['high'].iloc[index-1]
        second_low = data['low'].iloc[index-1]
        
        is_small_body = abs(second_close - second_open) < (second_high - second_low) * 0.3
        if not (self._is_doji(second_open, second_close, second_high, second_low) or is_small_body):
            return False
        
        # Dritte Kerze: Bullish
        third_open = data['open'].iloc[index]
        third_close = data['close'].iloc[index]
        if third_close <= third_open:
            return False
        
        # Gap-Down zwischen erster und zweiter Kerze
        max_second = max(second_open, second_close)
        if max_second >= first_close:
            return False
        
        # Gap-Up zwischen zweiter und dritter Kerze
        min_third = min(third_open, third_close)
        if min_third <= max_second:
            return False
        
        # Dritte Kerze schließt über der Mitte der ersten Kerze
        midpoint_first = (first_open + first_close) / 2
        return third_close > midpoint_first
    
    def _is_evening_star(self, data: pd.DataFrame, index: int) -> bool:
        """
        Prüft, ob ein Evening-Star-Muster vorliegt (Umkehrmuster am Höhepunkt).
        
        Args:
            data: DataFrame mit Preisdaten
            index: Index der aktuellen Kerze
            
        Returns:
            bool: True, wenn ein Evening-Star-Muster vorliegt
        """
        if index < 2:
            return False
        
        # Erste Kerze: Bullish
        first_open = data['open'].iloc[index-2]
        first_close = data['close'].iloc[index-2]
        if first_close <= first_open:
            return False
        
        # Zweite Kerze: Doji oder kleiner Körper
        second_open = data['open'].iloc[index-1]
        second_close = data['close'].iloc[index-1]
        second_high = data['high'].iloc[index-1]
        second_low = data['low'].iloc[index-1]
        
        is_small_body = abs(second_close - second_open) < (second_high - second_low) * 0.3
        if not (self._is_doji(second_open, second_close, second_high, second_low) or is_small_body):
            return False
        
        # Dritte Kerze: Bearish
        third_open = data['open'].iloc[index]
        third_close = data['close'].iloc[index]
        if third_close >= third_open:
            return False
        
        # Gap-Up zwischen erster und zweiter Kerze
        min_second = min(second_open, second_close)
        if min_second <= first_close:
            return False
        
        # Gap-Down zwischen zweiter und dritter Kerze
        max_third = max(third_open, third_close)
        if max_third >= min_second:
            return False
        
        # Dritte Kerze schließt unter der Mitte der ersten Kerze
        midpoint_first = (first_open + first_close) / 2
        return third_close < midpoint_first
    
    def _detect_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Erkennt Candlestick-Muster in den Preisdaten.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            pd.DataFrame: DataFrame mit erkannten Mustern
        """
        # Kopie der Daten erstellen
        result = data.copy()
        
        # Spalten für Muster hinzufügen
        result['is_doji'] = False
        result['is_hammer'] = False
        result['is_shooting_star'] = False
        result['is_bullish_engulfing'] = False
        result['is_bearish_engulfing'] = False
        result['is_morning_star'] = False
        result['is_evening_star'] = False
        
        # Muster-Erkennung
        active_patterns = self.parameters['active_patterns']
        
        # Für jeden Datenpunkt Muster erkennen
        for i in range(1, len(result)):
            current_open = result['open'].iloc[i]
            current_close = result['close'].iloc[i]
            current_high = result['high'].iloc[i]
            current_low = result['low'].iloc[i]
            
            prev_open = result['open'].iloc[i-1]
            prev_close = result['close'].iloc[i-1]
            
            # Doji-Erkennung
            if active_patterns.get('doji', 0) > 0:
                result.loc[result.index[i], 'is_doji'] = self._is_doji(current_open, current_close, current_high, current_low)
            
            # Hammer-Erkennung
            if active_patterns.get('hammer', 0) > 0:
                result.loc[result.index[i], 'is_hammer'] = self._is_hammer(current_open, current_close, current_high, current_low)
            
            # Shooting-Star-Erkennung
            if active_patterns.get('shooting_star', 0) > 0:
                result.loc[result.index[i], 'is_shooting_star'] = self._is_shooting_star(current_open, current_close, current_high, current_low)
            
            # Engulfing-Muster-Erkennung
            if active_patterns.get('bullish_engulfing', 0) > 0:
                result.loc[result.index[i], 'is_bullish_engulfing'] = self._is_bullish_engulfing(current_open, current_close, prev_open, prev_close)
            
            if active_patterns.get('bearish_engulfing', 0) > 0:
                result.loc[result.index[i], 'is_bearish_engulfing'] = self._is_bearish_engulfing(current_open, current_close, prev_open, prev_close)
            
            # Star-Muster-Erkennung
            if i >= 2:
                if active_patterns.get('morning_star', 0) > 0:
                    result.loc[result.index[i], 'is_morning_star'] = self._is_morning_star(result, i)
                
                if active_patterns.get('evening_star', 0) > 0:
                    result.loc[result.index[i], 'is_evening_star'] = self._is_evening_star(result, i)
        
        return result
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Candlestick-Mustern.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        trend_period = self.parameters['trend_period']
        trend_filter = self.parameters['trend_filter']
        use_ema = self.parameters['use_ema']
        volume_filter = self.parameters['volume_filter']
        confirmation_period = self.parameters['confirmation_period']
        
        # Trend-Indikatoren berechnen, falls notwendig
        if trend_filter:
            if use_ema:
                if f'ema_{trend_period}' not in data.columns:
                    data[f'ema_{trend_period}'] = data['close'].ewm(span=trend_period, adjust=False).mean()
                trend_ma = f'ema_{trend_period}'
            else:
                if f'sma_{trend_period}' not in data.columns:
                    data[f'sma_{trend_period}'] = data['close'].rolling(window=trend_period).mean()
                trend_ma = f'sma_{trend_period}'
            
            # Trend-Richtung bestimmen
            data['trend'] = 0
            data.loc[data['close'] > data[trend_ma], 'trend'] = 1
            data.loc[data['close'] < data[trend_ma], 'trend'] = -1
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Candlestick-Muster erkennen
        data = self._detect_patterns(data)
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Bullische Signale
        bullish_conditions = (
            (data['is_hammer']) |
            (data['is_bullish_engulfing']) |
            (data['is_morning_star'])
        )
        
        # Bärische Signale
        bearish_conditions = (
            (data['is_shooting_star']) |
            (data['is_bearish_engulfing']) |
            (data['is_evening_star'])
        )
        
        # Doji als Unentschieden oder leicht in Trend-Richtung
        if 'trend' in data.columns:
            data.loc[data['is_doji'] & (data['trend'] > 0), 'signal'] = 0.5
            data.loc[data['is_doji'] & (data['trend'] < 0), 'signal'] = -0.5
        
        # Trend-Filterung anwenden, falls aktiviert
        if trend_filter:
            # Nur long gehen, wenn der Trend aufwärts gerichtet ist
            long_condition = bullish_conditions & (data['trend'] > 0) & volume_filter_condition
            
            # Nur short gehen, wenn der Trend abwärts gerichtet ist
            short_condition = bearish_conditions & (data['trend'] < 0) & volume_filter_condition
        else:
            # Ohne Trend-Filter
            long_condition = bullish_conditions & volume_filter_condition
            short_condition = bearish_conditions & volume_filter_condition
        
        # Signale setzen
        data.loc[long_condition, 'signal'] = 1
        data.loc[short_condition, 'signal'] = -1
        
        # Bestätigungsperiode anwenden (nur Signale halten, die für N Perioden bestehen)
        if confirmation_period > 1:
            for i in range(confirmation_period, len(data)):
                # Wenn das Signal nicht konsistent ist, zurücksetzen
                if data['signal'].iloc[i] != data['signal'].iloc[i-1]:
                    confirmation_count = 1
                else:
                    confirmation_count += 1
                
                # Signal nur behalten, wenn es für confirmation_period Perioden besteht
                if confirmation_count < confirmation_period:
                    data.loc[data.index[i], 'signal'] = 0
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
