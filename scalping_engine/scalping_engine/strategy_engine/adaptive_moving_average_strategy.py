import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class AdaptiveMovingAverageStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf adaptiven gleitenden Durchschnitten.
    Verwendet Kaufman's Adaptive Moving Average (KAMA) oder Arnaud Legoux Moving Average (ALMA).
    """
    
    def __init__(
        self,
        name: str = "Adaptive Moving Average Strategy",
        description: str = "Handelt auf Basis adaptiver gleitender Durchschnitte",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Adaptive-Moving-Average-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'ma_type': 'kama',               # Type des Moving Average: 'kama' oder 'alma'
            'fast_kama_period': 10,          # Schnelle KAMA-Periode
            'slow_kama_period': 50,          # Langsame KAMA-Periode
            'kama_efficiency_ratio': 2,      # Effizienz-Ratio für KAMA
            'alma_window': 20,               # Fenstergröße für ALMA
            'alma_offset': 0.85,             # Offset für ALMA (0 bis 1)
            'alma_sigma': 6,                 # Sigma für die Gauß'sche Verteilung bei ALMA
            'signal_period': 9,              # Periode für die Signal-Linie
            'trend_filter': True,            # Trend-Filter verwenden
            'trend_period': 200,             # Periode für Trend-Bestimmung (SMA/EMA)
            'use_ema_for_trend': True,       # EMA statt SMA für Trend-Filterung verwenden
            'crossover_confirmation': 1,     # Anzahl der Kerzen für die Bestätigung eines Crossovers
            'signal_quality_threshold': 0.3, # Mindestqualität für ein Signal (0-1)
            'volume_filter': True,           # Volumen-Filter verwenden
            'volatility_filter': True,       # Volatilitäts-Filter verwenden
            'atr_period': 14,                # ATR-Periode für Volatilitäts-Filter
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def _calculate_kama(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Berechnet den Kaufman Adaptive Moving Average (KAMA).
        
        Args:
            data: DataFrame mit Preisdaten
            price_column: Spaltenname für die Preisdaten
            
        Returns:
            pd.DataFrame: DataFrame mit KAMA-Werten
        """
        # Parameter extrahieren
        fast_period = self.parameters['fast_kama_period']
        slow_period = self.parameters['slow_kama_period']
        er_period = self.parameters['kama_efficiency_ratio']
        
        # Kopie der Daten erstellen
        result = data.copy()
        
        # Change berechnen
        change = abs(result[price_column] - result[price_column].shift(er_period))
        
        # Volatilität berechnen (Summe der absoluten Preisänderungen)
        volatility = result[price_column].diff().abs().rolling(window=er_period).sum()
        
        # Efficiency Ratio berechnen (Verhältnis von Preisänderung zu Volatilität)
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Smoothing-Konstanten berechnen
        fast_alpha = 2 / (fast_period + 1)
        slow_alpha = 2 / (slow_period + 1)
        
        # Smoothing Constant berechnen
        sc = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # KAMA berechnen
        kama = pd.Series(index=result.index, dtype='float64')
        kama.iloc[er_period] = result[price_column].iloc[er_period]  # Initialisierung
        
        # KAMA-Berechnung durchführen
        for i in range(er_period + 1, len(result)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (result[price_column].iloc[i] - kama.iloc[i-1])
        
        result['kama'] = kama
        
        return result
    
    def _calculate_alma(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Berechnet den Arnaud Legoux Moving Average (ALMA).
        
        Args:
            data: DataFrame mit Preisdaten
            price_column: Spaltenname für die Preisdaten
            
        Returns:
            pd.DataFrame: DataFrame mit ALMA-Werten
        """
        # Parameter extrahieren
        window = self.parameters['alma_window']
        offset = self.parameters['alma_offset']
        sigma = self.parameters['alma_sigma']
        
        # Kopie der Daten erstellen
        result = data.copy()
        
        # ALMA-Gewichtung berechnen
        m = offset * (window - 1)
        s = window / sigma
        
        # Gewichte für die Fensterperiode berechnen
        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(window)])
        weights = weights / weights.sum()  # Normalisieren
        
        # ALMA berechnen
        result['alma'] = result[price_column].rolling(window=window).apply(
            lambda x: np.sum(np.array(x) * weights), raw=True
        )
        
        return result
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf adaptiven gleitenden Durchschnitten.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        ma_type = self.parameters['ma_type'].lower()
        signal_period = self.parameters['signal_period']
        trend_filter = self.parameters['trend_filter']
        use_ema_for_trend = self.parameters['use_ema_for_trend']
        trend_period = self.parameters['trend_period']
        crossover_confirmation = self.parameters['crossover_confirmation']
        volume_filter = self.parameters['volume_filter']
        volatility_filter = self.parameters['volatility_filter']
        
        # Adaptiven Moving Average berechnen
        if ma_type == 'kama':
            data = self._calculate_kama(data)
            ma_column = 'kama'
        elif ma_type == 'alma':
            data = self._calculate_alma(data)
            ma_column = 'alma'
        else:
            logger.error(f"Unbekannter Moving-Average-Typ: {ma_type}")
            return pd.DataFrame()
        
        # Signal-Linie berechnen (EMA des adaptiven MA)
        data[f'{ma_column}_signal'] = data[ma_column].ewm(span=signal_period, adjust=False).mean()
        
        # Trend-Filter berechnen, falls aktiviert
        if trend_filter:
            if use_ema_for_trend:
                data['trend_ma'] = data['close'].ewm(span=trend_period, adjust=False).mean()
            else:
                data['trend_ma'] = data['close'].rolling(window=trend_period).mean()
            
            # Trend-Richtung bestimmen
            data['trend'] = 0
            data.loc[data['close'] > data['trend_ma'], 'trend'] = 1
            data.loc[data['close'] < data['trend_ma'], 'trend'] = -1
        
        # Volatilitäts-Filter berechnen, falls aktiviert
        if volatility_filter:
            atr_period = self.parameters['atr_period']
            if 'atr' not in data.columns:
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift())
                low_close = abs(data['low'] - data['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['atr'] = true_range.rolling(window=atr_period).mean()
            
            # Normalisierte ATR (Prozent des Preises)
            data['atr_pct'] = data['atr'] / data['close'] * 100
            
            # Volatilitäts-Filter: True, wenn ATR im akzeptablen Bereich liegt
            median_atr_pct = data['atr_pct'].median()
            max_atr_pct = median_atr_pct * 2  # Maximal doppelt so hoch wie der Median
            data['volatility_ok'] = (data['atr_pct'] > 0) & (data['atr_pct'] < max_atr_pct)
        else:
            data['volatility_ok'] = True
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Crossover berechnen
        data['ma_above_signal'] = data[ma_column] > data[f'{ma_column}_signal']
        
        # Crossover-Signale mit Bestätigung
        for i in range(crossover_confirmation, len(data)):
            # Prüfen, ob es ein bestätigtes Crossover gibt
            ma_above_signal_streak = 0
            ma_below_signal_streak = 0
            
            for j in range(crossover_confirmation):
                if data['ma_above_signal'].iloc[i-j]:
                    ma_above_signal_streak += 1
                else:
                    ma_below_signal_streak += 1
            
            # Long-Signal: MA kreuzt Signal-Linie von unten nach oben
            if (ma_above_signal_streak == crossover_confirmation and 
                not data['ma_above_signal'].iloc[i-crossover_confirmation]):
                
                # Zusätzliche Filter anwenden
                if (not trend_filter or data['trend'].iloc[i] > 0) and volume_filter_condition.iloc[i] and data['volatility_ok'].iloc[i]:
                    data.loc[data.index[i], 'signal'] = 1
            
            # Short-Signal: MA kreuzt Signal-Linie von oben nach unten
            elif (ma_below_signal_streak == crossover_confirmation and 
                  data['ma_above_signal'].iloc[i-crossover_confirmation]):
                
                # Zusätzliche Filter anwenden
                if (not trend_filter or data['trend'].iloc[i] < 0) and volume_filter_condition.iloc[i] and data['volatility_ok'].iloc[i]:
                    data.loc[data.index[i], 'signal'] = -1
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
