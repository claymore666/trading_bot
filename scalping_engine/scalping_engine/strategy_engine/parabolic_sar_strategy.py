import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class ParabolicSARStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf dem Parabolic SAR Indikator.
    """
    
    def __init__(
        self,
        name: str = "Parabolic SAR Strategy",
        description: str = "Handelt auf Basis des Parabolic SAR Indikators",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Parabolic-SAR-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'initial_af': 0.02,        # Initial Acceleration Factor
            'max_af': 0.2,             # Maximum Acceleration Factor
            'af_step': 0.02,           # Acceleration Factor Increment
            'filter_method': 'atr',    # Filter-Methode: 'atr', 'adx', 'rsi', 'none'
            'atr_period': 14,          # ATR-Periode für Volatilitäts-Filter
            'atr_multiplier': 1.0,     # ATR-Multiplikator für Stop-Loss
            'adx_period': 14,          # ADX-Periode für Trend-Filter
            'adx_threshold': 25,       # ADX-Schwellenwert für Trendstärke
            'rsi_period': 14,          # RSI-Periode
            'rsi_overbought': 70,      # RSI-Niveau für überkaufte Bedingung
            'rsi_oversold': 30,        # RSI-Niveau für überverkaufte Bedingung
            'volume_filter': True,     # Volumen-Filter verwenden
            'trend_filter': True,      # Trend-Filter verwenden
            'trend_period': 50,        # Periode für Trend-Bestimmung
            'confirmation_period': 1,  # Anzahl der Bestätigungskerzen
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def _calculate_psar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet den Parabolic SAR Indikator.
        
        Args:
            data: DataFrame mit OHLCV-Daten
            
        Returns:
            pd.DataFrame: DataFrame mit berechneten PSAR-Werten
        """
        # Parameter extrahieren
        initial_af = self.parameters['initial_af']
        max_af = self.parameters['max_af']
        af_step = self.parameters['af_step']
        
        # Kopie der Daten erstellen
        result = data.copy()
        
        # Arrays für Berechnung initialisieren
        high = result['high'].values
        low = result['low'].values
        close = result['close'].values
        
        # Ergebnis-Arrays initialisieren
        length = len(result)
        psar = np.zeros(length)
        psarbull = np.zeros(length)
        psarbear = np.zeros(length)
        bull = np.zeros(length)
        af = np.zeros(length)
        ep = np.zeros(length)
        
        # Initialisierung für die ersten 2 Werte
        # Wir nehmen an, dass wir mit einem Aufwärtstrend beginnen
        bull[0] = True
        af[0] = initial_af
        ep[0] = high[0]
        psar[0] = low[0]
        
        # Zweiter Wert
        if high[1] > high[0]:
            ep[1] = high[1]
            af[1] = af[0] + af_step
            if af[1] > max_af:
                af[1] = max_af
        else:
            ep[1] = ep[0]
            af[1] = af[0]
        
        # Wenn wir im Aufwärtstrend sind, ist PSAR der niedrigste Wert der letzten beiden Perioden
        if bull[0]:
            psar[1] = psar[0] + af[0] * (ep[0] - psar[0])
            psar[1] = min(psar[1], low[0], low[1])
            bull[1] = True
            psarbull[1] = psar[1]
        # Wenn wir im Abwärtstrend sind, ist PSAR der höchste Wert der letzten beiden Perioden
        else:
            psar[1] = psar[0] + af[0] * (ep[0] - psar[0])
            psar[1] = max(psar[1], high[0], high[1])
            bull[1] = False
            psarbear[1] = psar[1]
        
        # Hauptberechnungsschleife
        for i in range(2, length):
            # Wenn der vorherige Punkt ein Aufwärtstrend war
            if bull[i-1]:
                # Trendumkehr?
                if low[i] < psar[i-1]:
                    # Trendwechsel zu Abwärtstrend
                    bull[i] = False
                    psar[i] = ep[i-1]
                    psarbear[i] = psar[i]
                    ep[i] = low[i]
                    af[i] = initial_af
                else:
                    # Fortsetzen des Aufwärtstrends
                    bull[i] = True
                    psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                    psar[i] = min(psar[i], low[i-1], low[i-2])
                    psarbull[i] = psar[i]
                    
                    # Neues Hoch?
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_step, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            # Wenn der vorherige Punkt ein Abwärtstrend war
            else:
                # Trendumkehr?
                if high[i] > psar[i-1]:
                    # Trendwechsel zu Aufwärtstrend
                    bull[i] = True
                    psar[i] = ep[i-1]
                    psarbull[i] = psar[i]
                    ep[i] = high[i]
                    af[i] = initial_af
                else:
                    # Fortsetzen des Abwärtstrends
                    bull[i] = False
                    psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                    psar[i] = max(psar[i], high[i-1], high[i-2])
                    psarbear[i] = psar[i]
                    
                    # Neues Tief?
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_step, max_af)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        # Ergebnisse in DataFrame speichern
        result['psar'] = psar
        result['psarbull'] = psarbull
        result['psarbear'] = psarbear
        result['psar_bull'] = bull.astype(bool)
        
        return result
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet den Average Directional Index (ADX).
        
        Args:
            data: DataFrame mit OHLCV-Daten
            
        Returns:
            pd.DataFrame: DataFrame mit berechneten ADX-Werten
        """
        # Parameter extrahieren
        period = self.parameters['adx_period']
        
        # Kopie der Daten erstellen
        result = data.copy()
        
        # True Range berechnen
        result['tr'] = 0
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        
        # Directional Movement berechnen
        result['dmplus'] = 0
        result['dmminus'] = 0
        
        # +DM
        result.loc[result['high'] - result['high'].shift(1) > result['low'].shift(1) - result['low'], 'dmplus'] = \
            np.maximum(result['high'] - result['high'].shift(1), 0)
        # -DM
        result.loc[result['low'].shift(1) - result['low'] > result['high'] - result['high'].shift(1), 'dmminus'] = \
            np.maximum(result['low'].shift(1) - result['low'], 0)
        
        # Smoothed TR und DM
        result['atr'] = result['tr'].rolling(window=period).mean()
        result['dmplus_smooth'] = result['dmplus'].rolling(window=period).mean()
        result['dmminus_smooth'] = result['dmminus'].rolling(window=period).mean()
        
        # Directional Indicators
        result['di_plus'] = 100 * result['dmplus_smooth'] / result['atr']
        result['di_minus'] = 100 * result['dmminus_smooth'] / result['atr']
        
        # Directional Index
        result['dx'] = 100 * np.abs(result['di_plus'] - result['di_minus']) / (result['di_plus'] + result['di_minus'])
        
        # Average Directional Index
        result['adx'] = result['dx'].rolling(window=period).mean()
        
        return result
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf dem Parabolic SAR Indikator.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        filter_method = self.parameters['filter_method'].lower()
        adx_threshold = self.parameters['adx_threshold']
        rsi_period = self.parameters['rsi_period']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        trend_filter = self.parameters['trend_filter']
        trend_period = self.parameters['trend_period']
        volume_filter = self.parameters['volume_filter']
        confirmation_period = self.parameters['confirmation_period']
        
        # PSAR berechnen
        data = self._calculate_psar(data)
        
        # Filter vorbereiten
        # 1. ADX für Trendstärke
        if filter_method == 'adx':
            data = self._calculate_adx(data)
            data['trend_strong'] = data['adx'] > adx_threshold
        else:
            data['trend_strong'] = True
        
        # 2. RSI für überkaufte/überverkaufte Bedingungen
        if filter_method == 'rsi' and 'rsi' not in data.columns:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # RSI-Filter
            data['rsi_long_ok'] = data['rsi'] < rsi_oversold
            data['rsi_short_ok'] = data['rsi'] > rsi_overbought
        else:
            data['rsi_long_ok'] = True
            data['rsi_short_ok'] = True
        
        # 3. ATR für Stop-Loss-Berechnung
        if 'atr' not in data.columns:
            atr_period = self.parameters['atr_period']
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # 4. Trend-Filter
        if trend_filter:
            data['trend_ma'] = data['close'].rolling(window=trend_period).mean()
            data['trend_up'] = data['close'] > data['trend_ma']
            data['trend_down'] = data['close'] < data['trend_ma']
        else:
            data['trend_up'] = True
            data['trend_down'] = True
        
        # 5. Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # SAR-Signale mit Bestätigung und Filter
        # PSAR-Trendwechsel in kombination mit den Filtern identifizieren
        for i in range(confirmation_period, len(data)):
            # Bullenmarkt (Preis über PSAR)
            if all(data['psar_bull'].iloc[i-j] for j in range(confirmation_period)):
                # Aufwärtstrend beginnt (vorher war es ein Bärenmarkt)
                if not all(data['psar_bull'].iloc[i-confirmation_period-j] for j in range(1, confirmation_period+1)):
                    # Zusätzliche Filter anwenden
                    if (data['trend_strong'].iloc[i] and
                        (not trend_filter or data['trend_up'].iloc[i]) and
                        (filter_method != 'rsi' or data['rsi_long_ok'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = 1
                        logger.debug(f"Long-Signal bei {data.index[i]}, Preis: {data['close'].iloc[i]}, PSAR: {data['psar'].iloc[i]}")
            
            # Bärenmarkt (Preis unter PSAR)
            elif not any(data['psar_bull'].iloc[i-j] for j in range(confirmation_period)):
                # Abwärtstrend beginnt (vorher war es ein Bullenmarkt)
                if any(data['psar_bull'].iloc[i-confirmation_period-j] for j in range(1, confirmation_period+1)):
                    # Zusätzliche Filter anwenden
                    if (data['trend_strong'].iloc[i] and
                        (not trend_filter or data['trend_down'].iloc[i]) and
                        (filter_method != 'rsi' or data['rsi_short_ok'].iloc[i]) and
                        volume_filter_condition.iloc[i]):
                        
                        data.loc[data.index[i], 'signal'] = -1
                        logger.debug(f"Short-Signal bei {data.index[i]}, Preis: {data['close'].iloc[i]}, PSAR: {data['psar'].iloc[i]}")
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
