import pandas as pd
import numpy as np
from typing import Dict, Any, List
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase
from scalping_engine.strategy_engine.breakout_detector import BreakoutDetector
from scalping_engine.strategy_engine.consolidation_detector import ConsolidationDetector
from scalping_engine.strategy_engine.signal_filter import SignalFilter

@StrategyBase.register
class BreakoutStrategy(StrategyBase):
    """
    Implementierung einer Breakout-Trading-Strategie, die Ausbrüche aus Konsolidierungszonen erkennt.
    
    Die Strategie identifiziert Konsolidierungsphasen mit Bollinger-Band-Kontraktionen
    und handelt Ausbrüche mit Volumenbestätigung und mehrstufiger Filterung.
    """
    
    def __init__(
        self,
        name: str = "Breakout Strategy",
        description: str = "Handelt auf Basis von Ausbrüchen aus Konsolidierungsphasen",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Breakout-Strategie mit den angegebenen Parametern.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            # Bollinger Band Parameter
            'bb_period': 20,              # Periode für Bollinger-Bänder
            'bb_std_dev': 2.0,            # Standardabweichungsfaktor für die Bänder
            'bb_squeeze_factor': 0.8,     # Kontraktionsfaktor (niedriger = stärkere Kontraktion)
            
            # Breakout-Parameter
            'breakout_threshold_pct': 0.5, # Prozentsatz für Ausbruchsbestätigung
            'consolidation_periods': 5,    # Mindestanzahl von Perioden in Konsolidierung
            'breakout_volume_factor': 1.5, # Volumenfaktor für Ausbruchsbestätigung
            
            # Filter
            'rsi_period': 14,             # RSI-Periode
            'rsi_overbought': 70,         # RSI-Niveau für überkaufte Bedingung
            'rsi_oversold': 30,           # RSI-Niveau für überverkaufte Bedingung
            'macd_fast': 12,              # Schnelle MACD-Periode
            'macd_slow': 26,              # Langsame MACD-Periode
            'macd_signal': 9,             # MACD-Signal-Periode
            
            # Trendfilter
            'trend_ema_period': 50,       # EMA-Periode für Trendbestimmung
            'trend_filter_on': True,      # Trendfilter verwenden?
            
            # Volatilitätsfilter
            'atr_period': 14,             # ATR-Periode
            'atr_multiplier': 1.5,        # ATR-Multiplikator für volatilitätsbasierte Stopps
            'min_atr_pct': 0.5,           # Mindest-ATR in Prozent für Trades
            
            # Volumenfilter
            'volume_filter_on': True,     # Volumenfilter verwenden?
            'volume_ma_period': 20,       # Volumen-MA-Periode
            
            # Bestätigung
            'confirmation_periods': 1      # Anzahl der Perioden für Ausbruchsbestätigung
        }
        
        # Default-Parameter mit benutzerdefinierten Parametern überschreiben
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
        
        # Initialisieren der Submodule
        self.consolidation_detector = ConsolidationDetector(
            bb_period=self.parameters['bb_period'],
            bb_std_dev=self.parameters['bb_std_dev'],
            bb_squeeze_factor=self.parameters['bb_squeeze_factor'],
            consolidation_periods=self.parameters['consolidation_periods']
        )
        
        self.breakout_detector = BreakoutDetector(
            breakout_threshold_pct=self.parameters['breakout_threshold_pct'],
            confirmation_periods=self.parameters['confirmation_periods'],
            volume_filter_on=self.parameters['volume_filter_on'],
            breakout_volume_factor=self.parameters['breakout_volume_factor']
        )
        
        self.signal_filter = SignalFilter(
            trend_filter_on=self.parameters['trend_filter_on'],
            trend_ema_period=self.parameters['trend_ema_period'],
            rsi_oversold=self.parameters['rsi_oversold'],
            rsi_overbought=self.parameters['rsi_overbought'],
            min_atr_pct=self.parameters['min_atr_pct']
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet Indikatoren, die für die Strategie benötigt werden.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            pd.DataFrame: DataFrame mit den berechneten Indikatoren
        """
        # Kopie der Daten erstellen
        result = data.copy()
        
        # Parameter abrufen
        bb_period = self.parameters['bb_period']
        bb_std_dev = self.parameters['bb_std_dev']
        trend_ema_period = self.parameters['trend_ema_period']
        rsi_period = self.parameters['rsi_period']
        atr_period = self.parameters['atr_period']
        volume_ma_period = self.parameters['volume_ma_period']
        macd_fast = self.parameters['macd_fast']
        macd_slow = self.parameters['macd_slow']
        macd_signal = self.parameters['macd_signal']
        
        # Bollinger Bands berechnen
        if 'bollinger_mid' not in result.columns:
            result['bollinger_mid'] = result['close'].rolling(window=bb_period).mean()
            result['bollinger_std'] = result['close'].rolling(window=bb_period).std()
            result['bollinger_upper'] = result['bollinger_mid'] + bb_std_dev * result['bollinger_std']
            result['bollinger_lower'] = result['bollinger_mid'] - bb_std_dev * result['bollinger_std']
        
        # Trend-EMA berechnen
        if 'ema_trend' not in result.columns:
            result['ema_trend'] = result['close'].ewm(span=trend_ema_period, adjust=False).mean()
        
        # RSI berechnen
        if 'rsi' not in result.columns:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR berechnen
        if 'atr' not in result.columns:
            high_low = result['high'] - result['low']
            high_close = abs(result['high'] - result['close'].shift())
            low_close = abs(result['low'] - result['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result['atr'] = true_range.rolling(window=atr_period).mean()
            result['atr_pct'] = result['atr'] / result['close'] * 100
        
        # Volumen-Indikator
        if 'volume' in result.columns and 'volume_ma' not in result.columns:
            result['volume_ma'] = result['volume'].rolling(window=volume_ma_period).mean()
        
        # MACD berechnen
        if 'macd' not in result.columns:
            result['macd_fast_ema'] = result['close'].ewm(span=macd_fast, adjust=False).mean()
            result['macd_slow_ema'] = result['close'].ewm(span=macd_slow, adjust=False).mean()
            result['macd'] = result['macd_fast_ema'] - result['macd_slow_ema']
            result['macd_signal'] = result['macd'].ewm(span=macd_signal, adjust=False).mean()
            result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        return result
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Breakouts aus Konsolidierungsphasen.
        
        Returns:
            pd.DataFrame: DataFrame mit den Daten und Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Indikatoren zu den Daten hinzufügen
        data = self._calculate_indicators(self._data)
        
        # Konsolidierung erkennen
        consolidation = self.consolidation_detector.detect_consolidation(data)
        
        # Breakouts identifizieren
        breakout_signals = self.breakout_detector.detect_breakouts(data, consolidation)
        
        # Filter anwenden
        filtered_signals = self.signal_filter.apply_filters(data, breakout_signals)
        
        # Signale dem DataFrame hinzufügen
        data['signal'] = filtered_signals
        
        # NaNs aufgrund von Rolling Windows entfernen
        data = data.dropna()
        
        logger.info(f"Strategie hat {len(data[data['signal'] != 0])} Signale generiert")
        
        return data
