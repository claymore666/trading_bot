import pandas as pd
from typing import Dict, Any
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class MovingAverageCrossStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf dem Kreuzen von gleitenden Durchschnitten.
    """
    
    def __init__(
        self,
        name: str = "Moving Average Cross Strategy",
        description: str = "Handelt auf Basis von Kreuzungen gleitender Durchschnitte",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Moving-Average-Cross-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie (fast_ma, slow_ma, etc.)
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'fast_ma': 10,
            'slow_ma': 50,
            'signal_ma': 9,
            'use_ema': True,
            'volume_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf dem Kreuzen von gleitenden Durchschnitten.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        fast_ma = self.parameters['fast_ma']
        slow_ma = self.parameters['slow_ma']
        signal_ma = self.parameters['signal_ma']
        use_ema = self.parameters['use_ema']
        volume_filter = self.parameters['volume_filter']
        
        # Gleitende Durchschnitte berechnen, falls nicht vorhanden
        if use_ema:
            if f'ema_{fast_ma}' not in data.columns:
                data[f'ema_{fast_ma}'] = data['close'].ewm(span=fast_ma, adjust=False).mean()
            if f'ema_{slow_ma}' not in data.columns:
                data[f'ema_{slow_ma}'] = data['close'].ewm(span=slow_ma, adjust=False).mean()
            
            # MACD berechnen
            data['macd'] = data[f'ema_{fast_ma}'] - data[f'ema_{slow_ma}']
            data['macd_signal'] = data['macd'].ewm(span=signal_ma, adjust=False).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        else:
            if f'sma_{fast_ma}' not in data.columns:
                data[f'sma_{fast_ma}'] = data['close'].rolling(window=fast_ma).mean()
            if f'sma_{slow_ma}' not in data.columns:
                data[f'sma_{slow_ma}'] = data['close'].rolling(window=slow_ma).mean()
            
            # MACD berechnen (mit SMA statt EMA)
            data['macd'] = data[f'sma_{fast_ma}'] - data[f'sma_{slow_ma}']
            data['macd_signal'] = data['macd'].rolling(window=signal_ma).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Long-Signal: MACD kreuzt Signal-Linie von unten nach oben
        long_condition = (
            (data['macd'] > data['macd_signal']) & 
            (data['macd'].shift(1) <= data['macd_signal'].shift(1)) &
            volume_filter_condition
        )
        
        # Short-Signal: MACD kreuzt Signal-Linie von oben nach unten
        short_condition = (
            (data['macd'] < data['macd_signal']) & 
            (data['macd'].shift(1) >= data['macd_signal'].shift(1)) &
            volume_filter_condition
        )
        
        # Signale setzen
        data.loc[long_condition, 'signal'] = 1
        data.loc[short_condition, 'signal'] = -1
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
