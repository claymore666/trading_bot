import pandas as pd
from typing import Dict, Any
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class BollingerBandsStrategy(StrategyBase):
    """
    Implementierung einer Handelsstrategie basierend auf Bollinger-Bändern.
    """
    
    def __init__(
        self,
        name: str = "Bollinger Bands Strategy",
        description: str = "Handelt auf Basis von Bollinger-Band-Breakouts",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Bollinger-Bands-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie (window, std_dev, etc.)
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'window': 20,
            'std_dev': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Bollinger-Bändern.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        window = self.parameters['window']
        std_dev = self.parameters['std_dev']
        rsi_period = self.parameters['rsi_period']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        volume_filter = self.parameters['volume_filter']
        
        # Bollinger-Bänder berechnen, falls nicht vorhanden
        if 'sma_20' not in data.columns:
            data['sma_20'] = data['close'].rolling(window=window).mean()
        
        if 'bollinger_upper' not in data.columns or 'bollinger_lower' not in data.columns:
            data['bollinger_std'] = data['close'].rolling(window=window).std()
            data['bollinger_upper'] = data['sma_20'] + std_dev * data['bollinger_std']
            data['bollinger_lower'] = data['sma_20'] - std_dev * data['bollinger_std']
        
        # RSI berechnen, falls nicht vorhanden
        if 'rsi' not in data.columns:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        # Signal-Spalte erstellen
        data['signal'] = 0
        
        # Volumen-Filter
        if volume_filter and 'volume' in data.columns and 'volume_sma_20' in data.columns:
            volume_filter_condition = data['volume'] > data['volume_sma_20']
        else:
            volume_filter_condition = pd.Series(True, index=data.index)
        
        # Long-Signale: Preis kreuzt unteres Band nach oben und RSI < oversold
        long_condition = (
            (data['close'] > data['bollinger_lower']) & 
            (data['close'].shift(1) <= data['bollinger_lower'].shift(1)) &
            (data['rsi'] < rsi_oversold) &
            volume_filter_condition
        )
        
        # Short-Signale: Preis kreuzt oberes Band nach unten und RSI > overbought
        short_condition = (
            (data['close'] < data['bollinger_upper']) & 
            (data['close'].shift(1) >= data['bollinger_upper'].shift(1)) &
            (data['rsi'] > rsi_overbought) &
            volume_filter_condition
        )
        
        # Signale setzen
        data.loc[long_condition, 'signal'] = 1
        data.loc[short_condition, 'signal'] = -1
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
