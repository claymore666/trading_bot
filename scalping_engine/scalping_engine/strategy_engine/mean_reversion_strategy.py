import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from scalping_engine.strategy_engine.strategy_base import StrategyBase

@StrategyBase.register
class MeanReversionStrategy(StrategyBase):
    """
    Implementierung einer Mean-Reversion-Strategie mit statistischen Bändern.
    """
    
    def __init__(
        self,
        name: str = "Mean Reversion Strategy",
        description: str = "Handelt auf Basis statistischer Abweichungen vom Mittelwert",
        parameters: Dict[str, Any] = None,
        risk_per_trade: float = 1.0
    ):
        """
        Initialisiert die Mean-Reversion-Strategie.
        
        Args:
            name: Name der Strategie
            description: Beschreibung der Strategie
            parameters: Parameter der Strategie
            risk_per_trade: Risiko pro Trade in Prozent des Kapitals
        """
        default_params = {
            'lookback_period': 20,        # Periode für die Berechnung des gleitenden Durchschnitts
            'z_score_threshold': 2.0,     # Z-Score-Schwellenwert für Handelssignale
            'entry_threshold': 2.0,       # Z-Score-Schwellenwert für Handelseinstiege
            'exit_threshold': 0.5,        # Z-Score-Schwellenwert für Handelsausstiege
            'max_holding_periods': 10,    # Maximale Haltedauer in Perioden
            'use_bollinger': True,        # Bollinger-Bänder anstelle von Z-Score verwenden
            'bollinger_std': 2.0,         # Standardabweichungsmultiplikator für Bollinger-Bänder
            'atr_period': 14,             # Periode für ATR-Berechnung
            'volume_filter': True,        # Volumen-Filter verwenden
            'rsi_filter': True,           # RSI-Filter verwenden
            'rsi_period': 14,             # Periode für RSI-Berechnung
            'rsi_overbought': 70,         # RSI-Niveau für überkaufte Bedingung
            'rsi_oversold': 30            # RSI-Niveau für überverkaufte Bedingung
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name, description, default_params, risk_per_trade)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf Mean-Reversion-Prinzipien.
        
        Returns:
            pd.DataFrame: DataFrame mit Signalen (1 für Long, -1 für Short, 0 für neutral)
        """
        if not self.is_initialized or self._data is None or self._data.empty:
            logger.error("Strategie nicht initialisiert oder keine Daten vorhanden")
            return pd.DataFrame()
        
        # Daten kopieren
        data = self._data.copy()
        
        # Parameter extrahieren
        lookback_period = self.parameters['lookback_period']
        entry_threshold = self.parameters['entry_threshold']
        exit_threshold = self.parameters['exit_threshold']
        use_bollinger = self.parameters['use_bollinger']
        bollinger_std = self.parameters['bollinger_std']
        volume_filter = self.parameters['volume_filter']
        rsi_filter = self.parameters['rsi_filter']
        rsi_period = self.parameters['rsi_period']
        rsi_overbought = self.parameters['rsi_overbought']
        rsi_oversold = self.parameters['rsi_oversold']
        
        # Berechnung des gleitenden Durchschnitts
        if f'sma_{lookback_period}' not in data.columns:
            data[f'sma_{lookback_period}'] = data['close'].rolling(window=lookback_period).mean()
        
        # Zwei verschiedene Ansätze: Z-Score oder Bollinger-Bänder
        if use_bollinger:
            # Bollinger-Bänder berechnen
            if 'bollinger_std' not in data.columns:
                data['bollinger_std'] = data['close'].rolling(window=lookback_period).std()
            if 'bollinger_upper' not in data.columns:
                data['bollinger_upper'] = data[f'sma_{lookback_period}'] + bollinger_std * data['bollinger_std']
            if 'bollinger_lower' not in data.columns:
                data['bollinger_lower'] = data[f'sma_{lookback_period}'] - bollinger_std * data['bollinger_std']
            
            # Berechnung des Z-Scores basierend auf Bollinger-Bändern
            # Z-Score = (Preis - Durchschnitt) / Standardabweichung
            data['z_score'] = (data['close'] - data[f'sma_{lookback_period}']) / data['bollinger_std']
        else:
            # Direkten Z-Score berechnen
            data['rolling_std'] = data['close'].rolling(window=lookback_period).std()
            data['z_score'] = (data['close'] - data[f'sma_{lookback_period}']) / data['rolling_std']
        
        # RSI berechnen falls notwendig
        if rsi_filter and 'rsi' not in data.columns:
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
        
        # Mean-Reversion-Logik:
        # 1. Long, wenn Preis stark unterbewertet ist (niedriger Z-Score)
        # 2. Short, wenn Preis stark überbewertet ist (hoher Z-Score)
        
        # Long-Signale (Z-Score < -entry_threshold)
        long_z_score = data['z_score'] < -entry_threshold
        
        # Optional RSI-Filter hinzufügen (nur Long, wenn RSI überverkauft ist)
        if rsi_filter:
            long_rsi = data['rsi'] < rsi_oversold
            long_condition = long_z_score & long_rsi & volume_filter_condition
        else:
            long_condition = long_z_score & volume_filter_condition
        
        # Short-Signale (Z-Score > entry_threshold)
        short_z_score = data['z_score'] > entry_threshold
        
        # Optional RSI-Filter hinzufügen (nur Short, wenn RSI überkauft ist)
        if rsi_filter:
            short_rsi = data['rsi'] > rsi_overbought
            short_condition = short_z_score & short_rsi & volume_filter_condition
        else:
            short_condition = short_z_score & volume_filter_condition
        
        # Signale setzen
        data.loc[long_condition, 'signal'] = 1
        data.loc[short_condition, 'signal'] = -1
        
        # NaNs entfernen (am Anfang wegen Rolling Windows)
        data = data.dropna()
        
        return data
