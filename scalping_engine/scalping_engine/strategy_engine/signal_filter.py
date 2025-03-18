import pandas as pd
import numpy as np
from loguru import logger


class SignalFilter:
    """
    Komponente zur Filterung von Handelssignalen basierend auf verschiedenen Kriterien.
    
    Wendet Filter wie Trendrichtung, RSI-Niveaus und Volatilitätsfilter auf generierte Signale an,
    um die Qualität der Signale zu verbessern.
    """
    
    def __init__(
        self,
        trend_filter_on: bool = True,
        trend_ema_period: int = 50,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        min_atr_pct: float = 0.5
    ):
        """
        Initialisiert den Signal-Filter.
        
        Args:
            trend_filter_on: Ob Trendfilter aktiviert werden soll
            trend_ema_period: EMA-Periode für Trendbestimmung
            rsi_oversold: RSI-Niveau für überverkaufte Bedingung
            rsi_overbought: RSI-Niveau für überkaufte Bedingung
            min_atr_pct: Mindest-ATR in Prozent für Trades
        """
        self.trend_filter_on = trend_filter_on
        self.trend_ema_period = trend_ema_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_atr_pct = min_atr_pct
    
    def apply_trend_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Wendet einen Trendfilter auf die Signale an.
        
        Long-Signale nur im Aufwärtstrend, Short-Signale nur im Abwärtstrend.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            signals: Serie mit Handelssignalen
            
        Returns:
            pd.Series: Gefilterte Signale
        """
        if not self.trend_filter_on or 'ema_trend' not in data.columns:
            return signals
        
        # Kopie der Signale erstellen
        filtered_signals = signals.copy()
        
        # Für jedes Signal prüfen
        for i in range(len(data)):
            if signals.iloc[i] != 0:
                # Long-Signal im Abwärtstrend oder Short-Signal im Aufwärtstrend filtern
                if ((signals.iloc[i] > 0 and data['close'].iloc[i] < data['ema_trend'].iloc[i]) or
                    (signals.iloc[i] < 0 and data['close'].iloc[i] > data['ema_trend'].iloc[i])):
                    filtered_signals.iloc[i] = 0
        
        # Anzahl der gefilterten Signale loggen
        removed_count = (signals != 0).sum() - (filtered_signals != 0).sum()
        if removed_count > 0:
            logger.debug(f"Trendfilter hat {removed_count} Signale entfernt")
        
        return filtered_signals
    
    def apply_rsi_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Wendet einen RSI-Filter an, um überkaufte/überverkaufte Situationen zu vermeiden.
        
        Long-Signale nicht im überkauften Bereich, Short-Signale nicht im überverkauften Bereich.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            signals: Serie mit Handelssignalen
            
        Returns:
            pd.Series: Gefilterte Signale
        """
        if 'rsi' not in data.columns:
            return signals
        
        # Kopie der Signale erstellen
        filtered_signals = signals.copy()
        
        # Für jedes Signal prüfen
        for i in range(len(data)):
            if signals.iloc[i] != 0:
                # Long-Signal im überkauften RSI-Bereich filtern
                if signals.iloc[i] > 0 and data['rsi'].iloc[i] > self.rsi_overbought:
                    filtered_signals.iloc[i] = 0
                
                # Short-Signal im überverkauften RSI-Bereich filtern
                elif signals.iloc[i] < 0 and data['rsi'].iloc[i] < self.rsi_oversold:
                    filtered_signals.iloc[i] = 0
        
        # Anzahl der gefilterten Signale loggen
        removed_count = (signals != 0).sum() - (filtered_signals != 0).sum()
        if removed_count > 0:
            logger.debug(f"RSI-Filter hat {removed_count} Signale entfernt")
        
        return filtered_signals
    
    def apply_macd_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Wendet einen MACD-Filter an, um Signale in Übereinstimmung mit dem MACD zu bringen.
        
        Long-Signale nur bei positivem MACD-Histogramm, Short-Signale nur bei negativem MACD-Histogramm.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            signals: Serie mit Handelssignalen
            
        Returns:
            pd.Series: Gefilterte Signale
        """
        if 'macd_histogram' not in data.columns:
            return signals
        
        # Kopie der Signale erstellen
        filtered_signals = signals.copy()
        
        # Für jedes Signal prüfen
        for i in range(len(data)):
            if signals.iloc[i] != 0:
                # Long-Signal mit negativem MACD-Histogramm filtern
                if signals.iloc[i] > 0 and data['macd_histogram'].iloc[i] < 0:
                    filtered_signals.iloc[i] = 0
                
                # Short-Signal mit positivem MACD-Histogramm filtern
                elif signals.iloc[i] < 0 and data['macd_histogram'].iloc[i] > 0:
                    filtered_signals.iloc[i] = 0
        
        # Anzahl der gefilterten Signale loggen
        removed_count = (signals != 0).sum() - (filtered_signals != 0).sum()
        if removed_count > 0:
            logger.debug(f"MACD-Filter hat {removed_count} Signale entfernt")
        
        return filtered_signals
    
    def apply_volatility_filter(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Wendet einen Volatilitätsfilter an, um zu geringe Volatilität zu vermeiden.
        
        Signale nur bei ausreichender Volatilität (ATR-Prozent).
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            signals: Serie mit Handelssignalen
            
        Returns:
            pd.Series: Gefilterte Signale
        """
        if 'atr_pct' not in data.columns:
            return signals
        
        # Kopie der Signale erstellen
        filtered_signals = signals.copy()
        
        # Für jedes Signal prüfen
        for i in range(len(data)):
            if signals.iloc[i] != 0:
                # Signal bei zu geringer Volatilität filtern
                if data['atr_pct'].iloc[i] < self.min_atr_pct:
                    filtered_signals.iloc[i] = 0
        
        # Anzahl der gefilterten Signale loggen
        removed_count = (signals != 0).sum() - (filtered_signals != 0).sum()
        if removed_count > 0:
            logger.debug(f"Volatilitätsfilter hat {removed_count} Signale entfernt")
        
        return filtered_signals
    
    def apply_filters(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        Wendet alle Filter auf die generierten Signale an.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            signals: Serie mit Handelssignalen
            
        Returns:
            pd.Series: Gefilterte Signale
        """
        # Kopie der Signale erstellen
        filtered_signals = signals.copy()
        
        # Trendfilter anwenden
        filtered_signals = self.apply_trend_filter(data, filtered_signals)
        
        # RSI-Filter anwenden
        filtered_signals = self.apply_rsi_filter(data, filtered_signals)
        
        # MACD-Filter anwenden
        filtered_signals = self.apply_macd_filter(data, filtered_signals)
        
        # Volatilitätsfilter anwenden
        filtered_signals = self.apply_volatility_filter(data, filtered_signals)
        
        # Gesamtergebnis loggen
        total_original = (signals != 0).sum()
        total_filtered = (filtered_signals != 0).sum()
        
        if total_original > 0:
            percentage = (total_filtered / total_original) * 100
            logger.info(f"Signalfilterung: {total_filtered}/{total_original} Signale behalten ({percentage:.1f}%)")
        
        return filtered_signals
