import pandas as pd
import numpy as np
from loguru import logger


class ConsolidationDetector:
    """
    Komponente zur Erkennung von Konsolidierungsphasen im Markt.
    
    Identifiziert Konsolidierungsphasen durch die Erkennung von Bollinger-Band-Kontraktionen
    und anderen Volatilitätsmustern.
    """
    
    def __init__(
        self, 
        bb_period: int = 20, 
        bb_std_dev: float = 2.0,
        bb_squeeze_factor: float = 0.8,
        consolidation_periods: int = 5
    ):
        """
        Initialisiert den Konsolidierungsdetektor.
        
        Args:
            bb_period: Periode für Bollinger-Bänder
            bb_std_dev: Standardabweichungsfaktor für die Bänder
            bb_squeeze_factor: Kontraktionsfaktor (niedriger = stärkere Kontraktion)
            consolidation_periods: Mindestanzahl von Perioden in Konsolidierung
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.bb_squeeze_factor = bb_squeeze_factor
        self.consolidation_periods = consolidation_periods
    
    def check_bollinger_squeeze(self, data: pd.DataFrame) -> pd.Series:
        """
        Prüft, ob eine Bollinger-Band-Kontraktion (Squeeze) vorliegt.
        
        Args:
            data: DataFrame mit Preisdaten und Bollinger-Bändern
            
        Returns:
            pd.Series: Boolsche Serie, die angibt, wo Squeezes vorliegen
        """
        # Parameter überprüfen
        required_columns = ['bollinger_upper', 'bollinger_lower', 'close']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Benötigte Spalten für Bollinger-Squeeze fehlen: {required_columns}")
            return pd.Series(False, index=data.index)
        
        # Bandbreite berechnen (Prozentsatz vom Preis)
        bandwidth = (data['bollinger_upper'] - data['bollinger_lower']) / data['close'] * 100
        
        # Mittlere Bandbreite über die letzten 100 Perioden
        avg_bandwidth = bandwidth.rolling(window=100, min_periods=20).mean()
        
        # Squeeze identifizieren: Aktuelle Bandbreite < squeeze_factor * mittlere Bandbreite
        return bandwidth < (avg_bandwidth * self.bb_squeeze_factor)
    
    def check_low_volatility(self, data: pd.DataFrame) -> pd.Series:
        """
        Prüft auf niedrige Volatilität basierend auf ATR.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            pd.Series: Boolsche Serie für niedrige Volatilität
        """
        if 'atr_pct' not in data.columns:
            return pd.Series(False, index=data.index)
        
        # Mittlere ATR über die letzten 100 Perioden
        avg_atr = data['atr_pct'].rolling(window=100, min_periods=20).mean()
        
        # Niedrige Volatilität: ATR < 70% des Durchschnitts
        return data['atr_pct'] < (avg_atr * 0.7)
    
    def check_price_range(self, data: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Prüft, ob der Preis in einer engen Range verbleibt.
        
        Args:
            data: DataFrame mit Preisdaten
            window: Fenstergröße für die Berechnung der Range
            
        Returns:
            pd.Series: Boolsche Serie für enge Preisrange
        """
        # Preisrange über das Fenster
        rolling_high = data['high'].rolling(window=window).max()
        rolling_low = data['low'].rolling(window=window).min()
        price_range = (rolling_high - rolling_low) / data['close'] * 100
        
        # Mittlere Range über die letzten 100 Perioden
        avg_range = price_range.rolling(window=100, min_periods=20).mean()
        
        # Enge Range: Aktuelle Range < 60% der durchschnittlichen Range
        return price_range < (avg_range * 0.6)
    
    def detect_consolidation(self, data: pd.DataFrame) -> pd.Series:
        """
        Erkennt Konsolidierungsphasen im Preischart basierend auf mehreren Kriterien.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            
        Returns:
            pd.Series: Boolsche Serie, die Konsolidierungsphasen anzeigt
        """
        # Bollinger-Squeeze identifizieren
        squeeze = self.check_bollinger_squeeze(data)
        
        # Niedrige Volatilität identifizieren
        low_volatility = self.check_low_volatility(data)
        
        # Enge Preisrange identifizieren
        tight_range = self.check_price_range(data)
        
        # Kombiniertes Konsolidierungssignal
        # Eine Konsolidierung liegt vor, wenn mindestens zwei der drei Kriterien erfüllt sind
        raw_consolidation = (squeeze & low_volatility) | (squeeze & tight_range) | (low_volatility & tight_range)
        
        # Kontinuierliche Konsolidierung für mindestens N Perioden
        result = pd.Series(False, index=data.index)
        consolidation_count = 0
        
        for i in range(len(data)):
            if raw_consolidation.iloc[i]:
                consolidation_count += 1
            else:
                consolidation_count = 0
            
            result.iloc[i] = consolidation_count >= self.consolidation_periods
        
        return result
