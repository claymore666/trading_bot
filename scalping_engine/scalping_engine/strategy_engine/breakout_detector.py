import pandas as pd
import numpy as np
from loguru import logger


class BreakoutDetector:
    """
    Komponente zur Erkennung von Breakouts aus Konsolidierungsphasen.
    
    Identifiziert Ausbrüche aus Konsolidierungszonen basierend auf Preisbewegungen
    und Volumenbestätigung.
    """
    
    def __init__(
        self,
        breakout_threshold_pct: float = 0.5,
        confirmation_periods: int = 1,
        volume_filter_on: bool = True,
        breakout_volume_factor: float = 1.5
    ):
        """
        Initialisiert den Breakout-Detektor.
        
        Args:
            breakout_threshold_pct: Prozentsatz für Ausbruchsbestätigung
            confirmation_periods: Anzahl der Perioden für Ausbruchsbestätigung
            volume_filter_on: Ob Volumenfilter aktiviert werden soll
            breakout_volume_factor: Volumenfaktor für Ausbruchsbestätigung
        """
        self.breakout_threshold_pct = breakout_threshold_pct / 100  # In Dezimal umwandeln
        self.confirmation_periods = confirmation_periods
        self.volume_filter_on = volume_filter_on
        self.breakout_volume_factor = breakout_volume_factor
    
    def detect_price_breakout(self, data: pd.DataFrame, i: int) -> int:
        """
        Erkennt einen Preisausbruch an einem bestimmten Index.
        
        Args:
            data: DataFrame mit Preisdaten
            i: Index des zu prüfenden Datenpunkts
            
        Returns:
            int: 1 für Aufwärtsausbruch, -1 für Abwärtsausbruch, 0 für keinen Ausbruch
        """
        # Prüfen, ob genügend Daten vorhanden sind
        if i < 1 or i >= len(data):
            return 0
        
        # Bandbreite berechnen
        bb_range = data['bollinger_upper'].iloc[i-1] - data['bollinger_lower'].iloc[i-1]
        threshold = bb_range * self.breakout_threshold_pct
        
        # Preisbewegung seit dem letzten Datenpunkt
        price_move = abs(data['close'].iloc[i] - data['close'].iloc[i-1])
        
        # Aufwärtsausbruch
        if (price_move > threshold and data['close'].iloc[i] > data['bollinger_upper'].iloc[i-1]):
            return 1
        
        # Abwärtsausbruch
        elif (price_move > threshold and data['close'].iloc[i] < data['bollinger_lower'].iloc[i-1]):
            return -1
        
        # Kein Ausbruch
        return 0
    
    def check_volume_confirmation(self, data: pd.DataFrame, i: int, lookback: int = 5) -> bool:
        """
        Prüft, ob ein Ausbruch durch erhöhtes Volumen bestätigt wird.
        
        Args:
            data: DataFrame mit Preis- und Volumendaten
            i: Index des zu prüfenden Datenpunkts
            lookback: Anzahl der zurückblickenden Perioden für den Volumenvergleich
            
        Returns:
            bool: True wenn Volumen erhöht ist, sonst False
        """
        # Volumenfilter deaktiviert oder keine Volumendaten vorhanden
        if not self.volume_filter_on or 'volume' not in data.columns:
            return True
        
        # Prüfen, ob genügend Daten vorhanden sind
        if i < lookback:
            return True
        
        # Durchschnittliches Volumen der vorherigen Perioden
        avg_volume = data['volume'].iloc[i-lookback:i].mean()
        
        # Aktuelle Periode hat erhöhtes Volumen
        return data['volume'].iloc[i] > avg_volume * self.breakout_volume_factor
    
    def check_breakout_quality(self, data: pd.DataFrame, i: int, direction: int) -> float:
        """
        Bewertet die Qualität eines Ausbruchs auf einer Skala von 0 bis 1.
        
        Args:
            data: DataFrame mit Preisdaten
            i: Index des zu prüfenden Datenpunkts
            direction: Richtung des Ausbruchs (1 für Aufwärts, -1 für Abwärts)
            
        Returns:
            float: Qualitätsbewertung des Ausbruchs (0 bis 1)
        """
        # Prüfen, ob genügend Daten vorhanden sind
        if i < 1 or i >= len(data):
            return 0.0
        
        quality_factors = []
        
        # Faktor 1: Stärke der Preisbewegung
        bb_range = data['bollinger_upper'].iloc[i-1] - data['bollinger_lower'].iloc[i-1]
        price_move = abs(data['close'].iloc[i] - data['close'].iloc[i-1])
        move_strength = min(1.0, price_move / (bb_range * 0.5))
        quality_factors.append(move_strength)
        
        # Faktor 2: Volumenbestätigung
        if 'volume' in data.columns:
            avg_volume = data['volume'].iloc[i-5:i].mean() if i >= 5 else data['volume'].iloc[:i].mean()
            volume_strength = min(1.0, data['volume'].iloc[i] / (avg_volume * self.breakout_volume_factor))
            quality_factors.append(volume_strength)
        
        # Faktor 3: Kerzenformation
        if direction > 0:  # Aufwärtsausbruch
            candle_body = max(0, data['close'].iloc[i] - data['open'].iloc[i])
            candle_range = data['high'].iloc[i] - data['low'].iloc[i]
        else:  # Abwärtsausbruch
            candle_body = max(0, data['open'].iloc[i] - data['close'].iloc[i])
            candle_range = data['high'].iloc[i] - data['low'].iloc[i]
        
        candle_strength = min(1.0, candle_body / (candle_range * 0.6)) if candle_range > 0 else 0
        quality_factors.append(candle_strength)
        
        # Gesamtqualität berechnen (Durchschnitt der Faktoren)
        return sum(quality_factors) / len(quality_factors)
    
    def detect_breakouts(self, data: pd.DataFrame, consolidation: pd.Series) -> pd.Series:
        """
        Erkennt Ausbrüche aus Konsolidierungsphasen.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            consolidation: Serie mit Konsolidierungsphasen
            
        Returns:
            pd.Series: Serie mit Ausbruchssignalen (1 für Long, -1 für Short, 0 für neutral)
        """
        # Ergebnis-Serie initialisieren
        breakout = pd.Series(0, index=data.index)
        
        # Für jeden Datenpunkt prüfen
        for i in range(self.confirmation_periods, len(data)):
            # Nur prüfen, wenn es eine Konsolidierungsphase war und jetzt keine mehr ist
            if i > 0 and consolidation.iloc[i-1] and not consolidation.iloc[i]:
                # Preisausbruch erkennen
                direction = self.detect_price_breakout(data, i)
                
                if direction != 0:
                    # Volumenbestätigung prüfen
                    volume_confirmed = self.check_volume_confirmation(data, i)
                    
                    # Qualität des Ausbruchs prüfen
                    breakout_quality = self.check_breakout_quality(data, i, direction)
                    
                    # Nur signifikante und volumenbestätigte Ausbrüche berücksichtigen
                    if volume_confirmed and breakout_quality >= 0.6:
                        # Aufwärts- oder Abwärtsausbruch
                        breakout.iloc[i] = direction
                        
                        logger.debug(f"Breakout erkannt bei {data.index[i]}: Richtung={direction}, "
                                     f"Qualität={breakout_quality:.2f}, Volumen={'bestätigt' if volume_confirmed else 'nicht bestätigt'}")
        
        return breakout
