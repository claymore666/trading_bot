import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from loguru import logger

class MarketAnalyzer:
    """
    Eine verbesserte Komponente zur Analyse von Marktdaten und Bestimmung optimaler Handelsstrategien.
    Bietet fortschrittliche Methoden zur Erkennung von Markttypen, Volatilität, Trendstärke und anderen
    wichtigen Eigenschaften für verschiedene Assetklassen.
    """
    
    # Asset-spezifische Parameter
    ASSET_PARAMETERS = {
        # Standard-Parameter für unbekannte Assets
        "default": {
            "volatility_thresholds": {"low": 1.0, "medium": 3.0},  # ATR in % vom Preis
            "trend_strength_thresholds": {"weak": 0.1, "moderate": 0.3},  # Trendstärke-Maß
            "mean_reversion_thresholds": {"weak": 0, "moderate": -0.2, "strong": -0.4},  # Autokorrelation
            "min_data_days": 60,  # Mindestanforderung an Datenmenge in Tagen
            "lookback_windows": {"short": 20, "medium": 50, "long": 200},  # Perioden für Analyse
            "volume_profile_bins": 50  # Anzahl der Bins für Volume Profile
        },
        # Bitcoin-spezifische Parameter
        "BTC/USDT": {
            "volatility_thresholds": {"low": 2.0, "medium": 5.0},  # Bitcoin hat höhere Volatilität
            "trend_strength_thresholds": {"weak": 0.15, "moderate": 0.35},
            "mean_reversion_thresholds": {"weak": 0, "moderate": -0.15, "strong": -0.3},
            "min_data_days": 120,  # Bitcoin benötigt mehr Daten für gute Analyse
            "lookback_windows": {"short": 20, "medium": 50, "long": 200},
            "volume_profile_bins": 100  # Mehr Bins für detaillierteres Volume Profile
        },
        # Aktien-Parameter (Standard)
        "STOCK": {
            "volatility_thresholds": {"low": 1.0, "medium": 2.5},
            "trend_strength_thresholds": {"weak": 0.08, "moderate": 0.25},
            "mean_reversion_thresholds": {"weak": 0, "moderate": -0.25, "strong": -0.45},
            "min_data_days": 252,  # Ein Handelsjahr
            "lookback_windows": {"short": 20, "medium": 50, "long": 200},
            "volume_profile_bins": 50
        },
        # Forex-Parameter
        "FOREX": {
            "volatility_thresholds": {"low": 0.5, "medium": 1.5},  # Forex typischerweise weniger volatil
            "trend_strength_thresholds": {"weak": 0.05, "moderate": 0.2},
            "mean_reversion_thresholds": {"weak": 0, "moderate": -0.3, "strong": -0.5},
            "min_data_days": 90,
            "lookback_windows": {"short": 20, "medium": 50, "long": 200},
            "volume_profile_bins": 30
        }
    }
    
    # Mapping von speziellen Assets zu Kategorien
    ASSET_CATEGORY_MAP = {
        "EUR/USD": "FOREX",
        "GBP/USD": "FOREX",
        "USD/JPY": "FOREX",
        "AAPL": "STOCK",
        "MSFT": "STOCK",
        "GOOGL": "STOCK",
        "TSLA": "STOCK"
    }
    
    def __init__(self, symbol: str):
        """
        Initialisiert den MarketAnalyzer für ein bestimmtes Symbol.
        
        Args:
            symbol: Das zu analysierende Handelssymbol (z.B. 'BTC/USDT', 'EUR/USD')
        """
        self.symbol = symbol
        
        # Parameter für das spezifische Asset bestimmen
        if symbol in self.ASSET_PARAMETERS:
            self.params = self.ASSET_PARAMETERS[symbol]
        elif symbol in self.ASSET_CATEGORY_MAP:
            category = self.ASSET_CATEGORY_MAP[symbol]
            self.params = self.ASSET_PARAMETERS[category]
        else:
            self.params = self.ASSET_PARAMETERS["default"]
            logger.info(f"Verwende Standard-Parameter für unbekanntes Symbol: {symbol}")
    
    def check_data_sufficiency(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prüft, ob genügend Daten für eine aussagekräftige Analyse vorhanden sind.
        
        Args:
            data: Dictionary mit DataFrames für verschiedene Zeitrahmen
            
        Returns:
            bool: True, wenn ausreichend Daten vorhanden sind, sonst False
        """
        # Mindestens ein Timeframe muss vorhanden sein
        if not data:
            logger.warning("Keine Daten für die Analyse vorhanden.")
            return False
        
        # Prüfen des längsten verfügbaren Zeitrahmens
        longest_tf = None
        max_days = 0
        
        for tf, df in data.items():
            if df.empty:
                continue
            
            # Berechnen der Anzahl von Tagen im DataFrame
            days = (df.index[-1] - df.index[0]).days
            
            if days > max_days:
                max_days = days
                longest_tf = tf
        
        # Prüfen, ob die Mindestanforderung erfüllt ist
        min_required = self.params["min_data_days"]
        
        if max_days < min_required:
            logger.warning(f"Nicht genügend Daten für robuste Analyse. Vorhanden: {max_days} Tage, " 
                          f"Erforderlich: {min_required} Tage. Verwende verfügbare Daten.")
            # Wir setzen trotzdem True, arbeiten aber mit einem Warnhinweis
            return True
        
        return True
    
    def analyze_volatility(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Analysiert die Volatilität des Assets basierend auf ATR.
        
        Args:
            data: DataFrame mit Preisdaten und Indikatoren
            
        Returns:
            Tuple[str, float]: Volatilitätskategorie und ATR-Prozent-Wert
        """
        # ATR berechnen, falls nicht vorhanden
        if 'atr' not in data.columns:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr'] = true_range.rolling(window=14).mean()
        
        # ATR als Prozentsatz vom Preis
        data['atr_pct'] = (data['atr'] / data['close'] * 100)
        
        # Mittlerer ATR-Prozentsatz für die letzten 20 Tage
        atr_pct = data['atr_pct'].tail(20).mean()
        
        # Volatilitätskategorie basierend auf Schwellenwerten
        thresholds = self.params["volatility_thresholds"]
        
        if atr_pct < thresholds["low"]:
            return "low", atr_pct
        elif atr_pct < thresholds["medium"]:
            return "medium", atr_pct
        else:
            return "high", atr_pct
    
    def analyze_trend_strength(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Analysiert die Stärke des Trends mit mehreren Methoden.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            Tuple[str, float]: Trendstärke-Kategorie und -Wert
        """
        # ADX berechnen, falls nicht in Daten
        if 'adx' not in data.columns:
            # Vereinfachte ADX-Berechnung
            # Für echte Implementierung würde eine vollständige ADX-Berechnung benötigt
            # Hier verwenden wir eine vereinfachte Annäherung
            
            # Gleitender Durchschnitt
            lookback = self.params["lookback_windows"]["medium"]
            data['ma'] = data['close'].rolling(window=lookback).mean()
            
            # Preisänderungen
            data['price_change'] = data['close'].pct_change(lookback)
            
            # Annäherung der Trendstärke durch Verhältnis von Richtung zu Volatilität
            price_changes = data['close'].pct_change(lookback)
            trend_strength = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0
        else:
            # Falls ADX bereits vorhanden
            adx_values = data['adx'].tail(20).values
            trend_strength = adx_values.mean() / 100.0  # Normalisieren auf 0-1
        
        # Trendstärke-Kategorie basierend auf Schwellenwerten
        thresholds = self.params["trend_strength_thresholds"]
        
        if trend_strength < thresholds["weak"]:
            return "weak", trend_strength
        elif trend_strength < thresholds["moderate"]:
            return "moderate", trend_strength
        else:
            return "strong", trend_strength
    
    def analyze_mean_reversion(self, data: pd.DataFrame) -> Tuple[str, float]:
        """
        Analysiert Mean-Reversion-Eigenschaften durch mehrere statistische Methoden.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            Tuple[str, float]: Mean-Reversion-Kategorie und Autokorrelationswert
        """
        # Mindestens 30 Datenpunkte für Autokorrelation
        if len(data) < 30:
            return "unknown", 0
        
        # Berechnung der Renditen
        returns = data['close'].pct_change().dropna()
        
        # Autokorrelation der Renditen (Lag 1)
        if len(returns) > 1:
            # Pandas autocorrelation mit lag=1
            autocorr = returns.autocorr(lag=1)
        else:
            autocorr = 0
        
        # Negative Autokorrelation deutet auf Mean Reversion hin
        # Positive Autokorrelation deutet auf Momentum/Trend hin
        thresholds = self.params["mean_reversion_thresholds"]
        
        if autocorr <= thresholds["strong"]:
            return "strong", autocorr
        elif autocorr <= thresholds["moderate"]:
            return "moderate", autocorr
        elif autocorr <= thresholds["weak"]:
            return "weak", autocorr
        else:
            return "momentum", autocorr  # Positive Autokorrelation = Momentum
    
    def analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identifiziert wichtige Support- und Resistance-Levels.
        
        Args:
            data: DataFrame mit Preisdaten
            
        Returns:
            Dict[str, List[float]]: Support- und Resistance-Levels
        """
        if len(data) < 50:
            return {"support": [], "resistance": []}
        
        # Zur Vereinfachung verwenden wir hier Swing-Hochs und -Tiefs
        lookback = 10  # Anzahl der Perioden für Swing-Punkt-Identifikation
        
        # Listen für Swing-Hochs und -Tiefs
        swing_highs = []
        swing_lows = []
        
        # Preisdaten
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # Iteration über die Daten (wir ignorieren die ersten und letzten lookback Perioden)
        for i in range(lookback, len(data) - lookback):
            # Prüfen, ob es sich um ein Swing-Hoch handelt
            is_swing_high = True
            current_high = high_prices[i]
            
            for j in range(1, lookback + 1):
                if high_prices[i - j] > current_high or high_prices[i + j] > current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(current_high)
            
            # Prüfen, ob es sich um ein Swing-Tief handelt
            is_swing_low = True
            current_low = low_prices[i]
            
            for j in range(1, lookback + 1):
                if low_prices[i - j] < current_low or low_prices[i + j] < current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(current_low)
        
        # Gruppieren ähnlicher Levels (Clustering)
        def cluster_levels(levels, proximity_pct=0.5):
            if not levels:
                return []
            
            current_price = data['close'].iloc[-1]
            proximity = current_price * proximity_pct / 100
            
            sorted_levels = sorted(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if level - current_cluster[-1] <= proximity:
                    current_cluster.append(level)
                else:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
            
            if current_cluster:
                clusters.append(sum(current_cluster) / len(current_cluster))
            
            return clusters
        
        support_levels = cluster_levels(swing_lows)
        resistance_levels = cluster_levels(swing_highs)
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analysiert das Volumen-Profil, um wichtige Preisniveaus zu identifizieren.
        
        Args:
            data: DataFrame mit Preis- und Volumendaten
            
        Returns:
            Dict[str, Any]: Volume-Profile-Analyseergebnisse
        """
        if 'volume' not in data.columns or len(data) < 50:
            return {"poc": None, "value_area": {"low": None, "high": None}}
        
        # Preisspanne definieren
        price_min = data['low'].min()
        price_max = data['high'].max()
        
        # Anzahl der Bins für das Histogramm
        bins = self.params["volume_profile_bins"]
        
        # Preisbereiche erstellen
        price_ranges = np.linspace(price_min, price_max, bins + 1)
        
        # Volumen pro Preisbereich berechnen
        volume_by_price = np.zeros(bins)
        
        for i in range(len(data)):
            for j in range(bins):
                if data['low'].iloc[i] <= price_ranges[j+1] and data['high'].iloc[i] >= price_ranges[j]:
                    # Volumen proportional zum überlappenden Preisbereich zuweisen
                    overlap_low = max(data['low'].iloc[i], price_ranges[j])
                    overlap_high = min(data['high'].iloc[i], price_ranges[j+1])
                    overlap_ratio = (overlap_high - overlap_low) / (data['high'].iloc[i] - data['low'].iloc[i])
                    volume_by_price[j] += data['volume'].iloc[i] * overlap_ratio
        
        # Point of Control (POC) - Preisniveau mit dem höchsten Volumen
        poc_index = np.argmax(volume_by_price)
        poc_price = (price_ranges[poc_index] + price_ranges[poc_index + 1]) / 2
        
        # Value Area berechnen (70% des gesamten Volumens)
        total_volume = np.sum(volume_by_price)
        target_volume = total_volume * 0.7
        
        # Von POC aus in beide Richtungen erweitern
        current_volume = volume_by_price[poc_index]
        va_lower_idx = poc_index
        va_upper_idx = poc_index
        
        while current_volume < target_volume and (va_lower_idx > 0 or va_upper_idx < bins - 1):
            # Bestimmen, ob nach unten oder oben erweitern
            add_lower = False
            add_upper = False
            
            if va_lower_idx > 0 and va_upper_idx < bins - 1:
                if volume_by_price[va_lower_idx - 1] > volume_by_price[va_upper_idx + 1]:
                    add_lower = True
                else:
                    add_upper = True
            elif va_lower_idx > 0:
                add_lower = True
            elif va_upper_idx < bins - 1:
                add_upper = True
            
            # Value Area erweitern
            if add_lower:
                va_lower_idx -= 1
                current_volume += volume_by_price[va_lower_idx]
            if add_upper:
                va_upper_idx += 1
                current_volume += volume_by_price[va_upper_idx]
        
        value_area_low = price_ranges[va_lower_idx]
        value_area_high = price_ranges[va_upper_idx + 1]
        
        return {
            "poc": poc_price,
            "value_area": {
                "low": value_area_low,
                "high": value_area_high
            }
        }
    
    def determine_market_type(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Bestimmt den Markttyp basierend auf allen verfügbaren Analysen.
        
        Args:
            data: Dictionary mit DataFrames für verschiedene Zeitrahmen
            
        Returns:
            Dict[str, Any]: Markttyp und detaillierte Analyseergebnisse
        """
        # Prüfen, ob genügend Daten vorhanden sind
        if not self.check_data_sufficiency(data):
            return {
                "market_type": "unknown",
                "reason": "Nicht genügend Daten für Analyse",
                "details": {}
            }
        
        # Primären Zeitrahmen für die Analyse bestimmen (bevorzugt 1h, dann 4h, dann täglich)
        primary_tf = None
        for tf in ["1h", "4h", "1d"]:
            if tf in data and not data[tf].empty:
                primary_tf = tf
                break
        
        if not primary_tf:
            # Wenn keiner der bevorzugten Zeitrahmen verfügbar ist, den ersten nehmen
            primary_tf = list(data.keys())[0]
        
        primary_data = data[primary_tf]
        
        # Verschiedene Marktaspekte analysieren
        volatility, volatility_value = self.analyze_volatility(primary_data)
        trend_strength, trend_value = self.analyze_trend_strength(primary_data)
        mean_reversion, mr_value = self.analyze_mean_reversion(primary_data)
        
        support_resistance = self.analyze_support_resistance(primary_data)
        volume_profile = self.analyze_volume_profile(primary_data)
        
        # Markttyp basierend auf Analysen bestimmen
        market_type = "unknown"
        reason = ""
        
        # Regel 1: Strong trend + low to medium mean reversion = trending
        if trend_strength == "strong" and mean_reversion != "strong":
            market_type = "trending"
            reason = "Starker Trend mit niedriger Mean Reversion"
        # Regel 2: Moderate/strong mean reversion = mean_reverting
        elif mean_reversion in ["moderate", "strong"]:
            market_type = "mean_reverting"
            reason = f"{mean_reversion.capitalize()} Mean Reversion erkannt"
        # Regel 3: High volatility + weak trend + weak mean reversion = volatile
        elif volatility == "high" and trend_strength == "weak" and mean_reversion == "weak":
            market_type = "volatile"
            reason = "Hohe Volatilität ohne klaren Trend oder Mean Reversion"
        # Regel 4: Low volatility + weak trend + weak mean reversion = ranging
        elif volatility == "low" and trend_strength == "weak" and mean_reversion == "weak":
            market_type = "ranging"
            reason = "Niedrige Volatilität in seitwärts gerichtetem Markt"
        # Regel 5: Medium volatility + moderate trend = mixed
        else:
            market_type = "mixed"
            reason = "Gemischte Marktsignale ohne dominantes Muster"
        
        # Detaillierte Ergebnisse zusammenstellen
        details = {
            "volatility": {
                "category": volatility,
                "value": volatility_value
            },
            "trend_strength": {
                "category": trend_strength,
                "value": trend_value
            },
            "mean_reversion": {
                "category": mean_reversion,
                "value": mr_value
            },
            "support_resistance": support_resistance,
            "volume_profile": volume_profile,
            "timeframe_used": primary_tf,
            "data_points": len(primary_data),
            "symbol": self.symbol
        }
        
        return {
            "market_type": market_type,
            "reason": reason,
            "details": details
        }
    
    def suggest_strategies(self, market_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Schlägt Handelsstrategien basierend auf der Marktanalyse vor.
        
        Args:
            market_analysis: Ergebnisse der Marktanalyse
            
        Returns:
            List[Dict[str, Any]]: Liste empfohlener Strategien mit Begründungen und Parametern
        """
        market_type = market_analysis.get("market_type", "unknown")
        details = market_analysis.get("details", {})
        
        # Strategieempfehlungen basierend auf Markttyp
        strategies = []
        
        if market_type == "trending":
            # Strategien für Trendmärkte
            strategies.append({
                "strategy": "MovingAverageCrossStrategy",
                "reason": "Eignet sich gut für trendstarke Märkte mit klaren Richtungsbewegungen.",
                "suggested_params": {
                    "fast_ma": 10,
                    "slow_ma": 30,
                    "signal_ma": 9,
                    "use_ema": True
                }
            })
            
            strategies.append({
                "strategy": "ParabolicSARStrategy",
                "reason": "Effektiv für die Verfolgung von Trends mit automatischer Anpassung an die Marktdynamik.",
                "suggested_params": {
                    "initial_af": 0.02,
                    "max_af": 0.2,
                    "trend_filter": True
                }
            })
            
            # Wenn starker Trend mit moderater Volatilität
            if details.get("volatility", {}).get("category") in ["medium", "high"]:
                strategies.append({
                    "strategy": "BreakoutStrategy",
                    "reason": "Kann neue Trendfortsetzungen nach Konsolidierungen identifizieren.",
                    "suggested_params": {
                        "bb_squeeze_factor": 0.7,
                        "breakout_threshold_pct": 0.5,
                        "volume_filter_on": True
                    }
                })
        
        elif market_type == "mean_reverting":
            # Strategien für Mean-Reverting Märkte
            strategies.append({
                "strategy": "BollingerBandsStrategy",
                "reason": "Optimal für Märkte mit Mean-Reversion-Eigenschaften und klaren Unterstützungs-/Widerstandsbereichen.",
                "suggested_params": {
                    "window": 20,
                    "std_dev": 2.0,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            })
            
            strategies.append({
                "strategy": "MeanReversionStrategy",
                "reason": "Speziell für Mean-Reversion-Dynamik entwickelt mit statistischen Abweichungen vom Mittelwert.",
                "suggested_params": {
                    "lookback_period": 20,
                    "z_score_threshold": 2.0,
                    "rsi_filter": True
                }
            })
            
            # Starke Mean Reversion könnte auch RSI-Extremwerte nutzen
            if details.get("mean_reversion", {}).get("category") == "strong":
                strategies.append({
                    "strategy": "RSIOscillatorStrategy",  # Hypothetische Strategie
                    "reason": "Nutzt extreme RSI-Werte in stark mean-reverting Märkten.",
                    "suggested_params": {
                        "rsi_period": 7,  # Kürzerer Zeitraum für schnellere Reaktion
                        "rsi_overbought": 75,
                        "rsi_oversold": 25,
                        "volume_confirmation": True
                    }
                })
        
        elif market_type == "volatile":
            # Strategien für volatile Märkte
            strategies.append({
                "strategy": "BreakoutStrategy",
                "reason": "Nutzt Ausbrüche aus Konsolidierungsphasen in volatilen Märkten.",
                "suggested_params": {
                    "bb_squeeze_factor": 0.7,
                    "breakout_threshold_pct": 0.5,
                    "volume_filter_on": True
                }
            })
            
            strategies.append({
                "strategy": "BollingerBandsStrategy",
                "reason": "Kann in volatilen Märkten effektiv sein durch Identifikation von Extremwerten.",
                "suggested_params": {
                    "window": 20,
                    "std_dev": 2.5,  # Höhere Standardabweichung für volatile Märkte
                    "rsi_period": 14,
                    "rsi_overbought": 75,  # Angepasste RSI-Niveaus für Volatilität
                    "rsi_oversold": 25
                }
            })
            
            # Für höchst volatile Märkte auch Support/Resistance beachten
            strategies.append({
                "strategy": "SupportResistanceStrategy",
                "reason": "Identifiziert Schlüsselniveaus für Ein- und Ausstiege in volatilen Märkten.",
                "suggested_params": {
                    "lookback_periods": 20,
                    "threshold_pct": 0.5,
                    "num_touches": 2,
                    "volume_filter": True
                }
            })
        
        elif market_type == "ranging":
            # Strategien für Seitwärtsmärkte
            strategies.append({
                "strategy": "BollingerBandsStrategy",
                "reason": "Effektiv in Seitwärtsmärkten durch Handel zwischen oberen und unteren Bändern.",
                "suggested_params": {
                    "window": 20,
                    "std_dev": 1.8,  # Leicht engere Bänder für Seitwärtsmärkte
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            })
            
            strategies.append({
                "strategy": "SupportResistanceStrategy",
                "reason": "Ideal für Handel innerhalb definierter Bereiche in Seitwärtsmärkten.",
                "suggested_params": {
                    "lookback_periods": 20,
                    "threshold_pct": 0.3,  # Niedrigerer Schwellenwert für enge Ranges
                    "num_touches": 2,
                    "volume_filter": True
                }
            })
        
        else:  # mixed oder unknown
            # Allgemeine Strategien für gemischte Märkte
            strategies.append({
                "strategy": "BollingerBandsStrategy",
                "reason": "Vielseitige Strategie, die in verschiedenen Marktbedingungen funktionieren kann.",
                "suggested_params": {
                    "window": 20,
                    "std_dev": 2.0,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            })
            
            strategies.append({
                "strategy": "MovingAverageCrossStrategy",
                "reason": "Klassische Strategie, die in verschiedenen Marktbedingungen funktionieren kann.",
                "suggested_params": {
                    "fast_ma": 10,
                    "slow_ma": 30,
                    "signal_ma": 9,
                    "use_ema": True
                }
            })
            
            strategies.append({
                "strategy": "AdaptiveMovingAverageStrategy",
                "reason": "Passt sich an wechselnde Marktbedingungen durch adaptive Indikatoren an.",
                "suggested_params": {
                    "ma_type": "kama",  # Kaufman's Adaptive Moving Average
                    "fast_kama_period": 10,
"slow_kama_period": 30,
                    "signal_period": 9,
                    "trend_filter": True
                }
            })
        
        # Assetspezifische Anpassungen für die empfohlenen Strategien
        if self.symbol.endswith("/USDT") or self.symbol in ["BTC/USD", "ETH/USD"]:
            # Krypto-spezifische Anpassungen
            for strategy in strategies:
                if strategy["strategy"] == "BollingerBandsStrategy":
                    # Krypto ist volatiler, daher leicht breitere Bänder
                    strategy["suggested_params"]["std_dev"] += 0.3
                elif strategy["strategy"] == "MovingAverageCrossStrategy":
                    # Schnellere Perioden für Kryptowährungen
                    strategy["suggested_params"]["fast_ma"] = max(5, strategy["suggested_params"]["fast_ma"] - 2)
                    strategy["suggested_params"]["slow_ma"] = max(20, strategy["suggested_params"]["slow_ma"] - 5)
        elif self.symbol in self.ASSET_CATEGORY_MAP and self.ASSET_CATEGORY_MAP[self.symbol] == "FOREX":
            # Forex-spezifische Anpassungen
            for strategy in strategies:
                if strategy["strategy"] == "BollingerBandsStrategy":
                    # Engere Bänder für Forex
                    strategy["suggested_params"]["std_dev"] = max(1.5, strategy["suggested_params"]["std_dev"] - 0.3)
        
        # Volatilitätsbasierte Anpassungen für Stop-Loss und Risk Management
        volatility_category = details.get("volatility", {}).get("category", "medium")
        for strategy in strategies:
            if volatility_category == "high":
                strategy["risk_management"] = {
                    "position_size_pct": 1.0,  # Kleinere Position bei hoher Volatilität
                    "stop_loss_atr_multiplier": 2.5,  # Weiterer Stop-Loss bei hoher Volatilität
                    "take_profit_ratio": 2.0    # Risk-Reward-Ratio anpassen
                }
            elif volatility_category == "medium":
                strategy["risk_management"] = {
                    "position_size_pct": 1.5,  # Standard-Position
                    "stop_loss_atr_multiplier": 2.0,
                    "take_profit_ratio": 1.8
                }
            else:  # "low"
                strategy["risk_management"] = {
                    "position_size_pct": 2.0,  # Größere Position bei niedriger Volatilität
                    "stop_loss_atr_multiplier": 1.5,  # Engerer Stop-Loss möglich
                    "take_profit_ratio": 1.5
                }
        
        return strategies
    
    def optimize_strategy_parameters(self, 
                                    strategy_name: str, 
                                    base_params: Dict[str, Any], 
                                    market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimiert die Strategieparameter basierend auf der Marktanalyse.
        
        Args:
            strategy_name: Name der zu optimierenden Strategie
            base_params: Basis-Parameter der Strategie
            market_analysis: Ergebnisse der Marktanalyse
            
        Returns:
            Dict[str, Any]: Optimierte Strategieparameter
        """
        # Kopie der Basis-Parameter erstellen, um das Original nicht zu verändern
        optimized_params = base_params.copy()
        
        market_type = market_analysis.get("market_type", "unknown")
        details = market_analysis.get("details", {})
        
        # Extrahiere Marktdetails für die Optimierung
        volatility = details.get("volatility", {}).get("category", "medium")
        trend_strength = details.get("trend_strength", {}).get("category", "moderate")
        mean_reversion = details.get("mean_reversion", {}).get("category", "moderate")
        
        # Strategie-spezifische Optimierungen
        if strategy_name == "BollingerBandsStrategy":
            # Bollinger-Band-Parameter anpassen
            if volatility == "high":
                optimized_params["std_dev"] = min(3.0, optimized_params.get("std_dev", 2.0) + 0.5)
            elif volatility == "low":
                optimized_params["std_dev"] = max(1.5, optimized_params.get("std_dev", 2.0) - 0.3)
            
            # RSI-Parameter basierend auf Mean Reversion anpassen
            if mean_reversion == "strong":
                optimized_params["rsi_overbought"] = min(75, optimized_params.get("rsi_overbought", 70) + 5)
                optimized_params["rsi_oversold"] = max(25, optimized_params.get("rsi_oversold", 30) - 5)
            elif mean_reversion == "weak":
                optimized_params["rsi_overbought"] = max(65, optimized_params.get("rsi_overbought", 70) - 5)
                optimized_params["rsi_oversold"] = min(35, optimized_params.get("rsi_oversold", 30) + 5)
        
        elif strategy_name == "MovingAverageCrossStrategy":
            # Moving-Average-Parameter anpassen
            if trend_strength == "strong":
                # Bei starkem Trend können MAs etwas langsamer sein
                optimized_params["fast_ma"] = min(15, optimized_params.get("fast_ma", 10) + 3)
                optimized_params["slow_ma"] = min(50, optimized_params.get("slow_ma", 30) + 10)
            elif trend_strength == "weak":
                # Bei schwachem Trend schnellere MAs für frühere Signale
                optimized_params["fast_ma"] = max(5, optimized_params.get("fast_ma", 10) - 3)
                optimized_params["slow_ma"] = max(20, optimized_params.get("slow_ma", 30) - 5)
            
            # EMA vs SMA basierend auf Volatilität
            optimized_params["use_ema"] = volatility in ["medium", "high"]
        
        elif strategy_name == "BreakoutStrategy":
            # Breakout-Parameter anpassen
            if volatility == "high":
                optimized_params["breakout_threshold_pct"] = min(1.0, optimized_params.get("breakout_threshold_pct", 0.5) + 0.3)
            elif volatility == "low":
                optimized_params["breakout_threshold_pct"] = max(0.2, optimized_params.get("breakout_threshold_pct", 0.5) - 0.2)
            
            # Volumen-Filter basierend auf Markttyp
            optimized_params["volume_filter_on"] = market_type in ["trending", "volatile"]
        
        elif strategy_name == "MeanReversionStrategy":
            # Z-Score-Threshold basierend auf Mean-Reversion-Stärke anpassen
            if mean_reversion == "strong":
                optimized_params["z_score_threshold"] = max(1.5, optimized_params.get("z_score_threshold", 2.0) - 0.3)
            elif mean_reversion == "weak":
                optimized_params["z_score_threshold"] = min(2.5, optimized_params.get("z_score_threshold", 2.0) + 0.3)
            
            # Lookback-Periode basierend auf Volatilität
            if volatility == "high":
                optimized_params["lookback_period"] = min(30, optimized_params.get("lookback_period", 20) + 5)
            elif volatility == "low":
                optimized_params["lookback_period"] = max(10, optimized_params.get("lookback_period", 20) - 5)
        
        elif strategy_name == "ParabolicSARStrategy":
            # SAR-Parameter basierend auf Trend und Volatilität anpassen
            if trend_strength == "strong":
                optimized_params["initial_af"] = min(0.03, optimized_params.get("initial_af", 0.02) + 0.005)
                optimized_params["max_af"] = min(0.3, optimized_params.get("max_af", 0.2) + 0.05)
            elif volatility == "high":
                optimized_params["initial_af"] = max(0.01, optimized_params.get("initial_af", 0.02) - 0.005)
                optimized_params["max_af"] = min(0.25, optimized_params.get("max_af", 0.2) + 0.02)
        
        # Support/Resistance-Elemente für jede Strategie berücksichtigen
        support_levels = details.get("support_resistance", {}).get("support", [])
        resistance_levels = details.get("support_resistance", {}).get("resistance", [])
        
        if support_levels and resistance_levels:
            # Aktuelle Preisnähe zu Support oder Resistance überprüfen
            current_price = details.get("current_price", None)
            if current_price:
                # Finde nächsten Support und Resistance-Level
                next_support = max([s for s in support_levels if s < current_price], default=None)
                next_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                
                if next_support and next_resistance:
                    # Berechne Distanz zu Support und Resistance
                    support_distance = (current_price - next_support) / current_price
                    resistance_distance = (next_resistance - current_price) / current_price
                    
                    # Optimiere basierend auf relativer Nähe
                    if support_distance < resistance_distance * 0.5:
                        # Nahe am Support, Long-Bias optimieren
                        if strategy_name == "BollingerBandsStrategy":
                            optimized_params["rsi_oversold"] = max(20, optimized_params.get("rsi_oversold", 30) - 5)
                    elif resistance_distance < support_distance * 0.5:
                        # Nahe am Resistance, Short-Bias optimieren
                        if strategy_name == "BollingerBandsStrategy":
                            optimized_params["rsi_overbought"] = min(80, optimized_params.get("rsi_overbought", 70) + 5)
        
        # Volume Profile für jede Strategie berücksichtigen
        volume_profile = details.get("volume_profile", {})
        poc = volume_profile.get("poc")
        value_area = volume_profile.get("value_area", {})
        
        if poc and value_area.get("low") and value_area.get("high"):
            # Value Area kann für Range-Strategien nützlich sein
            if strategy_name == "BollingerBandsStrategy" and market_type == "ranging":
                va_width = (value_area["high"] - value_area["low"]) / poc
                if va_width < 0.03:  # Enge Value Area
                    optimized_params["std_dev"] = max(1.2, optimized_params.get("std_dev", 2.0) - 0.4)
                elif va_width > 0.08:  # Breite Value Area
                    optimized_params["std_dev"] = min(2.5, optimized_params.get("std_dev", 2.0) + 0.3)
        
        # Assetspezifische finale Anpassungen
        if self.symbol.endswith("/USDT") or "BTC" in self.symbol or "ETH" in self.symbol:
            # Krypto-spezifische Parameter-Tweaks
            if "volume_filter" in optimized_params:
                optimized_params["volume_filter"] = True  # Volumen ist wichtig in Krypto
            
            if "atr_multiplier" in optimized_params:
                optimized_params["atr_multiplier"] += 0.2  # Leicht weiterer Stop für Krypto
        
        return optimized_params
    
    def get_risk_management_recommendations(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gibt Risikomanagement-Empfehlungen basierend auf der Marktanalyse.
        
        Args:
            market_analysis: Ergebnisse der Marktanalyse
            
        Returns:
            Dict[str, Any]: Risikomanagement-Empfehlungen
        """
        details = market_analysis.get("details", {})
        market_type = market_analysis.get("market_type", "unknown")
        
        # Extrahiere relevante Marktdetails
        volatility = details.get("volatility", {}).get("category", "medium")
        trend_strength = details.get("trend_strength", {}).get("category", "moderate")
        
        # Grundlegende Risikomanagement-Parameter basierend auf Volatilität
        if volatility == "high":
            position_size_pct = 1.0  # Weniger Risiko bei hoher Volatilität
            stop_loss_atr = 2.5      # Weiterer Stop bei hoher Volatilität
            risk_per_trade = 1.0     # Geringeres Risiko pro Trade
        elif volatility == "medium":
            position_size_pct = 1.5
            stop_loss_atr = 2.0
            risk_per_trade = 1.2
        else:  # "low"
            position_size_pct = 2.0
            stop_loss_atr = 1.5
            risk_per_trade = 1.5
        
        # Risk-Reward-Ratio basierend auf Markttyp
        if market_type == "trending":
            risk_reward_ratio = 1.5  # Niedriger RRR in Trendmärkten, aber höhere Win-Rate
        elif market_type == "volatile":
            risk_reward_ratio = 2.5  # Höherer RRR in volatilen Märkten, um niedrigere Win-Rate auszugleichen
        else:
            risk_reward_ratio = 2.0  # Standard
        
        # Korrelationsverwaltung und maximale Exposition
        max_correlated_positions = 2 if market_type == "volatile" else 3
        max_exposure_pct = 6 if volatility == "high" else (8 if volatility == "medium" else 10)
        
        # Take-Profit-Strategien basierend auf Markttyp
        take_profit_strategies = []
        
        if market_type == "trending":
            take_profit_strategies.append({
                "type": "trailing_stop",
                "settings": {"activation_pct": 1.0, "trail_pct": 0.8}
            })
        elif market_type == "mean_reverting":
            take_profit_strategies.append({
                "type": "fixed_target",
                "settings": {"target_atr_multiple": 1.5}
            })
        elif market_type == "volatile":
            take_profit_strategies.append({
                "type": "partial_exits",
                "settings": {"levels": [1.0, 2.0, 3.0], "percentages": [30, 30, 40]}
            })
        else:
            take_profit_strategies.append({
                "type": "fixed_target",
                "settings": {"target_atr_multiple": 2.0}
            })
        
        # Assetspezifische Anpassungen
        if self.symbol.endswith("/USDT") or "BTC" in self.symbol or "ETH" in self.symbol:
            # Krypto erfordert vorsichtigeres Risikomanagement
            position_size_pct = max(0.8, position_size_pct - 0.2)
            max_exposure_pct = max(5, max_exposure_pct - 1)
        elif self.symbol in self.ASSET_CATEGORY_MAP and self.ASSET_CATEGORY_MAP[self.symbol] == "FOREX":
            # Forex kann größere Positionsgrößen haben, da Volatilität oft niedriger ist
            position_size_pct = min(3.0, position_size_pct + 0.5)
        
        return {
            "position_size_pct": position_size_pct,
            "stop_loss": {
                "type": "atr_based",
                "atr_multiple": stop_loss_atr
            },
            "take_profit": take_profit_strategies,
            "risk_per_trade": risk_per_trade,
            "risk_reward_ratio": risk_reward_ratio,
            "max_correlated_positions": max_correlated_positions,
            "max_exposure_pct": max_exposure_pct,
            "volatility_adjustment": volatility == "high"
        }
    
    def get_comprehensive_trading_plan(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generiert einen umfassenden Handelsplan basierend auf Marktanalyse und Strategieempfehlungen.
        
        Args:
            data: Dictionary mit DataFrames für verschiedene Zeitrahmen
            
        Returns:
            Dict[str, Any]: Umfassender Handelsplan
        """
        # Marktanalyse durchführen
        market_analysis = self.determine_market_type(data)
        
        # Strategien vorschlagen
        recommended_strategies = self.suggest_strategies(market_analysis)
        
        # Risikomanagement-Empfehlungen
        risk_management = self.get_risk_management_recommendations(market_analysis)
        
        # Optimierte Parameter für empfohlene Strategien
        optimized_strategies = []
        for strategy in recommended_strategies:
            strategy_name = strategy["strategy"]
            base_params = strategy["suggested_params"]
            
            optimized_params = self.optimize_strategy_parameters(
                strategy_name=strategy_name,
                base_params=base_params,
                market_analysis=market_analysis
            )
            
            optimized_strategies.append({
                "strategy": strategy_name,
                "reason": strategy["reason"],
                "original_params": base_params,
                "optimized_params": optimized_params
            })
        
        # Marktbedingungen zusammenfassen
        market_conditions = {
            "type": market_analysis["market_type"],
            "reason": market_analysis["reason"],
            "volatility": market_analysis["details"].get("volatility", {}).get("category", "unknown"),
            "trend_strength": market_analysis["details"].get("trend_strength", {}).get("category", "unknown"),
            "mean_reversion": market_analysis["details"].get("mean_reversion", {}).get("category", "unknown"),
        }
        
        # Primären Zeitrahmen und seine Eigenschaften extrahieren
        primary_tf = market_analysis["details"].get("timeframe_used", "1h")
        
        # Wenn Support/Resistance-Levels vorhanden sind, aktuelle Preisnähe berechnen
        if primary_tf in data and not data[primary_tf].empty:
            current_price = data[primary_tf]['close'].iloc[-1] if not data[primary_tf].empty else None
            
            support_levels = market_analysis["details"].get("support_resistance", {}).get("support", [])
            resistance_levels = market_analysis["details"].get("support_resistance", {}).get("resistance", [])
            
            next_support = max([s for s in support_levels if s < current_price], default=None) if support_levels and current_price else None
            next_resistance = min([r for r in resistance_levels if r > current_price], default=None) if resistance_levels and current_price else None
            
            price_analysis = {
                "current_price": current_price,
                "next_support": next_support,
                "next_resistance": next_resistance,
                "support_distance_pct": ((current_price - next_support) / current_price * 100) if next_support and current_price else None,
                "resistance_distance_pct": ((next_resistance - current_price) / current_price * 100) if next_resistance and current_price else None
            }
        else:
            price_analysis = {"current_price": None}
        
        # Zusammenstellung des Handelsplans
        trading_plan = {
            "symbol": self.symbol,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_conditions": market_conditions,
            "price_analysis": price_analysis,
            "top_recommended_strategy": recommended_strategies[0] if recommended_strategies else None,
            "all_recommended_strategies": optimized_strategies,
            "risk_management": risk_management
        }
        
        return trading_plan
