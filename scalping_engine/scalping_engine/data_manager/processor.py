import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from loguru import logger

class DataProcessor:
    """
    Verarbeitet rohe Marktdaten für die weitere Analyse und Strategie-Ausführung.
    """
    
    @staticmethod
    def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Bereinigt Marktdaten:
        - Entfernt Duplikate
        - Behandelt fehlende Werte
        - Entfernt Ausreißer
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            pd.DataFrame: Bereinigter DataFrame
        """
        if df.empty:
            return df
            
        # Kopie erstellen, um das Original nicht zu verändern
        df_clean = df.copy()
        
        # Duplikate entfernen
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        # Fehlende Werte behandeln
        # Bei Preisdaten können wir fehlende Werte mit dem letzten bekannten Wert füllen
        # für eine fortgeschrittenere Lösung könnte man Interpolation verwenden
        df_clean = df_clean.fillna(method='ffill')
        
        # Ausreißer bei Volumen behandeln (z.B. abnormal hohe Volumina)
        # Wir verwenden hier einen einfachen Z-Score-Filter
        if 'volume' in df_clean.columns and len(df_clean) > 10:
            vol_mean = df_clean['volume'].mean()
            vol_std = df_clean['volume'].std()
            
            if vol_std > 0:  # Vermeidet Division durch Null
                z_scores = (df_clean['volume'] - vol_mean) / vol_std
                df_clean['volume_outlier'] = abs(z_scores) > 3
                
                logger.debug(f"Gefunden: {df_clean['volume_outlier'].sum()} Volumen-Ausreißer")
                
                # Ausreißer durch Mittelwert ersetzen
                # (alternativ könnte man sie auch behalten, aber markieren)
                df_clean.loc[df_clean['volume_outlier'], 'volume'] = vol_mean
                
                # Hilfsspalte entfernen
                df_clean = df_clean.drop('volume_outlier', axis=1)
        
        return df_clean
    
    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Resampling der Daten auf einen anderen Zeitrahmen.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            timeframe: Ziel-Zeitrahmen (z.B. '1h', '4h', '1d')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        if df.empty:
            return df
            
        # Pandas-Resampling-Regeln basierend auf Zeitrahmen
        timeframe_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1w': '1W'
        }
        
        if timeframe not in timeframe_map:
            logger.error(f"Ungültiger Zeitrahmen für Resampling: {timeframe}")
            return df
            
        rule = timeframe_map[timeframe]
        
        # OHLCV-Resampling
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled
    
    @staticmethod
    def add_indicators(df: pd.DataFrame, indicators: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Fügt technische Indikatoren zu den Marktdaten hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            indicators: Liste der zu berechnenden Indikatoren mit Parametern
            
        Returns:
            pd.DataFrame: DataFrame mit hinzugefügten Indikatoren
        """
        if df.empty:
            return df
            
        # Kopie erstellen, um das Original nicht zu verändern
        df_with_indicators = df.copy()
        
        # Standardindikatoren, falls keine angegeben wurden
        if indicators is None:
            indicators = [
                {'type': 'sma', 'params': {'window': 20}},
                {'type': 'ema', 'params': {'window': 20}},
                {'type': 'bollinger_bands', 'params': {'window': 20, 'std_dev': 2.0}},
                {'type': 'rsi', 'params': {'window': 14}},
                {'type': 'macd', 'params': {'fast_window': 12, 'slow_window': 26, 'signal_window': 9}},
                {'type': 'atr', 'params': {'window': 14}}
            ]
        
        # Indikatoren berechnen
        for indicator in indicators:
            indicator_type = indicator.get('type', '')
            params = indicator.get('params', {})
            
            # Simple Moving Average (SMA)
            if indicator_type == 'sma':
                window = params.get('window', 20)
                df_with_indicators[f'sma_{window}'] = df_with_indicators['close'].rolling(window=window).mean()
            
            # Exponential Moving Average (EMA)
            elif indicator_type == 'ema':
                window = params.get('window', 20)
                df_with_indicators[f'ema_{window}'] = df_with_indicators['close'].ewm(span=window, adjust=False).mean()
            
            # Bollinger Bands
            elif indicator_type == 'bollinger_bands':
                window = params.get('window', 20)
                std_dev = params.get('std_dev', 2.0)
                sma = df_with_indicators['close'].rolling(window=window).mean()
                std = df_with_indicators['close'].rolling(window=window).std()
                df_with_indicators[f'bollinger_mid'] = sma
                df_with_indicators[f'bollinger_upper'] = sma + std_dev * std
                df_with_indicators[f'bollinger_lower'] = sma - std_dev * std
            
            # Relative Strength Index (RSI)
            elif indicator_type == 'rsi':
                window = params.get('window', 14)
                delta = df_with_indicators['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                df_with_indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Average Convergence Divergence (MACD)
            elif indicator_type == 'macd':
                fast_window = params.get('fast_window', 12)
                slow_window = params.get('slow_window', 26)
                signal_window = params.get('signal_window', 9)
                
                fast_ema = df_with_indicators['close'].ewm(span=fast_window, adjust=False).mean()
                slow_ema = df_with_indicators['close'].ewm(span=slow_window, adjust=False).mean()
                df_with_indicators['macd'] = fast_ema - slow_ema
                df_with_indicators['macd_signal'] = df_with_indicators['macd'].ewm(span=signal_window, adjust=False).mean()
                df_with_indicators['macd_histogram'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
            
            # Average True Range (ATR)
            elif indicator_type == 'atr':
                window = params.get('window', 14)
                high_low = df_with_indicators['high'] - df_with_indicators['low']
                high_close = (df_with_indicators['high'] - df_with_indicators['close'].shift()).abs()
                low_close = (df_with_indicators['low'] - df_with_indicators['close'].shift()).abs()
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df_with_indicators['true_range'] = true_range
                df_with_indicators['atr'] = true_range.rolling(window=window).mean()
            
            # Stochastischer Oszillator
            elif indicator_type == 'stochastic':
                k_window = params.get('k_window', 14)
                d_window = params.get('d_window', 3)
                
                low_min = df_with_indicators['low'].rolling(window=k_window).min()
                high_max = df_with_indicators['high'].rolling(window=k_window).max()
                
                df_with_indicators['stoch_k'] = 100 * ((df_with_indicators['close'] - low_min) / (high_max - low_min))
                df_with_indicators['stoch_d'] = df_with_indicators['stoch_k'].rolling(window=d_window).mean()
            
            # Volumen-SMA
            elif indicator_type == 'volume_sma':
                window = params.get('window', 20)
                df_with_indicators[f'volume_sma_{window}'] = df_with_indicators['volume'].rolling(window=window).mean()
            
            # Weitere Indikatoren können hier hinzugefügt werden
            else:
                logger.warning(f"Unbekannter Indikator-Typ: {indicator_type}")
        
        return df_with_indicators
    
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Feature-Engineering-Merkmale zu den Marktdaten hinzu.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            pd.DataFrame: DataFrame mit hinzugefügten Features
        """
        if df.empty:
            return df
            
        # Kopie erstellen, um das Original nicht zu verändern
        df_with_features = df.copy()
        
        # Tageszeit-Feature (kann für Marktphasen relevant sein)
        if isinstance(df_with_features.index, pd.DatetimeIndex):
            df_with_features['hour'] = df_with_features.index.hour
            df_with_features['day_of_week'] = df_with_features.index.dayofweek
        
        # Price Action Features
        df_with_features['body_size'] = abs(df_with_features['close'] - df_with_features['open'])
        df_with_features['upper_wick'] = df_with_features['high'] - df_with_features[['open', 'close']].max(axis=1)
        df_with_features['lower_wick'] = df_with_features[['open', 'close']].min(axis=1) - df_with_features['low']
        
        # Kerzenformen
        df_with_features['is_bullish'] = df_with_features['close'] > df_with_features['open']
        df_with_features['is_doji'] = df_with_features['body_size'] < (df_with_features['high'] - df_with_features['low']) * 0.1
        
        # Volatilität
        df_with_features['volatility'] = (df_with_features['high'] - df_with_features['low']) / df_with_features['open']
        
        # Volumen-Features
        if 'volume' in df_with_features.columns:
            df_with_features['volume_change'] = df_with_features['volume'].pct_change()
            df_with_features['volume_sma_20'] = df_with_features['volume'].rolling(window=20).mean()
            df_with_features['is_volume_spike'] = df_with_features['volume'] > df_with_features['volume_sma_20'] * 2
        
        # Verhältnisse und Berechnungen für Marktdynamik
        df_with_features['price_change'] = df_with_features['close'].pct_change()
        df_with_features['price_sma_20'] = df_with_features['close'].rolling(window=20).mean()
        df_with_features['price_above_sma'] = df_with_features['close'] > df_with_features['price_sma_20']
        
        return df_with_features
    
    @staticmethod
    def detect_market_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Erkennt Marktlücken (Gaps) in den Daten.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            pd.DataFrame: DataFrame mit Informationen über Gaps
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Kopie erstellen, um das Original nicht zu verändern
        df_with_gaps = df.copy()
        
        # Gap-up und Gap-down Erkennung
        df_with_gaps['gap_up'] = df_with_gaps['low'] > df_with_gaps['high'].shift(1)
        df_with_gaps['gap_down'] = df_with_gaps['high'] < df_with_gaps['low'].shift(1)
        
        # Gap-Größe in Prozent
        df_with_gaps['gap_size'] = np.where(
            df_with_gaps['gap_up'],
            (df_with_gaps['low'] / df_with_gaps['high'].shift(1) - 1) * 100,
            np.where(
                df_with_gaps['gap_down'],
                (df_with_gaps['high'] / df_with_gaps['low'].shift(1) - 1) * 100,
                0
            )
        )
        
        # Zeitlücken in den Daten erkennen (fehlende Zeitpunkte)
        if len(df_with_gaps) > 1:
            # Erwartete Zeitdifferenz basierend auf den ersten beiden Einträgen
            expected_diff = df_with_gaps.index[1] - df_with_gaps.index[0]
            
            # Tatsächliche Zeitdifferenzen
            actual_diffs = df_with_gaps.index.to_series().diff()
            
            # Zeitlücken markieren
            df_with_gaps['time_gap'] = actual_diffs > expected_diff * 1.5
            
            # Größe der Zeitlücke in regulären Perioden
            df_with_gaps['time_gap_size'] = (actual_diffs / expected_diff).round(1)
        
        return df_with_gaps

# Hilfsfunktion für externe Aufrufe
def process_market_data(
    df: pd.DataFrame,
    clean: bool = True,
    add_features: bool = True,
    add_volume: bool = True,
    detect_market_gaps: bool = True,
    indicators: Optional[List[Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Verarbeitet Marktdaten für die Analyse und Strategie-Ausführung.
    
    Args:
        df: DataFrame mit OHLCV-Daten
        clean: Ob die Daten bereinigt werden sollen
        add_features: Ob Features hinzugefügt werden sollen
        add_volume: Ob Volumen-Features hinzugefügt werden sollen
        detect_market_gaps: Ob Marktlücken erkannt werden sollen
        indicators: Liste der zu berechnenden Indikatoren mit Parametern
        
    Returns:
        pd.DataFrame: Verarbeiteter DataFrame
    """
    processor = DataProcessor()
    
    # Daten kopieren, um das Original nicht zu verändern
    data = df.copy()
    
    # Sicherstellen, dass wir einen DatetimeIndex haben
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    else:
        logger.warning("DataFrame hat keinen DatetimeIndex, Sortierung könnte ungenau sein")
        data = data.sort_index()
    
    # Daten bereinigen
    if clean:
        data = processor.clean_market_data(data)
    
    # Indikatoren hinzufügen
    data = processor.add_indicators(data, indicators)
    
    # Features hinzufügen
    if add_features:
        data = processor.add_features(data)
    
    # Marktlücken erkennen
    if detect_market_gaps:
        data = processor.detect_market_gaps(data)
    
    return data
