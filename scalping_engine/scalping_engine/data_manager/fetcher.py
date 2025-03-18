import asyncio
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import ccxt.async_support as ccxtasync
import ccxt
from typing import Dict, List, Optional, Union, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select

from scalping_engine.models.market_data import MarketData
from scalping_engine.utils.db import get_async_db


class RateLimiter:
    """
    Implementiert einen Token-Bucket-Algorithmus, um API-Ratengrenzwerte einzuhalten.
    Speziell für Binance-Limits konzipiert.
    """
    def __init__(self, max_tokens: int = 6000, refill_rate: float = 100, time_window: int = 60):
        """
        Initialisiert den Rate Limiter.
        
        Args:
            max_tokens: Maximale Anzahl von Tokens im Bucket (entspricht dem Weight-Limit von Binance)
            refill_rate: Anzahl der Tokens, die pro Sekunde nachgefüllt werden
            time_window: Zeitfenster in Sekunden, in dem das Limit gilt
        """
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.time_window = time_window  # in Sekunden
        self.last_refill_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, weight: int = 1) -> bool:
        """
        Versucht, Tokens zu erwerben. Gibt True zurück, wenn erfolgreich, sonst False.
        
        Args:
            weight: Gewicht der Anfrage (unterschiedliche Binance-Endpunkte haben unterschiedliche Gewichte)
        
        Returns:
            bool: True, wenn die Tokens erworben wurden, sonst False
        """
        async with self.lock:
            self._refill_tokens()
            
            if self.tokens >= weight:
                self.tokens -= weight
                return True
            else:
                wait_time = (weight - self.tokens) / self.refill_rate
                logger.debug(f"Rate Limit erreicht. Warte {wait_time:.2f}s auf Token-Nachfüllung")
                await asyncio.sleep(wait_time)
                self._refill_tokens()
                self.tokens -= weight
                return True
    
    def _refill_tokens(self):
        """
        Füllt Tokens basierend auf der vergangenen Zeit seit der letzten Nachfüllung nach.
        """
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        
        if new_tokens > 0:
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.last_refill_time = now


class DataFetcher:
    """
    Verantwortlich für den Abruf und die Speicherung von Marktdaten aus verschiedenen Quellen.
    """
    def __init__(self):
        # Binance-Ratengrenze: 6000 Gewicht pro Minute
        self.binance_rate_limiter = RateLimiter(max_tokens=6000, refill_rate=100, time_window=60)
        self.exchange_clients = {}
        
    async def get_exchange_client(self, exchange_name: str = 'binance'):
        """
        Gibt einen CCXT-Client für die angegebene Börse zurück.
        Erstellt einen neuen Client, wenn noch keiner existiert.
        
        Args:
            exchange_name: Name der Börse (z.B. 'binance', 'ftx', 'coinbase')
            
        Returns:
            ccxt.Exchange: CCXT-Exchange-Client
        """
        if exchange_name.lower() not in self.exchange_clients:
            # Binance-Client erstellen
            if exchange_name.lower() == 'binance':
                self.exchange_clients[exchange_name.lower()] = ccxtasync.binance({
                    'enableRateLimit': True,  # CCXT-eigener Rate-Limiter
                    'options': {
                        'defaultType': 'spot',  # Standardmäßig Spot-Markt verwenden
                    }
                })
            else:
                # Andere Börsen hier hinzufügen
                exchange_class = getattr(ccxtasync, exchange_name.lower())
                self.exchange_clients[exchange_name.lower()] = exchange_class({
                    'enableRateLimit': True,
                })
        
        return self.exchange_clients[exchange_name.lower()]

    async def close_clients(self):
        """
        Schließt alle offenen Exchange-Clients ordnungsgemäß.
        """
        for exchange_name, client in self.exchange_clients.items():
            await client.close()
        self.exchange_clients = {}
        
    async def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: str = 'binance'
    ) -> pd.DataFrame:
        """
        Ruft Marktdaten für ein Symbol ab.
        
        Args:
            symbol: Symbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen (z.B. '1m', '5m', '1h', '1d')
            start_date: Startdatum
            end_date: Enddatum
            exchange: Name der Börse
            
        Returns:
            pd.DataFrame: DataFrame mit OHLCV-Daten
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        logger.info(f"Rufe Daten für {symbol} von {start_date} bis {end_date} ab")
        
        client = await self.get_exchange_client(exchange)
        
        # Binance erlaubt maximal 1000 Kerzen pro Anfrage
        # Wir müssen also mehrere Anfragen machen, wenn der Zeitraum zu groß ist
        timeframe_seconds = self._get_timeframe_seconds(timeframe)
        if timeframe_seconds == 0:
            raise ValueError(f"Ungültiger Zeitrahmen: {timeframe}")
            
        # Anfangs- und Endzeitstempel in Millisekunden
        from_ts = int(start_date.timestamp() * 1000)
        to_ts = int(end_date.timestamp() * 1000)
        
        # Maximale Anzahl von Kerzen pro Anfrage
        max_candles_per_request = 1000
        
        # Berechne, wie viele Millisekunden der gewählte Zeitrahmen entspricht
        timeframe_ms = timeframe_seconds * 1000
        
        # Berechne die Anzahl der benötigten Anfragen
        time_range_ms = to_ts - from_ts
        num_candles = time_range_ms // timeframe_ms
        num_requests = (num_candles + max_candles_per_request - 1) // max_candles_per_request
        
        logger.debug(f"Für {symbol}: {num_candles} Kerzen, {num_requests} Anfragen notwendig")
        
        all_candles = []
        current_ts = from_ts
        
        # Weight für /klines Endpunkt bei Binance ist 1
        request_weight = 1
        
        for i in range(num_requests):
            # Warten, bis wir die Rate-Grenze einhalten
            await self.binance_rate_limiter.acquire(request_weight)
            
            try:
                # Kerzen abrufen
                candles = await client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=max_candles_per_request
                )
                
                if not candles or len(candles) == 0:
                    logger.warning(f"Keine Daten für {symbol} im Zeitraum gefunden")
                    break
                    
                all_candles.extend(candles)
                
                # Timestamp für die nächste Anfrage aktualisieren
                # Wir verwenden den letzten Timestamp der aktuellen Antwort plus eine Kerze
                current_ts = candles[-1][0] + timeframe_ms
                
                # Prüfen, ob wir bereits alle benötigten Daten haben
                if current_ts >= to_ts:
                    break
                    
                logger.debug(f"Fortschritt für {symbol}: {i+1}/{num_requests} Anfragen, {len(all_candles)}/{num_candles} Kerzen")
                
            except Exception as e:
                logger.error(f"Fehler beim Abrufen von Daten für {symbol}: {str(e)}")
                break
                
            # Kurze Pause, um die API nicht zu überlasten
            await asyncio.sleep(0.1)
        
        # CCXT-Format zu Pandas-DataFrame konvertieren
        if not all_candles:
            logger.warning(f"Keine Daten für {symbol} abgerufen")
            return pd.DataFrame()
            
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Timestamp von Millisekunden in Datetime umwandeln
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Duplikate entfernen und nach Timestamp sortieren
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        # Timestamp als Index setzen
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Daten für {symbol} abgerufen: {len(df)} Einträge")
        
        return df
        
    async def save_market_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str = 'binance',
        db: Optional[AsyncSession] = None
    ) -> bool:
        """
        Speichert abgerufene Marktdaten in der Datenbank.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Symbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen (z.B. '1m', '5m', '1h', '1d')
            source: Datenquelle
            db: Datenbanksession (optional)
            
        Returns:
            bool: True, wenn erfolgreich gespeichert
        """
        if df.empty:
            logger.warning(f"Keine Daten zum Speichern für {symbol}")
            return False
            
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Daten in die Datenbank schreiben
            # Wir verwenden Bulk-Insert für bessere Performance
            data = []
            
            for idx, row in df.reset_index().iterrows():
                data.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": row['timestamp'],
                    "open": row['open'],
                    "high": row['high'],
                    "low": row['low'],
                    "close": row['close'],
                    "volume": row['volume'],
                    "source": source,
                    "additional_data": json.dumps({})
                })
            
            # Optimierter Bulk-Insert mit ON CONFLICT DO NOTHING
            if data:
                insert_query = text("""
                    INSERT INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume, source, additional_data)
                    VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :source, :additional_data)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source,
                        additional_data = EXCLUDED.additional_data
                """)
                
                await db.execute(insert_query, data)
                await db.commit()
                
                logger.info(f"{len(data)} Datenpunkte für {symbol} gespeichert")
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Daten für {symbol}: {str(e)}")
            await db.rollback()
            return False
            
        finally:
            if close_db:
                await db.close()
                
        return False
        
    async def fetch_and_save_market_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: str = 'binance',
        db: Optional[AsyncSession] = None
    ) -> pd.DataFrame:
        """
        Kombiniert Abruf und Speicherung von Marktdaten in einem Schritt.
        
        Args:
            symbol: Symbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen (z.B. '1m', '5m', '1h', '1d')
            start_date: Startdatum
            end_date: Enddatum
            exchange: Name der Börse
            db: Datenbanksession (optional)
            
        Returns:
            pd.DataFrame: DataFrame mit OHLCV-Daten
        """
        df = await self.fetch_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange
        )
        
        if not df.empty:
            await self.save_market_data(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                source=exchange,
                db=db
            )
            
        return df
        
    async def load_market_data_from_db(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        db: Optional[AsyncSession] = None
    ) -> pd.DataFrame:
        """
        Lädt Marktdaten aus der Datenbank.
        
        Args:
            symbol: Symbol (z.B. 'BTC/USDT')
            timeframe: Zeitrahmen (z.B. '1m', '5m', '1h', '1d')
            start_date: Startdatum
            end_date: Enddatum
            db: Datenbanksession (optional)
            
        Returns:
            pd.DataFrame: DataFrame mit OHLCV-Daten
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
            
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
            
        try:
            # Daten aus der Datenbank abrufen
            query = select(MarketData).where(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp)
            
            result = await db.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                logger.warning(f"Keine Daten in der Datenbank für {symbol} im angegebenen Zeitraum gefunden")
                return pd.DataFrame()
                
            # Daten in DataFrame konvertieren
            data = []
            for row in rows:
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                })
                
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Daten für {symbol} aus Datenbank geladen: {len(df)} Einträge")
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten für {symbol} aus der Datenbank: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if close_db:
                await db.close()
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """
        Konvertiert einen Zeitrahmen-String in Sekunden.
        
        Args:
            timeframe: Zeitrahmen im Format 'Xm', 'Xh', 'Xd', 'Xw'
            
        Returns:
            int: Zeitrahmen in Sekunden
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 60 * 60 * 24
        elif unit == 'w':
            return value * 60 * 60 * 24 * 7
        else:
            return 0


# Singleton-Instanz
data_fetcher = DataFetcher()


# Hilfsfunktion für externe Aufrufe
async def fetch_market_data(
    symbol: str,
    timeframe: str = '1h',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    exchange: str = 'binance'
) -> pd.DataFrame:
    """
    Hilfsfunktion zum Abrufen von Marktdaten.
    
    Args:
        symbol: Symbol (z.B. 'BTC/USDT')
        timeframe: Zeitrahmen (z.B. '1m', '5m', '1h', '1d')
        start_date: Startdatum
        end_date: Enddatum
        exchange: Name der Börse
        
    Returns:
        pd.DataFrame: DataFrame mit OHLCV-Daten
    """
    global data_fetcher
    
    try:
        # Prüfen, ob die Daten bereits in der Datenbank vorhanden sind
        db_gen = get_async_db()
        db = await anext(db_gen)
        
        db_data = await data_fetcher.load_market_data_from_db(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            db=db
        )
        
        # Wenn es Daten in der Datenbank gibt, verwenden wir diese
        if not db_data.empty:
            return db_data
            
        # Sonst holen wir die Daten von der API und speichern sie
        return await data_fetcher.fetch_and_save_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            exchange=exchange,
            db=db
        )
        
    except Exception as e:
        logger.error(f"Fehler beim Abrufen von Marktdaten: {str(e)}")
        return pd.DataFrame()
