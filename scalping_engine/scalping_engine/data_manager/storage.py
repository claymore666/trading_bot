import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from loguru import logger
from sqlalchemy import text, select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from scalping_engine.models.market_data import MarketData
from scalping_engine.utils.db import get_async_db


class DataStorage:
    """
    Verantwortlich für das Speichern, Abrufen und Verwalten von Marktdaten in der Datenbank.
    """
    
    @staticmethod
    async def store_dataframe(
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str = 'binance',
        replace_existing: bool = True,
        db: Optional[AsyncSession] = None
    ) -> bool:
        """
        Speichert einen Dataframe mit Marktdaten in der Datenbank.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            source: Datenquelle
            replace_existing: Ob bestehende Daten überschrieben werden sollen
            db: Datenbanksession (optional)
            
        Returns:
            bool: True, wenn erfolgreich gespeichert
        """
        if df.empty:
            logger.warning(f"Leerer DataFrame für {symbol} - nichts zu speichern")
            return False
        
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # DataFrame für das Speichern vorbereiten
            # Index zurücksetzen, um auf 'timestamp' Spalte zuzugreifen
            df_save = df.reset_index() if 'timestamp' in df.index.names else df.copy()
            
            # Wenn 'timestamp' nicht als Spalte vorhanden ist, aber das Index-Format bekannt ist
            if 'timestamp' not in df_save.columns and isinstance(df_save.index, pd.DatetimeIndex):
                df_save['timestamp'] = df_save.index
            
            # Prüfen, ob alle erforderlichen Spalten vorhanden sind
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df_save.columns]
            
            if missing_columns:
                logger.error(f"Fehlende Spalten im DataFrame: {missing_columns}")
                return False
            
            # Daten für den Bulk-Insert vorbereiten
            data = []
            for _, row in df_save.iterrows():
                data.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": row['timestamp'] if isinstance(row['timestamp'], datetime) 
                                 else pd.to_datetime(row['timestamp']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "source": source,
                    "additional_data": json.dumps({})  # Kann später erweitert werden
                })
            
            if replace_existing:
                # Bestehende Datenpunkte aktualisieren oder neue einfügen
                # Wir verwenden die ON CONFLICT Klausel für einen UPSERT-Vorgang
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
            else:
                # Nur neue Datenpunkte einfügen, bestehende ignorieren
                insert_query = text("""
                    INSERT INTO market_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume, source, additional_data)
                    VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :source, :additional_data)
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """)
            
            if data:
                # Chunks von 1000 Datenpunkten für bessere Performance
                chunk_size = 1000
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    await db.execute(insert_query, chunk)
                
                await db.commit()
                logger.info(f"{len(data)} Datenpunkte für {symbol}/{timeframe} gespeichert")
                return True
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Daten: {str(e)}")
            await db.rollback()
            return False
        
        finally:
            if close_db:
                await db.close()
        
        return False
    
    @staticmethod
    async def load_market_data(
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        db: Optional[AsyncSession] = None
    ) -> pd.DataFrame:
        """
        Lädt Marktdaten aus der Datenbank.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            start_date: Startdatum
            end_date: Enddatum
            limit: Maximale Anzahl von Datenpunkten
            db: Datenbanksession (optional)
            
        Returns:
            pd.DataFrame: DataFrame mit OHLCV-Daten
        """
        # Standardwerte für Start- und Enddatum
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Query aufbauen
            query = select(MarketData).where(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.timestamp >= start_date,
                    MarketData.timestamp <= end_date
                )
            ).order_by(MarketData.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            # Query ausführen
            result = await db.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                logger.warning(f"Keine Daten für {symbol}/{timeframe} im angegebenen Zeitraum gefunden")
                return pd.DataFrame()
            
            # Ergebnisse in DataFrame konvertieren
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
            # Timestamp als Index setzen
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"{len(df)} Datenpunkte für {symbol}/{timeframe} geladen")
            
            return df
        
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {str(e)}")
            return pd.DataFrame()
        
        finally:
            if close_db:
                await db.close()
    
    @staticmethod
    async def delete_market_data(
        symbol: str,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        db: Optional[AsyncSession] = None
    ) -> int:
        """
        Löscht Marktdaten aus der Datenbank.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen (wenn None, werden alle Zeitrahmen gelöscht)
            start_date: Startdatum (wenn None, ab Beginn der Daten)
            end_date: Enddatum (wenn None, bis Ende der Daten)
            db: Datenbanksession (optional)
            
        Returns:
            int: Anzahl der gelöschten Datenpunkte
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Bedingungen für das Löschen aufbauen
            conditions = [MarketData.symbol == symbol]
            
            if timeframe:
                conditions.append(MarketData.timeframe == timeframe)
            
            if start_date:
                conditions.append(MarketData.timestamp >= start_date)
            
            if end_date:
                conditions.append(MarketData.timestamp <= end_date)
            
            # Anzahl der zu löschenden Datenpunkte ermitteln
            count_query = select(func.count()).select_from(MarketData).where(and_(*conditions))
            count_result = await db.execute(count_query)
            count = count_result.scalar()
            
            if count == 0:
                logger.warning(f"Keine Daten zum Löschen gefunden")
                return 0
            
            # Löschquery aufbauen und ausführen
            delete_query = text(
                "DELETE FROM market_data WHERE " + 
                " AND ".join([f"{cond.left.key} = :val_{i}" for i, cond in enumerate(conditions)])
            )
            
            # Parameter für die Query erstellen
            params = {}
            for i, cond in enumerate(conditions):
                params[f"val_{i}"] = cond.right
            
            await db.execute(delete_query, params)
            await db.commit()
            
            logger.info(f"{count} Datenpunkte für {symbol}/{timeframe or 'alle Zeitrahmen'} gelöscht")
            
            return count
        
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Daten: {str(e)}")
            await db.rollback()
            return 0
        
        finally:
            if close_db:
                await db.close()
    
    @staticmethod
    async def get_available_symbols(db: Optional[AsyncSession] = None) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren Symbole in der Datenbank zurück.
        
        Args:
            db: Datenbanksession (optional)
            
        Returns:
            List[str]: Liste der Symbole
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            query = select(MarketData.symbol).distinct()
            result = await db.execute(query)
            symbols = [row[0] for row in result.all()]
            
            return symbols
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der verfügbaren Symbole: {str(e)}")
            return []
        
        finally:
            if close_db:
                await db.close()
    
    @staticmethod
    async def get_available_timeframes(symbol: str, db: Optional[AsyncSession] = None) -> List[str]:
        """
        Gibt eine Liste aller verfügbaren Zeitrahmen für ein Symbol zurück.
        
        Args:
            symbol: Handelssymbol
            db: Datenbanksession (optional)
            
        Returns:
            List[str]: Liste der Zeitrahmen
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            query = select(MarketData.timeframe).distinct().where(MarketData.symbol == symbol)
            result = await db.execute(query)
            timeframes = [row[0] for row in result.all()]
            
            return timeframes
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der verfügbaren Zeitrahmen für {symbol}: {str(e)}")
            return []
        
        finally:
            if close_db:
                await db.close()
    
    @staticmethod
    async def get_data_range(
        symbol: str,
        timeframe: str,
        db: Optional[AsyncSession] = None
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Gibt den verfügbaren Datenzeitraum für ein Symbol und einen Zeitrahmen zurück.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            db: Datenbanksession (optional)
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: (Start, Ende) des Datenzeitraums
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Frühesten Zeitstempel abrufen
            start_query = select(func.min(MarketData.timestamp)).where(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            )
            start_result = await db.execute(start_query)
            start_date = start_result.scalar()
            
            # Spätesten Zeitstempel abrufen
            end_query = select(func.max(MarketData.timestamp)).where(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            )
            end_result = await db.execute(end_query)
            end_date = end_result.scalar()
            
            return start_date, end_date
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Datenzeitraums für {symbol}/{timeframe}: {str(e)}")
            return None, None
        
        finally:
            if close_db:
                await db.close()
    
    @staticmethod
    async def get_data_statistics(
        symbol: str,
        timeframe: str,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """
        Gibt Statistiken über die verfügbaren Daten für ein Symbol und einen Zeitrahmen zurück.
        
        Args:
            symbol: Handelssymbol
            timeframe: Zeitrahmen
            db: Datenbanksession (optional)
            
        Returns:
            Dict[str, Any]: Statistiken
        """
        # Datenbank-Session abrufen, falls nicht übergeben
        close_db = False
        if db is None:
            db_gen = get_async_db()
            db = await anext(db_gen)
            close_db = True
        
        try:
            # Datenzeitraum abrufen
            start_date, end_date = await DataStorage.get_data_range(symbol, timeframe, db)
            
            if not start_date or not end_date:
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": 0,
                    "start_date": None,
                    "end_date": None,
                    "duration_days": 0,
                    "completeness": 0
                }
            
            # Anzahl der Datenpunkte abrufen
            count_query = select(func.count()).select_from(MarketData).where(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe
                )
            )
            count_result = await db.execute(count_query)
            count = count_result.scalar()
            
            # Dauer in Tagen berechnen
            duration = (end_date - start_date).total_seconds() / 86400  # Umrechnung in Tage
            
            # Erwartete Anzahl von Datenpunkten berechnen
            timeframe_seconds = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '2h': 7200,
                '4h': 14400,
                '6h': 21600,
                '12h': 43200,
                '1d': 86400,
                '3d': 259200,
                '1w': 604800
            }
            
            if timeframe in timeframe_seconds:
                expected_count = int(duration * 86400 / timeframe_seconds[timeframe])
                completeness = min(100, round(count / max(1, expected_count) * 100, 2))
            else:
                expected_count = 0
                completeness = 0
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "count": count,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": round(duration, 2),
                "expected_count": expected_count,
                "completeness": completeness
            }
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Datenstatistiken für {symbol}/{timeframe}: {str(e)}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": str(e)
            }
        
        finally:
            if close_db:
                await db.close()
