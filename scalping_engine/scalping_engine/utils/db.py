import os
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from loguru import logger

from scalping_engine.utils.config import get_settings

settings = get_settings()

# SQLAlchemy Datenbank-URL
SQLALCHEMY_DATABASE_URL = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
ASYNC_SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

# SQLAlchemy Engine und Session erstellen
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Asynchrone Datenbank-Engine
async_engine = create_async_engine(ASYNC_SQLALCHEMY_DATABASE_URL)
AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine, 
    class_=AsyncSession
)

# Base-Klasse für Modelle
Base = declarative_base()
metadata = MetaData()

# Dependency für FastAPI-Routen
def get_db():
    """
    Erstellt eine neue Datenbankverbindung für jede Anfrage.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Asynchrone Dependency für FastAPI-Routen
async def get_async_db():
    """
    Erstellt eine neue asynchrone Datenbankverbindung für jede Anfrage.
    """
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """
    Initialisiert die Datenbank und erstellt alle Tabellen.
    """
    try:
        # Verbindung zur Datenbank prüfen
        async with async_engine.begin() as conn:
            # TimescaleDB Erweiterung aktivieren (mit richtigem SQLAlchemy Text-Objekt)
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                logger.info("TimescaleDB-Erweiterung aktiviert oder bereits vorhanden")
            except Exception as e:
                logger.warning(f"Konnte TimescaleDB-Erweiterung nicht aktivieren: {e}. Fortfahren ohne TimescaleDB.")
            
            # Alle Tabellen erstellen
            # Importiere hier erst Base, um zirkuläre Importe zu vermeiden
            from scalping_engine.models.base_model import Base
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Datenbank erfolgreich initialisiert")
        return True
    except Exception as e:
        logger.error(f"Fehler bei der Datenbankinitialisierung: {e}")
        raise
