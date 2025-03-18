import os
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Konfigurationseinstellungen für die Anwendung, basierend auf Umgebungsvariablen.
    """
    # Datenbank-Einstellungen
    DB_HOST: str = os.environ.get("DB_HOST", "timescaledb")
    DB_PORT: str = os.environ.get("DB_PORT", "5432")
    DB_USER: str = os.environ.get("DB_USER", "scalpuser")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "scalppass")
    DB_NAME: str = os.environ.get("DB_NAME", "scalpingdb")
    
    # Logging-Einstellungen
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    
    # API-Einstellungen
    API_PORT: int = int(os.environ.get("API_PORT", "8001"))
    
    # Backtesting-Einstellungen
    BACKTEST_DAYS: int = int(os.environ.get("BACKTEST_DAYS", "365"))
    INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "1000"))
    MAX_RISK_PER_TRADE: float = float(os.environ.get("MAX_RISK_PER_TRADE", "1.5"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Gibt die Anwendungseinstellungen zurück, gecached für bessere Performance.
    """
    return Settings()
