import os
import sys
from loguru import logger


def setup_logging(log_level="INFO"):
    """
    Konfiguriert das Logging für die Anwendung.
    
    Args:
        log_level (str): Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Stelle sicher, dass das Logs-Verzeichnis existiert
    os.makedirs("logs", exist_ok=True)
    
    # Entferne alle Standard-Handler
    logger.remove()
    
    # Konsolenausgabe hinzufügen
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # Dateiausgabe hinzufügen
    logger.add(
        "logs/scalping_engine.log",
        rotation="10 MB",
        retention="1 week",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    # In Produktionsumgebungen kann ein zentralisiertes Logging-System hinzugefügt werden
    # z.B. ELK Stack, Graylog, etc.
    
    logger.info(f"Logging eingerichtet mit Level: {log_level}")
