from sqlalchemy import Column, Integer, DateTime, func
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime

from scalping_engine.utils.db import Base


class BaseModel(Base):
    """
    Basis-Modell mit ID für alle Tabellen.
    """
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    @declared_attr
    def __tablename__(cls):
        """
        Tabellennamen automatisch vom Klassennamen ableiten.
        """
        return cls.__name__.lower()


class BaseTimeStampedModel(BaseModel):
    """
    Basis-Modell mit Zeitstempeln für Erstellung und Aktualisierung.
    """
    __abstract__ = True
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, 
        default=datetime.utcnow, 
        onupdate=datetime.utcnow, 
        nullable=False
    )
