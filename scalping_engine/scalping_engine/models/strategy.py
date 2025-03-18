from sqlalchemy import Column, String, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB

from scalping_engine.models.base_model import BaseTimeStampedModel


class Strategy(BaseTimeStampedModel):
    """
    Modell für Handelsstrategien.
    """
    __tablename__ = "strategies"
    
    # Basisinformationen
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    
    # Strategie-Konfiguration
    parameters = Column(JSONB, nullable=False, default={})
    indicators = Column(JSONB, nullable=False, default=[])
    entry_conditions = Column(JSONB, nullable=False, default=[])
    exit_conditions = Column(JSONB, nullable=False, default=[])
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_optimized = Column(Boolean, default=False, nullable=False)
    
    def __repr__(self):
        return f"<Strategy(name='{self.name}', active={self.is_active})>"
