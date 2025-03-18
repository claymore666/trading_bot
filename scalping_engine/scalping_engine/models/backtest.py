from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime

from scalping_engine.models.base_model import BaseTimeStampedModel


class Backtest(BaseTimeStampedModel):
    """
    Modell für Backtesting-Ergebnisse.
    """
    __tablename__ = "backtests"
    
    # Beziehung zur Strategie
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    strategy = relationship("Strategy", backref="backtests")
    
    # Backtesting-Parameter
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False, default=1000.0)
    
    # Status
    status = Column(String(20), nullable=False, default="running")
    completed_at = Column(DateTime, nullable=True)
    
    # Ergebnisse
    results = Column(JSONB, nullable=True)
    metrics = Column(JSONB, nullable=True)
    trades = Column(JSONB, nullable=True)
    
    # Fehler
    error = Column(Text, nullable=True)
    
    def __repr__(self):
        return (
            f"<Backtest(strategy='{self.strategy.name if self.strategy else None}', "
            f"symbol='{self.symbol}', status='{self.status}')>"
        )
