from sqlalchemy import Column, String, Float, DateTime, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from scalping_engine.models.base_model import BaseModel


class MarketData(BaseModel):
    """
    Modell für Marktdaten mit OHLCV-Werten.
    """
    __tablename__ = "market_data"
    
    # Identifikationsfelder
    symbol = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    
    # Zeitstempel
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # OHLCV-Daten
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Zusätzliche Metadaten
    source = Column(String(50), nullable=True)
    additional_data = Column(JSONB, nullable=True)
    
    # Indizes und Constraints
    __table_args__ = (
        # Zusammengesetzter Index für häufige Abfragen
        Index('idx_market_data_symbol_timeframe_timestamp', 
              'symbol', 'timeframe', 'timestamp'),
        
        # Keine Duplikate erlauben
        UniqueConstraint('symbol', 'timeframe', 'timestamp', 
                         name='uq_market_data_symbol_timeframe_timestamp'),
    )
    
    def __repr__(self):
        return (
            f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', "
            f"timestamp='{self.timestamp}', close={self.close})>"
        )
