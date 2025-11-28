"""Candle model for OHLCV data."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import Direction


class Candle(BaseModel):
    """Candlestick data model with OHLCV information and computed attributes."""
    
    # Basic OHLCV data
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    
    # Metadata
    timestamp: datetime = Field(..., description="Candle timestamp")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    index: Optional[int] = Field(None, description="Index in series")
    
    # Validation flags
    is_valid: bool = Field(True, description="Whether candle data is valid")
    
    @model_validator(mode="after")
    def validate_price_relationships(self) -> "Candle":
        """Validate that high/low relationships are correct."""
        if self.high < self.open:
            raise ValueError("High price must be >= open price")
        if self.high < self.close:
            raise ValueError("High price must be >= close price")
        if self.low > self.open:
            raise ValueError("Low price must be <= open price")
        if self.low > self.close:
            raise ValueError("Low price must be <= close price")
        return self
    
    @computed_field
    @property
    def range(self) -> float:
        """Calculate price range (high - low)."""
        return self.high - self.low
    
    @computed_field
    @property
    def body(self) -> float:
        """Calculate candle body (abs(close - open))."""
        return abs(self.close - self.open)
    
    @computed_field
    @property
    def body_percentage(self) -> float:
        """Calculate body as percentage of range."""
        if self.range == 0:
            return 0.0
        return (self.body / self.range) * 100
    
    @computed_field
    @property
    def upper_wick(self) -> float:
        """Calculate upper wick length."""
        return self.high - max(self.open, self.close)
    
    @computed_field
    @property
    def lower_wick(self) -> float:
        """Calculate lower wick length."""
        return min(self.open, self.close) - self.low
    
    @computed_field
    @property
    def direction(self) -> Direction:
        """Determine candle direction."""
        return Direction.BULLISH if self.close > self.open else Direction.BEARISH
    
    @computed_field
    @property
    def is_doji(self) -> bool:
        """Check if candle is a doji (open ≈ close)."""
        if self.range == 0:
            return False
        tolerance = self.range * 0.01  # 1% of range
        return abs(self.close - self.open) <= tolerance
    
    @classmethod
    def from_ohlcv_array(
        cls,
        ohlcv: list,
        symbol: str,
        timeframe: str,
        index: Optional[int] = None,
    ) -> "Candle":
        """Create candle from OHLCV array format [timestamp, open, high, low, close, volume]."""
        if len(ohlcv) != 6:
            raise ValueError("OHLCV array must have exactly 6 elements")
        
        timestamp = datetime.fromtimestamp(ohlcv[0] / 1000)  # Convert from ms
        open_price, high_price, low_price, close_price, volume = ohlcv[1:6]
        
        return cls(
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            index=index,
        )