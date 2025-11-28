"""Market structure event model for tracking market structure changes."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .enums import StructureType


class MarketStructureEvent(BaseModel):
    """Market structure event representing breaks in market structure."""
    
    # Event data
    structure_type: StructureType = Field(..., description="Type of market structure")
    price: float = Field(..., gt=0, description="Break price level")
    timestamp: datetime = Field(..., description="Event timestamp")
    index: int = Field(..., ge=0, description="Index in the series")
    
    # Context information
    previous_pivot_price: float = Field(..., gt=0, description="Previous pivot price that was broken")
    previous_pivot_index: int = Field(..., ge=0, description="Previous pivot index")
    
    # Metadata
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    is_confirmed: bool = Field(True, description="Whether break is confirmed")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    
    # Additional context
    volume_at_break: Optional[float] = Field(None, ge=0, description="Volume at break point")
    break_strength: Optional[float] = Field(None, description="Strength of the break")
    
    @model_validator(mode="after")
    def validate_structure_consistency(self) -> "MarketStructureEvent":
        """Validate that structure type matches price relationship."""
        if self.structure_type in [StructureType.HIGHER_HIGH, StructureType.LOWER_HIGH]:
            # For high structures, break price should be above previous pivot
            if self.price <= self.previous_pivot_price:
                raise ValueError(
                    f"For {self.structure_type}, break price must be above previous pivot"
                )
        elif self.structure_type in [StructureType.HIGHER_LOW, StructureType.LOWER_LOW]:
            # For low structures, break price should be below previous pivot
            if self.price >= self.previous_pivot_price:
                raise ValueError(
                    f"For {self.structure_type}, break price must be below previous pivot"
                )
        
        return self
    
    @property
    def is_bullish_structure(self) -> bool:
        """Check if this is a bullish market structure event."""
        return self.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW]
    
    @property
    def is_bearish_structure(self) -> bool:
        """Check if this is a bearish market structure event."""
        return self.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW]
    
    @property
    def price_difference(self) -> float:
        """Calculate the difference between break price and previous pivot."""
        return abs(self.price - self.previous_pivot_price)
    
    @property
    def price_difference_percentage(self) -> float:
        """Calculate the percentage difference between break price and previous pivot."""
        return (self.price_difference / self.previous_pivot_price) * 100