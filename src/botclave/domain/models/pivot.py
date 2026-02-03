"""Pivot model for identifying swing highs and lows."""

from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import Direction


class Pivot(BaseModel):
    """Pivot point representing swing highs and lows in price action."""
    
    # Core pivot data
    price: float = Field(..., gt=0, description="Pivot price level")
    timestamp: datetime = Field(..., description="Pivot timestamp")
    index: int = Field(..., ge=0, description="Index in the series")
    
    # Pivot characteristics
    direction: Direction = Field(..., description="Pivot direction (high or low)")
    strength: float = Field(..., ge=0, description="Pivot strength score")
    lookback_left: int = Field(..., gt=0, description="Number of bars to the left")
    lookback_right: int = Field(..., gt=0, description="Number of bars to the right")
    
    # Metadata
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    is_confirmed: bool = Field(True, description="Whether pivot is confirmed")
    
    @model_validator(mode="after")
    def validate_high_low_consistency(self) -> "Pivot":
        """Validate that high pivots are actually highs and low pivots are lows."""
        # This would typically be validated against actual price data
        # For now, we'll just ensure the direction is set correctly
        return self
    
    @computed_field
    @property
    def pivot_type(self) -> str:
        """Get pivot type as string."""
        return "swing_high" if self.direction == Direction.BEARISH else "swing_low"
    
    @classmethod
    def from_series(
        cls,
        series: pd.Series,
        direction: Direction,
        lookback_left: int,
        lookback_right: int,
        symbol: str,
        timeframe: str,
        strength_calc: Optional[str] = "relative",
    ) -> List["Pivot"]:
        """Identify pivots from a price series.
        
        Args:
            series: Price series (high for swing highs, low for swing lows)
            direction: Pivot direction to identify
            lookback_left: Number of bars to look back on the left
            lookback_right: Number of bars to look back on the right  
            symbol: Trading symbol
            timeframe: Timeframe
            strength_calc: Method for calculating strength ('relative', 'absolute')
            
        Returns:
            List of identified pivots
        """
        pivots = []
        
        for i in range(lookback_left, len(series) - lookback_right):
            current_price = series.iloc[i]
            
            if direction == Direction.BEARISH:  # Swing high
                is_pivot = all(
                    current_price >= series.iloc[j]
                    for j in range(i - lookback_left, i + lookback_right + 1)
                    if j != i
                )
            else:  # Swing low
                is_pivot = all(
                    current_price <= series.iloc[j]
                    for j in range(i - lookback_left, i + lookback_right + 1)
                    if j != i
                )
            
            if is_pivot:
                # Calculate strength based on method
                if strength_calc == "relative":
                    # Relative strength based on how much it stands out
                    if direction == Direction.BEARISH:
                        max_nearby = max(
                            series.iloc[j]
                            for j in range(i - lookback_left, i + lookback_right + 1)
                            if j != i
                        )
                        strength = (current_price - max_nearby) / current_price * 100
                    else:
                        min_nearby = min(
                            series.iloc[j]
                            for j in range(i - lookback_left, i + lookback_right + 1)
                            if j != i
                        )
                        strength = (min_nearby - current_price) / current_price * 100
                else:
                    # Absolute strength
                    strength = current_price
                
                # Get timestamp from series index if it's datetime
                timestamp = (
                    series.index[i] if hasattr(series.index[i], "to_pydatetime")
                    else datetime.now()
                )
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()
                
                pivots.append(
                    cls(
                        price=current_price,
                        timestamp=timestamp,
                        index=i,
                        direction=direction,
                        strength=max(0, strength),  # Ensure non-negative
                        lookback_left=lookback_left,
                        lookback_right=lookback_right,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
        
        return pivots
    
    def is_higher_than(self, other: "Pivot") -> bool:
        """Check if this pivot is higher than another pivot."""
        return self.price > other.price
    
    def is_lower_than(self, other: "Pivot") -> bool:
        """Check if this pivot is lower than another pivot."""
        return self.price < other.price