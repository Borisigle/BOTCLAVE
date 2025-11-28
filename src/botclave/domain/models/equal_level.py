"""Equal level model for identifying equal highs and lows."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class EqualLevel(BaseModel):
    """Equal level model representing price levels with multiple touches."""
    
    # Core level data
    price: float = Field(..., gt=0, description="Equal price level")
    timestamp: datetime = Field(..., description="Most recent touch timestamp")
    indices: List[int] = Field(..., min_length=2, description="Indices where this level was touched")
    
    # Level characteristics
    touches: int = Field(..., ge=2, description="Number of touches at this level")
    tolerance: float = Field(..., ge=0, description="Tolerance for price equality")
    level_type: str = Field(..., description="Type of level (high/low)")
    
    # Metadata
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    is_active: bool = Field(True, description="Whether level is still relevant")
    
    # Additional context
    volume_at_touches: Optional[List[float]] = Field(None, description="Volume at each touch")
    reaction_strength: Optional[List[float]] = Field(None, description="Reaction strength at each touch")
    
    @model_validator(mode="after")
    def validate_consistency(self) -> "EqualLevel":
        """Validate model consistency."""
        if len(self.indices) != self.touches:
            self.touches = len(self.indices)
        
        if self.volume_at_touches is not None and len(self.volume_at_touches) != self.touches:
            raise ValueError("Volume list length must match number of touches")
        
        if self.reaction_strength is not None and len(self.reaction_strength) != self.touches:
            raise ValueError("Reaction strength list length must match number of touches")
        
        return self
    
    @computed_field
    @property
    def first_touch_index(self) -> int:
        """Get the index of the first touch."""
        return min(self.indices)
    
    @computed_field
    @property
    def last_touch_index(self) -> int:
        """Get the index of the last touch."""
        return max(self.indices)
    
    @computed_field
    @property
    def age_in_bars(self) -> int:
        """Calculate how many bars since first touch."""
        return self.last_touch_index - self.first_touch_index
    
    @computed_field
    @property
    def average_volume(self) -> Optional[float]:
        """Calculate average volume at touches."""
        if not self.volume_at_touches:
            return None
        return sum(self.volume_at_touches) / len(self.volume_at_touches)
    
    @computed_field
    @property
    def average_reaction_strength(self) -> Optional[float]:
        """Calculate average reaction strength."""
        if not self.reaction_strength:
            return None
        return sum(self.reaction_strength) / len(self.reaction_strength)
    
    def is_price_at_level(self, price: float) -> bool:
        """Check if price is at this equal level within tolerance."""
        return abs(price - self.price) <= self.tolerance
    
    def add_touch(
        self,
        index: int,
        timestamp: datetime,
        volume: Optional[float] = None,
        reaction_strength: Optional[float] = None,
    ) -> None:
        """Add a new touch to this equal level.
        
        Args:
            index: Index of the new touch
            timestamp: Timestamp of the new touch
            volume: Volume at the touch
            reaction_strength: Reaction strength at the touch
        """
        if index not in self.indices:
            self.indices.append(index)
            self.touches += 1
            self.timestamp = timestamp  # Update to most recent
            
            if volume is not None:
                if self.volume_at_touches is None:
                    self.volume_at_touches = [None] * (self.touches - 1)  # Initialize with None for previous touches
                self.volume_at_touches.append(volume)
            
            if reaction_strength is not None:
                if self.reaction_strength is None:
                    self.reaction_strength = [None] * (self.touches - 1)  # Initialize with None for previous touches
                self.reaction_strength.append(reaction_strength)
    
    @classmethod
    def find_equal_levels(
        cls,
        prices: List[float],
        indices: List[int],
        timestamps: List[datetime],
        symbol: str,
        timeframe: str,
        tolerance: float,
        level_type: str,
        min_touches: int = 2,
    ) -> List["EqualLevel"]:
        """Find equal levels from a series of prices.
        
        Args:
            prices: List of prices to analyze
            indices: Corresponding indices
            timestamps: Corresponding timestamps
            symbol: Trading symbol
            timeframe: Timeframe
            tolerance: Tolerance for price equality
            level_type: Type of level ('high' or 'low')
            min_touches: Minimum number of touches to consider a level
            
        Returns:
            List of equal levels found
        """
        if len(prices) != len(indices) or len(prices) != len(timestamps):
            raise ValueError("Prices, indices, and timestamps must have same length")
        
        levels = []
        level_map = {}  # Map rounded price to level data
        
        for i, (price, index, timestamp) in enumerate(zip(prices, indices, timestamps)):
            # Round price to tolerance precision for grouping
            rounded_price = round(price / tolerance) * tolerance
            
            if rounded_price not in level_map:
                level_map[rounded_price] = {
                    "actual_price": price,
                    "indices": [index],
                    "timestamps": [timestamp],
                }
            else:
                level_map[rounded_price]["indices"].append(index)
                level_map[rounded_price]["timestamps"].append(timestamp)
        
        # Create level objects for those with enough touches
        for rounded_price, level_data in level_map.items():
            if len(level_data["indices"]) >= min_touches:
                levels.append(
                    cls(
                        price=level_data["actual_price"],
                        timestamp=level_data["timestamps"][-1],  # Most recent
                        indices=level_data["indices"],
                        touches=len(level_data["indices"]),
                        tolerance=tolerance,
                        level_type=level_type,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
        
        return levels