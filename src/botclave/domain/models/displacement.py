"""Displacement event model for tracking strong price movements."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import Direction


class DisplacementEvent(BaseModel):
    """Displacement event representing strong, directional price movements."""
    
    # Core displacement data
    start_price: float = Field(..., gt=0, description="Starting price of displacement")
    end_price: float = Field(..., gt=0, description="Ending price of displacement")
    start_timestamp: datetime = Field(..., description="Start timestamp")
    end_timestamp: datetime = Field(..., description="End timestamp")
    start_index: int = Field(..., ge=0, description="Start index in series")
    end_index: int = Field(..., ge=0, description="End index in series")
    
    # Displacement characteristics
    direction: Direction = Field(..., description="Direction of displacement")
    magnitude: float = Field(..., ge=0, description="Price magnitude of displacement")
    duration_bars: int = Field(..., ge=1, description="Duration in bars")
    
    # Volume data
    entry_volume: Optional[float] = Field(None, ge=0, description="Volume at entry")
    peak_volume: Optional[float] = Field(None, ge=0, description="Peak volume during displacement")
    exit_volume: Optional[float] = Field(None, ge=0, description="Volume at exit")
    
    # Metadata
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    is_valid: bool = Field(True, description="Whether displacement is valid")
    
    # Additional metrics
    velocity: Optional[float] = Field(None, description="Price change per bar")
    acceleration: Optional[float] = Field(None, description="Change in velocity")
    
    @model_validator(mode="after")
    def validate_displacement_consistency(self) -> "DisplacementEvent":
        """Validate displacement consistency and compute derived fields."""
        # Validate temporal consistency
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("End timestamp must be after start timestamp")
        
        if self.end_index < self.start_index:
            raise ValueError("End index must be after start index")
        
        # Validate direction consistency
        if self.direction == Direction.BULLISH and self.end_price <= self.start_price:
            raise ValueError("Bullish displacement must have end_price > start_price")
        elif self.direction == Direction.BEARISH and self.end_price >= self.start_price:
            raise ValueError("Bearish displacement must have end_price < start_price")
        
        # Validate magnitude
        expected_magnitude = abs(self.end_price - self.start_price)
        if abs(self.magnitude - expected_magnitude) > 0.00001:
            self.magnitude = expected_magnitude
        
        # Compute duration
        self.duration_bars = self.end_index - self.start_index + 1
        
        # Compute velocity if not provided
        if self.velocity is None and self.duration_bars > 0:
            self.velocity = self.magnitude / self.duration_bars
        
        return self
    
    @computed_field
    @property
    def midpoint_price(self) -> float:
        """Calculate the midpoint price of the displacement."""
        return (self.start_price + self.end_price) / 2
    
    @computed_field
    @property
    def percentage_change(self) -> float:
        """Calculate the percentage change of the displacement."""
        return (self.magnitude / self.start_price) * 100
    
    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Calculate the duration in seconds."""
        return (self.end_timestamp - self.start_timestamp).total_seconds()
    
    @computed_field
    @property
    def is_strong(self) -> bool:
        """Determine if this is a strong displacement based on magnitude and duration."""
        # Strong displacement: large magnitude relative to duration
        if self.velocity is None:
            return False
        
        # Threshold can be configurable based on symbol/timeframe
        velocity_threshold = 0.001  # 0.1% per bar as default threshold
        return self.velocity > velocity_threshold
    
    def get_fibonacci_level(self, level: float) -> float:
        """Calculate Fibonacci retracement level.
        
        Args:
            level: Fibonacci level (0.0 to 1.0, e.g., 0.382, 0.5, 0.618)
            
        Returns:
            Price at the Fibonacci level
        """
        if not 0 <= level <= 1:
            raise ValueError("Fibonacci level must be between 0 and 1")
        
        if self.direction == Direction.BULLISH:
            return self.end_price - (self.magnitude * level)
        else:  # BEARISH
            return self.end_price + (self.magnitude * level)
    
    def is_price_in_displacement(self, price: float) -> bool:
        """Check if a price is within the displacement range."""
        if self.direction == Direction.BULLISH:
            return self.start_price <= price <= self.end_price
        else:  # BEARISH
            return self.end_price <= price <= self.start_price
    
    @classmethod
    def from_price_series(
        cls,
        start_index: int,
        end_index: int,
        start_price: float,
        end_price: float,
        start_timestamp: datetime,
        end_timestamp: datetime,
        symbol: str,
        timeframe: str,
        entry_volume: Optional[float] = None,
        peak_volume: Optional[float] = None,
        exit_volume: Optional[float] = None,
    ) -> "DisplacementEvent":
        """Create displacement event from price series data.
        
        Args:
            start_index: Start index
            end_index: End index
            start_price: Start price
            end_price: End price
            start_timestamp: Start timestamp
            end_timestamp: End timestamp
            symbol: Trading symbol
            timeframe: Timeframe
            entry_volume: Volume at entry
            peak_volume: Peak volume
            exit_volume: Volume at exit
            
        Returns:
            DisplacementEvent object
        """
        # Determine direction
        direction = Direction.BULLISH if end_price > start_price else Direction.BEARISH
        
        # Calculate magnitude
        magnitude = abs(end_price - start_price)
        
        # Calculate duration
        duration_bars = end_index - start_index + 1
        
        # Calculate velocity
        velocity = magnitude / duration_bars if duration_bars > 0 else 0
        
        return cls(
            start_price=start_price,
            end_price=end_price,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            start_index=start_index,
            end_index=end_index,
            direction=direction,
            magnitude=magnitude,
            duration_bars=duration_bars,
            entry_volume=entry_volume,
            peak_volume=peak_volume,
            exit_volume=exit_volume,
            symbol=symbol,
            timeframe=timeframe,
            velocity=velocity,
        )