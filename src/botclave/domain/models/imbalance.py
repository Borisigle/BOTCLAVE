"""Imbalance model for identifying price imbalances (FVG)."""

from datetime import datetime
from typing import Optional, Tuple

from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import ImbalanceType


class Imbalance(BaseModel):
    """Price imbalance (Fair Value Gap) model representing inefficient price movement."""
    
    # Core imbalance data
    top: float = Field(..., gt=0, description="Top of the imbalance zone")
    bottom: float = Field(..., gt=0, description="Bottom of the imbalance zone")
    timestamp: datetime = Field(..., description="Creation timestamp")
    index: int = Field(..., ge=0, description="Index where imbalance was created")
    
    # Imbalance characteristics
    imbalance_type: ImbalanceType = Field(..., description="Type of imbalance")
    size: float = Field(..., ge=0, description="Size of the imbalance in price units")
    candle_count: int = Field(3, ge=3, description="Number of candles that created the imbalance")
    
    # Metadata
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    is_filled: bool = Field(False, description="Whether imbalance has been filled")
    is_mitigated: bool = Field(False, description="Whether imbalance has been mitigated")
    
    # Fill tracking
    fill_timestamp: Optional[datetime] = Field(None, description="Timestamp when filled")
    fill_price: Optional[float] = Field(None, gt=0, description="Price at which it was filled")
    
    # Validation with tolerance
    tolerance_pips: float = Field(0.0, ge=0, description="Tolerance in pips for fill detection")
    
    @model_validator(mode="after")
    def validate_price_relationships(self) -> "Imbalance":
        """Validate that top and bottom prices are correctly ordered."""
        if self.top <= self.bottom:
            raise ValueError("Top price must be greater than bottom price")
        
        # Validate that size matches the price difference
        expected_size = self.top - self.bottom
        if abs(self.size - expected_size) > 0.00001:  # Small tolerance for floating point
            self.size = expected_size
        
        # Validate type consistency with price context
        # This would typically be validated against actual candle data
        # For buy imbalances, we expect a 3-candle rally
        # For sell imbalances, we expect a 3-candle decline
        
        return self
    
    @computed_field
    @property
    def midpoint(self) -> float:
        """Calculate the midpoint of the imbalance zone."""
        return (self.top + self.bottom) / 2
    
    @computed_field
    @property
    def range_percentage(self) -> float:
        """Calculate the range as a percentage of the midpoint price."""
        return (self.size / self.midpoint) * 100
    
    def is_price_in_imbalance(self, price: float) -> bool:
        """Check if a price is within the imbalance zone."""
        tolerance = self.tolerance_pips / 10000  # Convert pips to price units
        return self.bottom - tolerance <= price <= self.top + tolerance
    
    def check_if_filled(self, current_price: float, timestamp: Optional[datetime] = None) -> bool:
        """Check and update fill status based on current price.
        
        Args:
            current_price: Current price to check against
            timestamp: Timestamp of the current price
            
        Returns:
            True if imbalance is now filled, False otherwise
        """
        if self.is_filled:
            return True
        
        if self.is_price_in_imbalance(current_price):
            self.is_filled = True
            self.is_mitigated = True
            self.fill_timestamp = timestamp or datetime.now()
            self.fill_price = current_price
            return True
        
        return False
    
    @classmethod
    def from_three_candles(
        cls,
        candle1_high: float,
        candle1_low: float,
        candle2_high: float,
        candle2_low: float,
        candle3_high: float,
        candle3_low: float,
        timestamp: datetime,
        index: int,
        symbol: str,
        timeframe: str,
        tolerance_pips: float = 0.0,
    ) -> Optional["Imbalance"]:
        """Create imbalance from three consecutive candles.
        
        Args:
            candle1_high: High of first candle
            candle1_low: Low of first candle
            candle2_high: High of second candle  
            candle2_low: Low of second candle
            candle3_high: High of third candle
            candle3_low: Low of third candle
            timestamp: Creation timestamp
            index: Index in series
            symbol: Trading symbol
            timeframe: Timeframe
            tolerance_pips: Tolerance for fill detection
            
        Returns:
            Imbalance object if one exists, None otherwise
        """
        # Check for buy imbalance (FVG) - strong upward move
        if candle2_low > candle1_high and candle3_low > candle2_low:
            # Buy imbalance between candle1_high and candle2_low
            top = candle2_low
            bottom = candle1_high
            size = top - bottom
            
            if size > 0:  # Valid imbalance
                return cls(
                    top=top,
                    bottom=bottom,
                    size=size,
                    timestamp=timestamp,
                    index=index,
                    imbalance_type=ImbalanceType.BUY,
                    symbol=symbol,
                    timeframe=timeframe,
                    tolerance_pips=tolerance_pips,
                )
        
        # Check for sell imbalance (FVG) - strong downward move
        elif candle2_high < candle1_low and candle3_high < candle2_high:
            # Sell imbalance between candle1_low and candle2_high
            top = candle1_low
            bottom = candle2_high
            size = top - bottom
            
            if size > 0:  # Valid imbalance
                return cls(
                    top=top,
                    bottom=bottom,
                    size=size,
                    timestamp=timestamp,
                    index=index,
                    imbalance_type=ImbalanceType.SELL,
                    symbol=symbol,
                    timeframe=timeframe,
                    tolerance_pips=tolerance_pips,
                )
        
        return None