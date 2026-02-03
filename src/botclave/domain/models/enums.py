"""Enums for domain models."""

from enum import Enum


class Timeframe(str, Enum):
    """Timeframe enumeration for different chart periods."""
    
    SCALPER = "ST"  # Short-term (scalper)
    INTRADAY = "IT"  # Intraday 
    LONGTERM = "LT"  # Long-term


class Direction(str, Enum):
    """Direction enumeration for price movement."""
    
    BULLISH = "bullish"
    BEARISH = "bearish"


class StructureType(str, Enum):
    """Market structure type enumeration."""
    
    HIGHER_HIGH = "higher_high"
    LOWER_LOW = "lower_low"
    HIGHER_LOW = "higher_low"
    LOWER_HIGH = "lower_high"


class ImbalanceType(str, Enum):
    """Imbalance type enumeration."""
    
    BUY = "buy"  # FVG (Fair Value Gap) - bullish imbalance
    SELL = "sell"  # FVG - bearish imbalance