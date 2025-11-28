"""Configuration schema for domain models."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PivotConfig(BaseModel):
    """Configuration for pivot detection."""
    
    lookback_left: int = Field(5, ge=1, description="Number of bars to look back on the left")
    lookback_right: int = Field(5, ge=1, description="Number of bars to look back on the right")
    min_strength: float = Field(0.1, ge=0, description="Minimum strength threshold")
    strength_calculation: str = Field("relative", description="Strength calculation method")


class ImbalanceConfig(BaseModel):
    """Configuration for imbalance detection."""
    
    min_size_pips: int = Field(5, ge=0, description="Minimum imbalance size in pips")
    tolerance_pips: int = Field(2, ge=0, description="Tolerance for fill detection in pips")
    max_age_bars: Optional[int] = Field(100, ge=1, description="Maximum age in bars before invalidating")
    require_volume_confirmation: bool = Field(True, description="Require volume confirmation")


class EqualLevelConfig(BaseModel):
    """Configuration for equal level detection."""
    
    tolerance_pips: int = Field(10, ge=0, description="Tolerance for price equality in pips")
    min_touches: int = Field(2, ge=2, description="Minimum number of touches")
    max_age_bars: Optional[int] = Field(200, ge=1, description="Maximum age in bars")
    min_reaction_strength: float = Field(0.5, ge=0, description="Minimum reaction strength")


class DisplacementConfig(BaseModel):
    """Configuration for displacement detection."""
    
    min_bars: int = Field(3, ge=1, description="Minimum displacement duration in bars")
    min_velocity: float = Field(0.001, ge=0, description="Minimum velocity (price change per bar)")
    min_percentage_change: float = Field(0.5, ge=0, description="Minimum percentage change")
    volume_confirmation: bool = Field(True, description="Require volume confirmation")


class MarketStructureConfig(BaseModel):
    """Configuration for market structure analysis."""
    
    confirmation_bars: int = Field(1, ge=0, description="Bars to wait for confirmation")
    min_confidence: float = Field(0.7, ge=0, le=1, description="Minimum confidence level")
    volume_threshold: float = Field(1.5, ge=1, description="Volume multiplier for confirmation")


class DomainModelsConfig(BaseModel):
    """Configuration for all domain models."""
    
    pivot: PivotConfig = Field(default_factory=PivotConfig)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    equal_level: EqualLevelConfig = Field(default_factory=EqualLevelConfig)
    displacement: DisplacementConfig = Field(default_factory=DisplacementConfig)
    market_structure: MarketStructureConfig = Field(default_factory=MarketStructureConfig)
    
    # Global settings
    symbol: str = Field("BTC/USDT", description="Default trading symbol")
    timeframe: str = Field("1h", description="Default timeframe")
    
    def get_pip_value(self, symbol: Optional[str] = None) -> float:
        """Get pip value for a symbol.
        
        Args:
            symbol: Trading symbol (uses default if None)
            
        Returns:
            Pip value in price units
        """
        symbol = symbol or self.symbol
        
        # Common forex pairs
        if any(pair in symbol for pair in ["EUR", "GBP", "USD", "JPY", "CHF", "CAD", "AUD", "NZD"]):
            if "/JPY" in symbol:
                return 0.01  # JPY pairs
            else:
                return 0.0001  # Most forex pairs
        
        # Crypto typically uses smaller price units
        return 0.0001
    
    def tolerance_to_price(self, tolerance_pips: int, symbol: Optional[str] = None) -> float:
        """Convert tolerance in pips to price units.
        
        Args:
            tolerance_pips: Tolerance in pips
            symbol: Trading symbol (uses default if None)
            
        Returns:
            Tolerance in price units
        """
        pip_value = self.get_pip_value(symbol)
        return tolerance_pips * pip_value