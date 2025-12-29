"""
Order Flow Engine Module

This module contains the core order flow analysis components including
depth analysis, footprint charting, DOM building, indicators, and strategy logic.
"""

from .depth import (
    DepthAnalyzer,
    DeOrder,
    Depth,
    LocalDepthCache,
)
from .footprint import (
    FootprintChart,
    NPoc,
    Trade,
    GroupedTrades,
    PointOfControl,
    KlineFootprint,
    KlineDataPoint,
)
from .dom_builder import DOMBuilder
from .indicators import (
    OrderFlowIndicators,
    SMCIndicator,
    SwingDetector,
    BreakOfStructureDetector,
    FairValueGapDetector,
    ChangeOfCharacterDetector,
    OrderBlockDetector,
    LiquidityDetector,
    Swing,
    BreakOfStructure,
    FairValueGap,
    ChangeOfCharacter,
    OrderBlock,
    LiquidityCluster,
    RetracementLevels,
    calculate_retracement_levels,
    get_previous_highs_lows,
)
from .strategy import (
    TradingStrategy,
    RiskRewardSetup,
    Signal,
    OrderflowAnalyzer,
    MultiTimeframeStrategy,
)

__all__ = [
    # Foundational depth classes (from flowsurface port)
    "DeOrder",
    "Depth",
    "LocalDepthCache",
    # Foundational footprint classes (from flowsurface port)
    "NPoc",
    "Trade",
    "GroupedTrades",
    "PointOfControl",
    "KlineFootprint",
    "KlineDataPoint",
    # Analyzer classes
    "DepthAnalyzer",
    "FootprintChart",
    "DOMBuilder",
    "OrderFlowIndicators",
    "TradingStrategy",
    "RiskRewardSetup",
    "Signal",
    "OrderflowAnalyzer",
    "MultiTimeframeStrategy",
    # SMC Indicators
    "SMCIndicator",
    "SwingDetector",
    "BreakOfStructureDetector",
    "FairValueGapDetector",
    "ChangeOfCharacterDetector",
    "OrderBlockDetector",
    "LiquidityDetector",
    # SMC Data classes
    "Swing",
    "BreakOfStructure",
    "FairValueGap",
    "ChangeOfCharacter",
    "OrderBlock",
    "LiquidityCluster",
    "RetracementLevels",
    # SMC Helper functions
    "calculate_retracement_levels",
    "get_previous_highs_lows",
]
