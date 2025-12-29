"""
Depth Analysis Module

Analyzes order book depth to identify areas of absorption, imbalances,
and liquidity concentration.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class DepthLevel(BaseModel):
    """Represents a single level in the order book depth."""

    price: float = Field(..., description="Price level")
    bid_volume: float = Field(0.0, description="Total bid volume at this level")
    ask_volume: float = Field(0.0, description="Total ask volume at this level")
    timestamp: int = Field(..., description="Timestamp of the depth snapshot")


class DepthSnapshot(BaseModel):
    """Complete order book depth snapshot."""

    timestamp: int = Field(..., description="Snapshot timestamp")
    bids: List[DepthLevel] = Field(default_factory=list, description="Bid levels")
    asks: List[DepthLevel] = Field(default_factory=list, description="Ask levels")
    symbol: str = Field(..., description="Trading pair symbol")


class AbsorptionZone(BaseModel):
    """Identifies a zone where significant order absorption occurred."""

    price_start: float = Field(..., description="Start price of absorption zone")
    price_end: float = Field(..., description="End price of absorption zone")
    volume_absorbed: float = Field(..., description="Total volume absorbed")
    side: str = Field(..., description="Side of absorption (bid/ask)")
    timestamp: int = Field(..., description="When absorption was detected")
    strength: float = Field(..., description="Absorption strength (0-1)")


class DepthAnalyzer:
    """
    Analyzes order book depth data to identify absorption zones,
    imbalances, and significant liquidity levels.
    """

    def __init__(
        self,
        absorption_threshold: float = 2.0,
        imbalance_threshold: float = 1.5,
        min_volume: float = 1.0,
    ):
        """
        Initialize the DepthAnalyzer.

        Args:
            absorption_threshold: Multiplier for detecting absorption events
            imbalance_threshold: Ratio threshold for imbalance detection
            min_volume: Minimum volume to consider for analysis
        """
        self.absorption_threshold = absorption_threshold
        self.imbalance_threshold = imbalance_threshold
        self.min_volume = min_volume
        self.depth_history: List[DepthSnapshot] = []

    def add_snapshot(self, snapshot: DepthSnapshot) -> None:
        """
        Add a new depth snapshot to the history.

        Args:
            snapshot: The depth snapshot to add
        """
        self.depth_history.append(snapshot)

    def calculate_imbalance(
        self, bids: List[DepthLevel], asks: List[DepthLevel]
    ) -> float:
        """
        Calculate the order book imbalance ratio.

        Args:
            bids: List of bid levels
            asks: List of ask levels

        Returns:
            Imbalance ratio (positive = bid pressure, negative = ask pressure)
        """
        total_bid_volume = sum(level.bid_volume for level in bids)
        total_ask_volume = sum(level.ask_volume for level in asks)

        if total_ask_volume == 0:
            return float("inf") if total_bid_volume > 0 else 0.0

        return total_bid_volume / total_ask_volume

    def detect_absorption_zones(
        self, lookback_periods: int = 10
    ) -> List[AbsorptionZone]:
        """
        Detect absorption zones from recent depth snapshots.

        Args:
            lookback_periods: Number of recent snapshots to analyze

        Returns:
            List of detected absorption zones
        """
        if len(self.depth_history) < lookback_periods:
            return []

        absorption_zones: List[AbsorptionZone] = []
        recent_snapshots = self.depth_history[-lookback_periods:]

        return absorption_zones

    def get_liquidity_heatmap(
        self, price_levels: int = 50
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a liquidity heatmap from depth history.

        Args:
            price_levels: Number of price levels to include

        Returns:
            Tuple of (bid_heatmap, ask_heatmap) DataFrames
        """
        if not self.depth_history:
            return pd.DataFrame(), pd.DataFrame()

        bid_data = []
        ask_data = []

        for snapshot in self.depth_history:
            bid_prices = [level.price for level in snapshot.bids[:price_levels]]
            bid_volumes = [level.bid_volume for level in snapshot.bids[:price_levels]]
            ask_prices = [level.price for level in snapshot.asks[:price_levels]]
            ask_volumes = [level.ask_volume for level in snapshot.asks[:price_levels]]

            bid_data.append(
                {
                    "timestamp": snapshot.timestamp,
                    "prices": bid_prices,
                    "volumes": bid_volumes,
                }
            )
            ask_data.append(
                {
                    "timestamp": snapshot.timestamp,
                    "prices": ask_prices,
                    "volumes": ask_volumes,
                }
            )

        return pd.DataFrame(bid_data), pd.DataFrame(ask_data)

    def identify_support_resistance(
        self, min_touches: int = 3
    ) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels from depth concentration.

        Args:
            min_touches: Minimum number of times a level must be tested

        Returns:
            Dictionary with 'support' and 'resistance' price levels
        """
        support_levels = []
        resistance_levels = []

        return {"support": support_levels, "resistance": resistance_levels}

    def get_depth_delta(
        self, snapshot1: DepthSnapshot, snapshot2: DepthSnapshot
    ) -> Dict[str, float]:
        """
        Calculate the change in depth between two snapshots.

        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot

        Returns:
            Dictionary with bid_delta and ask_delta
        """
        bid_delta = sum(level.bid_volume for level in snapshot2.bids) - sum(
            level.bid_volume for level in snapshot1.bids
        )
        ask_delta = sum(level.ask_volume for level in snapshot2.asks) - sum(
            level.ask_volume for level in snapshot1.asks
        )

        return {"bid_delta": bid_delta, "ask_delta": ask_delta}
