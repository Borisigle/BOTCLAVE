"""
Depth Analysis Module

Analyzes order book depth to identify areas of absorption, imbalances,
and liquidity concentration.

This module provides foundational order book management classes ported
from flowsurface (Rust) for real-time order book maintenance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


@dataclass
class DeOrder:
    """Represents an order in the order book.

    Attributes:
        price: Price level of the order
        qty: Quantity/volume at this price level
    """

    price: float
    qty: float


class Depth:
    """Maintains the order book (bids + asks) with real-time updates.

    This class provides efficient order book management similar to flowsurface's
    depth implementation, allowing for both full snapshots and incremental updates.

    Example:
        >>> depth = Depth()
        >>> depth.update([(100.0, 10.0), (99.5, 5.0)], [(101.0, 8.0)])
        >>> depth.mid_price()
        100.5
    """

    def __init__(self) -> None:
        """Initialize an empty order book."""
        self.bids: Dict[float, float] = {}  # {price: qty}
        self.asks: Dict[float, float] = {}  # {price: qty}

    def update(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> None:
        """Update bids/asks with new values.

        If qty=0, removes the price level. If qty>0, updates or inserts.

        Args:
            bids: List of (price, qty) tuples for bid levels
            asks: List of (price, qty) tuples for ask levels
        """
        # Update bids
        for price, qty in bids:
            if qty == 0:
                # Remove level if qty is 0
                self.bids.pop(price, None)
            else:
                # Update or insert level
                self.bids[price] = qty

        # Update asks
        for price, qty in asks:
            if qty == 0:
                # Remove level if qty is 0
                self.asks.pop(price, None)
            else:
                # Update or insert level
                self.asks[price] = qty

    def snapshot(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> None:
        """Replace the entire order book with a snapshot.

        Args:
            bids: List of (price, qty) tuples for bid levels
            asks: List of (price, qty) tuples for ask levels
        """
        self.bids.clear()
        self.asks.clear()
        self.update(bids, asks)

    def mid_price(self) -> Optional[float]:
        """Calculate the mid price (best_bid + best_ask) / 2.

        Returns:
            Mid price or None if book is empty
        """
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if best_bid is None or best_ask is None:
            return None

        return (best_bid[0] + best_ask[0]) / 2.0

    def best_bid(self) -> Optional[Tuple[float, float]]:
        """Get the best bid (highest price).

        Returns:
            Tuple of (price, qty) for best bid or None if no bids
        """
        if not self.bids:
            return None

        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])

    def best_ask(self) -> Optional[Tuple[float, float]]:
        """Get the best ask (lowest price).

        Returns:
            Tuple of (price, qty) for best ask or None if no asks
        """
        if not self.asks:
            return None

        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])

    def get_level(
        self, price: float, side: str = "both"
    ) -> Optional[Tuple[float, float, float]]:
        """Get bid_qty, ask_qty, mid_price at a specific price level.

        Args:
            price: Price level to query
            side: 'bid', 'ask', or 'both' to specify which side to query

        Returns:
            Tuple of (bid_qty, ask_qty, mid_price) or None if no data
        """
        bid_qty = self.bids.get(price, 0.0)
        ask_qty = self.asks.get(price, 0.0)
        mid_price = self.mid_price()

        if side == "bid" and bid_qty == 0:
            return None
        if side == "ask" and ask_qty == 0:
            return None
        if side == "both" and bid_qty == 0 and ask_qty == 0:
            return None

        return (bid_qty, ask_qty, mid_price if mid_price else 0.0)


class LocalDepthCache:
    """Maintains a cached, updated depth object from WebSocket updates.

    This class manages order book synchronization similar to flowsurface's
    LocalDepthCache, handling both initial snapshots and incremental diff updates.

    Example:
        >>> cache = LocalDepthCache()
        >>> cache.update_snapshot([(100.0, 10.0)], [(101.0, 8.0)], 1, 1000)
        >>> cache.update_diff([(100.5, 5.0)], [], 2, 1001)
        >>> depth = cache.get_depth()
    """

    def __init__(self) -> None:
        """Initialize an empty depth cache."""
        self.depth = Depth()
        self.last_update_id = 0
        self.time_ms = 0

    def update_snapshot(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
        update_id: int, time_ms: int
    ) -> None:
        """Receive a complete snapshot (full reset of order book).

        Args:
            bids: List of (price, qty) tuples for bid levels
            asks: List of (price, qty) tuples for ask levels
            update_id: Update sequence ID
            time_ms: Timestamp in milliseconds
        """
        self.depth.snapshot(bids, asks)
        self.last_update_id = update_id
        self.time_ms = time_ms

    def update_diff(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
        update_id: int, time_ms: int
    ) -> None:
        """Receive incremental diff updates to the order book.

        Args:
            bids: List of (price, qty) tuples for bid level changes
            asks: List of (price, qty) tuples for ask level changes
            update_id: Update sequence ID
            time_ms: Timestamp in milliseconds
        """
        self.depth.update(bids, asks)
        self.last_update_id = update_id
        self.time_ms = time_ms

    def get_depth(self) -> Depth:
        """Get the current Depth object.

        Returns:
            Current Depth object with latest order book state
        """
        return self.depth


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
