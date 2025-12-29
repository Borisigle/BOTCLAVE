"""
DOM (Depth of Market) Builder Module

Builds and maintains a real-time Depth of Market view from order book
snapshots and updates.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from pydantic import BaseModel, Field


class OrderBookLevel(BaseModel):
    """Represents a single level in the order book."""

    price: float = Field(..., description="Price level")
    volume: float = Field(..., description="Volume at this price")
    orders: int = Field(0, description="Number of orders at this level")
    side: str = Field(..., description="Side (bid/ask)")


class DOMState(BaseModel):
    """Complete state of the Depth of Market."""

    timestamp: int = Field(..., description="State timestamp")
    bids: Dict[float, OrderBookLevel] = Field(
        default_factory=dict, description="Bid side of the book"
    )
    asks: Dict[float, OrderBookLevel] = Field(
        default_factory=dict, description="Ask side of the book"
    )
    best_bid: Optional[float] = Field(None, description="Best bid price")
    best_ask: Optional[float] = Field(None, description="Best ask price")
    spread: float = Field(0.0, description="Bid-ask spread")


class LiquidityLevel(BaseModel):
    """Represents a significant liquidity level."""

    price: float = Field(..., description="Price level")
    volume: float = Field(..., description="Total volume")
    side: str = Field(..., description="Side (bid/ask)")
    strength: float = Field(..., description="Relative strength (0-1)")
    first_seen: int = Field(..., description="First appearance timestamp")
    last_seen: int = Field(..., description="Last appearance timestamp")


class DOMBuilder:
    """
    Builds and maintains a Depth of Market view from order book data.
    Tracks liquidity levels, spread changes, and order book dynamics.
    """

    def __init__(
        self,
        max_depth: int = 50,
        significant_level_threshold: float = 2.0,
        track_history: bool = True,
    ):
        """
        Initialize the DOMBuilder.

        Args:
            max_depth: Maximum depth levels to maintain
            significant_level_threshold: Multiplier for significant level detection
            track_history: Whether to track historical states
        """
        self.max_depth = max_depth
        self.significant_level_threshold = significant_level_threshold
        self.track_history = track_history

        self.current_state: Optional[DOMState] = None
        self.state_history: List[DOMState] = []
        self.liquidity_levels: Dict[float, LiquidityLevel] = {}

    def update_from_snapshot(
        self, timestamp: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
    ) -> DOMState:
        """
        Update DOM state from a complete order book snapshot.

        Args:
            timestamp: Snapshot timestamp
            bids: List of (price, volume) tuples for bids
            asks: List of (price, volume) tuples for asks

        Returns:
            Updated DOMState
        """
        bid_levels = {}
        for price, volume in bids[: self.max_depth]:
            bid_levels[price] = OrderBookLevel(
                price=price, volume=volume, side="bid"
            )

        ask_levels = {}
        for price, volume in asks[: self.max_depth]:
            ask_levels[price] = OrderBookLevel(
                price=price, volume=volume, side="ask"
            )

        best_bid = max(bid_levels.keys()) if bid_levels else None
        best_ask = min(ask_levels.keys()) if ask_levels else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0

        self.current_state = DOMState(
            timestamp=timestamp,
            bids=bid_levels,
            asks=ask_levels,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
        )

        if self.track_history:
            self.state_history.append(self.current_state)

        self._update_liquidity_levels(timestamp)

        return self.current_state

    def _update_liquidity_levels(self, timestamp: int) -> None:
        """
        Update tracked liquidity levels based on current state.

        Args:
            timestamp: Current timestamp
        """
        if not self.current_state:
            return

        avg_bid_volume = np.mean(
            [level.volume for level in self.current_state.bids.values()]
        ) if self.current_state.bids else 0.0

        avg_ask_volume = np.mean(
            [level.volume for level in self.current_state.asks.values()]
        ) if self.current_state.asks else 0.0

        for price, level in self.current_state.bids.items():
            if level.volume >= avg_bid_volume * self.significant_level_threshold:
                if price in self.liquidity_levels:
                    self.liquidity_levels[price].last_seen = timestamp
                    self.liquidity_levels[price].volume = level.volume
                else:
                    strength = (
                        level.volume / avg_bid_volume if avg_bid_volume > 0 else 1.0
                    )
                    self.liquidity_levels[price] = LiquidityLevel(
                        price=price,
                        volume=level.volume,
                        side="bid",
                        strength=min(strength, 1.0),
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )

        for price, level in self.current_state.asks.items():
            if level.volume >= avg_ask_volume * self.significant_level_threshold:
                if price in self.liquidity_levels:
                    self.liquidity_levels[price].last_seen = timestamp
                    self.liquidity_levels[price].volume = level.volume
                else:
                    strength = (
                        level.volume / avg_ask_volume if avg_ask_volume > 0 else 1.0
                    )
                    self.liquidity_levels[price] = LiquidityLevel(
                        price=price,
                        volume=level.volume,
                        side="ask",
                        strength=min(strength, 1.0),
                        first_seen=timestamp,
                        last_seen=timestamp,
                    )

    def get_spread_history(self, periods: int = 100) -> pd.Series:
        """
        Get historical spread values.

        Args:
            periods: Number of recent periods to retrieve

        Returns:
            Pandas Series of spread values
        """
        if not self.state_history:
            return pd.Series(dtype=float)

        recent_states = self.state_history[-periods:]
        spreads = [state.spread for state in recent_states]
        timestamps = [state.timestamp for state in recent_states]

        return pd.Series(spreads, index=timestamps)

    def get_bid_ask_imbalance(self) -> float:
        """
        Calculate the current bid-ask imbalance ratio.

        Returns:
            Imbalance ratio (positive = bid pressure, negative = ask pressure)
        """
        if not self.current_state:
            return 0.0

        total_bid_volume = sum(
            level.volume for level in self.current_state.bids.values()
        )
        total_ask_volume = sum(
            level.volume for level in self.current_state.asks.values()
        )

        if total_ask_volume == 0:
            return 1.0 if total_bid_volume > 0 else 0.0

        return (total_bid_volume - total_ask_volume) / (
            total_bid_volume + total_ask_volume
        )

    def get_liquidity_at_price(self, price: float, tolerance: float = 0.01) -> Dict[str, float]:
        """
        Get liquidity information near a specific price.

        Args:
            price: Target price
            tolerance: Price tolerance range (percentage)

        Returns:
            Dictionary with bid and ask liquidity near the price
        """
        if not self.current_state:
            return {"bid": 0.0, "ask": 0.0}

        price_range = price * tolerance
        bid_liquidity = sum(
            level.volume
            for p, level in self.current_state.bids.items()
            if abs(p - price) <= price_range
        )
        ask_liquidity = sum(
            level.volume
            for p, level in self.current_state.asks.items()
            if abs(p - price) <= price_range
        )

        return {"bid": bid_liquidity, "ask": ask_liquidity}

    def get_significant_levels(self, min_age: int = 0) -> List[LiquidityLevel]:
        """
        Get all significant liquidity levels.

        Args:
            min_age: Minimum age in milliseconds for a level to be considered

        Returns:
            List of significant liquidity levels
        """
        if not self.current_state:
            return []

        current_time = self.current_state.timestamp
        return [
            level
            for level in self.liquidity_levels.values()
            if (current_time - level.first_seen) >= min_age
        ]

    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert current DOM state to DataFrames.

        Returns:
            Tuple of (bids_df, asks_df)
        """
        if not self.current_state:
            return pd.DataFrame(), pd.DataFrame()

        bids_data = [
            {"price": level.price, "volume": level.volume, "orders": level.orders}
            for level in self.current_state.bids.values()
        ]
        asks_data = [
            {"price": level.price, "volume": level.volume, "orders": level.orders}
            for level in self.current_state.asks.values()
        ]

        bids_df = pd.DataFrame(bids_data).sort_values("price", ascending=False)
        asks_df = pd.DataFrame(asks_data).sort_values("price", ascending=True)

        return bids_df, asks_df
