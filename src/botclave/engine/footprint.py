"""
Footprint Chart Module

Implements footprint charting functionality to visualize order flow
at each price level within a candlestick.

This module provides foundational footprint classes ported from flowsurface (Rust)
for trade-level order flow analysis and absorption detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class NPoc(Enum):
    """State of the Point of Control (POC).

    The POC can change from one candle to another, indicating
    where the most volume has occurred.
    """

    UNCHANGED = "unchanged"
    HIGHER = "higher"
    LOWER = "lower"


@dataclass
class Trade:
    """Represents an executed trade.

    Attributes:
        price: Execution price
        qty: Quantity/volume traded
        is_buy: True if buyer-initiated (market buy), False if seller-initiated
        time_ms: Timestamp in milliseconds
    """

    price: float
    qty: float
    is_buy: bool
    time_ms: int


@dataclass
class GroupedTrades:
    """Aggregates trades grouped by price level.

    This class tracks buy/sell volume and counts at a specific price level,
    enabling absorption signal detection through delta calculation.

    Attributes:
        buy_qty: Total buy volume at this price
        sell_qty: Total sell volume at this price
        first_time_ms: Timestamp of first trade at this price
        last_time_ms: Timestamp of most recent trade at this price
        buy_count: Number of buy trades at this price
        sell_count: Number of sell trades at this price
    """

    buy_qty: float = 0.0
    sell_qty: float = 0.0
    first_time_ms: int = 0
    last_time_ms: int = 0
    buy_count: int = 0
    sell_count: int = 0

    @property
    def total_qty(self) -> float:
        """Total volume (buy + sell) at this price level."""
        return self.buy_qty + self.sell_qty

    @property
    def delta(self) -> float:
        """Delta = buy_qty - sell_qty (ABSORPTION SIGNAL).

        Positive delta indicates net buying pressure (potential absorption of asks).
        Negative delta indicates net selling pressure (potential absorption of bids).
        """
        return self.buy_qty - self.sell_qty

    @property
    def delta_percent(self) -> float:
        """Delta as percentage of total volume (0-1 range).

        Higher values indicate stronger imbalance/absorption signals.
        """
        total = self.total_qty
        if total == 0:
            return 0.0
        return abs(self.delta) / total


@dataclass
class PointOfControl:
    """Price level with maximum volume.

    The Point of Control (POC) represents the price where the most volume
    has traded, often indicating institutional activity or fair value.

    Attributes:
        price: POC price level
        qty: Total volume at POC
        status: NPoc state (unchanged, higher, or lower vs previous POC)
    """

    price: float
    qty: float
    status: NPoc = NPoc.UNCHANGED


class KlineFootprint:
    """Maintains the footprint (trades grouped by price) for a single candle.

    This class provides efficient trade-level order flow analysis similar to
    flowsurface's kline footprint implementation, enabling absorption detection
    through delta analysis at each price level.

    Example:
        >>> fp = KlineFootprint(price_step=1.0)
        >>> fp.add_trade(Trade(price=100.5, qty=10.0, is_buy=True, time_ms=1000))
        >>> fp.calculate_poc()
        101.0
        >>> stats = fp.get_stats()
    """

    def __init__(self, price_step: float = 1.0) -> None:
        """Initialize a KlineFootprint.

        Args:
            price_step: Price rounding step (e.g., 1.0, 0.5, 0.25, 0.01)
                       Trades are grouped by rounding to this step
        """
        self.price_step = price_step
        self.trades: Dict[float, GroupedTrades] = {}  # {price: GroupedTrades}
        self.poc: Optional[PointOfControl] = None

    def _round_price(self, price: float) -> float:
        """Round price to nearest price_step.

        Args:
            price: Original price

        Returns:
            Price rounded to price_step
        """
        return round(price / self.price_step) * self.price_step

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the footprint, grouped by rounded price.

        Args:
            trade: Trade to add
        """
        rounded_price = self._round_price(trade.price)

        if rounded_price not in self.trades:
            self.trades[rounded_price] = GroupedTrades(first_time_ms=trade.time_ms)

        gt = self.trades[rounded_price]

        if trade.is_buy:
            gt.buy_qty += trade.qty
            gt.buy_count += 1
        else:
            gt.sell_qty += trade.qty
            gt.sell_count += 1

        gt.last_time_ms = max(gt.last_time_ms, trade.time_ms)

    def add_trade_batch(self, trades: List[Trade]) -> None:
        """Add multiple trades efficiently.

        Args:
            trades: List of trades to add
        """
        for trade in trades:
            self.add_trade(trade)

    def calculate_poc(self) -> Optional[float]:
        """Calculate Point of Control (price with max volume).

        Returns:
            POC price or None if no trades
        """
        if not self.trades:
            return None

        max_qty = 0.0
        poc_price = None

        for price, gt in self.trades.items():
            if gt.total_qty > max_qty:
                max_qty = gt.total_qty
                poc_price = price

        return poc_price

    def calculate_delta(self) -> Dict[float, float]:
        """Calculate delta (buy - sell) for each price level.

        Returns:
            Dictionary mapping price to delta value
        """
        return {price: gt.delta for price, gt in self.trades.items()}

    def calculate_delta_profile(self) -> Dict[float, float]:
        """Calculate delta percent for each price level.

        Returns:
            Dictionary mapping price to delta_percent value (0-1 range)
        """
        return {price: gt.delta_percent for price, gt in self.trades.items()}

    def get_volume_profile(self) -> Dict[float, float]:
        """Get total volume (buy + sell) per price level.

        Returns:
            Dictionary mapping price to total volume
        """
        return {price: gt.total_qty for price, gt in self.trades.items()}

    def get_imbalance(
        self, price: float, threshold: float = 0.65
    ) -> Optional[str]:
        """Detect imbalance (absorption) at a specific price.

        Args:
            price: Price level to check
            threshold: Delta percent threshold (0-1) for considering it an imbalance

        Returns:
            'buy' if delta > 0 and delta_percent >= threshold,
            'sell' if delta < 0 and delta_percent >= threshold,
            None otherwise
        """
        if price not in self.trades:
            return None

        gt = self.trades[price]
        delta_pct = gt.delta_percent

        if delta_pct >= threshold:
            return 'buy' if gt.delta > 0 else 'sell'
        return None

    def get_highest_lowest_prices(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the highest and lowest prices in the footprint.

        Returns:
            Tuple of (highest_price, lowest_price) or (None, None) if empty
        """
        if not self.trades:
            return (None, None)

        prices = list(self.trades.keys())
        return (max(prices), min(prices))

    def clear(self) -> None:
        """Clear all trades from the footprint."""
        self.trades.clear()
        self.poc = None

    def get_stats(self) -> Dict:
        """Get footprint statistics.

        Returns:
            Dictionary with total_volume, total_buy, total_sell,
            delta, price_levels, and poc
        """
        total_buy = sum(gt.buy_qty for gt in self.trades.values())
        total_sell = sum(gt.sell_qty for gt in self.trades.values())
        total = total_buy + total_sell

        return {
            'total_volume': total,
            'total_buy': total_buy,
            'total_sell': total_sell,
            'delta': total_buy - total_sell,
            'price_levels': len(self.trades),
            'poc': self.poc.price if self.poc else None,
        }


@dataclass
class KlineDataPoint:
    """Combines a Kline (candlestick) with its Footprint.

    This class merges traditional OHLCV data with trade-level footprint
    analysis, enabling comprehensive order flow analysis per candle.

    Attributes:
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Total volume (from exchange)
        time_ms: Candle timestamp in milliseconds
        footprint: KlineFootprint with trade-level data
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    time_ms: int
    footprint: KlineFootprint = field(default_factory=lambda: KlineFootprint())

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to this candle's footprint.

        Args:
            trade: Trade to add
        """
        self.footprint.add_trade(trade)

    def get_footprint_stats(self) -> Dict:
        """Get footprint statistics for this candle.

        Returns:
            Dictionary with footprint statistics
        """
        return self.footprint.get_stats()


class FootprintBar(BaseModel):
    """Represents a single bar in the footprint chart."""

    timestamp: int = Field(..., description="Bar timestamp")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Total volume")
    buy_volume: float = Field(0.0, description="Buy side volume")
    sell_volume: float = Field(0.0, description="Sell side volume")
    delta: float = Field(0.0, description="Volume delta (buy - sell)")
    price_levels: Dict[float, Dict[str, float]] = Field(
        default_factory=dict, description="Volume at each price level"
    )


class FootprintMetrics(BaseModel):
    """Metrics calculated from footprint analysis."""

    cumulative_delta: float = Field(..., description="Cumulative volume delta")
    poc: float = Field(..., description="Point of Control (max volume price)")
    value_area_high: float = Field(..., description="Value Area High")
    value_area_low: float = Field(..., description="Value Area Low")
    imbalance_levels: List[float] = Field(
        default_factory=list, description="Price levels with significant imbalances"
    )


class FootprintChart:
    """
    Generates and analyzes footprint charts from trade data.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        imbalance_ratio: float = 1.5,
        value_area_percent: float = 0.7,
    ):
        """
        Initialize the FootprintChart.

        Args:
            tick_size: Minimum price movement (tick size)
            imbalance_ratio: Ratio to detect imbalances
            value_area_percent: Percentage for value area calculation (0.7 = 70%)
        """
        self.tick_size = tick_size
        self.imbalance_ratio = imbalance_ratio
        self.value_area_percent = value_area_percent
        self.bars: List[FootprintBar] = []

    def build_from_trades(
        self, trades_df: pd.DataFrame, timeframe: str = "5T"
    ) -> List[FootprintBar]:
        """
        Build footprint bars from raw trade data.

        Args:
            trades_df: DataFrame with columns [timestamp, price, volume, side]
            timeframe: Pandas timeframe string (e.g., '5T' for 5 minutes)

        Returns:
            List of FootprintBar objects
        """
        if trades_df.empty:
            return []

        bars = []

        return bars

    def calculate_delta(self, bar: FootprintBar) -> float:
        """
        Calculate volume delta for a footprint bar.

        Args:
            bar: The footprint bar

        Returns:
            Volume delta (buy_volume - sell_volume)
        """
        return bar.buy_volume - bar.sell_volume

    def find_point_of_control(self, bar: FootprintBar) -> float:
        """
        Find the Point of Control (price with highest volume).

        Args:
            bar: The footprint bar

        Returns:
            POC price level
        """
        if not bar.price_levels:
            return bar.close

        max_volume = 0.0
        poc_price = bar.close

        for price, volumes in bar.price_levels.items():
            total_volume = volumes.get("buy", 0.0) + volumes.get("sell", 0.0)
            if total_volume > max_volume:
                max_volume = total_volume
                poc_price = price

        return poc_price

    def calculate_value_area(self, bar: FootprintBar) -> tuple[float, float]:
        """
        Calculate Value Area High and Low.

        Args:
            bar: The footprint bar

        Returns:
            Tuple of (value_area_high, value_area_low)
        """
        if not bar.price_levels:
            return bar.high, bar.low

        total_volume = sum(
            volumes.get("buy", 0.0) + volumes.get("sell", 0.0)
            for volumes in bar.price_levels.values()
        )

        target_volume = total_volume * self.value_area_percent

        sorted_levels = sorted(
            bar.price_levels.items(),
            key=lambda x: x[1].get("buy", 0.0) + x[1].get("sell", 0.0),
            reverse=True,
        )

        accumulated_volume = 0.0
        value_area_prices = []

        for price, volumes in sorted_levels:
            volume = volumes.get("buy", 0.0) + volumes.get("sell", 0.0)
            accumulated_volume += volume
            value_area_prices.append(price)
            if accumulated_volume >= target_volume:
                break

        if not value_area_prices:
            return bar.high, bar.low

        return max(value_area_prices), min(value_area_prices)

    def detect_imbalances(self, bar: FootprintBar) -> List[float]:
        """
        Detect price levels with significant buy/sell imbalances.

        Args:
            bar: The footprint bar

        Returns:
            List of price levels with imbalances
        """
        imbalance_prices = []

        for price, volumes in bar.price_levels.items():
            buy_vol = volumes.get("buy", 0.0)
            sell_vol = volumes.get("sell", 0.0)

            if sell_vol == 0 and buy_vol > 0:
                imbalance_prices.append(price)
            elif buy_vol == 0 and sell_vol > 0:
                imbalance_prices.append(price)
            elif buy_vol > 0 and sell_vol > 0:
                ratio = max(buy_vol, sell_vol) / min(buy_vol, sell_vol)
                if ratio >= self.imbalance_ratio:
                    imbalance_prices.append(price)

        return imbalance_prices

    def calculate_cumulative_delta(self, bars: Optional[List[FootprintBar]] = None) -> pd.Series:
        """
        Calculate cumulative delta across multiple bars.

        Args:
            bars: List of bars to analyze (uses self.bars if None)

        Returns:
            Pandas Series of cumulative delta values
        """
        if bars is None:
            bars = self.bars

        if not bars:
            return pd.Series(dtype=float)

        deltas = [self.calculate_delta(bar) for bar in bars]
        return pd.Series(deltas).cumsum()

    def get_metrics(self, bar: FootprintBar) -> FootprintMetrics:
        """
        Calculate all footprint metrics for a bar.

        Args:
            bar: The footprint bar

        Returns:
            FootprintMetrics object with all calculated metrics
        """
        poc = self.find_point_of_control(bar)
        vah, val = self.calculate_value_area(bar)
        imbalances = self.detect_imbalances(bar)
        cumulative_delta = self.calculate_delta(bar)

        return FootprintMetrics(
            cumulative_delta=cumulative_delta,
            poc=poc,
            value_area_high=vah,
            value_area_low=val,
            imbalance_levels=imbalances,
        )

    def to_dataframe(self, bars: Optional[List[FootprintBar]] = None) -> pd.DataFrame:
        """
        Convert footprint bars to a DataFrame.

        Args:
            bars: List of bars to convert (uses self.bars if None)

        Returns:
            DataFrame with footprint data
        """
        if bars is None:
            bars = self.bars

        if not bars:
            return pd.DataFrame()

        data = []
        for bar in bars:
            data.append(
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "buy_volume": bar.buy_volume,
                    "sell_volume": bar.sell_volume,
                    "delta": bar.delta,
                }
            )

        return pd.DataFrame(data)
