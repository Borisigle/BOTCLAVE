"""
Footprint Chart Module

Implements footprint charting functionality to visualize order flow
at each price level within a candlestick.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


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
