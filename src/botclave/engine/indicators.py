"""
Order Flow Indicators Module

Implements various order flow indicators including cumulative delta,
volume profile, and ICT-specific indicators.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class VolumeProfile(BaseModel):
    """Volume profile calculation result."""

    price_levels: List[float] = Field(..., description="Price levels")
    volumes: List[float] = Field(..., description="Volume at each level")
    poc: float = Field(..., description="Point of Control")
    value_area_high: float = Field(..., description="Value Area High")
    value_area_low: float = Field(..., description="Value Area Low")


class OrderFlowSignal(BaseModel):
    """Order flow trading signal."""

    timestamp: int = Field(..., description="Signal timestamp")
    signal_type: str = Field(..., description="Signal type (buy/sell/neutral)")
    strength: float = Field(..., description="Signal strength (0-1)")
    price: float = Field(..., description="Price at signal")
    indicators: Dict[str, float] = Field(
        default_factory=dict, description="Contributing indicator values"
    )
    description: str = Field(..., description="Signal description")


class OrderFlowIndicators:
    """
    Calculates various order flow indicators for trading analysis.
    """

    def __init__(
        self,
        value_area_percent: float = 0.7,
        delta_period: int = 20,
        cvd_period: int = 50,
    ):
        """
        Initialize OrderFlowIndicators.

        Args:
            value_area_percent: Percentage for value area calculation
            delta_period: Period for delta calculations
            cvd_period: Period for cumulative volume delta
        """
        self.value_area_percent = value_area_percent
        self.delta_period = delta_period
        self.cvd_period = cvd_period

    def calculate_cumulative_delta(
        self, buy_volume: pd.Series, sell_volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Cumulative Volume Delta (CVD).

        Args:
            buy_volume: Series of buy volumes
            sell_volume: Series of sell volumes

        Returns:
            Cumulative delta series
        """
        delta = buy_volume - sell_volume
        return delta.cumsum()

    def calculate_volume_profile(
        self, df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume"
    ) -> VolumeProfile:
        """
        Calculate volume profile for the given data.

        Args:
            df: DataFrame with price and volume data
            price_col: Name of price column
            volume_col: Name of volume column

        Returns:
            VolumeProfile object
        """
        if df.empty:
            return VolumeProfile(
                price_levels=[], volumes=[], poc=0.0, value_area_high=0.0, value_area_low=0.0
            )

        min_price = df[price_col].min()
        max_price = df[price_col].max()
        num_levels = 50

        price_levels = np.linspace(min_price, max_price, num_levels)
        volumes = []

        for i in range(len(price_levels) - 1):
            mask = (df[price_col] >= price_levels[i]) & (
                df[price_col] < price_levels[i + 1]
            )
            volume_at_level = df.loc[mask, volume_col].sum()
            volumes.append(volume_at_level)

        volumes.append(0.0)

        poc_idx = np.argmax(volumes)
        poc = price_levels[poc_idx]

        total_volume = sum(volumes)
        target_volume = total_volume * self.value_area_percent

        sorted_indices = np.argsort(volumes)[::-1]
        accumulated_volume = 0.0
        value_area_indices = []

        for idx in sorted_indices:
            accumulated_volume += volumes[idx]
            value_area_indices.append(idx)
            if accumulated_volume >= target_volume:
                break

        value_area_prices = [price_levels[i] for i in value_area_indices]
        value_area_high = max(value_area_prices) if value_area_prices else max_price
        value_area_low = min(value_area_prices) if value_area_prices else min_price

        return VolumeProfile(
            price_levels=price_levels.tolist(),
            volumes=volumes,
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
        )

    def calculate_delta_divergence(
        self, price: pd.Series, delta: pd.Series, lookback: int = 14
    ) -> pd.Series:
        """
        Detect divergences between price and volume delta.

        Args:
            price: Price series
            delta: Volume delta series
            lookback: Lookback period for divergence detection

        Returns:
            Series with divergence signals (1 = bullish, -1 = bearish, 0 = none)
        """
        if len(price) < lookback or len(delta) < lookback:
            return pd.Series([0] * len(price), index=price.index)

        divergence = pd.Series([0] * len(price), index=price.index)

        for i in range(lookback, len(price)):
            price_window = price.iloc[i - lookback : i]
            delta_window = delta.iloc[i - lookback : i]

            price_trend = price_window.iloc[-1] - price_window.iloc[0]
            delta_trend = delta_window.iloc[-1] - delta_window.iloc[0]

            if price_trend > 0 and delta_trend < 0:
                divergence.iloc[i] = -1
            elif price_trend < 0 and delta_trend > 0:
                divergence.iloc[i] = 1

        return divergence

    def calculate_absorption(
        self,
        price: pd.Series,
        volume: pd.Series,
        delta: pd.Series,
        threshold: float = 2.0,
    ) -> pd.Series:
        """
        Detect absorption events (high volume with little price movement).

        Args:
            price: Price series
            volume: Volume series
            delta: Volume delta series
            threshold: Volume threshold multiplier

        Returns:
            Series with absorption signals
        """
        if len(price) < 2:
            return pd.Series([0] * len(price), index=price.index)

        price_change = price.pct_change().abs()
        volume_avg = volume.rolling(window=20, min_periods=1).mean()
        volume_std = volume.rolling(window=20, min_periods=1).std()

        absorption = pd.Series([0] * len(price), index=price.index)

        for i in range(1, len(price)):
            if volume.iloc[i] > volume_avg.iloc[i] + threshold * volume_std.iloc[i]:
                if price_change.iloc[i] < price_change.mean():
                    absorption.iloc[i] = 1 if delta.iloc[i] > 0 else -1

        return absorption

    def calculate_liquidity_voids(
        self, df: pd.DataFrame, volume_threshold: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Identify liquidity voids (areas with abnormally low volume).

        Args:
            df: DataFrame with OHLCV data
            volume_threshold: Threshold as fraction of average volume

        Returns:
            List of (start_price, end_price) tuples representing voids
        """
        if df.empty or "volume" not in df.columns:
            return []

        avg_volume = df["volume"].mean()
        threshold_volume = avg_volume * volume_threshold

        voids = []
        void_start = None

        for idx, row in df.iterrows():
            if row["volume"] < threshold_volume:
                if void_start is None:
                    void_start = row["low"]
            else:
                if void_start is not None:
                    void_end = row["high"]
                    voids.append((void_start, void_end))
                    void_start = None

        return voids

    def calculate_imbalance_score(
        self, buy_volume: pd.Series, sell_volume: pd.Series
    ) -> pd.Series:
        """
        Calculate order flow imbalance score.

        Args:
            buy_volume: Buy volume series
            sell_volume: Sell volume series

        Returns:
            Imbalance score series (-1 to 1)
        """
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / total_volume.replace(0, np.nan)
        return imbalance.fillna(0)

    def generate_signal(
        self,
        timestamp: int,
        price: float,
        indicators: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> OrderFlowSignal:
        """
        Generate a trading signal from indicator values.

        Args:
            timestamp: Signal timestamp
            price: Current price
            indicators: Dictionary of indicator values
            thresholds: Optional custom thresholds

        Returns:
            OrderFlowSignal object
        """
        if thresholds is None:
            thresholds = {
                "cvd": 0.5,
                "imbalance": 0.3,
                "absorption": 0.7,
            }

        signal_strength = 0.0
        signal_type = "neutral"
        contributing_factors = []

        cvd = indicators.get("cumulative_delta", 0.0)
        if abs(cvd) > thresholds["cvd"]:
            signal_strength += 0.3
            signal_type = "buy" if cvd > 0 else "sell"
            contributing_factors.append("CVD")

        imbalance = indicators.get("imbalance", 0.0)
        if abs(imbalance) > thresholds["imbalance"]:
            signal_strength += 0.3
            contributing_factors.append("Imbalance")

        absorption = indicators.get("absorption", 0.0)
        if abs(absorption) > thresholds["absorption"]:
            signal_strength += 0.4
            contributing_factors.append("Absorption")

        signal_strength = min(signal_strength, 1.0)

        description = (
            f"{signal_type.upper()} signal "
            f"(strength: {signal_strength:.2f}) "
            f"based on: {', '.join(contributing_factors)}"
        )

        return OrderFlowSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            strength=signal_strength,
            price=price,
            indicators=indicators,
            description=description,
        )

    def calculate_all_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate all available indicators for the given DataFrame.

        Args:
            df: DataFrame with OHLCV and order flow data

        Returns:
            DataFrame with all calculated indicators
        """
        result_df = df.copy()

        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            result_df["cvd"] = self.calculate_cumulative_delta(
                df["buy_volume"], df["sell_volume"]
            )
            result_df["imbalance"] = self.calculate_imbalance_score(
                df["buy_volume"], df["sell_volume"]
            )

        if "volume" in df.columns and "close" in df.columns:
            delta = df.get("buy_volume", df["volume"] / 2) - df.get(
                "sell_volume", df["volume"] / 2
            )
            result_df["delta"] = delta

            if "close" in df.columns:
                result_df["divergence"] = self.calculate_delta_divergence(
                    df["close"], delta
                )
                result_df["absorption"] = self.calculate_absorption(
                    df["close"], df["volume"], delta
                )

        return result_df
