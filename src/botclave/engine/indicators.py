"""
Order Flow Indicators Module

Implements various order flow indicators including cumulative delta,
volume profile, and ICT-specific indicators, plus SMC (Smart Money Concepts)
indicators for advanced market structure analysis.
"""

from dataclasses import dataclass
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


# ============================================================================
# SMC (Smart Money Concepts) INDICATORS
# ============================================================================


@dataclass
class Swing:
    """Represents a swing high/low (structural pivot)."""

    index: int  # Position in DataFrame
    price: float  # Price of the swing
    time: str  # Timestamp
    swing_type: str  # 'high' or 'low'


@dataclass
class BreakOfStructure:
    """Represents a Break of Structure."""

    index: int  # Position where it occurs
    price: float  # Price of the break
    time: str
    direction: str  # 'bullish' or 'bearish'
    broken_level: float  # The level that was broken


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap."""

    index: int  # Starting position
    time: str
    top_price: float  # Upper price of the gap
    bottom_price: float  # Lower price of the gap
    direction: str  # 'bullish' or 'bearish'
    size_percent: float  # Size as % of current range


@dataclass
class ChangeOfCharacter:
    """Represents a change of market character."""

    index: int
    time: str
    previous_trend: str  # 'uptrend' or 'downtrend'
    new_trend: str
    trigger_price: float  # Price where change was confirmed


@dataclass
class OrderBlock:
    """Represents an Order Block (absorption zone)."""

    index: int  # Starting index
    time: str
    high_price: float  # Top of the candle
    low_price: float  # Bottom of the candle
    direction: str  # 'bullish' or 'bearish'
    mitigated_index: Optional[int]  # Where it was mitigated (if applicable)
    volume: float  # Candle volume


@dataclass
class LiquidityCluster:
    """Liquidity zone (support/resistance)."""

    price: float
    strength: int  # How many times it was touched
    last_touched_index: int
    direction: str  # 'support' or 'resistance'


@dataclass
class RetracementLevels:
    """Retracement levels (Fibonacci, etc)."""

    level_0: float  # 0% = start
    level_236: float  # 23.6%
    level_382: float  # 38.2%
    level_500: float  # 50%
    level_618: float  # 61.8%
    level_786: float  # 78.6%
    level_1: float  # 100% = end


class SwingDetector:
    """Detects swing highs and lows (structural pivots)."""

    def __init__(self, left_bars: int = 2, right_bars: int = 2):
        """
        Initialize swing detector.

        Args:
            left_bars: How many bars to look left
            right_bars: How many bars to look right
        """
        self.left_bars = left_bars
        self.right_bars = right_bars

    def find_swings(self, df: pd.DataFrame) -> List[Swing]:
        """
        Find all swings in the dataframe.

        A swing high = candle where high > all neighboring candles (left+right)
        A swing low = candle where low < all neighboring candles (left+right)

        Args:
            df: DataFrame with 'high', 'low' columns and datetime index

        Returns:
            List[Swing]
        """
        if len(df) < self.left_bars + self.right_bars + 1:
            return []

        swings = []

        # Iterate through valid range (excluding edges)
        for i in range(self.left_bars, len(df) - self.right_bars):
            # Check for swing high
            is_swing_high = True
            current_high = df["high"].iloc[i]

            for j in range(i - self.left_bars, i + self.right_bars + 1):
                if j != i and df["high"].iloc[j] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                swings.append(
                    Swing(index=i, price=current_high, time=time_str, swing_type="high")
                )

            # Check for swing low
            is_swing_low = True
            current_low = df["low"].iloc[i]

            for j in range(i - self.left_bars, i + self.right_bars + 1):
                if j != i and df["low"].iloc[j] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                swings.append(
                    Swing(index=i, price=current_low, time=time_str, swing_type="low")
                )

        # Sort by index
        swings.sort(key=lambda x: x.index)

        return swings

    def get_last_swing(self, df: pd.DataFrame, swing_type: str) -> Optional[Swing]:
        """
        Get the most recent swing high/low.

        Args:
            df: DataFrame
            swing_type: 'high' or 'low'

        Returns:
            Most recent Swing of the specified type, or None
        """
        swings = self.find_swings(df)
        filtered = [s for s in swings if s.swing_type == swing_type]
        return filtered[-1] if filtered else None

    def get_last_n_swings(self, df: pd.DataFrame, n: int = 5) -> List[Swing]:
        """
        Get the last N swings (alternating high/low).

        Args:
            df: DataFrame
            n: Number of swings to return

        Returns:
            List of last N swings
        """
        swings = self.find_swings(df)
        return swings[-n:] if len(swings) >= n else swings


class BreakOfStructureDetector:
    """Detects BOS (changes in market structure)."""

    def __init__(self):
        pass

    def find_bos(self, df: pd.DataFrame, swings: List[Swing]) -> List[BreakOfStructure]:
        """
        Find Break of Structure events.

        BOS Bullish = when price closes above previous swing high
        BOS Bearish = when price closes below previous swing low

        Args:
            df: DataFrame
            swings: Previously detected swings

        Returns:
            List[BreakOfStructure]
        """
        if len(swings) < 2:
            return []

        bos_list = []

        # Separate swing highs and lows
        swing_highs = [s for s in swings if s.swing_type == "high"]
        swing_lows = [s for s in swings if s.swing_type == "low"]

        # Check for bullish BOS (break above swing high)
        for swing_high in swing_highs:
            # Look for candles after the swing
            for i in range(swing_high.index + 1, len(df)):
                close_price = df["close"].iloc[i]
                if close_price > swing_high.price:
                    time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                    bos_list.append(
                        BreakOfStructure(
                            index=i,
                            price=close_price,
                            time=time_str,
                            direction="bullish",
                            broken_level=swing_high.price,
                        )
                    )
                    break  # Only first break counts

        # Check for bearish BOS (break below swing low)
        for swing_low in swing_lows:
            # Look for candles after the swing
            for i in range(swing_low.index + 1, len(df)):
                close_price = df["close"].iloc[i]
                if close_price < swing_low.price:
                    time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                    bos_list.append(
                        BreakOfStructure(
                            index=i,
                            price=close_price,
                            time=time_str,
                            direction="bearish",
                            broken_level=swing_low.price,
                        )
                    )
                    break  # Only first break counts

        # Sort by index
        bos_list.sort(key=lambda x: x.index)

        return bos_list

    def get_last_bos(
        self, df: pd.DataFrame, swings: List[Swing], direction: Optional[str] = None
    ) -> Optional[BreakOfStructure]:
        """
        Get the last BOS.

        Args:
            df: DataFrame
            swings: Detected swings
            direction: Optional filter by 'bullish' or 'bearish'

        Returns:
            Most recent BOS, or None
        """
        bos_list = self.find_bos(df, swings)

        if direction:
            bos_list = [b for b in bos_list if b.direction == direction]

        return bos_list[-1] if bos_list else None


class FairValueGapDetector:
    """Detects Fair Value Gaps (gaps without fill)."""

    def __init__(self, min_size_percent: float = 0.02):
        """
        Initialize FFG detector.

        Args:
            min_size_percent: Minimum gap size as % of range
        """
        self.min_size_percent = min_size_percent

    def find_ffg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Find Fair Value Gaps.

        FFG Bullish = gap between low[i] and high[i-2] (where candle[i-1] doesn't fill)
        FFG Bearish = gap between high[i] and low[i-2] (where candle[i-1] doesn't fill)

        Args:
            df: DataFrame

        Returns:
            List[FairValueGap]
        """
        if len(df) < 3:
            return []

        ffg_list = []
        atr = df["high"].rolling(14).max() - df["low"].rolling(14).min()
        avg_range = atr.mean() if not atr.empty else 1.0

        for i in range(2, len(df)):
            # Check for bullish FFG
            gap_bottom = df["high"].iloc[i - 2]
            gap_top = df["low"].iloc[i]
            middle_high = df["high"].iloc[i - 1]
            middle_low = df["low"].iloc[i - 1]

            if gap_top > gap_bottom and middle_high < gap_top and middle_low > gap_bottom:
                gap_size = gap_top - gap_bottom
                size_percent = gap_size / avg_range if avg_range > 0 else 0

                if size_percent >= self.min_size_percent:
                    time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                    ffg_list.append(
                        FairValueGap(
                            index=i,
                            time=time_str,
                            top_price=gap_top,
                            bottom_price=gap_bottom,
                            direction="bullish",
                            size_percent=size_percent,
                        )
                    )

            # Check for bearish FFG
            gap_top = df["low"].iloc[i - 2]
            gap_bottom = df["high"].iloc[i]
            middle_high = df["high"].iloc[i - 1]
            middle_low = df["low"].iloc[i - 1]

            if gap_bottom < gap_top and middle_low > gap_bottom and middle_high < gap_top:
                gap_size = gap_top - gap_bottom
                size_percent = gap_size / avg_range if avg_range > 0 else 0

                if size_percent >= self.min_size_percent:
                    time_str = str(df.index[i]) if hasattr(df.index[i], "__str__") else str(i)
                    ffg_list.append(
                        FairValueGap(
                            index=i,
                            time=time_str,
                            top_price=gap_top,
                            bottom_price=gap_bottom,
                            direction="bearish",
                            size_percent=size_percent,
                        )
                    )

        return ffg_list

    def get_active_ffg(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Get FFGs that haven't been filled yet (active).

        Args:
            df: DataFrame

        Returns:
            List of active FFGs
        """
        all_ffg = self.find_ffg(df)
        active = []

        for ffg in all_ffg:
            filled_index = self.fill_ffg(df, ffg)
            if filled_index is None:
                active.append(ffg)

        return active

    def fill_ffg(self, df: pd.DataFrame, ffg: FairValueGap) -> Optional[int]:
        """
        Check if an FFG has been filled.

        Args:
            df: DataFrame
            ffg: FairValueGap to check

        Returns:
            Index where it was filled, or None if still open
        """
        # Check candles after the FFG
        for i in range(ffg.index + 1, len(df)):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]

            # FFG is filled if price moves through the gap
            if ffg.direction == "bullish":
                if low <= ffg.bottom_price:
                    return i
            else:  # bearish
                if high >= ffg.top_price:
                    return i

        return None


class ChangeOfCharacterDetector:
    """Detects changes in market character (trend reversals)."""

    def __init__(self):
        pass

    def find_choch(self, df: pd.DataFrame, swings: List[Swing]) -> List[ChangeOfCharacter]:
        """
        Find Change of Character events.

        CHoCH = confirmation of trend reversal
        - Uptrend confirmed = swing low > previous swing low
        - Downtrend confirmed = swing high < previous swing high

        Args:
            df: DataFrame
            swings: Detected swings

        Returns:
            List[ChangeOfCharacter]
        """
        if len(swings) < 4:
            return []

        choch_list = []

        # Separate swings by type
        swing_highs = [s for s in swings if s.swing_type == "high"]
        swing_lows = [s for s in swings if s.swing_type == "low"]

        # Detect downtrend to uptrend (higher low)
        for i in range(1, len(swing_lows)):
            if swing_lows[i].price > swing_lows[i - 1].price:
                time_str = (
                    str(df.index[swing_lows[i].index])
                    if hasattr(df.index[swing_lows[i].index], "__str__")
                    else str(swing_lows[i].index)
                )
                choch_list.append(
                    ChangeOfCharacter(
                        index=swing_lows[i].index,
                        time=time_str,
                        previous_trend="downtrend",
                        new_trend="uptrend",
                        trigger_price=swing_lows[i].price,
                    )
                )

        # Detect uptrend to downtrend (lower high)
        for i in range(1, len(swing_highs)):
            if swing_highs[i].price < swing_highs[i - 1].price:
                time_str = (
                    str(df.index[swing_highs[i].index])
                    if hasattr(df.index[swing_highs[i].index], "__str__")
                    else str(swing_highs[i].index)
                )
                choch_list.append(
                    ChangeOfCharacter(
                        index=swing_highs[i].index,
                        time=time_str,
                        previous_trend="uptrend",
                        new_trend="downtrend",
                        trigger_price=swing_highs[i].price,
                    )
                )

        # Sort by index
        choch_list.sort(key=lambda x: x.index)

        return choch_list


class OrderBlockDetector:
    """Detects Order Blocks (zones where smart money entered)."""

    def __init__(self):
        pass

    def find_order_blocks(
        self, df: pd.DataFrame, bos: List[BreakOfStructure], volume_threshold: float = 1.5
    ) -> List[OrderBlock]:
        """
        Find Order Blocks.

        Order Block = candle BEFORE the BOS that showed absorption
        - Bullish OB: candle before bullish BOS
        - Bearish OB: candle before bearish BOS

        Args:
            df: DataFrame with 'high', 'low', 'volume'
            bos: Previously detected BOS
            volume_threshold: Volume multiplier threshold

        Returns:
            List[OrderBlock]
        """
        if len(bos) == 0 or "volume" not in df.columns:
            return []

        order_blocks = []
        avg_volume = df["volume"].mean()

        for b in bos:
            # Get the candle before the BOS
            if b.index > 0:
                ob_index = b.index - 1
                volume = df["volume"].iloc[ob_index]

                # Check if volume is elevated
                if volume >= avg_volume * volume_threshold:
                    time_str = (
                        str(df.index[ob_index])
                        if hasattr(df.index[ob_index], "__str__")
                        else str(ob_index)
                    )
                    order_blocks.append(
                        OrderBlock(
                            index=ob_index,
                            time=time_str,
                            high_price=df["high"].iloc[ob_index],
                            low_price=df["low"].iloc[ob_index],
                            direction=b.direction,
                            mitigated_index=None,
                            volume=volume,
                        )
                    )

        return order_blocks

    def find_mitigated_blocks(self, df: pd.DataFrame, blocks: List[OrderBlock]) -> List[OrderBlock]:
        """
        Mark which order blocks have been mitigated (price touched their range).

        Args:
            df: DataFrame
            blocks: Order blocks to check

        Returns:
            Updated list of order blocks with mitigation info
        """
        for block in blocks:
            # Check candles after the order block
            for i in range(block.index + 1, len(df)):
                high = df["high"].iloc[i]
                low = df["low"].iloc[i]

                # Check if price entered the order block range
                if low <= block.high_price and high >= block.low_price:
                    block.mitigated_index = i
                    break

        return blocks


class LiquidityDetector:
    """Detects liquidity clusters (key levels)."""

    def __init__(self, tolerance_percent: float = 0.005):
        """
        Initialize liquidity detector.

        Args:
            tolerance_percent: Price range for grouping (e.g., 0.5% = ±0.5%)
        """
        self.tolerance_percent = tolerance_percent

    def find_liquidity_clusters(
        self, df: pd.DataFrame, swings: List[Swing]
    ) -> List[LiquidityCluster]:
        """
        Detect zones where market returns frequently (clustered swing highs/lows).

        Args:
            df: DataFrame
            swings: Detected swings

        Returns:
            List[LiquidityCluster]
        """
        if len(swings) == 0:
            return []

        clusters = []

        # Group swings by proximity
        for swing in swings:
            # Check if this swing belongs to an existing cluster
            found_cluster = False

            for cluster in clusters:
                price_diff = abs(swing.price - cluster.price) / cluster.price
                if price_diff <= self.tolerance_percent:
                    # Update cluster
                    cluster.strength += 1
                    cluster.last_touched_index = max(cluster.last_touched_index, swing.index)
                    # Update average price
                    cluster.price = (cluster.price * (cluster.strength - 1) + swing.price) / cluster.strength
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                direction = "resistance" if swing.swing_type == "high" else "support"
                clusters.append(
                    LiquidityCluster(
                        price=swing.price,
                        strength=1,
                        last_touched_index=swing.index,
                        direction=direction,
                    )
                )

        # Filter clusters with strength > 1 (touched multiple times)
        significant_clusters = [c for c in clusters if c.strength > 1]

        return significant_clusters


def get_previous_highs_lows(
    df: pd.DataFrame, periods: Optional[List[int]] = None
) -> Dict[str, List[float]]:
    """
    Get significant highs/lows from the past.

    Args:
        df: DataFrame
        periods: List of periods to check (e.g., [20, 50, 200])

    Returns:
        Dictionary with 'highs' and 'lows' lists
    """
    if periods is None:
        periods = [20, 50, 200]

    levels = {"highs": [], "lows": []}

    for period in periods:
        if len(df) >= period:
            # Get high/low of the period
            period_high = df["high"].iloc[-period:].max()
            period_low = df["low"].iloc[-period:].min()

            levels["highs"].append(period_high)
            levels["lows"].append(period_low)

    return levels


def calculate_retracement_levels(start_price: float, end_price: float) -> RetracementLevels:
    """
    Calculate Fibonacci retracement levels between two points.

    Args:
        start_price: Starting price (0%)
        end_price: Ending price (100%)

    Returns:
        RetracementLevels object
    """
    diff = end_price - start_price

    return RetracementLevels(
        level_0=start_price,
        level_236=start_price + diff * 0.236,
        level_382=start_price + diff * 0.382,
        level_500=start_price + diff * 0.500,
        level_618=start_price + diff * 0.618,
        level_786=start_price + diff * 0.786,
        level_1=end_price,
    )


class SMCIndicator:
    """Master class that encapsulates all SMC indicators."""

    def __init__(
        self, left_bars: int = 2, right_bars: int = 2, min_ffg_percent: float = 0.02
    ):
        """
        Initialize SMC indicator suite.

        Args:
            left_bars: Bars to look left for swing detection
            right_bars: Bars to look right for swing detection
            min_ffg_percent: Minimum FFG size as % of range
        """
        self.swing_detector = SwingDetector(left_bars, right_bars)
        self.bos_detector = BreakOfStructureDetector()
        self.ffg_detector = FairValueGapDetector(min_ffg_percent)
        self.choch_detector = ChangeOfCharacterDetector()
        self.ob_detector = OrderBlockDetector()
        self.liquidity_detector = LiquidityDetector()

        # Cache for last calculations
        self._last_df = None
        self._swings_cache = []
        self._bos_cache = []
        self._ffg_cache = []
        self._ob_cache = []

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze a DataFrame and return all SMC indicators.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume'

        Returns:
            Dictionary with all SMC indicators
        """
        # 1. Detect swings
        swings = self.swing_detector.find_swings(df)

        # 2. Detect BOS based on swings
        bos = self.bos_detector.find_bos(df, swings)

        # 3. Detect FFG
        ffg = self.ffg_detector.find_ffg(df)
        active_ffg = self.ffg_detector.get_active_ffg(df)

        # 4. Detect CHoCH
        choch = self.choch_detector.find_choch(df, swings)

        # 5. Detect Order Blocks
        order_blocks = self.ob_detector.find_order_blocks(df, bos)
        self.ob_detector.find_mitigated_blocks(df, order_blocks)

        # 6. Detect liquidity
        liquidity = self.liquidity_detector.find_liquidity_clusters(df, swings)

        # Cache results
        self._last_df = df
        self._swings_cache = swings
        self._bos_cache = bos
        self._ffg_cache = ffg
        self._ob_cache = order_blocks

        # 7. Return complete analysis
        return {
            "swings": swings,
            "bos": bos,
            "ffg": ffg,
            "choch": choch,
            "order_blocks": order_blocks,
            "liquidity": liquidity,
            "last_swing": swings[-1] if swings else None,
            "last_bos": bos[-1] if bos else None,
            "active_ffg": active_ffg,
            "last_high": df["high"].iloc[-1],
            "last_low": df["low"].iloc[-1],
        }

    def get_bias(self, df: pd.DataFrame) -> str:
        """
        Get current market bias: 'bullish', 'bearish', or 'neutral'.

        Based on:
        - Swing structure
        - Last BOS
        - Trend

        Args:
            df: DataFrame

        Returns:
            Market bias string
        """
        swings = self.swing_detector.find_swings(df)
        bos = self.bos_detector.find_bos(df, swings)

        if not swings or len(swings) < 2:
            return "neutral"

        # Check last BOS
        if bos:
            last_bos = bos[-1]
            if last_bos.direction == "bullish":
                return "bullish"
            elif last_bos.direction == "bearish":
                return "bearish"

        # Check swing structure
        swing_highs = [s for s in swings if s.swing_type == "high"]
        swing_lows = [s for s in swings if s.swing_type == "low"]

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Higher highs and higher lows = bullish
            if (
                swing_highs[-1].price > swing_highs[-2].price
                and swing_lows[-1].price > swing_lows[-2].price
            ):
                return "bullish"

            # Lower highs and lower lows = bearish
            if (
                swing_highs[-1].price < swing_highs[-2].price
                and swing_lows[-1].price < swing_lows[-2].price
            ):
                return "bearish"

        return "neutral"

    def get_entry_level(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """
        Get recommended entry level for a direction.

        Args:
            df: DataFrame
            direction: 'long' or 'short'

        Returns:
            Entry price level, or None
        """
        # Get active FFGs and order blocks
        active_ffg = self.ffg_detector.get_active_ffg(df)
        order_blocks = self._ob_cache if self._ob_cache else self.ob_detector.find_order_blocks(
            df, self._bos_cache
        )

        if direction == "long":
            # Look for bullish FFG or bullish OB
            bullish_ffg = [f for f in active_ffg if f.direction == "bullish"]
            bullish_ob = [ob for ob in order_blocks if ob.direction == "bullish" and ob.mitigated_index is None]

            if bullish_ffg:
                # Enter at bottom of FFG
                return bullish_ffg[-1].bottom_price

            if bullish_ob:
                # Enter at top of order block
                return bullish_ob[-1].high_price

        elif direction == "short":
            # Look for bearish FFG or bearish OB
            bearish_ffg = [f for f in active_ffg if f.direction == "bearish"]
            bearish_ob = [ob for ob in order_blocks if ob.direction == "bearish" and ob.mitigated_index is None]

            if bearish_ffg:
                # Enter at top of FFG
                return bearish_ffg[-1].top_price

            if bearish_ob:
                # Enter at bottom of order block
                return bearish_ob[-1].low_price

        return None

    def get_stop_loss(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """
        Get recommended stop loss level.

        Args:
            df: DataFrame
            direction: 'long' or 'short'

        Returns:
            Stop loss price, or None
        """
        swings = self._swings_cache if self._swings_cache else self.swing_detector.find_swings(df)

        if not swings:
            return None

        if direction == "long":
            # SL below last swing low
            swing_lows = [s for s in swings if s.swing_type == "low"]
            if swing_lows:
                return swing_lows[-1].price * 0.998  # Slight buffer

        elif direction == "short":
            # SL above last swing high
            swing_highs = [s for s in swings if s.swing_type == "high"]
            if swing_highs:
                return swing_highs[-1].price * 1.002  # Slight buffer

        return None

    def get_take_profit_levels(
        self, df: pd.DataFrame, direction: str, count: int = 3
    ) -> List[float]:
        """
        Get multiple take profit levels.

        Args:
            df: DataFrame
            direction: 'long' or 'short'
            count: Number of TP levels to return

        Returns:
            List of take profit prices
        """
        swings = self._swings_cache if self._swings_cache else self.swing_detector.find_swings(df)
        liquidity = self.liquidity_detector.find_liquidity_clusters(df, swings)

        tp_levels = []

        if direction == "long":
            # TP at swing highs and resistance clusters
            swing_highs = [s for s in swings if s.swing_type == "high"]
            resistances = [liq for liq in liquidity if liq.direction == "resistance"]

            # Add swing highs
            for swing in reversed(swing_highs):
                if len(tp_levels) < count:
                    tp_levels.append(swing.price)

            # Add resistance levels
            for res in reversed(resistances):
                if len(tp_levels) < count:
                    tp_levels.append(res.price)

        elif direction == "short":
            # TP at swing lows and support clusters
            swing_lows = [s for s in swings if s.swing_type == "low"]
            supports = [liq for liq in liquidity if liq.direction == "support"]

            # Add swing lows
            for swing in reversed(swing_lows):
                if len(tp_levels) < count:
                    tp_levels.append(swing.price)

            # Add support levels
            for sup in reversed(supports):
                if len(tp_levels) < count:
                    tp_levels.append(sup.price)

        return tp_levels[:count]
