"""
Order Flow Strategy Module

Implements the core trading strategy logic based on order flow analysis,
ICT concepts, and absorption patterns.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from .indicators import OrderFlowIndicators, OrderFlowSignal
from .depth import DepthAnalyzer, AbsorptionZone
from .footprint import FootprintChart


class PositionSide(str, Enum):
    """Position side enumeration."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeSignal(BaseModel):
    """Trading signal with entry, stop loss, and take profit levels."""

    timestamp: int = Field(..., description="Signal timestamp")
    symbol: str = Field(..., description="Trading symbol")
    side: PositionSide = Field(..., description="Position side")
    entry_price: float = Field(..., description="Entry price")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: List[float] = Field(..., description="Take profit levels")
    confidence: float = Field(..., description="Signal confidence (0-1)")
    reasoning: str = Field(..., description="Signal reasoning")
    risk_reward_ratio: float = Field(..., description="Risk/reward ratio")


class StrategyConfig(BaseModel):
    """Configuration for the order flow strategy."""

    min_confidence: float = Field(0.6, description="Minimum signal confidence")
    risk_reward_ratio: float = Field(2.0, description="Minimum risk/reward ratio")
    use_absorption: bool = Field(True, description="Use absorption analysis")
    use_imbalance: bool = Field(True, description="Use order imbalance")
    use_footprint: bool = Field(True, description="Use footprint analysis")
    position_size_pct: float = Field(0.02, description="Position size as % of capital")
    max_positions: int = Field(3, description="Maximum concurrent positions")
    timeframes: List[str] = Field(
        default_factory=lambda: ["5m", "15m", "1h"], description="Analysis timeframes"
    )


class Position(BaseModel):
    """Open trading position."""

    symbol: str = Field(..., description="Trading symbol")
    side: PositionSide = Field(..., description="Position side")
    entry_price: float = Field(..., description="Entry price")
    quantity: float = Field(..., description="Position quantity")
    entry_time: int = Field(..., description="Entry timestamp")
    stop_loss: float = Field(..., description="Stop loss price")
    take_profit: List[float] = Field(..., description="Take profit levels")
    current_pnl: float = Field(0.0, description="Current P&L")


class OrderFlowStrategy:
    """
    Main trading strategy based on order flow analysis.
    Combines multiple indicators and ICT concepts for trading decisions.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize the OrderFlowStrategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or StrategyConfig()
        self.indicators = OrderFlowIndicators()
        self.depth_analyzer = DepthAnalyzer()
        self.footprint_chart = FootprintChart()

        self.open_positions: Dict[str, Position] = {}
        self.signal_history: List[TradeSignal] = []

    def analyze_market_structure(
        self, df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Analyze market structure from OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with market structure analysis
        """
        structure = {
            "trend": "neutral",
            "strength": 0.0,
            "swing_highs": [],
            "swing_lows": [],
            "key_levels": [],
        }

        if df.empty or len(df) < 20:
            return structure

        close_prices = df["close"]
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()

        if len(sma_20) > 0 and len(sma_50) > 0:
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                structure["trend"] = "bullish"
                structure["strength"] = min(
                    (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1], 1.0
                )
            elif sma_20.iloc[-1] < sma_50.iloc[-1]:
                structure["trend"] = "bearish"
                structure["strength"] = min(
                    (sma_50.iloc[-1] - sma_20.iloc[-1]) / sma_50.iloc[-1], 1.0
                )

        highs = df["high"]
        lows = df["low"]

        for i in range(2, len(df) - 2):
            if (
                highs.iloc[i] > highs.iloc[i - 1]
                and highs.iloc[i] > highs.iloc[i - 2]
                and highs.iloc[i] > highs.iloc[i + 1]
                and highs.iloc[i] > highs.iloc[i + 2]
            ):
                structure["swing_highs"].append((i, highs.iloc[i]))

            if (
                lows.iloc[i] < lows.iloc[i - 1]
                and lows.iloc[i] < lows.iloc[i - 2]
                and lows.iloc[i] < lows.iloc[i + 1]
                and lows.iloc[i] < lows.iloc[i + 2]
            ):
                structure["swing_lows"].append((i, lows.iloc[i]))

        return structure

    def detect_order_blocks(
        self, df: pd.DataFrame
    ) -> List[Dict[str, any]]:
        """
        Detect ICT order blocks from price action.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected order blocks
        """
        order_blocks = []

        if df.empty or len(df) < 10:
            return order_blocks

        for i in range(3, len(df) - 1):
            close_change = abs(df["close"].iloc[i] - df["close"].iloc[i - 1])
            avg_change = df["close"].diff().abs().rolling(window=20).mean().iloc[i]

            if close_change > avg_change * 1.5:
                order_block = {
                    "type": "bullish" if df["close"].iloc[i] > df["open"].iloc[i] else "bearish",
                    "high": df["high"].iloc[i - 1],
                    "low": df["low"].iloc[i - 1],
                    "timestamp": df.index[i] if hasattr(df.index, "__getitem__") else i,
                    "strength": min(close_change / avg_change, 3.0) / 3.0,
                }
                order_blocks.append(order_block)

        return order_blocks

    def calculate_entry_exit_levels(
        self,
        current_price: float,
        signal_side: PositionSide,
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> Tuple[float, float, List[float]]:
        """
        Calculate entry, stop loss, and take profit levels.

        Args:
            current_price: Current market price
            signal_side: Position side
            support_levels: List of support levels
            resistance_levels: List of resistance levels

        Returns:
            Tuple of (entry_price, stop_loss, take_profit_levels)
        """
        entry_price = current_price

        if signal_side == PositionSide.LONG:
            nearby_support = [s for s in support_levels if s < current_price]
            stop_loss = (
                max(nearby_support) * 0.995
                if nearby_support
                else current_price * 0.98
            )

            risk = current_price - stop_loss
            take_profit = [
                current_price + risk * self.config.risk_reward_ratio,
                current_price + risk * self.config.risk_reward_ratio * 1.5,
                current_price + risk * self.config.risk_reward_ratio * 2.0,
            ]

        else:
            nearby_resistance = [r for r in resistance_levels if r > current_price]
            stop_loss = (
                min(nearby_resistance) * 1.005
                if nearby_resistance
                else current_price * 1.02
            )

            risk = stop_loss - current_price
            take_profit = [
                current_price - risk * self.config.risk_reward_ratio,
                current_price - risk * self.config.risk_reward_ratio * 1.5,
                current_price - risk * self.config.risk_reward_ratio * 2.0,
            ]

        return entry_price, stop_loss, take_profit

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_price: float,
        timestamp: int,
    ) -> Optional[TradeSignal]:
        """
        Generate a trading signal based on order flow analysis.

        Args:
            symbol: Trading symbol
            df: DataFrame with market data
            current_price: Current market price
            timestamp: Current timestamp

        Returns:
            TradeSignal if conditions are met, None otherwise
        """
        if df.empty or len(df) < 50:
            return None

        df_with_indicators = self.indicators.calculate_all_indicators(df)

        market_structure = self.analyze_market_structure(df_with_indicators)
        order_blocks = self.detect_order_blocks(df_with_indicators)

        confidence = 0.0
        reasoning_parts = []
        signal_side = PositionSide.NEUTRAL

        if "cvd" in df_with_indicators.columns:
            cvd = df_with_indicators["cvd"].iloc[-1]
            if cvd > 0:
                confidence += 0.2
                signal_side = PositionSide.LONG
                reasoning_parts.append("Positive CVD")
            elif cvd < 0:
                confidence += 0.2
                signal_side = PositionSide.SHORT
                reasoning_parts.append("Negative CVD")

        if "imbalance" in df_with_indicators.columns:
            imbalance = df_with_indicators["imbalance"].iloc[-1]
            if abs(imbalance) > 0.3:
                confidence += 0.2
                reasoning_parts.append(f"Strong imbalance ({imbalance:.2f})")

        if market_structure["trend"] != "neutral":
            confidence += 0.3
            reasoning_parts.append(f"{market_structure['trend'].capitalize()} trend")

        if len(order_blocks) > 0:
            confidence += 0.3
            reasoning_parts.append(f"{len(order_blocks)} order blocks detected")

        if confidence < self.config.min_confidence:
            return None

        support_levels = [level[1] for level in market_structure["swing_lows"]]
        resistance_levels = [level[1] for level in market_structure["swing_highs"]]

        entry_price, stop_loss, take_profit = self.calculate_entry_exit_levels(
            current_price, signal_side, support_levels, resistance_levels
        )

        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit[0] - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self.config.risk_reward_ratio:
            return None

        signal = TradeSignal(
            timestamp=timestamp,
            symbol=symbol,
            side=signal_side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts),
            risk_reward_ratio=risk_reward,
        )

        self.signal_history.append(signal)
        return signal

    def update_position(
        self, position: Position, current_price: float
    ) -> Dict[str, any]:
        """
        Update position status and check for exit conditions.

        Args:
            position: The position to update
            current_price: Current market price

        Returns:
            Dictionary with update status and actions
        """
        result = {"should_exit": False, "reason": "", "pnl": 0.0}

        if position.side == PositionSide.LONG:
            pnl = (current_price - position.entry_price) * position.quantity
            position.current_pnl = pnl

            if current_price <= position.stop_loss:
                result["should_exit"] = True
                result["reason"] = "Stop loss hit"
            elif current_price >= position.take_profit[0]:
                result["should_exit"] = True
                result["reason"] = "Take profit reached"

        else:
            pnl = (position.entry_price - current_price) * position.quantity
            position.current_pnl = pnl

            if current_price >= position.stop_loss:
                result["should_exit"] = True
                result["reason"] = "Stop loss hit"
            elif current_price <= position.take_profit[0]:
                result["should_exit"] = True
                result["reason"] = "Take profit reached"

        result["pnl"] = position.current_pnl
        return result

    def get_statistics(self) -> Dict[str, any]:
        """
        Get strategy performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        total_signals = len(self.signal_history)
        long_signals = sum(
            1 for s in self.signal_history if s.side == PositionSide.LONG
        )
        short_signals = sum(
            1 for s in self.signal_history if s.side == PositionSide.SHORT
        )

        avg_confidence = (
            sum(s.confidence for s in self.signal_history) / total_signals
            if total_signals > 0
            else 0.0
        )

        return {
            "total_signals": total_signals,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "avg_confidence": avg_confidence,
            "open_positions": len(self.open_positions),
        }
