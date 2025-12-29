"""
Tests for order flow strategy module.
"""

import pytest
import pandas as pd
import numpy as np

from botclave.engine.strategy import (
    OrderFlowStrategy,
    StrategyConfig,
    PositionSide,
    Position,
)


class TestOrderFlowStrategy:
    """Test suite for OrderFlowStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = StrategyConfig(
            min_confidence=0.6,
            risk_reward_ratio=2.0,
        )
        strategy = OrderFlowStrategy(config)

        assert strategy.config.min_confidence == 0.6
        assert strategy.config.risk_reward_ratio == 2.0
        assert len(strategy.open_positions) == 0

    def test_analyze_market_structure(self):
        """Test market structure analysis."""
        strategy = OrderFlowStrategy()

        df = pd.DataFrame(
            {
                "open": np.random.randn(100) * 100 + 50000,
                "high": np.random.randn(100) * 100 + 50100,
                "low": np.random.randn(100) * 100 + 49900,
                "close": np.random.randn(100) * 100 + 50000,
                "volume": np.random.randn(100) * 50 + 100,
            }
        )

        structure = strategy.analyze_market_structure(df)

        assert "trend" in structure
        assert "strength" in structure
        assert "swing_highs" in structure
        assert "swing_lows" in structure
        assert structure["trend"] in ["bullish", "bearish", "neutral"]

    def test_detect_order_blocks(self):
        """Test order block detection."""
        strategy = OrderFlowStrategy()

        df = pd.DataFrame(
            {
                "open": [50000, 50100, 50050, 50200, 50150] * 4,
                "high": [50100, 50200, 50150, 50300, 50250] * 4,
                "low": [49900, 50000, 49950, 50100, 50050] * 4,
                "close": [50050, 50150, 50100, 50250, 50200] * 4,
                "volume": [100, 150, 120, 180, 140] * 4,
            }
        )

        order_blocks = strategy.detect_order_blocks(df)

        assert isinstance(order_blocks, list)

    def test_calculate_entry_exit_levels(self):
        """Test entry/exit level calculation."""
        strategy = OrderFlowStrategy()

        current_price = 50000.0
        support_levels = [49500.0, 49800.0]
        resistance_levels = [50500.0, 50800.0]

        entry, stop_loss, take_profit = strategy.calculate_entry_exit_levels(
            current_price,
            PositionSide.LONG,
            support_levels,
            resistance_levels,
        )

        assert entry > 0
        assert stop_loss < entry
        assert all(tp > entry for tp in take_profit)

    def test_generate_signal(self):
        """Test signal generation."""
        strategy = OrderFlowStrategy()

        df = pd.DataFrame(
            {
                "open": [50000 + i * 10 for i in range(100)],
                "high": [50100 + i * 10 for i in range(100)],
                "low": [49900 + i * 10 for i in range(100)],
                "close": [50050 + i * 10 for i in range(100)],
                "volume": [100 + i for i in range(100)],
                "buy_volume": [60 + i * 0.5 for i in range(100)],
                "sell_volume": [40 + i * 0.5 for i in range(100)],
            }
        )

        timestamp = int(pd.Timestamp.now().timestamp() * 1000)
        current_price = df["close"].iloc[-1]

        signal = strategy.generate_signal("BTC/USDT", df, current_price, timestamp)

        if signal:
            assert signal.symbol == "BTC/USDT"
            assert signal.side in [PositionSide.LONG, PositionSide.SHORT]
            assert signal.confidence >= strategy.config.min_confidence

    def test_update_position(self):
        """Test position update."""
        strategy = OrderFlowStrategy()

        position = Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=50000.0,
            quantity=1.0,
            entry_time=int(pd.Timestamp.now().timestamp() * 1000),
            stop_loss=49500.0,
            take_profit=[51000.0, 51500.0, 52000.0],
        )

        result = strategy.update_position(position, 50500.0)

        assert "should_exit" in result
        assert "reason" in result
        assert "pnl" in result

    def test_get_statistics(self):
        """Test statistics retrieval."""
        strategy = OrderFlowStrategy()

        stats = strategy.get_statistics()

        assert "total_signals" in stats
        assert "long_signals" in stats
        assert "short_signals" in stats
        assert "avg_confidence" in stats
        assert stats["total_signals"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
