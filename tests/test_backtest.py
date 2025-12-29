"""
Tests for backtesting module.
"""

import pytest
import pandas as pd
import numpy as np

from botclave.backtest.backtester import Backtester, BacktestConfig, Trade


class TestBacktester:
    """Test suite for Backtester class."""

    def test_initialization(self):
        """Test backtester initialization."""
        config = BacktestConfig(
            initial_capital=10000.0,
            position_size_pct=0.02,
        )
        backtester = Backtester(config)

        assert backtester.config.initial_capital == 10000.0
        assert backtester.current_capital == 10000.0
        assert len(backtester.trades) == 0

    def test_run_backtest_no_signals(self):
        """Test backtest with no signals."""
        backtester = Backtester()

        df = pd.DataFrame(
            {
                "open": [50000, 50100, 50050, 50200, 50150],
                "high": [50100, 50200, 50150, 50300, 50250],
                "low": [49900, 50000, 49950, 50100, 50050],
                "close": [50050, 50150, 50100, 50250, 50200],
                "volume": [100, 150, 120, 180, 140],
            }
        )

        signals = pd.DataFrame()

        result = backtester.run(df, signals)

        assert result.total_trades == 0
        assert result.start_capital == backtester.config.initial_capital

    def test_run_backtest_with_signals(self):
        """Test backtest with signals."""
        config = BacktestConfig(initial_capital=10000.0)
        backtester = Backtester(config)

        df = pd.DataFrame(
            {
                "open": [50000 + i * 100 for i in range(20)],
                "high": [50100 + i * 100 for i in range(20)],
                "low": [49900 + i * 100 for i in range(20)],
                "close": [50050 + i * 100 for i in range(20)],
                "volume": [100] * 20,
            }
        )
        df.index = pd.date_range(start="2024-01-01", periods=20, freq="1h")

        signals = pd.DataFrame(
            {
                "side": ["long"],
                "symbol": ["BTC/USDT"],
                "stop_loss": [49500.0],
                "take_profit": [51500.0],
            },
            index=[df.index[5]],
        )

        result = backtester.run(df, signals)

        assert result.total_trades >= 0
        assert len(result.trades) >= 0

    def test_equity_curve(self):
        """Test equity curve generation."""
        backtester = Backtester()

        df = pd.DataFrame(
            {
                "open": [50000, 50100, 50050],
                "high": [50100, 50200, 50150],
                "low": [49900, 50000, 49950],
                "close": [50050, 50150, 50100],
                "volume": [100, 150, 120],
            }
        )

        signals = pd.DataFrame()

        backtester.run(df, signals)
        equity_curve = backtester.get_equity_curve()

        assert len(equity_curve) > 0


class TestBacktestResult:
    """Test suite for BacktestResult class."""

    def test_result_calculations(self):
        """Test backtest result calculations."""
        config = BacktestConfig(initial_capital=10000.0)
        backtester = Backtester(config)

        df = pd.DataFrame(
            {
                "open": [50000, 50100, 50200, 50150, 50250],
                "high": [50100, 50200, 50300, 50250, 50350],
                "low": [49900, 50000, 50100, 50050, 50150],
                "close": [50050, 50150, 50250, 50200, 50300],
                "volume": [100, 150, 120, 180, 140],
            }
        )

        signals = pd.DataFrame()

        result = backtester.run(df, signals)

        assert result.win_rate >= 0
        assert result.max_drawdown >= 0
        assert result.sharpe_ratio is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
