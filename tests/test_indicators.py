"""
Tests for order flow indicators module.
"""

import pytest
import pandas as pd
import numpy as np

from botclave.engine.indicators import OrderFlowIndicators, VolumeProfile


class TestOrderFlowIndicators:
    """Test suite for OrderFlowIndicators class."""

    def test_initialization(self):
        """Test indicators initialization."""
        indicators = OrderFlowIndicators(
            value_area_percent=0.7,
            delta_period=20,
            cvd_period=50,
        )

        assert indicators.value_area_percent == 0.7
        assert indicators.delta_period == 20
        assert indicators.cvd_period == 50

    def test_calculate_cumulative_delta(self):
        """Test cumulative delta calculation."""
        indicators = OrderFlowIndicators()

        buy_volume = pd.Series([10, 15, 12, 18, 20])
        sell_volume = pd.Series([8, 10, 15, 12, 16])

        cvd = indicators.calculate_cumulative_delta(buy_volume, sell_volume)

        assert len(cvd) == 5
        assert cvd.iloc[-1] == sum(buy_volume - sell_volume)

    def test_calculate_volume_profile(self):
        """Test volume profile calculation."""
        indicators = OrderFlowIndicators()

        df = pd.DataFrame(
            {
                "close": [50000, 50100, 50050, 50200, 50150],
                "volume": [100, 150, 120, 180, 140],
            }
        )

        profile = indicators.calculate_volume_profile(df)

        assert isinstance(profile, VolumeProfile)
        assert len(profile.price_levels) > 0
        assert len(profile.volumes) > 0
        assert profile.poc > 0

    def test_calculate_delta_divergence(self):
        """Test delta divergence detection."""
        indicators = OrderFlowIndicators()

        price = pd.Series([50000 + i * 100 for i in range(20)])
        delta = pd.Series([10, 8, 6, 4, 2, 0, -2, -4, -6, -8] + [0] * 10)

        divergence = indicators.calculate_delta_divergence(price, delta, lookback=14)

        assert len(divergence) == len(price)
        assert divergence.iloc[-1] in [-1, 0, 1]

    def test_calculate_absorption(self):
        """Test absorption detection."""
        indicators = OrderFlowIndicators()

        price = pd.Series([50000.0] * 10)
        volume = pd.Series([100.0] * 5 + [500.0] + [100.0] * 4)
        delta = pd.Series([10.0] * 10)

        absorption = indicators.calculate_absorption(price, volume, delta)

        assert len(absorption) == len(price)
        assert any(abs(x) > 0 for x in absorption)

    def test_calculate_imbalance_score(self):
        """Test imbalance score calculation."""
        indicators = OrderFlowIndicators()

        buy_volume = pd.Series([100, 150, 120, 180, 160])
        sell_volume = pd.Series([80, 90, 100, 70, 90])

        imbalance = indicators.calculate_imbalance_score(buy_volume, sell_volume)

        assert len(imbalance) == 5
        assert all(-1 <= x <= 1 for x in imbalance)

    def test_generate_signal(self):
        """Test signal generation."""
        indicators = OrderFlowIndicators()

        timestamp = int(pd.Timestamp.now().timestamp() * 1000)
        price = 50000.0
        indicator_values = {
            "cumulative_delta": 0.8,
            "imbalance": 0.4,
            "absorption": 0.6,
        }

        signal = indicators.generate_signal(timestamp, price, indicator_values)

        assert signal.timestamp == timestamp
        assert signal.price == price
        assert signal.signal_type in ["buy", "sell", "neutral"]
        assert 0 <= signal.strength <= 1

    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        indicators = OrderFlowIndicators()

        df = pd.DataFrame(
            {
                "open": [50000, 50100, 50050, 50200, 50150],
                "high": [50100, 50200, 50150, 50300, 50250],
                "low": [49900, 50000, 49950, 50100, 50050],
                "close": [50050, 50150, 50100, 50250, 50200],
                "volume": [100, 150, 120, 180, 140],
                "buy_volume": [60, 90, 70, 110, 85],
                "sell_volume": [40, 60, 50, 70, 55],
            }
        )

        result_df = indicators.calculate_all_indicators(df)

        assert "cvd" in result_df.columns
        assert "imbalance" in result_df.columns
        assert "delta" in result_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
