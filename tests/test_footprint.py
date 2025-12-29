"""
Tests for footprint chart module.
"""

import pytest
import pandas as pd
from datetime import datetime

from botclave.engine.footprint import FootprintChart, FootprintBar


class TestFootprintChart:
    """Test suite for FootprintChart class."""

    def test_initialization(self):
        """Test footprint chart initialization."""
        chart = FootprintChart(
            tick_size=0.01,
            imbalance_ratio=1.5,
            value_area_percent=0.7,
        )

        assert chart.tick_size == 0.01
        assert chart.imbalance_ratio == 1.5
        assert chart.value_area_percent == 0.7

    def test_calculate_delta(self):
        """Test delta calculation."""
        chart = FootprintChart()

        bar = FootprintBar(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            buy_volume=60.0,
            sell_volume=40.0,
            delta=20.0,
        )

        delta = chart.calculate_delta(bar)
        assert delta == 20.0

    def test_find_point_of_control(self):
        """Test POC finding."""
        chart = FootprintChart()

        bar = FootprintBar(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            price_levels={
                50000.0: {"buy": 30.0, "sell": 20.0},
                50050.0: {"buy": 40.0, "sell": 35.0},
                50100.0: {"buy": 25.0, "sell": 15.0},
            },
        )

        poc = chart.find_point_of_control(bar)
        assert poc == 50050.0

    def test_calculate_value_area(self):
        """Test value area calculation."""
        chart = FootprintChart()

        bar = FootprintBar(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            price_levels={
                49900.0: {"buy": 10.0, "sell": 5.0},
                50000.0: {"buy": 30.0, "sell": 20.0},
                50050.0: {"buy": 40.0, "sell": 35.0},
                50100.0: {"buy": 25.0, "sell": 15.0},
            },
        )

        vah, val = chart.calculate_value_area(bar)
        assert vah >= val
        assert val >= bar.low
        assert vah <= bar.high

    def test_detect_imbalances(self):
        """Test imbalance detection."""
        chart = FootprintChart(imbalance_ratio=2.0)

        bar = FootprintBar(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            price_levels={
                50000.0: {"buy": 50.0, "sell": 10.0},
                50050.0: {"buy": 20.0, "sell": 20.0},
            },
        )

        imbalances = chart.detect_imbalances(bar)
        assert 50000.0 in imbalances

    def test_calculate_cumulative_delta(self):
        """Test cumulative delta calculation."""
        chart = FootprintChart()

        bars = [
            FootprintBar(
                timestamp=i,
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=100.0,
                buy_volume=60.0,
                sell_volume=40.0,
                delta=20.0,
            )
            for i in range(5)
        ]

        chart.bars = bars
        cumulative_delta = chart.calculate_cumulative_delta()
        assert len(cumulative_delta) == 5
        assert cumulative_delta.iloc[-1] == 100.0

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        chart = FootprintChart()

        bars = [
            FootprintBar(
                timestamp=i,
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=100.0,
                buy_volume=60.0,
                sell_volume=40.0,
                delta=20.0,
            )
            for i in range(3)
        ]

        df = chart.to_dataframe(bars)
        assert not df.empty
        assert len(df) == 3
        assert "delta" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
