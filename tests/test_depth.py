"""
Tests for depth analysis module.
"""

import pytest
from datetime import datetime

from botclave.engine.depth import (
    DepthAnalyzer,
    DepthLevel,
    DepthSnapshot,
    AbsorptionZone,
)


class TestDepthAnalyzer:
    """Test suite for DepthAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = DepthAnalyzer(
            absorption_threshold=2.0,
            imbalance_threshold=1.5,
            min_volume=1.0,
        )

        assert analyzer.absorption_threshold == 2.0
        assert analyzer.imbalance_threshold == 1.5
        assert analyzer.min_volume == 1.0
        assert len(analyzer.depth_history) == 0

    def test_add_snapshot(self):
        """Test adding depth snapshots."""
        analyzer = DepthAnalyzer()

        snapshot = DepthSnapshot(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTC/USDT",
            bids=[
                DepthLevel(price=50000.0, bid_volume=1.5, timestamp=0),
                DepthLevel(price=49990.0, bid_volume=2.0, timestamp=0),
            ],
            asks=[
                DepthLevel(price=50010.0, ask_volume=1.8, timestamp=0),
                DepthLevel(price=50020.0, ask_volume=2.2, timestamp=0),
            ],
        )

        analyzer.add_snapshot(snapshot)
        assert len(analyzer.depth_history) == 1

    def test_calculate_imbalance(self):
        """Test imbalance calculation."""
        analyzer = DepthAnalyzer()

        bids = [
            DepthLevel(price=50000.0, bid_volume=10.0, timestamp=0),
            DepthLevel(price=49990.0, bid_volume=8.0, timestamp=0),
        ]

        asks = [
            DepthLevel(price=50010.0, ask_volume=5.0, timestamp=0),
            DepthLevel(price=50020.0, ask_volume=3.0, timestamp=0),
        ]

        imbalance = analyzer.calculate_imbalance(bids, asks)
        assert imbalance > 1.0

    def test_detect_absorption_zones(self):
        """Test absorption zone detection."""
        analyzer = DepthAnalyzer()

        for i in range(15):
            snapshot = DepthSnapshot(
                timestamp=int(datetime.now().timestamp() * 1000) + i,
                symbol="BTC/USDT",
                bids=[DepthLevel(price=50000.0, bid_volume=1.0, timestamp=0)],
                asks=[DepthLevel(price=50010.0, ask_volume=1.0, timestamp=0)],
            )
            analyzer.add_snapshot(snapshot)

        zones = analyzer.detect_absorption_zones(lookback_periods=10)
        assert isinstance(zones, list)

    def test_get_liquidity_heatmap(self):
        """Test liquidity heatmap generation."""
        analyzer = DepthAnalyzer()

        snapshot = DepthSnapshot(
            timestamp=int(datetime.now().timestamp() * 1000),
            symbol="BTC/USDT",
            bids=[DepthLevel(price=50000.0, bid_volume=1.5, timestamp=0)],
            asks=[DepthLevel(price=50010.0, ask_volume=1.8, timestamp=0)],
        )
        analyzer.add_snapshot(snapshot)

        bid_heatmap, ask_heatmap = analyzer.get_liquidity_heatmap()
        assert not bid_heatmap.empty
        assert not ask_heatmap.empty

    def test_get_depth_delta(self):
        """Test depth delta calculation."""
        analyzer = DepthAnalyzer()

        snapshot1 = DepthSnapshot(
            timestamp=1000,
            symbol="BTC/USDT",
            bids=[DepthLevel(price=50000.0, bid_volume=10.0, timestamp=0)],
            asks=[DepthLevel(price=50010.0, ask_volume=8.0, timestamp=0)],
        )

        snapshot2 = DepthSnapshot(
            timestamp=2000,
            symbol="BTC/USDT",
            bids=[DepthLevel(price=50000.0, bid_volume=12.0, timestamp=0)],
            asks=[DepthLevel(price=50010.0, ask_volume=7.0, timestamp=0)],
        )

        delta = analyzer.get_depth_delta(snapshot1, snapshot2)
        assert "bid_delta" in delta
        assert "ask_delta" in delta
        assert delta["bid_delta"] == 2.0
        assert delta["ask_delta"] == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
