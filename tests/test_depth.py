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
    DeOrder,
    Depth,
    LocalDepthCache,
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


class TestDeOrder:
    """Test suite for DeOrder dataclass."""

    def test_deorder_creation(self):
        """Test DeOrder creation."""
        order = DeOrder(price=50000.0, qty=10.0)
        assert order.price == 50000.0
        assert order.qty == 10.0


class TestDepth:
    """Test suite for Depth class (foundational order book)."""

    def test_depth_initialization(self):
        """Test Depth initialization."""
        depth = Depth()
        assert len(depth.bids) == 0
        assert len(depth.asks) == 0

    def test_depth_update(self):
        """Test updating bids/asks."""
        depth = Depth()

        # Add initial levels
        depth.update(
            [(100.0, 10.0), (99.5, 5.0)],
            [(101.0, 8.0), (102.0, 6.0)]
        )

        assert len(depth.bids) == 2
        assert len(depth.asks) == 2
        assert depth.bids[100.0] == 10.0
        assert depth.asks[101.0] == 8.0

        # Update existing level
        depth.update([(100.0, 15.0)], [])
        assert depth.bids[100.0] == 15.0

        # Remove level with qty=0
        depth.update([(99.5, 0.0)], [])
        assert 99.5 not in depth.bids
        assert len(depth.bids) == 1

    def test_depth_snapshot(self):
        """Test snapshot replacement."""
        depth = Depth()

        # Add initial data
        depth.update([(100.0, 10.0)], [(101.0, 8.0)])

        # Replace with snapshot
        depth.snapshot([(105.0, 5.0)], [(106.0, 4.0)])

        assert len(depth.bids) == 1
        assert len(depth.asks) == 1
        assert 100.0 not in depth.bids
        assert 101.0 not in depth.asks
        assert depth.bids[105.0] == 5.0
        assert depth.asks[106.0] == 4.0

    def test_mid_price(self):
        """Test mid price calculation."""
        depth = Depth()

        # Empty book
        assert depth.mid_price() is None

        # Add levels
        depth.update([(100.0, 10.0)], [(102.0, 8.0)])
        mid = depth.mid_price()
        assert mid == (100.0 + 102.0) / 2.0
        assert mid == 101.0

    def test_best_bid_ask(self):
        """Test best bid and best ask."""
        depth = Depth()

        # Empty book
        assert depth.best_bid() is None
        assert depth.best_ask() is None

        # Add levels
        depth.update(
            [(100.0, 10.0), (99.5, 5.0), (99.0, 3.0)],
            [(101.0, 8.0), (101.5, 6.0), (102.0, 4.0)]
        )

        best_bid = depth.best_bid()
        best_ask = depth.best_ask()

        assert best_bid is not None
        assert best_ask is not None
        assert best_bid[0] == 100.0  # Highest bid
        assert best_bid[1] == 10.0
        assert best_ask[0] == 101.0  # Lowest ask
        assert best_ask[1] == 8.0

    def test_get_level(self):
        """Test getting specific price level."""
        depth = Depth()

        depth.update([(100.0, 10.0)], [(101.0, 8.0)])

        # Get level that exists
        level = depth.get_level(100.0)
        assert level is not None
        assert level[0] == 10.0  # bid_qty
        assert level[1] == 0.0   # ask_qty

        # Get non-existent level
        level = depth.get_level(99.0)
        assert level is None

        # Get level with both sides
        level = depth.get_level(100.0, side='both')
        assert level is not None

        # Get level with wrong side
        level = depth.get_level(100.0, side='ask')
        assert level is None


class TestLocalDepthCache:
    """Test suite for LocalDepthCache class."""

    def test_local_depth_cache_initialization(self):
        """Test LocalDepthCache initialization."""
        cache = LocalDepthCache()
        assert cache.last_update_id == 0
        assert cache.time_ms == 0
        assert isinstance(cache.depth, Depth)

    def test_update_snapshot(self):
        """Test snapshot update."""
        cache = LocalDepthCache()

        cache.update_snapshot(
            [(100.0, 10.0), (99.5, 5.0)],
            [(101.0, 8.0)],
            update_id=1,
            time_ms=1000
        )

        assert cache.last_update_id == 1
        assert cache.time_ms == 1000
        assert len(cache.depth.bids) == 2
        assert len(cache.depth.asks) == 1

    def test_update_diff(self):
        """Test incremental diff update."""
        cache = LocalDepthCache()

        # Initial snapshot
        cache.update_snapshot(
            [(100.0, 10.0)],
            [(101.0, 8.0)],
            update_id=1,
            time_ms=1000
        )

        # Apply diff
        cache.update_diff(
            [(100.5, 5.0), (100.0, 0.0)],  # Add new level, remove old
            [(101.5, 3.0)],
            update_id=2,
            time_ms=1001
        )

        assert cache.last_update_id == 2
        assert cache.time_ms == 1001
        assert 100.0 not in cache.depth.bids  # Removed
        assert 100.5 in cache.depth.bids     # Added
        assert 101.5 in cache.depth.asks     # Added

    def test_get_depth(self):
        """Test getting depth object."""
        cache = LocalDepthCache()

        cache.update_snapshot(
            [(100.0, 10.0)],
            [(101.0, 8.0)],
            update_id=1,
            time_ms=1000
        )

        depth = cache.get_depth()
        assert isinstance(depth, Depth)
        assert depth.best_bid()[0] == 100.0
        assert depth.best_ask()[0] == 101.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
