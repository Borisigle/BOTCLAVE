"""
Tests for footprint chart module.
"""

import pytest
import pandas as pd
from datetime import datetime

from botclave.engine.footprint import (
    FootprintChart,
    FootprintBar,
    NPoc,
    Trade,
    GroupedTrades,
    PointOfControl,
    KlineFootprint,
    KlineDataPoint,
)


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


class TestNPoc:
    """Test suite for NPoc enum."""

    def test_npoc_values(self):
        """Test NPoc enum values."""
        assert NPoc.UNCHANGED.value == "unchanged"
        assert NPoc.HIGHER.value == "higher"
        assert NPoc.LOWER.value == "lower"


class TestTrade:
    """Test suite for Trade dataclass."""

    def test_trade_creation(self):
        """Test Trade creation."""
        trade = Trade(
            price=50000.0,
            qty=10.0,
            is_buy=True,
            time_ms=1000
        )
        assert trade.price == 50000.0
        assert trade.qty == 10.0
        assert trade.is_buy is True
        assert trade.time_ms == 1000


class TestGroupedTrades:
    """Test suite for GroupedTrades dataclass."""

    def test_grouped_trades_delta(self):
        """Test delta calculation."""
        gt = GroupedTrades(buy_qty=1000.0, sell_qty=600.0)
        assert gt.delta == 400.0  # buy - sell
        assert gt.total_qty == 1600.0

    def test_delta_percent(self):
        """Test delta percent calculation."""
        gt = GroupedTrades(buy_qty=1000.0, sell_qty=600.0)
        # delta = 400, total = 1600, delta_percent = 400/1600 = 0.25
        assert gt.delta_percent == 0.25

    def test_empty_grouped_trades(self):
        """Test empty GroupedTrades."""
        gt = GroupedTrades()
        assert gt.buy_qty == 0.0
        assert gt.sell_qty == 0.0
        assert gt.total_qty == 0.0
        assert gt.delta == 0.0
        assert gt.delta_percent == 0.0

    def test_accumulating_trades(self):
        """Test accumulating trades in GroupedTrades."""
        gt = GroupedTrades()

        # Add buys
        gt.buy_qty = 500.0
        gt.buy_count = 5
        # Add sells
        gt.sell_qty = 300.0
        gt.sell_count = 3

        assert gt.buy_qty == 500.0
        assert gt.sell_qty == 300.0
        assert gt.delta == 200.0
        assert gt.buy_count == 5
        assert gt.sell_count == 3


class TestPointOfControl:
    """Test suite for PointOfControl dataclass."""

    def test_poc_creation(self):
        """Test PointOfControl creation."""
        poc = PointOfControl(price=50000.0, qty=1000.0)
        assert poc.price == 50000.0
        assert poc.qty == 1000.0
        assert poc.status == NPoc.UNCHANGED

    def test_poc_with_status(self):
        """Test PointOfControl with status."""
        poc = PointOfControl(
            price=50100.0,
            qty=1500.0,
            status=NPoc.HIGHER
        )
        assert poc.price == 50100.0
        assert poc.status == NPoc.HIGHER


class TestKlineFootprint:
    """Test suite for KlineFootprint class."""

    def test_footprint_initialization(self):
        """Test KlineFootprint initialization."""
        fp = KlineFootprint(price_step=1.0)
        assert fp.price_step == 1.0
        assert len(fp.trades) == 0
        assert fp.poc is None

    def test_footprint_add_trade(self):
        """Test adding trades with price rounding."""
        fp = KlineFootprint(price_step=1.0)

        # Add buy trade
        fp.add_trade(Trade(price=100.5, qty=10.0, is_buy=True, time_ms=1000))
        # Add sell trade at similar price
        fp.add_trade(Trade(price=100.4, qty=15.0, is_buy=False, time_ms=1001))

        # Both should round to 100 or 101
        assert len(fp.trades) <= 2

        # Check trades were aggregated
        total_qty = sum(gt.total_qty for gt in fp.trades.values())
        assert total_qty == 25.0

    def test_footprint_add_trade_batch(self):
        """Test adding multiple trades at once."""
        fp = KlineFootprint()

        trades = [
            Trade(price=100.0, qty=10.0, is_buy=True, time_ms=1000),
            Trade(price=101.0, qty=5.0, is_buy=False, time_ms=1001),
            Trade(price=100.0, qty=8.0, is_buy=True, time_ms=1002),
        ]

        fp.add_trade_batch(trades)

        # Should have 2 price levels (100 and 101)
        assert len(fp.trades) == 2

        # Price 100 should have aggregated buys
        assert 100.0 in fp.trades
        assert fp.trades[100.0].buy_qty == 18.0

    def test_footprint_poc(self):
        """Test POC calculation."""
        fp = KlineFootprint()

        # Add many trades at price 100
        for _ in range(10):
            fp.add_trade(Trade(price=100.0, qty=10.0, is_buy=True, time_ms=1000))

        # Add few trades at price 101
        fp.add_trade(Trade(price=101.0, qty=5.0, is_buy=False, time_ms=1001))

        poc = fp.calculate_poc()
        assert poc == 100.0  # Price 100 has more volume

    def test_calculate_delta(self):
        """Test delta calculation."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=100.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=100.0, qty=60.0, is_buy=False, time_ms=1001))

        delta_profile = fp.calculate_delta()
        assert 100.0 in delta_profile
        assert delta_profile[100.0] == 40.0  # 100 - 60

    def test_calculate_delta_profile(self):
        """Test delta percent profile."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=100.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=100.0, qty=60.0, is_buy=False, time_ms=1001))

        delta_profile = fp.calculate_delta_profile()
        assert 100.0 in delta_profile
        # delta = 40, total = 160, delta_percent = 40/160 = 0.25
        assert delta_profile[100.0] == 0.25

    def test_get_volume_profile(self):
        """Test volume profile."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=50.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=100.0, qty=30.0, is_buy=False, time_ms=1001))
        fp.add_trade(Trade(price=101.0, qty=20.0, is_buy=True, time_ms=1002))

        volume_profile = fp.get_volume_profile()
        assert volume_profile[100.0] == 80.0
        assert volume_profile[101.0] == 20.0

    def test_get_imbalance(self):
        """Test imbalance detection."""
        fp = KlineFootprint()

        # Strong buy imbalance at price 100
        fp.add_trade(Trade(price=100.0, qty=100.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=100.0, qty=30.0, is_buy=False, time_ms=1001))

        # delta = 70, total = 130, delta_percent = 70/130 ≈ 0.538
        # With threshold=0.65, this should NOT trigger
        imbalance = fp.get_imbalance(price=100.0, threshold=0.65)
        assert imbalance is None

        # With threshold=0.5, this SHOULD trigger
        imbalance = fp.get_imbalance(price=100.0, threshold=0.5)
        assert imbalance == 'buy'

    def test_get_highest_lowest_prices(self):
        """Test getting price range."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=10.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=105.0, qty=10.0, is_buy=False, time_ms=1001))
        fp.add_trade(Trade(price=102.5, qty=10.0, is_buy=True, time_ms=1002))

        high, low = fp.get_highest_lowest_prices()
        assert high == 105.0
        assert low == 100.0

    def test_footprint_clear(self):
        """Test clearing footprint."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=10.0, is_buy=True, time_ms=1000))
        assert len(fp.trades) > 0

        fp.clear()
        assert len(fp.trades) == 0
        assert fp.poc is None

    def test_get_stats(self):
        """Test getting footprint statistics."""
        fp = KlineFootprint()

        fp.add_trade(Trade(price=100.0, qty=60.0, is_buy=True, time_ms=1000))
        fp.add_trade(Trade(price=100.0, qty=40.0, is_buy=False, time_ms=1001))
        fp.add_trade(Trade(price=101.0, qty=30.0, is_buy=True, time_ms=1002))
        fp.add_trade(Trade(price=101.0, qty=20.0, is_buy=False, time_ms=1003))

        stats = fp.get_stats()
        assert stats['total_volume'] == 150.0
        assert stats['total_buy'] == 90.0
        assert stats['total_sell'] == 60.0
        assert stats['delta'] == 30.0
        assert stats['price_levels'] == 2


class TestKlineDataPoint:
    """Test suite for KlineDataPoint class."""

    def test_kline_datapoint_creation(self):
        """Test KlineDataPoint creation."""
        kdp = KlineDataPoint(
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
            time_ms=1000
        )

        assert kdp.open == 100.0
        assert kdp.high == 105.0
        assert kdp.low == 99.0
        assert kdp.close == 104.0
        assert kdp.volume == 1000.0
        assert kdp.time_ms == 1000
        assert isinstance(kdp.footprint, KlineFootprint)

    def test_kline_datapoint_add_trade(self):
        """Test adding trade to KlineDataPoint."""
        kdp = KlineDataPoint(
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
            time_ms=1000
        )

        kdp.add_trade(Trade(price=102.0, qty=100.0, is_buy=True, time_ms=1000))

        stats = kdp.get_footprint_stats()
        assert stats['total_volume'] == 100.0

    def test_kline_datapoint_with_price_step(self):
        """Test KlineDataPoint with custom price_step."""
        kdp = KlineDataPoint(
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
            time_ms=1000,
            price_step=0.5
        )

        assert kdp.footprint.price_step == 0.5

        kdp.add_trade(Trade(price=102.3, qty=10.0, is_buy=True, time_ms=1000))

        # Should round to 102.5
        assert 102.5 in kdp.footprint.trades

    def test_get_footprint_stats(self):
        """Test getting footprint stats from KlineDataPoint."""
        kdp = KlineDataPoint(
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
            time_ms=1000
        )

        kdp.add_trade(Trade(price=102.0, qty=60.0, is_buy=True, time_ms=1000))
        kdp.add_trade(Trade(price=102.0, qty=40.0, is_buy=False, time_ms=1001))

        stats = kdp.get_footprint_stats()
        assert stats['total_volume'] == 100.0
        assert stats['delta'] == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
