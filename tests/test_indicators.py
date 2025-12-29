"""
Tests for order flow indicators module.
"""

import pytest
import pandas as pd
import numpy as np

from botclave.engine.indicators import (
    OrderFlowIndicators,
    VolumeProfile,
    SwingDetector,
    BreakOfStructureDetector,
    FairValueGapDetector,
    ChangeOfCharacterDetector,
    OrderBlockDetector,
    LiquidityDetector,
    SMCIndicator,
    calculate_retracement_levels,
    get_previous_highs_lows,
)


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


# ============================================================================
# SMC INDICATORS TESTS
# ============================================================================


@pytest.fixture
def sample_df():
    """Create a test DataFrame with realistic data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(100) * 0.2,
            "high": close + abs(np.random.randn(100) * 0.5),
            "low": close - abs(np.random.randn(100) * 0.5),
            "close": close,
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )


class TestSwingDetector:
    """Test suite for SwingDetector class."""

    def test_initialization(self):
        """Test swing detector initialization."""
        detector = SwingDetector(left_bars=3, right_bars=3)
        assert detector.left_bars == 3
        assert detector.right_bars == 3

    def test_swing_detection(self, sample_df):
        """Test: detects swings correctly."""
        detector = SwingDetector(left_bars=2, right_bars=2)
        swings = detector.find_swings(sample_df)

        # Should have some swings
        assert len(swings) > 0

        # Each swing should have valid properties
        for swing in swings:
            assert swing.index >= 0
            assert swing.price > 0
            assert swing.swing_type in ["high", "low"]

    def test_swing_alternation(self, sample_df):
        """Test: swings can be alternating or same type (depending on data)."""
        detector = SwingDetector(left_bars=2, right_bars=2)
        swings = detector.find_swings(sample_df)

        # Should have some swings
        assert len(swings) > 0

        # All swings should have proper types
        for swing in swings:
            assert swing.swing_type in ["high", "low"]

    def test_get_last_swing(self, sample_df):
        """Test: get last swing of specific type."""
        detector = SwingDetector()
        last_high = detector.get_last_swing(sample_df, "high")
        last_low = detector.get_last_swing(sample_df, "low")

        if last_high:
            assert last_high.swing_type == "high"
        if last_low:
            assert last_low.swing_type == "low"

    def test_get_last_n_swings(self, sample_df):
        """Test: get last N swings."""
        detector = SwingDetector()
        swings = detector.get_last_n_swings(sample_df, n=5)

        assert len(swings) <= 5


class TestBreakOfStructureDetector:
    """Test suite for BreakOfStructureDetector class."""

    def test_bos_detection(self, sample_df):
        """Test: detects BOS when there's a structural change."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        bos_detector = BreakOfStructureDetector()
        bos = bos_detector.find_bos(sample_df, swings)

        # Should return a list (may be empty)
        assert isinstance(bos, list)

        # If BOS exist, they should have proper structure
        for b in bos:
            assert b.direction in ["bullish", "bearish"]
            assert b.price > 0
            assert b.broken_level > 0

    def test_get_last_bos(self, sample_df):
        """Test: get last BOS."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        bos_detector = BreakOfStructureDetector()
        last_bos = bos_detector.get_last_bos(sample_df, swings)

        # May be None if no BOS
        if last_bos:
            assert last_bos.direction in ["bullish", "bearish"]

    def test_get_last_bos_filtered(self, sample_df):
        """Test: get last BOS filtered by direction."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        bos_detector = BreakOfStructureDetector()
        bullish_bos = bos_detector.get_last_bos(sample_df, swings, direction="bullish")
        bearish_bos = bos_detector.get_last_bos(sample_df, swings, direction="bearish")

        if bullish_bos:
            assert bullish_bos.direction == "bullish"
        if bearish_bos:
            assert bearish_bos.direction == "bearish"


class TestFairValueGapDetector:
    """Test suite for FairValueGapDetector class."""

    def test_ffg_detection(self, sample_df):
        """Test: detects Fair Value Gaps."""
        detector = FairValueGapDetector()
        ffg = detector.find_ffg(sample_df)

        # Should return a list (may be empty)
        assert isinstance(ffg, list)

        # If FFG exist, validate structure
        for gap in ffg:
            assert gap.direction in ["bullish", "bearish"]
            assert gap.top_price > gap.bottom_price
            assert gap.size_percent >= 0

    def test_get_active_ffg(self, sample_df):
        """Test: get active (unfilled) FFGs."""
        detector = FairValueGapDetector()
        active_ffg = detector.get_active_ffg(sample_df)

        assert isinstance(active_ffg, list)

        # All active FFGs should be unfilled
        for ffg in active_ffg:
            filled_index = detector.fill_ffg(sample_df, ffg)
            assert filled_index is None

    def test_fill_ffg(self, sample_df):
        """Test: FFG filling detection."""
        detector = FairValueGapDetector(min_size_percent=0.0)  # Allow small gaps
        all_ffg = detector.find_ffg(sample_df)

        # Test fill detection
        for ffg in all_ffg:
            filled_index = detector.fill_ffg(sample_df, ffg)
            # Should be either None or valid index
            if filled_index is not None:
                assert filled_index > ffg.index
                assert filled_index < len(sample_df)


class TestChangeOfCharacterDetector:
    """Test suite for ChangeOfCharacterDetector class."""

    def test_choch_detection(self, sample_df):
        """Test: detects Change of Character."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        choch_detector = ChangeOfCharacterDetector()
        choch = choch_detector.find_choch(sample_df, swings)

        # Should return a list
        assert isinstance(choch, list)

        # Validate structure
        for ch in choch:
            assert ch.previous_trend in ["uptrend", "downtrend"]
            assert ch.new_trend in ["uptrend", "downtrend"]
            assert ch.previous_trend != ch.new_trend


class TestOrderBlockDetector:
    """Test suite for OrderBlockDetector class."""

    def test_order_block_detection(self, sample_df):
        """Test: detects Order Blocks."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        bos_detector = BreakOfStructureDetector()
        bos = bos_detector.find_bos(sample_df, swings)

        ob_detector = OrderBlockDetector()
        order_blocks = ob_detector.find_order_blocks(sample_df, bos)

        # Should return a list
        assert isinstance(order_blocks, list)

        # Validate structure
        for ob in order_blocks:
            assert ob.direction in ["bullish", "bearish"]
            assert ob.high_price >= ob.low_price
            assert ob.volume > 0

    def test_mitigated_blocks(self, sample_df):
        """Test: detects mitigated order blocks."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        bos_detector = BreakOfStructureDetector()
        bos = bos_detector.find_bos(sample_df, swings)

        ob_detector = OrderBlockDetector()
        order_blocks = ob_detector.find_order_blocks(sample_df, bos)
        mitigated = ob_detector.find_mitigated_blocks(sample_df, order_blocks)

        # Should return same list (modified in place)
        assert len(mitigated) == len(order_blocks)


class TestLiquidityDetector:
    """Test suite for LiquidityDetector class."""

    def test_liquidity_clusters(self, sample_df):
        """Test: detects liquidity clusters."""
        swing_detector = SwingDetector()
        swings = swing_detector.find_swings(sample_df)

        liquidity_detector = LiquidityDetector()
        clusters = liquidity_detector.find_liquidity_clusters(sample_df, swings)

        # Should return a list
        assert isinstance(clusters, list)

        # Validate structure
        for cluster in clusters:
            assert cluster.direction in ["support", "resistance"]
            assert cluster.strength >= 2  # Significant clusters only
            assert cluster.price > 0


class TestSMCIndicator:
    """Test suite for SMCIndicator master class."""

    def test_initialization(self):
        """Test SMC indicator initialization."""
        smc = SMCIndicator(left_bars=3, right_bars=3, min_ffg_percent=0.03)
        assert smc.swing_detector.left_bars == 3
        assert smc.swing_detector.right_bars == 3
        assert smc.ffg_detector.min_size_percent == 0.03

    def test_smc_master_analysis(self, sample_df):
        """Test: SMCIndicator.analyze() returns complete dict."""
        smc = SMCIndicator()
        result = smc.analyze(sample_df)

        # Should have all required keys
        required_keys = [
            "swings",
            "bos",
            "ffg",
            "choch",
            "order_blocks",
            "liquidity",
            "last_swing",
            "last_bos",
            "active_ffg",
            "last_high",
            "last_low",
        ]
        for key in required_keys:
            assert key in result

        # Validate types
        assert isinstance(result["swings"], list)
        assert isinstance(result["bos"], list)
        assert isinstance(result["ffg"], list)
        assert isinstance(result["choch"], list)
        assert isinstance(result["order_blocks"], list)
        assert isinstance(result["liquidity"], list)
        assert isinstance(result["active_ffg"], list)
        assert isinstance(result["last_high"], (int, float))
        assert isinstance(result["last_low"], (int, float))

    def test_bias_detection(self, sample_df):
        """Test: detects market bias."""
        smc = SMCIndicator()
        bias = smc.get_bias(sample_df)

        assert bias in ["bullish", "bearish", "neutral"]

    def test_entry_level(self, sample_df):
        """Test: returns entry level."""
        smc = SMCIndicator()
        # Run analysis first to populate cache
        smc.analyze(sample_df)

        entry_long = smc.get_entry_level(sample_df, "long")
        entry_short = smc.get_entry_level(sample_df, "short")

        # Should be a number or None
        assert entry_long is None or isinstance(entry_long, (int, float))
        assert entry_short is None or isinstance(entry_short, (int, float))

    def test_stop_loss(self, sample_df):
        """Test: returns stop loss level."""
        smc = SMCIndicator()
        smc.analyze(sample_df)  # Populate cache

        sl_long = smc.get_stop_loss(sample_df, "long")
        sl_short = smc.get_stop_loss(sample_df, "short")

        # Should be a number or None
        assert sl_long is None or isinstance(sl_long, (int, float))
        assert sl_short is None or isinstance(sl_short, (int, float))

    def test_take_profit_levels(self, sample_df):
        """Test: returns multiple TP levels."""
        smc = SMCIndicator()
        smc.analyze(sample_df)  # Populate cache

        tp_long = smc.get_take_profit_levels(sample_df, "long", count=3)
        tp_short = smc.get_take_profit_levels(sample_df, "short", count=3)

        # Should be lists
        assert isinstance(tp_long, list)
        assert isinstance(tp_short, list)

        # Validate TP values
        if tp_long:
            assert all(isinstance(tp, (int, float)) for tp in tp_long)
        if tp_short:
            assert all(isinstance(tp, (int, float)) for tp in tp_short)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_get_previous_highs_lows(self, sample_df):
        """Test: gets previous highs/lows."""
        levels = get_previous_highs_lows(sample_df, periods=[20, 50])

        assert "highs" in levels
        assert "lows" in levels
        assert isinstance(levels["highs"], list)
        assert isinstance(levels["lows"], list)

        # Should have levels for each period
        if len(sample_df) >= 50:
            assert len(levels["highs"]) == 2
            assert len(levels["lows"]) == 2

    def test_calculate_retracement_levels(self):
        """Test: calculates Fibonacci retracement levels."""
        start_price = 100.0
        end_price = 200.0

        levels = calculate_retracement_levels(start_price, end_price)

        # Validate all levels
        assert levels.level_0 == start_price
        assert levels.level_1 == end_price
        assert start_price < levels.level_236 < end_price
        assert start_price < levels.level_382 < end_price
        assert start_price < levels.level_500 < end_price
        assert start_price < levels.level_618 < end_price
        assert start_price < levels.level_786 < end_price

        # Validate ordering
        assert levels.level_236 < levels.level_382
        assert levels.level_382 < levels.level_500
        assert levels.level_500 < levels.level_618
        assert levels.level_618 < levels.level_786


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
