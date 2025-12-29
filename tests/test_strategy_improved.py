"""
Test suite for enhanced TradingStrategy with multi-timeframe confluence, session filtering, and ATR-based risk management.

Tests new features:
- SessionFilter: Market session detection
- ATRCalculator: Dynamic stop loss/take profit sizing
- MultiTimeframeStrategy: Enhanced confluence analysis
- Signal explanations: Detailed trade justification
- Backward compatibility: Existing functionality preserved
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from botclave.engine.strategy import (
    TradingStrategy, MultiTimeframeStrategy, SessionFilter, ATRCalculator,
    RiskRewardSetup, Signal, OrderflowAnalyzer
)
from botclave.engine.footprint import KlineFootprint, Trade


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Base DataFrame fixture for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Bullish trend
    close = 100 + np.cumsum(np.random.randn(100) * 0.3 + 0.2)
    
    return pd.DataFrame({
        'open': close + np.random.randn(100) * 0.2,
        'high': close + abs(np.random.randn(100) * 0.5),
        'low': close - abs(np.random.randn(100) * 0.5),
        'close': close,
        'volume': np.random.uniform(1000, 10000, 100),
    }, index=dates)


@pytest.fixture
def sample_footprints(sample_df):
    """Footprints with simulated absorption"""
    footprints = []
    for i in range(len(sample_df)):
        fp = KlineFootprint(price_step=0.5)
        
        # Simulate buy absorption on some candles
        if i % 10 == 0:
            # Buy absorption: many more buys than sells
            for _ in range(100):
                fp.add_trade(Trade(
                    price=sample_df['close'].iloc[i],
                    qty=10,
                    is_buy=True,
                    time_ms=int(i * 1000)
                ))
            for _ in range(20):
                fp.add_trade(Trade(
                    price=sample_df['close'].iloc[i],
                    qty=10, 
                    is_buy=False,
                    time_ms=int(i * 1000)
                ))
        else:
            # Normal trading
            for _ in range(60):
                fp.add_trade(Trade(
                    price=sample_df['close'].iloc[i],
                    qty=10,
                    is_buy=np.random.random() > 0.5,
                    time_ms=int(i * 1000)
                ))
        
        footprints.append(fp)
    
    return footprints


@pytest.fixture
def london_session_df():
    """DataFrame during London session (3:00-12:00 UTC)"""
    # Create times during London session (8am-12pm UTC)
    dates = pd.date_range('2024-01-01 08:00', periods=20, freq='15min', tz='UTC')
    close = 100 + np.cumsum(np.random.randn(20) * 0.1)
    
    return pd.DataFrame({
        'open': close,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': np.random.uniform(1000, 5000, 20),
    }, index=dates)


@pytest.fixture
def off_session_df():
    """DataFrame outside active sessions (2:00 UTC - between sessions)"""
    dates = pd.date_range('2024-01-01 02:00', periods=20, freq='15min', tz='UTC')
    close = 100 + np.cumsum(np.random.randn(20) * 0.05)
    
    return pd.DataFrame({
        'open': close,
        'high': close + 0.3,
        'low': close - 0.3,
        'close': close,
        'volume': np.random.uniform(500, 2000, 20),
    }, index=dates)


@pytest.fixture
def mtf_dataframes():
    """Multi-timeframe data for confluence testing"""
    np.random.seed(42)
    base_price = 100
    
    # 4H DataFrame - strong bullish bias
    dates_4h = pd.date_range('2024-01-01', periods=50, freq='4H')
    close_4h = base_price + np.cumsum(np.random.randn(50) * 0.8 + 0.4)
    df_4h = pd.DataFrame({
        'open': close_4h,
        'high': close_4h + 1.0,
        'low': close_4h - 1.0,
        'close': close_4h,
        'volume': np.random.uniform(5000, 15000, 50),
    }, index=dates_4h)
    
    # 1H DataFrame - bullish with pullback
    dates_1h = pd.date_range('2024-01-01', periods=200, freq='1H')
    close_1h = base_price + np.cumsum(np.random.randn(200) * 0.4 + 0.1)
    df_1h = pd.DataFrame({
        'open': close_1h,
        'high': close_1h + 0.5,
        'low': close_1h - 0.5,
        'close': close_1h,
        'volume': np.random.uniform(2000, 8000, 200),
    }, index=dates_1h)
    
    # 15m DataFrame - recent candles with FFG potential
    dates_15m = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
    close_15m = base_price + np.cumsum(np.random.randn(300) * 0.2)
    df_15m = pd.DataFrame({
        'open': close_15m,
        'high': close_15m + 0.3,
        'low': close_15m - 0.3,
        'close': close_15m,
        'volume': np.random.uniform(1000, 4000, 300),
    }, index=dates_15m)
    
    return {
        '4h': df_4h,
        '1h': df_1h, 
        '15m': df_15m
    }


# =============================================================================
# SESSION FILTER TESTS
# =============================================================================

class TestSessionFilter:
    """Test suite for SessionFilter functionality"""
    
    def test_session_filter_initialization(self):
        """Test SessionFilter initializes with default sessions"""
        session_filter = SessionFilter()
        assert 'London' in session_filter.active_sessions
        assert 'New York' in session_filter.active_sessions
        assert 'Asian Kill Zone' in session_filter.active_sessions
    
    def test_session_filter_custom_sessions(self):
        """Test SessionFilter with custom session list"""
        session_filter = SessionFilter(['London', 'Tokyo'])
        assert 'London' in session_filter.active_sessions
        assert 'Tokyo' in session_filter.active_sessions
        assert 'New York' not in session_filter.active_sessions
    
    def test_london_session_detection(self, london_session_df):
        """Test detecting London session (3:00-12:00 UTC)"""
        session_filter = SessionFilter(['London'])
        
        # Should be active at 8:00 UTC (London session)
        assert session_filter.is_active_session(london_session_df)
        assert session_filter.get_current_session(london_session_df) == 'London'
    
    def test_off_session_detection(self, off_session_df):
        """Test detecting outside active sessions (2:00 UTC)"""
        session_filter = SessionFilter(['London', 'New York'])
        
        # Should be inactive at 2:00 UTC (between sessions)
        assert not session_filter.is_active_session(off_session_df)
        assert session_filter.get_current_session(off_session_df) == 'Outside Session'
    
    def test_nyc_session_detection(self):
        """Test detecting New York session (13:00-22:00 UTC)"""
        nyc_time = pd.date_range('2024-01-01 15:00', periods=5, freq='15min', tz='UTC')
        df_nyc = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
        }, index=nyc_time)
        
        session_filter = SessionFilter(['New York'])
        assert session_filter.is_active_session(df_nyc)
        assert session_filter.get_current_session(df_nyc) == 'New York'
    
    def test_asian_kill_zone_detection(self):
        """Test detecting Asian Kill Zone (0:00-4:00 UTC)"""
        asian_time = pd.date_range('2024-01-01 02:30', periods=5, freq='15min', tz='UTC')
        df_asian = pd.DataFrame({
            'open': [100] * 5,
            'high': [101] * 5,
            'low': [99] * 5,
            'close': [100] * 5,
            'volume': [1000] * 5,
        }, index=asian_time)
        
        session_filter = SessionFilter(['Asian Kill Zone'])
        assert session_filter.is_active_session(df_asian)
        assert session_filter.get_current_session(df_asian) == 'Asian Kill Zone'


# =============================================================================
# ATR CALCULATOR TESTS
# =============================================================================

class TestATRCalculator:
    """Test suite for ATRCalculator functionality"""
    
    def test_atr_calculator_initialization(self):
        """Test ATRCalculator initializes with default period"""
        atr_calc = ATRCalculator()
        assert atr_calc.period == 14
    
    def test_atr_calculator_custom_period(self):
        """Test ATRCalculator with custom period"""
        atr_calc = ATRCalculator(period=20)
        assert atr_calc.period == 20
    
    def test_atr_calculation(self, sample_df):
        """Test ATR calculation on sample data"""
        atr_calc = ATRCalculator(period=14)
        atr_series = atr_calc.calculate(sample_df)
        
        # Should return pandas Series
        assert isinstance(atr_series, pd.Series)
        assert len(atr_series) == len(sample_df)
        
        # ATR should be positive values
        assert (atr_series >= 0).all()
        
        # Should handle insufficient data
        df_short = sample_df.head(5)
        atr_short = atr_calc.calculate(df_short)
        assert isinstance(atr_short, pd.Series)
        assert len(atr_short) == len(df_short)
    
    def test_sl_price_calculation(self):
        """Test stop loss price calculation"""
        atr_calc = ATRCalculator()
        entry_price = 100.0
        atr = 2.0
        
        # Long position SL = entry - 2*ATR
        sl_long = atr_calc.get_sl_price(entry_price, atr, 'long')
        assert sl_long == 96.0  # 100 - 2*2
        
        # Short position SL = entry + 2*ATR
        sl_short = atr_calc.get_sl_price(entry_price, atr, 'short')
        assert sl_short == 104.0  # 100 + 2*2
    
    def test_sl_price_custom_multiplier(self):
        """Test stop loss with custom ATR multiplier"""
        atr_calc = ATRCalculator()
        entry_price = 100.0
        atr = 2.0
        
        # Long with 3x ATR multiplier
        sl_long = atr_calc.get_sl_price(entry_price, atr, 'long', multiplier=3.0)
        assert sl_long == 94.0  # 100 - 3*2
    
    def test_tp_price_calculation(self):
        """Test take profit price calculation"""
        atr_calc = ATRCalculator()
        entry_price = 100.0
        atr = 2.0
        
        # Long position TP = entry + (2*ATR * RR_ratio)
        # RR_ratio = 2 means TP = entry + 4*ATR
        tp_long = atr_calc.get_tp_price(entry_price, atr, 'long', rr_ratio=2.0)
        assert tp_long == 108.0  # 100 + (2*2)*2 = 100 + 8
        
        # Short position TP = entry - (2*ATR * RR_ratio)
        tp_short = atr_calc.get_tp_price(entry_price, atr, 'short', rr_ratio=2.0)
        assert tp_short == 92.0  # 100 - 8
    
    def test_rr_setup_creation(self):
        """Test complete RiskRewardSetup creation"""
        atr_calc = ATRCalculator()
        entry_price = 100.0
        atr = 2.0
        
        # Create setup for long position
        setup = atr_calc.get_rr_setup(
            entry_price=entry_price,
            atr=atr,
            direction='long',
            rr_ratio=2.0,
            atr_multiplier=2.0
        )
        
        assert isinstance(setup, RiskRewardSetup)
        assert setup.entry_price == 100.0
        assert setup.stop_loss_price == 96.0  # 100 - 2*2
        assert setup.take_profit_price == 108.0  # 100 + 8
        assert setup.risk_reward_ratio == 2.0  # 8/4 = 2.0
    
    def test_invalid_direction_raises_error(self):
        """Test that invalid direction raises ValueError"""
        atr_calc = ATRCalculator()
        
        with pytest.raises(ValueError, match="Direction must be 'long' or 'short'"):
            atr_calc.get_sl_price(100.0, 2.0, 'invalid')
        
        with pytest.raises(ValueError, match="Direction must be 'long' or 'short'"):
            atr_calc.get_tp_price(100.0, 2.0, 'invalid')


# =============================================================================
# MULTI-TIMEFRAME SIGNAL GENERATION TESTS
# =============================================================================

class TestMultiTimeframeSignalGeneration:
    """Test suite for enhanced multi-timeframe signal generation"""
    
    def test_strategy_initializes_with_new_components(self):
        """Test TradingStrategy initializes with SessionFilter and ATRCalculator"""
        strategy = TradingStrategy(
            active_sessions=['London', 'New York'],
            atr_period=14
        )
        
        # Should have new components
        assert hasattr(strategy, 'session_filter')
        assert hasattr(strategy, 'atr_calc')
        assert isinstance(strategy.session_filter, SessionFilter)
        assert isinstance(strategy.atr_calc, ATRCalculator)
    
    def test_generate_signal_requires_active_session(self, mtf_dataframes, sample_footprints):
        """Test signal generation only during active sessions"""
        strategy = TradingStrategy(active_sessions=['London'])
        
        # Use data from non-London session (should already be off-session by default setup)
        result = strategy.generate_signal(
            df_15m=mtf_dataframes['15m'],
            df_1h=mtf_dataframes['1h'],
            df_4h=mtf_dataframes['4h'],
            footprints_15m=sample_footprints[:100]  # Match 15m timeframe length
        )
        
        # Should return None outside active session
        assert result is None
    
    def test_generate_signal_with_london_session(self, mtf_dataframes, sample_footprints):
        """Test signal generation during London session"""
        strategy = TradingStrategy(
            active_sessions=['London'],
            min_confidence=0.5  # Lower for testing
        )
        
        # Use London session timeframes
        london_15m = mtf_dataframes['15m'].copy()
        london_15m.index = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
        
        london_1h = mtf_dataframes['1h'].copy()
        london_1h.index = pd.date_range('2024-01-01 08:00', periods=200, freq='1H')
        
        london_4h = mtf_dataframes['4h'].copy()
        london_4h.index = pd.date_range('2024-01-01 08:00', periods=50, freq='4H')
        
        result = strategy.generate_signal(
            df_15m=london_15m,
            df_1h=london_1h,
            df_4h=london_4h,
            footprints_15m=sample_footprints[:300]
        )
        
        # May or may not generate signal depending on data, but should not error
        if result:
            assert isinstance(result, Signal)
            assert result.confidence >= 0.5  # Meets minimum
    
    def test_signal_explanation_contains_required_sections(self, mtf_dataframes, sample_footprints):
        """Test that signal explanation contains all required sections"""
        strategy = TradingStrategy(
            active_sessions=['London'],
            min_confidence=0.0  # Allow all signals
        )
        
        london_15m = mtf_dataframes['15m'].copy()
        london_15m.index = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
        
        london_1h = mtf_dataframes['1h'].copy()
        london_1h.index = pd.date_range('2024-01-01 08:00', periods=200, freq='1H')
        
        london_4h = mtf_dataframes['4h'].copy()
        london_4h.index = pd.date_range('2024-01-01 08:00', periods=50, freq='4H')
        
        result = strategy.generate_signal(
            df_15m=london_15m,
            df_1h=london_1h,
            df_4h=london_4h,
            footprints_15m=sample_footprints[:300]
        )
        
        if result:
            explanation = result.reason
            
            # Check for required sections in explanation
            assert '📅 SESSION:' in explanation
            assert '📊 4H BIAS:' in explanation
            assert '🔍 1H CONFIRMATION:' in explanation
            assert '🎯 15m ENTRY:' in explanation
            assert '💰 ORDERFLOW:' in explanation
            assert '📏 RISK:' in explanation
            assert '🎲 RR RATIO:' in explanation
            assert '✅ CONFIDENCE:' in explanation
    
    def test_atr_based_sl_tp(self, mtf_dataframes, sample_footprints):
        """Test that signals use ATR-based stop loss and take profit"""
        strategy = TradingStrategy(
            active_sessions=['London'],
            min_confidence=0.0
        )
        
        london_15m = mtf_dataframes['15m'].copy()
        london_15m.index = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
        
        london_1h = mtf_dataframes['1h'].copy()
        london_1h.index = pd.date_range('2024-01-01 08:00', periods=200, freq='1H')
        
        london_4h = mtf_dataframes['4h'].copy()
        london_4h.index = pd.date_range('2024-01-01 08:00', periods=50, freq='4H')
        
        result = strategy.generate_signal(
            df_15m=london_15m,
            df_1h=london_1h,
            df_4h=london_4h,
            footprints_15m=sample_footprints[:300]
        )
        
        if result and result.entry_setup:
            # Verify setup exists
            assert isinstance(result.entry_setup, RiskRewardSetup)
            assert result.entry_setup.entry_price > 0
            assert result.entry_setup.stop_loss_price > 0
            assert result.entry_setup.take_profit_price > 0
            
            # Check RR ratio meets minimum
            assert result.entry_setup.risk_reward_ratio >= 3.0


# =============================================================================
# MULTITIMEFRAME STRATEGY TESTS
# =============================================================================

class TestMultiTimeframeStrategyEnhanced:
    """Test enhanced MultiTimeframeStrategy with new confluence logic"""
    
    def test_multitimeframe_strategy_initialization(self):
        """Test MultiTimeframeStrategy initializes correctly"""
        strategy = MultiTimeframeStrategy(['15m', '1h', '4h'])
        assert len(strategy.timeframes) == 3
        assert '15m' in strategy.timeframes
        assert '1h' in strategy.timeframes
        assert '4h' in strategy.timeframes
        assert hasattr(strategy, 'strategy')
        assert isinstance(strategy.strategy, TradingStrategy)
    
    def test_analyze_returns_enhanced_format(self, mtf_dataframes, sample_footprints):
        """Test analyze() returns new enhanced format with session info"""
        strategy = MultiTimeframeStrategy(['15m', '1h', '4h'])
        
        # Set to London session
        london_15m = mtf_dataframes['15m'].copy()
        london_15m.index = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
        
        london_1h = mtf_dataframes['1h'].copy()
        london_1h.index = pd.date_range('2024-01-01 08:00', periods=200, freq='1H')
        
        london_4h = mtf_dataframes['4h'].copy()
        london_4h.index = pd.date_range('2024-01-01 08:00', periods=50, freq='4H')
        
        footprints_15m = sample_footprints[:300]
        
        result = strategy.analyze(
            df_dict={'15m': london_15m, '1h': london_1h, '4h': london_4h},
            footprints_dict={'15m': footprints_15m}
        )
        
        # Check enhanced return format
        assert 'entry_signal' in result
        assert 'stop_signal' in result
        assert 'bias_signal' in result
        assert 'confluence_score' in result
        assert 'recommendation' in result
        assert 'session_filter' in result
        assert 'mtf_analysis' in result
        
        # MTF analysis details
        mtf_details = result['mtf_analysis']
        assert mtf_details['has_4h_bias'] is True
        assert mtf_details['has_1h_confirmation'] is True
        assert mtf_details['has_orderflow'] is True
    
    def test_legacy_fallback_when_missing_timeframes(self, sample_df, sample_footprints):
        """Test fallback to legacy method when missing required timeframes"""
        strategy = MultiTimeframeStrategy(['15m'])
        
        # Missing 4h timeframe
        result = strategy.analyze(
            df_dict={'15m': sample_df},
            footprints_dict={'15m': sample_footprints[:100]}
        )
        
        # Should still work with legacy method
        assert 'entry_signal' in result
        assert 'confluence_score' in result
        assert 'recommendation' in result


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing functionality still works"""
    
    def test_existing_analyze_method_works(self, sample_df, sample_footprints):
        """Test original analyze() method still works"""
        strategy = TradingStrategy()
        result = strategy.analyze(sample_df, sample_footprints)
        
        assert isinstance(result, dict)
        assert 'smc' in result
        assert 'orderflow' in result
        assert 'signals' in result
        assert 'current_bias' in result
        assert 'confluence_level' in result
    
    def test_get_current_signal_still_works(self, sample_df, sample_footprints):
        """Test get_current_signal() method still works"""
        strategy = TradingStrategy()
        signal = strategy.get_current_signal(sample_df, sample_footprints)
        
        # Signal may be None or Signal depending on data
        if signal:
            assert isinstance(signal, Signal)
    
    def test_risk_reward_setup_unchanged(self):
        """Test RiskRewardSetup calculations unchanged"""
        setup = RiskRewardSetup(
            entry_price=100.0,
            stop_loss_price=99.0,  # $1 risk
            take_profit_price=103.0,  # $3 reward
            position_size=1.0
        )
        
        assert setup.risk_amount == 1.0
        assert setup.reward_amount == 3.0
        assert setup.risk_reward_ratio == 3.0
        assert setup.is_valid_rr(3.0) is True
    
    def test_orderflow_analyzer_unchanged(self):
        """Test OrderflowAnalyzer still works"""
        analyzer = OrderflowAnalyzer(delta_threshold=0.65)
        
        # Create footprint with buy absorption
        fp = KlineFootprint()
        for _ in range(100):
            fp.add_trade(Trade(price=100.0, qty=10, is_buy=True, time_ms=1000))
        for _ in range(20):
            fp.add_trade(Trade(price=100.0, qty=10, is_buy=False, time_ms=1001))
        
        absorption = analyzer.detect_absorption(fp, 100.0, 100.0)
        assert absorption == 'BUY'


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests for enhanced strategy"""
    
    def test_full_enhanced_strategy_workflow(self, mtf_dataframes, sample_footprints):
        """Test complete enhanced workflow from session check to signal explanation"""
        
        # Setup strategy with custom configuration
        strategy = TradingStrategy(
            min_rr=2.0,  # Lower for testing
            min_confidence=0.4,
            active_sessions=['London'],
            atr_period=14
        )
        
        # Prepare data for London session
        london_15m = mtf_dataframes['15m'].copy()
        london_15m.index = pd.date_range('2024-01-01 08:00', periods=300, freq='15min')
        
        london_1h = mtf_dataframes['1h'].copy()
        london_1h.index = pd.date_range('2024-01-01 08:00', periods=200, freq='1H')
        
        london_4h = mtf_dataframes['4h'].copy()
        london_4h.index = pd.date_range('2024-01-01 08:00', periods=50, freq='4H')
        
        footprints_15m = sample_footprints[:300]
        
        # 1. Check session
        session_active = strategy.session_filter.is_active_session(london_15m)
        assert session_active is True
        
        # 2. Generate signal
        signal = strategy.generate_signal(
            df_15m=london_15m,
            df_1h=london_1h,
            df_4h=london_4h,
            footprints_15m=footprints_15m
        )
        
        # 3. Verify signal if generated
        if signal:
            assert isinstance(signal, Signal)
            assert 0.4 <= signal.confidence <= 1.0
            assert signal.entry_setup is not None
            assert signal.reason is not None
            assert len(signal.reason) > 100  # Should be detailed
            
            # 4. Verify risk/reward meets minimum
            assert signal.entry_setup.risk_reward_ratio >= 2.0


if __name__ == '__main__':
    # Run tests manually if needed
    pytest.main([__file__, '-v'])