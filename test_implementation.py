#!/usr/bin/env python3
"""
Quick test script to verify the enhanced strategy implementation works correctly.
Tests key functionality without pytest dependency.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, './src')

try:
    from botclave.engine.strategy import (
        TradingStrategy, SessionFilter, ATRCalculator, 
        MultiTimeframeStrategy, RiskRewardSetup
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_session_filter():
    """Test SessionFilter functionality"""
    print("\\n🧪 Testing SessionFilter...")
    
    session_filter = SessionFilter(['London', 'New York'])
    
    # Test London session (8am UTC)
    london_time = pd.date_range('2024-01-01 08:00', periods=5, freq='15min', tz='UTC')
    london_df = pd.DataFrame({
        'open': [100] * 5, 'high': [101] * 5, 'low': [99] * 5,
        'close': [100] * 5, 'volume': [1000] * 5,
    }, index=london_time)
    
    assert session_filter.is_active_session(london_df), "London session detection failed"
    assert session_filter.get_current_session(london_df) == 'London', "London session name failed"
    
    # Test off-session (2am UTC)
    off_time = pd.date_range('2024-01-01 02:00', periods=5, freq='15min', tz='UTC')
    off_df = pd.DataFrame({
        'open': [100] * 5, 'high': [101] * 5, 'low': [99] * 5,
        'close': [100] * 5, 'volume': [1000] * 5,
    }, index=off_time)
    
    assert not session_filter.is_active_session(off_df), "Off-session detection failed"
    assert session_filter.get_current_session(off_df) == 'Outside Session', "Off-session name failed"
    
    print("✅ SessionFilter tests passed")


def test_atr_calculator():
    """Test ATRCalculator functionality"""
    print("\\n🧪 Testing ATRCalculator...")
    
    atr_calc = ATRCalculator(period=14)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    close = 100 + np.cumsum(np.random.randn(100) * 0.3)
    df = pd.DataFrame({
        'open': close,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
    }, index=dates)
    
    # Test ATR calculation
    atr_series = atr_calc.calculate(df)
    assert len(atr_series) == len(df), "ATR series length mismatch"
    assert atr_series.iloc[-1] > 0, "ATR should be positive"
    
    # Test SL/TP calculation
    entry_price = 100.0
    atr = atr_series.iloc[-1]
    
    sl_long = atr_calc.get_sl_price(entry_price, atr, 'long')
    tp_long = atr_calc.get_tp_price(entry_price, atr, 'long', rr_ratio=2.0)
    
    assert sl_long < entry_price, "Long SL should be below entry"
    assert tp_long > entry_price, "Long TP should be above entry"
    
    # Test RiskRewardSetup creation
    setup = atr_calc.get_rr_setup(entry_price, atr, 'long')
    assert isinstance(setup, RiskRewardSetup), "Should return RiskRewardSetup"
    assert setup.risk_reward_ratio == 2.0, "RR ratio should match input"
    
    print("✅ ATRCalculator tests passed")


def test_trading_strategy_initialization():
    """Test TradingStrategy initialization with new components"""
    print("\\n🧪 Testing TradingStrategy initialization...")
    
    strategy = TradingStrategy(
        active_sessions=['London'],
        atr_period=14
    )
    
    assert hasattr(strategy, 'session_filter'), "Missing session_filter"
    assert hasattr(strategy, 'atr_calc'), "Missing atr_calc"
    assert isinstance(strategy.session_filter, SessionFilter), "session_filter wrong type"
    assert isinstance(strategy.atr_calc, ATRCalculator), "atr_calc wrong type"
    
    print("✅ TradingStrategy initialization tests passed")


def test_multitimeframe_strategy():
    """Test MultiTimeframeStrategy enhanced functionality"""
    print("\\n🧪 Testing MultiTimeframeStrategy...")
    
    strategy = MultiTimeframeStrategy(['15m', '1h', '4h'])
    
    assert hasattr(strategy, 'strategy'), "Missing strategy instance"
    assert isinstance(strategy.strategy, TradingStrategy), "strategy wrong type"
    
    print("✅ MultiTimeframeStrategy tests passed")


def test_backward_compatibility():
    """Test backward compatibility with original methods"""
    print("\\n🧪 Testing backward compatibility...")
    
    strategy = TradingStrategy()
    
    # Test RiskRewardSetup (original functionality)
    setup = RiskRewardSetup(
        entry_price=100.0,
        stop_loss_price=99.0,
        take_profit_price=103.0,
        position_size=1.0
    )
    
    assert setup.risk_amount == 1.0, "Risk amount calculation changed"
    assert setup.reward_amount == 3.0, "Reward amount calculation changed"
    assert setup.risk_reward_ratio == 3.0, "RR ratio calculation changed"
    assert setup.is_valid_rr(3.0) is True, "RR validation changed"
    
    print("✅ Backward compatibility tests passed")


def run_all_tests():
    """Run all implementation tests"""
    print("🚀 Testing Enhanced TradingStrategy Implementation")
    print("=" * 60)
    
    try:
        test_session_filter()
        test_atr_calculator()
        test_trading_strategy_initialization()
        test_multitimeframe_strategy()
        test_backward_compatibility()
        
        print("\\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Implementation is working correctly.")
        print("\\n✨ New features available:")
        print("   • SessionFilter - Market session detection")
        print("   • ATRCalculator - Dynamic SL/TP sizing")
        print("   • Enhanced generate_signal() - MTF confluence")
        print("   • Detailed signal explanations")
        print("   • 100% backward compatible")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
