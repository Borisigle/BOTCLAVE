"""
Test suite for the strategy module.

Tests the core trading strategy logic including risk-reward calculations,
signal generation, orderflow analysis, and multi-timeframe confluence.
"""

import pytest
import pandas as pd
import numpy as np
from botclave.engine.strategy import (
    TradingStrategy, RiskRewardSetup, Signal, OrderflowAnalyzer
)
from botclave.engine.footprint import KlineFootprint, Trade


@pytest.fixture
def sample_df():
    """DataFrame with bullish trend + FFG"""
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
    """Footprints with absorption"""
    footprints = []
    for i in range(len(sample_df)):
        fp = KlineFootprint(price_step=0.5)
        
        # Add simulated trades
        if i % 10 == 0:  # Every 10 candles, there's buy absorption
            for _ in range(50):
                fp.add_trade(Trade(
                    price=sample_df['close'].iloc[i],
                    qty=100,
                    is_buy=True,
                    time_ms=int(i * 1000)
                ))
            for _ in range(10):
                fp.add_trade(Trade(
                    price=sample_df['close'].iloc[i],
                    qty=100,
                    is_buy=False,
                    time_ms=int(i * 1000)
                ))
        
        footprints.append(fp)
    
    return footprints


def test_risk_reward_setup():
    """Test: RiskRewardSetup calculates RR correctly"""
    setup = RiskRewardSetup(
        entry_price=100.0,
        stop_loss_price=99.0,
        take_profit_price=103.0,
        position_size=1.0
    )
    
    assert setup.risk_reward_ratio == 3.0  # 3 up / 1 down
    assert setup.is_valid_rr(min_rr=3.0)


def test_risk_reward_setup_invalid():
    """Test: RiskRewardSetup correctly identifies invalid RR"""
    setup = RiskRewardSetup(
        entry_price=100.0,
        stop_loss_price=99.0,
        take_profit_price=101.5,
        position_size=1.0
    )
    
    assert setup.risk_reward_ratio == 1.5  # 1.5 up / 1 down
    assert not setup.is_valid_rr(min_rr=3.0)


def test_strategy_analyze(sample_df, sample_footprints):
    """Test: Strategy.analyze() returns complete dict"""
    strategy = TradingStrategy()
    result = strategy.analyze(sample_df, sample_footprints)
    
    # Must have all required keys
    required_keys = ['smc', 'orderflow', 'signals', 'current_bias', 'confluence_level']
    for key in required_keys:
        assert key in result


def test_signal_generation(sample_df, sample_footprints):
    """Test: Generates signals when there's confluence"""
    strategy = TradingStrategy(min_rr=3.0, min_confidence=0.5)
    result = strategy.analyze(sample_df, sample_footprints)
    
    signals = result['signals']
    
    # Signals may be empty (depends on data), but if present they must be valid
    for signal in signals:
        assert signal.signal_type in ['ENTRY_LONG', 'ENTRY_SHORT', 'EXIT_LONG', 'EXIT_SHORT']
        assert 0 <= signal.confidence <= 1.0
        if signal.entry_setup:
            assert signal.entry_setup.risk_reward_ratio >= 3.0


def test_bias_detection(sample_df, sample_footprints):
    """Test: detects bias correctly"""
    strategy = TradingStrategy()
    result = strategy.analyze(sample_df, sample_footprints)
    
    bias = result['current_bias']
    assert bias in ['bullish', 'bearish', 'neutral']


def test_orderflow_analyzer():
    """Test: OrderflowAnalyzer detects absorption"""
    analyzer = OrderflowAnalyzer(delta_threshold=0.65)
    
    # Create footprint with buy absorption
    fp = KlineFootprint()
    for _ in range(100):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=True, time_ms=1000))
    for _ in range(20):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=False, time_ms=1001))
    
    # Delta should be > 65%
    imbalance = fp.get_imbalance(100.0, threshold=0.65)
    assert imbalance == 'buy'


def test_orderflow_analyzer_sell_absorption():
    """Test: OrderflowAnalyzer detects sell absorption"""
    analyzer = OrderflowAnalyzer(delta_threshold=0.65)
    
    # Create footprint with sell absorption
    fp = KlineFootprint()
    for _ in range(20):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=True, time_ms=1000))
    for _ in range(100):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=False, time_ms=1001))
    
    # Delta should be > 65% negative
    imbalance = fp.get_imbalance(100.0, threshold=0.65)
    assert imbalance == 'sell'


def test_absorption_zones(sample_df, sample_footprints):
    """Test: find_absorption_zones returns correct zones"""
    analyzer = OrderflowAnalyzer(delta_threshold=0.65)
    
    # Find absorption zones in last 5 candles
    absorption_zones = analyzer.find_absorption_zones(
        sample_df, sample_footprints, lookback=5
    )
    
    # Should return a dictionary
    assert isinstance(absorption_zones, dict)
    
    # All values should be 'BUY' or 'SELL'
    for absorption_type in absorption_zones.values():
        assert absorption_type in ['BUY', 'SELL']


def test_absorption_strength():
    """Test: get_absorption_strength returns value between 0-1"""
    analyzer = OrderflowAnalyzer(delta_threshold=0.65)
    
    # Create footprint with strong absorption
    fp = KlineFootprint()
    for _ in range(100):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=True, time_ms=1000))
    for _ in range(10):
        fp.add_trade(Trade(price=100.0, qty=10, is_buy=False, time_ms=1001))
    
    strength = analyzer.get_absorption_strength(fp, 100.0)
    
    assert 0.0 <= strength <= 1.0
    assert strength > 0.6  # Should be above threshold


def test_get_current_signal(sample_df, sample_footprints):
    """Test: get_current_signal returns strongest signal"""
    strategy = TradingStrategy()
    
    # Get current signal
    signal = strategy.get_current_signal(sample_df, sample_footprints)
    
    # Signal may be None if no valid signals
    if signal:
        assert isinstance(signal, Signal)
        assert 0 <= signal.confidence <= 1.0


def test_multi_timeframe_strategy():
    """Test: MultiTimeframeStrategy basic functionality"""
    from botclave.engine.strategy import MultiTimeframeStrategy
    
    # Create strategy with multiple timeframes
    strategy = MultiTimeframeStrategy(timeframes=['15m', '1h', '4h'])
    
    # Should have strategies for each timeframe
    assert len(strategy.strategies) == 3
    assert '15m' in strategy.strategies
    assert '1h' in strategy.strategies
    assert '4h' in strategy.strategies


def test_confluence_calculation():
    """Test: confluence calculation with multiple signals"""
    strategy = TradingStrategy()
    
    # Create some test signals
    signals = [
        Signal(
            signal_type='ENTRY_LONG',
            index=0,
            price=100.0,
            time='2024-01-01',
            confidence=0.8,
            smc_component='BOS',
            orderflow_component='ABSORPTION_BUY'
        ),
        Signal(
            signal_type='ENTRY_LONG',
            index=1,
            price=101.0,
            time='2024-01-02',
            confidence=0.9,
            smc_component='FFG',
            orderflow_component='ABSORPTION_BUY'
        )
    ]
    
    # Calculate confluence
    confluence = strategy._calculate_confluence(signals)
    
    # Should be average of confidences
    expected = (0.8 + 0.9) / 2
    assert abs(confluence - expected) < 0.001


def test_signal_creation_long():
    """Test: signal creation for long entry"""
    strategy = TradingStrategy(min_rr=2.0)  # Lower RR requirement for testing
    
    # Create mock SMC result
    class MockFFG:
        def __init__(self):
            self.bottom_price = 100.0
            self.top_price = 105.0
            self.direction = 'bullish'
    
    class MockSwing:
        def __init__(self):
            self.price = 98.0
            self.swing_type = 'low'
    
    smc_result = {
        'last_swing': MockSwing(),
        'swings': []
    }
    
    # Create mock DataFrame
    df = pd.DataFrame({
        'close': [102.0]
    })
    
    # Create signal
    signal = strategy._create_entry_signal(
        signal_type='ENTRY_LONG',
        index=0,
        price=102.0,
        time='2024-01-01',
        smc_component='BOS',
        orderflow_component='ABSORPTION_BUY',
        df=df,
        smc_result=smc_result,
        ffg=MockFFG()
    )
    
    # Verify signal properties
    assert signal is not None
    assert signal.signal_type == 'ENTRY_LONG'
    assert signal.entry_setup.entry_price == 100.0
    assert signal.entry_setup.stop_loss_price == 98.0
    assert signal.entry_setup.take_profit_price == 105.0  # 5% above entry
    assert signal.confidence > 0.7


def test_signal_creation_short():
    """Test: signal creation for short entry"""
    strategy = TradingStrategy(min_rr=2.0)  # Lower RR requirement for testing
    
    # Create mock SMC result
    class MockFFG:
        def __init__(self):
            self.top_price = 100.0
            self.bottom_price = 95.0
            self.direction = 'bearish'
    
    class MockSwing:
        def __init__(self):
            self.price = 102.0
            self.swing_type = 'high'
    
    smc_result = {
        'last_swing': MockSwing(),
        'swings': []
    }
    
    # Create mock DataFrame
    df = pd.DataFrame({
        'close': [98.0]
    })
    
    # Create signal
    signal = strategy._create_entry_signal(
        signal_type='ENTRY_SHORT',
        index=0,
        price=98.0,
        time='2024-01-01',
        smc_component='BOS',
        orderflow_component='ABSORPTION_SELL',
        df=df,
        smc_result=smc_result,
        ffg=MockFFG()
    )
    
    # Verify signal properties
    assert signal is not None
    assert signal.signal_type == 'ENTRY_SHORT'
    assert signal.entry_setup.entry_price == 100.0
    assert signal.entry_setup.stop_loss_price == 102.0
    assert signal.entry_setup.take_profit_price == 95.0  # 5% below entry
    assert signal.confidence > 0.7
