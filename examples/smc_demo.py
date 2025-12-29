"""
SMC Indicators Demo Script

Demonstrates the usage of Smart Money Concepts indicators
for BTC/XAU trading analysis.
"""

import pandas as pd
import numpy as np
from botclave.engine.indicators import (
    SMCIndicator,
    SwingDetector,
    FairValueGapDetector,
    OrderBlockDetector,
    calculate_retracement_levels,
)


def create_sample_data(periods: int = 200) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h")
    np.random.seed(42)
    
    # Create trending price data
    trend = np.linspace(50000, 52000, periods)
    noise = np.cumsum(np.random.randn(periods) * 100)
    close = trend + noise
    
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(periods) * 50,
            "high": close + abs(np.random.randn(periods) * 100),
            "low": close - abs(np.random.randn(periods) * 100),
            "close": close,
            "volume": np.random.uniform(100, 1000, periods),
        },
        index=dates,
    )
    
    return df


def main():
    """Run SMC indicators demo."""
    print("=" * 70)
    print("SMC INDICATORS DEMO - BTC/XAU Trading Analysis")
    print("=" * 70)
    print()
    
    # Create sample data
    print("📊 Creating sample market data...")
    df = create_sample_data(periods=200)
    print(f"   Timeframe: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"   Total candles: {len(df)}")
    print()
    
    # Initialize SMC indicator
    print("🔧 Initializing SMC Indicator...")
    smc = SMCIndicator(left_bars=2, right_bars=2, min_ffg_percent=0.01)
    print()
    
    # Run full analysis
    print("📈 Running SMC Analysis...")
    analysis = smc.analyze(df)
    print()
    
    # Display results
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    # Market bias
    bias = smc.get_bias(df)
    bias_emoji = "🟢" if bias == "bullish" else "🔴" if bias == "bearish" else "⚪"
    print(f"{bias_emoji} Market Bias: {bias.upper()}")
    print()
    
    # Swings
    print(f"📍 Swing Highs/Lows: {len(analysis['swings'])} detected")
    if analysis['swings']:
        swing_highs = [s for s in analysis['swings'] if s.swing_type == 'high']
        swing_lows = [s for s in analysis['swings'] if s.swing_type == 'low']
        print(f"   - Swing Highs: {len(swing_highs)}")
        print(f"   - Swing Lows: {len(swing_lows)}")
        if analysis['last_swing']:
            print(f"   - Last Swing: {analysis['last_swing'].swing_type} @ ${analysis['last_swing'].price:.2f}")
    print()
    
    # Break of Structure
    print(f"🔄 Break of Structure (BOS): {len(analysis['bos'])} detected")
    if analysis['bos']:
        bullish_bos = [b for b in analysis['bos'] if b.direction == 'bullish']
        bearish_bos = [b for b in analysis['bos'] if b.direction == 'bearish']
        print(f"   - Bullish BOS: {len(bullish_bos)}")
        print(f"   - Bearish BOS: {len(bearish_bos)}")
        if analysis['last_bos']:
            print(f"   - Last BOS: {analysis['last_bos'].direction} @ ${analysis['last_bos'].price:.2f}")
    print()
    
    # Fair Value Gaps
    print(f"📊 Fair Value Gaps (FFG): {len(analysis['ffg'])} total")
    print(f"   - Active (unfilled): {len(analysis['active_ffg'])}")
    if analysis['active_ffg']:
        print(f"   Active gaps:")
        for i, ffg in enumerate(analysis['active_ffg'][:3], 1):  # Show first 3
            print(f"      {i}. {ffg.direction} FFG: ${ffg.bottom_price:.2f} - ${ffg.top_price:.2f}")
    print()
    
    # Order Blocks
    print(f"🎯 Order Blocks: {len(analysis['order_blocks'])} detected")
    if analysis['order_blocks']:
        bullish_ob = [ob for ob in analysis['order_blocks'] if ob.direction == 'bullish']
        bearish_ob = [ob for ob in analysis['order_blocks'] if ob.direction == 'bearish']
        print(f"   - Bullish OB: {len(bullish_ob)}")
        print(f"   - Bearish OB: {len(bearish_ob)}")
        
        # Show unmitigated order blocks
        unmitigated = [ob for ob in analysis['order_blocks'] if ob.mitigated_index is None]
        if unmitigated:
            print(f"   - Unmitigated: {len(unmitigated)}")
    print()
    
    # Change of Character
    print(f"⚡ Change of Character (CHoCH): {len(analysis['choch'])} detected")
    if analysis['choch']:
        for i, ch in enumerate(analysis['choch'][-3:], 1):  # Show last 3
            print(f"   {i}. {ch.previous_trend} → {ch.new_trend} @ ${ch.trigger_price:.2f}")
    print()
    
    # Liquidity Clusters
    print(f"💧 Liquidity Clusters: {len(analysis['liquidity'])} detected")
    if analysis['liquidity']:
        for i, liq in enumerate(sorted(analysis['liquidity'], key=lambda x: x.strength, reverse=True)[:3], 1):
            print(f"   {i}. {liq.direction} @ ${liq.price:.2f} (strength: {liq.strength})")
    print()
    
    # Trading levels
    print("=" * 70)
    print("TRADE SETUP RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Long setup
    print("📈 LONG SETUP:")
    long_entry = smc.get_entry_level(df, 'long')
    long_sl = smc.get_stop_loss(df, 'long')
    long_tps = smc.get_take_profit_levels(df, 'long', count=3)
    
    if long_entry:
        print(f"   Entry: ${long_entry:.2f}")
        if long_sl:
            print(f"   Stop Loss: ${long_sl:.2f}")
            risk = abs(long_entry - long_sl)
            print(f"   Risk: ${risk:.2f}")
        if long_tps:
            print(f"   Take Profits:")
            for i, tp in enumerate(long_tps, 1):
                if long_sl:
                    reward = abs(tp - long_entry)
                    rr_ratio = reward / risk if risk > 0 else 0
                    print(f"      TP{i}: ${tp:.2f} (R:R = 1:{rr_ratio:.2f})")
                else:
                    print(f"      TP{i}: ${tp:.2f}")
    else:
        print("   No clear long setup at current levels")
    print()
    
    # Short setup
    print("📉 SHORT SETUP:")
    short_entry = smc.get_entry_level(df, 'short')
    short_sl = smc.get_stop_loss(df, 'short')
    short_tps = smc.get_take_profit_levels(df, 'short', count=3)
    
    if short_entry:
        print(f"   Entry: ${short_entry:.2f}")
        if short_sl:
            print(f"   Stop Loss: ${short_sl:.2f}")
            risk = abs(short_entry - short_sl)
            print(f"   Risk: ${risk:.2f}")
        if short_tps:
            print(f"   Take Profits:")
            for i, tp in enumerate(short_tps, 1):
                if short_sl:
                    reward = abs(short_entry - tp)
                    rr_ratio = reward / risk if risk > 0 else 0
                    print(f"      TP{i}: ${tp:.2f} (R:R = 1:{rr_ratio:.2f})")
                else:
                    print(f"      TP{i}: ${tp:.2f}")
    else:
        print("   No clear short setup at current levels")
    print()
    
    # Fibonacci retracement example
    if len(analysis['swings']) >= 2:
        print("=" * 70)
        print("FIBONACCI RETRACEMENT LEVELS")
        print("=" * 70)
        print()
        
        # Get last two significant swings
        swing_highs = [s for s in analysis['swings'] if s.swing_type == 'high']
        swing_lows = [s for s in analysis['swings'] if s.swing_type == 'low']
        
        if swing_highs and swing_lows:
            last_high = swing_highs[-1]
            last_low = swing_lows[-1]
            
            fib_levels = calculate_retracement_levels(last_low.price, last_high.price)
            
            print(f"From Low: ${last_low.price:.2f} to High: ${last_high.price:.2f}")
            print()
            print(f"   0.0% (Low):   ${fib_levels.level_0:.2f}")
            print(f"  23.6%:         ${fib_levels.level_236:.2f}")
            print(f"  38.2%:         ${fib_levels.level_382:.2f}")
            print(f"  50.0%:         ${fib_levels.level_500:.2f}")
            print(f"  61.8%:         ${fib_levels.level_618:.2f}")
            print(f"  78.6%:         ${fib_levels.level_786:.2f}")
            print(f" 100.0% (High): ${fib_levels.level_1:.2f}")
            print()
    
    print("=" * 70)
    print("✅ SMC Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
