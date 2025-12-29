# SMC (Smart Money Concepts) Indicators

## Overview

The SMC indicators module provides a comprehensive suite of Smart Money Concepts indicators designed for institutional-level market structure analysis. These indicators help identify where "smart money" (institutions, banks, hedge funds) are likely entering and exiting positions.

## Core Concepts

### What is Smart Money Concepts?

Smart Money Concepts (SMC) is a trading methodology that focuses on understanding how institutional traders operate in the markets. Unlike retail traders who often chase price movements, institutional traders leave footprints in market structure that can be identified and exploited.

Key principles:
- **Market Structure**: Understanding swing highs and lows to identify trend direction
- **Liquidity**: Where orders are concentrated (stop losses, breakout traders)
- **Order Blocks**: Zones where institutions placed large orders
- **Fair Value Gaps**: Price imbalances that may get filled
- **Break of Structure**: Confirmations of trend changes

## Available Indicators

### 1. SwingDetector

Identifies swing highs and lows - structural pivots that define market direction.

```python
from botclave.engine.indicators import SwingDetector

detector = SwingDetector(left_bars=2, right_bars=2)
swings = detector.find_swings(df)

for swing in swings:
    print(f"{swing.swing_type} @ ${swing.price:.2f}")
```

**Parameters:**
- `left_bars`: Number of bars to check on the left (default: 2)
- `right_bars`: Number of bars to check on the right (default: 2)

**Returns:**
- List of `Swing` objects with:
  - `index`: Position in DataFrame
  - `price`: Price of the swing
  - `time`: Timestamp
  - `swing_type`: 'high' or 'low'

### 2. BreakOfStructureDetector

Detects when price breaks above previous swing highs (bullish) or below previous swing lows (bearish).

```python
from botclave.engine.indicators import BreakOfStructureDetector

bos_detector = BreakOfStructureDetector()
bos = bos_detector.find_bos(df, swings)

for b in bos:
    print(f"{b.direction} BOS @ ${b.price:.2f}")
```

**Returns:**
- List of `BreakOfStructure` objects with:
  - `index`: Where the break occurred
  - `price`: Price of the break
  - `direction`: 'bullish' or 'bearish'
  - `broken_level`: The swing level that was broken

**Trading Significance:**
- Bullish BOS = Potential trend reversal to upside
- Bearish BOS = Potential trend reversal to downside

### 3. FairValueGapDetector

Identifies Fair Value Gaps (FVGs) - areas where price moved quickly leaving an imbalance.

```python
from botclave.engine.indicators import FairValueGapDetector

ffg_detector = FairValueGapDetector(min_size_percent=0.02)
ffg = ffg_detector.find_ffg(df)
active_ffg = ffg_detector.get_active_ffg(df)

for gap in active_ffg:
    print(f"{gap.direction} FFG: ${gap.bottom_price:.2f} - ${gap.top_price:.2f}")
```

**Parameters:**
- `min_size_percent`: Minimum gap size as % of range (default: 0.02 = 2%)

**Returns:**
- List of `FairValueGap` objects with:
  - `index`: Starting position
  - `top_price`: Upper bound of the gap
  - `bottom_price`: Lower bound of the gap
  - `direction`: 'bullish' or 'bearish'
  - `size_percent`: Gap size as percentage

**Trading Significance:**
- Bullish FVG = Support zone where price may bounce
- Bearish FVG = Resistance zone where price may reject
- Active (unfilled) FVGs are high-probability entry zones

### 4. ChangeOfCharacterDetector

Detects trend reversals by analyzing swing patterns.

```python
from botclave.engine.indicators import ChangeOfCharacterDetector

choch_detector = ChangeOfCharacterDetector()
choch = choch_detector.find_choch(df, swings)

for ch in choch:
    print(f"CHoCH: {ch.previous_trend} → {ch.new_trend}")
```

**Returns:**
- List of `ChangeOfCharacter` objects with:
  - `previous_trend`: 'uptrend' or 'downtrend'
  - `new_trend`: 'uptrend' or 'downtrend'
  - `trigger_price`: Where the change was confirmed

**Trading Significance:**
- CHoCH = Early warning of trend reversal
- Stronger signal than a single BOS

### 5. OrderBlockDetector

Identifies order blocks - zones where institutions placed large orders.

```python
from botclave.engine.indicators import OrderBlockDetector

ob_detector = OrderBlockDetector()
order_blocks = ob_detector.find_order_blocks(df, bos, volume_threshold=1.5)

for ob in order_blocks:
    if ob.mitigated_index is None:
        print(f"{ob.direction} OB @ ${ob.low_price:.2f}-${ob.high_price:.2f}")
```

**Parameters:**
- `volume_threshold`: Volume multiplier for detection (default: 1.5x average)

**Returns:**
- List of `OrderBlock` objects with:
  - `high_price`, `low_price`: Order block range
  - `direction`: 'bullish' or 'bearish'
  - `mitigated_index`: Where price returned to the block (None if unmitigated)
  - `volume`: Volume of the order block candle

**Trading Significance:**
- Bullish OB = High-probability long entry zone
- Bearish OB = High-probability short entry zone
- Unmitigated OBs are especially powerful

### 6. LiquidityDetector

Detects liquidity clusters - price levels where many traders have orders.

```python
from botclave.engine.indicators import LiquidityDetector

liq_detector = LiquidityDetector(tolerance_percent=0.005)
clusters = liq_detector.find_liquidity_clusters(df, swings)

for cluster in clusters:
    print(f"{cluster.direction} @ ${cluster.price:.2f} (strength: {cluster.strength})")
```

**Parameters:**
- `tolerance_percent`: Price range for grouping (default: 0.005 = 0.5%)

**Returns:**
- List of `LiquidityCluster` objects with:
  - `price`: Cluster price level
  - `strength`: Number of times the level was tested
  - `direction`: 'support' or 'resistance'

**Trading Significance:**
- High-strength clusters = Strong S/R levels
- Price often reacts at these levels

### 7. SMCIndicator (Master Class)

Combines all SMC indicators into a single, easy-to-use interface.

```python
from botclave.engine.indicators import SMCIndicator

smc = SMCIndicator(left_bars=2, right_bars=2, min_ffg_percent=0.02)
analysis = smc.analyze(df)

# Get market bias
bias = smc.get_bias(df)  # 'bullish', 'bearish', or 'neutral'

# Get trade setup
entry = smc.get_entry_level(df, 'long')
stop_loss = smc.get_stop_loss(df, 'long')
take_profits = smc.get_take_profit_levels(df, 'long', count=3)

print(f"Bias: {bias}")
print(f"Entry: ${entry:.2f}")
print(f"SL: ${stop_loss:.2f}")
print(f"TPs: {[f'${tp:.2f}' for tp in take_profits]}")
```

**Methods:**

#### `analyze(df)`
Performs complete SMC analysis.

**Returns:**
```python
{
    'swings': List[Swing],
    'bos': List[BreakOfStructure],
    'ffg': List[FairValueGap],
    'choch': List[ChangeOfCharacter],
    'order_blocks': List[OrderBlock],
    'liquidity': List[LiquidityCluster],
    'last_swing': Swing,
    'last_bos': BreakOfStructure,
    'active_ffg': List[FairValueGap],
    'last_high': float,
    'last_low': float,
}
```

#### `get_bias(df)`
Returns market bias: 'bullish', 'bearish', or 'neutral'

#### `get_entry_level(df, direction)`
Returns recommended entry price for 'long' or 'short'

#### `get_stop_loss(df, direction)`
Returns recommended stop loss price

#### `get_take_profit_levels(df, direction, count=3)`
Returns list of take profit levels

## Helper Functions

### calculate_retracement_levels()

Calculates Fibonacci retracement levels between two prices.

```python
from botclave.engine.indicators import calculate_retracement_levels

levels = calculate_retracement_levels(start_price=50000, end_price=52000)

print(f"0.0%: ${levels.level_0:.2f}")
print(f"23.6%: ${levels.level_236:.2f}")
print(f"38.2%: ${levels.level_382:.2f}")
print(f"50.0%: ${levels.level_500:.2f}")
print(f"61.8%: ${levels.level_618:.2f}")
print(f"78.6%: ${levels.level_786:.2f}")
print(f"100.0%: ${levels.level_1:.2f}")
```

### get_previous_highs_lows()

Gets significant highs/lows from previous periods.

```python
from botclave.engine.indicators import get_previous_highs_lows

levels = get_previous_highs_lows(df, periods=[20, 50, 200])

print(f"Previous highs: {levels['highs']}")
print(f"Previous lows: {levels['lows']}")
```

## Trading Strategies

### Strategy 1: BOS + Order Block Entry

```python
smc = SMCIndicator()
analysis = smc.analyze(df)

# Look for bullish BOS
last_bos = analysis['last_bos']
if last_bos and last_bos.direction == 'bullish':
    # Find unmitigated bullish order blocks
    bullish_obs = [ob for ob in analysis['order_blocks'] 
                   if ob.direction == 'bullish' and ob.mitigated_index is None]
    
    if bullish_obs:
        # Enter at order block high
        entry = bullish_obs[-1].high_price
        stop_loss = bullish_obs[-1].low_price * 0.995  # Below OB
        take_profit = smc.get_take_profit_levels(df, 'long', count=1)[0]
```

### Strategy 2: Fair Value Gap Reversal

```python
smc = SMCIndicator()
analysis = smc.analyze(df)

# Look for active bullish FVGs
active_ffg = [ffg for ffg in analysis['active_ffg'] if ffg.direction == 'bullish']

if active_ffg and smc.get_bias(df) == 'bullish':
    # Enter at bottom of FVG
    entry = active_ffg[-1].bottom_price
    stop_loss = entry * 0.98  # 2% below
    take_profit = smc.get_take_profit_levels(df, 'long', count=1)[0]
```

### Strategy 3: Liquidity Sweep

```python
smc = SMCIndicator()
analysis = smc.analyze(df)

# Find strong support levels
supports = [liq for liq in analysis['liquidity'] 
            if liq.direction == 'support' and liq.strength >= 3]

if supports:
    # Wait for price to sweep below support then reclaim
    support_level = supports[-1].price
    
    # Entry: when price closes back above support
    # Stop: below the sweep low
    # TP: next resistance cluster
```

## Best Practices

1. **Combine Multiple Timeframes**: Use higher timeframes for bias, lower for entries
2. **Respect Order Blocks**: Unmitigated OBs are the highest-probability zones
3. **Wait for Confirmation**: Don't enter on structure break alone
4. **FVG + OB = High Probability**: When an FVG aligns with an OB
5. **Liquidity Grabs**: Price often sweeps liquidity before reversing
6. **Risk Management**: Always use stop losses below/above structure

## Example: Complete Analysis

```python
import pandas as pd
from botclave.engine.indicators import SMCIndicator

# Load your data
df = pd.read_csv('BTC_USDT_15m.csv', index_col=0, parse_dates=True)

# Initialize SMC
smc = SMCIndicator(left_bars=2, right_bars=2)

# Run analysis
analysis = smc.analyze(df)

# Get market context
bias = smc.get_bias(df)
print(f"Market Bias: {bias}")

# Display structure
print(f"\nStructure:")
print(f"- Swings: {len(analysis['swings'])}")
print(f"- BOS: {len(analysis['bos'])}")
print(f"- Active FVGs: {len(analysis['active_ffg'])}")
print(f"- Order Blocks: {len(analysis['order_blocks'])}")
print(f"- CHoCH: {len(analysis['choch'])}")

# Get trade setup
if bias == 'bullish':
    entry = smc.get_entry_level(df, 'long')
    sl = smc.get_stop_loss(df, 'long')
    tps = smc.get_take_profit_levels(df, 'long', count=3)
    
    print(f"\nLong Setup:")
    print(f"Entry: ${entry:.2f}")
    print(f"Stop Loss: ${sl:.2f}")
    print(f"Take Profits: {', '.join([f'${tp:.2f}' for tp in tps])}")
    
    # Calculate R:R
    if entry and sl:
        risk = abs(entry - sl)
        for i, tp in enumerate(tps, 1):
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            print(f"  TP{i} R:R = 1:{rr:.2f}")
```

## Integration with Order Flow

SMC indicators work best when combined with order flow data:

```python
from botclave.engine.indicators import SMCIndicator, OrderFlowIndicators

# SMC for structure
smc = SMCIndicator()
smc_analysis = smc.analyze(df)

# Order flow for confirmation
of_indicators = OrderFlowIndicators()
of_analysis = of_indicators.calculate_all_indicators(df)

# Combined signal
bias = smc.get_bias(df)
cvd = of_analysis['cvd'].iloc[-1]

if bias == 'bullish' and cvd > 0:
    print("Strong bullish confluence!")
    entry = smc.get_entry_level(df, 'long')
    # ... execute trade
```

## References

- **ICT (Inner Circle Trader)**: Original SMC methodology
- **Order Blocks**: Institutional order placement zones
- **Fair Value Gaps**: Price imbalances (also called "imbalance" or "void")
- **BOS vs CHoCH**: BOS is break, CHoCH is confirmed reversal

## Support

For questions or issues with SMC indicators:
- Check the tests: `tests/test_indicators.py`
- Run the demo: `python examples/smc_demo.py`
- Review architecture: `docs/ARCHITECTURE.md`

---

**Last Updated**: 2024-01-15
