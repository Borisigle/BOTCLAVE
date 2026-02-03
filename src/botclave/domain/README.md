# Domain Models

This module provides strongly typed Pydantic models for ICT Order Flow cryptocurrency trading analysis.

## Overview

The domain models module contains the following core components:

### Models

- **Candle**: OHLCV candlestick data with computed properties (range, body, wicks, direction, doji detection)
- **Pivot**: Swing high and low pivot points with strength calculations
- **MarketStructureEvent**: Market structure breaks (higher highs, lower lows, etc.)
- **Imbalance**: Price imbalances (Fair Value Gaps) with fill tracking
- **EqualLevel**: Price levels with multiple touches
- **DisplacementEvent**: Strong directional price movements with Fibonacci levels

### Enums

- **Timeframe**: SCALPER (ST), INTRADAY (IT), LONGTERM (LT)
- **Direction**: BULLISH, BEARISH
- **StructureType**: HIGHER_HIGH, LOWER_LOW, HIGHER_LOW, LOWER_HIGH
- **ImbalanceType**: BUY, SELL

### Configuration

- **DomainModelsConfig**: Central configuration with threshold settings
- **PivotConfig**: Pivot detection configuration
- **ImbalanceConfig**: Imbalance detection configuration
- **EqualLevelConfig**: Equal level detection configuration
- **DisplacementConfig**: Displacement detection configuration
- **MarketStructureConfig**: Market structure analysis configuration

## Key Features

### Strong Typing & Validation
All models use Pydantic for runtime validation and type safety:

```python
from botclave.domain.models import Candle

candle = Candle(
    open=100.0,
    high=105.0,
    low=98.0,
    close=103.0,
    volume=1000.0,
    timestamp=datetime.now(),
    symbol="BTC/USDT",
    timeframe="1h",
)

# Invalid data raises validation errors
try:
    invalid_candle = Candle(
        open=100.0,
        high=99.0,  # Invalid: high < open
        low=98.0,
        close=101.0,
        volume=1000.0,
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        timeframe="1h",
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Computed Properties
Models include useful computed attributes:

```python
# Candle properties
print(f"Range: {candle.range}")  # high - low
print(f"Body: {candle.body}")  # abs(close - open)
print(f"Body %: {candle.body_percentage}")  # body / range * 100
print(f"Direction: {candle.direction}")  # BULLISH or BEARISH
print(f"Is doji: {candle.is_doji}")  # open ≈ close

# Pivot properties
print(f"Pivot type: {pivot.pivot_type}")  # swing_high or swing_low
print(f"Is higher: {pivot.is_higher_than(other_pivot)}")

# Imbalance properties
print(f"Midpoint: {imbalance.midpoint}")
print(f"Range %: {imbalance.range_percentage}")
print(f"Is filled: {imbalance.is_filled}")

# Displacement properties
print(f"Midpoint: {displacement.midpoint_price}")
print(f"% change: {displacement.percentage_change}")
print(f"Velocity: {displacement.velocity}")
print(f"Fib 50%: {displacement.get_fibonacci_level(0.5)}")
```

### Helper Constructors
Convenient factory methods for common operations:

```python
# From OHLCV array (common from exchange APIs)
ohlcv = [timestamp_ms, open, high, low, close, volume]
candle = Candle.from_ohlcv_array(
    ohlcv=ohlcv,
    symbol="BTC/USDT",
    timeframe="1h",
    index=0,
)

# Pivots from price series
import pandas as pd
prices = pd.Series([100, 102, 105, 103, 101, 98, 96])
swing_highs = Pivot.from_series(
    series=prices,
    direction=Direction.BEARISH,
    lookback_left=5,
    lookback_right=5,
    symbol="BTC/USDT",
    timeframe="1h",
)

# Imbalances from three candles
imbalance = Imbalance.from_three_candles(
    candle1_high=100.0,
    candle1_low=99.0,
    candle2_high=103.0,
    candle2_low=101.0,  # Creates buy imbalance
    candle3_high=105.0,
    candle3_low=102.0,
    timestamp=datetime.now(),
    index=2,
    symbol="BTC/USDT",
    timeframe="1h",
)

# Displacement from price series
displacement = DisplacementEvent.from_price_series(
    start_index=0,
    end_index=5,
    start_price=100.0,
    end_price=105.0,
    start_timestamp=start_time,
    end_timestamp=end_time,
    symbol="BTC/USDT",
    timeframe="1h",
)

# Equal levels from price list
levels = EqualLevel.find_equal_levels(
    prices=[100.0, 100.1, 99.9, 100.05],
    indices=[0, 1, 2, 3],
    timestamps=[datetime.now()] * 4,
    symbol="BTC/USDT",
    timeframe="1h",
    tolerance=0.5,
    level_type="high",
    min_touches=2,
)
```

### Configuration Integration
Models integrate with configuration thresholds:

```python
# Create configuration
config = DomainModelsConfig(
    symbol="BTC/USDT",
    timeframe="1h",
    pivot=PivotConfig(lookback_left=10, lookback_right=10),
    imbalance=ImbalanceConfig(min_size_pips=10, tolerance_pips=5),
)

# Use pip conversion utilities
pip_value = config.get_pip_value("BTC/USDT")
tolerance_price = config.tolerance_to_price(10, "BTC/USDT")

print(f"Pip value: {pip_value}")
print(f"Tolerance in price units: {tolerance_price}")
```

### Serialization
All models support serialization to dict and JSON for API usage:

```python
# To dictionary
candle_dict = candle.model_dump()

# To JSON string
candle_json = candle.model_dump_json(indent=2)

# Both include computed properties
print(f"Direction: {candle_dict['direction']}")
print(f"Body %: {candle_dict['body_percentage']}")
```

## Validation Examples

The models include comprehensive validation:

### Candle Validation
- High must be >= open and close
- Low must be <= open and close
- All prices must be positive
- Volume must be non-negative

### Pivot Validation
- Lookback periods must be positive
- Strength must be non-negative

### Imbalance Validation
- Top must be > bottom
- Size must match price difference
- Type must be consistent with price context

### Market Structure Validation
- Price relationships must match structure type
- Confidence must be 0-1
- Temporal consistency

### Displacement Validation
- End timestamp must be after start
- Direction must match price relationship
- Magnitude must be calculated correctly

## Integration with Existing Code

The domain models integrate seamlessly with the existing configuration system:

```python
from botclave.config.manager import Config
from botclave.domain.models import DomainModelsConfig, Candle

# Use existing config
config_manager = Config()

# Create domain models config
domain_config = DomainModelsConfig(
    symbol=config_manager.get("trading.symbols", ["BTC/USDT"])[0],
    timeframe=config_manager.get("data.default_timeframe", "1h"),
)

# Create candle with config values
candle = Candle(
    open=100.0,
    high=105.0,
    low=98.0,
    close=103.0,
    volume=1000.0,
    timestamp=datetime.now(),
    symbol=domain_config.symbol,
    timeframe=domain_config.timeframe,
)
```

## Testing

Comprehensive unit tests ensure model validation and behavior:

```bash
# Run domain models tests
python -m pytest tests/unit/test_domain_models.py -v

# Run integration tests
python -m pytest tests/unit/test_domain_models_integration.py -v
```

## Examples

See `examples/domain_models_example.py` for a complete usage example demonstrating all models.