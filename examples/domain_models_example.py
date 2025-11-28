"""Example usage of domain models."""

from datetime import datetime, timedelta
import pandas as pd

from botclave.domain.models import (
    Candle,
    DisplacementEvent,
    DomainModelsConfig,
    EqualLevel,
    EqualLevelConfig,
    Imbalance,
    ImbalanceConfig,
    ImbalanceType,
    MarketStructureEvent,
    Pivot,
    StructureType,
    Timeframe,
    Direction,
)


def main():
    """Demonstrate domain models usage."""
    
    # Create configuration
    config = DomainModelsConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        imbalance=ImbalanceConfig(min_size_pips=10, tolerance_pips=5),
        equal_level=EqualLevelConfig(min_touches=3, tolerance_pips=15),
    )
    
    print(f"Configuration for {config.symbol} on {config.timeframe}")
    print(f"Pip value: {config.get_pip_value()}")
    print(f"Tolerance in price units: {config.tolerance_to_price(10)}")
    print()
    
    # Create candles from OHLCV data
    ohlcv_data = [
        [int((datetime.now() - timedelta(hours=i)).timestamp() * 1000), 
         100.0 + i, 100.5 + i, 99.5 + i, 100.2 + i, 1000.0 + i * 100]
        for i in range(10, 0, -1)
    ]
    
    candles = []
    for i, ohlcv in enumerate(ohlcv_data):
        candle = Candle.from_ohlcv_array(
            ohlcv=ohlcv,
            symbol=config.symbol,
            timeframe=config.timeframe,
            index=i,
        )
        candles.append(candle)
    
    print(f"Created {len(candles)} candles")
    print(f"First candle: {candles[0].direction}, body%: {candles[0].body_percentage:.2f}")
    print()
    
    # Find pivots from candle data
    high_prices = pd.Series([c.high for c in candles])
    low_prices = pd.Series([c.low for c in candles])
    
    swing_highs = Pivot.from_series(
        series=high_prices,
        direction=Direction.BEARISH,
        lookback_left=config.pivot.lookback_left,
        lookback_right=config.pivot.lookback_right,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )
    
    swing_lows = Pivot.from_series(
        series=low_prices,
        direction=Direction.BULLISH,
        lookback_left=config.pivot.lookback_left,
        lookback_right=config.pivot.lookback_right,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )
    
    print(f"Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
    if swing_highs:
        print(f"Highest swing high: {max(swing_highs, key=lambda p: p.price).price}")
    if swing_lows:
        print(f"Lowest swing low: {min(swing_lows, key=lambda p: p.price).price}")
    print()
    
    # Create market structure events
    if swing_highs and len(swing_highs) >= 2:
        prev_high = swing_highs[-2]
        current_high = swing_highs[-1]
        
        if current_high.price > prev_high.price:
            structure_event = MarketStructureEvent(
                structure_type=StructureType.HIGHER_HIGH,
                price=current_high.price,
                timestamp=current_high.timestamp,
                index=current_high.index,
                previous_pivot_price=prev_high.price,
                previous_pivot_index=prev_high.index,
                symbol=config.symbol,
                timeframe=config.timeframe,
                confidence=0.85,
            )
            
            print(f"Market structure event: {structure_event.structure_type}")
            print(f"Price difference: {structure_event.price_difference_percentage:.2f}%")
            print()
    
    # Find imbalances from three consecutive candles
    for i in range(len(candles) - 2):
        imbalance = Imbalance.from_three_candles(
            candle1_high=candles[i].high,
            candle1_low=candles[i].low,
            candle2_high=candles[i + 1].high,
            candle2_low=candles[i + 1].low,
            candle3_high=candles[i + 2].high,
            candle3_low=candles[i + 2].low,
            timestamp=candles[i + 1].timestamp,
            index=i + 1,
            symbol=config.symbol,
            timeframe=config.timeframe,
            tolerance_pips=config.imbalance.tolerance_pips,
        )
        
        if imbalance:
            print(f"Found {imbalance.imbalance_type} imbalance")
            print(f"Size: {imbalance.size}, range%: {imbalance.range_percentage:.2f}%")
            break
    else:
        print("No imbalances found in sample data")
    print()
    
    # Find equal levels from price data
    high_prices_list = [c.high for c in candles]
    high_indices = list(range(len(candles)))
    high_timestamps = [c.timestamp for c in candles]
    
    equal_highs = EqualLevel.find_equal_levels(
        prices=high_prices_list,
        indices=high_indices,
        timestamps=high_timestamps,
        symbol=config.symbol,
        timeframe=config.timeframe,
        tolerance=config.tolerance_to_price(config.equal_level.tolerance_pips),
        level_type="high",
        min_touches=config.equal_level.min_touches,
    )
    
    print(f"Found {len(equal_highs)} equal high levels")
    for level in equal_highs:
        print(f"Level at {level.price}: {level.touches} touches")
    print()
    
    # Create displacement event
    if len(candles) >= 5:
        displacement = DisplacementEvent.from_price_series(
            start_index=0,
            end_index=4,
            start_price=candles[0].close,
            end_price=candles[4].close,
            start_timestamp=candles[0].timestamp,
            end_timestamp=candles[4].timestamp,
            symbol=config.symbol,
            timeframe=config.timeframe,
            entry_volume=candles[0].volume,
            peak_volume=max(c.volume for c in candles[:5]),
            exit_volume=candles[4].volume,
        )
        
        print(f"Displacement: {displacement.direction}")
        print(f"Magnitude: {displacement.magnitude:.2f} ({displacement.percentage_change:.2f}%)")
        print(f"Velocity: {displacement.velocity:.4f} per bar")
        print(f"Duration: {displacement.duration_bars} bars")
        
        # Calculate Fibonacci levels
        fib_50 = displacement.get_fibonacci_level(0.5)
        print(f"50% Fibonacci level: {fib_50:.2f}")
    
    # Demonstrate serialization
    print("\n" + "="*50)
    print("SERIALIZATION EXAMPLE")
    print("="*50)
    
    sample_candle = candles[0]
    print("Candle as dict:")
    print(sample_candle.model_dump())
    
    print("\nCandle as JSON:")
    print(sample_candle.model_dump_json(indent=2))
    
    # Show validation
    print("\n" + "="*50)
    print("VALIDATION EXAMPLE")
    print("="*50)
    
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
        print(f"Validation caught error: {e}")


if __name__ == "__main__":
    main()