"""Unit tests for domain models."""

from datetime import datetime, timedelta
import pytest
import pandas as pd

from botclave.domain.models import (
    Candle,
    DisplacementConfig,
    DisplacementEvent,
    DomainModelsConfig,
    Direction,
    EqualLevel,
    EqualLevelConfig,
    Imbalance,
    ImbalanceConfig,
    ImbalanceType,
    MarketStructureConfig,
    MarketStructureEvent,
    Pivot,
    PivotConfig,
    StructureType,
    Timeframe,
)


class TestTimeframe:
    """Test Timeframe enum."""
    
    def test_timeframe_values(self):
        """Test timeframe enum values."""
        assert Timeframe.SCALPER == "ST"
        assert Timeframe.INTRADAY == "IT"
        assert Timeframe.LONGTERM == "LT"


class TestDirection:
    """Test Direction enum."""
    
    def test_direction_values(self):
        """Test direction enum values."""
        assert Direction.BULLISH == "bullish"
        assert Direction.BEARISH == "bearish"


class TestStructureType:
    """Test StructureType enum."""
    
    def test_structure_type_values(self):
        """Test structure type enum values."""
        assert StructureType.HIGHER_HIGH == "higher_high"
        assert StructureType.LOWER_LOW == "lower_low"
        assert StructureType.HIGHER_LOW == "higher_low"
        assert StructureType.LOWER_HIGH == "lower_high"


class TestImbalanceType:
    """Test ImbalanceType enum."""
    
    def test_imbalance_type_values(self):
        """Test imbalance type enum values."""
        assert ImbalanceType.BUY == "buy"
        assert ImbalanceType.SELL == "sell"


class TestCandle:
    """Test Candle model."""
    
    def test_valid_candle_creation(self):
        """Test creating a valid candle."""
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
        
        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 98.0
        assert candle.close == 103.0
        assert candle.volume == 1000.0
        assert candle.range == 7.0
        assert candle.body == 3.0
        assert candle.body_percentage == pytest.approx(42.857, rel=1e-3)
        assert candle.upper_wick == 2.0
        assert candle.lower_wick == 2.0
        assert candle.direction == Direction.BULLISH
    
    def test_bearish_candle(self):
        """Test bearish candle properties."""
        candle = Candle(
            open=105.0,
            high=106.0,
            low=100.0,
            close=101.0,
            volume=1000.0,
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert candle.direction == Direction.BEARISH
        assert candle.body == 4.0
        assert candle.upper_wick == 1.0
        assert candle.lower_wick == 1.0
    
    def test_doji_candle(self):
        """Test doji candle detection."""
        candle = Candle(
            open=100.0,
            high=102.0,
            low=98.0,
            close=100.03,  # Very close to open (within 1% of range)
            volume=1000.0,
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert candle.is_doji
    
    def test_invalid_price_relationships(self):
        """Test validation of invalid price relationships."""
        with pytest.raises(ValueError, match="High price must be >= open price"):
            Candle(
                open=100.0,
                high=99.0,  # Invalid: high < open
                low=98.0,
                close=101.0,
                volume=1000.0,
                timestamp=datetime.now(),
                symbol="BTC/USDT",
                timeframe="1h",
            )
        
        with pytest.raises(ValueError, match="Low price must be <= open price"):
            Candle(
                open=100.0,
                high=105.0,
                low=101.0,  # Invalid: low > open
                close=99.0,
                volume=1000.0,
                timestamp=datetime.now(),
                symbol="BTC/USDT",
                timeframe="1h",
            )
    
    def test_from_ohlcv_array(self):
        """Test creating candle from OHLCV array."""
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        ohlcv = [timestamp_ms, 100.0, 105.0, 98.0, 103.0, 1000.0]
        
        candle = Candle.from_ohlcv_array(
            ohlcv=ohlcv,
            symbol="BTC/USDT",
            timeframe="1h",
            index=0,
        )
        
        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 98.0
        assert candle.close == 103.0
        assert candle.volume == 1000.0
        assert candle.symbol == "BTC/USDT"
        assert candle.timeframe == "1h"
        assert candle.index == 0
    
    def test_invalid_ohlcv_array(self):
        """Test invalid OHLCV array."""
        with pytest.raises(ValueError, match="OHLCV array must have exactly 6 elements"):
            Candle.from_ohlcv_array(
                ohlcv=[100, 200],  # Too short
                symbol="BTC/USDT",
                timeframe="1h",
            )


class TestPivot:
    """Test Pivot model."""
    
    def test_valid_pivot_creation(self):
        """Test creating a valid pivot."""
        pivot = Pivot(
            price=100.0,
            timestamp=datetime.now(),
            index=10,
            direction=Direction.BEARISH,
            strength=5.0,
            lookback_left=5,
            lookback_right=5,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert pivot.price == 100.0
        assert pivot.direction == Direction.BEARISH
        assert pivot.pivot_type == "swing_high"
    
    def test_pivot_type_property(self):
        """Test pivot type property."""
        high_pivot = Pivot(
            price=100.0,
            timestamp=datetime.now(),
            index=10,
            direction=Direction.BEARISH,
            strength=5.0,
            lookback_left=5,
            lookback_right=5,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        low_pivot = Pivot(
            price=90.0,
            timestamp=datetime.now(),
            index=20,
            direction=Direction.BULLISH,
            strength=3.0,
            lookback_left=5,
            lookback_right=5,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert high_pivot.pivot_type == "swing_high"
        assert low_pivot.pivot_type == "swing_low"
    
    def test_from_series(self):
        """Test creating pivots from price series."""
        # Create a simple price series with clear highs and lows
        prices = pd.Series([100, 102, 105, 103, 101, 98, 96, 99, 102, 104, 103, 100])
        
        # Find swing highs
        high_pivots = Pivot.from_series(
            series=prices,
            direction=Direction.BEARISH,
            lookback_left=2,
            lookback_right=2,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        # Find swing lows
        low_pivots = Pivot.from_series(
            series=prices,
            direction=Direction.BULLISH,
            lookback_left=2,
            lookback_right=2,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert len(high_pivots) > 0
        assert len(low_pivots) > 0
        assert all(p.direction == Direction.BEARISH for p in high_pivots)
        assert all(p.direction == Direction.BULLISH for p in low_pivots)
    
    def test_pivot_comparisons(self):
        """Test pivot comparison methods."""
        pivot1 = Pivot(
            price=100.0,
            timestamp=datetime.now(),
            index=10,
            direction=Direction.BEARISH,
            strength=5.0,
            lookback_left=5,
            lookback_right=5,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        pivot2 = Pivot(
            price=95.0,
            timestamp=datetime.now(),
            index=20,
            direction=Direction.BEARISH,
            strength=3.0,
            lookback_left=5,
            lookback_right=5,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert pivot1.is_higher_than(pivot2)
        assert not pivot1.is_lower_than(pivot2)
        assert pivot2.is_lower_than(pivot1)


class TestMarketStructureEvent:
    """Test MarketStructureEvent model."""
    
    def test_valid_market_structure_creation(self):
        """Test creating a valid market structure event."""
        event = MarketStructureEvent(
            structure_type=StructureType.HIGHER_HIGH,
            price=105.0,
            timestamp=datetime.now(),
            index=15,
            previous_pivot_price=100.0,
            previous_pivot_index=10,
            symbol="BTC/USDT",
            timeframe="1h",
            confidence=0.8,
        )
        
        assert event.structure_type == StructureType.HIGHER_HIGH
        assert event.price == 105.0
        assert event.previous_pivot_price == 100.0
        assert event.is_bullish_structure
        assert not event.is_bearish_structure
    
    def test_structure_type_properties(self):
        """Test structure type properties."""
        bullish_event = MarketStructureEvent(
            structure_type=StructureType.HIGHER_HIGH,
            price=105.0,
            timestamp=datetime.now(),
            index=15,
            previous_pivot_price=100.0,
            previous_pivot_index=10,
            symbol="BTC/USDT",
            timeframe="1h",
            confidence=0.8,
        )
        
        bearish_event = MarketStructureEvent(
            structure_type=StructureType.LOWER_LOW,
            price=95.0,
            timestamp=datetime.now(),
            index=15,
            previous_pivot_price=100.0,
            previous_pivot_index=10,
            symbol="BTC/USDT",
            timeframe="1h",
            confidence=0.8,
        )
        
        assert bullish_event.is_bullish_structure
        assert not bullish_event.is_bearish_structure
        assert bearish_event.is_bearish_structure
        assert not bearish_event.is_bullish_structure
    
    def test_price_differences(self):
        """Test price difference calculations."""
        event = MarketStructureEvent(
            structure_type=StructureType.HIGHER_HIGH,
            price=105.0,
            timestamp=datetime.now(),
            index=15,
            previous_pivot_price=100.0,
            previous_pivot_index=10,
            symbol="BTC/USDT",
            timeframe="1h",
            confidence=0.8,
        )
        
        assert event.price_difference == 5.0
        assert event.price_difference_percentage == 5.0
    
    def test_invalid_structure_consistency(self):
        """Test validation of invalid structure consistency."""
        with pytest.raises(ValueError, match="must be above previous pivot"):
            MarketStructureEvent(
                structure_type=StructureType.HIGHER_HIGH,
                price=95.0,  # Invalid: lower than previous pivot
                timestamp=datetime.now(),
                index=15,
                previous_pivot_price=100.0,
                previous_pivot_index=10,
                symbol="BTC/USDT",
                timeframe="1h",
                confidence=0.8,
            )


class TestImbalance:
    """Test Imbalance model."""
    
    def test_valid_imbalance_creation(self):
        """Test creating a valid imbalance."""
        imbalance = Imbalance(
            top=102.0,
            bottom=98.0,
            timestamp=datetime.now(),
            index=10,
            imbalance_type=ImbalanceType.BUY,
            size=4.0,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert imbalance.top == 102.0
        assert imbalance.bottom == 98.0
        assert imbalance.size == 4.0
        assert imbalance.midpoint == 100.0
        assert imbalance.range_percentage == 4.0
    
    def test_invalid_price_relationships(self):
        """Test validation of invalid price relationships."""
        with pytest.raises(ValueError, match="Top price must be greater than bottom price"):
            Imbalance(
                top=98.0,  # Invalid: top <= bottom
                bottom=100.0,
                timestamp=datetime.now(),
                index=10,
                imbalance_type=ImbalanceType.BUY,
                size=2.0,
                symbol="BTC/USDT",
                timeframe="1h",
            )
    
    def test_price_in_imbalance(self):
        """Test checking if price is in imbalance."""
        imbalance = Imbalance(
            top=102.0,
            bottom=98.0,
            timestamp=datetime.now(),
            index=10,
            imbalance_type=ImbalanceType.BUY,
            size=4.0,
            symbol="BTC/USDT",
            timeframe="1h",
            tolerance_pips=10,  # 10 pips tolerance
        )
        
        assert imbalance.is_price_in_imbalance(100.0)  # Inside
        assert imbalance.is_price_in_imbalance(101.999)  # Inside with tolerance
        assert imbalance.is_price_in_imbalance(98.001)  # Inside with tolerance
        assert not imbalance.is_price_in_imbalance(103.0)  # Outside
        assert not imbalance.is_price_in_imbalance(97.0)  # Outside
    
    def test_fill_checking(self):
        """Test fill checking and updating."""
        imbalance = Imbalance(
            top=102.0,
            bottom=98.0,
            timestamp=datetime.now(),
            index=10,
            imbalance_type=ImbalanceType.BUY,
            size=4.0,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        # Check fill with price outside range
        assert not imbalance.check_if_filled(105.0)
        assert not imbalance.is_filled
        
        # Check fill with price inside range
        fill_time = datetime.now()
        assert imbalance.check_if_filled(100.0, fill_time)
        assert imbalance.is_filled
        assert imbalance.is_mitigated
        assert imbalance.fill_timestamp == fill_time
        assert imbalance.fill_price == 100.0
    
    def test_from_three_candles(self):
        """Test creating imbalance from three candles."""
        # Test buy imbalance (strong upward move)
        buy_imbalance = Imbalance.from_three_candles(
            candle1_high=100.0,
            candle1_low=99.0,
            candle2_high=103.0,
            candle2_low=101.0,  # Higher than candle1 high
            candle3_high=105.0,
            candle3_low=102.0,  # Higher than candle2 low
            timestamp=datetime.now(),
            index=2,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert buy_imbalance is not None
        assert buy_imbalance.imbalance_type == ImbalanceType.BUY
        assert buy_imbalance.bottom == 100.0
        assert buy_imbalance.top == 101.0
        
        # Test sell imbalance (strong downward move)
        sell_imbalance = Imbalance.from_three_candles(
            candle1_high=100.0,
            candle1_low=99.0,
            candle2_high=98.0,
            candle2_low=97.0,  # Lower than candle1 low
            candle3_high=96.0,
            candle3_low=95.0,  # Lower than candle2 high
            timestamp=datetime.now(),
            index=2,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert sell_imbalance is not None
        assert sell_imbalance.imbalance_type == ImbalanceType.SELL
        assert sell_imbalance.bottom == 98.0
        assert sell_imbalance.top == 99.0
        
        # Test no imbalance
        no_imbalance = Imbalance.from_three_candles(
            candle1_high=100.0,
            candle1_low=99.0,
            candle2_high=101.0,
            candle2_low=100.0,  # Not creating imbalance
            candle3_high=102.0,
            candle3_low=101.0,
            timestamp=datetime.now(),
            index=2,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert no_imbalance is None


class TestEqualLevel:
    """Test EqualLevel model."""
    
    def test_valid_equal_level_creation(self):
        """Test creating a valid equal level."""
        level = EqualLevel(
            price=100.0,
            timestamp=datetime.now(),
            indices=[5, 15, 25],
            touches=3,
            tolerance=0.5,
            level_type="high",
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert level.price == 100.0
        assert level.touches == 3
        assert level.first_touch_index == 5
        assert level.last_touch_index == 25
        assert level.age_in_bars == 20
    
    def test_price_at_level(self):
        """Test checking if price is at level."""
        level = EqualLevel(
            price=100.0,
            timestamp=datetime.now(),
            indices=[5, 15, 25],
            touches=3,
            tolerance=0.5,
            level_type="high",
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert level.is_price_at_level(100.0)  # Exact
        assert level.is_price_at_level(100.3)  # Within tolerance
        assert level.is_price_at_level(99.8)  # Within tolerance
        assert not level.is_price_at_level(101.0)  # Outside tolerance
        assert not level.is_price_at_level(99.0)  # Outside tolerance
    
    def test_add_touch(self):
        """Test adding a new touch to equal level."""
        level = EqualLevel(
            price=100.0,
            timestamp=datetime.now(),
            indices=[5, 15, 25],
            touches=3,
            tolerance=0.5,
            level_type="high",
            symbol="BTC/USDT",
            timeframe="1h",
            volume_at_touches=[1000.0, 1200.0, 1100.0],  # Initialize with existing data
            reaction_strength=[0.5, 0.7, 0.6],
        )
        
        original_touches = level.touches
        new_timestamp = datetime.now()
        
        level.add_touch(
            index=35,
            timestamp=new_timestamp,
            volume=1500.0,
            reaction_strength=0.8,
        )
        
        assert level.touches == original_touches + 1
        assert len(level.indices) == 4
        assert 35 in level.indices
        assert level.timestamp == new_timestamp
        assert len(level.volume_at_touches) == 4
        assert level.volume_at_touches[-1] == 1500.0
        assert len(level.reaction_strength) == 4
        assert level.reaction_strength[-1] == 0.8
    
    def test_find_equal_levels(self):
        """Test finding equal levels from price series."""
        prices = [100.0, 100.2, 99.8, 100.1, 100.0, 99.9, 100.3]
        indices = list(range(len(prices)))
        timestamps = [datetime.now() + timedelta(minutes=i) for i in indices]
        
        levels = EqualLevel.find_equal_levels(
            prices=prices,
            indices=indices,
            timestamps=timestamps,
            symbol="BTC/USDT",
            timeframe="1h",
            tolerance=0.5,
            level_type="high",
            min_touches=2,
        )
        
        assert len(levels) > 0
        assert all(level.touches >= 2 for level in levels)
        assert all(level.level_type == "high" for level in levels)
    
    def test_invalid_inputs(self):
        """Test validation of invalid inputs."""
        with pytest.raises(ValueError, match="Prices, indices, and timestamps must have same length"):
            EqualLevel.find_equal_levels(
                prices=[100.0, 101.0],
                indices=[0],  # Different length
                timestamps=[datetime.now(), datetime.now()],
                symbol="BTC/USDT",
                timeframe="1h",
                tolerance=0.5,
                level_type="high",
            )


class TestDisplacementEvent:
    """Test DisplacementEvent model."""
    
    def test_valid_displacement_creation(self):
        """Test creating a valid displacement event."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=5)
        
        displacement = DisplacementEvent(
            start_price=100.0,
            end_price=105.0,
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_index=10,
            end_index=15,
            direction=Direction.BULLISH,
            magnitude=5.0,
            duration_bars=6,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        assert displacement.direction == Direction.BULLISH
        assert displacement.midpoint_price == 102.5
        assert displacement.percentage_change == 5.0
        assert displacement.duration_seconds == 5 * 3600
    
    def test_invalid_displacement_consistency(self):
        """Test validation of invalid displacement consistency."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=5)
        
        # Test end timestamp before start timestamp
        with pytest.raises(ValueError, match="End timestamp must be after start timestamp"):
            DisplacementEvent(
                start_price=100.0,
                end_price=105.0,
                start_timestamp=end_time,  # Invalid: after end
                end_timestamp=start_time,  # Invalid: before start
                start_index=10,
                end_index=15,
                direction=Direction.BULLISH,
                magnitude=5.0,
                duration_bars=6,
                symbol="BTC/USDT",
                timeframe="1h",
            )
        
        # Test bearish displacement with wrong price relationship
        with pytest.raises(ValueError, match="Bearish displacement must have end_price < start_price"):
            DisplacementEvent(
                start_price=100.0,
                end_price=105.0,  # Invalid: higher than start for bearish
                start_timestamp=start_time,
                end_timestamp=end_time,
                start_index=10,
                end_index=15,
                direction=Direction.BEARISH,
                magnitude=5.0,
                duration_bars=6,
                symbol="BTC/USDT",
                timeframe="1h",
            )
    
    def test_fibonacci_levels(self):
        """Test Fibonacci level calculations."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=5)
        
        displacement = DisplacementEvent(
            start_price=100.0,
            end_price=105.0,
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_index=10,
            end_index=15,
            direction=Direction.BULLISH,
            magnitude=5.0,
            duration_bars=6,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        # Test Fibonacci levels for bullish displacement
        assert displacement.get_fibonacci_level(0.0) == 105.0  # 0% = end price
        assert displacement.get_fibonacci_level(0.5) == 102.5  # 50% = midpoint
        assert displacement.get_fibonacci_level(1.0) == 100.0  # 100% = start price
        
        # Test invalid Fibonacci level
        with pytest.raises(ValueError, match="Fibonacci level must be between 0 and 1"):
            displacement.get_fibonacci_level(1.5)
    
    def test_price_in_displacement(self):
        """Test checking if price is in displacement range."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=5)
        
        bullish_displacement = DisplacementEvent(
            start_price=100.0,
            end_price=105.0,
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_index=10,
            end_index=15,
            direction=Direction.BULLISH,
            magnitude=5.0,
            duration_bars=6,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        bearish_displacement = DisplacementEvent(
            start_price=105.0,
            end_price=100.0,
            start_timestamp=start_time,
            end_timestamp=end_time,
            start_index=10,
            end_index=15,
            direction=Direction.BEARISH,
            magnitude=5.0,
            duration_bars=6,
            symbol="BTC/USDT",
            timeframe="1h",
        )
        
        # Test bullish displacement
        assert bullish_displacement.is_price_in_displacement(102.5)  # Inside
        assert bullish_displacement.is_price_in_displacement(100.0)  # At start
        assert bullish_displacement.is_price_in_displacement(105.0)  # At end
        assert not bullish_displacement.is_price_in_displacement(99.0)  # Below
        assert not bullish_displacement.is_price_in_displacement(106.0)  # Above
        
        # Test bearish displacement
        assert bearish_displacement.is_price_in_displacement(102.5)  # Inside
        assert bearish_displacement.is_price_in_displacement(105.0)  # At start
        assert bearish_displacement.is_price_in_displacement(100.0)  # At end
        assert not bearish_displacement.is_price_in_displacement(106.0)  # Above
        assert not bearish_displacement.is_price_in_displacement(99.0)  # Below
    
    def test_from_price_series(self):
        """Test creating displacement from price series data."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=5)
        
        displacement = DisplacementEvent.from_price_series(
            start_index=10,
            end_index=15,
            start_price=100.0,
            end_price=105.0,
            start_timestamp=start_time,
            end_timestamp=end_time,
            symbol="BTC/USDT",
            timeframe="1h",
            entry_volume=1000.0,
            peak_volume=2000.0,
            exit_volume=1500.0,
        )
        
        assert displacement.direction == Direction.BULLISH
        assert displacement.magnitude == 5.0
        assert displacement.duration_bars == 6
        assert displacement.velocity == pytest.approx(0.833, rel=1e-3)
        assert displacement.entry_volume == 1000.0
        assert displacement.peak_volume == 2000.0
        assert displacement.exit_volume == 1500.0


class TestDomainModelsConfig:
    """Test DomainModelsConfig model."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = DomainModelsConfig()
        
        assert config.pivot.lookback_left == 5
        assert config.pivot.lookback_right == 5
        assert config.imbalance.min_size_pips == 5
        assert config.equal_level.min_touches == 2
        assert config.displacement.min_bars == 3
        assert config.market_structure.min_confidence == 0.7
        assert config.symbol == "BTC/USDT"
        assert config.timeframe == "1h"
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = DomainModelsConfig(
            pivot=PivotConfig(lookback_left=10, lookback_right=10, min_strength=0.2),
            imbalance=ImbalanceConfig(min_size_pips=10, tolerance_pips=5),
            symbol="ETH/USDT",
            timeframe="4h",
        )
        
        assert config.pivot.lookback_left == 10
        assert config.pivot.lookback_right == 10
        assert config.pivot.min_strength == 0.2
        assert config.imbalance.min_size_pips == 10
        assert config.imbalance.tolerance_pips == 5
        assert config.symbol == "ETH/USDT"
        assert config.timeframe == "4h"
    
    def test_get_pip_value(self):
        """Test pip value calculation."""
        config = DomainModelsConfig()
        
        # Test JPY pair
        jpy_pip = config.get_pip_value("USD/JPY")
        assert jpy_pip == 0.01
        
        # Test standard forex pair
        forex_pip = config.get_pip_value("EUR/USD")
        assert forex_pip == 0.0001
        
        # Test crypto (default)
        crypto_pip = config.get_pip_value("BTC/USDT")
        assert crypto_pip == 0.0001
    
    def test_tolerance_to_price(self):
        """Test converting tolerance pips to price."""
        config = DomainModelsConfig()
        
        # Test forex pair
        forex_tolerance = config.tolerance_to_price(10, "EUR/USD")
        assert forex_tolerance == 0.001  # 10 * 0.0001
        
        # Test JPY pair
        jpy_tolerance = config.tolerance_to_price(5, "USD/JPY")
        assert jpy_tolerance == 0.05  # 5 * 0.01
        
        # Test crypto
        crypto_tolerance = config.tolerance_to_price(50, "BTC/USDT")
        assert crypto_tolerance == 0.005  # 50 * 0.0001


class TestPivotConfig:
    """Test PivotConfig model."""
    
    def test_valid_pivot_config(self):
        """Test creating valid pivot config."""
        config = PivotConfig(
            lookback_left=8,
            lookback_right=8,
            min_strength=0.15,
            strength_calculation="absolute",
        )
        
        assert config.lookback_left == 8
        assert config.lookback_right == 8
        assert config.min_strength == 0.15
        assert config.strength_calculation == "absolute"
    
    def test_invalid_pivot_config(self):
        """Test invalid pivot config validation."""
        with pytest.raises(ValueError):
            PivotConfig(lookback_left=0)  # Must be >= 1
        
        with pytest.raises(ValueError):
            PivotConfig(lookback_right=-1)  # Must be >= 1
        
        with pytest.raises(ValueError):
            PivotConfig(min_strength=-0.1)  # Must be >= 0


class TestImbalanceConfig:
    """Test ImbalanceConfig model."""
    
    def test_valid_imbalance_config(self):
        """Test creating valid imbalance config."""
        config = ImbalanceConfig(
            min_size_pips=10,
            tolerance_pips=3,
            max_age_bars=200,
            require_volume_confirmation=False,
        )
        
        assert config.min_size_pips == 10
        assert config.tolerance_pips == 3
        assert config.max_age_bars == 200
        assert config.require_volume_confirmation is False
    
    def test_invalid_imbalance_config(self):
        """Test invalid imbalance config validation."""
        with pytest.raises(ValueError):
            ImbalanceConfig(min_size_pips=-1)  # Must be >= 0
        
        with pytest.raises(ValueError):
            ImbalanceConfig(tolerance_pips=-5)  # Must be >= 0
        
        with pytest.raises(ValueError):
            ImbalanceConfig(max_age_bars=0)  # Must be >= 1


class TestEqualLevelConfig:
    """Test EqualLevelConfig model."""
    
    def test_valid_equal_level_config(self):
        """Test creating valid equal level config."""
        config = EqualLevelConfig(
            tolerance_pips=15,
            min_touches=3,
            max_age_bars=300,
            min_reaction_strength=0.8,
        )
        
        assert config.tolerance_pips == 15
        assert config.min_touches == 3
        assert config.max_age_bars == 300
        assert config.min_reaction_strength == 0.8
    
    def test_invalid_equal_level_config(self):
        """Test invalid equal level config validation."""
        with pytest.raises(ValueError):
            EqualLevelConfig(tolerance_pips=-1)  # Must be >= 0
        
        with pytest.raises(ValueError):
            EqualLevelConfig(min_touches=1)  # Must be >= 2
        
        with pytest.raises(ValueError):
            EqualLevelConfig(min_reaction_strength=-0.1)  # Must be >= 0


class TestDisplacementConfig:
    """Test DisplacementConfig model."""
    
    def test_valid_displacement_config(self):
        """Test creating valid displacement config."""
        config = DisplacementConfig(
            min_bars=5,
            min_velocity=0.002,
            min_percentage_change=1.0,
            volume_confirmation=False,
        )
        
        assert config.min_bars == 5
        assert config.min_velocity == 0.002
        assert config.min_percentage_change == 1.0
        assert config.volume_confirmation is False
    
    def test_invalid_displacement_config(self):
        """Test invalid displacement config validation."""
        with pytest.raises(ValueError):
            DisplacementConfig(min_bars=0)  # Must be >= 1
        
        with pytest.raises(ValueError):
            DisplacementConfig(min_velocity=-0.001)  # Must be >= 0
        
        with pytest.raises(ValueError):
            DisplacementConfig(min_percentage_change=-1.0)  # Must be >= 0


class TestMarketStructureConfig:
    """Test MarketStructureConfig model."""
    
    def test_valid_market_structure_config(self):
        """Test creating valid market structure config."""
        config = MarketStructureConfig(
            confirmation_bars=3,
            min_confidence=0.8,
            volume_threshold=2.0,
        )
        
        assert config.confirmation_bars == 3
        assert config.min_confidence == 0.8
        assert config.volume_threshold == 2.0
    
    def test_invalid_market_structure_config(self):
        """Test invalid market structure config validation."""
        with pytest.raises(ValueError):
            MarketStructureConfig(confirmation_bars=-1)  # Must be >= 0
        
        with pytest.raises(ValueError):
            MarketStructureConfig(min_confidence=-0.1)  # Must be >= 0
        
        with pytest.raises(ValueError):
            MarketStructureConfig(min_confidence=1.1)  # Must be <= 1
        
        with pytest.raises(ValueError):
            MarketStructureConfig(volume_threshold=0.5)  # Must be >= 1