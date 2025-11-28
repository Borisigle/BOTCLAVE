"""Integration test for domain models with configuration system."""

from datetime import datetime
import pytest

from botclave.config.manager import Config
from botclave.domain.models import (
    Candle,
    DomainModelsConfig,
    Imbalance,
    Pivot,
)


class TestDomainModelsIntegration:
    """Test integration of domain models with configuration system."""
    
    def test_config_integration(self):
        """Test that domain models work with existing config system."""
        # Create a config manager
        config_manager = Config()
        
        # Create domain models config
        domain_config = DomainModelsConfig(
            symbol=config_manager.get("trading.symbols", ["BTC/USDT"])[0],
            timeframe=config_manager.get("trading.timeframes", ["1h"])[0],
        )
        
        assert domain_config.symbol in config_manager.get("trading.symbols", [])
        assert domain_config.timeframe in config_manager.get("trading.timeframes", [])
    
    def test_candle_with_config(self):
        """Test creating candle with configuration values."""
        config = Config()
        symbol = config.get("trading.symbols", ["BTC/USDT"])[0]
        timeframe = config.get("trading.timeframes", ["1h"])[0]
        
        candle = Candle(
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
        )
        
        assert candle.symbol == symbol
        assert candle.timeframe == timeframe
    
    def test_domain_config_from_manager(self):
        """Test creating domain config from config manager values."""
        config_manager = Config()
        
        # Get values from config manager
        symbol = config_manager.get("trading.symbols", ["BTC/USDT"])[0]
        default_timeframe = config_manager.get("data.default_timeframe", "1h")
        
        # Create domain config
        domain_config = DomainModelsConfig(
            symbol=symbol,
            timeframe=default_timeframe,
        )
        
        assert domain_config.symbol == symbol
        assert domain_config.timeframe == default_timeframe
    
    def test_serialization_compatibility(self):
        """Test that domain models serialize properly for API usage."""
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
        
        # Test dict serialization
        candle_dict = candle.model_dump()
        assert isinstance(candle_dict, dict)
        assert "open" in candle_dict
        assert "close" in candle_dict
        assert "direction" in candle_dict
        
        # Test JSON serialization
        candle_json = candle.model_dump_json()
        assert isinstance(candle_json, str)
        assert "open" in candle_json
        assert "close" in candle_json
    
    def test_metadata_fields(self):
        """Test that models include required metadata fields."""
        candle = Candle(
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            timeframe="1h",
            index=5,
        )
        
        # Check metadata fields exist
        assert hasattr(candle, 'index')
        assert hasattr(candle, 'symbol')
        assert hasattr(candle, 'timeframe')
        assert hasattr(candle, 'timestamp')
        assert hasattr(candle, 'is_valid')
        
        # Check values
        assert candle.index == 5
        assert candle.symbol == "BTC/USDT"
        assert candle.timeframe == "1h"
        assert candle.is_valid is True