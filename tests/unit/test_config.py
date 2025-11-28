"""Unit tests for configuration module."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from botclave.config.manager import Config


class TestConfig:
    """Test cases for Config class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_init_with_default_config(self):
        """Test initialization with default config when no file exists."""
        config = Config()

        # Should have default values
        assert config.get("data.default_exchange") == "binance"
        assert config.get("data.default_timeframe") == "4h"
        assert config.get("data.default_limit") == 1500

    def test_init_with_yaml_file(self):
        """Test initialization with YAML config file."""
        config_data = {
            "data": {"default_exchange": "coinbase", "custom_setting": "test_value"}
        }

        config_file = self.temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = Config(config_file)

        assert config.get("data.default_exchange") == "coinbase"
        assert config.get("data.custom_setting") == "test_value"
        # Should still have other defaults
        assert config.get("data.default_timeframe") == "4h"

    def test_init_with_json_file(self):
        """Test initialization with JSON config file."""
        config_data = {"data": {"default_exchange": "kraken", "default_limit": 500}}

        config_file = self.temp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = Config(config_file)

        assert config.get("data.default_exchange") == "kraken"
        assert config.get("data.default_limit") == 500

    def test_get_with_dot_notation(self):
        """Test getting values with dot notation."""
        config = Config()

        assert config.get("data.default_exchange") == "binance"
        assert config.get("indicators.ema_periods") == [20, 50, 200]
        assert config.get("api.port") == 8000

    def test_get_with_default(self):
        """Test getting values with default fallback."""
        config = Config()

        # Existing key
        assert config.get("data.default_exchange", "fallback") == "binance"

        # Non-existent key with default
        assert config.get("nonexistent.key", "fallback") == "fallback"

        # Non-existent key without default
        assert config.get("nonexistent.key") is None

    def test_set_with_dot_notation(self):
        """Test setting values with dot notation."""
        config = Config()

        # Set existing key
        config.set("data.default_exchange", "coinbase")
        assert config.get("data.default_exchange") == "coinbase"

        # Set new nested key
        config.set("new.nested.key", "new_value")
        assert config.get("new.nested.key") == "new_value"

        # Set new top-level key
        config.set("top_level", "value")
        assert config.get("top_level") == "value"

    def test_save_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config()
        config.set("data.default_exchange", "coinbase")
        config.set("custom.setting", "test")

        config_file = self.temp_dir / "saved_config.yaml"
        config.config_path = config_file
        config.save()

        # Verify file was created and contains correct data
        assert config_file.exists()

        with open(config_file, "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["data"]["default_exchange"] == "coinbase"
        assert saved_data["custom"]["setting"] == "test"

    def test_save_json(self):
        """Test saving configuration to JSON file."""
        config = Config()
        config.set("data.default_exchange", "kraken")

        config_file = self.temp_dir / "saved_config.json"
        config.config_path = config_file
        config.save()

        # Verify file was created and contains correct data
        assert config_file.exists()

        with open(config_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["data"]["default_exchange"] == "kraken"

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        config_file = self.temp_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(RuntimeError, match="Failed to load config"):
            Config(config_file)

    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        config_file = self.temp_dir / "config.txt"
        config_file.write_text("some text content")

        with pytest.raises(RuntimeError, match="Failed to load config"):
            Config(config_file)

    def test_default_config_structure(self):
        """Test that default config has expected structure."""
        config = Config()

        # Test data section
        assert "cache_dir" in config._config["data"]
        assert "output_dir" in config._config["data"]
        assert "default_exchange" in config._config["data"]
        assert "default_timeframe" in config._config["data"]
        assert "default_limit" in config._config["data"]

        # Test indicators section
        assert "ema_periods" in config._config["indicators"]
        assert "rsi_period" in config._config["indicators"]
        assert "rsi_overbought" in config._config["indicators"]
        assert "rsi_oversold" in config._config["indicators"]

        # Test trading section
        assert "symbols" in config._config["trading"]
        assert "timeframes" in config._config["trading"]
        assert "risk_per_trade" in config._config["trading"]
        assert "max_positions" in config._config["trading"]

        # Test api section
        assert "host" in config._config["api"]
        assert "port" in config._config["api"]
        assert "debug" in config._config["api"]
