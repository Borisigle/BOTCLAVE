"""Configuration management for Botclave."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for Botclave."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default paths.
        """
        self.config_path = config_path or self._find_config_file()
        self._config: Dict[str, Any] = {}
        self.load()

    def _find_config_file(self) -> Path:
        """Find configuration file in standard locations."""
        possible_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path("config.json"),
            Path("config/config.yaml"),
            Path("config/config.yml"),
            Path("config/config.json"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Return default config path
        return Path("config/config.yaml")

    def load(self) -> None:
        """Load configuration from file."""
        # Start with default config
        self._config = self._get_default_config()

        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r") as f:
                if self.config_path.suffix.lower() in [".yaml", ".yml"]:
                    loaded_config = yaml.safe_load(f) or {}
                elif self.config_path.suffix.lower() == ".json":
                    import json

                    loaded_config = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported config format: {self.config_path.suffix}"
                    )

                # Merge loaded config with defaults (deep merge)
                self._config = self._deep_merge(self._config, loaded_config)

        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def _deep_merge(self, base: dict, update: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            if self.config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            elif self.config_path.suffix.lower() == ".json":
                import json

                json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation).

        Args:
            key: Configuration key (e.g., 'data.cache_dir')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation).

        Args:
            key: Configuration key (e.g., 'data.cache_dir')
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "cache_dir": "data/cache",
                "output_dir": "data/output",
                "default_exchange": "binance",
                "default_timeframe": "4h",
                "default_limit": 1500,
            },
            "indicators": {
                "ema_periods": [20, 50, 200],
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "atr_period": 14,
                "bb_period": 20,
                "bb_std": 2,
            },
            "trading": {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframes": ["1h", "4h", "1d"],
                "risk_per_trade": 0.02,
                "max_positions": 5,
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
            },
        }


# Global configuration instance
config = Config()
