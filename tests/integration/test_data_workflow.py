"""Integration tests for the complete data workflow."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from botclave.data.ingestion import (
    DataIngestion,
    fetch_data,
    save_dataset,
    load_dataset,
)
from botclave.config.manager import Config


class TestDataWorkflow:
    """Integration tests for complete data workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_ohlcv = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47800.0, 47000.0, 47600.0, 1200.0],
            [1641002400000, 47600.0, 48000.0, 47300.0, 47800.0, 900.0],
            [1641006000000, 47800.0, 48200.0, 47500.0, 48000.0, 1100.0],
            [1641009600000, 48000.0, 48500.0, 47700.0, 48300.0, 1300.0],
        ]

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_complete_fetch_cache_save_load_workflow(self, mock_exchange_class):
        """Test complete workflow: fetch -> cache -> save -> load."""
        # Setup mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        mock_exchange_class.return_value = mock_exchange

        # Initialize data ingestion
        ingestion = DataIngestion()
        cache_file = self.temp_dir / "btc_usdt_4h.csv"
        output_file = self.temp_dir / "output_dataset.csv"

        # Step 1: Fetch data (should create cache)
        df1 = ingestion.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="4h",
            limit=5,
            use_cache=True,
            cache_file=cache_file,
        )

        assert len(df1) == 5
        assert cache_file.exists()
        assert mock_exchange.fetch_ohlcv.call_count == 1

        # Step 2: Fetch again (should use cache)
        df2 = ingestion.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="4h",
            limit=5,
            use_cache=True,
            cache_file=cache_file,
        )

        assert len(df2) == 5
        assert mock_exchange.fetch_ohlcv.call_count == 1  # Should not increase
        pd.testing.assert_frame_equal(df1, df2)

        # Step 3: Save to different format
        save_dataset(df1, output_file)
        assert output_file.exists()

        # Step 4: Load from saved file
        df3 = load_dataset(output_file)
        assert len(df3) == 5
        pd.testing.assert_frame_equal(df1, df3)

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_config_integration(self, mock_exchange_class):
        """Test integration with configuration system."""
        # Setup mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv[:3]
        mock_exchange_class.return_value = mock_exchange

        # Create custom config
        config_data = {
            "data": {
                "default_exchange": "binance",
                "default_timeframe": "4h",
                "default_limit": 3,
                "cache_dir": str(self.temp_dir / "cache"),
            }
        }

        config_file = self.temp_dir / "config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Test that DataIngestion uses config
        with patch("botclave.data.ingestion.config", Config(config_file)):
            ingestion = DataIngestion()

            assert ingestion.exchange_name == "binance"
            assert ingestion.cache_dir == Path(self.temp_dir / "cache")

            df = ingestion.fetch_ohlcv("BTC/USDT", "4h", 3, use_cache=False)
            assert len(df) == 3

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_multiple_symbols_timeframes(self, mock_exchange_class):
        """Test fetching data for multiple symbols and timeframes."""
        # Setup mock exchange to return different data for different calls
        mock_exchange = Mock()

        def fetch_ohlcv_side_effect(symbol, timeframe, limit=None):
            if symbol == "BTC/USDT":
                return self.sample_ohlcv[:3]
            elif symbol == "ETH/USDT":
                return [
                    [1640995200000, 3200.0, 3250.0, 3150.0, 3220.0, 500.0],
                    [1640998800000, 3220.0, 3280.0, 3200.0, 3260.0, 600.0],
                    [1641002400000, 3260.0, 3300.0, 3230.0, 3280.0, 450.0],
                ]
            else:
                return []

        mock_exchange.fetch_ohlcv.side_effect = fetch_ohlcv_side_effect
        mock_exchange_class.return_value = mock_exchange

        # Test fetching different symbols
        btc_df = fetch_data("BTC/USDT", "4h", 3, use_cache=False)
        eth_df = fetch_data("ETH/USDT", "4h", 3, use_cache=False)

        assert len(btc_df) == 3
        assert len(eth_df) == 3
        assert btc_df["close"].iloc[0] == 47200.0
        assert eth_df["close"].iloc[0] == 3220.0

        # Verify exchange was called correctly
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_cache_fallback_behavior(self, mock_exchange_class):
        """Test cache fallback when cache is corrupted."""
        # Setup mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv[:3]
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()
        cache_file = self.temp_dir / "corrupted_cache.csv"

        # Create corrupted cache file
        cache_file.write_text("invalid,csv,content\nnot,proper,data")

        # Should fall back to exchange fetch
        df = ingestion.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="4h",
            limit=3,
            use_cache=True,
            cache_file=cache_file,
        )

        assert len(df) == 3
        assert mock_exchange.fetch_ohlcv.call_count == 1

        # Should overwrite corrupted cache with valid data
        df_cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        assert len(df_cached) == 3

    def test_file_format_roundtrip(self):
        """Test saving and loading different file formats."""
        # Create test data
        df = pd.DataFrame(
            {
                "open": [47000.0, 47200.0, 47600.0],
                "high": [47500.0, 47800.0, 48000.0],
                "low": [46800.0, 47000.0, 47300.0],
                "close": [47200.0, 47600.0, 47800.0],
                "volume": [1000.0, 1200.0, 900.0],
            },
            index=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"]),
        )

        # Test CSV format
        csv_file = self.temp_dir / "test.csv"
        save_dataset(df, csv_file)
        loaded_csv = load_dataset(csv_file)
        pd.testing.assert_frame_equal(df, loaded_csv)

        # Test JSON format
        json_file = self.temp_dir / "test.json"
        save_dataset(df, json_file)
        loaded_json = load_dataset(json_file)
        # JSON doesn't preserve index the same way, so compare data
        assert len(loaded_json) == 3
        assert list(loaded_json["close"]) == [47200.0, 47600.0, 47800.0]

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_error_handling_and_recovery(self, mock_exchange_class):
        """Test error handling and recovery scenarios."""
        # Setup mock exchange that fails initially
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("Network error"),  # First call fails
            self.sample_ohlcv[:3],  # Second call succeeds
        ]
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()

        # First attempt should fail
        with pytest.raises(RuntimeError, match="Failed to fetch data"):
            ingestion.fetch_ohlcv("BTC/USDT", "4h", 3, use_cache=False)

        # Second attempt should succeed
        df = ingestion.fetch_ohlcv("BTC/USDT", "4h", 3, use_cache=False)
        assert len(df) == 3
        assert mock_exchange.fetch_ohlcv.call_count == 2
