"""Unit tests for data ingestion module."""

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


class TestDataIngestion:
    """Test cases for DataIngestion class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_ohlcv = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47800.0, 47000.0, 47600.0, 1200.0],
            [1641002400000, 47600.0, 48000.0, 47300.0, 47800.0, 900.0],
        ]

    def test_init_default_exchange(self):
        """Test initialization with default exchange."""
        with patch("botclave.data.ingestion.ccxt.binance") as mock_exchange:
            ingestion = DataIngestion()
            mock_exchange.assert_called_once()
            assert ingestion.exchange_name == "binance"

    def test_init_custom_exchange(self):
        """Test initialization with custom exchange."""
        with patch("botclave.data.ingestion.ccxt.coinbase") as mock_exchange:
            ingestion = DataIngestion("coinbase")
            mock_exchange.assert_called_once()
            assert ingestion.exchange_name == "coinbase"

    def test_ohlcv_to_dataframe(self):
        """Test conversion of OHLCV data to DataFrame."""
        with patch("botclave.data.ingestion.ccxt.binance"):
            ingestion = DataIngestion()
            df = ingestion._ohlcv_to_dataframe(self.sample_ohlcv)

            assert len(df) == 3
            assert list(df.columns) == [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            assert df.index.name == "datetime"
            assert df["open"].dtype == "float64"
            assert df["close"].iloc[0] == 47200.0

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_fetch_ohlcv_success(self, mock_exchange_class):
        """Test successful OHLCV data fetch."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()
        df = ingestion.fetch_ohlcv("BTC/USDT", "4h", 3, use_cache=False)

        assert len(df) == 3
        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "4h", limit=3)

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_fetch_ohlcv_with_cache(self, mock_exchange_class):
        """Test OHLCV data fetch with caching."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()
        cache_file = self.temp_dir / "test_cache.csv"

        # First fetch (should call exchange)
        df1 = ingestion.fetch_ohlcv(
            "BTC/USDT", "4h", 3, use_cache=True, cache_file=cache_file
        )
        assert len(df1) == 3
        assert mock_exchange.fetch_ohlcv.call_count == 1
        assert cache_file.exists()

        # Second fetch (should use cache)
        df2 = ingestion.fetch_ohlcv(
            "BTC/USDT", "4h", 3, use_cache=True, cache_file=cache_file
        )
        assert len(df2) == 3
        assert mock_exchange.fetch_ohlcv.call_count == 1  # Should not increase
        pd.testing.assert_frame_equal(df1, df2)

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_fetch_ohlcv_exchange_error(self, mock_exchange_class):
        """Test handling of exchange fetch errors."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()

        with pytest.raises(RuntimeError, match="Failed to fetch data"):
            ingestion.fetch_ohlcv("BTC/USDT", "4h", 3, use_cache=False)

    def test_get_cache_info_no_cache(self):
        """Test cache info when no cache exists."""
        with patch("botclave.data.ingestion.ccxt.binance"):
            ingestion = DataIngestion()
            # Use a specific file that we know doesn't exist
            cache_info = ingestion.get_cache_info("NONEXISTENT/PAIR", "1m")

            assert cache_info["exists"] is False

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_get_cache_info_with_cache(self, mock_exchange_class):
        """Test cache info when cache exists."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        mock_exchange_class.return_value = mock_exchange

        ingestion = DataIngestion()
        cache_file = self.temp_dir / "test_cache.csv"

        # Create cache
        ingestion.fetch_ohlcv(
            "BTC/USDT", "4h", 3, use_cache=True, cache_file=cache_file
        )

        # Get cache info
        cache_info = ingestion.get_cache_info("BTC/USDT", "4h")

        assert cache_info["exists"] is True
        assert cache_info["candles_count"] == 3
        assert "earliest_date" in cache_info
        assert "latest_date" in cache_info


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_ohlcv = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 1000.0],
            [1640998800000, 47200.0, 47800.0, 47000.0, 47600.0, 1200.0],
        ]

    @patch("botclave.data.ingestion.ccxt.binance")
    def test_fetch_data_function(self, mock_exchange_class):
        """Test fetch_data convenience function."""
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = self.sample_ohlcv
        mock_exchange_class.return_value = mock_exchange

        df = fetch_data("BTC/USDT", "4h", 2, use_cache=False)

        assert len(df) == 2
        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "4h", limit=2)

    def test_save_dataset_csv(self):
        """Test saving dataset to CSV."""
        df = pd.DataFrame(
            {
                "open": [47000.0, 47200.0],
                "high": [47500.0, 47800.0],
                "low": [46800.0, 47000.0],
                "close": [47200.0, 47600.0],
                "volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2022-01-01", "2022-01-02"]),
        )

        file_path = self.temp_dir / "test_data.csv"
        save_dataset(df, file_path)

        assert file_path.exists()
        loaded_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        assert len(loaded_df) == 2
        assert loaded_df["close"].iloc[0] == 47200.0

    def test_save_dataset_json(self):
        """Test saving dataset to JSON."""
        df = pd.DataFrame(
            {
                "open": [47000.0, 47200.0],
                "close": [47200.0, 47600.0],
            }
        )

        file_path = self.temp_dir / "test_data.json"
        save_dataset(df, file_path)

        assert file_path.exists()

    def test_save_dataset_unsupported_format(self):
        """Test saving to unsupported format raises error."""
        df = pd.DataFrame({"close": [47200.0]})
        file_path = self.temp_dir / "test_data.txt"

        with pytest.raises(ValueError, match="Unsupported file format"):
            save_dataset(df, file_path)

    def test_load_dataset_csv(self):
        """Test loading dataset from CSV."""
        # First save a test file
        df = pd.DataFrame(
            {
                "open": [47000.0, 47200.0],
                "close": [47200.0, 47600.0],
            },
            index=pd.to_datetime(["2022-01-01", "2022-01-02"]),
        )

        file_path = self.temp_dir / "test_data.csv"
        df.to_csv(file_path)

        # Load it back
        loaded_df = load_dataset(file_path)

        assert len(loaded_df) == 2
        assert loaded_df["close"].iloc[0] == 47200.0

    def test_load_dataset_not_found(self):
        """Test loading non-existent file raises error."""
        file_path = self.temp_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_dataset(file_path)

    def test_load_dataset_unsupported_format(self):
        """Test loading unsupported format raises error."""
        file_path = self.temp_dir / "test.txt"
        file_path.write_text("test content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(file_path)
