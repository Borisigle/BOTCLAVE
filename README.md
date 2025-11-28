# Botclave

ICT Order Flow port for cryptocurrency trading analysis.

## Overview

Botclave is a Python package designed for cryptocurrency trading analysis with a focus on ICT (Inner Circle Trader) Order Flow concepts. It provides tools for data ingestion, technical analysis, and trading strategy development.

## Features

- **Data Ingestion**: Fetch OHLCV data from multiple exchanges via CCXT
- **Caching**: Intelligent caching system to avoid redundant API calls
- **Configuration**: Flexible YAML/JSON configuration management
- **CLI Tools**: Command-line interface for data fetching and management
- **Type Safety**: Full type hints and Pydantic models
- **Testing**: Comprehensive test suite with mocked dependencies

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd botclave

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Using pip

```bash
# Install dependencies
pip install pandas numpy plotly pydantic ccxt pyyaml fastapi uvicorn pytest pytest-asyncio ruff black
```

## Quick Start

### 1. Configuration

The package uses a configuration system with sensible defaults. The default configuration is located at `config/config.yaml`:

```yaml
data:
  cache_dir: "data/cache"
  output_dir: "data/output"
  default_exchange: "binance"
  default_timeframe: "4h"
  default_limit: 1500

indicators:
  ema_periods: [20, 50, 200]
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30

trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframes: ["1h", "4h", "1d"]
```

### 2. Fetch Data via CLI

```bash
# Basic usage - fetches 1500 BTC/USDT 4H candles
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 1500

# Custom cache file
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 1000 --cache data/btcusdt_4h.csv

# Save to file
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 500 --save data/btc_4h.csv

# Skip cache and fetch fresh data
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 1000 --no-cache

# Check cache information
python -m botclave --symbol BTC/USDT --timeframe 4h --cache-info

# Verbose output
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 100 --verbose
```

### 3. Use in Python Code

```python
from botclave.data.ingestion import fetch_data, save_dataset
from botclave.config.manager import config

# Fetch data with caching
df = fetch_data(
    symbol="BTC/USDT",
    timeframe="4h",
    limit=1000,
    use_cache=True
)

print(f"Fetched {len(df)} candles")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Save dataset
save_dataset(df, "data/btc_4h.csv")

# Access configuration
exchange = config.get("data.default_exchange")
print(f"Default exchange: {exchange}")
```

### 4. Advanced Usage

```python
from botclave.data.ingestion import DataIngestion
from pathlib import Path

# Create custom ingestion instance
ingestion = DataIngestion(exchange_name="coinbase")

# Fetch with custom cache file
df = ingestion.fetch_ohlcv(
    symbol="ETH/USDT",
    timeframe="1h",
    limit=500,
    cache_file=Path("custom_cache/eth_1h.csv")
)

# Get cache information
cache_info = ingestion.get_cache_info("ETH/USDT", "1h")
if cache_info["exists"]:
    print(f"Cache has {cache_info['candles_count']} candles")

# Fetch latest candles (bypasses cache)
latest = ingestion.fetch_latest_candles("BTC/USDT", "4h", limit=5)
print(latest)
```

## Project Structure

```
botclave/
├── src/
│   └── botclave/
│       ├── __init__.py
│       ├── __main__.py              # CLI entry point
│       ├── config/
│       │   ├── __init__.py
│       │   └── manager.py           # Configuration management
│       ├── data/
│       │   ├── __init__.py
│       │   └── ingestion.py         # Data ingestion module
│       └── scripts/
│           ├── __init__.py
│           └── fetch_data.py        # CLI script
├── config/
│   └── config.yaml                  # Default configuration
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   └── test_data_ingestion.py
│   └── integration/
│       └── test_data_workflow.py
├── pyproject.toml                   # Project configuration
└── README.md
```

## Configuration

### Configuration File Locations

The configuration system searches for config files in this order:
1. `config.yaml`
2. `config.yml`
3. `config.json`
4. `config/config.yaml`
5. `config/config.yml`
6. `config/config.json`

### Configuration Structure

```yaml
data:
  cache_dir: "data/cache"           # Directory for cached data
  output_dir: "data/output"         # Directory for output files
  default_exchange: "binance"       # Default exchange
  default_timeframe: "4h"           # Default timeframe
  default_limit: 1500               # Default number of candles

indicators:
  ema_periods: [20, 50, 200]        # EMA periods
  rsi_period: 14                    # RSI period
  rsi_overbought: 70                # RSI overbought level
  rsi_oversold: 30                  # RSI oversold level
  atr_period: 14                    # ATR period
  bb_period: 20                     # Bollinger Bands period
  bb_std: 2                         # Bollinger Bands standard deviations

trading:
  symbols: ["BTC/USDT", "ETH/USDT"] # Default trading symbols
  timeframes: ["1h", "4h", "1d"]    # Default timeframes
  risk_per_trade: 0.02              # Risk per trade (2%)
  max_positions: 5                  # Maximum open positions

api:
  host: "0.0.0.0"                   # API host
  port: 8000                        # API port
  debug: false                      # Debug mode
```

## Data Formats

### Supported Exchanges

Botclave supports all exchanges available in the CCXT library, including:
- Binance
- Coinbase
- Kraken
- Bitfinex
- Bybit
- And many more...

### Supported Timeframes

- `1m`, `3m`, `5m`, `15m`, `30m` (minutes)
- `1h`, `2h`, `4h`, `6h`, `8h`, `12h` (hours)
- `1d`, `3d` (days)
- `1w` (weeks)
- `1M` (months)

### Data Storage Formats

- **CSV**: Default format, preserves datetime index
- **JSON**: Record-oriented format
- **Parquet**: Efficient binary format (requires pyarrow)

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=botclave --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_data_ingestion.py

# Run with verbose output
poetry run pytest -v
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking (if using mypy)
poetry run mypy src/
```

### Adding New Features

1. Add modules under `src/botclave/`
2. Follow the existing code style and patterns
3. Add comprehensive tests
4. Update documentation
5. Ensure all tests pass

## API Reference

### DataIngestion Class

```python
class DataIngestion:
    def __init__(self, exchange_name: Optional[str] = None)
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, 
                   use_cache: bool = True, cache_file: Optional[Path] = None) -> pd.DataFrame
    def fetch_latest_candles(self, symbol: str, timeframe: str, limit: int = 1) -> pd.DataFrame
    def get_cache_info(self, symbol: str, timeframe: str) -> dict
```

### Configuration Class

```python
class Config:
    def __init__(self, config_path: Optional[Path] = None)
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
    def load(self) -> None
    def save(self) -> None
```

### Utility Functions

```python
def fetch_data(symbol: str, timeframe: str, limit: int, 
              exchange: Optional[str] = None, use_cache: bool = True,
              cache_file: Optional[Path] = None) -> pd.DataFrame

def save_dataset(df: pd.DataFrame, file_path: Path) -> None
def load_dataset(file_path: Path) -> pd.DataFrame
```

## Troubleshooting

### Common Issues

1. **Exchange API Errors**: Check if the exchange is available and has the required symbol/timeframe
2. **Cache Issues**: Delete cache files and re-fetch data if corrupted
3. **Configuration Errors**: Verify YAML syntax and file paths
4. **Memory Issues**: Use smaller `limit` values for large datasets

### Debug Mode

Enable verbose output for debugging:

```bash
python -m botclave --symbol BTC/USDT --timeframe 4h --limit 100 --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Add your license information here]

## Changelog

### v0.1.0
- Initial release
- Data ingestion with CCXT integration
- Configuration management system
- CLI tools for data fetching
- Comprehensive test suite
- Documentation and examples