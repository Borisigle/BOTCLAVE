# Setup Guide

Complete installation and configuration guide for BOTCLAVE.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [API Setup](#api-setup)
5. [Testing Installation](#testing-installation)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB, recommended 8GB
- **Disk Space**: Minimum 1GB for installation and data

### Required Software

```bash
# Python 3.9+
python --version

# pip (Python package manager)
pip --version

# Git (for cloning repository)
git --version
```

## Installation

### Method 1: Using Poetry (Recommended)

Poetry provides better dependency management and virtual environment handling.

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yourusername/botclave.git
cd botclave

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/botclave.git
cd botclave

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Method 3: Docker (Coming Soon)

```bash
# Build Docker image
docker build -t botclave:latest .

# Run container
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  botclave:latest
```

## Configuration

### 1. Strategy Configuration

Edit `config/strategy_params.yaml`:

```yaml
strategy:
  name: "OrderFlowAbsorption"
  version: "1.0.0"
  enabled: true

risk_management:
  initial_capital: 10000.0      # Starting capital in USD
  position_size_percent: 2.0    # 2% of capital per trade
  max_positions: 3              # Maximum concurrent positions
  max_drawdown_percent: 15.0    # Stop trading if drawdown exceeds 15%

order_flow:
  absorption_threshold: 2.0     # Multiplier for absorption detection
  imbalance_threshold: 1.5      # Ratio for imbalance detection
  min_volume: 1.0              # Minimum volume to consider

indicators:
  cvd_period: 50               # Cumulative Volume Delta lookback
  delta_period: 20             # Delta calculation period
  value_area_percent: 0.7      # 70% value area

entry:
  min_confidence: 0.6          # Minimum signal confidence (60%)
  min_risk_reward: 2.0         # Minimum 2:1 R:R ratio

exit:
  use_trailing_stop: true
  trailing_stop_percent: 2.0   # 2% trailing stop
  take_profit_levels: 3        # Three TP levels
```

### 2. Exchange Configuration

Edit `config/exchange_config.yaml`:

```yaml
exchange:
  name: "binance"
  testnet: true                # Start with testnet!
  enable_rate_limit: true
  timeout: 30000

api:
  use_env_vars: true           # Use environment variables
  env_key_name: "BINANCE_API_KEY"
  env_secret_name: "BINANCE_API_SECRET"

fees:
  maker: 0.0002                # 0.02%
  taker: 0.0004                # 0.04%

slippage:
  model: "percentage"
  percentage: 0.01             # 0.01% slippage estimate
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# Binance API credentials (testnet)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here

# Optional: Production credentials (keep separate!)
BINANCE_PROD_API_KEY=your_production_api_key
BINANCE_PROD_API_SECRET=your_production_api_secret
```

**Important**: Add `.env` to your `.gitignore` to prevent committing credentials!

### 4. Directory Structure

Create necessary directories:

```bash
# Create data directories
mkdir -p data/historical
mkdir -p data/cache
mkdir -p data/backtest_results

# Create log directory
mkdir -p logs

# Create reports directory
mkdir -p reports
```

## API Setup

### Binance Testnet

1. **Visit Binance Testnet**: https://testnet.binance.vision/
2. **Create Account**: Register for a testnet account
3. **Generate API Keys**:
   - Go to API Management
   - Create new API key
   - Save the key and secret securely
4. **Configure Permissions**:
   - Enable "Spot & Margin Trading"
   - Enable "Read" permissions

### Binance Production (When Ready)

1. **Visit Binance**: https://www.binance.com/
2. **Enable 2FA**: Set up two-factor authentication
3. **Create API Key**:
   - Go to Account → API Management
   - Create API key with trading permissions
   - Restrict API key to specific IPs (recommended)
4. **Security**:
   - Never share your API keys
   - Use IP whitelisting
   - Enable withdrawal restrictions

## Testing Installation

### 1. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_strategy.py -v

# Run with coverage
pytest tests/ --cov=botclave --cov-report=html
```

### 2. Test Exchange Connection

```bash
# Test connection (using testnet)
python -c "
from botclave.exchange.binance_connector import BinanceConnector
import asyncio

async def test():
    connector = BinanceConnector()
    ticker = await connector.fetch_ticker('BTC/USDT')
    print(f'BTC/USDT: {ticker}')
    connector.close()

asyncio.run(test())
"
```

### 3. Fetch Sample Data

```bash
# Download 7 days of 15m data
python scripts/fetch_historical_data.py \
    --symbol BTC/USDT \
    --timeframe 15m \
    --days 7 \
    --output data/historical
```

### 4. Run a Simple Backtest

```bash
# Run backtest on downloaded data
python scripts/backtest.py \
    --data data/historical/BTC_USDT_15m_7d.csv \
    --capital 10000 \
    --position-size 0.02
```

### 5. Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/botclave/dashboard/app.py
```

The dashboard should open in your browser at `http://localhost:8501`

## Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Dependencies installed successfully
- [ ] Configuration files created
- [ ] Environment variables set
- [ ] API keys configured (testnet)
- [ ] Tests pass successfully
- [ ] Can fetch market data
- [ ] Can run backtests
- [ ] Dashboard launches

## Troubleshooting

### Common Issues

#### Issue: Import errors

```bash
# Solution: Ensure you're in the virtual environment
poetry shell  # or source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

#### Issue: API connection fails

```bash
# Check your API keys are set
echo $BINANCE_API_KEY

# Verify testnet setting
# Check config/exchange_config.yaml - testnet should be true

# Test basic connectivity
ping testnet.binance.vision
```

#### Issue: Module not found

```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install package
pip install -e .
```

#### Issue: Rate limit errors

```bash
# Enable rate limiting in config
# config/exchange_config.yaml:
#   enable_rate_limit: true
```

#### Issue: Missing dependencies

```bash
# Update pip
pip install --upgrade pip

# Reinstall all dependencies
poetry install --no-cache
# or
pip install -r requirements.txt --force-reinstall
```

### Getting Help

If you encounter issues:

1. Check the [documentation](../README.md)
2. Search [existing issues](https://github.com/yourusername/botclave/issues)
3. Join our [Discord community](#)
4. Create a new issue with:
   - Python version
   - Operating system
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Read the Documentation**:
   - [Architecture](ARCHITECTURE.md)
   - [Development Guide](DEVELOPMENT.md)

2. **Run Example Scripts**:
   ```bash
   python examples/basic_analysis.py
   python examples/backtest_example.py
   ```

3. **Customize Configuration**:
   - Adjust risk parameters
   - Set your trading symbols
   - Configure indicators

4. **Paper Trading**:
   - Test strategy with testnet
   - Monitor performance
   - Refine parameters

5. **Production Deployment** (when ready):
   - Switch to production API
   - Start with small capital
   - Monitor closely

## Security Best Practices

1. **API Keys**:
   - Never commit to git
   - Use environment variables
   - Restrict IP addresses
   - Enable withdrawal restrictions

2. **Configuration**:
   - Keep testnet and production separate
   - Use separate API keys
   - Test thoroughly before production

3. **Monitoring**:
   - Set up alerts
   - Monitor logs regularly
   - Track performance metrics

---

**Ready to start trading? Check out the [Development Guide](DEVELOPMENT.md)!**

*Last Updated: 2024-01-15*
