# 🤖 BOTCLAVE - BTC/XAU Order Flow Absorption Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**BOTCLAVE** is a sophisticated Python package for ICT (Inner Circle Trader) Order Flow cryptocurrency trading analysis, with specialized support for BTC/XAU pair trading. The bot analyzes order book depth, footprint charts, and market structure to identify high-probability absorption zones and trading opportunities.

## 🎯 Features

- **📊 Order Flow Analysis**
  - Real-time depth of market (DOM) analysis
  - Footprint charting with buy/sell imbalance detection
  - Cumulative volume delta (CVD) tracking
  - Absorption zone identification
  
- **🔍 ICT Methodology**
  - Order block detection
  - Market structure analysis (swing highs/lows)
  - Liquidity void identification
  - Fair value gap detection

- **💹 Exchange Integration**
  - Binance connector via CCXT
  - WebSocket support for real-time data
  - Order management system
  - Position tracking

- **🧪 Backtesting & Validation**
  - Historical strategy simulation
  - Walk-forward analysis
  - Monte Carlo simulation
  - Statistical validation

- **📈 Visualization Dashboard**
  - Streamlit-based interactive dashboard
  - Real-time order flow visualization
  - Performance metrics and charts
  - Trade analysis tools

## 📊 Implementation Status

| Module | Status | Progress | Description |
|--------|--------|----------|-------------|
| Engine | ✅ | 5/5 | Core order flow analysis components |
| Exchange | ✅ | 2/2 | Exchange connectivity and order management |
| Backtest | ✅ | 2/2 | Backtesting and strategy validation |
| Dashboard | ✅ | 3/3 | Streamlit visualization dashboard |

*Last updated: 2024-01-15*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/botclave.git
cd botclave

# Install dependencies using Poetry
poetry install

# Or using pip
pip install -e .
```

### Basic Usage

```python
from botclave.engine.strategy import OrderFlowStrategy, StrategyConfig
from botclave.exchange.binance_connector import BinanceConnector
import pandas as pd

# Initialize strategy
config = StrategyConfig(min_confidence=0.6, risk_reward_ratio=2.0)
strategy = OrderFlowStrategy(config)

# Connect to exchange
connector = BinanceConnector()
df = await connector.fetch_ohlcv("BTC/USDT", "15m", limit=500)

# Generate trading signals
signal = strategy.generate_signal(
    symbol="BTC/USDT",
    df=df,
    current_price=df['close'].iloc[-1],
    timestamp=int(pd.Timestamp.now().timestamp() * 1000)
)

if signal:
    print(f"Signal: {signal.side} at {signal.entry_price}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Take Profit: {signal.take_profit}")
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/botclave/dashboard/app.py
```

### Fetching Historical Data

```bash
# Download historical data
python scripts/fetch_historical_data.py \
    --symbol BTC/USDT \
    --timeframe 15m \
    --days 30 \
    --output data/historical
```

### Running Backtests

```bash
# Run a backtest
python scripts/backtest.py \
    --data data/historical/BTC_USDT_15m_30d.csv \
    --capital 10000 \
    --position-size 0.02 \
    --validate
```

## 📁 Project Structure

```
botclave/
├── src/botclave/
│   ├── engine/              # Core order flow analysis
│   │   ├── depth.py         # Order book depth analysis
│   │   ├── footprint.py     # Footprint charting
│   │   ├── dom_builder.py   # Depth of Market builder
│   │   ├── indicators.py    # Order flow indicators
│   │   └── strategy.py      # Trading strategy logic
│   ├── exchange/            # Exchange connectivity
│   │   ├── binance_connector.py  # Binance integration
│   │   └── order_manager.py      # Order management
│   ├── backtest/            # Backtesting framework
│   │   ├── backtester.py    # Backtest engine
│   │   └── validator.py     # Strategy validation
│   └── dashboard/           # Visualization
│       ├── app.py           # Main dashboard app
│       ├── charts.py        # Chart generation
│       └── metrics.py       # Metrics calculation
├── config/                  # Configuration files
│   ├── strategy_params.yaml
│   └── exchange_config.yaml
├── scripts/                 # Utility scripts
│   ├── fetch_historical_data.py
│   ├── backtest.py
│   ├── generate_report.py
│   └── update_readme.py
├── tests/                   # Test suite
│   ├── test_depth.py
│   ├── test_footprint.py
│   ├── test_indicators.py
│   ├── test_strategy.py
│   └── test_backtest.py
└── docs/                    # Documentation
    ├── ARCHITECTURE.md
    ├── SETUP.md
    └── DEVELOPMENT.md
```

## 📖 Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and design
- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[Development](docs/DEVELOPMENT.md)** - Contributing guidelines

## 🔧 Configuration

Configuration files are located in the `config/` directory:

### Strategy Parameters (`config/strategy_params.yaml`)

```yaml
strategy:
  name: "OrderFlowAbsorption"
  enabled: true

risk_management:
  initial_capital: 10000.0
  position_size_percent: 2.0
  max_positions: 3
  max_drawdown_percent: 15.0

order_flow:
  absorption_threshold: 2.0
  imbalance_threshold: 1.5
  use_footprint: true
  use_depth_analysis: true
```

### Exchange Configuration (`config/exchange_config.yaml`)

```yaml
exchange:
  name: "binance"
  testnet: true
  enable_rate_limit: true

api:
  use_env_vars: true
  env_key_name: "BINANCE_API_KEY"
  env_secret_name: "BINANCE_API_SECRET"
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_strategy.py -v

# Run with coverage
pytest tests/ --cov=botclave --cov-report=html
```

## 📊 Key Indicators

The bot uses several proprietary indicators for order flow analysis:

1. **Cumulative Volume Delta (CVD)** - Tracks buying vs selling pressure
2. **Order Book Imbalance** - Identifies bid/ask imbalances
3. **Absorption Zones** - Detects areas where large orders are absorbed
4. **Volume Profile** - Shows volume distribution across price levels
5. **Footprint Patterns** - Identifies specific order flow patterns

## 🎯 Trading Strategy

The bot implements an ICT-based order flow strategy:

1. **Market Structure Analysis** - Identifies trend and key levels
2. **Order Block Detection** - Finds institutional order blocks
3. **Liquidity Analysis** - Tracks where liquidity is concentrated
4. **Entry Signal Generation** - Combines multiple indicators
5. **Risk Management** - Dynamic position sizing and stop losses

## 📈 Performance Metrics

Track strategy performance with comprehensive metrics:

- Win Rate
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Average R:R Ratio
- Recovery Factor
- Calmar Ratio

## 🛠️ Development

### Code Style

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for static typing
- **Pydantic** for data validation

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch from `main`
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## 🐛 Known Issues & Roadmap

### Current Limitations
- WebSocket streaming not yet implemented
- Live trading mode under development
- Limited to Binance exchange

### Roadmap
- [ ] Multi-exchange support
- [ ] Machine learning integration
- [ ] Advanced risk management
- [ ] Portfolio management
- [ ] Mobile notifications
- [ ] Cloud deployment support

## 🤝 Contributing

Contributions are welcome! Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies carries risk. Always do your own research and never risk more than you can afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/botclave/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/botclave/discussions)
- **Email**: support@botclave.io

## 🙏 Acknowledgments

- ICT (Inner Circle Trader) for order flow concepts
- CCXT library for exchange connectivity
- Streamlit for dashboard framework
- The Python trading community

---

**Built with ❤️ by the BOTCLAVE Team**

*Last Updated: 2024-01-15*
