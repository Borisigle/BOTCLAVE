# Botclave Streamlit Dashboard Guide

## Overview

The Botclave Streamlit Dashboard is a professional trading interface that provides real-time monitoring, Smart Money Concepts (SMC) analysis, and order flow visualization for cryptocurrency trading. It features automated signal generation with alerts, interactive charts, and comprehensive metrics.

## Features

### 🚨 Real-time Alerts
- **Alert Panel**: Displays current trading signals prominently at the top
- **Sound Notifications**: Optional audio alerts for new signals
- **Signal Details**: Shows entry, stop loss, take profit, confidence, and risk-reward ratio
- **Signal Components**: Displays SMC and orderflow components that generated the signal
- **Reason Explanation**: Detailed explanation of why a signal was generated

### 📊 Interactive Charts
- **Candlestick Chart**: Professional OHLCV visualization
- **SMC Overlays**:
  - Swing highs/lows markers (red/green triangles)
  - Fair Value Gaps (shaded zones)
  - Order Blocks (horizontal lines with shaded regions)
  - Entry/SL/TP levels for active signals
  - Break of Structure markers
- **Interactive Features**: Hover for details, zoom, pan

### 📈 Live Metrics
- **Market Bias**: Current trend direction (bullish/bearish/neutral)
- **Current Price**: Real-time price display
- **ATR (Volatility)**: Average True Range for volatility measurement
- **Confluence Level**: Overall signal strength based on multiple indicators
- **Orderflow Metrics**:
  - Cumulative Volume Delta (CVD)
  - Buy/Sell Ratio
  - Absorption Zones count
- **SMC Indicators Count**: Active FFGs, Order Blocks, Swings

### 🌊 Orderflow Analysis
- **Order Book Heatmap**: Visual representation of depth of market
- **Footprint Chart**: Buy/sell imbalance visualization
- **Absorption Zones**: Detected buy/sell absorption levels

### 📋 Signals Log
- **Complete History**: All generated signals with timestamps
- **Filterable**: Filter by signal type (long/short/neutral)
- **Export to CSV**: Download signal history for analysis
- **Statistics**: Summary of signal performance metrics

### ⚙️ Configuration
- **Symbol Selection**: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT
- **Timeframe Selection**: 1m, 5m, 15m, 1h, 4h
- **Lookback Period**: 50-1000 candles
- **Signal Filters**:
  - Minimum confidence level (0.5-1.0)
  - Minimum risk-reward ratio (1.0-5.0)
- **Auto-refresh**: Configurable refresh interval (1-60 seconds)
- **Alert Settings**: Enable/disable alerts and sound notifications

## Installation

### Prerequisites
- Python 3.9 or higher
- Botclave package installed

### Running the Dashboard

```bash
# From the project root
streamlit run src/botclave/dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage Guide

### Getting Started

1. **Launch the Dashboard**
   ```bash
   streamlit run src/botclave/dashboard/app.py
   ```

2. **Configure Settings (Sidebar)**
   - Select your trading symbol (e.g., BTC/USDT)
   - Choose a timeframe (15m is recommended for SMC analysis)
   - Adjust lookback period (200 candles is a good starting point)
   - Set minimum confidence (0.7 recommended)
   - Set minimum risk-reward ratio (3.0 recommended)

3. **Load Data**
   - Click the "🔄 Load Data" button in the sidebar
   - Wait for data to load and analysis to complete

4. **Monitor Alerts**
   - Check the Alert Panel at the top for any active signals
   - Enable sound alerts if desired (checkbox in Alert Panel)
   - Review the signal reason and components

5. **Explore Tabs**
   - **Chart Tab**: View candlesticks with SMC overlays
   - **Metrics Tab**: Analyze live market metrics
   - **Orderflow Tab**: Study orderflow patterns
   - **Signals Log Tab**: Review historical signals

### Understanding the Alerts

When a signal is generated, you'll see:

```
🟢 ENTRY_LONG
Symbol @ $45,000.00
Time: 2024-01-15 14:30:00

Confidence: 87%
RR Ratio: 3.5:1

Entry: $44,500.00
Stop Loss: $44,000.00
Take Profit: $45,750.00

SMC: BOS+FFG
Orderflow: ABSORPTION_BUY

Why this signal:
BOS+FFG + ABSORPTION_BUY @ $44,500.00, RR=3.5
```

**Signal Types:**
- `ENTRY_LONG`: Long entry signal
- `ENTRY_SHORT`: Short entry signal
- `EXIT_LONG`: Exit long position
- `EXIT_SHORT`: Exit short position
- `NEUTRAL`: No trading opportunity

**Signal Components:**
- `BOS`: Break of Structure
- `FFG`: Fair Value Gap
- `OB`: Order Block
- `ABSORPTION_BUY`: Buy absorption detected
- `ABSORPTION_SELL`: Sell absorption detected
- `IMBALANCE`: Order flow imbalance

### Reading the Charts

**Swing Markers:**
- 🔺 Red triangle: Swing high
- 🔻 Green triangle: Swing low

**Fair Value Gaps:**
- Green shaded zone: Bullish FFG (price gap upwards)
- Red shaded zone: Bearish FFG (price gap downwards)

**Order Blocks:**
- Purple zone: Bullish Order Block (support)
- Orange zone: Bearish Order Block (resistance)

**Trade Levels (when signal active):**
- Green dashed line: Entry price
- Red solid line: Stop loss
- Blue solid line: Take profit

### Interpreting Metrics

**Market Bias:**
- 🟢 BULLISH: Uptrend expected
- 🔴 BEARISH: Downtrend expected
- ⚪ NEUTRAL: No clear direction

**CVD (Cumulative Volume Delta):**
- Positive: More buying pressure
- Negative: More selling pressure
- Trend direction matters more than absolute value

**Buy/Sell Ratio:**
- > 1.0: More buyers than sellers
- < 1.0: More sellers than buyers
- 1.0: Balanced market

**ATR (Average True Range):**
- Higher = More volatility
- Lower = Less volatility
- Use for position sizing

## Architecture

### Component Structure

```
src/botclave/dashboard/
├── app.py                    # Main Streamlit application
├── components/
│   ├── alert_panel.py        # Alert display and management
│   ├── chart_panel.py        # Plotly charts with SMC overlays
│   ├── metrics_panel.py      # Live metrics display
│   ├── orderflow_panel.py    # Orderflow visualization
│   └── signals_log.py        # Signal history and export
└── utils/
    ├── data_loader.py        # Data fetching from Binance
    └── alert_manager.py      # Alert management and notifications
```

### Data Flow

1. **Data Loading**: `DataLoader` fetches OHLCV from Binance
2. **Analysis**:
   - `SMCIndicator` performs SMC analysis
   - `TradingStrategy` generates signals
   - `OrderflowAnalyzer` detects absorption
3. **Alert Management**: `AlertManager` tracks and notifies
4. **Display**: Components render results in UI

### Key Classes

**DataLoader**
- `fetch_ohlcv_sync()`: Fetch candlestick data
- `generate_footprints()`: Create footprint objects
- `calculate_orderflow_metrics()`: Compute orderflow metrics

**AlertManager**
- `process_signal()`: Handle new signals
- `get_current_signal()`: Get active signal
- `render_settings()`: Display alert settings

**Components**
- `show_alert_panel()`: Display alert panel
- `create_smc_chart()`: Create chart with SMC overlays
- `show_metrics_panel()`: Display live metrics
- `show_orderflow_panel()`: Show orderflow analysis
- `show_signals_log()`: Display signal history

## Customization

### Adding New Signal Types

Edit `botclave/engine/strategy.py` to add new signal logic:

```python
# In TradingStrategy class
def _create_custom_signal(self, ...):
    # Your custom signal logic
    return Signal(
        signal_type='CUSTOM_SIGNAL',
        ...
    )
```

### Modifying Chart Appearance

Edit `src/botclave/dashboard/components/chart_panel.py`:

```python
# Modify colors, markers, overlays
fig.add_trace(go.Candlestick(
    increasing_line_color="#00FF00",  # Custom green
    decreasing_line_color="#FF0000",  # Custom red
    ...
))
```

### Adding New Metrics

Edit `src/botclave/dashboard/components/metrics_panel.py`:

```python
# Add new metric
st.metric("Custom Metric", value)
```

## Performance Tips

1. **Lookback Period**: Use smaller lookback (100-200) for faster analysis
2. **Refresh Interval**: 5-10 seconds is optimal; 1 second may be excessive
3. **Timeframe**: Higher timeframes (1h, 4h) require less frequent refreshes
4. **Browser**: Use modern browsers (Chrome, Firefox, Edge) for best performance

## Troubleshooting

### Data Not Loading

**Problem**: Clicking "Load Data" shows error

**Solutions**:
- Check internet connection
- Verify Binance API is accessible
- Try a different symbol or timeframe
- Check logs for specific error messages

### No Signals Generated

**Problem**: Analysis completes but no signals appear

**Reasons**:
- Market conditions don't meet criteria
- Filters are too strict (lower min_confidence or min_rr)
- Not enough data (increase lookback period)

**Solutions**:
- Lower minimum confidence to 0.6
- Lower minimum RR to 2.0
- Increase lookback to 300+ candles
- Try different timeframe

### Chart Not Updating

**Problem**: Chart shows old data

**Solutions**:
- Ensure auto-refresh is enabled
- Manually click "Load Data"
- Check browser console for errors
- Refresh the page (F5)

### Alerts Not Working

**Problem**: No sound or notifications

**Solutions**:
- Enable alerts in sidebar
- Enable sound alerts checkbox
- Check browser audio permissions
- Ensure browser allows auto-play

## Best Practices

1. **Start with 15m timeframe**: Good balance of detail and stability
2. **Use minimum RR of 3.0**: Ensures positive expectancy
3. **Wait for confluence**: Don't trade on single indicator
4. **Monitor absorption zones**: They indicate smart money activity
5. **Review signal history**: Learn from past signals
6. **Export signals regularly**: Keep records for analysis
7. **Adjust filters based on market**: Volatile markets may need looser filters

## Advanced Features

### Multi-Symbol Monitoring

Run multiple dashboard instances with different symbols:

```bash
# Terminal 1
streamlit run src/botclave/dashboard/app.py

# Terminal 2 (with different config)
streamlit run src/botclave/dashboard/app.py -- --symbol ETH/USDT
```

### Custom Alert Notifications

Extend `AlertManager` to add:
- Email notifications
- Telegram alerts
- Slack integration
- Push notifications

Example in `utils/alert_manager.py`:

```python
def _send_email_alert(self, signal: Signal):
    # Email sending logic
    pass

def _send_telegram_alert(self, signal: Signal):
    # Telegram bot logic
    pass
```

### Backtesting Integration

Compare dashboard signals with backtest results:

1. Run backtest using `scripts/backtest.py`
2. Export signals from dashboard
3. Compare signal performance

## API Reference

### DataLoader

```python
data_loader = DataLoader()

# Synchronous fetch
df = data_loader.fetch_ohlcv_sync("BTC/USDT", "15m", 200)

# Generate footprints
footprints = data_loader.generate_footprints(df)

# Calculate metrics
metrics = data_loader.calculate_orderflow_metrics(df, footprints)
```

### AlertManager

```python
alert_manager = AlertManager()

# Process signal
alert_manager.process_signal(signal)

# Get current signal
current = alert_manager.get_current_signal()

# Get signal history
history = alert_manager.get_signal_history()

# Get statistics
stats = alert_manager.get_alert_stats()
```

## Contributing

To contribute to the dashboard:

1. Follow the existing code style (Black formatter, 88 char line limit)
2. Add type hints to all functions
3. Include docstrings (Google style)
4. Test components before submitting
5. Update documentation for new features

## License

Part of the Botclave project. See main LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Check the main README.md
- Review existing GitHub issues
- Open a new issue with detailed description
- Include error logs and steps to reproduce

## Changelog

### Version 1.0.0
- Initial release
- Real-time alerts
- SMC analysis charts
- Orderflow visualization
- Signal history and export
- Auto-refresh functionality
- Configurable filters

---

**Happy Trading! 🚀**
