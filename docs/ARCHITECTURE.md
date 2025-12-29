# Architecture Documentation

## System Overview

BOTCLAVE is designed with a modular architecture that separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Dashboard Layer                          │
│              (Streamlit UI, Visualization)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                   Strategy Layer                             │
│        (Order Flow Strategy, Signal Generation)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    Engine Layer                              │
│  (Depth Analysis, Footprint, Indicators, DOM Builder)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                  Exchange Layer                              │
│         (Binance Connector, Order Manager)                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Engine Layer

#### depth.py
Analyzes order book depth to identify:
- Absorption zones
- Order book imbalances
- Support/resistance levels from liquidity concentration
- Depth heatmaps

**Key Classes:**
- `DepthAnalyzer`: Main analysis engine
- `DepthSnapshot`: Order book snapshot model
- `AbsorptionZone`: Detected absorption zone

#### footprint.py
Implements footprint charting:
- Build footprint bars from trade data
- Calculate Point of Control (POC)
- Detect volume imbalances
- Calculate cumulative delta

**Key Classes:**
- `FootprintChart`: Main footprint engine
- `FootprintBar`: Single bar representation
- `FootprintMetrics`: Calculated metrics

#### dom_builder.py
Real-time Depth of Market construction:
- Maintain live order book state
- Track liquidity levels over time
- Calculate bid/ask imbalance
- Identify significant levels

**Key Classes:**
- `DOMBuilder`: DOM state manager
- `DOMState`: Current market state
- `LiquidityLevel`: Significant liquidity level

#### indicators.py
Order flow indicators:
- Cumulative Volume Delta (CVD)
- Volume Profile
- Delta divergence
- Absorption detection
- Liquidity voids

**Key Classes:**
- `OrderFlowIndicators`: Indicator calculator
- `VolumeProfile`: Volume profile result
- `OrderFlowSignal`: Trading signal

#### strategy.py
Trading strategy implementation:
- Market structure analysis
- Order block detection
- Signal generation
- Position management
- Risk management

**Key Classes:**
- `OrderFlowStrategy`: Main strategy engine
- `TradeSignal`: Generated trading signal
- `Position`: Open position model

### 2. Exchange Layer

#### binance_connector.py
Binance exchange integration:
- CCXT-based connectivity
- OHLCV data fetching
- Order book retrieval
- Trade history
- WebSocket subscriptions

**Key Classes:**
- `BinanceConnector`: Exchange connector
- `ExchangeConfig`: Configuration model
- `MarketData`: Market data snapshot

#### order_manager.py
Order lifecycle management:
- Create market/limit orders
- Stop loss orders
- Take profit orders
- Order status tracking
- Order history

**Key Classes:**
- `OrderManager`: Order lifecycle manager
- `Order`: Order model
- `OrderStatus`: Order status enum

### 3. Backtest Layer

#### backtester.py
Historical strategy simulation:
- Realistic order execution
- Fee calculation
- Slippage modeling
- Equity curve tracking
- Performance metrics

**Key Classes:**
- `Backtester`: Backtest engine
- `BacktestResult`: Results with metrics
- `Trade`: Completed trade model

#### validator.py
Strategy validation:
- In-sample/out-of-sample testing
- Walk-forward analysis
- Monte Carlo simulation
- Overfitting detection
- Performance validation

**Key Classes:**
- `StrategyValidator`: Validation engine
- `ValidationResult`: Validation results
- `ValidationConfig`: Validation settings

### 4. Dashboard Layer

#### app.py
Main Streamlit application:
- Interactive UI
- Multiple tabs (Order Flow, Strategy, Depth)
- Real-time updates
- Configuration controls

#### charts.py
Visualization generation:
- Candlestick charts
- Footprint charts
- Volume profiles
- Equity curves
- Drawdown charts

**Key Classes:**
- `ChartGenerator`: Chart creation engine

#### metrics.py
Performance metrics calculation:
- Win rate
- Sharpe ratio
- Profit factor
- Max drawdown
- Trade statistics

**Key Classes:**
- `MetricsCalculator`: Metrics engine

## Data Flow

### 1. Historical Data Analysis

```
Raw Data → DepthAnalyzer → Footprint Chart → Indicators → Strategy → Signals
                                                              ↓
                                                         Backtester
                                                              ↓
                                                      Performance Metrics
```

### 2. Live Trading Flow

```
Exchange → WebSocket → DOM Builder → Real-time Analysis → Signal Generation
                                                               ↓
                                                         Order Manager
                                                               ↓
                                                         Execute Orders
```

### 3. Backtest Flow

```
Historical CSV → DataFrame → Signal Generation → Backtester → Results
                                                                  ↓
                                                            Validator
                                                                  ↓
                                                          Report Generation
```

## Configuration Management

Configuration is managed through YAML files:

1. **strategy_params.yaml**: Strategy settings
   - Risk management parameters
   - Order flow thresholds
   - Indicator settings
   - Entry/exit rules

2. **exchange_config.yaml**: Exchange settings
   - API credentials
   - Rate limits
   - Fee structure
   - Symbol configurations

## Error Handling

The system implements multi-layer error handling:

1. **Exchange Layer**: Network errors, API failures
2. **Engine Layer**: Invalid data, calculation errors
3. **Strategy Layer**: Signal validation errors
4. **Dashboard Layer**: UI exceptions

## Performance Considerations

### Optimization Strategies

1. **Data Caching**: Historical data cached to disk
2. **Vectorized Operations**: NumPy/Pandas for performance
3. **Async Operations**: Async I/O for exchange calls
4. **Batch Processing**: Bulk order book updates

### Scalability

- Modular design allows horizontal scaling
- Stateless components enable distributed deployment
- Configuration-driven for easy adaptation

## Testing Strategy

1. **Unit Tests**: Component-level testing
2. **Integration Tests**: Cross-component testing
3. **Backtest Validation**: Historical performance
4. **Paper Trading**: Live simulation

## Security Considerations

1. **API Keys**: Environment variables only
2. **Testnet First**: Default to testnet
3. **Input Validation**: Pydantic models
4. **Error Logging**: Comprehensive logging

## Future Enhancements

1. **Machine Learning**: Pattern recognition
2. **Multi-Exchange**: Support more exchanges
3. **Portfolio Mode**: Multi-asset trading
4. **Cloud Deployment**: AWS/GCP integration
5. **Real-time Alerts**: Telegram/Email notifications

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `ccxt`: Exchange connectivity
- `pydantic`: Data validation
- `plotly`: Visualization
- `streamlit`: Dashboard

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Type checking

## Design Principles

1. **Modularity**: Loosely coupled components
2. **Type Safety**: Full type hints
3. **Testability**: Comprehensive test coverage
4. **Documentation**: Clear docstrings
5. **Configuration**: External configuration
6. **Error Handling**: Graceful degradation

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                 Load Balancer                    │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐  ┌────────▼────────┐
│  Dashboard     │  │   Strategy      │
│  Instance      │  │   Engine        │
└───────┬────────┘  └────────┬────────┘
        │                    │
        └──────────┬─────────┘
                   │
        ┌──────────▼───────────┐
        │  Exchange Connector   │
        └──────────────────────┘
```

## Monitoring & Logging

1. **Application Logs**: Structured logging
2. **Performance Metrics**: Execution time tracking
3. **Error Tracking**: Exception monitoring
4. **Trade Logging**: Complete audit trail

---

*Last Updated: 2024-01-15*
