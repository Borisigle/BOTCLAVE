"""
Streamlit Dashboard Application

Professional trading dashboard with real-time alerts, SMC analysis,
and order flow visualization for BTC/XAU trading.
"""

import streamlit as st
import pandas as pd
import time
from typing import Optional

# Import components
from .components.alert_panel import show_alert_panel
from .components.chart_panel import create_smc_chart
from .components.metrics_panel import show_metrics_panel
from .components.orderflow_panel import show_orderflow_panel
from .components.signals_log import show_signals_log, export_signals_to_csv

# Import utilities
from .utils.data_loader import DataLoader
from .utils.alert_manager import AlertManager

# Import bot engine
from botclave.engine.indicators import SMCIndicator
from botclave.engine.strategy import TradingStrategy


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "signals" not in st.session_state:
        st.session_state.signals = []
    if "smc_analysis" not in st.session_state:
        st.session_state.smc_analysis = {}
    if "orderflow_metrics" not in st.session_state:
        st.session_state.orderflow_metrics = {}
    if "footprints" not in st.session_state:
        st.session_state.footprints = []
    if "absorption_zones" not in st.session_state:
        st.session_state.absorption_zones = {}
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader()
    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager()


def render_sidebar() -> tuple:
    """
    Render sidebar configuration controls.

    Returns:
        Tuple of (symbol, timeframe, lookback, min_confidence, min_rr, auto_refresh)
    """
    st.sidebar.title("⚙️ Configuration")

    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Symbol",
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        index=0,
    )

    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h"],
        index=2,
    )

    # Lookback period
    lookback = st.sidebar.slider(
        "Lookback Period (candles)",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
    )

    st.sidebar.markdown("---")

    # Signal filters
    st.sidebar.markdown("### 📊 Signal Filters")

    min_confidence = st.sidebar.slider(
        "Min Confidence",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence level to show signals",
    )

    min_rr = st.sidebar.slider(
        "Min Risk/Reward",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="Minimum risk-reward ratio for signals",
    )

    st.sidebar.markdown("---")

    # Auto-refresh settings
    st.sidebar.markdown("### 🔄 Auto-refresh")

    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-refresh",
        value=True,
        help="Automatically refresh data at specified interval",
    )

    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=1,
        max_value=60,
        value=5,
        disabled=not auto_refresh,
    )

    st.sidebar.markdown("---")

    # Alert settings
    if "alert_manager" in st.session_state:
        st.session_state.alert_manager.render_settings()

    return (
        symbol,
        timeframe,
        lookback,
        min_confidence,
        min_rr,
        auto_refresh,
        refresh_interval,
    )


def load_and_analyze_data(
    symbol: str,
    timeframe: str,
    lookback: int,
    min_confidence: float,
    min_rr: float,
) -> None:
    """
    Load market data and perform analysis.

    Args:
        symbol: Trading pair symbol
        timeframe: Chart timeframe
        lookback: Number of candles to analyze
        min_confidence: Minimum confidence threshold
        min_rr: Minimum risk-reward ratio
    """
    data_loader = st.session_state.data_loader

    with st.spinner(f"Loading {lookback} candles for {symbol} ({timeframe})..."):
        # Fetch OHLCV data
        df = data_loader.fetch_ohlcv_sync(symbol, timeframe, lookback)

        if df.empty:
            st.error("Failed to load market data. Please try again.")
            return

        st.session_state.df = df

        # Generate footprints (simplified for now)
        footprints = data_loader.generate_footprints(df)
        st.session_state.footprints = footprints

        # Calculate orderflow metrics
        orderflow_metrics = data_loader.calculate_orderflow_metrics(df, footprints)
        st.session_state.orderflow_metrics = orderflow_metrics

        # Perform SMC analysis
        smc_indicator = SMCIndicator()
        smc_analysis = smc_indicator.analyze(df)
        st.session_state.smc_analysis = smc_analysis

        # Run trading strategy
        strategy = TradingStrategy(
            min_confidence=min_confidence,
            min_rr=min_rr,
        )

        result = strategy.analyze(df, footprints)
        st.session_state.absorption_zones = result.get("absorption_zones", {})

        # Get latest signal
        signals = result.get("signals", [])
        st.session_state.signals = signals

        # Update alert manager with latest signal
        if signals:
            current_signal = signals[-1] if len(signals) > 0 else None
            st.session_state.alert_manager.process_signal(current_signal)
        else:
            st.session_state.alert_manager.process_signal(None)

        st.session_state.data_loaded = True

        st.success(
            f"✅ Loaded {len(df)} candles for {symbol} ({timeframe})\n"
            f"📊 Found {len(smc_analysis.get('swings', []))} swings, "
            f"{len(smc_analysis.get('active_ffg', []))} active FFGs, "
            f"{len(smc_analysis.get('order_blocks', []))} order blocks\n"
            f"🎯 Generated {len(signals)} signal(s)"
        )


def main():
    """Main dashboard application."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="BOTCLAVE - BTC/XAU Trader",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Page header
    st.title("🤖 BOTCLAVE - BTC/XAU Order Flow Trading Bot")
    st.markdown(
        """
        Professional trading dashboard with Smart Money Concepts (SMC) analysis,
        order flow visualization, and real-time signal alerts.
        """
    )
    st.markdown("---")

    # Render sidebar and get configuration
    (
        symbol,
        timeframe,
        lookback,
        min_confidence,
        min_rr,
        auto_refresh,
        refresh_interval,
    ) = render_sidebar()

    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        load_and_analyze_data(symbol, timeframe, lookback, min_confidence, min_rr)

    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.info("👈 Configure settings and click 'Load Data' to start analysis")
        st.caption(
            "Select a symbol, timeframe, and adjust filters, then load data to begin"
        )
        return

    # Get current price
    df = st.session_state.df
    current_price = df["close"].iloc[-1]

    # Get current signal from alert manager
    alert_manager = st.session_state.alert_manager
    current_signal = alert_manager.get_current_signal()

    # SECTION 1: ALERT PANEL (Top - Sticky)
    show_alert_panel(current_signal)

    st.markdown("---")

    # SECTION 2: TABS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Chart", "📈 Metrics", "🌊 Orderflow", "📋 Signals Log"]
    )

    # TAB 1: CHART
    with tab1:
        st.subheader("Candlestick + SMC Analysis")

        # Create and display chart
        smc_analysis = st.session_state.smc_analysis
        fig = create_smc_chart(df, smc_analysis, current_signal)
        st.plotly_chart(fig, use_container_width=True)

        # Additional chart info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(
                f"**Current Price:** ${current_price:,.2f}\n\n"
                f"**ATR:** ${df['high'].iloc[-1] - df['low'].iloc[-1]:,.2f}"
            )

        with col2:
            st.info(
                f"**Active FFGs:** {len(smc_analysis.get('active_ffg', []))}\n\n"
                f"**Order Blocks:** {len(smc_analysis.get('order_blocks', []))}"
            )

        with col3:
            bos = smc_analysis.get("last_bos")
            bias = bos.direction if bos else "neutral"
            st.info(
                f"**Swings:** {len(smc_analysis.get('swings', []))}\n\n"
                f"**Market Bias:** {bias.upper()}"
            )

    # TAB 2: METRICS
    with tab2:
        st.subheader("Live Market Metrics")

        orderflow_metrics = st.session_state.orderflow_metrics
        absorption_zones = st.session_state.absorption_zones

        show_metrics_panel(
            df,
            smc_analysis,
            orderflow_metrics,
            current_price,
        )

    # TAB 3: ORDERFLOW
    with tab3:
        st.subheader("Order Flow Analysis")

        footprints = st.session_state.footprints

        show_orderflow_panel(
            df,
            footprints,
            absorption_zones,
            current_price,
        )

    # TAB 4: SIGNALS LOG
    with tab4:
        st.subheader("Signal History")

        signals_history = st.session_state.signals
        show_signals_log(signals_history)

        # Export button
        if signals_history:
            st.markdown("---")
            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("📥 Export to CSV"):
                    csv_data = export_signals_to_csv(signals_history)
                    st.download_button(
                        label="Download Signals CSV",
                        data=csv_data,
                        file_name=f"signals_{symbol}_{timeframe}.csv",
                        mime="text/csv",
                    )

    # Auto-refresh
    if auto_refresh and st.session_state.data_loaded:
        time.sleep(refresh_interval)
        load_and_analyze_data(symbol, timeframe, lookback, min_confidence, min_rr)
        st.rerun()


if __name__ == "__main__":
    main()
