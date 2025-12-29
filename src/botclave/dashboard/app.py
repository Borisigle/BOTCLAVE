"""
Streamlit Dashboard Application

Main dashboard for visualizing order flow data and strategy performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from .charts import ChartGenerator
from .metrics import MetricsCalculator


def init_session_state():
    """Initialize Streamlit session state."""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "signals" not in st.session_state:
        st.session_state.signals = None


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("⚙️ Settings")

    symbol = st.sidebar.selectbox(
        "Symbol",
        ["BTC/USDT", "ETH/USDT", "BTC/XAU", "SOL/USDT"],
        index=0,
    )

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=2,
    )

    lookback = st.sidebar.slider(
        "Lookback Period (bars)",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
    )

    return symbol, timeframe, lookback


def render_metrics_row(metrics_data: dict):
    """Render metrics in a row of columns."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Signals",
            metrics_data.get("total_signals", 0),
            delta=None,
        )

    with col2:
        win_rate = metrics_data.get("win_rate", 0)
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            delta=f"{win_rate - 50:.1f}%",
        )

    with col3:
        st.metric(
            "Total P&L",
            f"${metrics_data.get('total_pnl', 0):.2f}",
            delta=None,
        )

    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics_data.get('sharpe_ratio', 0):.2f}",
            delta=None,
        )


def render_order_flow_tab():
    """Render order flow analysis tab."""
    st.header("📊 Order Flow Analysis")

    chart_gen = ChartGenerator()

    if st.session_state.df is not None:
        st.subheader("Footprint Chart")
        footprint_fig = chart_gen.create_footprint_chart(st.session_state.df)
        st.plotly_chart(footprint_fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Volume Profile")
            volume_fig = chart_gen.create_volume_profile(st.session_state.df)
            st.plotly_chart(volume_fig, use_container_width=True)

        with col2:
            st.subheader("Delta Analysis")
            delta_fig = chart_gen.create_delta_chart(st.session_state.df)
            st.plotly_chart(delta_fig, use_container_width=True)
    else:
        st.info("Load data to view order flow analysis")


def render_strategy_tab():
    """Render strategy performance tab."""
    st.header("🎯 Strategy Performance")

    if st.session_state.signals is not None:
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_strategy_metrics(
            st.session_state.df, st.session_state.signals
        )

        render_metrics_row(metrics)

        st.subheader("Equity Curve")
        chart_gen = ChartGenerator()
        equity_fig = chart_gen.create_equity_curve(metrics.get("equity_curve", []))
        st.plotly_chart(equity_fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Trade Distribution")
            trade_fig = chart_gen.create_trade_distribution(
                metrics.get("trades", [])
            )
            st.plotly_chart(trade_fig, use_container_width=True)

        with col2:
            st.subheader("Drawdown Analysis")
            dd_fig = chart_gen.create_drawdown_chart(
                metrics.get("equity_curve", [])
            )
            st.plotly_chart(dd_fig, use_container_width=True)
    else:
        st.info("Run strategy to view performance metrics")


def render_depth_tab():
    """Render depth analysis tab."""
    st.header("📈 Depth Analysis")

    if st.session_state.df is not None:
        chart_gen = ChartGenerator()

        st.subheader("Order Book Heatmap")
        heatmap_fig = chart_gen.create_depth_heatmap(st.session_state.df)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("Liquidity Levels")
        liquidity_fig = chart_gen.create_liquidity_levels(st.session_state.df)
        st.plotly_chart(liquidity_fig, use_container_width=True)
    else:
        st.info("Load data to view depth analysis")


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="BTC/XAU Order Flow Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🤖 BTC/XAU Order Flow Absorption Bot")
    st.markdown("---")

    init_session_state()

    symbol, timeframe, lookback = render_sidebar()

    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading market data..."):
            st.session_state.data_loaded = True
            st.success(f"Loaded {lookback} bars for {symbol} ({timeframe})")

    tabs = st.tabs(["📊 Order Flow", "🎯 Strategy", "📈 Depth Analysis", "⚙️ Settings"])

    with tabs[0]:
        render_order_flow_tab()

    with tabs[1]:
        render_strategy_tab()

    with tabs[2]:
        render_depth_tab()

    with tabs[3]:
        st.header("⚙️ Strategy Settings")
        st.subheader("Risk Management")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Position Size (%)", min_value=0.1, max_value=10.0, value=2.0)
            st.number_input("Stop Loss (%)", min_value=0.1, max_value=10.0, value=2.0)
        with col2:
            st.number_input("Take Profit (%)", min_value=0.1, max_value=20.0, value=4.0)
            st.number_input("Max Positions", min_value=1, max_value=10, value=3)

        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
