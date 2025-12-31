"""
Orderflow Panel Component

Displays order flow analysis including heatmap, footprint, and absorption zones.
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional

from botclave.engine.footprint import KlineFootprint


def create_order_book_heatmap(
    depth_data: Optional[pd.DataFrame] = None,
    current_price: Optional[float] = None,
) -> go.Figure:
    """
    Create order book heatmap visualization.

    Args:
        depth_data: DataFrame with depth data (price, volume, timestamp)
        current_price: Current market price for reference line

    Returns:
        Plotly Figure with heatmap
    """
    fig = go.Figure()

    if depth_data is not None and not depth_data.empty:
        # Create heatmap from actual depth data
        if "price" in depth_data.columns and "volume" in depth_data.columns:
            # Reshape data for heatmap
            price_levels = depth_data["price"].unique()
            time_points = depth_data.get("timestamp", pd.RangeIndex(len(depth_data)))

            # Create volume matrix
            z_data = []
            for price in price_levels:
                price_data = depth_data[depth_data["price"] == price]
                z_data.append(price_data["volume"].tolist())

            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    y=price_levels,
                    x=time_points if len(z_data[0]) == len(time_points) else list(
                        range(len(z_data[0]))
                    ),
                    colorscale="RdYlGn_r",
                    name="Volume",
                    colorbar=dict(title="Volume"),
                )
            )
    else:
        # Generate placeholder data for demonstration
        z_data = np.random.rand(50, 20)

        fig.add_trace(
            go.Heatmap(
                z=z_data,
                colorscale="RdYlGn_r",
                name="Volume",
                colorbar=dict(title="Volume"),
            )
        )

    # Add current price line
    if current_price is not None:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="cyan",
            line_width=2,
            annotation_text="Current Price",
            annotation_position="right",
        )

    fig.update_layout(
        title="Order Book Heatmap (DOM)",
        yaxis_title="Price",
        xaxis_title="Time",
        height=500,
        template="plotly_dark",
    )

    return fig


def create_footprint_chart(
    df: pd.DataFrame,
    footprints: List[KlineFootprint],
    last_n: int = 20,
) -> go.Figure:
    """
    Create footprint chart showing buy/sell imbalances.

    Args:
        df: DataFrame with OHLCV data
        footprints: List of KlineFootprint objects
        last_n: Number of recent candles to display

    Returns:
        Plotly Figure with footprint visualization
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Footprint Chart", "Volume Imbalance"),
        vertical_spacing=0.05,
    )

    # Limit to last N candles
    df_tail = df.tail(last_n)
    footprints_tail = footprints[-last_n:] if footprints else []

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df_tail.index,
            open=df_tail["open"],
            high=df_tail["high"],
            low=df_tail["low"],
            close=df_tail["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # Volume imbalance bars
    if len(footprints_tail) == len(df_tail):
        buy_volumes = []
        sell_volumes = []
        imbalances = []

        for fp in footprints_tail:
            stats = fp.get_stats()
            buy_volumes.append(stats.get("total_buy", 0))
            sell_volumes.append(stats.get("total_sell", 0))
            total = stats.get("total_volume", 0)
            if total > 0:
                imbalance = (stats.get("total_buy", 0) - stats.get("total_sell", 0)) / total
            else:
                imbalance = 0
            imbalances.append(imbalance)

        # Colors based on imbalance
        colors = ["green" if imp > 0 else "red" for imp in imbalances]

        fig.add_trace(
            go.Bar(
                x=df_tail.index,
                y=buy_volumes,
                name="Buy Volume",
                marker_color="rgba(0, 255, 0, 0.5)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_tail.index,
                y=[-v for v in sell_volumes],
                name="Sell Volume",
                marker_color="rgba(255, 0, 0, 0.5)",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title="Footprint Analysis - Buy/Sell Imbalance",
        height=700,
        template="plotly_dark",
        showlegend=True,
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def show_orderflow_panel(
    df: pd.DataFrame,
    footprints: List[KlineFootprint],
    absorption_zones: Dict,
    current_price: float,
) -> None:
    """
    Display orderflow analysis panel with heatmap and footprint charts.

    Args:
        df: DataFrame with OHLCV data
        footprints: List of KlineFootprint objects
        absorption_zones: Dictionary of absorption zones {price: type}
        current_price: Current market price
    """
    # Top section: Order Book Heatmap and Footprint side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Order Book Heatmap (DOM)")
        depth_fig = create_order_book_heatmap(current_price=current_price)
        st.plotly_chart(depth_fig, use_container_width=True)
        st.caption(
            "Shows order book depth aggregated by price level. "
            "Brighter colors indicate higher volume."
        )

    with col2:
        st.subheader("Footprint Analysis")
        if footprints:
            footprint_fig = create_footprint_chart(df, footprints, last_n=20)
            st.plotly_chart(footprint_fig, use_container_width=True)
            st.caption(
                "Shows buy vs sell volume imbalance. Green = buying pressure, Red = selling pressure."
            )
        else:
            st.info("Footprint data not available")
            st.caption(
                "Load data with footprint analysis to see volume imbalances"
            )

    # Bottom section: Absorption zones
    st.markdown("---")
    st.subheader("Absorption Zones")

    if absorption_zones:
        col1, col2 = st.columns(2)

        with col1:
            buy_zones = {p: t for p, t in absorption_zones.items() if t == "BUY"}
            if buy_zones:
                st.info(f"**Buy Absorption Zones ({len(buy_zones)})**\n\n" +
                        "\n".join([f"• ${price:,.2f}" for price in sorted(buy_zones.keys())]))
            else:
                st.info("No buy absorption zones detected")

        with col2:
            sell_zones = {p: t for p, t in absorption_zones.items() if t == "SELL"}
            if sell_zones:
                st.info(f"**Sell Absorption Zones ({len(sell_zones)})**\n\n" +
                        "\n".join([f"• ${price:,.2f}" for price in sorted(sell_zones.keys())]))
            else:
                st.info("No sell absorption zones detected")

        st.caption(
            "Absorption zones indicate where smart money is accumulating positions "
            "(high volume with little price movement)"
        )
    else:
        st.info("No absorption zones detected")
        st.caption("Analyze more candles to detect absorption patterns")
