"""
Chart Panel Component

Creates interactive candlestick charts with SMC (Smart Money Concepts) overlays.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Dict, List

from botclave.engine.indicators import (
    SMCIndicator,
    Swing,
    FairValueGap,
    OrderBlock,
)
from botclave.engine.strategy import Signal


def create_smc_chart(
    df: pd.DataFrame,
    smc_analysis: Dict,
    current_signal: Optional[Signal] = None,
) -> go.Figure:
    """
    Create candlestick chart with SMC overlays.

    Args:
        df: DataFrame with OHLCV data
        smc_analysis: Dictionary with SMC indicators from SMCIndicator.analyze()
        current_signal: Current trading signal (optional)

    Returns:
        Plotly Figure with candlestick and SMC overlays
    """
    fig = go.Figure()

    # 1. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # 2. Swing highs (red triangle up)
    swings: List[Swing] = smc_analysis.get("swings", [])
    swing_highs = [s for s in swings if s.swing_type == "high"]

    if swing_highs:
        fig.add_trace(
            go.Scatter(
                x=df.index[[s.index for s in swing_highs]],
                y=[s.price for s in swing_highs],
                mode="markers",
                name="Swing High",
                marker=dict(size=10, color="red", symbol="triangle-up"),
                hovertemplate="<b>Swing High</b><br>Price: %{y:.2f}<extra></extra>",
            )
        )

    # 3. Swing lows (green triangle down)
    swing_lows = [s for s in swings if s.swing_type == "low"]

    if swing_lows:
        fig.add_trace(
            go.Scatter(
                x=df.index[[s.index for s in swing_lows]],
                y=[s.price for s in swing_lows],
                mode="markers",
                name="Swing Low",
                marker=dict(size=10, color="green", symbol="triangle-down"),
                hovertemplate="<b>Swing Low</b><br>Price: %{y:.2f}<extra></extra>",
            )
        )

    # 4. Fair Value Gaps (shaded zones)
    active_ffg: List[FairValueGap] = smc_analysis.get("active_ffg", [])

    for ffg in active_ffg:
        color = "rgba(0, 255, 0, 0.15)" if ffg.direction == "bullish" else "rgba(255, 0, 0, 0.15)"

        fig.add_hrect(
            y0=ffg.bottom_price,
            y1=ffg.top_price,
            fillcolor=color,
            opacity=0.3,
            line_width=0,
            annotation_text=f"FFG {ffg.direction}",
            annotation_position="right top",
            annotation_font_size=10,
            annotation_font_color="white",
        )

    # 5. Order Blocks (horizontal lines)
    order_blocks: List[OrderBlock] = smc_analysis.get("order_blocks", [])

    for ob in order_blocks:
        # Only show unmitigated blocks
        if ob.mitigated_index is None:
            color = "purple" if ob.direction == "bullish" else "orange"

            # Add shaded region for OB
            fig.add_hrect(
                y0=ob.low_price,
                y1=ob.high_price,
                fillcolor=color,
                opacity=0.15,
                line_width=1,
                line_color=color,
                annotation_text=f"OB {ob.direction}",
                annotation_position="right top",
                annotation_font_size=9,
            )

    # 6. Entry/SL/TP lines if there's an active signal
    if current_signal and current_signal.entry_setup:
        # Entry (green dashed)
        fig.add_hline(
            y=current_signal.entry_setup.entry_price,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text="ENTRY",
            annotation_position="left",
            annotation_font=dict(size=10, color="green"),
        )

        # Stop Loss (red solid)
        fig.add_hline(
            y=current_signal.entry_setup.stop_loss_price,
            line_dash="solid",
            line_color="red",
            line_width=3,
            annotation_text="SL",
            annotation_position="left",
            annotation_font=dict(size=10, color="red"),
        )

        # Take Profit (blue solid)
        fig.add_hline(
            y=current_signal.entry_setup.take_profit_price,
            line_dash="solid",
            line_color="blue",
            line_width=2,
            annotation_text="TP",
            annotation_position="left",
            annotation_font=dict(size=10, color="blue"),
        )

    # Layout settings
    fig.update_layout(
        title="Price Chart with SMC Analysis",
        yaxis_title="Price",
        xaxis_title="Time",
        template="plotly_dark",
        hovermode="x unified",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_cvd_chart(
    df: pd.DataFrame,
    cvd_column: str = "cvd",
) -> go.Figure:
    """
    Create Cumulative Volume Delta chart.

    Args:
        df: DataFrame with CVD data
        cvd_column: Name of CVD column

    Returns:
        Plotly Figure with CVD chart
    """
    fig = go.Figure()

    if cvd_column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[cvd_column],
                mode="lines",
                name="CVD",
                line=dict(color="cyan", width=2),
                fill="tozeroy",
            )
        )

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            line_width=1,
        )

    fig.update_layout(
        title="Cumulative Volume Delta",
        xaxis_title="Time",
        yaxis_title="CVD",
        template="plotly_dark",
        hovermode="x unified",
        height=300,
        showlegend=True,
    )

    return fig
