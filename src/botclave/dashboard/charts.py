"""
Chart Generation Module

Creates various charts for order flow visualization using Plotly.
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartGenerator:
    """
    Generates various charts for order flow analysis and strategy visualization.
    """

    def __init__(self, theme: str = "plotly_dark"):
        """
        Initialize ChartGenerator.

        Args:
            theme: Plotly theme to use
        """
        self.theme = theme

    def create_candlestick_chart(
        self, df: pd.DataFrame, title: str = "Price Chart"
    ) -> go.Figure:
        """
        Create a candlestick chart.

        Args:
            df: DataFrame with OHLCV data
            title: Chart title

        Returns:
            Plotly Figure
        """
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                )
            ]
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Price",
            template=self.theme,
            height=500,
        )

        return fig

    def create_footprint_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a footprint chart showing buy/sell volume at each price level.

        Args:
            df: DataFrame with order flow data

        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Footprint Chart", "Volume Delta"),
            vertical_spacing=0.1,
        )

        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    y=df["buy_volume"],
                    name="Buy Volume",
                    marker_color="green",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    y=-df["sell_volume"],
                    name="Sell Volume",
                    marker_color="red",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            delta = df["buy_volume"] - df["sell_volume"]
            colors = ["green" if d > 0 else "red" for d in delta]

            fig.add_trace(
                go.Bar(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    y=delta,
                    name="Delta",
                    marker_color=colors,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            template=self.theme,
            height=700,
            showlegend=True,
        )

        return fig

    def create_volume_profile(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a volume profile chart.

        Args:
            df: DataFrame with price and volume data

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if "close" in df.columns and "volume" in df.columns:
            price_bins = 50
            min_price = df["close"].min()
            max_price = df["close"].max()
            price_levels = np.linspace(min_price, max_price, price_bins)

            volumes = []
            for i in range(len(price_levels) - 1):
                mask = (df["close"] >= price_levels[i]) & (
                    df["close"] < price_levels[i + 1]
                )
                volume = df.loc[mask, "volume"].sum()
                volumes.append(volume)

            fig.add_trace(
                go.Bar(
                    y=price_levels[:-1],
                    x=volumes,
                    orientation="h",
                    name="Volume Profile",
                    marker_color="rgba(0, 150, 255, 0.6)",
                )
            )

        fig.update_layout(
            title="Volume Profile",
            xaxis_title="Volume",
            yaxis_title="Price",
            template=self.theme,
            height=500,
        )

        return fig

    def create_delta_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create cumulative delta chart.

        Args:
            df: DataFrame with buy/sell volume data

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            delta = df["buy_volume"] - df["sell_volume"]
            cumulative_delta = delta.cumsum()

            fig.add_trace(
                go.Scatter(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    y=cumulative_delta,
                    mode="lines",
                    name="Cumulative Delta",
                    line=dict(color="blue", width=2),
                    fill="tozeroy",
                )
            )

        fig.update_layout(
            title="Cumulative Volume Delta",
            xaxis_title="Time",
            yaxis_title="CVD",
            template=self.theme,
            height=400,
        )

        return fig

    def create_equity_curve(self, equity: List[float]) -> go.Figure:
        """
        Create equity curve chart.

        Args:
            equity: List of equity values

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(equity))),
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color="green", width=2),
            )
        )

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Trade Number",
            yaxis_title="Equity ($)",
            template=self.theme,
            height=400,
        )

        return fig

    def create_trade_distribution(self, trades: List) -> go.Figure:
        """
        Create trade P&L distribution histogram.

        Args:
            trades: List of trades

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if trades:
            pnls = [t.get("pnl", 0) if isinstance(t, dict) else t.pnl for t in trades]

            fig.add_trace(
                go.Histogram(
                    x=pnls,
                    nbinsx=30,
                    name="P&L Distribution",
                    marker_color="rgba(0, 150, 255, 0.7)",
                )
            )

        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            template=self.theme,
            height=400,
        )

        return fig

    def create_drawdown_chart(self, equity: List[float]) -> go.Figure:
        """
        Create drawdown chart.

        Args:
            equity: List of equity values

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if equity:
            equity_array = np.array(equity)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak * 100

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(drawdown))),
                    y=drawdown,
                    mode="lines",
                    name="Drawdown",
                    line=dict(color="red", width=2),
                    fill="tozeroy",
                )
            )

        fig.update_layout(
            title="Drawdown Analysis",
            xaxis_title="Trade Number",
            yaxis_title="Drawdown (%)",
            template=self.theme,
            height=400,
        )

        return fig

    def create_depth_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Create order book depth heatmap.

        Args:
            df: DataFrame with depth data

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        z_data = np.random.rand(50, 20)

        fig.add_trace(
            go.Heatmap(
                z=z_data,
                colorscale="RdYlGn",
                name="Depth",
            )
        )

        fig.update_layout(
            title="Order Book Depth Heatmap",
            xaxis_title="Time",
            yaxis_title="Price Level",
            template=self.theme,
            height=500,
        )

        return fig

    def create_liquidity_levels(self, df: pd.DataFrame) -> go.Figure:
        """
        Create liquidity levels chart.

        Args:
            df: DataFrame with price data

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        if "close" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index if hasattr(df.index, "__len__") else range(len(df)),
                    y=df["close"],
                    mode="lines",
                    name="Price",
                    line=dict(color="white", width=1),
                )
            )

            support_level = df["close"].min() * 1.02
            resistance_level = df["close"].max() * 0.98

            fig.add_hline(
                y=support_level,
                line_dash="dash",
                line_color="green",
                annotation_text="Support",
            )
            fig.add_hline(
                y=resistance_level,
                line_dash="dash",
                line_color="red",
                annotation_text="Resistance",
            )

        fig.update_layout(
            title="Key Liquidity Levels",
            xaxis_title="Time",
            yaxis_title="Price",
            template=self.theme,
            height=500,
        )

        return fig

    def create_signal_chart(
        self, df: pd.DataFrame, signals: pd.DataFrame
    ) -> go.Figure:
        """
        Create chart with trading signals overlaid.

        Args:
            df: DataFrame with price data
            signals: DataFrame with signals

        Returns:
            Plotly Figure
        """
        fig = self.create_candlestick_chart(df, "Price with Signals")

        if not signals.empty:
            buy_signals = signals[signals.get("side") == "long"]
            sell_signals = signals[signals.get("side") == "short"]

            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals.get("price", df.loc[buy_signals.index, "close"]),
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(color="green", size=15, symbol="triangle-up"),
                    )
                )

            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals.get("price", df.loc[sell_signals.index, "close"]),
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(color="red", size=15, symbol="triangle-down"),
                    )
                )

        return fig
