"""
Metrics Panel Component

Displays live market metrics and analysis results.
"""

import streamlit as st
import pandas as pd
from typing import Dict


def show_metrics_panel(
    df: pd.DataFrame,
    smc_analysis: Dict,
    orderflow_metrics: Dict,
    current_price: float,
) -> None:
    """
    Display live market metrics in a structured layout.

    Args:
        df: DataFrame with OHLCV data
        smc_analysis: Dictionary with SMC analysis results
        orderflow_metrics: Dictionary with orderflow metrics
        current_price: Current market price
    """
    # Row 1: Market structure metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Market bias
        last_bos = smc_analysis.get("last_bos")
        if last_bos:
            bias = last_bos.direction.upper()
            bias_emoji = "🟢" if bias == "BULLISH" else "🔴"
        else:
            bias = "NEUTRAL"
            bias_emoji = "⚪"

        st.metric("Market Bias", f"{bias_emoji} {bias}")

    with col2:
        st.metric("Current Price", f"${current_price:,.2f}")

    with col3:
        # ATR (volatility) - calculate if not present
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
        else:
            # Simple ATR calculation
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1] if len(df) >= 14 else high_low.iloc[-1]

        st.metric("ATR (Volatility)", f"${atr:,.2f}")

    with col4:
        # Confluence level (from analysis)
        confluence = smc_analysis.get("confluence_level", 0)
        st.metric("Confluence Level", f"{confluence * 100:.0f}%")

    st.markdown("---")

    # Row 2: Orderflow metrics
    st.subheader("Orderflow Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        cvd = orderflow_metrics.get("cvd", 0)
        cvd_delta = None
        if len(df) > 1 and "cvd" in df.columns:
            prev_cvd = df["cvd"].iloc[-2]
            cvd_delta = f"{cvd - prev_cvd:+,.0f}"

        cvd_display = f"{cvd:+,.0f}"
        if abs(cvd) > 1000000:
            cvd_display = f"{cvd / 1000000:+.1f}M"

        st.metric("CVD", cvd_display, delta=cvd_delta)

    with col2:
        buy_sell_ratio = orderflow_metrics.get("buy_sell_ratio", 1.0)
        st.metric("Buy/Sell Ratio", f"{buy_sell_ratio:.2f}x")

    with col3:
        # Absorption zones count
        absorption_zones = smc_analysis.get("absorption_zones", {})
        st.metric("Absorption Zones", f"{len(absorption_zones)}")

    st.markdown("---")

    # Row 3: SMC indicators count
    st.subheader("SMC Indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Active FFGs
        active_ffg = smc_analysis.get("active_ffg", [])
        bullish_ffg = [f for f in active_ffg if f.direction == "bullish"]
        bearish_ffg = [f for f in active_ffg if f.direction == "bearish"]
        ffg_text = f"{len(bullish_ffg)} 🟢 / {len(bearish_ffg)} 🔴"
        st.metric("Active FFGs", ffg_text)

    with col2:
        # Order Blocks
        order_blocks = smc_analysis.get("order_blocks", [])
        active_obs = [ob for ob in order_blocks if ob.mitigated_index is None]
        bullish_ob = [ob for ob in active_obs if ob.direction == "bullish"]
        bearish_ob = [ob for ob in active_obs if ob.direction == "bearish"]
        ob_text = f"{len(bullish_ob)} 🟢 / {len(bearish_ob)} 🔴"
        st.metric("Order Blocks", ob_text)

    with col3:
        # Swings detected
        swings = smc_analysis.get("swings", [])
        swing_highs = [s for s in swings if s.swing_type == "high"]
        swing_lows = [s for s in swings if s.swing_type == "low"]
        swings_text = f"{len(swing_highs)} 🔺 / {len(swing_lows)} 🔻"
        st.metric("Swings", swings_text)

    # Row 4: Market structure breakdown
    st.markdown("---")
    st.subheader("Market Structure")

    col1, col2 = st.columns(2)

    with col1:
        # Last swing analysis
        last_swing = smc_analysis.get("last_swing")
        if last_swing:
            st.info(
                f"Last Swing: {last_swing.swing_type.upper()} @ ${last_swing.price:,.2f}"
            )
        else:
            st.info("No swings detected yet")

    with col2:
        # BOS analysis
        last_bos = smc_analysis.get("last_bos")
        if last_bos:
            st.info(
                f"Last BOS: {last_bos.direction.upper()} @ ${last_bos.broken_level:,.2f}"
            )
        else:
            st.info("No BOS detected yet")

    # Additional metrics if available
    if "liquidity" in smc_analysis:
        liquidity = smc_analysis["liquidity"]
        st.markdown("---")
        st.subheader("Liquidity Zones")

        col1, col2 = st.columns(2)

        with col1:
            # Support levels
            supports = [l for l in liquidity if l.direction == "support"]
            if supports:
                support_text = "\n".join(
                    [f"${l.price:,.2f} (strength: {l.strength})" for l in supports[:3]]
                )
                st.info(f"Support Levels:\n{support_text}")

        with col2:
            # Resistance levels
            resistances = [l for l in liquidity if l.direction == "resistance"]
            if resistances:
                resistance_text = "\n".join(
                    [f"${l.price:,.2f} (strength: {l.strength})" for l in resistances[:3]]
                )
                st.info(f"Resistance Levels:\n{resistance_text}")
