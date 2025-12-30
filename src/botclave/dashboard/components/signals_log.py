"""
Signals Log Component

Displays a history of all generated trading signals.
"""

import streamlit as st
import pandas as pd
from typing import List
from datetime import datetime

from botclave.engine.strategy import Signal


def show_signals_log(signals_history: List[Signal]) -> None:
    """
    Display a table of all generated trading signals.

    Args:
        signals_history: List of Signal objects
    """
    st.subheader("📋 Signals History")

    if not signals_history:
        st.info("No signals generated yet")
        st.caption("Load data and run analysis to generate trading signals")
        return

    # Convert signals to DataFrame for display
    signals_data = []

    for signal in signals_history:
        # Format time nicely
        try:
            if isinstance(signal.time, str):
                time_str = signal.time
            else:
                time_str = str(signal.time)
        except Exception:
            time_str = "N/A"

        # Determine emoji based on signal type
        if "LONG" in signal.signal_type:
            emoji = "🟢"
        elif "SHORT" in signal.signal_type:
            emoji = "🔴"
        else:
            emoji = "⚪"

        # Format RR ratio
        rr_str = "-"
        if signal.entry_setup:
            rr_str = f"{signal.entry_setup.risk_reward_ratio:.1f}:1"

        # Truncate reason if too long
        reason = signal.reason[:60] + "..." if len(signal.reason) > 60 else signal.reason

        signals_data.append({
            "Time": time_str,
            "Signal": f"{emoji} {signal.signal_type}",
            "Price": f"${signal.price:,.2f}",
            "Confidence": f"{signal.confidence * 100:.0f}%",
            "RR": rr_str,
            "SMC": signal.smc_component,
            "Orderflow": signal.orderflow_component,
            "Reason": reason,
        })

    # Create DataFrame
    df_signals = pd.DataFrame(signals_data)

    # Display with search and filter
    col1, col2 = st.columns([1, 3])

    with col1:
        # Filter by signal type
        signal_types = ["All"] + list(df_signals["Signal"].str.replace("🟢 ", "")
                                                    .str.replace("🔴 ", "")
                                                    .str.replace("⚪ ", "")
                                                    .unique())
        selected_type = st.selectbox("Filter by Signal Type", signal_types)

    # Apply filter
    if selected_type != "All":
        df_filtered = df_signals[df_signals["Signal"].str.contains(selected_type)]
    else:
        df_filtered = df_signals

    # Display table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time": st.column_config.TextColumn("Time", width="medium"),
            "Signal": st.column_config.TextColumn("Signal", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Signal confidence level",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
            "RR": st.column_config.TextColumn("Risk/Reward", width="small"),
            "SMC": st.column_config.TextColumn("SMC Component", width="medium"),
            "Orderflow": st.column_config.TextColumn("Orderflow", width="medium"),
            "Reason": st.column_config.TextColumn("Reason", width="large"),
        },
    )

    # Statistics
    st.markdown("---")
    st.subheader("Signal Statistics")

    if len(signals_history) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Signals", len(signals_history))

        with col2:
            long_signals = [s for s in signals_history if "LONG" in s.signal_type]
            st.metric("Long Signals", len(long_signals))

        with col3:
            short_signals = [s for s in signals_history if "SHORT" in s.signal_type]
            st.metric("Short Signals", len(short_signals))

        with col4:
            # Average confidence
            avg_conf = sum(s.confidence for s in signals_history) / len(signals_history)
            st.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")

        # RR statistics for entry signals with setups
        entry_signals = [s for s in signals_history if s.entry_setup]
        if entry_signals:
            st.markdown("**Risk/Reward Statistics (Entry Signals):**")

            col1, col2, col3 = st.columns(3)

            rr_ratios = [s.entry_setup.risk_reward_ratio for s in entry_signals]

            with col1:
                st.metric("Avg RR", f"{sum(rr_ratios) / len(rr_ratios):.1f}:1")

            with col2:
                st.metric("Max RR", f"{max(rr_ratios):.1f}:1")

            with col3:
                st.metric("Min RR", f"{min(rr_ratios):.1f}:1")

            # Count signals meeting minimum RR thresholds
            st.markdown("**Signals by RR Threshold:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                rr_above_3 = sum(1 for rr in rr_ratios if rr >= 3.0)
                st.metric("RR >= 3:1", rr_above_3)

            with col2:
                rr_above_2 = sum(1 for rr in rr_ratios if rr >= 2.0)
                st.metric("RR >= 2:1", rr_above_2)

            with col3:
                rr_above_1_5 = sum(1 for rr in rr_ratios if rr >= 1.5)
                st.metric("RR >= 1.5:1", rr_above_1_5)


def export_signals_to_csv(signals_history: List[Signal]) -> bytes:
    """
    Export signals to CSV format.

    Args:
        signals_history: List of Signal objects

    Returns:
        CSV data as bytes
    """
    import io

    if not signals_history:
        return b""

    # Convert to DataFrame
    data = []
    for signal in signals_history:
        entry_price = signal.entry_setup.entry_price if signal.entry_setup else None
        sl_price = signal.entry_setup.stop_loss_price if signal.entry_setup else None
        tp_price = signal.entry_setup.take_profit_price if signal.entry_setup else None
        rr = signal.entry_setup.risk_reward_ratio if signal.entry_setup else None

        data.append({
            "timestamp": signal.time,
            "signal_type": signal.signal_type,
            "price": signal.price,
            "confidence": signal.confidence,
            "smc_component": signal.smc_component,
            "orderflow_component": signal.orderflow_component,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "risk_reward": rr,
            "reason": signal.reason,
        })

    df = pd.DataFrame(data)

    # Convert to CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")
