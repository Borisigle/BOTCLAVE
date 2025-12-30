"""
Alert Panel Component

Displays real-time trading signals with detailed information.
"""

import streamlit as st
from typing import Optional
from botclave.engine.strategy import Signal


def show_alert_panel(current_signal: Optional[Signal] = None) -> None:
    """
    Display alert panel when there is an active trading signal.

    Args:
        current_signal: Current trading signal or None
    """
    st.subheader("🚨 ALERTS")

    if current_signal:
        # Determine color based on signal type
        if "LONG" in current_signal.signal_type:
            color_emoji = "🟢"
            signal_color = "green"
        elif "SHORT" in current_signal.signal_type:
            color_emoji = "🔴"
            signal_color = "red"
        else:
            color_emoji = "⚪"
            signal_color = "gray"

        with st.container(border=True):
            # Signal header with color indicator
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(
                    f"""
                    ### {color_emoji} {current_signal.signal_type}
                    **Symbol** @ **${current_signal.price:,.2f}**
                    """
                )
                st.caption(f"Time: {current_signal.time}")

            with col2:
                st.metric(
                    "Confidence",
                    f"{current_signal.confidence * 100:.0f}%",
                )

            with col3:
                if current_signal.entry_setup:
                    st.metric(
                        "RR Ratio",
                        f"{current_signal.entry_setup.risk_reward_ratio:.1f}:1",
                    )

            # Entry setup details
            st.markdown("---")

            if current_signal.entry_setup:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Entry",
                        f"${current_signal.entry_setup.entry_price:,.2f}",
                    )

                with col2:
                    st.metric(
                        "Stop Loss",
                        f"${current_signal.entry_setup.stop_loss_price:,.2f}",
                    )

                with col3:
                    st.metric(
                        "Take Profit",
                        f"${current_signal.entry_setup.take_profit_price:,.2f}",
                    )

            # Signal components
            st.markdown("**Signal Components:**")
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.info(f"SMC: {current_signal.smc_component}")

            with comp_col2:
                st.info(f"Orderflow: {current_signal.orderflow_component}")

            # Reason for signal
            st.markdown("**Why this signal:**")
            st.success(current_signal.reason)

            # Sound notification (optional)
            if st.checkbox("🔊 Play Sound Alert", key="sound_alert"):
                # Placeholder for sound alert functionality
                st.info("Sound alerts would play here")

    else:
        st.info("⏳ Waiting for signals...")
        st.caption(
            "Load data to start monitoring the market for trading opportunities"
        )
