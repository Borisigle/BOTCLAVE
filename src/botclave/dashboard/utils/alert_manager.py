"""
Alert Manager Utility

Handles signal alerts, sound notifications, and alert history.
"""

import streamlit as st
from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass, field

from botclave.engine.strategy import Signal


@dataclass
class AlertLog:
    """Log entry for a signal alert."""

    signal: Signal
    timestamp: str
    notified: bool = False
    read: bool = False


class AlertManager:
    """
    Manages signal alerts and notifications.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alert_history: List[AlertLog] = []
        self.current_signal: Optional[Signal] = None

        # Initialize session state if not exists
        if "alert_history" not in st.session_state:
            st.session_state.alert_history = []
        if "current_signal" not in st.session_state:
            st.session_state.current_signal = None
        if "alert_enabled" not in st.session_state:
            st.session_state.alert_enabled = True
        if "sound_enabled" not in st.session_state:
            st.session_state.sound_enabled = False

    def process_signal(
        self,
        signal: Optional[Signal],
    ) -> None:
        """
        Process a new signal and create alert if needed.

        Args:
            signal: New signal to process
        """
        if signal is None:
            self.current_signal = None
            st.session_state.current_signal = None
            return

        # Check if this is a new signal (different from current)
        if self.current_signal is None or self._is_new_signal(signal):
            self.current_signal = signal
            st.session_state.current_signal = signal

            # Create alert log
            alert_log = AlertLog(
                signal=signal,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                notified=False,
                read=False,
            )

            # Add to history
            self.alert_history.append(alert_log)
            st.session_state.alert_history.append(alert_log)

            # Trigger notification if enabled
            if st.session_state.alert_enabled:
                self._trigger_alert(signal)

    def _is_new_signal(self, signal: Signal) -> bool:
        """
        Check if signal is different from current signal.

        Args:
            signal: Signal to check

        Returns:
            True if new signal
        """
        if self.current_signal is None:
            return True

        # Compare key properties
        return (
            signal.signal_type != self.current_signal.signal_type
            or signal.price != self.current_signal.price
            or signal.confidence != self.current_signal.confidence
        )

    def _trigger_alert(self, signal: Signal) -> None:
        """
        Trigger alert notification.

        Args:
            signal: Signal to alert about
        """
        # Sound notification if enabled
        if st.session_state.sound_enabled:
            self._play_sound_alert(signal)

        # Visual notification would be handled by Streamlit
        # This is a placeholder for future enhancements

    def _play_sound_alert(self, signal: Signal) -> None:
        """
        Play sound notification for signal.

        Args:
            signal: Signal to play sound for
        """
        # Placeholder for sound notification
        # In a real implementation, this would:
        # 1. Use JavaScript/HTML audio element
        # 2. Or use a Python library like playsound
        # 3. Or use Streamlit's audio component

        pass

    def get_current_signal(self) -> Optional[Signal]:
        """
        Get current active signal.

        Returns:
            Current signal or None
        """
        return self.current_signal

    def get_signal_history(self) -> List[Signal]:
        """
        Get history of all signals.

        Returns:
            List of Signal objects
        """
        return [log.signal for log in self.alert_history]

    def get_unread_alerts(self) -> List[AlertLog]:
        """
        Get list of unread alerts.

        Returns:
            List of unread AlertLog objects
        """
        return [log for log in self.alert_history if not log.read]

    def mark_as_read(self, alert_log: AlertLog) -> None:
        """
        Mark alert as read.

        Args:
            alert_log: Alert log to mark as read
        """
        alert_log.read = True

        # Update session state
        for log in st.session_state.alert_history:
            if log.signal.signal_type == alert_log.signal.signal_type:
                log.read = True

    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        st.session_state.alert_history.clear()

    def get_alert_stats(self) -> Dict:
        """
        Get statistics about alerts.

        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self.alert_history)
        unread_alerts = len(self.get_unread_alerts())

        # Count by signal type
        long_signals = sum(
            1 for log in self.alert_history
            if "LONG" in log.signal.signal_type
        )
        short_signals = sum(
            1 for log in self.alert_history
            if "SHORT" in log.signal.signal_type
        )
        other_signals = total_alerts - long_signals - short_signals

        return {
            "total_alerts": total_alerts,
            "unread_alerts": unread_alerts,
            "long_signals": long_signals,
            "short_signals": short_signals,
            "other_signals": other_signals,
        }

    def render_settings(self) -> None:
        """Render alert settings in sidebar."""
        st.sidebar.markdown("### 🔔 Alert Settings")

        # Enable/disable alerts
        st.session_state.alert_enabled = st.sidebar.checkbox(
            "Enable Alerts",
            value=st.session_state.alert_enabled,
            help="Show notifications when new signals are generated",
        )

        # Enable/disable sound
        st.session_state.sound_enabled = st.sidebar.checkbox(
            "Sound Alerts",
            value=st.session_state.sound_enabled,
            help="Play sound when new signals are generated",
        )

        # Clear history button
        if st.sidebar.button("Clear Alert History"):
            self.clear_history()
            st.sidebar.success("Alert history cleared!")

        # Alert statistics
        stats = self.get_alert_stats()
        st.sidebar.markdown("**Alert Statistics:**")
        st.sidebar.caption(f"Total: {stats['total_alerts']}")
        st.sidebar.caption(f"Unread: {stats['unread_alerts']}")
        st.sidebar.caption(f"Long: {stats['long_signals']} | Short: {stats['short_signals']}")
