"""
Metrics Calculation Module

Calculates performance metrics for strategy evaluation.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class MetricsCalculator:
    """
    Calculates various performance metrics for trading strategies.
    """

    def __init__(self):
        """Initialize MetricsCalculator."""
        pass

    def calculate_strategy_metrics(
        self, df: pd.DataFrame, signals: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Calculate comprehensive strategy metrics.

        Args:
            df: Price data DataFrame
            signals: Signals DataFrame

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            "total_signals": len(signals) if signals is not None else 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "equity_curve": [],
            "trades": [],
        }

        if signals is None or signals.empty:
            return metrics

        return metrics

    def calculate_win_rate(self, trades: List) -> float:
        """
        Calculate win rate from trades.

        Args:
            trades: List of trades

        Returns:
            Win rate percentage
        """
        if not trades:
            return 0.0

        winning_trades = sum(
            1
            for t in trades
            if (t.get("pnl", 0) if isinstance(t, dict) else t.pnl) > 0
        )
        return (winning_trades / len(trades)) * 100

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: List of equity values

        Returns:
            Maximum drawdown percentage
        """
        if not equity_curve:
            return 0.0

        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        return abs(drawdown.min()) * 100

    def calculate_profit_factor(self, trades: List) -> float:
        """
        Calculate profit factor.

        Args:
            trades: List of trades

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        gross_profit = sum(
            t.get("pnl", 0) if isinstance(t, dict) else t.pnl
            for t in trades
            if (t.get("pnl", 0) if isinstance(t, dict) else t.pnl) > 0
        )

        gross_loss = abs(
            sum(
                t.get("pnl", 0) if isinstance(t, dict) else t.pnl
                for t in trades
                if (t.get("pnl", 0) if isinstance(t, dict) else t.pnl) < 0
            )
        )

        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (downside risk adjusted).

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate

        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0

        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0

        return np.sqrt(252) * (excess_returns.mean() / downside_returns.std())

    def calculate_calmar_ratio(
        self, total_return: float, max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio.

        Args:
            total_return: Total return percentage
            max_drawdown: Maximum drawdown percentage

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return total_return / max_drawdown

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various risk metrics.

        Args:
            df: DataFrame with returns

        Returns:
            Dictionary of risk metrics
        """
        if df.empty or "returns" not in df.columns:
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            }

        returns = df["returns"]

        volatility = returns.std() * np.sqrt(252) * 100

        var_95 = np.percentile(returns, 5)

        tail_losses = returns[returns <= var_95]
        cvar_95 = tail_losses.mean() if not tail_losses.empty else 0.0

        return {
            "volatility": volatility,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

    def calculate_trade_statistics(self, trades: List) -> Dict[str, any]:
        """
        Calculate detailed trade statistics.

        Args:
            trades: List of trades

        Returns:
            Dictionary with trade statistics
        """
        if not trades:
            return {
                "total_trades": 0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_duration": 0.0,
            }

        pnls = [
            t.get("pnl", 0) if isinstance(t, dict) else t.pnl for t in trades
        ]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        return {
            "total_trades": len(trades),
            "avg_trade": np.mean(pnls) if pnls else 0.0,
            "avg_win": np.mean(winning_pnls) if winning_pnls else 0.0,
            "avg_loss": np.mean(losing_pnls) if losing_pnls else 0.0,
            "largest_win": max(pnls) if pnls else 0.0,
            "largest_loss": min(pnls) if pnls else 0.0,
            "avg_duration": 0.0,
        }

    def generate_report(
        self, df: pd.DataFrame, signals: pd.DataFrame, trades: List
    ) -> str:
        """
        Generate a text report of performance metrics.

        Args:
            df: Price data
            signals: Signals data
            trades: List of trades

        Returns:
            Formatted text report
        """
        metrics = self.calculate_strategy_metrics(df, signals)
        trade_stats = self.calculate_trade_statistics(trades)

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           STRATEGY PERFORMANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Trades:        {trade_stats['total_trades']:>8}                        ║
║ Win Rate:           {metrics['win_rate']:>8.2f}%                       ║
║ Total P&L:          ${metrics['total_pnl']:>8.2f}                       ║
║ Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}                        ║
║ Max Drawdown:       {metrics['max_drawdown']:>8.2f}%                       ║
║ Profit Factor:      {metrics['profit_factor']:>8.2f}                        ║
╠══════════════════════════════════════════════════════════════╣
║ Average Trade:      ${trade_stats['avg_trade']:>8.2f}                       ║
║ Average Win:        ${trade_stats['avg_win']:>8.2f}                       ║
║ Average Loss:       ${trade_stats['avg_loss']:>8.2f}                       ║
║ Largest Win:        ${trade_stats['largest_win']:>8.2f}                       ║
║ Largest Loss:       ${trade_stats['largest_loss']:>8.2f}                       ║
╚══════════════════════════════════════════════════════════════╝
        """

        return report
