"""
Backtesting Engine

Simulates strategy execution on historical data with realistic constraints.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class Trade(BaseModel):
    """Represents a completed trade in the backtest."""

    entry_time: int = Field(..., description="Entry timestamp")
    exit_time: int = Field(..., description="Exit timestamp")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (long/short)")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    quantity: float = Field(..., description="Trade quantity")
    pnl: float = Field(..., description="Profit/Loss")
    pnl_percent: float = Field(..., description="P&L percentage")
    fees: float = Field(..., description="Trading fees")
    exit_reason: str = Field(..., description="Reason for exit")
    max_drawdown: float = Field(0.0, description="Max drawdown during trade")


class BacktestResult(BaseModel):
    """Complete backtest results and statistics."""

    total_trades: int = Field(0, description="Total number of trades")
    winning_trades: int = Field(0, description="Number of winning trades")
    losing_trades: int = Field(0, description="Number of losing trades")
    win_rate: float = Field(0.0, description="Win rate percentage")
    total_pnl: float = Field(0.0, description="Total profit/loss")
    total_pnl_percent: float = Field(0.0, description="Total P&L percentage")
    max_drawdown: float = Field(0.0, description="Maximum drawdown")
    sharpe_ratio: float = Field(0.0, description="Sharpe ratio")
    profit_factor: float = Field(0.0, description="Profit factor")
    avg_win: float = Field(0.0, description="Average winning trade")
    avg_loss: float = Field(0.0, description="Average losing trade")
    largest_win: float = Field(0.0, description="Largest winning trade")
    largest_loss: float = Field(0.0, description="Largest losing trade")
    avg_trade_duration: float = Field(0.0, description="Average trade duration (hours)")
    total_fees: float = Field(0.0, description="Total fees paid")
    start_capital: float = Field(0.0, description="Starting capital")
    end_capital: float = Field(0.0, description="Ending capital")
    trades: List[Trade] = Field(default_factory=list, description="List of all trades")


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    initial_capital: float = Field(10000.0, description="Initial capital")
    position_size_pct: float = Field(0.02, description="Position size as % of capital")
    maker_fee: float = Field(0.0002, description="Maker fee (0.02%)")
    taker_fee: float = Field(0.0004, description="Taker fee (0.04%)")
    slippage: float = Field(0.0001, description="Slippage (0.01%)")
    max_positions: int = Field(1, description="Maximum concurrent positions")
    use_leverage: bool = Field(False, description="Use leverage")
    leverage: float = Field(1.0, description="Leverage multiplier")


class Backtester:
    """
    Backtests trading strategies on historical data.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the Backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_capital = self.config.initial_capital
        self.open_positions: Dict[str, Dict] = {}

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run backtest on historical data with generated signals.

        Args:
            df: DataFrame with OHLCV data
            signals: DataFrame with entry/exit signals

        Returns:
            BacktestResult with performance metrics
        """
        self.trades = []
        self.equity_curve = [self.current_capital]
        self.open_positions = {}
        self.current_capital = self.config.initial_capital

        if df.empty or signals.empty:
            return self._generate_result()

        for idx in range(len(df)):
            current_price = df["close"].iloc[idx]
            current_time = df.index[idx] if hasattr(df.index, "__getitem__") else idx

            self._check_exits(idx, df)

            if idx < len(signals) and self._can_open_position():
                signal = signals.iloc[idx]
                self._process_entry_signal(idx, df, signal)

            self.equity_curve.append(self.current_capital)

        self._close_all_positions(len(df) - 1, df)

        return self._generate_result()

    def _can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.open_positions) < self.config.max_positions

    def _process_entry_signal(
        self, idx: int, df: pd.DataFrame, signal: pd.Series
    ) -> None:
        """
        Process an entry signal.

        Args:
            idx: Current index
            df: Price data
            signal: Entry signal
        """
        if "side" not in signal or signal["side"] not in ["long", "short"]:
            return

        entry_price = df["close"].iloc[idx] * (1 + self.config.slippage)
        position_size = (
            self.current_capital * self.config.position_size_pct
        ) / entry_price

        if self.config.use_leverage:
            position_size *= self.config.leverage

        fees = position_size * entry_price * self.config.taker_fee

        position = {
            "side": signal["side"],
            "entry_price": entry_price,
            "entry_time": df.index[idx] if hasattr(df.index, "__getitem__") else idx,
            "quantity": position_size,
            "stop_loss": signal.get("stop_loss", entry_price * 0.95),
            "take_profit": signal.get("take_profit", entry_price * 1.05),
            "fees": fees,
            "peak_price": entry_price,
        }

        symbol = signal.get("symbol", "BTC/USDT")
        self.open_positions[symbol] = position
        self.current_capital -= fees

    def _check_exits(self, idx: int, df: pd.DataFrame) -> None:
        """
        Check if any open positions should be closed.

        Args:
            idx: Current index
            df: Price data
        """
        current_price = df["close"].iloc[idx]
        positions_to_close = []

        for symbol, position in self.open_positions.items():
            exit_reason = None

            if position["side"] == "long":
                position["peak_price"] = max(position["peak_price"], current_price)

                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"

            else:
                position["peak_price"] = min(position["peak_price"], current_price)

                if current_price >= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price <= position["take_profit"]:
                    exit_reason = "take_profit"

            if exit_reason:
                positions_to_close.append((symbol, exit_reason, idx, current_price))

        for symbol, reason, idx, price in positions_to_close:
            self._close_position(symbol, idx, df, price, reason)

    def _close_position(
        self,
        symbol: str,
        idx: int,
        df: pd.DataFrame,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """
        Close an open position.

        Args:
            symbol: Trading symbol
            idx: Current index
            df: Price data
            exit_price: Exit price
            exit_reason: Reason for closing
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        exit_price = exit_price * (1 - self.config.slippage)

        if position["side"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
            max_dd = (
                position["peak_price"] - position["entry_price"]
            ) / position["entry_price"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"]
            max_dd = (
                position["entry_price"] - position["peak_price"]
            ) / position["entry_price"]

        exit_fees = position["quantity"] * exit_price * self.config.taker_fee
        total_fees = position["fees"] + exit_fees
        net_pnl = pnl - total_fees

        pnl_percent = (
            net_pnl / (position["entry_price"] * position["quantity"]) * 100
        )

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=df.index[idx] if hasattr(df.index, "__getitem__") else idx,
            symbol=symbol,
            side=position["side"],
            entry_price=position["entry_price"],
            exit_price=exit_price,
            quantity=position["quantity"],
            pnl=net_pnl,
            pnl_percent=pnl_percent,
            fees=total_fees,
            exit_reason=exit_reason,
            max_drawdown=max_dd,
        )

        self.trades.append(trade)
        self.current_capital += net_pnl
        del self.open_positions[symbol]

    def _close_all_positions(self, idx: int, df: pd.DataFrame) -> None:
        """Close all open positions at the end of backtest."""
        symbols = list(self.open_positions.keys())
        current_price = df["close"].iloc[idx]

        for symbol in symbols:
            self._close_position(symbol, idx, df, current_price, "end_of_test")

    def _generate_result(self) -> BacktestResult:
        """Generate backtest result from completed trades."""
        if not self.trades:
            return BacktestResult(
                start_capital=self.config.initial_capital,
                end_capital=self.current_capital,
            )

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        returns = [t.pnl_percent for t in self.trades]
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) > 0
            else 0
        )

        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0

        trade_durations = [
            (t.exit_time - t.entry_time) / 3600000 for t in self.trades
        ]
        avg_duration = np.mean(trade_durations) if trade_durations else 0

        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=sum(t.pnl for t in self.trades),
            total_pnl_percent=(self.current_capital - self.config.initial_capital)
            / self.config.initial_capital
            * 100,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            avg_loss=np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            largest_win=max([t.pnl for t in winning_trades]) if winning_trades else 0,
            largest_loss=min([t.pnl for t in losing_trades]) if losing_trades else 0,
            avg_trade_duration=avg_duration,
            total_fees=sum(t.fees for t in self.trades),
            start_capital=self.config.initial_capital,
            end_capital=self.current_capital,
            trades=self.trades,
        )

    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve as a pandas Series."""
        return pd.Series(self.equity_curve)
