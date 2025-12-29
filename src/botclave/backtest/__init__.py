"""
Backtest Module

Provides backtesting and validation functionality for trading strategies.
"""

from .backtester import Backtester, BacktestResult, Trade
from .validator import StrategyValidator, ValidationResult

__all__ = [
    "Backtester",
    "BacktestResult",
    "Trade",
    "StrategyValidator",
    "ValidationResult",
]
