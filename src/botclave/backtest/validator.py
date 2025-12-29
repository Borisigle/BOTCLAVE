"""
Strategy Validator

Validates trading strategies using statistical tests and walk-forward analysis.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from .backtester import Backtester, BacktestResult, BacktestConfig


class ValidationResult(BaseModel):
    """Results from strategy validation."""

    passed: bool = Field(..., description="Whether validation passed")
    in_sample_result: Optional[BacktestResult] = Field(
        None, description="In-sample backtest result"
    )
    out_of_sample_result: Optional[BacktestResult] = Field(
        None, description="Out-of-sample backtest result"
    )
    walk_forward_results: List[BacktestResult] = Field(
        default_factory=list, description="Walk-forward test results"
    )
    overfitting_score: float = Field(0.0, description="Overfitting score (0-1)")
    consistency_score: float = Field(0.0, description="Consistency score (0-1)")
    issues: List[str] = Field(default_factory=list, description="Validation issues")
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class ValidationConfig(BaseModel):
    """Configuration for strategy validation."""

    in_sample_ratio: float = Field(0.7, description="In-sample data ratio")
    walk_forward_periods: int = Field(5, description="Number of walk-forward periods")
    min_trades: int = Field(30, description="Minimum trades required")
    min_win_rate: float = Field(45.0, description="Minimum win rate (%)")
    min_profit_factor: float = Field(1.2, description="Minimum profit factor")
    max_drawdown: float = Field(20.0, description="Maximum allowed drawdown (%)")
    min_sharpe_ratio: float = Field(0.5, description="Minimum Sharpe ratio")


class StrategyValidator:
    """
    Validates trading strategies using multiple techniques.
    """

    def __init__(
        self,
        backtest_config: Optional[BacktestConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
    ):
        """
        Initialize the StrategyValidator.

        Args:
            backtest_config: Backtesting configuration
            validation_config: Validation configuration
        """
        self.backtest_config = backtest_config or BacktestConfig()
        self.validation_config = validation_config or ValidationConfig()

    def validate(
        self, df: pd.DataFrame, signals: pd.DataFrame
    ) -> ValidationResult:
        """
        Perform complete validation of a strategy.

        Args:
            df: Historical price data
            signals: Trading signals

        Returns:
            ValidationResult with all validation metrics
        """
        issues = []
        recommendations = []

        if len(df) < 100:
            issues.append("Insufficient data for validation (minimum 100 bars)")
            return ValidationResult(
                passed=False,
                issues=issues,
                recommendations=["Collect more historical data"],
            )

        split_idx = int(len(df) * self.validation_config.in_sample_ratio)
        in_sample_df = df.iloc[:split_idx]
        in_sample_signals = signals.iloc[:split_idx]
        out_of_sample_df = df.iloc[split_idx:]
        out_of_sample_signals = signals.iloc[split_idx:]

        backtester = Backtester(self.backtest_config)
        in_sample_result = backtester.run(in_sample_df, in_sample_signals)

        backtester = Backtester(self.backtest_config)
        out_of_sample_result = backtester.run(out_of_sample_df, out_of_sample_signals)

        walk_forward_results = self._walk_forward_analysis(df, signals)

        issues, recommendations = self._check_criteria(
            in_sample_result, out_of_sample_result
        )

        overfitting_score = self._calculate_overfitting_score(
            in_sample_result, out_of_sample_result
        )

        consistency_score = self._calculate_consistency_score(walk_forward_results)

        passed = len(issues) == 0

        return ValidationResult(
            passed=passed,
            in_sample_result=in_sample_result,
            out_of_sample_result=out_of_sample_result,
            walk_forward_results=walk_forward_results,
            overfitting_score=overfitting_score,
            consistency_score=consistency_score,
            issues=issues,
            recommendations=recommendations,
        )

    def _walk_forward_analysis(
        self, df: pd.DataFrame, signals: pd.DataFrame
    ) -> List[BacktestResult]:
        """
        Perform walk-forward analysis.

        Args:
            df: Price data
            signals: Trading signals

        Returns:
            List of backtest results for each period
        """
        results = []
        period_size = len(df) // self.validation_config.walk_forward_periods

        for i in range(self.validation_config.walk_forward_periods):
            start_idx = i * period_size
            end_idx = (
                (i + 1) * period_size
                if i < self.validation_config.walk_forward_periods - 1
                else len(df)
            )

            period_df = df.iloc[start_idx:end_idx]
            period_signals = signals.iloc[start_idx:end_idx]

            backtester = Backtester(self.backtest_config)
            result = backtester.run(period_df, period_signals)
            results.append(result)

        return results

    def _check_criteria(
        self, in_sample: BacktestResult, out_of_sample: BacktestResult
    ) -> Tuple[List[str], List[str]]:
        """
        Check if results meet validation criteria.

        Args:
            in_sample: In-sample results
            out_of_sample: Out-of-sample results

        Returns:
            Tuple of (issues, recommendations)
        """
        issues = []
        recommendations = []

        if in_sample.total_trades < self.validation_config.min_trades:
            issues.append(
                f"Insufficient trades in-sample: {in_sample.total_trades} "
                f"(minimum: {self.validation_config.min_trades})"
            )
            recommendations.append("Adjust signal generation to produce more trades")

        if out_of_sample.total_trades < self.validation_config.min_trades:
            issues.append(
                f"Insufficient trades out-of-sample: {out_of_sample.total_trades} "
                f"(minimum: {self.validation_config.min_trades})"
            )

        if in_sample.win_rate < self.validation_config.min_win_rate:
            issues.append(
                f"Win rate too low in-sample: {in_sample.win_rate:.1f}% "
                f"(minimum: {self.validation_config.min_win_rate}%)"
            )
            recommendations.append("Improve signal quality or adjust entry criteria")

        if out_of_sample.win_rate < self.validation_config.min_win_rate:
            issues.append(
                f"Win rate too low out-of-sample: {out_of_sample.win_rate:.1f}% "
                f"(minimum: {self.validation_config.min_win_rate}%)"
            )

        if in_sample.profit_factor < self.validation_config.min_profit_factor:
            issues.append(
                f"Profit factor too low in-sample: {in_sample.profit_factor:.2f} "
                f"(minimum: {self.validation_config.min_profit_factor})"
            )
            recommendations.append("Improve risk/reward ratio or exit timing")

        if out_of_sample.profit_factor < self.validation_config.min_profit_factor:
            issues.append(
                f"Profit factor too low out-of-sample: {out_of_sample.profit_factor:.2f} "
                f"(minimum: {self.validation_config.min_profit_factor})"
            )

        if in_sample.max_drawdown > self.validation_config.max_drawdown:
            issues.append(
                f"Max drawdown too high in-sample: {in_sample.max_drawdown:.1f}% "
                f"(maximum: {self.validation_config.max_drawdown}%)"
            )
            recommendations.append("Tighten stop losses or reduce position size")

        if out_of_sample.max_drawdown > self.validation_config.max_drawdown:
            issues.append(
                f"Max drawdown too high out-of-sample: {out_of_sample.max_drawdown:.1f}% "
                f"(maximum: {self.validation_config.max_drawdown}%)"
            )

        if in_sample.sharpe_ratio < self.validation_config.min_sharpe_ratio:
            issues.append(
                f"Sharpe ratio too low in-sample: {in_sample.sharpe_ratio:.2f} "
                f"(minimum: {self.validation_config.min_sharpe_ratio})"
            )

        if out_of_sample.sharpe_ratio < self.validation_config.min_sharpe_ratio:
            issues.append(
                f"Sharpe ratio too low out-of-sample: {out_of_sample.sharpe_ratio:.2f} "
                f"(minimum: {self.validation_config.min_sharpe_ratio})"
            )

        return issues, recommendations

    def _calculate_overfitting_score(
        self, in_sample: BacktestResult, out_of_sample: BacktestResult
    ) -> float:
        """
        Calculate overfitting score (0 = no overfitting, 1 = severe overfitting).

        Args:
            in_sample: In-sample results
            out_of_sample: Out-of-sample results

        Returns:
            Overfitting score
        """
        if in_sample.total_pnl_percent == 0:
            return 1.0

        performance_ratio = (
            out_of_sample.total_pnl_percent / in_sample.total_pnl_percent
        )

        if performance_ratio >= 0.8:
            return 0.0
        elif performance_ratio >= 0.6:
            return 0.3
        elif performance_ratio >= 0.4:
            return 0.6
        else:
            return 1.0

    def _calculate_consistency_score(
        self, walk_forward_results: List[BacktestResult]
    ) -> float:
        """
        Calculate consistency score across walk-forward periods.

        Args:
            walk_forward_results: List of walk-forward results

        Returns:
            Consistency score (0-1)
        """
        if not walk_forward_results:
            return 0.0

        returns = [r.total_pnl_percent for r in walk_forward_results]
        positive_periods = sum(1 for r in returns if r > 0)

        if len(returns) == 0:
            return 0.0

        consistency = positive_periods / len(returns)

        std_dev = np.std(returns)
        mean_return = np.mean(returns)

        if mean_return <= 0:
            return 0.0

        coefficient_of_variation = std_dev / mean_return if mean_return != 0 else 10.0

        if coefficient_of_variation < 0.5:
            consistency_bonus = 0.2
        elif coefficient_of_variation < 1.0:
            consistency_bonus = 0.1
        else:
            consistency_bonus = 0.0

        return min(consistency + consistency_bonus, 1.0)

    def monte_carlo_simulation(
        self, trades: List, num_simulations: int = 1000
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on trade sequence.

        Args:
            trades: List of trades
            num_simulations: Number of simulations to run

        Returns:
            Dictionary with simulation statistics
        """
        if not trades:
            return {
                "mean_return": 0.0,
                "median_return": 0.0,
                "worst_case": 0.0,
                "best_case": 0.0,
                "probability_of_profit": 0.0,
            }

        returns = [t.pnl_percent for t in trades]
        simulated_returns = []

        for _ in range(num_simulations):
            shuffled = np.random.choice(returns, size=len(returns), replace=True)
            total_return = np.sum(shuffled)
            simulated_returns.append(total_return)

        return {
            "mean_return": np.mean(simulated_returns),
            "median_return": np.median(simulated_returns),
            "worst_case": np.percentile(simulated_returns, 5),
            "best_case": np.percentile(simulated_returns, 95),
            "probability_of_profit": np.sum(np.array(simulated_returns) > 0)
            / num_simulations,
        }
