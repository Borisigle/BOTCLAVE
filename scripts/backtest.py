#!/usr/bin/env python3
"""
Backtest Script

Run backtests on historical data with the order flow strategy.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from botclave.backtest.backtester import Backtester, BacktestConfig
from botclave.backtest.validator import StrategyValidator, ValidationConfig
from botclave.engine.strategy import OrderFlowStrategy, StrategyConfig


def load_data(filepath: str) -> pd.DataFrame:
    """Load historical data from CSV."""
    df = pd.read_csv(filepath)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

    return df


def generate_signals(df: pd.DataFrame, strategy_config: StrategyConfig) -> pd.DataFrame:
    """Generate trading signals using the strategy."""
    strategy = OrderFlowStrategy(strategy_config)

    signals = []

    for i in range(len(df)):
        if i < 50:
            continue

        current_df = df.iloc[: i + 1]
        current_price = df["close"].iloc[i]
        current_time = df.index[i] if hasattr(df.index, "__getitem__") else i

        signal = strategy.generate_signal(
            symbol="BTC/USDT",
            df=current_df,
            current_price=current_price,
            timestamp=int(current_time.timestamp() * 1000)
            if hasattr(current_time, "timestamp")
            else current_time,
        )

        if signal:
            signals.append(
                {
                    "timestamp": current_time,
                    "side": signal.side.value,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit[0],
                    "confidence": signal.confidence,
                }
            )

    return pd.DataFrame(signals).set_index("timestamp") if signals else pd.DataFrame()


def run_backtest(
    data_file: str,
    initial_capital: float = 10000.0,
    position_size: float = 0.02,
    validate: bool = False,
):
    """
    Run backtest on historical data.

    Args:
        data_file: Path to historical data CSV
        initial_capital: Initial capital
        position_size: Position size as % of capital
        validate: Run validation tests
    """
    print(f"Loading data from {data_file}...")
    df = load_data(data_file)
    print(f"Loaded {len(df)} bars")

    print("\nGenerating signals...")
    strategy_config = StrategyConfig()
    signals = generate_signals(df, strategy_config)
    print(f"Generated {len(signals)} signals")

    if signals.empty:
        print("No signals generated. Adjust strategy parameters.")
        return

    print("\nRunning backtest...")
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        position_size_pct=position_size,
    )

    backtester = Backtester(backtest_config)
    result = backtester.run(df, signals)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total Trades:        {result.total_trades}")
    print(f"Winning Trades:      {result.winning_trades}")
    print(f"Losing Trades:       {result.losing_trades}")
    print(f"Win Rate:           {result.win_rate:.2f}%")
    print(f"Total P&L:          ${result.total_pnl:.2f} ({result.total_pnl_percent:.2f}%)")
    print(f"Max Drawdown:       {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:.2f}")
    print(f"Profit Factor:      {result.profit_factor:.2f}")
    print(f"Average Win:        ${result.avg_win:.2f}")
    print(f"Average Loss:       ${result.avg_loss:.2f}")
    print(f"Largest Win:        ${result.largest_win:.2f}")
    print(f"Largest Loss:       ${result.largest_loss:.2f}")
    print(f"Total Fees:         ${result.total_fees:.2f}")
    print(f"Final Capital:      ${result.end_capital:.2f}")
    print("=" * 70)

    if validate:
        print("\nRunning validation tests...")
        validator = StrategyValidator(backtest_config, ValidationConfig())
        validation_result = validator.validate(df, signals)

        print(f"\nValidation: {'✅ PASSED' if validation_result.passed else '❌ FAILED'}")
        print(f"Overfitting Score: {validation_result.overfitting_score:.2f}")
        print(f"Consistency Score: {validation_result.consistency_score:.2f}")

        if validation_result.issues:
            print("\nIssues:")
            for issue in validation_result.issues:
                print(f"  ⚠️  {issue}")

        if validation_result.recommendations:
            print("\nRecommendations:")
            for rec in validation_result.recommendations:
                print(f"  💡 {rec}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to historical data CSV",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.02,
        help="Position size as decimal (0.02 = 2%%)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation tests",
    )

    args = parser.parse_args()

    run_backtest(
        data_file=args.data,
        initial_capital=args.capital,
        position_size=args.position_size,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
