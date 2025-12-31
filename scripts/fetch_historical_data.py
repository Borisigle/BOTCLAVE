#!/usr/bin/env python3
"""
Fetch Historical Data Script

Downloads historical market data from exchanges and saves to local cache.
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from botclave.exchange.binance_connector import BinanceConnector, ExchangeConfig


def fetch_data(
    symbol: str,
    timeframe: str,
    days: int,
    output_dir: str,
    testnet: bool = True,
):
    """
    Fetch historical data from exchange.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe (1m, 5m, 15m, 1h, etc.)
        days: Number of days to fetch
        output_dir: Output directory
        testnet: Use testnet
    """
    print(f"Fetching {days} days of {timeframe} data for {symbol}...")

    config = ExchangeConfig(testnet=testnet)
    connector = BinanceConnector(config)

    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_dfs = []
    batch_size = 1000

    try:
        while True:
            print(f"Fetching batch from {datetime.fromtimestamp(since/1000)}...")

            # fetch_ohlcv now returns a DataFrame directly
            df = connector.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=batch_size,
                since=since,
            )

            if df.empty:
                break

            all_dfs.append(df)

            if len(df) < batch_size:
                break

            # Get the last timestamp and increment for next batch
            since = int(df.index[-1].timestamp() * 1000) + 1

    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    finally:
        connector.close()

    if not all_dfs:
        print("No data fetched!")
        return

    # Concatenate all DataFrames
    final_df = pd.concat(all_dfs)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    final_df.sort_index(inplace=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol.replace('/', '_')}_{timeframe}_{days}d.csv"
    filepath = output_path / filename

    final_df.to_csv(filepath)

    print(f"\n✅ Saved {len(final_df)} bars to {filepath}")
    print(f"Date range: {final_df.index[0]} to {final_df.index[-1]}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fetch historical market data")
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading symbol (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="15m",
        help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to fetch",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical",
        help="Output directory",
    )
    parser.add_argument(
        "--no-testnet",
        action="store_true",
        help="Use production exchange (not testnet)",
    )

    args = parser.parse_args()

    # fetch_data is now synchronous
    fetch_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        output_dir=args.output,
        testnet=not args.no_testnet,
    )


if __name__ == "__main__":
    main()
