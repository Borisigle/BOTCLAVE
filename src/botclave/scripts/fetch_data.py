"""CLI script for fetching cryptocurrency data."""

import argparse
import sys

from ..data.ingestion import DataIngestion, fetch_data, save_dataset
from ..config.manager import config


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch cryptocurrency OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol BTC/USDT --timeframe 4h --limit 1500
  %(prog)s --symbol ETH/USDT --timeframe 1d --limit 500 --no-cache
  %(prog)s --symbol BTC/USDT --timeframe 4h --limit 1000 --cache custom_cache.csv
  %(prog)s --symbol BTC/USDT --timeframe 4h --limit 500 --save data/btc_4h.csv
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=config.get("trading.symbols", ["BTC/USDT"])[0],
        help="Trading symbol (default: %(default)s)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default=config.get("data.default_timeframe", "4h"),
        help="Timeframe (default: %(default)s)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=config.get("data.default_limit", 1500),
        help="Number of candles to fetch (default: %(default)s)",
    )

    parser.add_argument(
        "--exchange",
        type=str,
        default=config.get("data.default_exchange", "binance"),
        help="Exchange name (default: %(default)s)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip using cached data",
    )

    parser.add_argument(
        "--cache",
        type=str,
        help="Custom cache file path",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save fetched data to file",
    )

    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache information and exit",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.cache_info:
            # Show cache information
            ingestion = DataIngestion(args.exchange)
            cache_info = ingestion.get_cache_info(args.symbol, args.timeframe)

            if cache_info["exists"]:
                print(f"Cache file: {cache_info['file_path']}")
                print(f"File size: {cache_info['file_size']:,} bytes")
                print(f"Candles: {cache_info['candles_count']:,}")
                print(
                    f"Date range: {cache_info['earliest_date']} to {cache_info['latest_date']}"
                )
            else:
                print(f"No cache found for {args.symbol} {args.timeframe}")

            return 0

        # Fetch data
        if args.verbose:
            print(
                f"Fetching {args.limit} {args.symbol} {args.timeframe} candles from {args.exchange}"
            )
            if not args.no_cache:
                print("Cache enabled")

        df = fetch_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            exchange=args.exchange,
            use_cache=not args.no_cache,
            cache_file=args.cache,
        )

        print(f"Successfully fetched {len(df)} candles")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")

        if args.verbose:
            print("\nSample data:")
            print(df.head())
            print("\nData types:")
            print(df.dtypes)
            print("\nBasic statistics:")
            print(df.describe())

        # Save to file if requested
        if args.save:
            save_dataset(df, args.save)
            print(f"Data saved to: {args.save}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
