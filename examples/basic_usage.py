"""Example script demonstrating Botclave usage."""

import sys
from pathlib import Path

# Add src to path for demonstration
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from botclave.data.ingestion import fetch_data, save_dataset, DataIngestion
from botclave.config.manager import config


def main():
    """Demonstrate basic Botclave functionality."""
    print("Botclave Example Script")
    print("=" * 40)
    
    # Show configuration
    print("Current Configuration:")
    print(f"  Default Exchange: {config.get('data.default_exchange')}")
    print(f"  Default Timeframe: {config.get('data.default_timeframe')}")
    print(f"  Default Limit: {config.get('data.default_limit')}")
    print(f"  Cache Directory: {config.get('data.cache_dir')}")
    print()
    
    # Example 1: Simple data fetch
    print("Example 1: Fetching BTC/USDT data...")
    try:
        df = fetch_data(
            symbol="BTC/USDT",
            timeframe="4h",
            limit=10,  # Small limit for demo
            use_cache=True
        )
        
        print(f"  Successfully fetched {len(df)} candles")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"  Sample data:")
        print(df.head(3).to_string())
        print()
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Note: This requires internet connection and exchange API access")
        print()
    
    # Example 2: Using DataIngestion class directly
    print("Example 2: Using DataIngestion class...")
    try:
        ingestion = DataIngestion()
        
        # Get cache info
        cache_info = ingestion.get_cache_info("BTC/USDT", "4h")
        if cache_info["exists"]:
            print(f"  Cache exists with {cache_info['candles_count']} candles")
            print(f"  Cache file: {cache_info['file_path']}")
        else:
            print("  No cache found")
        
        print()
        
    except Exception as e:
        print(f"  Error: {e}")
        print()
    
    # Example 3: Configuration manipulation
    print("Example 3: Configuration manipulation...")
    original_limit = config.get("data.default_limit")
    config.set("data.default_limit", 500)
    print(f"  Changed default limit from {original_limit} to {config.get('data.default_limit')}")
    
    # Reset to original
    config.set("data.default_limit", original_limit)
    print(f"  Reset default limit to {config.get('data.default_limit')}")
    print()
    
    print("Example completed!")
    print("\nTo run the CLI tool:")
    print("  python -m botclave --help")
    print("  python -m botclave --symbol BTC/USDT --timeframe 4h --limit 100")


if __name__ == "__main__":
    main()