"""
Data Loader Utility

Handles fetching live and historical data from Binance.
"""

import random
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta

from botclave.exchange.binance_connector import BinanceConnector
from botclave.engine.footprint import KlineFootprint, Trade


class DataLoader:
    """
    Loads market data from Binance for the dashboard.
    """

    def __init__(self):
        """Initialize data loader with Binance connector."""
        self.connector = BinanceConnector()

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # fetch_ohlcv now returns a DataFrame directly
            df = self.connector.fetch_ohlcv(symbol, timeframe, limit)

            if df.empty:
                return pd.DataFrame()

            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()

    def fetch_ohlcv_sync(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for fetch_ohlcv (for Streamlit compatibility).

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        # fetch_ohlcv is now synchronous, so we can call it directly
        return self.fetch_ohlcv(symbol, timeframe, limit)

    def fetch_order_book(
        self,
        symbol: str = "BTC/USDT",
        limit: int = 20,
    ) -> dict:
        """
        Fetch current order book.

        Args:
            symbol: Trading pair symbol
            limit: Number of depth levels

        Returns:
            Dictionary with bids and asks
        """
        try:
            order_book = self.connector.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return {"bids": [], "asks": [], "timestamp": 0}

    def fetch_order_book_sync(
        self,
        symbol: str = "BTC/USDT",
        limit: int = 20,
    ) -> dict:
        """
        Synchronous wrapper for fetch_order_book.

        Args:
            symbol: Trading pair symbol
            limit: Number of depth levels

        Returns:
            Dictionary with bids and asks
        """
        # fetch_order_book is now synchronous, so we can call it directly
        return self.fetch_order_book(symbol, limit)

    def fetch_trades(
        self,
        symbol: str = "BTC/USDT",
        limit: int = 100,
    ) -> List[dict]:
        """
        Fetch recent trades for footprint analysis.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades

        Returns:
            List of trade dictionaries
        """
        try:
            trades = self.connector.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            print(f"Error fetching trades: {e}")
            return []

    def generate_footprints(
        self,
        df: pd.DataFrame,
        trades_list: Optional[List[List[dict]]] = None,
    ) -> List[KlineFootprint]:
        """
        Generate footprint objects for each candle.

        Args:
            df: DataFrame with OHLCV data
            trades_list: Optional list of trades for each candle

        Returns:
            List of KlineFootprint objects
        """
        footprints = []

        # Determine price step based on price range
        price_range = df["high"].max() - df["low"].min()
        if price_range > 10000:
            price_step = 10.0
        elif price_range > 1000:
            price_step = 1.0
        elif price_range > 100:
            price_step = 0.5
        elif price_range > 10:
            price_step = 0.1
        else:
            price_step = 0.01

        for i, (_, row) in enumerate(df.iterrows()):
            # Create footprint with appropriate price step
            footprint = KlineFootprint(price_step=price_step)

            # Add synthetic trades to populate footprint (simplified)
            # In production, you would use actual trade data from trades_list
            num_synthetic_trades = max(10, int(row["volume"] / 100))  # Simplified scaling, minimum 10 trades
            for _ in range(num_synthetic_trades):
                # Generate a random price within the candle range
                trade_price = random.uniform(row["low"], row["high"])
                is_buy = random.random() > 0.5  # 50/50 buy/sell

                # Create trade object
                trade = Trade(
                    price=trade_price,
                    qty=row["volume"] / num_synthetic_trades,
                    is_buy=is_buy,
                    time_ms=int(row.name.timestamp() * 1000),
                )

                footprint.add_trade(trade)

            footprints.append(footprint)

        return footprints

    def calculate_orderflow_metrics(
        self,
        df: pd.DataFrame,
        footprints: List[KlineFootprint],
    ) -> dict:
        """
        Calculate orderflow metrics from data.

        Args:
            df: DataFrame with OHLCV data
            footprints: List of footprint objects

        Returns:
            Dictionary with orderflow metrics
        """
        metrics = {
            "cvd": 0.0,
            "buy_sell_ratio": 1.0,
            "delta": 0.0,
        }

        if len(footprints) == 0:
            return metrics

        # Calculate CVD (Cumulative Volume Delta)
        buy_volumes = [fp.buy_volume for fp in footprints]
        sell_volumes = [fp.sell_volume for fp in footprints]

        delta = pd.Series(buy_volumes) - pd.Series(sell_volumes)
        cvd = delta.cumsum().iloc[-1] if len(delta) > 0 else 0.0

        # Buy/sell ratio
        total_buy = sum(buy_volumes)
        total_sell = sum(sell_volumes)
        ratio = total_buy / total_sell if total_sell > 0 else 1.0

        metrics["cvd"] = cvd
        metrics["buy_sell_ratio"] = ratio
        metrics["delta"] = delta.iloc[-1] if len(delta) > 0 else 0.0

        return metrics

    def close(self):
        """Close the exchange connection."""
        self.connector.close()
