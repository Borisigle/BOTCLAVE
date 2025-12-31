"""
Binance Exchange Connector

Handles connection to Binance exchange via CCXT and WebSocket streams
for real-time market data.
"""

from typing import Dict, List, Optional, Callable, Any
import asyncio
import json
import logging
import pandas as pd
import ccxt
from ccxt.base.errors import NetworkError, ExchangeError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExchangeConfig(BaseModel):
    """Configuration for exchange connection."""

    api_key: Optional[str] = Field(None, description="API key")
    api_secret: Optional[str] = Field(None, description="API secret")
    testnet: bool = Field(True, description="Use testnet")
    enable_rate_limit: bool = Field(True, description="Enable rate limiting")
    timeout: int = Field(30000, description="Request timeout in ms")


class MarketData(BaseModel):
    """Market data snapshot."""

    symbol: str = Field(..., description="Trading symbol")
    timestamp: int = Field(..., description="Data timestamp")
    bid: float = Field(..., description="Best bid price")
    ask: float = Field(..., description="Best ask price")
    last: float = Field(..., description="Last trade price")
    volume_24h: float = Field(..., description="24h volume")


class BinanceConnector:
    """
    Connector for Binance exchange using CCXT.
    Provides market data access and order execution capabilities.
    """

    def __init__(self, config: Optional[ExchangeConfig] = None):
        """
        Initialize Binance connector.

        Args:
            config: Exchange configuration
        """
        self.config = config or ExchangeConfig()
        self.exchange: Optional[ccxt.Exchange] = None
        self._websocket_tasks: List[asyncio.Task] = []
        self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """Initialize the CCXT exchange instance."""
        exchange_config = {
            "enableRateLimit": self.config.enable_rate_limit,
            "timeout": self.config.timeout,
        }

        if self.config.api_key and self.config.api_secret:
            exchange_config["apiKey"] = self.config.api_key
            exchange_config["secret"] = self.config.api_secret

        if self.config.testnet:
            self.exchange = ccxt.binance({
                **exchange_config,
                "options": {"defaultType": "future"},
            })
            if hasattr(self.exchange, "set_sandbox_mode"):
                self.exchange.set_sandbox_mode(True)
        else:
            self.exchange = ccxt.binance(exchange_config)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 100,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (default '15m')
            limit: Number of candles (default 100)
            since: Starting timestamp (optional)

        Returns:
            DataFrame with OHLCV columns indexed by timestamp

        Raises:
            RuntimeError: If exchange not initialized
            NetworkError: If network error occurs
            ExchangeError: If exchange error occurs
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        try:
            # CCXT fetch_ohlcv is synchronous, returns a list directly
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except (NetworkError, ExchangeError) as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            raise

    def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch current order book.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels to fetch

        Returns:
            Order book dictionary with bids and asks

        Raises:
            RuntimeError: If exchange not initialized
            NetworkError: If network error occurs
            ExchangeError: If exchange error occurs
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        try:
            # CCXT fetch_order_book is synchronous
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except (NetworkError, ExchangeError) as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker dictionary

        Raises:
            RuntimeError: If exchange not initialized
            NetworkError: If network error occurs
            ExchangeError: If exchange error occurs
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        try:
            # CCXT fetch_ticker is synchronous
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except (NetworkError, ExchangeError) as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise

    def fetch_trades(
        self, symbol: str, limit: int = 100, since: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent trades.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            since: Starting timestamp

        Returns:
            List of trade dictionaries

        Raises:
            RuntimeError: If exchange not initialized
            NetworkError: If network error occurs
            ExchangeError: If exchange error occurs
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        try:
            # CCXT fetch_trades is synchronous
            trades = self.exchange.fetch_trades(symbol, since, limit)
            return trades
        except (NetworkError, ExchangeError) as e:
            logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            raise

    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.

        Returns:
            Balance dictionary

        Raises:
            RuntimeError: If exchange not initialized
            NetworkError: If network error occurs
            ExchangeError: If exchange error occurs
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        try:
            # CCXT fetch_balance is synchronous
            balance = self.exchange.fetch_balance()
            return balance
        except (NetworkError, ExchangeError) as e:
            logger.error(f"Error fetching balance: {str(e)}")
            raise

    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        limit: int = 20,
    ) -> None:
        """
        Subscribe to order book updates via WebSocket.

        Args:
            symbol: Trading pair symbol
            callback: Callback function for updates
            limit: Number of levels to track
        """
        pass

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Subscribe to real-time trades via WebSocket.

        Args:
            symbol: Trading pair symbol
            callback: Callback function for trade updates
        """
        pass

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get market information for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Market info dictionary
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        markets = self.exchange.load_markets()
        return markets.get(symbol, {})

    def close(self) -> None:
        """Close the exchange connection and cleanup resources."""
        if self.exchange:
            self.exchange.close()

        for task in self._websocket_tasks:
            if not task.done():
                task.cancel()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()
