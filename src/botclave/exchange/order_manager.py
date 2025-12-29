"""
Order Manager Module

Handles order creation, tracking, and lifecycle management.
"""

from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import ccxt
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"


class Order(BaseModel):
    """Order model."""

    id: Optional[str] = Field(None, description="Exchange order ID")
    client_order_id: str = Field(..., description="Client order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price")
    status: OrderStatus = Field(OrderStatus.PENDING, description="Order status")
    filled_quantity: float = Field(0.0, description="Filled quantity")
    average_price: float = Field(0.0, description="Average fill price")
    created_at: int = Field(..., description="Creation timestamp")
    updated_at: int = Field(..., description="Last update timestamp")
    fees: float = Field(0.0, description="Trading fees")


class OrderManager:
    """
    Manages order lifecycle including creation, tracking, and cancellation.
    """

    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        """
        Initialize OrderManager.

        Args:
            exchange: CCXT exchange instance
        """
        self.exchange = exchange
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

    async def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a market order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            client_order_id: Optional client order ID

        Returns:
            Created Order object
        """
        if not client_order_id:
            client_order_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            created_at=int(datetime.now().timestamp() * 1000),
            updated_at=int(datetime.now().timestamp() * 1000),
        )

        if self.exchange:
            try:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side.value,
                    amount=quantity,
                )
                order.id = result.get("id")
                order.status = OrderStatus.CLOSED
                order.filled_quantity = result.get("filled", quantity)
                order.average_price = result.get("average", 0.0)
            except Exception as e:
                print(f"Error creating market order: {e}")
                order.status = OrderStatus.REJECTED

        self.orders[order.client_order_id] = order
        return order

    async def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Limit price
            client_order_id: Optional client order ID

        Returns:
            Created Order object
        """
        if not client_order_id:
            client_order_id = f"{symbol}_{side}_{int(datetime.now().timestamp())}"

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            created_at=int(datetime.now().timestamp() * 1000),
            updated_at=int(datetime.now().timestamp() * 1000),
        )

        if self.exchange:
            try:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side.value,
                    amount=quantity,
                    price=price,
                )
                order.id = result.get("id")
                order.status = OrderStatus.OPEN
            except Exception as e:
                print(f"Error creating limit order: {e}")
                order.status = OrderStatus.REJECTED

        self.orders[order.client_order_id] = order
        return order

    async def create_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a stop loss order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            stop_price: Stop trigger price
            client_order_id: Optional client order ID

        Returns:
            Created Order object
        """
        if not client_order_id:
            client_order_id = f"{symbol}_{side}_SL_{int(datetime.now().timestamp())}"

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_price,
            created_at=int(datetime.now().timestamp() * 1000),
            updated_at=int(datetime.now().timestamp() * 1000),
        )

        if self.exchange:
            try:
                result = await self.exchange.create_order(
                    symbol=symbol,
                    type="stop_loss",
                    side=side.value,
                    amount=quantity,
                    params={"stopPrice": stop_price},
                )
                order.id = result.get("id")
                order.status = OrderStatus.OPEN
            except Exception as e:
                print(f"Error creating stop loss order: {e}")
                order.status = OrderStatus.REJECTED

        self.orders[order.client_order_id] = order
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            True if canceled successfully
        """
        if self.exchange:
            try:
                await self.exchange.cancel_order(order_id, symbol)
                for order in self.orders.values():
                    if order.id == order_id:
                        order.status = OrderStatus.CANCELED
                        order.updated_at = int(datetime.now().timestamp() * 1000)
                return True
            except Exception as e:
                print(f"Error canceling order: {e}")
                return False
        return False

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """
        Get current order status from exchange.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Updated Order object or None
        """
        if not self.exchange:
            return None

        try:
            result = await self.exchange.fetch_order(order_id, symbol)
            for order in self.orders.values():
                if order.id == order_id:
                    order.status = OrderStatus(result.get("status", "open"))
                    order.filled_quantity = result.get("filled", 0.0)
                    order.average_price = result.get("average", 0.0)
                    order.updated_at = int(datetime.now().timestamp() * 1000)
                    return order
        except Exception as e:
            print(f"Error fetching order status: {e}")

        return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        if self.exchange:
            try:
                orders = await self.exchange.fetch_open_orders(symbol)
                return [
                    Order(
                        id=o.get("id"),
                        client_order_id=o.get("clientOrderId", ""),
                        symbol=o.get("symbol", ""),
                        side=OrderSide(o.get("side", "buy")),
                        type=OrderType(o.get("type", "limit")),
                        quantity=o.get("amount", 0.0),
                        price=o.get("price"),
                        status=OrderStatus(o.get("status", "open")),
                        filled_quantity=o.get("filled", 0.0),
                        average_price=o.get("average", 0.0),
                        created_at=o.get("timestamp", 0),
                        updated_at=o.get("timestamp", 0),
                    )
                    for o in orders
                ]
            except Exception as e:
                print(f"Error fetching open orders: {e}")

        return [
            order
            for order in self.orders.values()
            if order.status == OrderStatus.OPEN
            and (symbol is None or order.symbol == symbol)
        ]

    def get_order_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> List[Order]:
        """
        Get order history.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        history = [
            order
            for order in self.orders.values()
            if order.status
            in [OrderStatus.CLOSED, OrderStatus.CANCELED, OrderStatus.REJECTED]
            and (symbol is None or order.symbol == symbol)
        ]
        return sorted(history, key=lambda x: x.created_at, reverse=True)[:limit]

    def get_order_by_id(self, client_order_id: str) -> Optional[Order]:
        """
        Get order by client order ID.

        Args:
            client_order_id: Client order ID

        Returns:
            Order object or None
        """
        return self.orders.get(client_order_id)

    def clear_history(self) -> None:
        """Clear order history but keep open orders."""
        self.orders = {
            k: v
            for k, v in self.orders.items()
            if v.status == OrderStatus.OPEN
        }
