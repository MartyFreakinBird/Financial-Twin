"""
Mock Broker Implementation
Simulates trading operations for paper trading and testing
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class Order:
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    order_type: OrderType
    status: OrderStatus
    price: Optional[float] = None
    stop_price: Optional[float] = None
    fill_price: Optional[float] = None
    filled_quantity: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None

class MockBroker:
    """Mock broker for simulating trading operations"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.account_balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}  # symbol -> quantity
        self.orders = {}  # order_id -> Order
        self.trade_history = []
        self.fees_per_trade = 0.001  # 0.1% fee
        
        # Market data simulation
        self.market_prices = self._initialize_market_prices()
        
        logger.info(f"MockBroker initialized with balance: ${initial_balance:,.2f}")
    
    def _initialize_market_prices(self) -> Dict[str, float]:
        """Initialize mock market prices"""
        return {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 100.0,
            "ADA/USDT": 0.50,
            "DOT/USDT": 8.0,
            "USDJPY": 150.0,
            "EURUSD": 1.08,
            "GBPUSD": 1.26,
            "AUDUSD": 0.65,
            "USDCAD": 1.36
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        base_price = self.market_prices.get(symbol)
        if not base_price:
            return None
        
        # Add some random variation (±1%)
        import random
        variation = random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + variation)
        
        # Update stored price for consistency
        self.market_prices[symbol] = current_price
        
        return current_price
    
    def place_buy_order(self, symbol: str, quantity: float, 
                       order_type: OrderType = OrderType.MARKET,
                       price: Optional[float] = None) -> Dict[str, Any]:
        """Place a buy order"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {
                    "status": "rejected",
                    "message": f"Unknown symbol: {symbol}"
                }
            
            # Calculate order value
            if order_type == OrderType.MARKET:
                execution_price = current_price
            else:
                execution_price = price or current_price
            
            order_value = quantity * execution_price
            fee = order_value * self.fees_per_trade
            total_cost = order_value + fee
            
            # Check if sufficient balance
            if total_cost > self.account_balance:
                return {
                    "status": "rejected",
                    "message": f"Insufficient balance. Required: ${total_cost:.2f}, Available: ${self.account_balance:.2f}"
                }
            
            # Create order
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side="buy",
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.PENDING,
                price=price,
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            
            # For market orders, execute immediately
            if order_type == OrderType.MARKET:
                return self._execute_order(order_id, execution_price)
            
            return {
                "status": "pending",
                "order_id": order_id,
                "message": f"Buy order placed for {quantity} {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def place_sell_order(self, symbol: str, quantity: float,
                        order_type: OrderType = OrderType.MARKET,
                        price: Optional[float] = None) -> Dict[str, Any]:
        """Place a sell order"""
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return {
                    "status": "rejected",
                    "message": f"Unknown symbol: {symbol}"
                }
            
            # Check if sufficient position
            current_position = self.positions.get(symbol, 0.0)
            if quantity > current_position:
                return {
                    "status": "rejected",
                    "message": f"Insufficient position. Requested: {quantity}, Available: {current_position}"
                }
            
            # Create order
            order_id = str(uuid.uuid4())
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side="sell",
                quantity=quantity,
                order_type=order_type,
                status=OrderStatus.PENDING,
                price=price,
                created_at=datetime.now()
            )
            
            self.orders[order_id] = order
            
            # For market orders, execute immediately
            if order_type == OrderType.MARKET:
                execution_price = current_price
                return self._execute_order(order_id, execution_price)
            
            return {
                "status": "pending",
                "order_id": order_id,
                "message": f"Sell order placed for {quantity} {symbol}"
            }
            
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _execute_order(self, order_id: str, execution_price: float) -> Dict[str, Any]:
        """Execute a pending order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return {
                    "status": "error",
                    "message": "Order not found"
                }
            
            # Calculate execution details
            order_value = order.quantity * execution_price
            fee = order_value * self.fees_per_trade
            
            if order.side == "buy":
                total_cost = order_value + fee
                
                # Update account balance
                self.account_balance -= total_cost
                
                # Update position
                current_position = self.positions.get(order.symbol, 0.0)
                self.positions[order.symbol] = current_position + order.quantity
                
            else:  # sell
                net_proceeds = order_value - fee
                
                # Update account balance
                self.account_balance += net_proceeds
                
                # Update position
                current_position = self.positions.get(order.symbol, 0.0)
                self.positions[order.symbol] = current_position - order.quantity
                
                # Remove position if zero
                if self.positions[order.symbol] <= 0:
                    del self.positions[order.symbol]
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.fill_price = execution_price
            order.filled_quantity = order.quantity
            order.filled_at = datetime.now()
            
            # Add to trade history
            trade_record = {
                "trade_id": f"trade_{int(time.time())}",
                "order_id": order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": execution_price,
                "value": order_value,
                "fee": fee,
                "timestamp": datetime.now().isoformat()
            }
            self.trade_history.append(trade_record)
            
            logger.info(f"Order executed: {order.side} {order.quantity} {order.symbol} at ${execution_price:.2f}")
            
            return {
                "status": "filled",
                "trade_id": trade_record["trade_id"],
                "order_id": order_id,
                "fill_price": execution_price,
                "filled_quantity": order.quantity,
                "fee": fee,
                "message": f"Order filled: {order.side} {order.quantity} {order.symbol}"
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return {
                    "status": "error",
                    "message": "Order not found"
                }
            
            if order.status != OrderStatus.PENDING:
                return {
                    "status": "error",
                    "message": f"Cannot cancel order with status: {order.status.value}"
                }
            
            order.status = OrderStatus.CANCELLED
            
            return {
                "status": "cancelled",
                "order_id": order_id,
                "message": "Order cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        # Calculate portfolio value
        portfolio_value = 0.0
        position_values = {}
        
        for symbol, quantity in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                position_value = quantity * current_price
                portfolio_value += position_value
                position_values[symbol] = {
                    "quantity": quantity,
                    "current_price": current_price,
                    "value": position_value
                }
        
        total_value = self.account_balance + portfolio_value
        
        # Calculate P&L
        total_pnl = total_value - self.initial_balance
        pnl_percentage = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        return {
            "account_balance": self.account_balance,
            "portfolio_value": portfolio_value,
            "total_value": total_value,
            "initial_balance": self.initial_balance,
            "total_pnl": total_pnl,
            "pnl_percentage": pnl_percentage,
            "positions": position_values,
            "position_count": len(self.positions),
            "total_trades": len(self.trade_history)
        }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        position_details = {}
        
        for symbol, quantity in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                position_details[symbol] = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "current_price": current_price,
                    "market_value": quantity * current_price
                }
        
        return position_details
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        order = self.orders.get(order_id)
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "price": order.price,
            "fill_price": order.fill_price,
            "filled_quantity": order.filled_quantity,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None
        }
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        open_orders = []
        
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                open_orders.append(self.get_order_status(order.order_id))
        
        return open_orders
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        return self.trade_history[-limit:] if limit > 0 else self.trade_history
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_fees": 0.0
            }
        
        total_fees = sum(trade.get("fee", 0) for trade in self.trade_history)
        total_trades = len(self.trade_history)
        
        # Calculate realized P&L from completed buy/sell pairs
        buy_trades = [t for t in self.trade_history if t["side"] == "buy"]
        sell_trades = [t for t in self.trade_history if t["side"] == "sell"]
        
        winning_trades = 0
        losing_trades = 0
        total_realized_pnl = 0.0
        
        # Simple P&L calculation (first-in, first-out)
        for sell_trade in sell_trades:
            symbol = sell_trade["symbol"]
            sell_price = sell_trade["price"]
            sell_quantity = sell_trade["quantity"]
            
            # Find corresponding buy trades
            for buy_trade in buy_trades:
                if (buy_trade["symbol"] == symbol and 
                    buy_trade.get("matched", False) != True):
                    
                    buy_price = buy_trade["price"]
                    buy_quantity = buy_trade["quantity"]
                    
                    # Match quantities
                    matched_quantity = min(sell_quantity, buy_quantity)
                    pnl = matched_quantity * (sell_price - buy_price)
                    total_realized_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    # Mark as matched
                    buy_trade["matched"] = True
                    sell_quantity -= matched_quantity
                    
                    if sell_quantity <= 0:
                        break
        
        completed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
        
        account_info = self.get_account_info()
        
        return {
            "total_trades": total_trades,
            "completed_trades": completed_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_fees": total_fees,
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": account_info["total_pnl"] - total_realized_pnl,
            "total_pnl": account_info["total_pnl"],
            "return_percentage": account_info["pnl_percentage"]
        }
    
    def reset_account(self, new_balance: float = None):
        """Reset account to initial state"""
        self.account_balance = new_balance or self.initial_balance
        self.initial_balance = self.account_balance
        self.positions = {}
        self.orders = {}
        self.trade_history = []
        
        logger.info(f"Account reset with balance: ${self.account_balance:,.2f}")
    
    def simulate_market_movement(self, volatility: float = 0.02):
        """Simulate market price movements"""
        import random
        
        for symbol in self.market_prices:
            # Random price movement
            change_percent = random.uniform(-volatility, volatility)
            self.market_prices[symbol] *= (1 + change_percent)
            
            # Ensure prices don't go negative
            self.market_prices[symbol] = max(self.market_prices[symbol], 0.01)
    
    def get_broker_status(self) -> Dict[str, Any]:
        """Get broker status summary"""
        account_info = self.get_account_info()
        performance = self.calculate_performance_metrics()
        
        return {
            "broker_type": "MockBroker",
            "account_balance": account_info["account_balance"],
            "portfolio_value": account_info["portfolio_value"],
            "total_value": account_info["total_value"],
            "total_pnl": account_info["total_pnl"],
            "pnl_percentage": account_info["pnl_percentage"],
            "active_positions": len(self.positions),
            "open_orders": len(self.get_open_orders()),
            "total_trades": performance["total_trades"],
            "win_rate": performance["win_rate"],
            "status": "active",
            "last_updated": datetime.now().isoformat()
        }

