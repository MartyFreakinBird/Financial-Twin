"""
Market Data Provider
Simulates market data feeds for crypto, forex, and traditional assets
"""

import logging
import random
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDataProvider:
    """Provides market data for various financial instruments"""
    
    def __init__(self):
        self.base_prices = self._initialize_base_prices()
        self.price_history = {}
        self.market_hours = {"start": 9, "end": 17}  # 9 AM to 5 PM
        
        # Initialize price history for all symbols
        self._initialize_price_history()
        
        logger.info(f"MarketDataProvider initialized with {len(self.base_prices)} symbols")
    
    def _initialize_base_prices(self) -> Dict[str, float]:
        """Initialize base prices for various instruments"""
        return {
            # Crypto pairs
            "BTC/USDT": 45000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 100.0,
            "ADA/USDT": 0.50,
            "DOT/USDT": 8.0,
            "LINK/USDT": 15.0,
            "UNI/USDT": 7.0,
            "AVAX/USDT": 35.0,
            "MATIC/USDT": 0.80,
            "ATOM/USDT": 12.0,
            
            # Forex pairs
            "USDJPY": 150.0,
            "EURUSD": 1.08,
            "GBPUSD": 1.26,
            "AUDUSD": 0.65,
            "USDCAD": 1.36,
            "USDCHF": 0.91,
            "NZDUSD": 0.62,
            "EURGBP": 0.86,
            "EURJPY": 162.0,
            "GBPJPY": 189.0,
            
            # Stock indices (simplified)
            "SPY": 450.0,
            "QQQ": 380.0,
            "DIA": 350.0,
            "IWM": 200.0,
            
            # Commodities
            "GOLD": 2000.0,
            "SILVER": 25.0,
            "OIL": 75.0,
            "COPPER": 4.0
        }
    
    def _initialize_price_history(self):
        """Initialize price history for all symbols"""
        for symbol in self.base_prices:
            self.price_history[symbol] = self._generate_historical_data(symbol, days=30)
    
    def _generate_historical_data(self, symbol: str, days: int = 30) -> List[PriceBar]:
        """Generate historical price data for a symbol"""
        history = []
        base_price = self.base_prices[symbol]
        current_price = base_price
        
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24):  # Hourly data
            timestamp = start_date + timedelta(hours=i)
            
            # Generate price movement
            volatility = self._get_symbol_volatility(symbol)
            price_change = random.gauss(0, volatility) * current_price
            
            # Calculate OHLC
            open_price = current_price
            close_price = max(0.01, current_price + price_change)
            
            # High and low with some randomness
            high_range = abs(close_price - open_price) * random.uniform(1.1, 2.0)
            low_range = abs(close_price - open_price) * random.uniform(1.1, 2.0)
            
            high_price = max(open_price, close_price) + high_range
            low_price = min(open_price, close_price) - low_range
            low_price = max(0.01, low_price)  # Ensure positive price
            
            # Volume (simplified)
            base_volume = self._get_base_volume(symbol)
            volume = base_volume * random.uniform(0.5, 2.0)
            
            price_bar = PriceBar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            history.append(price_bar)
            current_price = close_price
        
        return history
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility parameter for a symbol"""
        if "BTC" in symbol or "ETH" in symbol:
            return 0.04  # 4% volatility for major crypto
        elif "/" in symbol and any(crypto in symbol for crypto in ["SOL", "ADA", "DOT", "LINK"]):
            return 0.06  # 6% volatility for altcoins
        elif symbol in ["USDJPY", "EURUSD", "GBPUSD"]:
            return 0.008  # 0.8% volatility for major forex
        elif symbol in ["SPY", "QQQ", "DIA"]:
            return 0.015  # 1.5% volatility for stock indices
        elif symbol in ["GOLD", "SILVER", "OIL"]:
            return 0.025  # 2.5% volatility for commodities
        else:
            return 0.02  # Default 2% volatility
    
    def _get_base_volume(self, symbol: str) -> float:
        """Get base volume for a symbol"""
        if "BTC" in symbol:
            return 1000000  # High volume for BTC
        elif "ETH" in symbol:
            return 2000000  # High volume for ETH
        elif "/" in symbol:
            return 500000   # Medium volume for other crypto
        elif symbol in ["USDJPY", "EURUSD", "GBPUSD"]:
            return 10000000  # Very high volume for major forex
        else:
            return 100000   # Default volume
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for a specific symbol"""
        if symbol not in self.base_prices:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        
        try:
            # Update price history with latest data
            self._update_symbol_price(symbol)
            
            # Get recent price bars
            recent_bars = self.price_history[symbol][-100:]  # Last 100 bars
            
            # Convert to dictionary format
            prices = []
            for bar in recent_bars:
                prices.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume
                })
            
            # Calculate basic statistics
            current_price = recent_bars[-1].close if recent_bars else self.base_prices[symbol]
            previous_price = recent_bars[-2].close if len(recent_bars) > 1 else current_price
            
            price_change = current_price - previous_price
            price_change_percent = (price_change / previous_price * 100) if previous_price > 0 else 0
            
            # Calculate 24h statistics
            if len(recent_bars) >= 24:
                day_ago_price = recent_bars[-24].close
                day_change = current_price - day_ago_price
                day_change_percent = (day_change / day_ago_price * 100) if day_ago_price > 0 else 0
            else:
                day_change = price_change
                day_change_percent = price_change_percent
            
            # Volume statistics
            recent_volumes = [bar.volume for bar in recent_bars[-24:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "day_change": day_change,
                "day_change_percent": day_change_percent,
                "volume": recent_bars[-1].volume if recent_bars else 0,
                "avg_volume": avg_volume,
                "high_24h": max(bar.high for bar in recent_bars[-24:]) if len(recent_bars) >= 24 else current_price,
                "low_24h": min(bar.low for bar in recent_bars[-24:]) if len(recent_bars) >= 24 else current_price,
                "prices": prices,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol data for {symbol}: {e}")
            return None
    
    def _update_symbol_price(self, symbol: str):
        """Update the latest price for a symbol"""
        if symbol not in self.price_history:
            return
        
        # Get the last price bar
        last_bar = self.price_history[symbol][-1]
        
        # Check if we need a new bar (more than 1 hour old)
        if datetime.now() - last_bar.timestamp > timedelta(hours=1):
            # Generate new price bar
            volatility = self._get_symbol_volatility(symbol)
            price_change = random.gauss(0, volatility) * last_bar.close
            
            open_price = last_bar.close
            close_price = max(0.01, last_bar.close + price_change)
            
            # Generate OHLC
            high_range = abs(close_price - open_price) * random.uniform(1.1, 2.0)
            low_range = abs(close_price - open_price) * random.uniform(1.1, 2.0)
            
            high_price = max(open_price, close_price) + high_range
            low_price = max(0.01, min(open_price, close_price) - low_range)
            
            base_volume = self._get_base_volume(symbol)
            volume = base_volume * random.uniform(0.5, 2.0)
            
            new_bar = PriceBar(
                timestamp=datetime.now(),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            self.price_history[symbol].append(new_bar)
            
            # Keep only last 1000 bars to prevent memory issues
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overall market overview"""
        try:
            crypto_symbols = [s for s in self.base_prices.keys() if "/" in s and "USDT" in s]
            forex_symbols = [s for s in self.base_prices.keys() if s in ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD"]]
            
            # Calculate market movements
            crypto_changes = []
            forex_changes = []
            
            for symbol in crypto_symbols[:5]:  # Top 5 crypto
                data = self.get_symbol_data(symbol)
                if data:
                    crypto_changes.append(data["day_change_percent"])
            
            for symbol in forex_symbols:
                data = self.get_symbol_data(symbol)
                if data:
                    forex_changes.append(data["day_change_percent"])
            
            # Calculate averages
            avg_crypto_change = sum(crypto_changes) / len(crypto_changes) if crypto_changes else 0
            avg_forex_change = sum(forex_changes) / len(forex_changes) if forex_changes else 0
            
            # Determine market sentiment
            overall_change = (avg_crypto_change + avg_forex_change) / 2
            if overall_change > 2:
                sentiment = "bullish"
            elif overall_change < -2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Determine volatility
            all_changes = crypto_changes + forex_changes
            if all_changes:
                volatility_score = sum(abs(change) for change in all_changes) / len(all_changes)
                if volatility_score > 3:
                    volatility = "high"
                elif volatility_score > 1.5:
                    volatility = "medium"
                else:
                    volatility = "low"
            else:
                volatility = "medium"
            
            return {
                "sentiment": sentiment,
                "volatility": volatility,
                "crypto_performance": avg_crypto_change,
                "forex_performance": avg_forex_change,
                "overall_performance": overall_change,
                "market_hours": self._is_market_hours(),
                "symbols_tracked": len(self.base_prices),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                "sentiment": "neutral",
                "volatility": "medium",
                "market_hours": True,
                "error": str(e)
            }
    
    def _is_market_hours(self) -> bool:
        """Check if it's currently market hours"""
        current_hour = datetime.now().hour
        return self.market_hours["start"] <= current_hour <= self.market_hours["end"]
    
    def get_multiple_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Get data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            data = self.get_symbol_data(symbol)
            if data:
                results[symbol] = data
        
        return results
    
    def get_price_alerts(self, symbol: str, upper_threshold: float, 
                        lower_threshold: float) -> Dict[str, Any]:
        """Check for price alerts"""
        data = self.get_symbol_data(symbol)
        if not data:
            return {"alerts": [], "symbol": symbol}
        
        current_price = data["current_price"]
        alerts = []
        
        if current_price >= upper_threshold:
            alerts.append({
                "type": "upper_threshold",
                "message": f"{symbol} price ${current_price:.2f} reached upper threshold ${upper_threshold:.2f}",
                "timestamp": datetime.now().isoformat()
            })
        
        if current_price <= lower_threshold:
            alerts.append({
                "type": "lower_threshold", 
                "message": f"{symbol} price ${current_price:.2f} reached lower threshold ${lower_threshold:.2f}",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "alerts": alerts
        }
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        data = self.get_symbol_data(symbol)
        if not data or not data.get("prices"):
            return {}
        
        prices = [float(p["close"]) for p in data["prices"]]
        
        if len(prices) < 20:
            return {"error": "Insufficient data for technical indicators"}
        
        # Simple Moving Averages
        sma_20 = sum(prices[-20:]) / 20
        sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else None
        
        # RSI calculation (simplified)
        rsi = self._calculate_rsi(prices)
        
        # MACD calculation (simplified)
        macd_line, signal_line = self._calculate_macd(prices)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
        
        current_price = prices[-1]
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "price_above_sma20": current_price > sma_20,
            "price_above_sma50": current_price > sma_50 if sma_50 else None,
            "rsi_overbought": rsi > 70 if rsi else False,
            "rsi_oversold": rsi < 30 if rsi else False,
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        try:
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return None
    
    def _calculate_macd(self, prices: List[float]) -> tuple[Optional[float], Optional[float]]:
        """Calculate MACD and signal line"""
        if len(prices) < 26:
            return None, None
        
        try:
            # Calculate EMAs
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            
            if ema_12 is None or ema_26 is None:
                return None, None
            
            macd_line = ema_12 - ema_26
            
            # Calculate signal line (EMA of MACD)
            # For simplicity, using a basic average instead of true EMA
            macd_values = [ema_12 - ema_26]  # Simplified
            signal_line = macd_line * 0.9  # Simplified signal
            
            return macd_line, signal_line
            
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return None, None
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        try:
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period  # Start with SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return None
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                  std_dev: float = 2) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        try:
            recent_prices = prices[-period:]
            middle = sum(recent_prices) / period
            
            # Calculate standard deviation
            variance = sum((price - middle) ** 2 for price in recent_prices) / period
            std = variance ** 0.5
            
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return upper, middle, lower
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return None, None, None
    
    def simulate_market_event(self, event_type: str, intensity: float = 1.0):
        """Simulate market events (news, economic data, etc.)"""
        logger.info(f"Simulating market event: {event_type} with intensity {intensity}")
        
        # Apply event effects to all symbols
        for symbol in self.base_prices:
            if event_type == "crypto_crash":
                if "/" in symbol and "USDT" in symbol:
                    # Crypto crash affects crypto pairs
                    multiplier = 1 - (0.1 * intensity)  # Up to 10% drop
                    self.base_prices[symbol] *= multiplier
                    
            elif event_type == "forex_volatility":
                if symbol in ["USDJPY", "EURUSD", "GBPUSD"]:
                    # Increased volatility for forex
                    change = random.uniform(-0.03, 0.03) * intensity
                    self.base_prices[symbol] *= (1 + change)
                    
            elif event_type == "market_rally":
                # General market rally
                change = random.uniform(0, 0.05) * intensity
                self.base_prices[symbol] *= (1 + change)
                
            elif event_type == "market_crash":
                # General market crash
                change = random.uniform(-0.08, 0) * intensity
                self.base_prices[symbol] *= (1 + change)
        
        # Ensure prices remain positive
        for symbol in self.base_prices:
            self.base_prices[symbol] = max(0.01, self.base_prices[symbol])
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get market data provider status"""
        return {
            "provider_type": "MockMarketData",
            "symbols_count": len(self.base_prices),
            "crypto_symbols": len([s for s in self.base_prices if "/" in s]),
            "forex_symbols": len([s for s in self.base_prices if s in ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD"]]),
            "data_points": sum(len(self.price_history.get(s, [])) for s in self.base_prices),
            "market_hours": self._is_market_hours(),
            "last_updated": datetime.now().isoformat(),
            "status": "active"
        }

