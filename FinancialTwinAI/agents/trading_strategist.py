"""
Trading Strategist Agent
Analyzes crypto/forex data and deploys trades through mock or real APIs
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import pandas as pd

from agents.base_agent import BaseAgent
from core.llm_client import llm_client
from trading.market_data import MarketDataProvider
from trading.mock_broker import MockBroker

logger = logging.getLogger(__name__)

class TradingStrategistAgent(BaseAgent):
    """Agent responsible for trading strategy and execution"""
    
    def __init__(self):
        super().__init__(
            name="TradingStrategist",
            description="Analyzes crypto/forex data and deploys trades through mock or real APIs"
        )
        self.market_data = MarketDataProvider()
        self.broker = MockBroker()
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions and trading opportunities"""
        try:
            # Get watchlist from memory or use defaults
            watchlist = self.get_memory("trading_watchlist") or ["BTC/USDT", "ETH/USDT", "USDJPY", "EURUSD"]
            
            analysis_data = {
                "watchlist": watchlist,
                "analysis_timestamp": datetime.now().isoformat(),
                "market_analysis": {},
                "technical_indicators": {},
                "trading_signals": {}
            }
            
            # Analyze each symbol in watchlist
            for symbol in watchlist:
                try:
                    # Get market data
                    market_data = self.market_data.get_symbol_data(symbol)
                    if not market_data:
                        continue
                    
                    # Calculate technical indicators
                    technical_analysis = self.calculate_technical_indicators(market_data)
                    
                    # Generate trading signals
                    trading_signals = self.generate_trading_signals(technical_analysis)
                    
                    analysis_data["market_analysis"][symbol] = market_data
                    analysis_data["technical_indicators"][symbol] = technical_analysis
                    analysis_data["trading_signals"][symbol] = trading_signals
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Get portfolio context
            portfolio_positions = self.db_manager.get_portfolio_positions()
            latest_balance = self.db_manager.get_latest_balance()
            
            analysis_data.update({
                "portfolio_positions": portfolio_positions,
                "account_balance": latest_balance,
                "available_capital": latest_balance.get("cash_balance", 0) if latest_balance else 0
            })
            
            # Use LLM for market sentiment analysis
            if llm_client.is_available():
                market_context = {
                    "symbols_analyzed": len(analysis_data["market_analysis"]),
                    "signals_generated": sum(1 for signals in analysis_data["trading_signals"].values() 
                                           if signals.get("signal") != "hold"),
                    "portfolio_value": latest_balance.get("total_portfolio_value", 0) if latest_balance else 0
                }
                
                llm_analysis = llm_client.analyze_financial_data(
                    market_context,
                    "Trading market analysis and sentiment evaluation"
                )
                if llm_analysis:
                    analysis_data["llm_insights"] = llm_analysis
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"TradingStrategist analysis failed: {e}")
            return {}
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make trading decisions based on analysis"""
        try:
            trading_signals = analysis.get("trading_signals", {})
            portfolio_positions = analysis.get("portfolio_positions", [])
            available_capital = analysis.get("available_capital", 0)
            
            # Filter for actionable signals
            actionable_signals = {}
            for symbol, signal_data in trading_signals.items():
                if signal_data.get("signal") in ["buy", "sell"] and signal_data.get("confidence", 0) > 0.6:
                    actionable_signals[symbol] = signal_data
            
            if not actionable_signals:
                return {
                    "action": "hold",
                    "reasoning": "No actionable trading signals with sufficient confidence",
                    "confidence": 0.3,
                    "risk_level": "low"
                }
            
            # Select best trading opportunity
            best_signal = max(
                actionable_signals.items(),
                key=lambda x: x[1].get("confidence", 0) * x[1].get("expected_return", 0)
            )
            
            symbol, signal_data = best_signal
            
            # Calculate position size
            position_size = self.calculate_position_size(
                signal_data, available_capital, portfolio_positions
            )
            
            # Use LLM for trading decision validation
            trading_decision = None
            if llm_client.is_available():
                market_data = analysis.get("market_analysis", {}).get(symbol, {})
                portfolio_data = {
                    "available_capital": available_capital,
                    "positions": portfolio_positions,
                    "target_symbol": symbol
                }
                
                trading_decision = llm_client.make_trading_decision(
                    market_data, portfolio_data, f"Signal: {signal_data}"
                )
            
            decision = {
                "action": signal_data["signal"],
                "symbol": symbol,
                "quantity": position_size,
                "signal_confidence": signal_data.get("confidence", 0),
                "expected_return": signal_data.get("expected_return", 0),
                "stop_loss": signal_data.get("stop_loss"),
                "take_profit": signal_data.get("take_profit"),
                "reasoning": self.generate_trading_reasoning(signal_data, analysis),
                "confidence": self.calculate_trading_confidence(signal_data, analysis),
                "risk_level": self.assess_trading_risk(signal_data, position_size, available_capital),
                "llm_decision": trading_decision,
                "decision_timestamp": datetime.now().isoformat()
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"TradingStrategist decision failed: {e}")
            return {}
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trading decision"""
        try:
            action = decision.get("action")
            
            if action == "hold":
                return {
                    "status": "success",
                    "message": "Holding positions, no trades executed",
                    "trade_id": None
                }
            
            symbol = decision.get("symbol")
            quantity = decision.get("quantity", 0)
            
            if not symbol or quantity <= 0:
                return {
                    "status": "error",
                    "message": "Invalid trading parameters"
                }
            
            # Execute trade through mock broker
            if action == "buy":
                trade_result = self.broker.place_buy_order(symbol, quantity)
            elif action == "sell":
                trade_result = self.broker.place_sell_order(symbol, quantity)
            else:
                return {
                    "status": "error", 
                    "message": f"Unknown action: {action}"
                }
            
            if trade_result.get("status") == "filled":
                # Log transaction in database
                transaction_id = self.db_manager.log_transaction(
                    transaction_type=action,
                    symbol=symbol,
                    quantity=quantity,
                    price=trade_result.get("fill_price", 0)
                )
                
                # Update portfolio position
                if action == "buy":
                    self.update_portfolio_after_buy(symbol, quantity, trade_result["fill_price"])
                elif action == "sell":
                    self.update_portfolio_after_sell(symbol, quantity, trade_result["fill_price"])
                
                # Store trade details
                trade_data = {
                    "trade_id": trade_result.get("trade_id"),
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "fill_price": trade_result.get("fill_price"),
                    "transaction_id": transaction_id,
                    "execution_time": datetime.now().isoformat()
                }
                
                self.store_memory("last_trade", trade_data)
                
                return {
                    "status": "success",
                    "trade_result": trade_result,
                    "transaction_id": transaction_id,
                    "message": f"Successfully executed {action} order for {quantity} {symbol}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Trade execution failed: {trade_result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"TradingStrategist execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def calculate_technical_indicators(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators from market data"""
        try:
            prices = market_data.get("prices", [])
            if len(prices) < 20:
                return {"error": "Insufficient data for technical analysis"}
            
            # Convert to pandas series for calculations
            price_series = pd.Series([p["close"] for p in prices])
            
            # Simple Moving Averages
            sma_20 = price_series.rolling(20).mean().iloc[-1]
            sma_50 = price_series.rolling(50).mean().iloc[-1] if len(prices) >= 50 else None
            
            # RSI calculation
            rsi = self.calculate_rsi(price_series)
            
            # MACD calculation
            macd_line, signal_line = self.calculate_macd(price_series)
            
            # Current price
            current_price = prices[-1]["close"]
            
            indicators = {
                "current_price": current_price,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi,
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "price_above_sma20": current_price > sma_20 if sma_20 else None,
                "price_above_sma50": current_price > sma_50 if sma_50 else None,
                "rsi_overbought": rsi > 70 if rsi else None,
                "rsi_oversold": rsi < 30 if rsi else None,
                "macd_bullish": macd_line > signal_line if macd_line and signal_line else None
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return {"error": str(e)}
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return None
    
    def calculate_macd(self, prices: pd.Series) -> tuple[Optional[float], Optional[float]]:
        """Calculate MACD and signal line"""
        try:
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            return macd_line.iloc[-1], signal_line.iloc[-1]
        except:
            return None, None
    
    def generate_trading_signals(self, technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on technical analysis"""
        if technical_analysis.get("error"):
            return {"signal": "hold", "confidence": 0, "reasoning": "Insufficient data"}
        
        signals = []
        confidence_factors = []
        
        # RSI signals
        if technical_analysis.get("rsi_oversold"):
            signals.append("buy")
            confidence_factors.append(0.3)
        elif technical_analysis.get("rsi_overbought"):
            signals.append("sell")
            confidence_factors.append(0.3)
        
        # Moving average signals
        if technical_analysis.get("price_above_sma20") and technical_analysis.get("price_above_sma50"):
            signals.append("buy")
            confidence_factors.append(0.2)
        elif not technical_analysis.get("price_above_sma20") and not technical_analysis.get("price_above_sma50"):
            signals.append("sell")
            confidence_factors.append(0.2)
        
        # MACD signals
        if technical_analysis.get("macd_bullish"):
            signals.append("buy")
            confidence_factors.append(0.25)
        elif technical_analysis.get("macd_bullish") is False:
            signals.append("sell")
            confidence_factors.append(0.25)
        
        # Determine final signal
        if not signals:
            return {"signal": "hold", "confidence": 0.3, "reasoning": "No clear signals"}
        
        # Count signal types
        buy_signals = signals.count("buy")
        sell_signals = signals.count("sell")
        
        if buy_signals > sell_signals:
            final_signal = "buy"
            confidence = sum(confidence_factors) * (buy_signals / len(signals))
        elif sell_signals > buy_signals:
            final_signal = "sell"
            confidence = sum(confidence_factors) * (sell_signals / len(signals))
        else:
            final_signal = "hold"
            confidence = 0.4
        
        # Calculate expected return and risk metrics
        current_price = technical_analysis.get("current_price", 0)
        sma_20 = technical_analysis.get("sma_20", current_price)
        
        expected_return = abs(current_price - sma_20) / current_price if current_price > 0 else 0
        
        # Set stop loss and take profit levels
        if final_signal == "buy":
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.04  # 4% take profit
        elif final_signal == "sell":
            stop_loss = current_price * 1.02  # 2% stop loss for short
            take_profit = current_price * 0.96  # 4% take profit for short
        else:
            stop_loss = None
            take_profit = None
        
        return {
            "signal": final_signal,
            "confidence": min(confidence, 1.0),
            "reasoning": f"Based on {len(signals)} technical indicators",
            "expected_return": expected_return,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "technical_factors": {
                "rsi": technical_analysis.get("rsi"),
                "price_vs_sma20": technical_analysis.get("price_above_sma20"),
                "macd_bullish": technical_analysis.get("macd_bullish")
            }
        }
    
    def calculate_position_size(self, signal_data: Dict[str, Any], 
                               available_capital: float, 
                               portfolio_positions: List[Dict[str, Any]]) -> float:
        """Calculate appropriate position size"""
        if available_capital <= 0:
            return 0
        
        # Base position size as percentage of available capital
        base_percentage = 0.1  # 10% of available capital
        
        # Adjust based on confidence
        confidence = signal_data.get("confidence", 0.5)
        confidence_multiplier = confidence * 1.5  # Scale up to 1.5x for high confidence
        
        # Adjust based on expected return
        expected_return = signal_data.get("expected_return", 0)
        return_multiplier = min(1.5, 1 + expected_return * 2)  # Cap at 1.5x
        
        # Calculate final percentage
        final_percentage = base_percentage * confidence_multiplier * return_multiplier
        final_percentage = min(final_percentage, 0.2)  # Cap at 20% of capital
        
        position_value = available_capital * final_percentage
        
        # Convert to quantity (assuming we have current price)
        current_price = signal_data.get("technical_factors", {}).get("current_price", 1)
        if current_price > 0:
            return position_value / current_price
        
        return 0
    
    def generate_trading_reasoning(self, signal_data: Dict[str, Any], 
                                  analysis: Dict[str, Any]) -> str:
        """Generate reasoning for trading decision"""
        signal = signal_data.get("signal", "hold")
        confidence = signal_data.get("confidence", 0)
        
        reasoning = f"Signal: {signal.upper()} with {confidence:.1%} confidence. "
        reasoning += signal_data.get("reasoning", "")
        
        technical_factors = signal_data.get("technical_factors", {})
        if technical_factors:
            reasoning += f" Technical factors: RSI={technical_factors.get('rsi', 'N/A'):.1f}, "
            reasoning += f"Price above SMA20: {technical_factors.get('price_vs_sma20', 'N/A')}, "
            reasoning += f"MACD bullish: {technical_factors.get('macd_bullish', 'N/A')}."
        
        return reasoning
    
    def calculate_trading_confidence(self, signal_data: Dict[str, Any],
                                    analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence in trading decision"""
        base_confidence = signal_data.get("confidence", 0.5)
        
        # Adjust based on market analysis quality
        market_analysis = analysis.get("market_analysis", {})
        if market_analysis:
            base_confidence += 0.1
        
        # Adjust based on LLM insights
        if analysis.get("llm_insights"):
            llm_confidence = analysis["llm_insights"].get("confidence", 0.5)
            base_confidence = (base_confidence + llm_confidence) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    def assess_trading_risk(self, signal_data: Dict[str, Any], 
                           position_size: float, available_capital: float) -> str:
        """Assess risk level of the trading decision"""
        position_percentage = (position_size / available_capital * 100) if available_capital > 0 else 0
        expected_return = signal_data.get("expected_return", 0)
        confidence = signal_data.get("confidence", 0.5)
        
        risk_score = position_percentage + (expected_return * 50) - (confidence * 30)
        
        if risk_score > 20:
            return "high"
        elif risk_score > 10:
            return "medium"
        else:
            return "low"
    
    def update_portfolio_after_buy(self, symbol: str, quantity: float, price: float):
        """Update portfolio position after buy order"""
        # Get existing position
        positions = self.db_manager.get_portfolio_positions()
        existing_position = next((p for p in positions if p["symbol"] == symbol), None)
        
        if existing_position:
            # Update existing position
            total_quantity = existing_position["quantity"] + quantity
            total_cost = (existing_position["quantity"] * existing_position["average_price"]) + (quantity * price)
            new_average_price = total_cost / total_quantity
            
            self.db_manager.update_portfolio_position(symbol, total_quantity, new_average_price, price)
        else:
            # Create new position
            self.db_manager.update_portfolio_position(symbol, quantity, price, price)
    
    def update_portfolio_after_sell(self, symbol: str, quantity: float, price: float):
        """Update portfolio position after sell order"""
        positions = self.db_manager.get_portfolio_positions()
        existing_position = next((p for p in positions if p["symbol"] == symbol), None)
        
        if existing_position:
            new_quantity = max(0, existing_position["quantity"] - quantity)
            if new_quantity > 0:
                self.db_manager.update_portfolio_position(
                    symbol, new_quantity, existing_position["average_price"], price
                )
            else:
                # Position closed
                self.db_manager.update_portfolio_position(symbol, 0, 0, price)
