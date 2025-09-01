"""
Reinvestor Agent
Analyzes idle cash and deploys based on thresholds (yield farming, DCA, margin, etc.)
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from agents.base_agent import BaseAgent
from core.llm_client import llm_client
from trading.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

class ReinvestorAgent(BaseAgent):
    """Agent responsible for reinvestment and yield optimization strategies"""
    
    def __init__(self):
        super().__init__(
            name="Reinvestor",
            description="Analyzes idle cash and deploys based on thresholds (yield farming, DCA, margin, etc.)"
        )
        self.market_data = MarketDataProvider()
        self.yield_strategies = [
            "stablecoin_lending", "dca_strategy", "yield_farming", 
            "low_risk_vaults", "money_market", "liquidity_provision"
        ]
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash position and yield opportunities"""
        try:
            # Get current financial state
            latest_balance = self.db_manager.get_latest_balance()
            portfolio_positions = self.db_manager.get_portfolio_positions()
            
            if not latest_balance:
                return {"error": "No balance information available"}
            
            cash_balance = latest_balance.get("cash_balance", 0)
            total_portfolio_value = latest_balance.get("total_portfolio_value", 0)
            total_value = cash_balance + total_portfolio_value
            
            # Calculate cash allocation percentage
            cash_percentage = (cash_balance / total_value * 100) if total_value > 0 else 0
            
            # Get reinvestment thresholds from memory or set defaults
            min_cash_threshold = self.get_memory("min_cash_threshold") or 10.0  # 10%
            max_cash_threshold = self.get_memory("max_cash_threshold") or 25.0  # 25%
            min_investment_amount = self.get_memory("min_investment_amount") or 100.0
            
            # Analyze yield opportunities
            yield_opportunities = await self.analyze_yield_opportunities()
            
            # Get DCA strategy status
            dca_analysis = self.analyze_dca_opportunities(portfolio_positions, cash_balance)
            
            # Get current market conditions
            market_overview = self.market_data.get_market_overview()
            
            analysis_data = {
                "cash_balance": cash_balance,
                "total_portfolio_value": total_portfolio_value,
                "total_value": total_value,
                "cash_percentage": cash_percentage,
                "min_cash_threshold": min_cash_threshold,
                "max_cash_threshold": max_cash_threshold,
                "excess_cash": max(0, cash_balance - (total_value * min_cash_threshold / 100)),
                "cash_deficit": max(0, (total_value * min_cash_threshold / 100) - cash_balance),
                "yield_opportunities": yield_opportunities,
                "dca_analysis": dca_analysis,
                "market_conditions": market_overview,
                "portfolio_positions": portfolio_positions,
                "reinvestment_eligible": cash_percentage > max_cash_threshold and cash_balance > min_investment_amount,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Use LLM for yield strategy analysis
            if llm_client.is_available():
                llm_context = {
                    "cash_percentage": cash_percentage,
                    "excess_cash": analysis_data["excess_cash"],
                    "available_strategies": len(yield_opportunities),
                    "market_sentiment": market_overview.get("sentiment", "neutral")
                }
                
                llm_analysis = llm_client.analyze_financial_data(
                    llm_context,
                    "Reinvestment and yield optimization analysis"
                )
                if llm_analysis:
                    analysis_data["llm_insights"] = llm_analysis
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Reinvestor analysis failed: {e}")
            return {}
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make reinvestment decisions based on analysis"""
        try:
            cash_percentage = analysis.get("cash_percentage", 0)
            excess_cash = analysis.get("excess_cash", 0)
            reinvestment_eligible = analysis.get("reinvestment_eligible", False)
            yield_opportunities = analysis.get("yield_opportunities", [])
            dca_analysis = analysis.get("dca_analysis", {})
            market_conditions = analysis.get("market_conditions", {})
            
            if not reinvestment_eligible:
                return {
                    "action": "hold_cash",
                    "reasoning": f"Cash level ({cash_percentage:.1f}%) within acceptable range or insufficient for reinvestment",
                    "confidence": 0.8,
                    "risk_level": "low"
                }
            
            # Evaluate reinvestment strategies
            strategy_scores = self.evaluate_reinvestment_strategies(
                yield_opportunities, dca_analysis, market_conditions, excess_cash
            )
            
            if not strategy_scores:
                return {
                    "action": "hold_cash",
                    "reasoning": "No suitable reinvestment strategies identified",
                    "confidence": 0.6,
                    "risk_level": "low"
                }
            
            # Select best strategy
            best_strategy = max(strategy_scores, key=lambda x: x["score"])
            
            # Calculate investment amount
            investment_amount = self.calculate_investment_amount(
                excess_cash, best_strategy, analysis
            )
            
            decision = {
                "action": "reinvest",
                "strategy": best_strategy["strategy"],
                "investment_amount": investment_amount,
                "target_allocation": best_strategy.get("allocation", {}),
                "expected_yield": best_strategy.get("expected_yield", 0),
                "risk_level": best_strategy.get("risk_level", "medium"),
                "time_horizon": best_strategy.get("time_horizon", "medium_term"),
                "reasoning": self.generate_reinvestment_reasoning(best_strategy, analysis),
                "confidence": self.calculate_reinvestment_confidence(best_strategy, analysis),
                "alternative_strategies": [s for s in strategy_scores if s != best_strategy][:2],
                "decision_timestamp": datetime.now().isoformat()
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Reinvestor decision failed: {e}")
            return {}
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reinvestment actions"""
        try:
            action = decision.get("action")
            
            if action == "hold_cash":
                return {
                    "status": "success",
                    "message": "Maintaining current cash position",
                    "action_taken": "hold_cash"
                }
            
            elif action == "reinvest":
                return await self.execute_reinvestment(decision)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown reinvestment action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Reinvestor execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_yield_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze available yield opportunities"""
        opportunities = []
        
        # Stablecoin lending opportunities
        stablecoin_yield = self.analyze_stablecoin_lending()
        if stablecoin_yield:
            opportunities.append(stablecoin_yield)
        
        # DeFi yield farming
        defi_opportunities = self.analyze_defi_yields()
        opportunities.extend(defi_opportunities)
        
        # Traditional money market
        money_market = self.analyze_money_market()
        if money_market:
            opportunities.append(money_market)
        
        # Low-risk vault strategies
        vault_strategies = self.analyze_vault_strategies()
        opportunities.extend(vault_strategies)
        
        return opportunities
    
    def analyze_stablecoin_lending(self) -> Optional[Dict[str, Any]]:
        """Analyze stablecoin lending opportunities"""
        return {
            "strategy": "stablecoin_lending",
            "description": "USDC/USDT lending on major platforms",
            "expected_yield": 0.05,  # 5% APY
            "risk_level": "low",
            "liquidity": "high",
            "time_horizon": "flexible",
            "platforms": ["Compound", "Aave", "Celsius"],
            "min_investment": 100,
            "max_allocation": 0.3  # 30% of excess cash
        }
    
    def analyze_defi_yields(self) -> List[Dict[str, Any]]:
        """Analyze DeFi yield farming opportunities"""
        opportunities = []
        
        # Liquidity provision
        opportunities.append({
            "strategy": "liquidity_provision",
            "description": "Provide liquidity to DEX pools",
            "expected_yield": 0.08,  # 8% APY
            "risk_level": "medium",
            "liquidity": "medium",
            "time_horizon": "medium_term",
            "platforms": ["Uniswap", "SushiSwap", "PancakeSwap"],
            "min_investment": 500,
            "max_allocation": 0.2
        })
        
        # Yield farming
        opportunities.append({
            "strategy": "yield_farming",
            "description": "Farm tokens in DeFi protocols",
            "expected_yield": 0.12,  # 12% APY
            "risk_level": "high",
            "liquidity": "medium",
            "time_horizon": "medium_term",
            "platforms": ["Yearn", "Harvest", "Convex"],
            "min_investment": 1000,
            "max_allocation": 0.15
        })
        
        return opportunities
    
    def analyze_money_market(self) -> Optional[Dict[str, Any]]:
        """Analyze traditional money market opportunities"""
        return {
            "strategy": "money_market",
            "description": "High-yield savings and money market accounts",
            "expected_yield": 0.03,  # 3% APY
            "risk_level": "very_low",
            "liquidity": "high",
            "time_horizon": "flexible",
            "platforms": ["Traditional Banks", "Online Banks"],
            "min_investment": 50,
            "max_allocation": 0.5
        }
    
    def analyze_vault_strategies(self) -> List[Dict[str, Any]]:
        """Analyze low-risk vault strategies"""
        return [
            {
                "strategy": "stable_vault",
                "description": "Conservative multi-strategy vault",
                "expected_yield": 0.06,  # 6% APY
                "risk_level": "low",
                "liquidity": "medium",
                "time_horizon": "medium_term",
                "platforms": ["Yearn Stable Vaults", "Rari Capital"],
                "min_investment": 250,
                "max_allocation": 0.25
            }
        ]
    
    def analyze_dca_opportunities(self, portfolio_positions: List[Dict[str, Any]], 
                                 cash_balance: float) -> Dict[str, Any]:
        """Analyze Dollar Cost Averaging opportunities"""
        # Get DCA settings from memory
        dca_enabled = self.get_memory("dca_enabled") or False
        dca_symbols = self.get_memory("dca_symbols") or ["BTC/USDT", "ETH/USDT"]
        dca_frequency = self.get_memory("dca_frequency") or "weekly"
        dca_amount = self.get_memory("dca_amount") or 100.0
        
        last_dca_date = self.get_memory("last_dca_date")
        
        dca_due = False
        if last_dca_date:
            try:
                last_date = datetime.fromisoformat(last_dca_date)
                days_since_dca = (datetime.now() - last_date).days
                
                if dca_frequency == "daily" and days_since_dca >= 1:
                    dca_due = True
                elif dca_frequency == "weekly" and days_since_dca >= 7:
                    dca_due = True
                elif dca_frequency == "monthly" and days_since_dca >= 30:
                    dca_due = True
            except:
                dca_due = True
        else:
            dca_due = True
        
        return {
            "dca_enabled": dca_enabled,
            "dca_symbols": dca_symbols,
            "dca_frequency": dca_frequency,
            "dca_amount": dca_amount,
            "dca_due": dca_due,
            "sufficient_funds": cash_balance >= dca_amount,
            "last_dca_date": last_dca_date
        }
    
    def evaluate_reinvestment_strategies(self, yield_opportunities: List[Dict[str, Any]],
                                       dca_analysis: Dict[str, Any],
                                       market_conditions: Dict[str, Any],
                                       excess_cash: float) -> List[Dict[str, Any]]:
        """Evaluate and score reinvestment strategies"""
        strategy_scores = []
        
        # Evaluate yield opportunities
        for opportunity in yield_opportunities:
            score = self.calculate_strategy_score(opportunity, market_conditions, excess_cash)
            if score > 0:
                strategy_scores.append({
                    "strategy": opportunity["strategy"],
                    "score": score,
                    "expected_yield": opportunity["expected_yield"],
                    "risk_level": opportunity["risk_level"],
                    "allocation": self.calculate_allocation(opportunity, excess_cash),
                    "details": opportunity
                })
        
        # Evaluate DCA strategy
        if dca_analysis.get("dca_enabled") and dca_analysis.get("dca_due") and dca_analysis.get("sufficient_funds"):
            dca_score = self.calculate_dca_score(dca_analysis, market_conditions)
            strategy_scores.append({
                "strategy": "dca_investment",
                "score": dca_score,
                "expected_yield": 0.07,  # Historical crypto returns
                "risk_level": "medium",
                "allocation": {"amount": dca_analysis["dca_amount"], "symbols": dca_analysis["dca_symbols"]},
                "details": dca_analysis
            })
        
        return sorted(strategy_scores, key=lambda x: x["score"], reverse=True)
    
    def calculate_strategy_score(self, opportunity: Dict[str, Any], 
                               market_conditions: Dict[str, Any], 
                               excess_cash: float) -> float:
        """Calculate score for a yield strategy"""
        if excess_cash < opportunity.get("min_investment", 0):
            return 0
        
        base_score = opportunity.get("expected_yield", 0) * 100  # Convert to percentage
        
        # Adjust for risk
        risk_multipliers = {"very_low": 1.2, "low": 1.1, "medium": 1.0, "high": 0.8, "very_high": 0.6}
        risk_level = opportunity.get("risk_level", "medium")
        base_score *= risk_multipliers.get(risk_level, 1.0)
        
        # Adjust for liquidity
        liquidity = opportunity.get("liquidity", "medium")
        liquidity_multipliers = {"high": 1.1, "medium": 1.0, "low": 0.9}
        base_score *= liquidity_multipliers.get(liquidity, 1.0)
        
        # Adjust for market conditions
        market_sentiment = market_conditions.get("sentiment", "neutral")
        if market_sentiment == "bullish" and risk_level in ["medium", "high"]:
            base_score *= 1.1
        elif market_sentiment == "bearish" and risk_level in ["low", "very_low"]:
            base_score *= 1.1
        elif market_sentiment == "bearish" and risk_level in ["high", "very_high"]:
            base_score *= 0.8
        
        return base_score
    
    def calculate_dca_score(self, dca_analysis: Dict[str, Any], 
                          market_conditions: Dict[str, Any]) -> float:
        """Calculate score for DCA strategy"""
        base_score = 70  # Base score for DCA
        
        # Increase score if market is down (good DCA opportunity)
        market_sentiment = market_conditions.get("sentiment", "neutral")
        if market_sentiment == "bearish":
            base_score += 20
        elif market_sentiment == "bullish":
            base_score += 5
        
        # Consistency bonus
        last_dca_date = dca_analysis.get("last_dca_date")
        if last_dca_date:
            base_score += 10  # Bonus for maintaining DCA schedule
        
        return base_score
    
    def calculate_allocation(self, opportunity: Dict[str, Any], excess_cash: float) -> Dict[str, Any]:
        """Calculate allocation for a yield strategy"""
        max_allocation_percent = opportunity.get("max_allocation", 0.2)
        max_amount = excess_cash * max_allocation_percent
        min_investment = opportunity.get("min_investment", 100)
        
        allocation_amount = max(min_investment, min(max_amount, excess_cash * 0.1))
        
        return {
            "amount": allocation_amount,
            "percentage": (allocation_amount / excess_cash * 100) if excess_cash > 0 else 0,
            "strategy": opportunity["strategy"]
        }
    
    def calculate_investment_amount(self, excess_cash: float, 
                                  best_strategy: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> float:
        """Calculate the optimal investment amount"""
        if best_strategy["strategy"] == "dca_investment":
            return best_strategy["allocation"]["amount"]
        
        allocation = best_strategy.get("allocation", {})
        return allocation.get("amount", excess_cash * 0.1)
    
    def generate_reinvestment_reasoning(self, strategy: Dict[str, Any], 
                                      analysis: Dict[str, Any]) -> str:
        """Generate reasoning for reinvestment decision"""
        strategy_name = strategy["strategy"]
        expected_yield = strategy.get("expected_yield", 0)
        risk_level = strategy.get("risk_level", "medium")
        cash_percentage = analysis.get("cash_percentage", 0)
        
        reasoning = f"Selected {strategy_name} strategy with {expected_yield:.1%} expected yield. "
        reasoning += f"Current cash allocation ({cash_percentage:.1f}%) exceeds target range. "
        reasoning += f"Strategy offers {risk_level} risk profile suitable for current market conditions."
        
        market_sentiment = analysis.get("market_conditions", {}).get("sentiment", "neutral")
        if market_sentiment != "neutral":
            reasoning += f" Market sentiment ({market_sentiment}) supports this allocation."
        
        return reasoning
    
    def calculate_reinvestment_confidence(self, strategy: Dict[str, Any],
                                        analysis: Dict[str, Any]) -> float:
        """Calculate confidence in reinvestment decision"""
        base_confidence = 0.7
        
        # Increase confidence for lower risk strategies
        risk_level = strategy.get("risk_level", "medium")
        risk_adjustments = {"very_low": 0.2, "low": 0.1, "medium": 0.0, "high": -0.1, "very_high": -0.2}
        base_confidence += risk_adjustments.get(risk_level, 0.0)
        
        # Increase confidence if we have good market data
        if analysis.get("market_conditions") and analysis.get("llm_insights"):
            base_confidence += 0.1
        
        # Increase confidence for established strategies (DCA)
        if strategy["strategy"] == "dca_investment":
            base_confidence += 0.1
        
        return min(1.0, max(0.3, base_confidence))
    
    async def execute_reinvestment(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the reinvestment strategy"""
        try:
            strategy = decision.get("strategy")
            investment_amount = decision.get("investment_amount", 0)
            
            if investment_amount <= 0:
                return {
                    "status": "error",
                    "message": "Invalid investment amount"
                }
            
            execution_result = {
                "status": "success",
                "strategy": strategy,
                "investment_amount": investment_amount,
                "actions_taken": []
            }
            
            if strategy == "dca_investment":
                result = await self.execute_dca_investment(decision)
                execution_result.update(result)
            
            elif strategy in ["stablecoin_lending", "money_market"]:
                result = await self.execute_lending_strategy(decision)
                execution_result.update(result)
            
            elif strategy in ["yield_farming", "liquidity_provision"]:
                result = await self.execute_defi_strategy(decision)
                execution_result.update(result)
            
            elif strategy == "stable_vault":
                result = await self.execute_vault_strategy(decision)
                execution_result.update(result)
            
            else:
                execution_result["status"] = "error"
                execution_result["message"] = f"Unknown strategy: {strategy}"
            
            # Log the reinvestment decision
            if execution_result.get("status") == "success":
                decision_id = self.db_manager.log_agent_decision(
                    agent_name=self.name,
                    decision_type="reinvestment",
                    decision_data=decision,
                    confidence=decision.get("confidence", 0.7)
                )
                execution_result["decision_id"] = decision_id
                
                # Update last reinvestment date
                self.store_memory("last_reinvestment_date", datetime.now().isoformat())
                
                if strategy == "dca_investment":
                    self.store_memory("last_dca_date", datetime.now().isoformat())
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Reinvestment execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_dca_investment(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DCA investment strategy"""
        try:
            allocation = decision.get("target_allocation", {})
            symbols = allocation.get("symbols", ["BTC/USDT"])
            total_amount = allocation.get("amount", 100)
            
            amount_per_symbol = total_amount / len(symbols)
            
            actions_taken = []
            for symbol in symbols:
                # In a real implementation, this would place actual buy orders
                # For now, we'll simulate the investment
                
                # Get current price
                market_data = self.market_data.get_symbol_data(symbol)
                if market_data and market_data.get("prices"):
                    current_price = market_data["prices"][-1]["close"]
                    quantity = amount_per_symbol / current_price
                    
                    # Log as a simulated transaction
                    transaction_id = self.db_manager.log_transaction(
                        transaction_type="buy",
                        symbol=symbol,
                        quantity=quantity,
                        price=current_price
                    )
                    
                    actions_taken.append({
                        "action": "dca_buy",
                        "symbol": symbol,
                        "amount": amount_per_symbol,
                        "quantity": quantity,
                        "price": current_price,
                        "transaction_id": transaction_id
                    })
            
            return {
                "status": "success",
                "message": f"DCA investment completed for {len(symbols)} symbols",
                "actions_taken": actions_taken,
                "total_invested": total_amount
            }
            
        except Exception as e:
            logger.error(f"DCA execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_lending_strategy(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute lending strategy (stablecoin lending, money market)"""
        try:
            investment_amount = decision.get("investment_amount", 0)
            strategy = decision.get("strategy")
            
            # Simulate lending transaction
            lending_record = {
                "strategy": strategy,
                "amount": investment_amount,
                "expected_yield": decision.get("expected_yield", 0),
                "start_date": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store lending position
            current_lending = self.get_memory("lending_positions") or []
            current_lending.append(lending_record)
            self.store_memory("lending_positions", current_lending)
            
            return {
                "status": "success",
                "message": f"Lending strategy initiated: {strategy}",
                "actions_taken": [
                    {
                        "action": "initiate_lending",
                        "strategy": strategy,
                        "amount": investment_amount,
                        "expected_yield": decision.get("expected_yield", 0)
                    }
                ],
                "lending_record": lending_record
            }
            
        except Exception as e:
            logger.error(f"Lending strategy execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_defi_strategy(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DeFi strategy (yield farming, liquidity provision)"""
        try:
            investment_amount = decision.get("investment_amount", 0)
            strategy = decision.get("strategy")
            
            # Simulate DeFi position
            defi_record = {
                "strategy": strategy,
                "amount": investment_amount,
                "expected_yield": decision.get("expected_yield", 0),
                "risk_level": decision.get("risk_level", "medium"),
                "start_date": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store DeFi position
            current_defi = self.get_memory("defi_positions") or []
            current_defi.append(defi_record)
            self.store_memory("defi_positions", current_defi)
            
            return {
                "status": "success",
                "message": f"DeFi strategy initiated: {strategy}",
                "actions_taken": [
                    {
                        "action": "initiate_defi_position",
                        "strategy": strategy,
                        "amount": investment_amount,
                        "expected_yield": decision.get("expected_yield", 0)
                    }
                ],
                "defi_record": defi_record
            }
            
        except Exception as e:
            logger.error(f"DeFi strategy execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_vault_strategy(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vault strategy"""
        try:
            investment_amount = decision.get("investment_amount", 0)
            
            vault_record = {
                "strategy": "stable_vault",
                "amount": investment_amount,
                "expected_yield": decision.get("expected_yield", 0),
                "start_date": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store vault position
            current_vaults = self.get_memory("vault_positions") or []
            current_vaults.append(vault_record)
            self.store_memory("vault_positions", current_vaults)
            
            return {
                "status": "success",
                "message": "Vault strategy initiated",
                "actions_taken": [
                    {
                        "action": "initiate_vault_position",
                        "amount": investment_amount,
                        "expected_yield": decision.get("expected_yield", 0)
                    }
                ],
                "vault_record": vault_record
            }
            
        except Exception as e:
            logger.error(f"Vault strategy execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_reinvestment_summary(self) -> Dict[str, Any]:
        """Get summary of current reinvestment positions"""
        lending_positions = self.get_memory("lending_positions") or []
        defi_positions = self.get_memory("defi_positions") or []
        vault_positions = self.get_memory("vault_positions") or []
        
        total_lending = sum(p.get("amount", 0) for p in lending_positions if p.get("status") == "active")
        total_defi = sum(p.get("amount", 0) for p in defi_positions if p.get("status") == "active")
        total_vaults = sum(p.get("amount", 0) for p in vault_positions if p.get("status") == "active")
        
        return {
            "lending_positions": len([p for p in lending_positions if p.get("status") == "active"]),
            "defi_positions": len([p for p in defi_positions if p.get("status") == "active"]),
            "vault_positions": len([p for p in vault_positions if p.get("status") == "active"]),
            "total_lending_amount": total_lending,
            "total_defi_amount": total_defi,
            "total_vault_amount": total_vaults,
            "total_reinvested": total_lending + total_defi + total_vaults,
            "last_reinvestment": self.get_memory("last_reinvestment_date"),
            "last_dca": self.get_memory("last_dca_date")
        }

