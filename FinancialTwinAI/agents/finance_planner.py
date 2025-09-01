"""
Finance Planner Agent
Creates weekly financial plans based on balance, risk model, and upcoming events
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from core.llm_client import llm_client
from trading.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

class FinancePlannerAgent(BaseAgent):
    """Agent responsible for financial planning and allocation strategies"""
    
    def __init__(self):
        super().__init__(
            name="FinancePlanner",
            description="Creates weekly financial plans based on balance, risk model, and upcoming events"
        )
        self.market_data = MarketDataProvider()
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current financial situation"""
        try:
            # Get current portfolio state
            portfolio_positions = self.db_manager.get_portfolio_positions()
            latest_balance = self.db_manager.get_latest_balance()
            
            # Get market overview
            market_overview = self.market_data.get_market_overview()
            
            # Retrieve planning parameters from memory
            risk_tolerance = self.get_memory("risk_tolerance") or "medium"
            investment_goals = self.get_memory("investment_goals") or ["capital_growth"]
            time_horizon = self.get_memory("time_horizon") or "medium_term"
            
            analysis_data = {
                "current_portfolio": portfolio_positions,
                "account_balance": latest_balance,
                "market_conditions": market_overview,
                "risk_tolerance": risk_tolerance,
                "investment_goals": investment_goals,
                "time_horizon": time_horizon,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Use LLM for advanced analysis
            if llm_client.is_available():
                llm_analysis = llm_client.analyze_financial_data(
                    analysis_data,
                    context="Weekly financial planning analysis"
                )
                if llm_analysis:
                    analysis_data.update(llm_analysis)
            
            # Calculate basic metrics
            total_portfolio_value = sum(
                pos.get("quantity", 0) * pos.get("current_price", pos.get("average_price", 0))
                for pos in portfolio_positions
            )
            
            cash_balance = latest_balance.get("cash_balance", 0) if latest_balance else 0
            total_value = total_portfolio_value + cash_balance
            
            analysis_data.update({
                "total_portfolio_value": total_portfolio_value,
                "cash_balance": cash_balance,
                "total_value": total_value,
                "cash_percentage": (cash_balance / total_value * 100) if total_value > 0 else 0,
                "portfolio_percentage": (total_portfolio_value / total_value * 100) if total_value > 0 else 0
            })
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"FinancePlanner analysis failed: {e}")
            return {}
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make financial planning decisions"""
        try:
            # Extract key metrics
            total_value = analysis.get("total_value", 0)
            cash_percentage = analysis.get("cash_percentage", 0)
            risk_tolerance = analysis.get("risk_tolerance", "medium")
            market_conditions = analysis.get("market_conditions", {})
            
            # Define allocation targets based on risk tolerance
            allocation_targets = self.get_allocation_targets(risk_tolerance, market_conditions)
            
            # Generate plan using LLM if available
            financial_plan = None
            if llm_client.is_available():
                financial_plan = llm_client.generate_financial_plan(
                    analysis,
                    analysis.get("investment_goals", ["capital_growth"]),
                    analysis.get("time_horizon", "medium_term")
                )
            
            decision = {
                "action": "create_weekly_plan",
                "allocation_targets": allocation_targets,
                "current_allocation": {
                    "cash": cash_percentage,
                    "investments": analysis.get("portfolio_percentage", 0)
                },
                "rebalancing_needed": abs(cash_percentage - allocation_targets["cash"]) > 10,
                "reasoning": self.generate_planning_reasoning(analysis, allocation_targets),
                "confidence": self.calculate_planning_confidence(analysis, market_conditions),
                "risk_level": self.assess_plan_risk(allocation_targets, market_conditions),
                "llm_plan": financial_plan,
                "decision_timestamp": datetime.now().isoformat()
            }
            
            # Add specific action items
            decision["action_items"] = self.generate_action_items(
                analysis, allocation_targets, financial_plan
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"FinancePlanner decision failed: {e}")
            return {}
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the financial planning action"""
        try:
            # Store the weekly plan
            plan_data = {
                "allocation_targets": decision["allocation_targets"],
                "action_items": decision["action_items"],
                "created_date": datetime.now().isoformat(),
                "risk_level": decision["risk_level"],
                "confidence": decision["confidence"]
            }
            
            self.store_memory("current_weekly_plan", plan_data)
            self.store_memory("last_plan_date", datetime.now().isoformat())
            
            # Update risk parameters if needed
            if decision.get("risk_level"):
                self.store_memory("current_risk_level", decision["risk_level"])
            
            # Log the decision for other agents
            decision_id = self.db_manager.log_agent_decision(
                agent_name=self.name,
                decision_type="weekly_financial_plan",
                decision_data=decision,
                confidence=decision.get("confidence", 0.5)
            )
            
            execution_result = {
                "status": "success",
                "decision_id": decision_id,
                "plan_stored": True,
                "message": "Weekly financial plan created and stored",
                "plan_summary": {
                    "target_allocations": decision["allocation_targets"],
                    "rebalancing_needed": decision.get("rebalancing_needed", False),
                    "action_items_count": len(decision.get("action_items", []))
                }
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"FinancePlanner execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_allocation_targets(self, risk_tolerance: str, 
                              market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Determine target allocations based on risk tolerance and market conditions"""
        base_allocations = {
            "conservative": {"cash": 40, "bonds": 40, "stocks": 15, "crypto": 5},
            "medium": {"cash": 20, "bonds": 30, "stocks": 40, "crypto": 10},
            "aggressive": {"cash": 10, "bonds": 15, "stocks": 55, "crypto": 20}
        }
        
        allocation = base_allocations.get(risk_tolerance, base_allocations["medium"])
        
        # Adjust based on market conditions
        market_volatility = market_conditions.get("volatility", "medium")
        if market_volatility == "high":
            # Increase cash allocation during high volatility
            allocation["cash"] += 10
            allocation["crypto"] -= 5
            allocation["stocks"] -= 5
        elif market_volatility == "low":
            # Decrease cash allocation during low volatility
            allocation["cash"] -= 5
            allocation["stocks"] += 3
            allocation["crypto"] += 2
        
        return allocation
    
    def generate_planning_reasoning(self, analysis: Dict[str, Any], 
                                   allocation_targets: Dict[str, float]) -> str:
        """Generate reasoning for the financial plan"""
        current_cash = analysis.get("cash_percentage", 0)
        target_cash = allocation_targets.get("cash", 20)
        
        reasoning = f"Current cash allocation: {current_cash:.1f}%, target: {target_cash:.1f}%. "
        
        if abs(current_cash - target_cash) > 10:
            if current_cash > target_cash:
                reasoning += "Excess cash should be deployed to investments. "
            else:
                reasoning += "Need to increase cash reserves for stability. "
        
        market_sentiment = analysis.get("market_conditions", {}).get("sentiment", "neutral")
        reasoning += f"Market sentiment: {market_sentiment}. "
        
        if market_sentiment == "bearish":
            reasoning += "Maintaining higher cash allocation due to market uncertainty."
        elif market_sentiment == "bullish":
            reasoning += "Favorable market conditions support increased investment exposure."
        
        return reasoning
    
    def calculate_planning_confidence(self, analysis: Dict[str, Any],
                                     market_conditions: Dict[str, Any]) -> float:
        """Calculate confidence in the financial plan"""
        base_confidence = 0.7
        
        # Increase confidence if we have complete data
        if analysis.get("total_value", 0) > 0:
            base_confidence += 0.1
        
        # Adjust based on market volatility
        volatility = market_conditions.get("volatility", "medium")
        if volatility == "low":
            base_confidence += 0.1
        elif volatility == "high":
            base_confidence -= 0.1
        
        # Consider data quality
        if analysis.get("market_conditions") and analysis.get("current_portfolio"):
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def assess_plan_risk(self, allocation_targets: Dict[str, float],
                        market_conditions: Dict[str, Any]) -> str:
        """Assess the risk level of the financial plan"""
        crypto_allocation = allocation_targets.get("crypto", 0)
        stocks_allocation = allocation_targets.get("stocks", 0)
        cash_allocation = allocation_targets.get("cash", 0)
        
        risk_score = crypto_allocation * 2 + stocks_allocation * 1 + cash_allocation * -0.5
        
        market_volatility = market_conditions.get("volatility", "medium")
        if market_volatility == "high":
            risk_score += 10
        elif market_volatility == "low":
            risk_score -= 5
        
        if risk_score > 70:
            return "high"
        elif risk_score > 40:
            return "medium"
        else:
            return "low"
    
    def generate_action_items(self, analysis: Dict[str, Any],
                             allocation_targets: Dict[str, float],
                             llm_plan: Optional[Dict[str, Any]]) -> List[str]:
        """Generate specific action items for the week"""
        action_items = []
        
        # Rebalancing actions
        current_cash = analysis.get("cash_percentage", 0)
        target_cash = allocation_targets.get("cash", 20)
        
        if abs(current_cash - target_cash) > 10:
            if current_cash > target_cash:
                action_items.append(f"Deploy excess cash: reduce from {current_cash:.1f}% to {target_cash:.1f}%")
            else:
                action_items.append(f"Build cash reserves: increase from {current_cash:.1f}% to {target_cash:.1f}%")
        
        # Market-based actions
        market_conditions = analysis.get("market_conditions", {})
        if market_conditions.get("sentiment") == "bullish":
            action_items.append("Consider increasing equity exposure in favorable market")
        elif market_conditions.get("sentiment") == "bearish":
            action_items.append("Review defensive positions and stop-loss orders")
        
        # LLM-generated actions
        if llm_plan and llm_plan.get("action_items"):
            action_items.extend(llm_plan["action_items"][:3])  # Limit to top 3
        
        return action_items

    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current weekly plan"""
        return self.get_memory("current_weekly_plan")
