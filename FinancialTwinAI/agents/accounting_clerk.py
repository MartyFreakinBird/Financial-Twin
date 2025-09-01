"""
Accounting Clerk Agent
Logs expenses, categorizes cash flow, and updates equity positions
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
from decimal import Decimal

from agents.base_agent import BaseAgent
from core.llm_client import llm_client

logger = logging.getLogger(__name__)

class AccountingClerkAgent(BaseAgent):
    """Agent responsible for accounting, bookkeeping, and financial record management"""
    
    def __init__(self):
        super().__init__(
            name="AccountingClerk",
            description="Logs expenses, categorizes cash flow, and updates equity positions"
        )
        self.expense_categories = [
            "trading_fees", "software_subscriptions", "data_feeds", 
            "hardware", "education", "taxes", "other"
        ]
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial records and transactions"""
        try:
            # Get recent transactions
            recent_transactions = self.get_recent_transactions()
            
            # Get current portfolio positions
            portfolio_positions = self.db_manager.get_portfolio_positions()
            
            # Get latest balance
            latest_balance = self.db_manager.get_latest_balance()
            
            # Calculate portfolio metrics
            portfolio_metrics = self.calculate_portfolio_metrics(portfolio_positions)
            
            # Analyze cash flow
            cash_flow_analysis = self.analyze_cash_flow(recent_transactions)
            
            # Get uncategorized transactions
            uncategorized_transactions = self.get_uncategorized_transactions()
            
            analysis_data = {
                "recent_transactions": recent_transactions,
                "portfolio_positions": portfolio_positions,
                "portfolio_metrics": portfolio_metrics,
                "cash_flow_analysis": cash_flow_analysis,
                "uncategorized_transactions": uncategorized_transactions,
                "current_balance": latest_balance,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Use LLM for financial analysis
            if llm_client.is_available() and recent_transactions:
                transaction_summary = {
                    "total_transactions": len(recent_transactions),
                    "transaction_types": list(set(t.get("transaction_type") for t in recent_transactions)),
                    "total_value": sum(t.get("total_value", 0) for t in recent_transactions),
                    "portfolio_value": portfolio_metrics.get("total_value", 0)
                }
                
                llm_analysis = llm_client.analyze_financial_data(
                    transaction_summary,
                    "Accounting and financial record analysis"
                )
                if llm_analysis:
                    analysis_data["llm_insights"] = llm_analysis
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"AccountingClerk analysis failed: {e}")
            return {}
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make accounting and bookkeeping decisions"""
        try:
            uncategorized_transactions = analysis.get("uncategorized_transactions", [])
            portfolio_positions = analysis.get("portfolio_positions", [])
            cash_flow_analysis = analysis.get("cash_flow_analysis", {})
            
            decisions = []
            
            # Decision 1: Categorize transactions
            if uncategorized_transactions:
                categorization_decision = {
                    "action": "categorize_transactions",
                    "transactions_to_categorize": len(uncategorized_transactions),
                    "priority": "high"
                }
                decisions.append(categorization_decision)
            
            # Decision 2: Update portfolio valuations
            portfolio_update_needed = self.check_portfolio_update_needed(portfolio_positions)
            if portfolio_update_needed:
                valuation_decision = {
                    "action": "update_portfolio_valuations",
                    "positions_to_update": len(portfolio_positions),
                    "priority": "medium"
                }
                decisions.append(valuation_decision)
            
            # Decision 3: Generate financial reports
            report_decision = {
                "action": "generate_financial_report",
                "report_type": "weekly_summary",
                "priority": "low"
            }
            decisions.append(report_decision)
            
            # Decision 4: Check for reconciliation needs
            reconciliation_needed = self.check_reconciliation_needed(analysis)
            if reconciliation_needed:
                reconciliation_decision = {
                    "action": "reconcile_accounts",
                    "priority": "high",
                    "details": reconciliation_needed
                }
                decisions.append(reconciliation_decision)
            
            # Select primary decision based on priority
            high_priority = [d for d in decisions if d.get("priority") == "high"]
            primary_decision = high_priority[0] if high_priority else decisions[0]
            
            decision = {
                "primary_action": primary_decision["action"],
                "all_decisions": decisions,
                "reasoning": self.generate_accounting_reasoning(analysis, decisions),
                "confidence": self.calculate_accounting_confidence(analysis),
                "risk_level": "low",  # Accounting is generally low risk
                "estimated_time": self.estimate_task_time(primary_decision),
                "decision_timestamp": datetime.now().isoformat()
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"AccountingClerk decision failed: {e}")
            return {}
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute accounting actions"""
        try:
            primary_action = decision.get("primary_action")
            execution_results = []
            
            if primary_action == "categorize_transactions":
                result = await self.categorize_transactions()
                execution_results.append(result)
            
            elif primary_action == "update_portfolio_valuations":
                result = await self.update_portfolio_valuations()
                execution_results.append(result)
            
            elif primary_action == "generate_financial_report":
                result = await self.generate_financial_report()
                execution_results.append(result)
            
            elif primary_action == "reconcile_accounts":
                result = await self.reconcile_accounts(decision.get("details", {}))
                execution_results.append(result)
            
            # Log the accounting decision
            decision_id = self.db_manager.log_agent_decision(
                agent_name=self.name,
                decision_type="accounting_operation",
                decision_data=decision,
                confidence=decision.get("confidence", 0.8)
            )
            
            # Update last accounting run
            self.store_memory("last_accounting_run", datetime.now().isoformat())
            
            execution_result = {
                "status": "success",
                "decision_id": decision_id,
                "primary_action": primary_action,
                "execution_results": execution_results,
                "message": f"Completed accounting operation: {primary_action}",
                "summary": self.generate_execution_summary(execution_results)
            }
            
            return execution_result
            
        except Exception as e:
            logger.error(f"AccountingClerk execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_recent_transactions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent transactions from the database"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_uncategorized_transactions(self) -> List[Dict[str, Any]]:
        """Get transactions that haven't been categorized"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.*, ad.decision_data 
                FROM transactions t
                LEFT JOIN agent_decisions ad ON t.agent_decision_id = ad.id
                WHERE ad.decision_data IS NULL OR 
                      JSON_EXTRACT(ad.decision_data, '$.category') IS NULL
                ORDER BY t.timestamp DESC
                LIMIT 50
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def calculate_portfolio_metrics(self, portfolio_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        if not portfolio_positions:
            return {"total_value": 0, "positions_count": 0}
        
        total_value = 0
        total_cost = 0
        positions_count = len(portfolio_positions)
        
        for position in portfolio_positions:
            quantity = position.get("quantity", 0)
            current_price = position.get("current_price", position.get("average_price", 0))
            average_price = position.get("average_price", 0)
            
            position_value = quantity * current_price
            position_cost = quantity * average_price
            
            total_value += position_value
            total_cost += position_cost
        
        unrealized_pnl = total_value - total_cost
        unrealized_pnl_percent = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_percent": unrealized_pnl_percent,
            "positions_count": positions_count
        }
    
    def analyze_cash_flow(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cash flow from transactions"""
        if not transactions:
            return {"inflows": 0, "outflows": 0, "net_flow": 0}
        
        inflows = 0
        outflows = 0
        
        for transaction in transactions:
            transaction_type = transaction.get("transaction_type", "")
            total_value = transaction.get("total_value", 0)
            
            if transaction_type == "sell":
                inflows += total_value
            elif transaction_type == "buy":
                outflows += total_value
        
        net_flow = inflows - outflows
        
        return {
            "inflows": inflows,
            "outflows": outflows,
            "net_flow": net_flow,
            "transaction_count": len(transactions)
        }
    
    def check_portfolio_update_needed(self, portfolio_positions: List[Dict[str, Any]]) -> bool:
        """Check if portfolio valuations need updating"""
        if not portfolio_positions:
            return False
        
        for position in portfolio_positions:
            last_updated = position.get("last_updated")
            if last_updated:
                try:
                    last_update_time = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                    if datetime.now() - last_update_time > timedelta(hours=4):
                        return True
                except:
                    return True
            else:
                return True
        
        return False
    
    def check_reconciliation_needed(self, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if account reconciliation is needed"""
        portfolio_metrics = analysis.get("portfolio_metrics", {})
        current_balance = analysis.get("current_balance", {})
        
        if not current_balance:
            return {"reason": "No current balance record found", "severity": "high"}
        
        # Check if portfolio value matches recorded value
        calculated_portfolio_value = portfolio_metrics.get("total_value", 0)
        recorded_portfolio_value = current_balance.get("total_portfolio_value", 0)
        
        variance = abs(calculated_portfolio_value - recorded_portfolio_value)
        variance_percent = (variance / recorded_portfolio_value * 100) if recorded_portfolio_value > 0 else 0
        
        if variance_percent > 5:  # 5% variance threshold
            return {
                "reason": "Portfolio value variance exceeds threshold",
                "severity": "medium",
                "variance": variance,
                "variance_percent": variance_percent,
                "calculated_value": calculated_portfolio_value,
                "recorded_value": recorded_portfolio_value
            }
        
        return None
    
    async def categorize_transactions(self) -> Dict[str, Any]:
        """Categorize uncategorized transactions"""
        try:
            uncategorized = self.get_uncategorized_transactions()
            categorized_count = 0
            
            for transaction in uncategorized:
                category = self.auto_categorize_transaction(transaction)
                if category:
                    # Update transaction with category
                    self.update_transaction_category(transaction["id"], category)
                    categorized_count += 1
            
            return {
                "action": "categorize_transactions",
                "processed": len(uncategorized),
                "categorized": categorized_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Transaction categorization failed: {e}")
            return {"action": "categorize_transactions", "success": False, "error": str(e)}
    
    async def update_portfolio_valuations(self) -> Dict[str, Any]:
        """Update portfolio position valuations"""
        try:
            positions = self.db_manager.get_portfolio_positions()
            updated_count = 0
            
            for position in positions:
                # In a real implementation, this would fetch current market prices
                # For now, we'll simulate price updates
                symbol = position["symbol"]
                current_price = self.get_current_price(symbol)
                
                if current_price:
                    self.db_manager.update_portfolio_position(
                        symbol, position["quantity"], position["average_price"], current_price
                    )
                    updated_count += 1
            
            return {
                "action": "update_portfolio_valuations",
                "positions_checked": len(positions),
                "positions_updated": updated_count,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Portfolio valuation update failed: {e}")
            return {"action": "update_portfolio_valuations", "success": False, "error": str(e)}
    
    async def generate_financial_report(self) -> Dict[str, Any]:
        """Generate financial summary report"""
        try:
            # Get data for report
            recent_transactions = self.get_recent_transactions(30)  # Last 30 days
            portfolio_positions = self.db_manager.get_portfolio_positions()
            portfolio_metrics = self.calculate_portfolio_metrics(portfolio_positions)
            cash_flow = self.analyze_cash_flow(recent_transactions)
            
            report_data = {
                "report_date": datetime.now().isoformat(),
                "report_period": "30_days",
                "portfolio_summary": portfolio_metrics,
                "cash_flow_summary": cash_flow,
                "transaction_count": len(recent_transactions),
                "active_positions": len([p for p in portfolio_positions if p.get("quantity", 0) > 0])
            }
            
            # Store report
            self.store_memory("latest_financial_report", report_data)
            
            return {
                "action": "generate_financial_report",
                "report_generated": True,
                "report_data": report_data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Financial report generation failed: {e}")
            return {"action": "generate_financial_report", "success": False, "error": str(e)}
    
    async def reconcile_accounts(self, reconciliation_details: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile account balances"""
        try:
            # Get current data
            portfolio_positions = self.db_manager.get_portfolio_positions()
            portfolio_metrics = self.calculate_portfolio_metrics(portfolio_positions)
            
            # Update account balance with calculated values
            latest_balance = self.db_manager.get_latest_balance()
            cash_balance = latest_balance.get("cash_balance", 0) if latest_balance else 0
            
            self.db_manager.update_account_balance(
                cash_balance, portfolio_metrics["total_value"]
            )
            
            reconciliation_result = {
                "action": "reconcile_accounts",
                "reconciliation_performed": True,
                "old_portfolio_value": reconciliation_details.get("recorded_value", 0),
                "new_portfolio_value": portfolio_metrics["total_value"],
                "variance_resolved": reconciliation_details.get("variance", 0),
                "success": True
            }
            
            return reconciliation_result
            
        except Exception as e:
            logger.error(f"Account reconciliation failed: {e}")
            return {"action": "reconcile_accounts", "success": False, "error": str(e)}
    
    def auto_categorize_transaction(self, transaction: Dict[str, Any]) -> Optional[str]:
        """Automatically categorize a transaction"""
        transaction_type = transaction.get("transaction_type", "").lower()
        symbol = transaction.get("symbol", "").lower()
        
        # Simple rule-based categorization
        if transaction_type in ["buy", "sell"]:
            if any(crypto in symbol for crypto in ["btc", "eth", "crypto"]):
                return "crypto_trading"
            elif any(forex in symbol for forex in ["usd", "eur", "jpy", "gbp"]):
                return "forex_trading"
            else:
                return "stock_trading"
        
        return "other"
    
    def update_transaction_category(self, transaction_id: int, category: str):
        """Update transaction category in database"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            # Find associated agent decision and update it
            cursor.execute("""
                UPDATE agent_decisions 
                SET decision_data = JSON_SET(
                    COALESCE(decision_data, '{}'), 
                    '$.category', ?
                )
                WHERE id = (
                    SELECT agent_decision_id FROM transactions 
                    WHERE id = ?
                )
            """, (category, transaction_id))
            conn.commit()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        # This would integrate with market data provider
        # For now, return a simulated price update
        from trading.market_data import MarketDataProvider
        market_data = MarketDataProvider()
        
        data = market_data.get_symbol_data(symbol)
        if data and data.get("prices"):
            return data["prices"][-1].get("close")
        
        return None
    
    def generate_accounting_reasoning(self, analysis: Dict[str, Any], 
                                    decisions: List[Dict[str, Any]]) -> str:
        """Generate reasoning for accounting decisions"""
        uncategorized_count = len(analysis.get("uncategorized_transactions", []))
        portfolio_count = len(analysis.get("portfolio_positions", []))
        
        reasoning = f"Accounting analysis found {uncategorized_count} uncategorized transactions "
        reasoning += f"and {portfolio_count} portfolio positions to manage. "
        
        high_priority_tasks = [d for d in decisions if d.get("priority") == "high"]
        if high_priority_tasks:
            reasoning += f"High priority tasks: {len(high_priority_tasks)} requiring immediate attention. "
        
        return reasoning
    
    def calculate_accounting_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in accounting decisions"""
        base_confidence = 0.8  # Accounting is generally high confidence
        
        # Reduce confidence if there are many uncategorized transactions
        uncategorized_count = len(analysis.get("uncategorized_transactions", []))
        if uncategorized_count > 10:
            base_confidence -= 0.1
        
        # Increase confidence if we have good data
        if analysis.get("portfolio_metrics") and analysis.get("cash_flow_analysis"):
            base_confidence += 0.1
        
        return min(1.0, max(0.5, base_confidence))
    
    def estimate_task_time(self, decision: Dict[str, Any]) -> str:
        """Estimate time required for the task"""
        action = decision.get("action", "")
        
        time_estimates = {
            "categorize_transactions": "5-10 minutes",
            "update_portfolio_valuations": "3-5 minutes",
            "generate_financial_report": "2-3 minutes",
            "reconcile_accounts": "10-15 minutes"
        }
        
        return time_estimates.get(action, "5 minutes")
    
    def generate_execution_summary(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of execution results"""
        successful_actions = [r for r in execution_results if r.get("success")]
        failed_actions = [r for r in execution_results if not r.get("success")]
        
        return {
            "total_actions": len(execution_results),
            "successful_actions": len(successful_actions),
            "failed_actions": len(failed_actions),
            "success_rate": len(successful_actions) / len(execution_results) if execution_results else 0,
            "actions_completed": [r.get("action") for r in successful_actions]
        }

    def get_financial_summary(self) -> Dict[str, Any]:
        """Get comprehensive financial summary"""
        latest_report = self.get_memory("latest_financial_report")
        portfolio_positions = self.db_manager.get_portfolio_positions()
        latest_balance = self.db_manager.get_latest_balance()
        
        return {
            "latest_report": latest_report,
            "active_positions": len([p for p in portfolio_positions if p.get("quantity", 0) > 0]),
            "total_positions": len(portfolio_positions),
            "current_balance": latest_balance,
            "last_accounting_run": self.get_memory("last_accounting_run")
        }
