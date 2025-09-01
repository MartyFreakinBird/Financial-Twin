"""
DeFi Analyst Agent
Integrates on-chain analytics and DeFi protocol analysis into the Digital Twin LAM system
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
import json

from agents.base_agent import BaseAgent
from defi.whale_tracker import WhaleTracker
from defi.protocol_analyzer import ProtocolAnalyzer

logger = logging.getLogger(__name__)

class DeFiAnalystAgent(BaseAgent):
    """Agent responsible for DeFi analytics, yield optimization, and on-chain intelligence"""
    
    def __init__(self):
        super().__init__(
            name="DeFiAnalyst",
            description="Analyzes DeFi protocols, whale movements, and yield opportunities for optimal capital allocation"
        )
        
        self.whale_tracker = WhaleTracker()
        self.protocol_analyzer = ProtocolAnalyzer()
        self.market_context = {}
        self.opportunity_cache = {}
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze DeFi landscape and on-chain activity"""
        trace_id = self.start_decision_trace(context)
        
        try:
            # Get whale intelligence
            whale_report = await self.whale_tracker.generate_whale_intelligence_report(
                timeframe_hours=context.get('timeframe_hours', 24)
            )
            self.log_reasoning_step("whale_analysis", {
                "transactions_analyzed": whale_report['summary']['total_transactions'],
                "risk_level": whale_report['summary']['risk_level'],
                "total_volume": whale_report['summary']['total_volume_usd']
            })
            
            # Get DeFi protocol analysis
            defi_report = await self.protocol_analyzer.generate_defi_intelligence_report()
            self.log_reasoning_step("defi_analysis", {
                "protocols_analyzed": defi_report['summary']['total_protocols_analyzed'],
                "avg_apy": defi_report['summary']['avg_risk_adjusted_apy'],
                "arbitrage_opportunities": defi_report['summary']['arbitrage_opportunities']
            })
            
            # Analyze yield opportunities based on portfolio context
            portfolio_balance = context.get('portfolio_balance', 10000)
            min_apy = context.get('min_apy', 5.0)
            
            yield_opportunities = await self.protocol_analyzer.analyze_yield_opportunities(
                min_apy=min_apy
            )
            self.log_reasoning_step("yield_analysis", {
                "opportunities_found": len(yield_opportunities),
                "best_apy": yield_opportunities[0]['risk_adjusted_apy'] if yield_opportunities else 0
            })
            
            # Cross-reference whale activity with protocol opportunities
            correlated_opportunities = await self._correlate_whale_activity_with_protocols(
                whale_report, yield_opportunities
            )
            self.log_reasoning_step("correlation_analysis", {
                "correlated_opportunities": len(correlated_opportunities)
            })
            
            analysis = {
                "whale_intelligence": whale_report,
                "defi_protocols": defi_report,
                "yield_opportunities": yield_opportunities[:10],  # Top 10
                "correlated_opportunities": correlated_opportunities,
                "market_conditions": await self._assess_defi_market_conditions(whale_report, defi_report),
                "risk_assessment": await self._comprehensive_risk_assessment(whale_report, defi_report),
                "capital_allocation_suggestions": await self._suggest_capital_allocation(
                    portfolio_balance, yield_opportunities, whale_report
                ),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in DeFi analysis: {e}")
            return {
                "error": str(e),
                "whale_intelligence": {"summary": {"risk_level": "UNKNOWN"}},
                "defi_protocols": {"summary": {"total_protocols_analyzed": 0}},
                "timestamp": datetime.now().isoformat()
            }
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make DeFi investment and allocation decisions"""
        try:
            # Extract key metrics
            whale_risk = analysis.get('whale_intelligence', {}).get('summary', {}).get('risk_level', 'MEDIUM')
            best_yield = analysis.get('yield_opportunities', [{}])[0] if analysis.get('yield_opportunities') else {}
            market_conditions = analysis.get('market_conditions', {})
            
            decisions = []
            
            # Primary yield optimization decision
            if best_yield:
                yield_decision = await self._make_yield_decision(best_yield, whale_risk, market_conditions)
                decisions.append(yield_decision)
            
            # Risk management decision based on whale activity
            if whale_risk == 'HIGH':
                risk_decision = await self._make_risk_management_decision(analysis)
                decisions.append(risk_decision)
            
            # Arbitrage opportunity decision
            arbitrage_opps = analysis.get('defi_protocols', {}).get('arbitrage_opportunities', [])
            if arbitrage_opps:
                arbitrage_decision = await self._make_arbitrage_decision(arbitrage_opps[0])
                decisions.append(arbitrage_decision)
            
            # Capital reallocation decision
            capital_suggestions = analysis.get('capital_allocation_suggestions', {})
            if capital_suggestions:
                reallocation_decision = await self._make_reallocation_decision(capital_suggestions)
                decisions.append(reallocation_decision)
            
            # Select primary decision
            primary_decision = self._select_primary_decision(decisions)
            
            return {
                "action": primary_decision.get("action", "monitor"),
                "details": primary_decision.get("details", {}),
                "confidence": primary_decision.get("confidence", 0.7),
                "reasoning": primary_decision.get("reasoning", "DeFi analysis completed"),
                "all_decisions": decisions,
                "market_sentiment": market_conditions.get('sentiment', 'NEUTRAL'),
                "risk_level": whale_risk,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making DeFi decision: {e}")
            return {
                "action": "error",
                "details": {"error": str(e)},
                "confidence": 0.0,
                "reasoning": f"Decision error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DeFi actions"""
        action = decision.get("action", "monitor")
        details = decision.get("details", {})
        
        try:
            if action == "enter_yield_position":
                result = await self._execute_yield_position(decision)
            elif action == "reduce_risk_exposure":
                result = await self._execute_risk_reduction(decision)
            elif action == "execute_arbitrage":
                result = await self._execute_arbitrage(decision)
            elif action == "reallocate_capital":
                result = await self._execute_reallocation(decision)
            elif action == "monitor":
                result = await self._execute_monitoring(decision)
            else:
                result = {
                    "status": "completed",
                    "message": f"DeFi monitoring action: {action}",
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log action for LAM training
            await self._log_defi_action(action, decision, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing DeFi action '{action}': {e}")
            return {
                "status": "failed",
                "error": str(e),
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _correlate_whale_activity_with_protocols(self, whale_report: Dict[str, Any], 
                                                     yield_opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate whale movements with protocol opportunities"""
        correlated = []
        
        whale_insights = whale_report.get('insights', [])
        whale_patterns = whale_report.get('pattern_analysis', {})
        
        for opportunity in yield_opportunities:
            correlation_score = 0
            correlation_factors = []
            
            # Check if whales are accumulating tokens related to this protocol
            if 'accumulation_phase' in whale_patterns and whale_patterns['accumulation_phase'] > 2:
                if opportunity['protocol_type'] in ['dex', 'lending']:
                    correlation_score += 30
                    correlation_factors.append("Whale accumulation pattern supports protocol type")
            
            # Check risk correlation
            whale_risk = whale_report['summary']['risk_level']
            protocol_risk = opportunity['risk_score']
            
            if whale_risk == 'LOW' and protocol_risk < 40:
                correlation_score += 25
                correlation_factors.append("Low whale risk aligns with low protocol risk")
            elif whale_risk == 'HIGH' and protocol_risk > 60:
                correlation_score -= 20
                correlation_factors.append("High whale risk compounds protocol risk")
            
            # Volume correlation
            if opportunity.get('volume_24h', 0) > 100_000_000:
                correlation_score += 15
                correlation_factors.append("High protocol volume suggests whale interest")
            
            if correlation_score >= 25:  # Minimum correlation threshold
                correlated_opportunity = opportunity.copy()
                correlated_opportunity['correlation_score'] = correlation_score
                correlated_opportunity['correlation_factors'] = correlation_factors
                correlated.append(correlated_opportunity)
        
        return sorted(correlated, key=lambda x: x['correlation_score'], reverse=True)
    
    async def _assess_defi_market_conditions(self, whale_report: Dict[str, Any], 
                                           defi_report: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall DeFi market conditions"""
        conditions = {
            'sentiment': 'NEUTRAL',
            'liquidity_conditions': 'NORMAL',
            'volatility_level': 'MEDIUM',
            'risk_appetite': 'MEDIUM',
            'yield_environment': 'NORMAL'
        }
        
        # Analyze whale sentiment
        whale_net_flow = whale_report['market_impact']['net_flow_usd']
        whale_risk = whale_report['summary']['risk_level']
        
        if whale_net_flow > 50_000_000:
            conditions['sentiment'] = 'BULLISH'
            conditions['risk_appetite'] = 'HIGH'
        elif whale_net_flow < -50_000_000:
            conditions['sentiment'] = 'BEARISH'
            conditions['risk_appetite'] = 'LOW'
        
        # Analyze yield environment
        avg_apy = defi_report['summary']['avg_risk_adjusted_apy']
        if avg_apy > 15:
            conditions['yield_environment'] = 'HIGH_YIELD'
        elif avg_apy < 5:
            conditions['yield_environment'] = 'LOW_YIELD'
        
        # Assess liquidity from protocol TVLs
        total_tvl = defi_report['summary']['total_tvl_analyzed']
        if total_tvl > 50_000_000_000:
            conditions['liquidity_conditions'] = 'ABUNDANT'
        elif total_tvl < 10_000_000_000:
            conditions['liquidity_conditions'] = 'TIGHT'
        
        return conditions
    
    async def _comprehensive_risk_assessment(self, whale_report: Dict[str, Any], 
                                           defi_report: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment across DeFi ecosystem"""
        risks = {
            'overall_risk_score': 0,
            'whale_activity_risk': 0,
            'protocol_concentration_risk': 0,
            'liquidity_risk': 0,
            'smart_contract_risk': 0,
            'regulatory_risk': 0
        }
        
        # Whale activity risk
        whale_risk_level = whale_report['summary']['risk_level']
        whale_risk_mapping = {'LOW': 20, 'MEDIUM': 50, 'HIGH': 80}
        risks['whale_activity_risk'] = whale_risk_mapping.get(whale_risk_level, 50)
        
        # Protocol risk analysis
        high_risk_protocols = defi_report['risk_analysis']['high_risk_protocols']
        total_protocols = defi_report['summary']['total_protocols_analyzed']
        
        if total_protocols > 0:
            risks['protocol_concentration_risk'] = (high_risk_protocols / total_protocols) * 100
        
        # Liquidity risk based on TVL distribution
        risks['liquidity_risk'] = 30  # Base liquidity risk
        
        # Smart contract risk (average across protocols)
        risks['smart_contract_risk'] = 40  # Base smart contract risk
        
        # Regulatory risk (current environment)
        risks['regulatory_risk'] = 35  # Current regulatory uncertainty
        
        # Calculate overall risk score
        risk_weights = {
            'whale_activity_risk': 0.3,
            'protocol_concentration_risk': 0.25,
            'liquidity_risk': 0.2,
            'smart_contract_risk': 0.15,
            'regulatory_risk': 0.1
        }
        
        risks['overall_risk_score'] = sum(
            risks[risk_type] * weight 
            for risk_type, weight in risk_weights.items()
        )
        
        return risks
    
    async def _suggest_capital_allocation(self, portfolio_balance: float, 
                                        yield_opportunities: List[Dict[str, Any]], 
                                        whale_report: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal capital allocation across DeFi opportunities"""
        if not yield_opportunities:
            return {'total_allocation': 0, 'positions': []}
        
        # Risk-based allocation
        whale_risk = whale_report['summary']['risk_level']
        base_allocation_pct = {
            'LOW': 0.7,    # 70% allocation in low risk
            'MEDIUM': 0.5, # 50% allocation in medium risk
            'HIGH': 0.3    # 30% allocation in high risk
        }.get(whale_risk, 0.5)
        
        total_defi_allocation = portfolio_balance * base_allocation_pct
        
        # Allocate across top opportunities
        allocations = []
        remaining_capital = total_defi_allocation
        
        for i, opportunity in enumerate(yield_opportunities[:5]):  # Top 5 opportunities
            # Weight allocation by risk-adjusted APY and inverse risk
            weight = (opportunity['risk_adjusted_apy'] * (100 - opportunity['risk_score']) / 100)
            
            # Decrease allocation for later opportunities
            position_weight = weight * (0.8 ** i)
            
            allocation_amount = min(
                remaining_capital * 0.4,  # Max 40% of remaining in single position
                total_defi_allocation * 0.3  # Max 30% of total DeFi allocation
            )
            
            if allocation_amount >= 100:  # Minimum position size
                allocations.append({
                    'protocol': opportunity['protocol_name'],
                    'chain': opportunity['chain'],
                    'allocation_usd': allocation_amount,
                    'allocation_pct': allocation_amount / portfolio_balance * 100,
                    'expected_apy': opportunity['risk_adjusted_apy'],
                    'risk_score': opportunity['risk_score'],
                    'reasoning': f"Risk-adjusted APY: {opportunity['risk_adjusted_apy']:.2f}%"
                })
                
                remaining_capital -= allocation_amount
        
        return {
            'total_allocation': sum(pos['allocation_usd'] for pos in allocations),
            'total_allocation_pct': sum(pos['allocation_pct'] for pos in allocations),
            'positions': allocations,
            'cash_reserve': portfolio_balance - sum(pos['allocation_usd'] for pos in allocations),
            'whale_risk_factor': whale_risk
        }
    
    async def _make_yield_decision(self, best_yield: Dict[str, Any], whale_risk: str, 
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Make yield farming/lending decision"""
        confidence = 0.7
        
        # Adjust confidence based on conditions
        if whale_risk == 'LOW' and best_yield.get('risk_score', 100) < 40:
            confidence += 0.15
        elif whale_risk == 'HIGH':
            confidence -= 0.2
        
        if best_yield.get('risk_adjusted_apy', 0) > 15:
            confidence += 0.1
        
        return {
            'action': 'enter_yield_position',
            'priority': 3,
            'confidence': min(confidence, 0.95),
            'details': {
                'protocol': best_yield.get('protocol_name'),
                'chain': best_yield.get('chain'),
                'expected_apy': best_yield.get('risk_adjusted_apy'),
                'risk_score': best_yield.get('risk_score'),
                'position_size_pct': 20 if whale_risk == 'LOW' else 10
            },
            'reasoning': f"Best yield opportunity: {best_yield.get('risk_adjusted_apy', 0):.2f}% APY on {best_yield.get('protocol_name', 'Unknown')}"
        }
    
    async def _make_risk_management_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make risk management decision based on whale activity"""
        return {
            'action': 'reduce_risk_exposure',
            'priority': 4,
            'confidence': 0.8,
            'details': {
                'risk_reduction_pct': 30,
                'move_to_stablecoins': True,
                'protocols_to_exit': ['high_risk_protocols'],
                'reason': 'High whale activity detected'
            },
            'reasoning': 'High whale activity risk detected - reducing exposure to volatile DeFi positions'
        }
    
    async def _make_arbitrage_decision(self, arbitrage_opp: Dict[str, Any]) -> Dict[str, Any]:
        """Make arbitrage execution decision"""
        estimated_profit = arbitrage_opp.get('estimated_profit', 0)
        
        if estimated_profit > 3.0:  # Minimum 3% profit
            return {
                'action': 'execute_arbitrage',
                'priority': 2,
                'confidence': 0.7,
                'details': arbitrage_opp,
                'reasoning': f"Arbitrage opportunity with {estimated_profit:.2f}% estimated profit"
            }
        
        return {
            'action': 'monitor',
            'priority': 1,
            'confidence': 0.6,
            'details': arbitrage_opp,
            'reasoning': f"Arbitrage profit {estimated_profit:.2f}% below minimum threshold"
        }
    
    async def _make_reallocation_decision(self, capital_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Make capital reallocation decision"""
        total_allocation_pct = capital_suggestions.get('total_allocation_pct', 0)
        
        return {
            'action': 'reallocate_capital',
            'priority': 2,
            'confidence': 0.75,
            'details': capital_suggestions,
            'reasoning': f"Optimal DeFi allocation: {total_allocation_pct:.1f}% of portfolio"
        }
    
    def _select_primary_decision(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the primary decision based on priority and confidence"""
        if not decisions:
            return {
                'action': 'monitor',
                'confidence': 0.5,
                'reasoning': 'No actionable opportunities identified'
            }
        
        # Sort by priority (higher is better) then confidence
        decisions.sort(key=lambda d: (d.get('priority', 0), d.get('confidence', 0)), reverse=True)
        return decisions[0]
    
    async def _execute_yield_position(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute yield position entry (simulated)"""
        details = decision.get('details', {})
        
        logger.info(f"Entering yield position: {details.get('protocol')} on {details.get('chain')}")
        
        return {
            'status': 'completed',
            'message': f"Entered yield position on {details.get('protocol', 'Unknown')}",
            'protocol': details.get('protocol'),
            'chain': details.get('chain'),
            'expected_apy': details.get('expected_apy'),
            'position_size_pct': details.get('position_size_pct'),
            'execution_mode': 'simulated',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_risk_reduction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk reduction measures (simulated)"""
        details = decision.get('details', {})
        
        logger.info(f"Reducing risk exposure by {details.get('risk_reduction_pct', 0)}%")
        
        return {
            'status': 'completed',
            'message': 'Risk exposure reduced due to high whale activity',
            'risk_reduction_pct': details.get('risk_reduction_pct'),
            'moved_to_stablecoins': details.get('move_to_stablecoins'),
            'execution_mode': 'simulated',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_arbitrage(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute arbitrage opportunity (simulated)"""
        details = decision.get('details', {})
        
        logger.info(f"Executing arbitrage between {details.get('source_chain')} and {details.get('target_chain')}")
        
        return {
            'status': 'completed',
            'message': 'Arbitrage opportunity executed',
            'estimated_profit': details.get('estimated_profit'),
            'source_chain': details.get('source_chain'),
            'target_chain': details.get('target_chain'),
            'execution_mode': 'simulated',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_reallocation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute capital reallocation (simulated)"""
        details = decision.get('details', {})
        
        logger.info(f"Reallocating capital across {len(details.get('positions', []))} DeFi positions")
        
        return {
            'status': 'completed',
            'message': 'Capital reallocation executed',
            'total_allocation_pct': details.get('total_allocation_pct'),
            'positions_count': len(details.get('positions', [])),
            'execution_mode': 'simulated',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_monitoring(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute monitoring and tracking"""
        logger.info("Continuing DeFi market monitoring")
        
        return {
            'status': 'completed',
            'message': 'DeFi monitoring active',
            'next_analysis': (datetime.now().timestamp() + 3600),  # Next analysis in 1 hour
            'execution_mode': 'active',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _log_defi_action(self, action: str, decision: Dict[str, Any], 
                             result: Dict[str, Any]) -> None:
        """Log DeFi actions for LAM training"""
        log_entry = {
            'agent': self.name,
            'action': action,
            'decision_context': {
                'confidence': decision.get('confidence'),
                'reasoning': decision.get('reasoning'),
                'details': decision.get('details', {})
            },
            'execution_result': {
                'status': result.get('status'),
                'message': result.get('message')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"DeFi action logged for LAM training: {json.dumps(log_entry, indent=2)}")