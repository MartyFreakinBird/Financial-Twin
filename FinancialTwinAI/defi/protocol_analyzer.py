"""
DeFi Protocol Analyzer
Analyzes yield opportunities, risks, and protocol health across DeFi ecosystems
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ProtocolType(Enum):
    LENDING = "lending"
    DEX = "dex"
    YIELD_FARMING = "yield_farming"
    LIQUID_STAKING = "liquid_staking"
    DERIVATIVES = "derivatives"
    BRIDGE = "bridge"

@dataclass
class ProtocolMetrics:
    name: str
    protocol_type: ProtocolType
    tvl_usd: float
    apy_range: tuple
    risk_score: int  # 0-100
    volume_24h: float
    token_address: str
    chain: str
    audit_status: str
    last_updated: datetime

class ProtocolAnalyzer:
    """Advanced DeFi protocol analysis and yield optimization"""
    
    def __init__(self):
        self.protocol_cache = {}
        self.risk_models = {}
        self.yield_history = {}
        
    async def analyze_yield_opportunities(self, chains: List[str] = None, 
                                        min_apy: float = 5.0) -> List[Dict[str, Any]]:
        """Analyze yield farming and lending opportunities across chains"""
        if chains is None:
            chains = ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism']
        
        opportunities = []
        
        for chain in chains:
            chain_protocols = await self._fetch_chain_protocols(chain)
            
            for protocol in chain_protocols:
                if protocol.apy_range[0] >= min_apy:
                    opportunity = await self._analyze_protocol_opportunity(protocol)
                    opportunities.append(opportunity)
        
        # Sort by risk-adjusted APY
        opportunities.sort(key=lambda x: x['risk_adjusted_apy'], reverse=True)
        
        return opportunities
    
    async def _fetch_chain_protocols(self, chain: str) -> List[ProtocolMetrics]:
        """Fetch protocol data for specific chain (mock data for now)"""
        # In production, integrate with DeFiLlama, DeBank, etc.
        mock_protocols = {
            'ethereum': [
                ProtocolMetrics(
                    name='Compound',
                    protocol_type=ProtocolType.LENDING,
                    tvl_usd=8_500_000_000,
                    apy_range=(3.2, 8.5),
                    risk_score=25,
                    volume_24h=150_000_000,
                    token_address='0xc00e94cb662c3520282e6f5717214004a7f26888',
                    chain='ethereum',
                    audit_status='audited',
                    last_updated=datetime.now()
                ),
                ProtocolMetrics(
                    name='Uniswap V3',
                    protocol_type=ProtocolType.DEX,
                    tvl_usd=4_200_000_000,
                    apy_range=(5.0, 25.0),
                    risk_score=35,
                    volume_24h=1_200_000_000,
                    token_address='0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
                    chain='ethereum',
                    audit_status='audited',
                    last_updated=datetime.now()
                )
            ],
            'bsc': [
                ProtocolMetrics(
                    name='PancakeSwap',
                    protocol_type=ProtocolType.DEX,
                    tvl_usd=2_800_000_000,
                    apy_range=(8.0, 45.0),
                    risk_score=40,
                    volume_24h=800_000_000,
                    token_address='0x0e09fabb73bd3ade0a17ecc321fd13a19e81ce82',
                    chain='bsc',
                    audit_status='audited',
                    last_updated=datetime.now()
                )
            ]
        }
        
        return mock_protocols.get(chain, [])
    
    async def _analyze_protocol_opportunity(self, protocol: ProtocolMetrics) -> Dict[str, Any]:
        """Analyze individual protocol opportunity"""
        # Calculate risk-adjusted APY
        max_apy = protocol.apy_range[1]
        risk_adjustment = (100 - protocol.risk_score) / 100
        risk_adjusted_apy = max_apy * risk_adjustment
        
        # Calculate impermanent loss risk for DEX protocols
        il_risk = 0
        if protocol.protocol_type == ProtocolType.DEX:
            il_risk = await self._calculate_impermanent_loss_risk(protocol)
        
        # Get protocol health metrics
        health_score = await self._calculate_protocol_health(protocol)
        
        opportunity = {
            'protocol_name': protocol.name,
            'chain': protocol.chain,
            'protocol_type': protocol.protocol_type.value,
            'max_apy': max_apy,
            'risk_score': protocol.risk_score,
            'risk_adjusted_apy': risk_adjusted_apy,
            'tvl_usd': protocol.tvl_usd,
            'volume_24h': protocol.volume_24h,
            'audit_status': protocol.audit_status,
            'health_score': health_score,
            'impermanent_loss_risk': il_risk,
            'recommendation': await self._generate_protocol_recommendation(protocol, risk_adjusted_apy),
            'entry_conditions': await self._determine_entry_conditions(protocol),
            'exit_strategy': await self._suggest_exit_strategy(protocol)
        }
        
        return opportunity
    
    async def _calculate_impermanent_loss_risk(self, protocol: ProtocolMetrics) -> float:
        """Calculate impermanent loss risk for DEX protocols"""
        if protocol.protocol_type != ProtocolType.DEX:
            return 0.0
        
        # Simulate IL calculation based on historical volatility
        base_il_risk = 5.0  # Base 5% IL risk
        
        # Adjust based on protocol volume and TVL ratio
        volume_tvl_ratio = protocol.volume_24h / protocol.tvl_usd if protocol.tvl_usd > 0 else 0
        
        if volume_tvl_ratio > 0.3:  # High volume relative to TVL
            il_risk_adjustment = 1.5
        elif volume_tvl_ratio > 0.1:
            il_risk_adjustment = 1.2
        else:
            il_risk_adjustment = 1.0
        
        return base_il_risk * il_risk_adjustment
    
    async def _calculate_protocol_health(self, protocol: ProtocolMetrics) -> int:
        """Calculate overall protocol health score (0-100)"""
        health_factors = {
            'tvl_score': min(protocol.tvl_usd / 1_000_000_000 * 30, 30),  # Max 30 points
            'volume_score': min(protocol.volume_24h / 100_000_000 * 25, 25),  # Max 25 points
            'audit_score': 25 if protocol.audit_status == 'audited' else 0,  # 25 points
            'age_score': 20  # Assume mature protocol, 20 points
        }
        
        total_score = sum(health_factors.values())
        return min(int(total_score), 100)
    
    async def _generate_protocol_recommendation(self, protocol: ProtocolMetrics, 
                                             risk_adjusted_apy: float) -> str:
        """Generate recommendation for protocol"""
        if risk_adjusted_apy > 15 and protocol.risk_score < 40:
            return "STRONG_BUY"
        elif risk_adjusted_apy > 8 and protocol.risk_score < 60:
            return "BUY"
        elif risk_adjusted_apy > 5:
            return "HOLD"
        else:
            return "AVOID"
    
    async def _determine_entry_conditions(self, protocol: ProtocolMetrics) -> List[str]:
        """Determine optimal entry conditions"""
        conditions = []
        
        if protocol.protocol_type == ProtocolType.LENDING:
            conditions.extend([
                f"Enter when utilization rate < 80%",
                f"Monitor for interest rate stability",
                f"Check protocol reserve levels"
            ])
        elif protocol.protocol_type == ProtocolType.DEX:
            conditions.extend([
                f"Enter during low volatility periods",
                f"Monitor token correlation for IL risk",
                f"Check liquidity depth and fees"
            ])
        elif protocol.protocol_type == ProtocolType.YIELD_FARMING:
            conditions.extend([
                f"Verify reward token sustainability",
                f"Check for potential reward token dumps",
                f"Monitor emission schedule"
            ])
        
        return conditions
    
    async def _suggest_exit_strategy(self, protocol: ProtocolMetrics) -> List[str]:
        """Suggest exit strategies"""
        strategies = []
        
        base_strategies = [
            f"Set stop-loss if APY drops below {protocol.apy_range[0] * 0.7:.1f}%",
            f"Monitor protocol TVL - exit if drops >30%",
            f"Watch for governance changes that increase risk"
        ]
        
        if protocol.protocol_type == ProtocolType.DEX:
            base_strategies.append("Exit if impermanent loss exceeds 10%")
        
        strategies.extend(base_strategies)
        return strategies
    
    async def analyze_cross_chain_arbitrage(self) -> List[Dict[str, Any]]:
        """Analyze arbitrage opportunities across chains"""
        arbitrage_opportunities = []
        
        # Get protocol data from multiple chains
        chains = ['ethereum', 'bsc', 'polygon', 'arbitrum']
        all_protocols = {}
        
        for chain in chains:
            all_protocols[chain] = await self._fetch_chain_protocols(chain)
        
        # Find arbitrage opportunities
        for base_chain in chains:
            for base_protocol in all_protocols[base_chain]:
                for target_chain in chains:
                    if target_chain == base_chain:
                        continue
                    
                    # Find similar protocols on other chains
                    similar_protocols = [
                        p for p in all_protocols[target_chain]
                        if p.protocol_type == base_protocol.protocol_type
                    ]
                    
                    for target_protocol in similar_protocols:
                        apy_diff = target_protocol.apy_range[1] - base_protocol.apy_range[1]
                        
                        if apy_diff > 2.0:  # Minimum 2% APY difference
                            opportunity = {
                                'source_chain': base_chain,
                                'target_chain': target_chain,
                                'source_protocol': base_protocol.name,
                                'target_protocol': target_protocol.name,
                                'apy_difference': apy_diff,
                                'estimated_profit': apy_diff - 1.0,  # Minus bridge costs
                                'risk_assessment': await self._assess_arbitrage_risk(
                                    base_protocol, target_protocol
                                ),
                                'bridge_cost_estimate': 1.0,  # Estimated bridge cost
                                'execution_complexity': 'MEDIUM'
                            }
                            
                            if opportunity['estimated_profit'] > 0:
                                arbitrage_opportunities.append(opportunity)
        
        return sorted(arbitrage_opportunities, key=lambda x: x['estimated_profit'], reverse=True)
    
    async def _assess_arbitrage_risk(self, source_protocol: ProtocolMetrics, 
                                   target_protocol: ProtocolMetrics) -> Dict[str, Any]:
        """Assess risks of cross-chain arbitrage"""
        risks = {
            'bridge_risk': 'MEDIUM',  # Cross-chain bridge risks
            'protocol_risk_diff': abs(target_protocol.risk_score - source_protocol.risk_score),
            'liquidity_risk': 'LOW' if min(source_protocol.tvl_usd, target_protocol.tvl_usd) > 100_000_000 else 'MEDIUM',
            'execution_risk': 'MEDIUM',  # Multi-step transaction risk
            'overall_risk': 'MEDIUM'
        }
        
        return risks
    
    async def generate_defi_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive DeFi intelligence report"""
        # Get yield opportunities
        yield_opportunities = await self.analyze_yield_opportunities(min_apy=5.0)
        
        # Get arbitrage opportunities
        arbitrage_opportunities = await self.analyze_cross_chain_arbitrage()
        
        # Calculate market statistics
        total_tvl = sum(opp['tvl_usd'] for opp in yield_opportunities)
        avg_risk_adjusted_apy = sum(opp['risk_adjusted_apy'] for opp in yield_opportunities) / len(yield_opportunities) if yield_opportunities else 0
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_protocols_analyzed': len(yield_opportunities),
                'total_tvl_analyzed': total_tvl,
                'avg_risk_adjusted_apy': avg_risk_adjusted_apy,
                'arbitrage_opportunities': len(arbitrage_opportunities),
                'best_yield_opportunity': yield_opportunities[0] if yield_opportunities else None,
                'best_arbitrage_opportunity': arbitrage_opportunities[0] if arbitrage_opportunities else None
            },
            'top_yield_opportunities': yield_opportunities[:10],
            'arbitrage_opportunities': arbitrage_opportunities[:5],
            'risk_analysis': {
                'high_risk_protocols': len([opp for opp in yield_opportunities if opp['risk_score'] > 70]),
                'medium_risk_protocols': len([opp for opp in yield_opportunities if 40 <= opp['risk_score'] <= 70]),
                'low_risk_protocols': len([opp for opp in yield_opportunities if opp['risk_score'] < 40])
            },
            'chain_analysis': await self._analyze_chain_performance(),
            'recommendations': await self._generate_defi_recommendations(yield_opportunities, arbitrage_opportunities)
        }
        
        return report
    
    async def _analyze_chain_performance(self) -> Dict[str, Any]:
        """Analyze performance across different chains"""
        chains = ['ethereum', 'bsc', 'polygon', 'arbitrum']
        chain_performance = {}
        
        for chain in chains:
            protocols = await self._fetch_chain_protocols(chain)
            
            if protocols:
                avg_apy = sum(p.apy_range[1] for p in protocols) / len(protocols)
                avg_risk = sum(p.risk_score for p in protocols) / len(protocols)
                total_tvl = sum(p.tvl_usd for p in protocols)
                
                chain_performance[chain] = {
                    'avg_apy': avg_apy,
                    'avg_risk_score': avg_risk,
                    'total_tvl': total_tvl,
                    'protocol_count': len(protocols),
                    'risk_adjusted_apy': avg_apy * ((100 - avg_risk) / 100)
                }
        
        return chain_performance
    
    async def _generate_defi_recommendations(self, yield_opportunities: List[Dict[str, Any]], 
                                           arbitrage_opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable DeFi recommendations"""
        recommendations = []
        
        if yield_opportunities:
            best_yield = yield_opportunities[0]
            recommendations.append(
                f"Best yield opportunity: {best_yield['protocol_name']} on {best_yield['chain']} "
                f"with {best_yield['risk_adjusted_apy']:.2f}% risk-adjusted APY"
            )
        
        if arbitrage_opportunities:
            best_arb = arbitrage_opportunities[0]
            recommendations.append(
                f"Best arbitrage: {best_arb['estimated_profit']:.2f}% profit potential "
                f"between {best_arb['source_chain']} and {best_arb['target_chain']}"
            )
        
        # Risk-based recommendations
        high_yield_low_risk = [
            opp for opp in yield_opportunities 
            if opp['risk_adjusted_apy'] > 10 and opp['risk_score'] < 50
        ]
        
        if high_yield_low_risk:
            recommendations.append(
                f"Found {len(high_yield_low_risk)} high-yield, low-risk opportunities"
            )
        
        recommendations.append("Monitor protocol health scores and exit if they drop below 60")
        recommendations.append("Diversify across chains to reduce concentration risk")
        
        return recommendations