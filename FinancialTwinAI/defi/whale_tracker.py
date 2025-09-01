"""
Whale Tracker - Advanced On-Chain Analytics
Tracks large wallet movements and behavioral patterns for LAM training
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class WhaleClassification(Enum):
    MEGA_WHALE = "mega_whale"  # >$100M
    LARGE_WHALE = "large_whale"  # $10M-$100M
    WHALE = "whale"  # $1M-$10M
    DOLPHIN = "dolphin"  # $100K-$1M

@dataclass
class WhaleTransaction:
    tx_hash: str
    from_address: str
    to_address: str
    token: str
    amount: float
    usd_value: float
    timestamp: datetime
    gas_price: float
    classification: WhaleClassification
    pattern_type: str
    confidence_score: float

class WhaleTracker:
    """Advanced whale intelligence and pattern recognition system"""
    
    def __init__(self):
        self.monitored_wallets = set()
        self.transaction_patterns = {}
        self.behavioral_models = {}
        self.market_impact_history = []
        
    async def track_whale_movements(self, addresses: List[str]) -> List[WhaleTransaction]:
        """Track movements from specified whale addresses"""
        transactions = []
        
        for address in addresses:
            # Simulate whale transaction data (in production, use web3 providers)
            whale_txs = await self._fetch_whale_transactions(address)
            
            for tx in whale_txs:
                classified_tx = await self._classify_transaction(tx)
                pattern = await self._detect_behavioral_pattern(classified_tx)
                
                whale_transaction = WhaleTransaction(
                    tx_hash=tx.get('hash', ''),
                    from_address=tx.get('from', ''),
                    to_address=tx.get('to', ''),
                    token=tx.get('token', 'ETH'),
                    amount=tx.get('amount', 0),
                    usd_value=tx.get('usd_value', 0),
                    timestamp=datetime.fromtimestamp(tx.get('timestamp', 0)),
                    gas_price=tx.get('gas_price', 0),
                    classification=self._classify_whale_size(tx.get('usd_value', 0)),
                    pattern_type=pattern,
                    confidence_score=tx.get('confidence', 0.8)
                )
                
                transactions.append(whale_transaction)
        
        return transactions
    
    async def _fetch_whale_transactions(self, address: str) -> List[Dict[str, Any]]:
        """Fetch recent transactions for whale address (mock data for now)"""
        # In production, this would query Ethereum/BSC/Polygon nodes
        mock_transactions = [
            {
                'hash': f'0x{"a" * 64}',
                'from': address,
                'to': '0x1234567890123456789012345678901234567890',
                'token': 'USDC',
                'amount': 5000000,
                'usd_value': 5000000,
                'timestamp': datetime.now().timestamp(),
                'gas_price': 25,
                'confidence': 0.95
            },
            {
                'hash': f'0x{"b" * 64}',
                'from': address,
                'to': '0x0987654321098765432109876543210987654321',
                'token': 'ETH',
                'amount': 2500,
                'usd_value': 4000000,
                'timestamp': (datetime.now() - timedelta(hours=2)).timestamp(),
                'gas_price': 30,
                'confidence': 0.87
            }
        ]
        
        return mock_transactions
    
    async def _classify_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ML classification to transaction"""
        # Simulate advanced transaction classification
        usd_value = tx.get('usd_value', 0)
        
        # Determine transaction type based on patterns
        if usd_value > 1000000:
            tx_type = 'large_transfer'
        elif tx.get('to', '').startswith('0x1111'):  # DEX router pattern
            tx_type = 'dex_swap'
        elif tx.get('to', '').startswith('0x2222'):  # Lending protocol
            tx_type = 'lending_operation'
        else:
            tx_type = 'standard_transfer'
        
        tx['classified_type'] = tx_type
        return tx
    
    async def _detect_behavioral_pattern(self, tx: Dict[str, Any]) -> str:
        """Detect whale behavioral patterns"""
        patterns = [
            'accumulation_phase',
            'distribution_phase', 
            'arbitrage_execution',
            'yield_farming',
            'market_making',
            'panic_selling',
            'coordinated_buying'
        ]
        
        # Simulate pattern detection based on transaction characteristics
        usd_value = tx.get('usd_value', 0)
        tx_type = tx.get('classified_type', '')
        
        if usd_value > 5000000 and tx_type == 'large_transfer':
            return 'distribution_phase'
        elif tx_type == 'dex_swap' and usd_value > 1000000:
            return 'arbitrage_execution'
        elif tx_type == 'lending_operation':
            return 'yield_farming'
        else:
            return 'accumulation_phase'
    
    def _classify_whale_size(self, usd_value: float) -> WhaleClassification:
        """Classify whale based on transaction size"""
        if usd_value >= 100_000_000:
            return WhaleClassification.MEGA_WHALE
        elif usd_value >= 10_000_000:
            return WhaleClassification.LARGE_WHALE
        elif usd_value >= 1_000_000:
            return WhaleClassification.WHALE
        else:
            return WhaleClassification.DOLPHIN
    
    async def analyze_market_impact(self, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Analyze potential market impact of whale movements"""
        impact_analysis = {
            'total_volume': sum(tx.usd_value for tx in transactions),
            'unique_whales': len(set(tx.from_address for tx in transactions)),
            'pattern_distribution': {},
            'time_clustering': {},
            'risk_assessment': {}
        }
        
        # Pattern distribution analysis
        for tx in transactions:
            pattern = tx.pattern_type
            impact_analysis['pattern_distribution'][pattern] = \
                impact_analysis['pattern_distribution'].get(pattern, 0) + 1
        
        # Time clustering (detect coordinated movements)
        time_windows = self._create_time_windows(transactions)
        impact_analysis['time_clustering'] = time_windows
        
        # Risk assessment
        impact_analysis['risk_assessment'] = await self._assess_market_risk(transactions)
        
        return impact_analysis
    
    def _create_time_windows(self, transactions: List[WhaleTransaction]) -> Dict[str, int]:
        """Detect coordinated movements in time windows"""
        windows = {}
        
        for tx in transactions:
            # Group by 1-hour windows
            window_key = tx.timestamp.strftime('%Y-%m-%d %H:00')
            windows[window_key] = windows.get(window_key, 0) + 1
        
        return windows
    
    async def _assess_market_risk(self, transactions: List[WhaleTransaction]) -> Dict[str, Any]:
        """Assess market risk from whale movements"""
        total_sell_volume = sum(
            tx.usd_value for tx in transactions 
            if tx.pattern_type in ['distribution_phase', 'panic_selling']
        )
        
        total_buy_volume = sum(
            tx.usd_value for tx in transactions 
            if tx.pattern_type in ['accumulation_phase', 'coordinated_buying']
        )
        
        net_flow = total_buy_volume - total_sell_volume
        
        # Risk scoring (0-100)
        volume_risk = min(total_sell_volume / 100_000_000 * 50, 50)  # Max 50 points
        coordination_risk = len([
            tx for tx in transactions 
            if tx.pattern_type == 'coordinated_buying'
        ]) * 10  # 10 points per coordinated action
        
        total_risk = min(volume_risk + coordination_risk, 100)
        
        return {
            'net_flow_usd': net_flow,
            'sell_pressure': total_sell_volume,
            'buy_pressure': total_buy_volume,
            'risk_score': total_risk,
            'risk_level': 'HIGH' if total_risk > 70 else 'MEDIUM' if total_risk > 40 else 'LOW'
        }
    
    async def generate_whale_intelligence_report(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive whale intelligence report"""
        # Get known whale addresses (in production, maintain database)
        whale_addresses = [
            '0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503',  # Binance
            '0x8894E0a0c962CB723c1976a4421c95949bE2D4E3',  # Binance 2
            '0x6262998Ced04146cA2c186D7dbE25d73A7E88a3D',  # Alameda Research
        ]
        
        # Track movements
        transactions = await self.track_whale_movements(whale_addresses)
        
        # Analyze market impact
        impact_analysis = await self.analyze_market_impact(transactions)
        
        # Generate insights
        insights = await self._generate_actionable_insights(transactions, impact_analysis)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'timeframe_hours': timeframe_hours,
            'summary': {
                'total_transactions': len(transactions),
                'total_volume_usd': impact_analysis['total_volume'],
                'unique_whales': impact_analysis['unique_whales'],
                'risk_level': impact_analysis['risk_assessment']['risk_level']
            },
            'transactions': [
                {
                    'tx_hash': tx.tx_hash[:10] + '...',
                    'token': tx.token,
                    'amount': tx.amount,
                    'usd_value': tx.usd_value,
                    'pattern': tx.pattern_type,
                    'confidence': tx.confidence_score
                }
                for tx in transactions[:10]  # Top 10 transactions
            ],
            'pattern_analysis': impact_analysis['pattern_distribution'],
            'market_impact': impact_analysis['risk_assessment'],
            'insights': insights,
            'recommendations': await self._generate_trading_recommendations(impact_analysis)
        }
        
        return report
    
    async def _generate_actionable_insights(self, transactions: List[WhaleTransaction], 
                                         impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable insights for LAM training"""
        insights = []
        
        risk_level = impact_analysis['risk_assessment']['risk_level']
        net_flow = impact_analysis['risk_assessment']['net_flow_usd']
        
        if risk_level == 'HIGH':
            insights.append("High whale activity detected - monitor for potential price volatility")
        
        if net_flow > 10_000_000:
            insights.append(f"Strong buying pressure: ${net_flow:,.0f} net inflow from whales")
        elif net_flow < -10_000_000:
            insights.append(f"Selling pressure detected: ${abs(net_flow):,.0f} net outflow from whales")
        
        # Pattern-based insights
        patterns = impact_analysis['pattern_distribution']
        if patterns.get('distribution_phase', 0) > 3:
            insights.append("Multiple whales in distribution phase - potential price weakness")
        
        if patterns.get('accumulation_phase', 0) > 3:
            insights.append("Whale accumulation pattern detected - potential bullish signal")
        
        return insights
    
    async def _generate_trading_recommendations(self, impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations based on whale activity"""
        recommendations = []
        
        risk_score = impact_analysis['risk_assessment']['risk_score']
        net_flow = impact_analysis['risk_assessment']['net_flow_usd']
        
        if risk_score > 70:
            recommendations.append("Consider reducing position sizes due to high whale activity risk")
        
        if net_flow > 50_000_000:
            recommendations.append("Strong whale accumulation - consider increasing long positions")
        elif net_flow < -50_000_000:
            recommendations.append("Whale distribution detected - consider defensive positioning")
        
        recommendations.append("Monitor whale movements for next 4-6 hours for trend confirmation")
        
        return recommendations