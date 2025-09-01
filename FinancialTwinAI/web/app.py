"""
Flask Web Application
Web interface for monitoring and controlling the Digital Twin LAM system
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import logging
from datetime import datetime
from typing import Dict, Any

from core.database import DatabaseManager
from core.orchestrator import AgentOrchestrator
from core.trace_logger import trace_logger
from trading.mock_broker import MockBroker
from trading.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'digital-twin-lam-secret-key'
    
    # Initialize components
    db_manager = DatabaseManager()
    market_data = MarketDataProvider()
    broker = MockBroker()
    
    @app.route('/')
    def dashboard():
        """Main dashboard"""
        try:
            # Get system status
            latest_balance = db_manager.get_latest_balance()
            portfolio_positions = db_manager.get_portfolio_positions()
            recent_transactions = get_recent_transactions(5)
            
            # Get market overview
            market_overview = market_data.get_market_overview()
            
            # Get broker status
            broker_status = broker.get_broker_status()
            
            # Get recent traces
            recent_traces = trace_logger.get_trace_statistics()
            
            dashboard_data = {
                "account_balance": latest_balance,
                "portfolio_positions": portfolio_positions,
                "recent_transactions": recent_transactions,
                "market_overview": market_overview,
                "broker_status": broker_status,
                "trace_statistics": recent_traces,
                "system_status": "active",
                "last_updated": datetime.now().isoformat()
            }
            
            return render_template('dashboard.html', data=dashboard_data)
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            return render_template('dashboard.html', error=str(e))
    
    @app.route('/agents')
    def agents():
        """Agents management page"""
        try:
            # This would normally get real orchestrator status
            # For now, return mock data structure
            agent_data = {
                "agents": [
                    {
                        "name": "FinancePlanner",
                        "status": "active",
                        "last_run": datetime.now().isoformat(),
                        "decisions_count": 5,
                        "success_rate": 85.5
                    },
                    {
                        "name": "TradingStrategist",
                        "status": "active",
                        "last_run": datetime.now().isoformat(),
                        "decisions_count": 12,
                        "success_rate": 72.3
                    },
                    {
                        "name": "AccountingClerk",
                        "status": "active",
                        "last_run": datetime.now().isoformat(),
                        "decisions_count": 8,
                        "success_rate": 95.2
                    },
                    {
                        "name": "AutomationPilot",
                        "status": "active",
                        "last_run": datetime.now().isoformat(),
                        "decisions_count": 3,
                        "success_rate": 67.8
                    },
                    {
                        "name": "Reinvestor",
                        "status": "active",
                        "last_run": datetime.now().isoformat(),
                        "decisions_count": 6,
                        "success_rate": 78.9
                    }
                ],
                "total_agents": 5,
                "active_agents": 5,
                "total_decisions": 34,
                "overall_success_rate": 79.9
            }
            
            return render_template('agents.html', data=agent_data)
            
        except Exception as e:
            logger.error(f"Agents page error: {e}")
            return render_template('agents.html', error=str(e))
    
    @app.route('/traces')
    def traces():
        """Decision traces page"""
        try:
            # Get trace data
            trace_stats = trace_logger.get_trace_statistics()
            recent_traces = db_manager.get_decision_traces(limit=20)
            
            # Format traces for display
            formatted_traces = []
            for trace in recent_traces:
                formatted_trace = {
                    "trace_id": trace["trace_id"],
                    "agent_name": trace["agent_name"],
                    "timestamp": trace["timestamp"],
                    "action_taken": trace["action_taken"],
                    "performance_score": trace.get("performance_score", 0),
                    "has_outcome": trace.get("outcome_data") is not None
                }
                formatted_traces.append(formatted_trace)
            
            trace_data = {
                "statistics": trace_stats,
                "recent_traces": formatted_traces,
                "total_traces": len(recent_traces)
            }
            
            return render_template('traces.html', data=trace_data)
            
        except Exception as e:
            logger.error(f"Traces page error: {e}")
            return render_template('traces.html', error=str(e))
    
    @app.route('/defi')
    def defi():
        """DeFi analytics page"""
        try:
            # Mock DeFi data for now
            defi_data = {
                "total_tvl_analyzed": 15_000_000_000,
                "avg_risk_adjusted_apy": 8.5,
                "whale_transactions": 24,
                "whale_risk_level": "MEDIUM"
            }
            
            return render_template('defi.html', data=defi_data)
            
        except Exception as e:
            logger.error(f"DeFi page error: {e}")
            return render_template('defi.html', error=str(e))
    
    # API Routes
    @app.route('/api/status')
    def api_status():
        """System status API"""
        try:
            status = {
                "system": "active",
                "database": "connected",
                "agents": 5,
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/portfolio')
    def api_portfolio():
        """Portfolio status API"""
        try:
            latest_balance = db_manager.get_latest_balance()
            positions = db_manager.get_portfolio_positions()
            broker_status = broker.get_account_info()
            
            portfolio_data = {
                "balance": latest_balance,
                "positions": positions,
                "broker_info": broker_status,
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(portfolio_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/market')
    def api_market():
        """Market data API"""
        try:
            market_overview = market_data.get_market_overview()
            
            # Get data for key symbols
            key_symbols = ["BTC/USDT", "ETH/USDT", "USDJPY", "EURUSD"]
            symbol_data = {}
            
            for symbol in key_symbols:
                data = market_data.get_symbol_data(symbol)
                if data:
                    symbol_data[symbol] = {
                        "price": data["current_price"],
                        "change": data["day_change_percent"],
                        "volume": data["volume"]
                    }
            
            market_response = {
                "overview": market_overview,
                "symbols": symbol_data,
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(market_response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/traces')
    def api_traces():
        """Decision traces API"""
        try:
            limit = request.args.get('limit', 10, type=int)
            agent_name = request.args.get('agent')
            
            traces = db_manager.get_decision_traces(agent_name=agent_name, limit=limit)
            trace_stats = trace_logger.get_trace_statistics()
            
            return jsonify({
                "traces": traces,
                "statistics": trace_stats,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/agent/<agent_name>/status')
    def api_agent_status(agent_name):
        """Individual agent status API"""
        try:
            # Get agent execution history
            recent_decisions = []
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM agent_decisions 
                    WHERE agent_name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """, (agent_name,))
                recent_decisions = [dict(row) for row in cursor.fetchall()]
            
            agent_data = {
                "name": agent_name,
                "status": "active",
                "recent_decisions": recent_decisions,
                "decision_count": len(recent_decisions),
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(agent_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/execute/<agent_name>', methods=['POST'])
    def api_execute_agent(agent_name):
        """Execute agent cycle API"""
        try:
            # This would normally trigger agent execution
            # For now, return success response
            result = {
                "agent": agent_name,
                "status": "executed",
                "message": f"Agent {agent_name} execution triggered",
                "timestamp": datetime.now().isoformat()
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/transactions')
    def api_transactions():
        """Recent transactions API"""
        try:
            limit = request.args.get('limit', 20, type=int)
            transactions = get_recent_transactions(limit)
            
            return jsonify({
                "transactions": transactions,
                "count": len(transactions),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # DeFi API Endpoints
    @app.route('/api/defi/whale-analysis', methods=['POST'])
    def api_whale_analysis():
        """Whale analysis API"""
        try:
            # Mock whale analysis data
            whale_data = {
                "summary": {
                    "total_transactions": 15,
                    "total_volume_usd": 125_000_000,
                    "unique_whales": 8,
                    "risk_level": "MEDIUM"
                },
                "transactions": [
                    {
                        "tx_hash": "0xabcd1234...",
                        "token": "USDC",
                        "usd_value": 5_000_000,
                        "pattern": "accumulation_phase",
                        "confidence": 0.89,
                        "risk_level": "MEDIUM"
                    },
                    {
                        "tx_hash": "0xefgh5678...", 
                        "token": "ETH",
                        "usd_value": 8_500_000,
                        "pattern": "distribution_phase",
                        "confidence": 0.92,
                        "risk_level": "HIGH"
                    }
                ],
                "market_impact": {
                    "net_flow_usd": 15_000_000,
                    "sell_pressure": 45_000_000,
                    "buy_pressure": 60_000_000,
                    "risk_score": 65
                },
                "insights": [
                    "Strong buying pressure detected from whale wallets",
                    "Accumulation patterns suggest potential price movement",
                    "Monitor next 4-6 hours for trend confirmation"
                ]
            }
            
            return jsonify(whale_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/defi/protocol-scan', methods=['POST'])
    def api_protocol_scan():
        """Protocol scan API"""
        try:
            request_data = request.get_json() or {}
            min_apy = request_data.get('min_apy', 5.0)
            
            # Mock protocol data
            protocol_data = {
                "summary": {
                    "total_protocols_analyzed": 12,
                    "avg_risk_adjusted_apy": 8.7,
                    "arbitrage_opportunities": 3
                },
                "top_yield_opportunities": [
                    {
                        "protocol_name": "Compound",
                        "chain": "ethereum",
                        "protocol_type": "lending",
                        "risk_adjusted_apy": 12.5,
                        "risk_score": 25,
                        "tvl_usd": 8_500_000_000,
                        "recommendation": "BUY"
                    },
                    {
                        "protocol_name": "PancakeSwap",
                        "chain": "bsc", 
                        "protocol_type": "dex",
                        "risk_adjusted_apy": 15.8,
                        "risk_score": 40,
                        "tvl_usd": 2_800_000_000,
                        "recommendation": "STRONG_BUY"
                    },
                    {
                        "protocol_name": "Uniswap V3",
                        "chain": "ethereum",
                        "protocol_type": "dex",
                        "risk_adjusted_apy": 9.2,
                        "risk_score": 35,
                        "tvl_usd": 4_200_000_000,
                        "recommendation": "HOLD"
                    }
                ],
                "arbitrage_opportunities": [
                    {
                        "source_chain": "ethereum",
                        "target_chain": "bsc",
                        "source_protocol": "Compound",
                        "target_protocol": "Venus",
                        "apy_difference": 4.5,
                        "estimated_profit": 3.2,
                        "bridge_cost_estimate": 1.3,
                        "risk_assessment": {"overall_risk": "MEDIUM"}
                    }
                ]
            }
            
            return jsonify(protocol_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/defi/enter-position', methods=['POST'])
    def api_enter_position():
        """Enter DeFi position API"""
        try:
            request_data = request.get_json() or {}
            protocol = request_data.get('protocol', 'Unknown')
            chain = request_data.get('chain', 'ethereum')
            
            # Mock position entry
            result = {
                "status": "completed",
                "message": f"Position entered on {protocol} ({chain})",
                "protocol": protocol,
                "chain": chain,
                "execution_mode": "simulated",
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/defi/execute-arbitrage', methods=['POST'])
    def api_execute_arbitrage():
        """Execute arbitrage API"""
        try:
            request_data = request.get_json() or {}
            source_chain = request_data.get('source_chain', 'ethereum')
            target_chain = request_data.get('target_chain', 'bsc')
            
            # Mock arbitrage execution
            result = {
                "status": "completed",
                "message": f"Arbitrage executed: {source_chain} → {target_chain}",
                "source_chain": source_chain,
                "target_chain": target_chain,
                "estimated_profit": 3.2,
                "execution_mode": "simulated",
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_recent_transactions(limit: int = 10) -> list:
        """Get recent transactions from database"""
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM transactions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('dashboard.html', error="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('dashboard.html', error="Internal server error"), 500
    
    return app

