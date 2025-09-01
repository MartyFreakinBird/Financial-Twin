"""
Database management for the Digital Twin LAM system
SQLite-based storage for transactions, agent decisions, and traces
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "financial_lam.db"):
        self.db_path = db_path
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Agent decisions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    decision_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    execution_status TEXT DEFAULT 'pending',
                    outcome TEXT
                )
            """)
            
            # Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_type TEXT NOT NULL,
                    symbol TEXT,
                    quantity REAL,
                    price REAL,
                    total_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    agent_decision_id INTEGER,
                    FOREIGN KEY (agent_decision_id) REFERENCES agent_decisions (id)
                )
            """)
            
            # Portfolio positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    current_price REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Account balance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_balance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cash_balance REAL NOT NULL,
                    total_portfolio_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Decision traces table (for LAM training)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT UNIQUE NOT NULL,
                    agent_name TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    decision_process TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    outcome_data TEXT,
                    performance_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Agent memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent_name, memory_key)
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def log_agent_decision(self, agent_name: str, decision_type: str, 
                          decision_data: Dict[str, Any], confidence: float) -> int:
        """Log an agent decision"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_decisions 
                (agent_name, decision_type, decision_data, confidence)
                VALUES (?, ?, ?, ?)
            """, (agent_name, decision_type, json.dumps(decision_data), confidence))
            conn.commit()
            return cursor.lastrowid
    
    def log_transaction(self, transaction_type: str, symbol: str, quantity: float, 
                       price: float, agent_decision_id: Optional[int] = None) -> int:
        """Log a financial transaction"""
        total_value = quantity * price
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions 
                (transaction_type, symbol, quantity, price, total_value, agent_decision_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (transaction_type, symbol, quantity, price, total_value, agent_decision_id))
            conn.commit()
            return cursor.lastrowid
    
    def update_portfolio_position(self, symbol: str, quantity: float, 
                                 average_price: float, current_price: Optional[float] = None):
        """Update or insert portfolio position"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO portfolio_positions 
                (symbol, quantity, average_price, current_price, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (symbol, quantity, average_price, current_price))
            conn.commit()
    
    def get_portfolio_positions(self) -> List[Dict[str, Any]]:
        """Get current portfolio positions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio_positions WHERE quantity > 0")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_account_balance(self, cash_balance: float, total_portfolio_value: float):
        """Update account balance"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO account_balance (cash_balance, total_portfolio_value)
                VALUES (?, ?)
            """, (cash_balance, total_portfolio_value))
            conn.commit()
    
    def get_latest_balance(self) -> Optional[Dict[str, Any]]:
        """Get the latest account balance"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM account_balance 
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def log_decision_trace(self, trace_id: str, agent_name: str, 
                          context_data: Dict[str, Any], decision_process: str,
                          action_taken: str, outcome_data: Optional[Dict[str, Any]] = None,
                          performance_score: Optional[float] = None):
        """Log a decision trace for LAM training"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO decision_traces 
                (trace_id, agent_name, context_data, decision_process, 
                 action_taken, outcome_data, performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (trace_id, agent_name, json.dumps(context_data), decision_process,
                  action_taken, json.dumps(outcome_data) if outcome_data else None,
                  performance_score))
            conn.commit()
    
    def get_decision_traces(self, agent_name: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Get decision traces for analysis"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if agent_name:
                cursor.execute("""
                    SELECT * FROM decision_traces 
                    WHERE agent_name = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (agent_name, limit))
            else:
                cursor.execute("""
                    SELECT * FROM decision_traces 
                    ORDER BY timestamp DESC LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def store_agent_memory(self, agent_name: str, memory_key: str, memory_value: Any):
        """Store agent memory"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agent_memory 
                (agent_name, memory_key, memory_value, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (agent_name, memory_key, json.dumps(memory_value)))
            conn.commit()
    
    def get_agent_memory(self, agent_name: str, memory_key: str) -> Optional[Any]:
        """Retrieve agent memory"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT memory_value FROM agent_memory 
                WHERE agent_name = ? AND memory_key = ?
            """, (agent_name, memory_key))
            row = cursor.fetchone()
            return json.loads(row['memory_value']) if row else None
