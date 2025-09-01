"""
Configuration settings for the Digital Twin LAM system
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    # Database settings
    DATABASE_PATH: str = "financial_lam.db"
    
    # Ollama settings
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    
    # Trading settings
    INITIAL_BALANCE: float = 10000.0
    MAX_POSITION_SIZE: float = 0.1  # 10% max position size
    STOP_LOSS_PERCENT: float = 0.02  # 2% stop loss
    
    # Agent settings
    DECISION_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_RETRIES: int = 3
    
    # Scheduler settings
    WEEKLY_EXECUTION_DAY: int = 0  # Monday
    WEEKLY_EXECUTION_HOUR: int = 9  # 9 AM
    
    # API Keys (from environment)
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    TRACE_LOG_PATH: str = "traces"
    
    @classmethod
    def get_trading_config(cls) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return {
            "initial_balance": cls.INITIAL_BALANCE,
            "max_position_size": cls.MAX_POSITION_SIZE,
            "stop_loss_percent": cls.STOP_LOSS_PERCENT
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM-specific configuration"""
        return {
            "host": cls.OLLAMA_HOST,
            "model": cls.OLLAMA_MODEL
        }

# Global config instance
config = Config()
