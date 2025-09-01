"""
Local LLM client for Ollama integration
Handles communication with local language models
"""

import requests
import json
from typing import Dict, Any, Optional, List
import logging
from config import config

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, host: str = None, model: str = None):
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL
        self.session = requests.Session()
        
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Generate text using the local LLM"""
        if not self.is_available():
            logger.error("Ollama server not available")
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                },
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"LLM generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating with LLM: {e}")
            return None
    
    def analyze_financial_data(self, data: Dict[str, Any], 
                              context: str = "") -> Optional[Dict[str, Any]]:
        """Analyze financial data and return structured insights"""
        system_prompt = """You are a financial analysis expert. Analyze the provided data and return insights in JSON format with the following structure:
        {
            "analysis": "detailed analysis text",
            "recommendations": ["list", "of", "recommendations"],
            "risk_level": "low|medium|high",
            "confidence": 0.0-1.0,
            "key_metrics": {"metric": "value"}
        }"""
        
        prompt = f"""
        Context: {context}
        
        Financial Data to Analyze:
        {json.dumps(data, indent=2)}
        
        Provide a comprehensive financial analysis in the specified JSON format.
        """
        
        response = self.generate(prompt, system_prompt, temperature=0.3)
        if response:
            try:
                # Extract JSON from response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing LLM analysis: {e}")
        
        return None
    
    def make_trading_decision(self, market_data: Dict[str, Any], 
                             portfolio_data: Dict[str, Any],
                             strategy_context: str = "") -> Optional[Dict[str, Any]]:
        """Make a trading decision based on market and portfolio data"""
        system_prompt = """You are a professional trading strategist. Based on the provided market data and portfolio information, make a trading decision in JSON format:
        {
            "action": "buy|sell|hold",
            "symbol": "trading symbol",
            "quantity": number,
            "reasoning": "detailed explanation",
            "confidence": 0.0-1.0,
            "risk_assessment": "low|medium|high",
            "stop_loss": number,
            "take_profit": number
        }"""
        
        prompt = f"""
        Strategy Context: {strategy_context}
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Portfolio Data:
        {json.dumps(portfolio_data, indent=2)}
        
        Make a trading decision based on this information.
        """
        
        response = self.generate(prompt, system_prompt, temperature=0.4)
        if response:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing trading decision: {e}")
        
        return None
    
    def generate_financial_plan(self, financial_data: Dict[str, Any],
                               goals: List[str], time_horizon: str) -> Optional[Dict[str, Any]]:
        """Generate a financial plan based on current situation and goals"""
        system_prompt = """You are a financial planning expert. Create a comprehensive financial plan in JSON format:
        {
            "weekly_targets": {
                "cash_allocation": percentage,
                "investment_allocation": percentage,
                "risk_allocation": percentage
            },
            "action_items": ["list", "of", "specific", "actions"],
            "risk_management": "risk management strategy",
            "monitoring_metrics": ["metrics", "to", "track"],
            "confidence": 0.0-1.0
        }"""
        
        prompt = f"""
        Current Financial Situation:
        {json.dumps(financial_data, indent=2)}
        
        Goals: {', '.join(goals)}
        Time Horizon: {time_horizon}
        
        Create a detailed financial plan for the upcoming week.
        """
        
        response = self.generate(prompt, system_prompt, temperature=0.3)
        if response:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except Exception as e:
                logger.error(f"Error parsing financial plan: {e}")
        
        return None

# Global LLM client instance
llm_client = OllamaClient()
