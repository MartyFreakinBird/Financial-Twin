"""
Base agent class for the Digital Twin LAM system
Provides common functionality for all specialized agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import uuid
from datetime import datetime

from core.llm_client import llm_client
from core.database import DatabaseManager
from core.trace_logger import trace_logger

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all financial agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.db_manager = DatabaseManager()
        self.memory = {}
        self.current_trace_id = None
        
    def get_memory(self, key: str) -> Optional[Any]:
        """Retrieve agent memory"""
        if key in self.memory:
            return self.memory[key]
        
        # Try to load from database
        value = self.db_manager.get_agent_memory(self.name, key)
        if value:
            self.memory[key] = value
        return value
    
    def store_memory(self, key: str, value: Any):
        """Store agent memory"""
        self.memory[key] = value
        self.db_manager.store_agent_memory(self.name, key, value)
    
    def start_decision_trace(self, context: Dict[str, Any]) -> str:
        """Start tracing a decision for LAM training"""
        self.current_trace_id = trace_logger.start_decision_trace(self.name, context)
        return self.current_trace_id
    
    def log_reasoning_step(self, step: str, data: Dict[str, Any]):
        """Log a reasoning step in the current trace"""
        if self.current_trace_id:
            trace_logger.log_reasoning_step(self.current_trace_id, step, data)
    
    def complete_decision_trace(self, decision: Dict[str, Any], 
                               action: str, confidence: float):
        """Complete the current decision trace"""
        if self.current_trace_id:
            trace_logger.complete_decision_trace(
                self.current_trace_id, decision, action, confidence
            )
            self.current_trace_id = None
    
    def log_decision_outcome(self, trace_id: str, outcome: Dict[str, Any],
                            performance_score: Optional[float] = None):
        """Log the outcome of a decision"""
        trace_logger.log_outcome(trace_id, outcome, performance_score)
    
    @abstractmethod
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the given context and return insights"""
        pass
    
    @abstractmethod
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision based on analysis"""
        pass
    
    @abstractmethod
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action"""
        pass
    
    async def run_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete agent cycle: analyze -> decide -> execute"""
        try:
            # Start decision trace
            trace_id = self.start_decision_trace(context)
            
            logger.info(f"Agent {self.name} starting cycle with trace {trace_id}")
            
            # Analysis phase
            self.log_reasoning_step("analysis_start", {"context": context})
            analysis = await self.analyze(context)
            self.log_reasoning_step("analysis_complete", {"analysis": analysis})
            
            if not analysis:
                logger.error(f"Agent {self.name} failed to analyze context")
                return {"status": "error", "message": "Analysis failed"}
            
            # Decision phase
            self.log_reasoning_step("decision_start", {"analysis": analysis})
            decision = await self.make_decision(analysis)
            self.log_reasoning_step("decision_complete", {"decision": decision})
            
            if not decision:
                logger.error(f"Agent {self.name} failed to make decision")
                return {"status": "error", "message": "Decision failed"}
            
            # Complete trace before execution
            confidence = decision.get("confidence", 0.5)
            action_description = f"{decision.get('action', 'unknown')} - {decision.get('reasoning', '')}"
            self.complete_decision_trace(decision, action_description, confidence)
            
            # Execution phase
            execution_result = await self.execute_action(decision)
            
            # Log outcome
            outcome = {
                "execution_result": execution_result,
                "decision": decision,
                "analysis": analysis
            }
            
            # Calculate performance score based on execution success
            performance_score = self.calculate_performance_score(
                decision, execution_result, analysis
            )
            
            self.log_decision_outcome(trace_id, outcome, performance_score)
            
            logger.info(f"Agent {self.name} completed cycle successfully")
            
            return {
                "status": "success",
                "trace_id": trace_id,
                "analysis": analysis,
                "decision": decision,
                "execution": execution_result,
                "performance_score": performance_score
            }
            
        except Exception as e:
            logger.error(f"Agent {self.name} cycle failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def calculate_performance_score(self, decision: Dict[str, Any],
                                   execution_result: Dict[str, Any],
                                   analysis: Dict[str, Any]) -> float:
        """Calculate performance score for the decision"""
        base_score = 0.5
        
        # Increase score if execution was successful
        if execution_result.get("status") == "success":
            base_score += 0.3
        
        # Increase score based on confidence if high confidence led to success
        if decision.get("confidence", 0) > 0.7 and execution_result.get("status") == "success":
            base_score += 0.2
        
        # Adjust based on risk level
        risk_level = decision.get("risk_level", "medium")
        if risk_level == "low" and execution_result.get("status") == "success":
            base_score += 0.1
        elif risk_level == "high" and execution_result.get("status") == "success":
            base_score += 0.2
        
        return min(1.0, max(0.0, base_score))
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "description": self.description,
            "memory_keys": list(self.memory.keys()),
            "active_trace": self.current_trace_id,
            "last_activity": datetime.now().isoformat()
        }
