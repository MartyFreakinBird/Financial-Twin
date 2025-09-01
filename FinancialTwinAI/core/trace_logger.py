"""
Decision trace logging system for LAM training data collection
Captures agent decisions, reasoning, and outcomes for future model training
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging
import os

from core.database import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class DecisionTrace:
    """Structure for a single decision trace"""
    trace_id: str
    agent_name: str
    timestamp: datetime
    context: Dict[str, Any]
    reasoning_process: str
    decision: Dict[str, Any]
    action_taken: str
    confidence: float
    outcome: Optional[Dict[str, Any]] = None
    performance_score: Optional[float] = None

class TraceLogger:
    """Manages decision trace logging for LAM training"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.trace_dir = "traces"
        os.makedirs(self.trace_dir, exist_ok=True)
        
    def create_trace_id(self) -> str:
        """Generate a unique trace ID"""
        return str(uuid.uuid4())
    
    def start_decision_trace(self, agent_name: str, context: Dict[str, Any]) -> str:
        """Start a new decision trace"""
        trace_id = self.create_trace_id()
        
        # Store initial trace data
        trace_data = {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "start_timestamp": datetime.now().isoformat(),
            "context": context,
            "status": "started"
        }
        
        # Save to file for immediate backup
        trace_file = os.path.join(self.trace_dir, f"{trace_id}.json")
        with open(trace_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        logger.info(f"Started decision trace {trace_id} for agent {agent_name}")
        return trace_id
    
    def log_reasoning_step(self, trace_id: str, step: str, data: Dict[str, Any]):
        """Log a reasoning step in the decision process"""
        trace_file = os.path.join(self.trace_dir, f"{trace_id}.json")
        
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            if "reasoning_steps" not in trace_data:
                trace_data["reasoning_steps"] = []
            
            trace_data["reasoning_steps"].append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging reasoning step for trace {trace_id}: {e}")
    
    def complete_decision_trace(self, trace_id: str, decision: Dict[str, Any],
                               action_taken: str, confidence: float) -> DecisionTrace:
        """Complete a decision trace with final decision and action"""
        trace_file = os.path.join(self.trace_dir, f"{trace_id}.json")
        
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            # Update trace with decision info
            trace_data.update({
                "decision": decision,
                "action_taken": action_taken,
                "confidence": confidence,
                "completion_timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
            
            # Save updated trace
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2)
            
            # Create DecisionTrace object
            decision_trace = DecisionTrace(
                trace_id=trace_id,
                agent_name=trace_data["agent_name"],
                timestamp=datetime.fromisoformat(trace_data["start_timestamp"]),
                context=trace_data["context"],
                reasoning_process=json.dumps(trace_data.get("reasoning_steps", [])),
                decision=decision,
                action_taken=action_taken,
                confidence=confidence
            )
            
            # Store in database
            self.db_manager.log_decision_trace(
                trace_id=trace_id,
                agent_name=trace_data["agent_name"],
                context_data=trace_data["context"],
                decision_process=decision_trace.reasoning_process,
                action_taken=action_taken
            )
            
            logger.info(f"Completed decision trace {trace_id}")
            return decision_trace
            
        except Exception as e:
            logger.error(f"Error completing decision trace {trace_id}: {e}")
            return None
    
    def log_outcome(self, trace_id: str, outcome: Dict[str, Any], 
                   performance_score: Optional[float] = None):
        """Log the outcome of a traced decision"""
        trace_file = os.path.join(self.trace_dir, f"{trace_id}.json")
        
        try:
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            trace_data.update({
                "outcome": outcome,
                "performance_score": performance_score,
                "outcome_timestamp": datetime.now().isoformat()
            })
            
            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2)
            
            # Update database
            self.db_manager.log_decision_trace(
                trace_id=trace_id,
                agent_name=trace_data["agent_name"],
                context_data=trace_data["context"],
                decision_process=json.dumps(trace_data.get("reasoning_steps", [])),
                action_taken=trace_data["action_taken"],
                outcome_data=outcome,
                performance_score=performance_score
            )
            
            logger.info(f"Logged outcome for trace {trace_id} with score {performance_score}")
            
        except Exception as e:
            logger.error(f"Error logging outcome for trace {trace_id}: {e}")
    
    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected traces"""
        traces = self.db_manager.get_decision_traces()
        
        if not traces:
            return {"total_traces": 0, "agents": {}}
        
        stats = {
            "total_traces": len(traces),
            "agents": {},
            "average_confidence": 0,
            "completed_with_outcomes": 0
        }
        
        agent_counts = {}
        total_confidence = 0
        outcome_count = 0
        
        for trace in traces:
            agent_name = trace["agent_name"]
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
            
            if trace.get("performance_score"):
                total_confidence += float(trace["performance_score"])
                outcome_count += 1
        
        stats["agents"] = agent_counts
        if outcome_count > 0:
            stats["average_confidence"] = total_confidence / outcome_count
        stats["completed_with_outcomes"] = outcome_count
        
        return stats
    
    def export_training_data(self, agent_name: Optional[str] = None,
                           min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Export decision traces as training data for LAM"""
        traces = self.db_manager.get_decision_traces(agent_name=agent_name)
        
        training_data = []
        for trace in traces:
            if trace.get("performance_score", 0) >= min_confidence:
                training_sample = {
                    "input": {
                        "context": json.loads(trace["context_data"]),
                        "agent_type": trace["agent_name"]
                    },
                    "reasoning": trace["decision_process"],
                    "output": trace["action_taken"],
                    "quality_score": trace.get("performance_score", 0)
                }
                training_data.append(training_sample)
        
        return training_data

# Global trace logger instance
trace_logger = TraceLogger()
