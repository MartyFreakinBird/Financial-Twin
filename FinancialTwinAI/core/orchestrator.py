"""
Agent Orchestrator
Manages multi-agent coordination and execution for the Digital Twin LAM system
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from agents.finance_planner import FinancePlannerAgent
from agents.trading_strategist import TradingStrategistAgent
from agents.accounting_clerk import AccountingClerkAgent
from agents.automation_pilot_simple import AutomationPilotAgent
from agents.reinvestor import ReinvestorAgent
from agents.defi_analyst import DeFiAnalystAgent
from core.database import DatabaseManager
from core.trace_logger import trace_logger

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"

@dataclass
class AgentTask:
    agent_name: str
    task_type: str
    context: Dict[str, Any]
    priority: int = 5  # 1-10, 10 being highest
    dependencies: List[str] = None
    timeout: int = 300  # 5 minutes default

class AgentOrchestrator:
    """Orchestrates multi-agent execution and coordination"""
    
    def __init__(self):
        self.agents = {}
        self.db_manager = DatabaseManager()
        self.execution_queue = asyncio.Queue()
        self.running = False
        self.current_execution_id = None
        
    async def initialize(self):
        """Initialize all agents"""
        try:
            logger.info("Initializing agent orchestrator...")
            
            # Initialize agents
            self.agents = {
                "FinancePlanner": FinancePlannerAgent(),
                "TradingStrategist": TradingStrategistAgent(),
                "AccountingClerk": AccountingClerkAgent(),
                "AutomationPilot": AutomationPilotAgent(),
                "Reinvestor": ReinvestorAgent(),
                "DeFiAnalyst": DeFiAnalystAgent()
            }
            
            logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
            
            # Verify agent readiness
            for name, agent in self.agents.items():
                status = agent.get_agent_status()
                logger.info(f"Agent {name} status: {status}")
            
            self.running = True
            logger.info("Agent orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Agent orchestrator initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        logger.info("Shutting down agent orchestrator...")
        self.running = False
        
        # Cancel any running tasks
        if self.current_execution_id:
            logger.info(f"Cancelling current execution: {self.current_execution_id}")
        
        logger.info("Agent orchestrator shutdown complete")
    
    async def execute_weekly_pipeline(self) -> Dict[str, Any]:
        """Execute the weekly automated pipeline"""
        pipeline_id = f"weekly_pipeline_{int(datetime.now().timestamp())}"
        self.current_execution_id = pipeline_id
        
        logger.info(f"Starting weekly pipeline execution: {pipeline_id}")
        
        try:
            # Define the weekly execution sequence
            pipeline_tasks = [
                AgentTask("FinancePlanner", "weekly_plan", {}, priority=10),
                AgentTask("AccountingClerk", "weekly_reconciliation", {}, priority=9),
                AgentTask("DeFiAnalyst", "defi_analysis", {}, priority=8.5),
                AgentTask("TradingStrategist", "market_analysis", {}, priority=8),
                AgentTask("Reinvestor", "yield_analysis", {}, priority=7),
                AgentTask("AutomationPilot", "report_generation", {}, priority=6)
            ]
            
            # Execute pipeline
            execution_results = await self.execute_task_sequence(
                pipeline_tasks, ExecutionMode.SEQUENTIAL
            )
            
            # Generate pipeline summary
            pipeline_summary = self.generate_pipeline_summary(execution_results)
            
            # Store pipeline execution record
            self.store_pipeline_execution(pipeline_id, execution_results, pipeline_summary)
            
            logger.info(f"Weekly pipeline execution completed: {pipeline_id}")
            
            return {
                "pipeline_id": pipeline_id,
                "status": "completed",
                "execution_results": execution_results,
                "summary": pipeline_summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Weekly pipeline execution failed: {e}")
            return {
                "pipeline_id": pipeline_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.current_execution_id = None
    
    async def execute_agent_cycle(self, agent_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single agent cycle"""
        if agent_name not in self.agents:
            return {"status": "error", "message": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        context = context or {}
        
        logger.info(f"Executing agent cycle: {agent_name}")
        
        try:
            # Add orchestrator context
            orchestrator_context = {
                "execution_mode": "single_agent",
                "orchestrator_timestamp": datetime.now().isoformat(),
                "available_agents": list(self.agents.keys())
            }
            context.update(orchestrator_context)
            
            # Execute agent cycle
            result = await agent.run_cycle(context)
            
            # Store execution result
            self.store_agent_execution(agent_name, result)
            
            logger.info(f"Agent cycle completed: {agent_name} - {result.get('status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent cycle execution failed for {agent_name}: {e}")
            return {
                "status": "error",
                "agent": agent_name,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_task_sequence(self, tasks: List[AgentTask], 
                                  mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> List[Dict[str, Any]]:
        """Execute a sequence of agent tasks"""
        if mode == ExecutionMode.SEQUENTIAL:
            return await self.execute_sequential(tasks)
        elif mode == ExecutionMode.PARALLEL:
            return await self.execute_parallel(tasks)
        elif mode == ExecutionMode.PRIORITY_BASED:
            return await self.execute_priority_based(tasks)
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
    
    async def execute_sequential(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially"""
        results = []
        shared_context = {}
        
        for task in tasks:
            logger.info(f"Executing task: {task.agent_name} - {task.task_type}")
            
            # Add shared context from previous tasks
            task_context = {**task.context, **shared_context}
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.execute_agent_cycle(task.agent_name, task_context),
                    timeout=task.timeout
                )
                
                # Update shared context with results
                if result.get("status") == "success":
                    agent_key = f"{task.agent_name}_result"
                    shared_context[agent_key] = result
                    
                    # Extract key insights for other agents
                    if task.agent_name == "FinancePlanner":
                        shared_context["financial_plan"] = result.get("decision", {})
                    elif task.agent_name == "TradingStrategist":
                        shared_context["trading_signals"] = result.get("analysis", {})
                    elif task.agent_name == "AccountingClerk":
                        shared_context["financial_summary"] = result.get("execution", {})
                
                results.append({
                    "task": task,
                    "result": result,
                    "execution_time": datetime.now().isoformat()
                })
                
            except asyncio.TimeoutError:
                logger.error(f"Task timeout: {task.agent_name} - {task.task_type}")
                results.append({
                    "task": task,
                    "result": {"status": "timeout", "message": "Task execution timeout"},
                    "execution_time": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Task execution failed: {task.agent_name} - {e}")
                results.append({
                    "task": task,
                    "result": {"status": "error", "message": str(e)},
                    "execution_time": datetime.now().isoformat()
                })
        
        return results
    
    async def execute_parallel(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel"""
        logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Create coroutines for all tasks
        task_coroutines = []
        for task in tasks:
            coro = self.execute_agent_cycle(task.agent_name, task.context)
            task_coroutines.append(asyncio.wait_for(coro, timeout=task.timeout))
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Format results
            formatted_results = []
            for i, (task, result) in enumerate(zip(tasks, results)):
                if isinstance(result, Exception):
                    formatted_result = {
                        "status": "error",
                        "message": str(result),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    formatted_result = result
                
                formatted_results.append({
                    "task": task,
                    "result": formatted_result,
                    "execution_time": datetime.now().isoformat()
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            return [{
                "task": task,
                "result": {"status": "error", "message": str(e)},
                "execution_time": datetime.now().isoformat()
            } for task in tasks]
    
    async def execute_priority_based(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks based on priority"""
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Group tasks by priority level
        priority_groups = {}
        for task in sorted_tasks:
            if task.priority not in priority_groups:
                priority_groups[task.priority] = []
            priority_groups[task.priority].append(task)
        
        all_results = []
        
        # Execute each priority group
        for priority in sorted(priority_groups.keys(), reverse=True):
            group_tasks = priority_groups[priority]
            logger.info(f"Executing priority {priority} tasks: {len(group_tasks)} tasks")
            
            # Execute high priority tasks sequentially, lower priority in parallel
            if priority >= 9:
                group_results = await self.execute_sequential(group_tasks)
            else:
                group_results = await self.execute_parallel(group_tasks)
            
            all_results.extend(group_results)
        
        return all_results
    
    def generate_pipeline_summary(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of pipeline execution"""
        total_tasks = len(execution_results)
        successful_tasks = len([r for r in execution_results if r["result"].get("status") == "success"])
        failed_tasks = total_tasks - successful_tasks
        
        # Extract key insights
        insights = {}
        for result in execution_results:
            agent_name = result["task"].agent_name
            if result["result"].get("status") == "success":
                insights[agent_name] = {
                    "completed": True,
                    "confidence": result["result"].get("performance_score", 0),
                    "key_decision": result["result"].get("decision", {}).get("action", "none")
                }
            else:
                insights[agent_name] = {
                    "completed": False,
                    "error": result["result"].get("message", "Unknown error")
                }
        
        # Calculate overall performance score
        performance_scores = [
            r["result"].get("performance_score", 0) 
            for r in execution_results 
            if r["result"].get("status") == "success"
        ]
        average_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_performance": average_performance,
            "agent_insights": insights,
            "execution_duration": self.calculate_execution_duration(execution_results),
            "summary_timestamp": datetime.now().isoformat()
        }
    
    def calculate_execution_duration(self, execution_results: List[Dict[str, Any]]) -> float:
        """Calculate total execution duration in seconds"""
        if not execution_results:
            return 0
        
        start_times = []
        end_times = []
        
        for result in execution_results:
            exec_time = result.get("execution_time")
            if exec_time:
                try:
                    timestamp = datetime.fromisoformat(exec_time)
                    start_times.append(timestamp)
                    end_times.append(timestamp)
                except:
                    pass
        
        if start_times and end_times:
            duration = (max(end_times) - min(start_times)).total_seconds()
            return max(duration, 0)
        
        return 0
    
    def store_pipeline_execution(self, pipeline_id: str, execution_results: List[Dict[str, Any]], 
                               summary: Dict[str, Any]):
        """Store pipeline execution record"""
        try:
            execution_record = {
                "pipeline_id": pipeline_id,
                "execution_results": execution_results,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in agent memory for the orchestrator
            executions = self.get_pipeline_executions()
            executions.append(execution_record)
            
            # Keep only last 50 executions
            if len(executions) > 50:
                executions = executions[-50:]
            
            self.store_orchestrator_memory("pipeline_executions", executions)
            
        except Exception as e:
            logger.error(f"Failed to store pipeline execution: {e}")
    
    def store_agent_execution(self, agent_name: str, result: Dict[str, Any]):
        """Store individual agent execution record"""
        try:
            executions = self.get_orchestrator_memory("agent_executions") or {}
            
            if agent_name not in executions:
                executions[agent_name] = []
            
            executions[agent_name].append({
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Keep only last 20 executions per agent
            if len(executions[agent_name]) > 20:
                executions[agent_name] = executions[agent_name][-20:]
            
            self.store_orchestrator_memory("agent_executions", executions)
            
        except Exception as e:
            logger.error(f"Failed to store agent execution: {e}")
    
    def get_pipeline_executions(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history"""
        return self.get_orchestrator_memory("pipeline_executions") or []
    
    def get_agent_executions(self, agent_name: str = None) -> Dict[str, Any]:
        """Get agent execution history"""
        all_executions = self.get_orchestrator_memory("agent_executions") or {}
        
        if agent_name:
            return all_executions.get(agent_name, [])
        
        return all_executions
    
    def store_orchestrator_memory(self, key: str, value: Any):
        """Store orchestrator memory"""
        self.db_manager.store_agent_memory("Orchestrator", key, value)
    
    def get_orchestrator_memory(self, key: str) -> Optional[Any]:
        """Get orchestrator memory"""
        return self.db_manager.get_agent_memory("Orchestrator", key)
    
    async def execute_agent_command(self, agent_name: str, command: str, 
                                   params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific command on an agent"""
        if agent_name not in self.agents:
            return {"status": "error", "message": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        params = params or {}
        
        try:
            if command == "analyze":
                return await agent.analyze(params)
            elif command == "decide":
                analysis = params.get("analysis", {})
                return await agent.make_decision(analysis)
            elif command == "execute":
                decision = params.get("decision", {})
                return await agent.execute_action(decision)
            elif command == "full_cycle":
                return await agent.run_cycle(params)
            elif command == "status":
                return agent.get_agent_status()
            else:
                return {"status": "error", "message": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Agent command execution failed: {agent_name}.{command} - {e}")
            return {"status": "error", "message": str(e)}
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status"""
        agent_statuses = {}
        for name, agent in self.agents.items():
            try:
                agent_statuses[name] = agent.get_agent_status()
            except Exception as e:
                agent_statuses[name] = {"status": "error", "message": str(e)}
        
        recent_pipelines = self.get_pipeline_executions()[-5:]  # Last 5 executions
        
        return {
            "orchestrator_running": self.running,
            "current_execution": self.current_execution_id,
            "agent_count": len(self.agents),
            "agent_statuses": agent_statuses,
            "recent_pipeline_executions": len(recent_pipelines),
            "last_pipeline_execution": recent_pipelines[-1]["timestamp"] if recent_pipelines else None,
            "status_timestamp": datetime.now().isoformat()
        }
    
    async def handle_agent_failure(self, agent_name: str, error: Exception) -> Dict[str, Any]:
        """Handle agent execution failures"""
        logger.error(f"Agent failure detected: {agent_name} - {error}")
        
        failure_record = {
            "agent_name": agent_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
            "recovery_action": "logged_for_analysis"
        }
        
        # Store failure record
        failures = self.get_orchestrator_memory("agent_failures") or []
        failures.append(failure_record)
        
        # Keep only last 100 failures
        if len(failures) > 100:
            failures = failures[-100:]
        
        self.store_orchestrator_memory("agent_failures", failures)
        
        # Attempt recovery based on agent type
        recovery_result = await self.attempt_agent_recovery(agent_name, error)
        failure_record["recovery_result"] = recovery_result
        
        return failure_record
    
    async def attempt_agent_recovery(self, agent_name: str, error: Exception) -> Dict[str, Any]:
        """Attempt to recover from agent failure"""
        try:
            # Simple recovery: reinitialize agent
            if agent_name in self.agents:
                logger.info(f"Attempting to recover agent: {agent_name}")
                
                # Get agent status
                status = self.agents[agent_name].get_agent_status()
                
                return {
                    "recovery_attempted": True,
                    "recovery_method": "status_check",
                    "agent_status": status,
                    "success": True
                }
            else:
                return {
                    "recovery_attempted": False,
                    "reason": "Agent not found",
                    "success": False
                }
                
        except Exception as recovery_error:
            logger.error(f"Agent recovery failed: {agent_name} - {recovery_error}")
            return {
                "recovery_attempted": True,
                "recovery_method": "status_check",
                "success": False,
                "recovery_error": str(recovery_error)
            }
    
    async def schedule_agent_task(self, agent_name: str, task_type: str, 
                                context: Dict[str, Any], 
                                priority: int = 5, delay: int = 0) -> str:
        """Schedule an agent task for future execution"""
        task_id = f"task_{int(datetime.now().timestamp())}_{agent_name}"
        
        task = AgentTask(
            agent_name=agent_name,
            task_type=task_type,
            context=context,
            priority=priority
        )
        
        # If delay is specified, schedule for later
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Add to execution queue
        await self.execution_queue.put((task_id, task))
        
        logger.info(f"Scheduled task: {task_id} for agent {agent_name}")
        return task_id
    
    async def process_task_queue(self):
        """Process scheduled tasks from the queue"""
        while self.running:
            try:
                # Wait for task with timeout
                task_id, task = await asyncio.wait_for(
                    self.execution_queue.get(), timeout=1.0
                )
                
                logger.info(f"Processing queued task: {task_id}")
                
                # Execute the task
                result = await self.execute_agent_cycle(task.agent_name, task.context)
                
                # Store task execution result
                self.store_task_execution(task_id, task, result)
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Task queue processing error: {e}")
                continue
    
    def store_task_execution(self, task_id: str, task: AgentTask, result: Dict[str, Any]):
        """Store scheduled task execution result"""
        try:
            task_executions = self.get_orchestrator_memory("task_executions") or {}
            
            task_executions[task_id] = {
                "task": {
                    "agent_name": task.agent_name,
                    "task_type": task.task_type,
                    "priority": task.priority
                },
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.store_orchestrator_memory("task_executions", task_executions)
            
        except Exception as e:
            logger.error(f"Failed to store task execution: {e}")

