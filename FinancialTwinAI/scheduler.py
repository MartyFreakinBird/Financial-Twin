"""
Weekly Scheduler
Manages automated weekly execution pipeline for the Digital Twin LAM system
"""

import asyncio
import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from core.orchestrator import AgentOrchestrator
from core.database import DatabaseManager
from config import config

logger = logging.getLogger(__name__)

class WeeklyScheduler:
    """Scheduler for automated weekly agent pipeline execution"""
    
    def __init__(self, orchestrator: AgentOrchestrator = None):
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.db_manager = DatabaseManager()
        self.is_running = False
        self.scheduler_thread = None
        self.current_execution = None
        
        # Schedule configuration
        self.execution_day = config.WEEKLY_EXECUTION_DAY  # 0 = Monday
        self.execution_hour = config.WEEKLY_EXECUTION_HOUR  # 9 AM
        
        logger.info(f"WeeklyScheduler initialized - execution scheduled for day {self.execution_day} at {self.execution_hour}:00")
    
    def setup_schedule(self):
        """Setup the weekly schedule"""
        try:
            # Clear any existing schedules
            schedule.clear()
            
            # Map day number to schedule day name
            day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = day_names[self.execution_day]
            
            # Schedule weekly execution
            scheduled_job = getattr(schedule.every(), day_name).at(f"{self.execution_hour:02d}:00")
            scheduled_job.do(self.trigger_weekly_execution)
            
            logger.info(f"Weekly execution scheduled for {day_name.title()} at {self.execution_hour:02d}:00")
            
            # Also schedule daily health checks
            schedule.every().day.at("08:00").do(self.daily_health_check)
            schedule.every().day.at("20:00").do(self.evening_summary)
            
            logger.info("Daily health checks scheduled for 08:00 and 20:00")
            
        except Exception as e:
            logger.error(f"Failed to setup schedule: {e}")
            raise
    
    def run(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            self.setup_schedule()
            self.is_running = True
            
            logger.info("Starting scheduler thread...")
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            logger.info("WeeklyScheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping WeeklyScheduler...")
        self.is_running = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("WeeklyScheduler stopped")
    
    def _run_scheduler(self):
        """Internal scheduler loop"""
        logger.info("Scheduler thread started")
        
        while self.is_running:
            try:
                # Run pending scheduled jobs
                schedule.run_pending()
                
                # Sleep for 1 minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Continue after error
        
        logger.info("Scheduler thread stopped")
    
    def trigger_weekly_execution(self):
        """Trigger the weekly pipeline execution"""
        if self.current_execution:
            logger.warning("Weekly execution already in progress, skipping")
            return
        
        logger.info("Triggering weekly pipeline execution...")
        
        try:
            # Create execution context
            execution_context = {
                "execution_type": "scheduled_weekly",
                "scheduled_time": datetime.now().isoformat(),
                "execution_day": self.execution_day,
                "execution_hour": self.execution_hour
            }
            
            # Start execution in background
            execution_thread = threading.Thread(
                target=self._execute_weekly_pipeline,
                args=(execution_context,),
                daemon=True
            )
            execution_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to trigger weekly execution: {e}")
    
    def _execute_weekly_pipeline(self, context: Dict[str, Any]):
        """Execute the weekly pipeline"""
        execution_id = f"weekly_{int(datetime.now().timestamp())}"
        self.current_execution = execution_id
        
        try:
            logger.info(f"Starting weekly pipeline execution: {execution_id}")
            
            # Record execution start
            self.record_execution_start(execution_id, context)
            
            # Run the weekly pipeline using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.orchestrator.execute_weekly_pipeline()
                )
                
                # Record successful execution
                self.record_execution_completion(execution_id, result, success=True)
                
                logger.info(f"Weekly pipeline execution completed successfully: {execution_id}")
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Weekly pipeline execution failed: {execution_id} - {e}")
            
            # Record failed execution
            error_result = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.record_execution_completion(execution_id, error_result, success=False)
            
        finally:
            self.current_execution = None
    
    def daily_health_check(self):
        """Perform daily system health check"""
        logger.info("Performing daily health check...")
        
        try:
            # Check system components
            health_status = {
                "database": self.check_database_health(),
                "agents": self.check_agents_health(),
                "orchestrator": self.check_orchestrator_health(),
                "scheduler": True,  # If we're here, scheduler is working
                "timestamp": datetime.now().isoformat()
            }
            
            # Log health status
            logger.info(f"Health check results: {health_status}")
            
            # Store health check results
            self.store_health_check(health_status)
            
            # Alert if any component is unhealthy
            unhealthy_components = [k for k, v in health_status.items() if v is False and k != "timestamp"]
            if unhealthy_components:
                logger.warning(f"Unhealthy components detected: {unhealthy_components}")
                self.alert_unhealthy_components(unhealthy_components)
            
        except Exception as e:
            logger.error(f"Daily health check failed: {e}")
    
    def evening_summary(self):
        """Generate evening summary of system activity"""
        logger.info("Generating evening summary...")
        
        try:
            # Get today's activity
            today = datetime.now().date()
            summary = self.generate_daily_summary(today)
            
            logger.info(f"Daily summary: {summary}")
            
            # Store summary
            self.store_daily_summary(summary)
            
        except Exception as e:
            logger.error(f"Evening summary generation failed: {e}")
    
    def check_database_health(self) -> bool:
        """Check database connectivity and basic operations"""
        try:
            # Test database connection
            latest_balance = self.db_manager.get_latest_balance()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_agents_health(self) -> bool:
        """Check agent system health"""
        try:
            # This would normally check agent availability
            # For now, return True if no critical errors
            return True
        except Exception as e:
            logger.error(f"Agents health check failed: {e}")
            return False
    
    def check_orchestrator_health(self) -> bool:
        """Check orchestrator health"""
        try:
            # Check if orchestrator is available and responding
            if hasattr(self.orchestrator, 'running'):
                return self.orchestrator.running
            return True
        except Exception as e:
            logger.error(f"Orchestrator health check failed: {e}")
            return False
    
    def record_execution_start(self, execution_id: str, context: Dict[str, Any]):
        """Record the start of a scheduled execution"""
        try:
            execution_record = {
                "execution_id": execution_id,
                "execution_type": "weekly_scheduled",
                "status": "started",
                "context": context,
                "start_time": datetime.now().isoformat()
            }
            
            # Store in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO scheduled_executions 
                    (execution_id, execution_type, status, context, start_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    execution_id,
                    execution_record["execution_type"],
                    execution_record["status"],
                    str(context),
                    execution_record["start_time"]
                ))
                conn.commit()
            
            logger.info(f"Recorded execution start: {execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to record execution start: {e}")
    
    def record_execution_completion(self, execution_id: str, result: Dict[str, Any], success: bool):
        """Record the completion of a scheduled execution"""
        try:
            # Update execution record
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE scheduled_executions 
                    SET status = ?, result = ?, end_time = ?, success = ?
                    WHERE execution_id = ?
                """, (
                    "completed" if success else "failed",
                    str(result),
                    datetime.now().isoformat(),
                    success,
                    execution_id
                ))
                conn.commit()
            
            logger.info(f"Recorded execution completion: {execution_id} - success: {success}")
            
        except Exception as e:
            logger.error(f"Failed to record execution completion: {e}")
    
    def store_health_check(self, health_status: Dict[str, Any]):
        """Store health check results"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS health_checks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        database_health BOOLEAN,
                        agents_health BOOLEAN,
                        orchestrator_health BOOLEAN,
                        scheduler_health BOOLEAN,
                        overall_health BOOLEAN
                    )
                """)
                
                overall_health = all(v for k, v in health_status.items() if k != "timestamp")
                
                cursor.execute("""
                    INSERT INTO health_checks 
                    (database_health, agents_health, orchestrator_health, scheduler_health, overall_health)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    health_status.get("database", False),
                    health_status.get("agents", False),
                    health_status.get("orchestrator", False),
                    health_status.get("scheduler", False),
                    overall_health
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store health check: {e}")
    
    def alert_unhealthy_components(self, components: list):
        """Alert about unhealthy system components"""
        logger.warning(f"ALERT: Unhealthy components detected: {', '.join(components)}")
        
        # In a production system, this would send notifications
        # For now, just log the alert
        alert_message = f"System Health Alert: {', '.join(components)} components are unhealthy"
        
        try:
            # Store alert in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT,
                        message TEXT,
                        components TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE
                    )
                """)
                
                cursor.execute("""
                    INSERT INTO system_alerts (alert_type, message, components)
                    VALUES (?, ?, ?)
                """, ("health_check", alert_message, ','.join(components)))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store system alert: {e}")
    
    def generate_daily_summary(self, date) -> Dict[str, Any]:
        """Generate summary of daily system activity"""
        try:
            # Get agent decisions for the day
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT agent_name, COUNT(*) as decisions, AVG(confidence) as avg_confidence
                    FROM agent_decisions 
                    WHERE DATE(timestamp) = DATE(?)
                    GROUP BY agent_name
                """, (date.isoformat(),))
                agent_activity = [dict(row) for row in cursor.fetchall()]
                
                # Get transactions for the day
                cursor.execute("""
                    SELECT transaction_type, COUNT(*) as count, SUM(total_value) as total_value
                    FROM transactions 
                    WHERE DATE(timestamp) = DATE(?)
                    GROUP BY transaction_type
                """, (date.isoformat(),))
                transaction_activity = [dict(row) for row in cursor.fetchall()]
            
            summary = {
                "date": date.isoformat(),
                "agent_activity": agent_activity,
                "transaction_activity": transaction_activity,
                "total_agent_decisions": sum(a.get("decisions", 0) for a in agent_activity),
                "total_transactions": sum(t.get("count", 0) for t in transaction_activity),
                "avg_confidence": sum(a.get("avg_confidence", 0) for a in agent_activity) / len(agent_activity) if agent_activity else 0,
                "generated_at": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate daily summary: {e}")
            return {
                "date": date.isoformat(),
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def store_daily_summary(self, summary: Dict[str, Any]):
        """Store daily summary in database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE,
                        summary_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_summaries (date, summary_data)
                    VALUES (?, ?)
                """, (summary["date"], str(summary)))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store daily summary: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        next_run = None
        if self.is_running:
            try:
                jobs = schedule.jobs
                if jobs:
                    next_run = min(job.next_run for job in jobs).isoformat()
            except:
                pass
        
        return {
            "is_running": self.is_running,
            "execution_day": self.execution_day,
            "execution_hour": self.execution_hour,
            "current_execution": self.current_execution,
            "next_scheduled_run": next_run,
            "scheduled_jobs": len(schedule.jobs) if self.is_running else 0,
            "status_timestamp": datetime.now().isoformat()
        }
    
    def force_weekly_execution(self) -> str:
        """Force immediate execution of weekly pipeline"""
        if self.current_execution:
            return f"Execution already in progress: {self.current_execution}"
        
        logger.info("Forcing immediate weekly pipeline execution...")
        
        execution_context = {
            "execution_type": "manual_trigger",
            "triggered_time": datetime.now().isoformat(),
            "forced": True
        }
        
        # Start execution in background
        execution_thread = threading.Thread(
            target=self._execute_weekly_pipeline,
            args=(execution_context,),
            daemon=True
        )
        execution_thread.start()
        
        return f"Weekly execution triggered manually at {datetime.now().isoformat()}"
    
    def get_execution_history(self, limit: int = 10) -> list:
        """Get history of scheduled executions"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT UNIQUE,
                        execution_type TEXT,
                        status TEXT,
                        context TEXT,
                        result TEXT,
                        start_time DATETIME,
                        end_time DATETIME,
                        success BOOLEAN
                    )
                """)
                
                cursor.execute("""
                    SELECT * FROM scheduled_executions 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get execution history: {e}")
            return []
    
    def cleanup_old_records(self, days: int = 30):
        """Clean up old execution and health check records"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old scheduled executions
                cursor.execute("""
                    DELETE FROM scheduled_executions 
                    WHERE start_time < ?
                """, (cutoff_date,))
                
                # Clean up old health checks
                cursor.execute("""
                    DELETE FROM health_checks 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                # Clean up old daily summaries
                cursor.execute("""
                    DELETE FROM daily_summaries 
                    WHERE date < DATE(?)
                """, (cutoff_date,))
                
                conn.commit()
                
            logger.info(f"Cleaned up records older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")


# Initialize database tables for scheduler
def initialize_scheduler_tables():
    """Initialize database tables required by the scheduler"""
    db_manager = DatabaseManager()
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Scheduled executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scheduled_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT UNIQUE,
                    execution_type TEXT,
                    status TEXT,
                    context TEXT,
                    result TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    success BOOLEAN
                )
            """)
            
            # Health checks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    database_health BOOLEAN,
                    agents_health BOOLEAN,
                    orchestrator_health BOOLEAN,
                    scheduler_health BOOLEAN,
                    overall_health BOOLEAN
                )
            """)
            
            # System alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    message TEXT,
                    components TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Daily summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE,
                    summary_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
        logger.info("Scheduler database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize scheduler tables: {e}")
        raise

# Initialize tables when module is imported
initialize_scheduler_tables()
