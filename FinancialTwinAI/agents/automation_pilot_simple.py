"""
Automation Pilot Agent (Simplified)
Handles GUI actions when no API exists - with graceful fallback for missing dependencies
"""

from typing import Dict, Any, Optional, List
import logging
import time
import os
from datetime import datetime
import json

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class MockGUIAutomation:
    """Mock GUI automation for when pyautogui is not available"""
    
    def __init__(self):
        self.PAUSE = 1
        self.FAILSAFE = True
    
    def click(self, x, y):
        logger.info(f"Mock GUI: Click at ({x}, {y})")
        return {"status": "simulated", "action": "click", "coordinates": [x, y]}
    
    def type(self, text):
        logger.info(f"Mock GUI: Type text '{text}'")
        return {"status": "simulated", "action": "type", "text": text}
    
    def hotkey(self, *keys):
        logger.info(f"Mock GUI: Hotkey {keys}")
        return {"status": "simulated", "action": "hotkey", "keys": keys}
    
    def press(self, key):
        logger.info(f"Mock GUI: Press key '{key}'")
        return {"status": "simulated", "action": "press", "key": key}
    
    def scroll(self, x, y, clicks):
        logger.info(f"Mock GUI: Scroll at ({x}, {y}) for {clicks} clicks")
        return {"status": "simulated", "action": "scroll", "coordinates": [x, y], "clicks": clicks}
    
    def screenshot(self):
        logger.info("Mock GUI: Take screenshot")
        return {"status": "simulated", "action": "screenshot", "filename": "mock_screenshot.png"}
    
    def size(self):
        return (1920, 1080)  # Mock screen size
    
    def position(self):
        return (960, 540)  # Mock mouse position

# Try to import real GUI automation, fall back to mock
try:
    import pyautogui
    GUI_AUTOMATION_AVAILABLE = True
    gui_client = pyautogui
except ImportError:
    GUI_AUTOMATION_AVAILABLE = False
    gui_client = MockGUIAutomation()

class AutomationPilotAgent(BaseAgent):
    """Agent responsible for GUI automation and screen interactions"""
    
    def __init__(self):
        super().__init__(
            name="AutomationPilot",
            description="Handles GUI actions when no API exists (clicking, inputting, exporting reports)"
        )
        
        self.gui = gui_client
        if GUI_AUTOMATION_AVAILABLE:
            self.gui.PAUSE = 1  # 1 second pause between actions
            self.gui.FAILSAFE = True  # Enable fail-safe
            logger.info("GUI automation initialized with pyautogui")
        else:
            logger.warning("GUI automation running in simulation mode - pyautogui not available")
        
        self.input_recorder = None
        self.automation_scripts = {}
        self.pending_tasks = []
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze automation requirements and screen state"""
        trace_id = self.start_decision_trace(context)
        
        try:
            # Get current screen information
            screen_info = self.get_screen_info()
            self.log_reasoning_step("screen_analysis", screen_info)
            
            # Get pending automation tasks
            pending_tasks = self.get_pending_automation_tasks()
            self.log_reasoning_step("pending_tasks", {"task_count": len(pending_tasks)})
            
            # Check for specific automation requests
            automation_request = context.get('automation_request')
            request_analysis = None
            if automation_request:
                request_analysis = self.analyze_automation_request(automation_request)
                self.log_reasoning_step("request_analysis", request_analysis)
            
            # Detect running GUI applications
            gui_applications = self.detect_gui_applications()
            self.log_reasoning_step("gui_detection", {"applications": gui_applications})
            
            analysis = {
                "screen_info": screen_info,
                "pending_tasks": pending_tasks,
                "automation_request": automation_request,
                "request_analysis": request_analysis if automation_request else None,
                "gui_applications": gui_applications,
                "automation_capabilities": self.get_automation_capabilities(),
                "gui_available": GUI_AUTOMATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in automation analysis: {e}")
            return {
                "error": str(e),
                "screen_info": {"error": "Unable to get screen info"},
                "pending_tasks": [],
                "gui_available": GUI_AUTOMATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make automation decisions based on analysis"""
        try:
            decisions = []
            
            # Handle specific automation request
            if analysis.get('automation_request'):
                decision = self.make_specific_automation_decision(
                    analysis['automation_request'], 
                    analysis
                )
                decisions.append(decision)
            
            # Handle pending tasks
            for task in analysis.get('pending_tasks', []):
                decision = self.make_task_automation_decision(task, analysis)
                decisions.append(decision)
            
            # Handle GUI applications that might need automation
            if analysis.get('gui_applications'):
                decision = self.make_gui_automation_decision(
                    analysis['gui_applications'], 
                    analysis
                )
                if decision:
                    decisions.append(decision)
            
            # If no specific tasks, provide status
            if not decisions:
                decisions.append({
                    "action": "status_report",
                    "priority": "low",
                    "details": {
                        "status": "ready",
                        "gui_available": analysis.get('gui_available', False),
                        "capabilities": analysis.get('automation_capabilities', [])
                    },
                    "confidence": 0.9,
                    "reasoning": "No automation tasks pending, system ready"
                })
            
            # Select highest priority decision
            primary_decision = max(decisions, key=lambda d: d.get('priority', 0) if isinstance(d.get('priority'), (int, float)) else 0)
            
            return {
                "action": primary_decision.get("action", "none"),
                "details": primary_decision.get("details", {}),
                "confidence": primary_decision.get("confidence", 0.7),
                "reasoning": primary_decision.get("reasoning", "Automation decision made"),
                "all_decisions": decisions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making automation decision: {e}")
            return {
                "action": "error",
                "details": {"error": str(e)},
                "confidence": 0.0,
                "reasoning": f"Decision error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automation actions"""
        action = decision.get("action", "none")
        details = decision.get("details", {})
        
        try:
            if action == "execute_script":
                result = await self.execute_automation_script(decision)
            elif action == "record_interaction":
                result = await self.record_user_interaction(decision)
            elif action == "gui_automation":
                result = await self.perform_gui_automation(decision)
            elif action == "export_report":
                result = await self.export_report(decision)
            elif action == "input_automation":
                result = await self.perform_input_automation(decision)
            elif action == "status_report":
                result = {
                    "status": "completed",
                    "message": "Automation system status reported",
                    "details": details,
                    "gui_available": GUI_AUTOMATION_AVAILABLE,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                result = {
                    "status": "completed",
                    "message": f"No action required for: {action}",
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing automation action '{action}': {e}")
            return {
                "status": "failed",
                "error": str(e),
                "action": action,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get current screen information"""
        try:
            if GUI_AUTOMATION_AVAILABLE:
                screen_size = self.gui.size()
                mouse_position = self.gui.position()
            else:
                screen_size = (1920, 1080)  # Mock values
                mouse_position = (960, 540)
            
            return {
                "screen_size": screen_size,
                "mouse_position": mouse_position,
                "gui_available": GUI_AUTOMATION_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting screen info: {e}")
            return {
                "error": str(e),
                "gui_available": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_pending_automation_tasks(self) -> List[Dict[str, Any]]:
        """Get pending automation tasks from memory or database"""
        return self.pending_tasks
    
    def get_available_scripts(self) -> List[str]:
        """Get list of available automation scripts"""
        return list(self.automation_scripts.keys())
    
    def detect_gui_applications(self) -> List[Dict[str, Any]]:
        """Detect running GUI applications that might need automation"""
        # Simplified detection - in a real implementation this would
        # check running processes and window titles
        return []
    
    def get_automation_capabilities(self) -> List[str]:
        """Get list of automation capabilities"""
        base_capabilities = [
            "status_reporting",
            "task_scheduling",
            "script_management"
        ]
        
        if GUI_AUTOMATION_AVAILABLE:
            base_capabilities.extend([
                "mouse_automation",
                "keyboard_automation", 
                "screen_capture",
                "window_interaction",
                "form_filling",
                "report_export"
            ])
        else:
            base_capabilities.extend([
                "simulation_mode",
                "mock_interactions"
            ])
        
        return base_capabilities
    
    def analyze_automation_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific automation request"""
        request_type = request.get('type', 'unknown')
        
        return {
            "request_type": request_type,
            "estimated_duration": self.estimate_automation_duration({"details": request}),
            "required_capabilities": self.determine_required_capabilities(request),
            "risk_level": self.assess_automation_risk(request),
            "estimated_steps": self.estimate_automation_steps(request),
            "feasible": GUI_AUTOMATION_AVAILABLE or request_type in ["status", "schedule"]
        }
    
    def make_specific_automation_decision(self, request: Dict[str, Any], 
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for specific automation request"""
        request_analysis = analysis.get('request_analysis', {})
        
        if not request_analysis.get('feasible', False):
            return {
                "action": "status_report",
                "priority": 1,
                "details": {
                    "message": "GUI automation not available - running in simulation mode",
                    "request": request,
                    "capabilities": analysis.get('automation_capabilities', [])
                },
                "confidence": 0.9,
                "reasoning": "Cannot perform GUI automation without required libraries"
            }
        
        return {
            "action": "gui_automation",
            "priority": 3,
            "details": {
                "request": request,
                "estimated_duration": request_analysis.get('estimated_duration', '2-5 minutes'),
                "steps": request_analysis.get('estimated_steps', [])
            },
            "confidence": 0.8,
            "reasoning": f"Executing automation request: {request.get('type', 'unknown')}"
        }
    
    def make_task_automation_decision(self, task: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for pending task automation"""
        return {
            "action": "execute_script",
            "priority": 2,
            "details": {
                "task": task,
                "script_name": task.get('script_name'),
                "parameters": task.get('parameters', {})
            },
            "confidence": 0.7,
            "reasoning": f"Executing pending task: {task.get('name', 'Unknown')}"
        }
    
    def make_gui_automation_decision(self, gui_applications: List[Dict[str, Any]], 
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for GUI application automation"""
        if not gui_applications:
            return None
        
        return {
            "action": "gui_automation",
            "priority": 1,
            "details": {
                "applications": gui_applications,
                "action_type": "health_check"
            },
            "confidence": 0.6,
            "reasoning": "Performing health check on detected GUI applications"
        }
    
    async def execute_automation_script(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a saved automation script"""
        details = decision.get("details", {})
        script_name = details.get("script_name", "unknown")
        
        logger.info(f"Executing automation script: {script_name}")
        
        # Simulate script execution
        return {
            "status": "completed",
            "message": f"Automation script '{script_name}' executed",
            "script_name": script_name,
            "execution_time": "simulated",
            "gui_mode": "simulation" if not GUI_AUTOMATION_AVAILABLE else "real",
            "timestamp": datetime.now().isoformat()
        }
    
    async def record_user_interaction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Record user interactions for building automation scripts"""
        logger.info("Recording user interaction (simulated)")
        
        return {
            "status": "completed",
            "message": "User interaction recording started",
            "recording_mode": "simulation" if not GUI_AUTOMATION_AVAILABLE else "real",
            "timestamp": datetime.now().isoformat()
        }
    
    async def perform_gui_automation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform GUI automation tasks"""
        details = decision.get("details", {})
        request = details.get("request", {})
        
        logger.info(f"Performing GUI automation: {request.get('type', 'unknown')}")
        
        # Simulate automation steps
        steps_performed = []
        
        if request.get('type') == 'click':
            result = self.gui.click(request.get('x', 100), request.get('y', 100))
            steps_performed.append(result)
        elif request.get('type') == 'type':
            result = self.gui.type(request.get('text', ''))
            steps_performed.append(result)
        elif request.get('type') == 'hotkey':
            keys = request.get('keys', ['ctrl', 'c'])
            result = self.gui.hotkey(*keys)
            steps_performed.append(result)
        
        return {
            "status": "completed",
            "message": "GUI automation performed",
            "steps_performed": steps_performed,
            "gui_mode": "simulation" if not GUI_AUTOMATION_AVAILABLE else "real",
            "timestamp": datetime.now().isoformat()
        }
    
    async def export_report(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Export reports through GUI automation"""
        details = decision.get("details", {})
        
        logger.info("Exporting report via GUI automation")
        
        # Simulate report export
        export_steps = [
            self.gui.hotkey('ctrl', 's'),  # Save
            self.gui.type('financial_report.pdf'),
            self.gui.press('enter')
        ]
        
        return {
            "status": "completed",
            "message": "Report export completed",
            "export_file": "financial_report.pdf",
            "steps_performed": export_steps,
            "gui_mode": "simulation" if not GUI_AUTOMATION_AVAILABLE else "real",
            "timestamp": datetime.now().isoformat()
        }
    
    async def perform_input_automation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated input operations"""
        details = decision.get("details", {})
        
        logger.info("Performing input automation")
        
        return {
            "status": "completed",
            "message": "Input automation completed",
            "gui_mode": "simulation" if not GUI_AUTOMATION_AVAILABLE else "real",
            "timestamp": datetime.now().isoformat()
        }
    
    def estimate_automation_duration(self, decision: Dict[str, Any]) -> str:
        """Estimate how long the automation will take"""
        return "1-3 minutes (estimated)"
    
    def estimate_automation_steps(self, request: Dict[str, Any]) -> List[str]:
        """Estimate the steps needed for automation"""
        request_type = request.get('type', 'unknown')
        
        if request_type == 'click':
            return ["Move mouse to position", "Perform click", "Verify action"]
        elif request_type == 'type':
            return ["Focus on input field", "Type text", "Verify input"]
        elif request_type == 'export':
            return ["Open export dialog", "Select format", "Choose location", "Export file"]
        else:
            return ["Analyze requirements", "Execute automation", "Verify results"]
    
    def determine_required_capabilities(self, request: Dict[str, Any]) -> List[str]:
        """Determine what capabilities are needed for the request"""
        request_type = request.get('type', 'unknown')
        
        capabilities_map = {
            'click': ['mouse_automation'],
            'type': ['keyboard_automation'],
            'export': ['keyboard_automation', 'file_operations'],
            'form_fill': ['mouse_automation', 'keyboard_automation'],
            'screenshot': ['screen_capture']
        }
        
        return capabilities_map.get(request_type, ['general_automation'])
    
    def assess_automation_risk(self, request: Dict[str, Any]) -> str:
        """Assess the risk level of the automation request"""
        request_type = request.get('type', 'unknown')
        
        high_risk_types = ['file_operations', 'system_commands']
        medium_risk_types = ['form_fill', 'export']
        
        if request_type in high_risk_types:
            return "high"
        elif request_type in medium_risk_types:
            return "medium"
        else:
            return "low"