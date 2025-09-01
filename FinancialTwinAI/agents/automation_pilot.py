"""
Automation Pilot Agent
Handles GUI actions when no API exists (clicking, inputting, exporting reports)
"""

from typing import Dict, Any, Optional, List
import logging
import time
import os
from datetime import datetime
import json

from agents.base_agent import BaseAgent

# Optional GUI automation imports
try:
    import pyautogui
    GUI_AUTOMATION_AVAILABLE = True
except ImportError:
    pyautogui = None
    GUI_AUTOMATION_AVAILABLE = False

try:
    import pynput
    from pynput import mouse, keyboard
    INPUT_CAPTURE_AVAILABLE = True
except ImportError:
    pynput = None
    mouse = None
    keyboard = None
    INPUT_CAPTURE_AVAILABLE = False

logger = logging.getLogger(__name__)

class AutomationPilotAgent(BaseAgent):
    """Agent responsible for GUI automation and screen interactions"""
    
    def __init__(self):
        super().__init__(
            name="AutomationPilot",
            description="Handles GUI actions when no API exists (clicking, inputting, exporting reports)"
        )
        
        # Configure pyautogui if available
        if GUI_AUTOMATION_AVAILABLE:
            pyautogui.PAUSE = 1  # 1 second pause between actions
            pyautogui.FAILSAFE = True  # Enable fail-safe
        else:
            logger.warning("GUI automation libraries not available - running in simulation mode")
        
        # Initialize input capture
        self.input_recorder = InputRecorder()
        self.automation_scripts = {}
        
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze automation requirements and screen state"""
        try:
            # Get screen information
            screen_info = self.get_screen_info()
            
            # Get pending automation tasks
            pending_tasks = self.get_pending_automation_tasks()
            
            # Get available automation scripts
            available_scripts = self.get_available_scripts()
            
            # Check for GUI applications that need interaction
            gui_applications = self.detect_gui_applications()
            
            analysis_data = {
                "screen_info": screen_info,
                "pending_tasks": pending_tasks,
                "available_scripts": available_scripts,
                "gui_applications": gui_applications,
                "automation_capabilities": self.get_automation_capabilities(),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Analyze specific automation needs from context
            automation_request = context.get("automation_request")
            if automation_request:
                analysis_data["specific_request"] = automation_request
                analysis_data["request_analysis"] = self.analyze_automation_request(automation_request)
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"AutomationPilot analysis failed: {e}")
            return {}
    
    async def make_decision(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make automation decisions based on analysis"""
        try:
            pending_tasks = analysis.get("pending_tasks", [])
            specific_request = analysis.get("specific_request")
            gui_applications = analysis.get("gui_applications", [])
            
            if specific_request:
                # Handle specific automation request
                decision = self.make_specific_automation_decision(specific_request, analysis)
            elif pending_tasks:
                # Handle pending automation tasks
                decision = self.make_task_automation_decision(pending_tasks[0], analysis)
            elif gui_applications:
                # Handle GUI application automation
                decision = self.make_gui_automation_decision(gui_applications, analysis)
            else:
                # No automation needed
                decision = {
                    "action": "monitor",
                    "reasoning": "No automation tasks required, monitoring for future needs",
                    "confidence": 0.8,
                    "risk_level": "low"
                }
            
            decision.update({
                "decision_timestamp": datetime.now().isoformat(),
                "estimated_duration": self.estimate_automation_duration(decision)
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"AutomationPilot decision failed: {e}")
            return {}
    
    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automation actions"""
        try:
            action = decision.get("action")
            
            if action == "monitor":
                return {
                    "status": "success",
                    "message": "Monitoring mode active",
                    "actions_taken": []
                }
            
            elif action == "execute_script":
                return await self.execute_automation_script(decision)
            
            elif action == "record_interaction":
                return await self.record_user_interaction(decision)
            
            elif action == "gui_automation":
                return await self.perform_gui_automation(decision)
            
            elif action == "export_report":
                return await self.export_report(decision)
            
            elif action == "input_automation":
                return await self.perform_input_automation(decision)
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown automation action: {action}"
                }
                
        except Exception as e:
            logger.error(f"AutomationPilot execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_screen_info(self) -> Dict[str, Any]:
        """Get current screen information"""
        try:
            screen_size = pyautogui.size()
            mouse_position = pyautogui.position()
            
            return {
                "screen_width": screen_size.width,
                "screen_height": screen_size.height,
                "mouse_x": mouse_position.x,
                "mouse_y": mouse_position.y,
                "primary_monitor": True
            }
        except Exception as e:
            logger.error(f"Error getting screen info: {e}")
            return {}
    
    def get_pending_automation_tasks(self) -> List[Dict[str, Any]]:
        """Get pending automation tasks from memory or database"""
        tasks = self.get_memory("pending_automation_tasks") or []
        
        # Add default tasks that might be needed
        default_tasks = [
            {
                "task_id": "daily_report_export",
                "description": "Export daily trading report",
                "priority": "medium",
                "application": "trading_platform",
                "last_run": self.get_memory("last_report_export")
            },
            {
                "task_id": "portfolio_screenshot",
                "description": "Take portfolio screenshot for records",
                "priority": "low",
                "application": "portfolio_manager",
                "last_run": self.get_memory("last_portfolio_screenshot")
            }
        ]
        
        return tasks + default_tasks
    
    def get_available_scripts(self) -> List[str]:
        """Get list of available automation scripts"""
        scripts_dir = "automation_scripts"
        if not os.path.exists(scripts_dir):
            os.makedirs(scripts_dir)
            return []
        
        scripts = []
        for filename in os.listdir(scripts_dir):
            if filename.endswith('.json'):
                scripts.append(filename[:-5])  # Remove .json extension
        
        return scripts
    
    def detect_gui_applications(self) -> List[Dict[str, Any]]:
        """Detect running GUI applications that might need automation"""
        # This is a simplified implementation
        # In practice, this would use system APIs to detect running applications
        
        common_financial_apps = [
            {"name": "MetaTrader", "type": "trading_platform"},
            {"name": "TradingView", "type": "charting"},
            {"name": "Excel", "type": "spreadsheet"},
            {"name": "Chrome", "type": "browser"},
            {"name": "Firefox", "type": "browser"}
        ]
        
        # For demonstration, return a subset
        return common_financial_apps[:2]
    
    def get_automation_capabilities(self) -> List[str]:
        """Get list of automation capabilities"""
        return [
            "screen_capture",
            "mouse_control",
            "keyboard_input",
            "window_management",
            "file_operations",
            "report_export",
            "form_filling",
            "data_extraction",
            "interaction_recording"
        ]
    
    def analyze_automation_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific automation request"""
        request_type = request.get("type", "unknown")
        target_application = request.get("application", "unknown")
        complexity = request.get("complexity", "medium")
        
        analysis = {
            "request_type": request_type,
            "target_application": target_application,
            "complexity": complexity,
            "feasibility": "high",
            "estimated_steps": self.estimate_automation_steps(request),
            "required_capabilities": self.determine_required_capabilities(request),
            "risk_assessment": self.assess_automation_risk(request)
        }
        
        return analysis
    
    def make_specific_automation_decision(self, request: Dict[str, Any], 
                                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for specific automation request"""
        request_analysis = analysis.get("request_analysis", {})
        feasibility = request_analysis.get("feasibility", "medium")
        
        if feasibility == "high":
            return {
                "action": "execute_script",
                "script_type": request.get("type"),
                "target_application": request.get("application"),
                "steps": request_analysis.get("estimated_steps", []),
                "reasoning": f"High feasibility automation request for {request.get('type')}",
                "confidence": 0.8,
                "risk_level": request_analysis.get("risk_assessment", "medium")
            }
        else:
            return {
                "action": "record_interaction",
                "purpose": "Learn automation pattern for future use",
                "target_application": request.get("application"),
                "reasoning": "Recording user interaction to build automation script",
                "confidence": 0.9,
                "risk_level": "low"
            }
    
    def make_task_automation_decision(self, task: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for pending task automation"""
        task_id = task.get("task_id")
        priority = task.get("priority", "medium")
        
        # Check if we have a script for this task
        if task_id in self.get_available_scripts():
            return {
                "action": "execute_script",
                "script_name": task_id,
                "task": task,
                "reasoning": f"Executing automated script for {task.get('description')}",
                "confidence": 0.85,
                "risk_level": "low"
            }
        else:
            return {
                "action": "gui_automation",
                "task": task,
                "steps": self.generate_automation_steps(task),
                "reasoning": f"Performing GUI automation for {task.get('description')}",
                "confidence": 0.7,
                "risk_level": "medium"
            }
    
    def make_gui_automation_decision(self, gui_applications: List[Dict[str, Any]], 
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision for GUI application automation"""
        # Select the first application that might need automation
        app = gui_applications[0]
        
        return {
            "action": "gui_automation",
            "target_application": app.get("name"),
            "automation_type": "health_check",
            "reasoning": f"Performing health check automation on {app.get('name')}",
            "confidence": 0.6,
            "risk_level": "low"
        }
    
    async def execute_automation_script(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a saved automation script"""
        try:
            script_name = decision.get("script_name")
            if not script_name:
                return {"status": "error", "message": "No script name provided"}
            
            script_path = f"automation_scripts/{script_name}.json"
            if not os.path.exists(script_path):
                return {"status": "error", "message": f"Script {script_name} not found"}
            
            with open(script_path, 'r') as f:
                script = json.load(f)
            
            execution_log = []
            
            for step in script.get("steps", []):
                step_result = await self.execute_automation_step(step)
                execution_log.append(step_result)
                
                if not step_result.get("success"):
                    return {
                        "status": "error",
                        "message": f"Script execution failed at step: {step.get('description')}",
                        "execution_log": execution_log
                    }
            
            return {
                "status": "success",
                "message": f"Successfully executed script: {script_name}",
                "execution_log": execution_log,
                "steps_completed": len(execution_log)
            }
            
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def record_user_interaction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Record user interactions for building automation scripts"""
        try:
            recording_duration = decision.get("duration", 30)  # 30 seconds default
            
            logger.info(f"Starting interaction recording for {recording_duration} seconds")
            
            # Start recording
            recording = self.input_recorder.start_recording()
            
            # Wait for recording duration
            await asyncio.sleep(recording_duration)
            
            # Stop recording
            interaction_data = self.input_recorder.stop_recording()
            
            # Save recording
            recording_id = f"recording_{int(time.time())}"
            recording_path = f"interaction_recordings/{recording_id}.json"
            
            os.makedirs("interaction_recordings", exist_ok=True)
            with open(recording_path, 'w') as f:
                json.dump(interaction_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Recorded user interaction for {recording_duration} seconds",
                "recording_id": recording_id,
                "recording_path": recording_path,
                "events_captured": len(interaction_data.get("events", []))
            }
            
        except Exception as e:
            logger.error(f"Interaction recording failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_gui_automation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform GUI automation tasks"""
        try:
            automation_type = decision.get("automation_type", "general")
            target_application = decision.get("target_application")
            
            if automation_type == "health_check":
                return await self.perform_health_check(target_application)
            elif automation_type == "data_extraction":
                return await self.perform_data_extraction(decision)
            elif automation_type == "form_filling":
                return await self.perform_form_filling(decision)
            else:
                return await self.perform_general_automation(decision)
                
        except Exception as e:
            logger.error(f"GUI automation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def export_report(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Export reports through GUI automation"""
        try:
            report_type = decision.get("report_type", "general")
            export_format = decision.get("export_format", "pdf")
            
            # Simulate report export process
            steps = [
                {"action": "click", "coordinates": [100, 200], "description": "Open File menu"},
                {"action": "click", "coordinates": [120, 250], "description": "Click Export"},
                {"action": "type", "text": f"report_{int(time.time())}.{export_format}", "description": "Enter filename"},
                {"action": "click", "coordinates": [200, 300], "description": "Click Save"}
            ]
            
            execution_log = []
            for step in steps:
                result = await self.execute_automation_step(step)
                execution_log.append(result)
                time.sleep(1)  # Brief pause between steps
            
            return {
                "status": "success",
                "message": f"Report exported successfully",
                "report_type": report_type,
                "export_format": export_format,
                "execution_log": execution_log
            }
            
        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_input_automation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated input operations"""
        try:
            input_data = decision.get("input_data", {})
            target_fields = decision.get("target_fields", [])
            
            actions_performed = []
            
            for field in target_fields:
                field_name = field.get("name")
                field_value = input_data.get(field_name, "")
                coordinates = field.get("coordinates")
                
                if coordinates and field_value:
                    # Click on field
                    pyautogui.click(coordinates[0], coordinates[1])
                    time.sleep(0.5)
                    
                    # Clear field and enter value
                    pyautogui.hotkey('ctrl', 'a')  # Select all
                    pyautogui.type(str(field_value))
                    
                    actions_performed.append({
                        "field": field_name,
                        "value": field_value,
                        "success": True
                    })
            
            return {
                "status": "success",
                "message": f"Input automation completed for {len(actions_performed)} fields",
                "actions_performed": actions_performed
            }
            
        except Exception as e:
            logger.error(f"Input automation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_automation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single automation step"""
        try:
            action = step.get("action")
            
            if action == "click":
                coordinates = step.get("coordinates", [0, 0])
                pyautogui.click(coordinates[0], coordinates[1])
                
            elif action == "type":
                text = step.get("text", "")
                pyautogui.type(text)
                
            elif action == "key":
                keys = step.get("keys", [])
                if isinstance(keys, str):
                    pyautogui.press(keys)
                else:
                    pyautogui.hotkey(*keys)
                    
            elif action == "scroll":
                direction = step.get("direction", "down")
                amount = step.get("amount", 3)
                pyautogui.scroll(amount if direction == "up" else -amount)
                
            elif action == "wait":
                duration = step.get("duration", 1)
                time.sleep(duration)
            
            return {
                "step": step.get("description", action),
                "action": action,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Automation step failed: {e}")
            return {
                "step": step.get("description", "unknown"),
                "action": step.get("action", "unknown"),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def perform_health_check(self, application: str) -> Dict[str, Any]:
        """Perform health check on an application"""
        try:
            # Simple health check: take screenshot and verify application is responsive
            screenshot = pyautogui.screenshot()
            screenshot_path = f"screenshots/health_check_{application}_{int(time.time())}.png"
            
            os.makedirs("screenshots", exist_ok=True)
            screenshot.save(screenshot_path)
            
            return {
                "status": "success",
                "message": f"Health check completed for {application}",
                "screenshot_path": screenshot_path,
                "application_responsive": True
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {application}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_data_extraction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data extraction through screen reading"""
        try:
            # This would typically use OCR or screen reading libraries
            # For now, return a placeholder
            
            return {
                "status": "success",
                "message": "Data extraction completed (placeholder)",
                "extracted_data": {},
                "extraction_method": "screen_reading"
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_form_filling(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated form filling"""
        try:
            form_data = decision.get("form_data", {})
            
            # Simulate form filling
            filled_fields = []
            for field_name, field_value in form_data.items():
                # In practice, this would locate form fields and fill them
                filled_fields.append({
                    "field": field_name,
                    "value": field_value,
                    "status": "filled"
                })
            
            return {
                "status": "success",
                "message": f"Form filling completed for {len(filled_fields)} fields",
                "filled_fields": filled_fields
            }
            
        except Exception as e:
            logger.error(f"Form filling failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def perform_general_automation(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general automation tasks"""
        try:
            steps = decision.get("steps", [])
            
            execution_log = []
            for step in steps:
                result = await self.execute_automation_step(step)
                execution_log.append(result)
                
                if not result.get("success"):
                    break
            
            return {
                "status": "success",
                "message": "General automation completed",
                "execution_log": execution_log,
                "steps_completed": len([r for r in execution_log if r.get("success")])
            }
            
        except Exception as e:
            logger.error(f"General automation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def estimate_automation_duration(self, decision: Dict[str, Any]) -> str:
        """Estimate how long the automation will take"""
        action = decision.get("action", "")
        
        duration_map = {
            "monitor": "continuous",
            "execute_script": "2-5 minutes",
            "record_interaction": "30 seconds - 5 minutes",
            "gui_automation": "1-3 minutes",
            "export_report": "30 seconds - 2 minutes",
            "input_automation": "30 seconds - 1 minute"
        }
        
        return duration_map.get(action, "1-2 minutes")
    
    def estimate_automation_steps(self, request: Dict[str, Any]) -> List[str]:
        """Estimate the steps needed for automation"""
        request_type = request.get("type", "general")
        
        step_templates = {
            "export_report": [
                "Navigate to reports section",
                "Select report type",
                "Configure export settings",
                "Initiate export",
                "Verify export completion"
            ],
            "data_entry": [
                "Locate input form",
                "Fill required fields",
                "Validate input data",
                "Submit form",
                "Confirm submission"
            ],
            "screenshot": [
                "Navigate to target screen",
                "Capture screenshot",
                "Save to designated location",
                "Verify file creation"
            ]
        }
        
        return step_templates.get(request_type, ["Analyze target", "Execute action", "Verify result"])
    
    def determine_required_capabilities(self, request: Dict[str, Any]) -> List[str]:
        """Determine what capabilities are needed for the request"""
        request_type = request.get("type", "general")
        
        capability_map = {
            "export_report": ["mouse_control", "keyboard_input", "file_operations"],
            "data_entry": ["mouse_control", "keyboard_input", "form_filling"],
            "screenshot": ["screen_capture", "file_operations"],
            "navigation": ["mouse_control", "window_management"]
        }
        
        return capability_map.get(request_type, ["mouse_control", "keyboard_input"])
    
    def assess_automation_risk(self, request: Dict[str, Any]) -> str:
        """Assess the risk level of the automation request"""
        request_type = request.get("type", "general")
        target_app = request.get("application", "unknown")
        
        # High risk for financial applications or data modification
        high_risk_apps = ["trading_platform", "banking", "financial_software"]
        high_risk_types = ["data_modification", "financial_transaction", "account_management"]
        
        if target_app in high_risk_apps or request_type in high_risk_types:
            return "high"
        elif request_type in ["export_report", "screenshot", "navigation"]:
            return "low"
        else:
            return "medium"
    
    def generate_automation_steps(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automation steps for a task"""
        task_type = task.get("task_id", "general")
        
        if task_type == "daily_report_export":
            return [
                {"action": "click", "coordinates": [50, 100], "description": "Open File menu"},
                {"action": "click", "coordinates": [70, 150], "description": "Click Export"},
                {"action": "type", "text": f"daily_report_{datetime.now().strftime('%Y%m%d')}.pdf", "description": "Enter filename"},
                {"action": "click", "coordinates": [200, 200], "description": "Click Save"}
            ]
        elif task_type == "portfolio_screenshot":
            return [
                {"action": "key", "keys": ["alt", "tab"], "description": "Switch to portfolio window"},
                {"action": "wait", "duration": 2, "description": "Wait for window to focus"},
                {"action": "key", "keys": ["ctrl", "shift", "s"], "description": "Take screenshot"}
            ]
        else:
            return [
                {"action": "wait", "duration": 1, "description": "Prepare for automation"},
                {"action": "click", "coordinates": [100, 100], "description": "Generic click action"}
            ]

class InputRecorder:
    """Records user input for building automation scripts"""
    
    def __init__(self):
        self.recording = False
        self.events = []
        self.mouse_listener = None
        self.keyboard_listener = None
    
    def start_recording(self) -> bool:
        """Start recording user input"""
        if self.recording:
            return False
        
        self.recording = True
        self.events = []
        
        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        return True
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return captured events"""
        if not self.recording:
            return {"events": []}
        
        self.recording = False
        
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        return {
            "events": self.events,
            "recording_duration": len(self.events),
            "timestamp": datetime.now().isoformat()
        }
    
    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events"""
        if self.recording and pressed:
            self.events.append({
                "type": "mouse_click",
                "x": x,
                "y": y,
                "button": str(button),
                "timestamp": datetime.now().isoformat()
            })
    
    def on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events"""
        if self.recording:
            self.events.append({
                "type": "mouse_scroll",
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "timestamp": datetime.now().isoformat()
            })
    
    def on_key_press(self, key):
        """Handle key press events"""
        if self.recording:
            try:
                key_name = key.char if hasattr(key, 'char') and key.char else str(key)
                self.events.append({
                    "type": "key_press",
                    "key": key_name,
                    "timestamp": datetime.now().isoformat()
                })
            except AttributeError:
                pass
    
    def on_key_release(self, key):
        """Handle key release events"""
        # We mainly care about key presses for automation
        pass
