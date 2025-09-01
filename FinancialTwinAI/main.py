#!/usr/bin/env python3
"""
Digital Twin LAM for Financial Engineering
Main entry point for the multi-agent financial system
"""

import asyncio
import os
import threading
import time
from datetime import datetime

from core.orchestrator import AgentOrchestrator
from core.database import DatabaseManager
from web.app import create_app
from scheduler import WeeklyScheduler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_lam.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_web_server():
    """Run the Flask web interface"""
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

async def main():
    """Main application entry point"""
    logger.info("Starting Digital Twin LAM Financial Engineering System")
    
    # Initialize database
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    
    # Initialize scheduler
    scheduler = WeeklyScheduler(orchestrator)
    
    # Start web server in separate thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    logger.info("Web interface started on http://0.0.0.0:5000")
    
    # Start scheduler
    scheduler_thread = threading.Thread(target=scheduler.run, daemon=True)
    scheduler_thread.start()
    logger.info("Weekly scheduler started")
    
    logger.info("System ready. Press Ctrl+C to stop.")
    
    try:
        # Keep main thread alive
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler.stop()
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
