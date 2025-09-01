# Digital Twin LAM - Financial Engineering System

## Overview

The Digital Twin LAM (Large Action Model) is a multi-agent financial engineering system that combines AI reasoning with automated action execution. The system orchestrates specialized agents to handle trading, accounting, financial planning, and reinvestment strategies. Built with Python and SQLite, it features both automated execution and web-based monitoring capabilities.

## System Architecture

### Core Architecture Pattern
- **Multi-Agent System**: Specialized agents for different financial domains
- **Event-Driven Orchestration**: Central orchestrator manages agent coordination
- **Local-First Design**: SQLite database with optional external API integration
- **Trace-Based Learning**: Decision logging for future LAM training
- **Web Interface**: Flask-based monitoring and control dashboard

### Technology Stack
- **Backend**: Python 3.x with Flask web framework
- **Database**: SQLite for local data persistence
- **LLM Integration**: Ollama for local AI reasoning (Mistral model)
- **GUI Automation**: pyautogui and pynput for screen interaction
- **Trading**: Mock broker implementation with real API extensibility
- **Scheduling**: Built-in weekly execution scheduler

## Key Components

### Agent System
The system implements six specialized agents:

1. **Finance Planner Agent**: Creates weekly financial plans based on balance, risk models, and market events
2. **Trading Strategist Agent**: Analyzes market data and executes trades through APIs or mock broker
3. **Accounting Clerk Agent**: Manages bookkeeping, expense categorization, and portfolio tracking
4. **Automation Pilot Agent**: Handles GUI interactions when APIs are unavailable
5. **Reinvestor Agent**: Optimizes idle cash deployment through yield strategies and DCA
6. **DeFi Analyst Agent**: Analyzes on-chain whale movements, protocol opportunities, and DeFi arbitrage for optimal capital allocation

### Core Services
- **Agent Orchestrator**: Manages multi-agent coordination and execution workflows
- **Database Manager**: Handles SQLite operations for transactions, decisions, and traces
- **LLM Client**: Interfaces with Ollama for local AI reasoning
- **Trace Logger**: Records decision processes for LAM training data collection
- **Market Data Provider**: Simulates market feeds for various financial instruments
- **Mock Broker**: Paper trading implementation with order management

### Web Dashboard
- **Real-time Monitoring**: Live dashboard showing system status and performance
- **Agent Management**: Interface for controlling and monitoring individual agents
- **Decision Traces**: Visualization of agent reasoning and decision history
- **Portfolio Tracking**: Account balance, positions, and transaction history

## Data Flow

### Weekly Execution Pipeline
1. **Scheduler** triggers weekly execution on configured day/time
2. **Orchestrator** coordinates agent execution based on dependencies
3. **Finance Planner** analyzes current financial state and creates plans
4. **Trading Strategist** evaluates market opportunities and executes trades
5. **Accounting Clerk** logs transactions and updates portfolio records
6. **Reinvestor** optimizes cash allocation and yield strategies
7. **Automation Pilot** handles any required GUI interactions

### Decision Tracing
- Each agent decision is logged with context, reasoning, and outcomes
- Traces include confidence scores and performance metrics
- Data structured for future LAM training and model improvement

### Database Schema
- **agent_decisions**: Agent choices with confidence and execution status
- **transactions**: Financial transaction records
- **agent_memory**: Persistent agent memory storage
- **decision_traces**: Detailed reasoning logs for LAM training

## External Dependencies

### Required Services
- **Ollama**: Local LLM server for AI reasoning (http://localhost:11434)
- **SQLite**: Embedded database (no external setup required)

### Optional Integrations
- **Alpha Vantage API**: Market data (requires API key)
- **Binance API**: Cryptocurrency trading (requires API keys)
- **GUI Applications**: For automation pilot interactions

### Python Dependencies
- Flask (web framework)
- SQLite3 (database)
- requests (HTTP client)
- pyautogui (GUI automation)
- pynput (input capture)
- pandas (data processing)
- schedule (task scheduling)

## Deployment Strategy

### Local Development
- Single-machine deployment with SQLite database
- Ollama server running locally for LLM inference
- Flask development server for web interface
- File-based logging for debugging and monitoring

### Production Considerations
- The system is designed for local/personal use
- Can be extended to support cloud deployment
- Database migrations supported through DatabaseManager
- Configuration via environment variables and config.py

### Scaling Options
- Agent system supports parallel execution
- Database can be migrated to PostgreSQL for multi-user scenarios
- Web interface supports containerization
- Trace logging can be extended to cloud storage

## Changelog
- August 15, 2025. **DeFi Analytics Integration**: Added comprehensive DeFi analytics capabilities to Digital Twin LAM system
  - Integrated DeFiAnalyst agent for on-chain whale tracking and protocol analysis
  - Added whale transaction monitoring with behavioral pattern recognition
  - Created protocol analyzer for cross-chain yield optimization and arbitrage detection
  - Built DeFi dashboard with real-time analytics and interactive controls
  - Enhanced LAM training data collection with DeFi behavioral patterns
  - System now supports both TradFi and DeFi decision-making for comprehensive financial AI
- June 29, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.