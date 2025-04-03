#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for running a trading system with historical data from PostgreSQL.

This script connects to a PostgreSQL database, loads historical trade data,
and runs the trading system using this data.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup path for local imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")

from dotenv import load_dotenv

# Import core modules
try:
    from src.core.db import PostgresDataManager
    from src.core.data import DataFeedManager
    from src.core.events import EventBus, EventType, EventHandlerWrapper
    from src.core.features import FeatureSystem
    from src.core.features.providers import EnhancedPriceProvider
    from src.core.trading import TradingSystemFactory
    print("Successfully imported all required modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nCurrent sys.path:")
    for p in sys.path:
        print(f"  - {p}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_historical.log', mode='w')
    ]
)
logger = logging.getLogger("historical_trading")

# Global shutdown signal
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info(f"Received shutdown signal: {sig}")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main async function to configure and run the trading system."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run trading system with historical data from PostgreSQL.')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--debug-mode', action='store_true', help='Enable debug mode')
    parser.add_argument('--use-mock', action='store_true', help='Use mock data instead of real data')
    parser.add_argument('--token-ids', type=str, help='Comma-separated list of token IDs to stream')
    parser.add_argument('--days', type=int, default=7, help='Number of days of historical data to load')
    parser.add_argument('--stream-delay', type=int, default=100, 
                       help='Delay between events in milliseconds (lower = faster simulation)')
    args = parser.parse_args()
    
    # Set logging level based on debug mode
    if args.debug_mode:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)
    
    # Load environment variables from .env file if it exists
    env_path = Path('.') / '.env'
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Get database connection parameters from environment or config
    db_host = os.getenv('DB_HOST', config.get('db_host', 'localhost'))
    db_port = os.getenv('DB_PORT', config.get('db_port', 5432))
    db_name = os.getenv('DB_NAME', config.get('db_name', 'postgres'))
    db_user = os.getenv('DB_USER', config.get('db_user', 'postgres'))
    db_password = os.getenv('DB_PASSWORD', config.get('db_password', 'postgres'))
    
    # Store connection string securely (without logging credentials)
    db_connection_params = {
        'host': db_host,
        'port': db_port,
        'dbname': db_name,
        'user': db_user,
        'password': db_password
    }
    
    # Set up date range for historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)
    
    # Parse token IDs if provided
    token_ids = None
    if args.token_ids:
        token_ids = [tid.strip() for tid in args.token_ids.split(',') if tid.strip()]
        logger.info(f"Processing data for specified tokens: {token_ids}")
    
    # Setup components for the trading system
    trading_system_components = None
    data_client = None
    db_client = None
    
    try:
        # Initialize event bus
        event_bus = EventBus(async_processing=True)
        event_bus.start_processing()
        logger.info("Event Bus initialized and started.")
        
        # Initialize feature system and price provider
        feature_system = FeatureSystem(event_bus=event_bus)
        price_provider = EnhancedPriceProvider(name='price_provider', max_history=500)
        feature_system.register_provider(price_provider)
        logger.info("Feature System and Price Provider initialized.")
        
        # Initialize data feed manager
        data_feed_manager = DataFeedManager(event_bus)
        logger.info("Data Feed Manager initialized.")
        
        # Initialize PostgreSQL data manager
        logger.info("Initializing PostgresDataManager...")
        db_client = PostgresDataManager(
            event_bus=event_bus,
            data_feed_manager=data_feed_manager,
            connection_params=db_connection_params,
            batch_size=100,
            streaming_delay_ms=args.stream_delay,
            debug_mode=args.debug_mode
        )
        
        # Connect to the database
        connect_success = await db_client.connect()
        if not connect_success:
            logger.error("Failed to connect to PostgreSQL database. Exiting.")
            return 1
        
        logger.info(f"Successfully connected to PostgreSQL database")
        
        # Create trading system components
        logger.info("Creating trading system components...")
        trading_system_components = TradingSystemFactory.create_complete_trading_system(
            event_bus=event_bus,
            config={
                'feature_system': feature_system,
                'data_feed_manager': data_feed_manager,
                'debug_mode': args.debug_mode
            }
        )
        
        if not trading_system_components:
            logger.error("Failed to create trading system components")
            return 1
            
        logger.info("Trading system components created successfully")
        
        # Start streaming historical data
        logger.info(f"Starting historical data streaming (date range: {start_time} to {end_time})...")
        stream_success = await db_client.start_streaming(
            token_ids=token_ids,
            start_time=start_time,
            end_time=end_time
        )
        
        if not stream_success:
            logger.error("Failed to start historical data streaming")
            return 1
            
        logger.info("Historical data streaming started successfully")
        
        # Main loop - keep running until shutdown is requested
        logger.info("Entering main loop. Press Ctrl+C to stop.")
        while not shutdown_requested:
            await asyncio.sleep(1.0)
            
            # Check if streaming is still active
            if not db_client.streaming_active:
                logger.info("Historical data streaming has completed")
                break
        
        logger.info("Main loop exited")
        return 0
        
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        return 1
        
    finally:
        # --- Graceful Shutdown Sequence ---
        logger.info("--- Initiating System Shutdown ---")
        
        # 1. Stop Database Client (stops new events)
        if db_client:
            logger.info("Stopping database client...")
            if hasattr(db_client, 'stop_streaming'):
                await db_client.stop_streaming()
            logger.info("Database client stopped.")
        
        # 2. Shutdown Trading System Components
        if trading_system_components:
            logger.info("Shutting down trading system components...")
            if hasattr(TradingSystemFactory, 'shutdown_trading_system'):
                await TradingSystemFactory.shutdown_trading_system(trading_system_components)
            logger.info("Trading system components shut down.")
            
            # Log final portfolio state
            portfolio_manager = trading_system_components.get('portfolio_manager')
            if portfolio_manager and hasattr(portfolio_manager, 'log_portfolio_summary'):
                logger.info("--- Final Portfolio ---")
                portfolio_manager.log_portfolio_summary(level=logging.INFO)
                logger.info("-----------------------")
        
        # 3. Stop Event Bus
        if 'event_bus' in locals() and event_bus:
            logger.info("Stopping event bus...")
            event_bus.stop_processing()
            logger.info("Event bus stopped.")
        
        # 4. Disconnect Database
        if db_client and hasattr(db_client, 'disconnect'):
            logger.info("Disconnecting database client...")
            await db_client.disconnect()
            logger.info("Database client disconnected.")
        
        logger.info("--- System Shutdown Complete ---")

if __name__ == "__main__":
    return_code = asyncio.run(main())
    sys.exit(return_code) 