#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SOL Trading Example - Refactored

This example demonstrates using the event-driven architecture to implement
a trading system using SOL as capital. It now imports core components
from the src directory.

The example shows:
- How to configure and initialize components imported from the core library.
- How to connect components using the event bus.
- Running the system with either mock or real WebSocket data.
"""

import asyncio
import logging
import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import signal

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dotenv for environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file.")
except ImportError:
    print("dotenv package not found. Skipping .env file loading.")

# --- Core Imports ---
from src.core.events import EventBus, EventType # Minimal core imports needed at top level
from src.core.features import FeatureSystem
from src.core.features.providers import EnhancedPriceProvider # Using enhanced provider
from src.core.trading.trading_factory import TradingSystemFactory

# --- Moved Class Imports ---
from src.core.data.data_feed_manager import DataFeedManager
from src.core.data.websocket_clients import RealWebSocketClient, MockWebSocketClient
from Archive.database_client import DatabaseClient
# TradingEngine and PortfolioManager are usually created via the factory, but import if needed directly
# from src.core.trading.trading_engine import TradingEngine
# from src.core.trading.portfolio_manager import PortfolioManager

# --- Logging Setup ---
log_filename = f"sol_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO, # Default level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
root_logger = logging.getLogger()
logger = logging.getLogger("sol_trading_example") # Logger for this specific script
logger.info(f"Logging to console and {log_filename}")

# Suppress noisy external logs
logging.getLogger('websockets').setLevel(logging.ERROR)

# --- Global Shutdown Signal ---
shutdown_requested = False
# Apply the flag to the websocket module (Workaround - TODO: Refactor)
try:
    import src.core.data.websocket_clients
    src.core.data.websocket_clients.shutdown_requested = shutdown_requested
except ImportError as e:
     logger.error(f"Could not import websocket_clients to set shutdown flag: {e}")

# --- Signal Handler ---
def signal_handler(sig, frame):
    """Handle signal interrupts."""
    global shutdown_requested
    if not shutdown_requested:
        signal_name = signal.Signals(sig).name
        logger.info(f"Shutdown signal {signal_name} received. Initiating graceful shutdown...")
        shutdown_requested = True
        # Update the flag in the websocket module as well
        if 'src.core.data.websocket_clients' in sys.modules:
            src.core.data.websocket_clients.shutdown_requested = True
    else:
        logger.warning("Shutdown already in progress.")

# Register signal handlers
for sig_name in ('SIGINT', 'SIGTERM'):
    if hasattr(signal, sig_name):
        try:
            sig_enum = getattr(signal, sig_name)
            signal.signal(sig_enum, signal_handler)
            logger.debug(f"Registered signal handler for {sig_name}")
        except (ValueError, OSError, AttributeError) as e:
            logger.warning(f"Could not register signal handler for {sig_name}: {e}")

# --- Helper Functions ---
def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        config_path = Path(config_file)
        # If the direct path doesn't exist, check relative to the script's directory
        if not config_path.is_file():
             script_dir = Path(__file__).parent
             config_path = script_dir / config_file
             if not config_path.is_file():
                  logger.error(f"Configuration file not found at {config_file} or {config_path}")
                  return {}
             else:
                  logger.info(f"Using configuration file from script directory: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {config_path}. Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}", exc_info=True)
        return {}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run SOL Trading System Example (Refactored)')
    parser.add_argument('--use-mock', action='store_true',
                        help='Use mock data client instead of real WebSocket')
    parser.add_argument('--websocket-url', type=str, default=None,
                        help='WebSocket URL for real-time data (e.g., wss://pumpportal.fun/api/data)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for data source (if required)')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging level (Default: INFO)')
    parser.add_argument('--db-connection', type=str, default=None,
                        help='Database connection string (currently mock implementation)')
    parser.add_argument('--config', type=str, default='examples/real_config.json',
                        help='Path to JSON configuration file (Default: examples/real_config.json)')
    parser.add_argument('--debug-mode', action='store_true',
                        help='Enable detailed debugging output')
    return parser.parse_args()

async def main():
    """Main async function to configure and run the trading system."""
    global shutdown_requested # Allow modification in case of early exit
    args = parse_args()
    config = load_config(args.config)

    # --- Determine Effective Configuration ---
    log_level_str = args.log_level or os.environ.get('TRADING_LOG_LEVEL') or config.get('log_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    debug_mode_arg = args.debug_mode
    debug_mode_env = os.environ.get('TRADING_DEBUG_MODE', 'false').lower() == 'true'
    debug_mode_conf = config.get('debug_mode', False)
    debug_mode = debug_mode_arg or debug_mode_env or debug_mode_conf

    # Apply log level (force DEBUG if debug_mode is True)
    effective_log_level = logging.DEBUG if debug_mode else log_level
    root_logger.setLevel(effective_log_level)
    for handler in root_logger.handlers: handler.setLevel(effective_log_level)
    logger.setLevel(effective_log_level)
    logger.info(f"Effective log level set to: {logging.getLevelName(effective_log_level)}")
    if debug_mode: logger.info("Debug mode is ON")

    websocket_uri = args.websocket_url or os.environ.get('TRADING_WEBSOCKET_URL') or config.get('websocket_uri') or config.get('websocket_url')
    api_key = args.api_key or os.environ.get('TRADING_API_KEY') or config.get('api_key')
    db_connection = args.db_connection or os.environ.get('TRADING_DB_CONNECTION') or config.get('db_connection')

    # Validate required settings for real mode
    if not args.use_mock and not websocket_uri:
        logger.critical("WebSocket URI is required for real mode. Set via --websocket-url, config, or env var.")
        return 1

    # Extract subscription keys (combining config and env)
    watched_accounts_env = [a for a in os.environ.get('TRADING_WATCHED_ACCOUNTS', '').split(',') if a]
    watched_tokens_env = [t for t in os.environ.get('TRADING_WATCHED_TOKENS', '').split(',') if t]
    subscription_keys = {
        'accounts': list(set(config.get('watched_accounts', []) + watched_accounts_env)),
        'tokens': list(set(config.get('watched_tokens', []) + watched_tokens_env))
    }
    enable_dynamic_discovery = config.get('enable_dynamic_token_discovery', False)

    # Log final configuration summary
    logger.info("--- System Configuration ---")
    logger.info(f"  Mode:          {'Mock' if args.use_mock else 'Real WebSocket'}")
    if not args.use_mock: logger.info(f"  WebSocket URI: {websocket_uri}")
    logger.info(f"  API Key:       {'Provided' if api_key else 'Not Provided'}")
    logger.info(f"  Watched Accts: {len(subscription_keys['accounts'])}")
    logger.info(f"  Watched Tokens: {len(subscription_keys['tokens'])}")
    logger.info(f"  Dynamic Disc:  {enable_dynamic_discovery}")
    logger.info(f"  DB Connection: {'Provided' if db_connection else 'Not Provided'}")
    logger.info(f"  Log Level:     {logging.getLevelName(effective_log_level)}")
    logger.info(f"  Debug Mode:    {debug_mode}")
    logger.info("--------------------------")

    # --- Initialize Core Components ---
    trading_system_components = None
    data_client = None
    db_client = None
    event_bus = None

    try:
        event_bus = EventBus(async_processing=True)
        event_bus.start_processing()
        logger.info("Event Bus initialized and started.")

        feature_system = FeatureSystem(event_bus=event_bus)
        price_provider = EnhancedPriceProvider(name='price_provider', max_history=500)
        feature_system.register_provider(price_provider)
        logger.info("Feature System and Price Provider initialized.")

        data_feed_manager = DataFeedManager(event_bus)
        logger.info("Data Feed Manager initialized.")

        if args.use_mock:
            logger.info("Initializing MockWebSocketClient...")
            data_client = MockWebSocketClient(
                event_bus=event_bus,
                data_feed_manager=data_feed_manager,
                feature_system=feature_system
            )
        else:
            logger.info("Initializing RealWebSocketClient...")
            data_client = RealWebSocketClient(
                event_bus=event_bus,
                websocket_uri=websocket_uri,
                data_feed_manager=data_feed_manager,
                api_key=api_key,
                subscription_keys=subscription_keys,
                enable_dynamic_token_discovery=enable_dynamic_discovery,
                debug_mode=debug_mode,
                heartbeat_interval=config.get('heartbeat_interval', 30)
            )

        if db_connection:
            logger.info("Initializing DatabaseClient...")
            db_client = DatabaseClient(event_bus, db_connection)
            # await db_client.connect() # Connect explicitly if needed at startup

        # --- Define Price Fetcher ---
        async def price_fetcher(token_id: str) -> Optional[float]:
            """Fetch price from DFM, fallback to Price Provider."""
            token_data = data_feed_manager.get_token_data(token_id)
            price = token_data.get('price')
            if price is not None: return float(price)

            latest_price_info = price_provider.get_latest_price(token_id)
            if latest_price_info: return latest_price_info['price']

            logger.warning(f"Price fetcher could not find price for {token_id}")
            return None

        # --- Create Trading System via Factory ---
        trading_config = config.get('trading', {}) # Get trading sub-config
        factory_config = {
            'event_bus': event_bus,
            'feature_system': feature_system,
            'data_feed_manager': data_feed_manager,
            'price_fetcher': price_fetcher,
            'initial_capital': config.get('initial_capital', 1000.0), # Get capital from main config or default
            'subscription_keys': subscription_keys,
            # Pass specific trading settings from config
            **trading_config
        }

        logger.info("Creating trading system components via TradingSystemFactory...")
        trading_system_components = TradingSystemFactory.create_complete_trading_system(
            event_bus=event_bus,
            config=factory_config
        )
        logger.info("Trading system components created successfully.")

        # Extract components for logging/verification (optional)
        portfolio_manager = trading_system_components.get('portfolio_manager')
        if portfolio_manager: logger.info(f"Portfolio Manager ready. Initial Capital: {portfolio_manager.initial_capital:.2f}")
        if trading_system_components.get('graceful_exit_manager'): logger.info("Graceful Exit Manager ready.")
        if trading_system_components.get('token_monitor_pool'): logger.info("Token Monitor Pool ready.")


        # --- Start Data Streaming ---
        logger.info("Starting data client streaming...")
        streaming_started = await data_client.start_streaming()
        if not streaming_started:
            raise RuntimeError("Data streaming failed to start. Check connection and logs.")
        logger.info("Data streaming active. System is running.")

        # --- Keep Main Task Alive ---
        while not shutdown_requested:
            await asyncio.sleep(1) # Check for shutdown request every second

        logger.info("Shutdown detected in main loop. Proceeding to cleanup...")

    except Exception as e:
        logger.critical(f"Critical error during system setup or main loop: {e}", exc_info=True)
        # Ensure shutdown is requested on error
        if not shutdown_requested:
            shutdown_requested = True
            if 'src.core.data.websocket_clients' in sys.modules:
                 src.core.data.websocket_clients.shutdown_requested = True
        return 1 # Indicate error
    finally:
        # --- Graceful Shutdown Sequence ---
        logger.info("--- Initiating System Shutdown ---")

        # 1. Stop Data Client (stops new events)
        if data_client and hasattr(data_client, 'stop_streaming'):
            logger.info("Stopping data client...")
            await data_client.stop_streaming()
            logger.info("Data client stopped.")

        # 2. Shutdown Trading System Components (allow graceful exit, etc.)
        if trading_system_components:
            logger.info("Shutting down trading system components...")
            if hasattr(TradingSystemFactory, 'shutdown_trading_system'):
                await TradingSystemFactory.shutdown_trading_system(trading_system_components) # Assume it might be async
            else:
                 logger.warning("TradingSystemFactory lacks shutdown method - manual component shutdown might be needed.")
            logger.info("Trading system components shut down.")

            # Log final portfolio state
            portfolio_manager = trading_system_components.get('portfolio_manager')
            if portfolio_manager and hasattr(portfolio_manager, 'log_portfolio_summary'):
                 logger.info("--- Final Portfolio ---")
                 portfolio_manager.log_portfolio_summary(level=logging.INFO)
                 logger.info("-----------------------")

        # 3. Stop Event Bus (process remaining events)
        if event_bus and hasattr(event_bus, 'stop_processing'):
            logger.info("Stopping event bus...")
            event_bus.stop_processing() # Should ideally wait for queue to empty
            logger.info("Event bus stopped.")

        # 4. Disconnect Database
        if db_client and hasattr(db_client, 'disconnect'):
            logger.info("Disconnecting database client...")
            await db_client.disconnect()
            logger.info("Database client disconnected.")

        logger.info("--- System Shutdown Complete ---")

    return 0 # Indicate success

# --- Script Entry Point ---
if __name__ == "__main__":
    exit_code = 1
    loop = None
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        exit_code = loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Main process interrupted. Shutdown handled in main().")
    except Exception as e:
        logger.critical(f"Fatal error at script level: {e}", exc_info=True)
    finally:
        logger.info(f"Exiting script with code {exit_code}.")
        sys.exit(exit_code)
