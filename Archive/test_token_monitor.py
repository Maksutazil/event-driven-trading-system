#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the TradingSystemFactory handles a None token_monitor during shutdown.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_token_monitor")

from src.core.events import EventBus
from src.core.trading.trading_factory import TradingSystemFactory

def test_token_monitor_none_handling():
    """Test that TradingSystemFactory correctly handles a None token_monitor during shutdown."""
    logger.info("Starting token_monitor=None handling test")
    
    # Create a trading system with None token_monitor
    trading_system = {
        'token_monitor': None,
        # Add some dummy components for testing
        'trading_engine': object(),
    }
    
    try:
        # Call shutdown_trading_system
        logger.info("Shutting down trading system with token_monitor=None")
        TradingSystemFactory.shutdown_trading_system(trading_system)
        logger.info("Shutdown completed without errors")
        
        # If we got here, no exception was thrown
        logger.info("Test completed successfully - token_monitor=None was handled correctly")
        return True
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_token_monitor_none_handling()
    sys.exit(0 if success else 1) 