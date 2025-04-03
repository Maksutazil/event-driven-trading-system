#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the ActivityAnalyzer's handling of SYSTEM events.
"""

import logging
import sys
import time
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
logger = logging.getLogger("test_activity_analyzer")

from src.core.events import Event, EventType, EventBus
from src.core.trading.activity_analyzer import ActivityAnalyzer

def test_system_event_handling():
    """Test that ActivityAnalyzer can handle SYSTEM events properly."""
    logger.info("Starting ActivityAnalyzer SYSTEM event handling test")
    
    # Create EventBus and ActivityAnalyzer
    event_bus = EventBus()
    analyzer = ActivityAnalyzer(event_bus)
    analyzer.start()
    
    # Create a SYSTEM event with 'shutdown' action
    event = Event(
        event_type=EventType.SYSTEM,
        data={'action': 'shutdown'},
        source='test_script'
    )
    
    # Handle the SYSTEM event
    try:
        logger.info("Sending SYSTEM event with 'shutdown' action")
        analyzer._handle_system_event(event)
        logger.info("SYSTEM event handled successfully")
        
        # Wait a moment for any async actions to complete
        time.sleep(1)
        
        # Clean up
        analyzer.cleanup()
        logger.info("Test completed successfully - ActivityAnalyzer handled SYSTEM event")
        return True
    except Exception as e:
        logger.error(f"Error handling SYSTEM event: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_system_event_handling()
    sys.exit(0 if success else 1) 