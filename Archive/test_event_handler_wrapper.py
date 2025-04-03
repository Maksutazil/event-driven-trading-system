#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the standardized EventHandlerWrapper correctly handles events and errors.
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
logger = logging.getLogger("test_event_handler_wrapper")

from src.core.events import Event, EventType, EventBus
from src.core.events.base import EventHandlerWrapper

def test_event_handler_wrapper():
    """Test that EventHandlerWrapper correctly handles events and errors."""
    logger.info("Starting EventHandlerWrapper test")
    
    # Create EventBus
    event_bus = EventBus(async_processing=False)  # Synchronous for easier testing
    
    # Test counter to verify callbacks are being called
    test_results = {'normal_callback_called': 0, 'error_callback_called': 0}
    
    # Create a normal callback function
    def normal_callback(event):
        logger.info(f"Normal callback called with event: {event}")
        test_results['normal_callback_called'] += 1
    
    # Create a callback that raises an exception
    def error_callback(event):
        logger.info(f"Error callback called with event: {event}")
        test_results['error_callback_called'] += 1
        raise ValueError("Test exception from error_callback")
    
    # Create EventHandlerWrapper instances
    normal_handler = EventHandlerWrapper(normal_callback, name="test_normal_handler")
    error_handler = EventHandlerWrapper(error_callback, name="test_error_handler")
    
    # Subscribe handlers to the event bus
    event_bus.subscribe(EventType.GENERIC, normal_handler)
    event_bus.subscribe(EventType.GENERIC, error_handler)
    
    # Create and publish a test event
    test_event = Event(
        event_type=EventType.GENERIC,
        data={'test_key': 'test_value'},
        source='test_script'
    )
    
    logger.info("Publishing test event")
    event_bus.publish(test_event)
    
    # Verify results
    if test_results['normal_callback_called'] != 1:
        logger.error(f"Normal callback was not called exactly once: {test_results['normal_callback_called']}")
        return False
    
    if test_results['error_callback_called'] != 1:
        logger.error(f"Error callback was not called exactly once: {test_results['error_callback_called']}")
        return False
    
    logger.info("Test completed successfully - EventHandlerWrapper handled normal and error cases correctly")
    return True

if __name__ == "__main__":
    success = test_event_handler_wrapper()
    sys.exit(0 if success else 1) 