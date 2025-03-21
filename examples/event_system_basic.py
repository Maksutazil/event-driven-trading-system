#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Event System Example

This example demonstrates the core functionality of the event system:
- Creating and starting an event bus
- Creating a subscriber to handle events
- Publishing events of different types
- Using callbacks for event handling
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any

# Add project root to Python path
import os
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.events.event_bus import EventBus
from src.core.events.event import Event, EventType, EventPriority
from src.core.events.base import BaseEventSubscriber, BaseEventPublisher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleEventHandler(BaseEventSubscriber):
    """
    A simple event subscriber that handles multiple event types.
    """
    def __init__(self, event_bus: EventBus):
        # Subscribe to multiple event types
        super().__init__(event_bus, {
            EventType.SYSTEM,
            EventType.DATA_NEW,
            EventType.DATA_UPDATE,
        })
        self.event_count = 0
        logger.info("SimpleEventHandler initialized and subscribed to events")
    
    def handle_system(self, event: Event) -> None:
        """Handle system events."""
        self.event_count += 1
        logger.info(f"SYSTEM Event #{self.event_count}: {event.data}")
    
    def handle_data_new(self, event: Event) -> None:
        """Handle new data events."""
        self.event_count += 1
        logger.info(f"DATA_NEW Event #{self.event_count}: {event.data}")
    
    def handle_data_update(self, event: Event) -> None:
        """Handle data update events."""
        self.event_count += 1
        logger.info(f"DATA_UPDATE Event #{self.event_count}: {event.data}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "event_count": self.event_count
        }

class SimpleEventPublisher(BaseEventPublisher):
    """
    A simple event publisher that publishes different types of events.
    """
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus, source="simple_publisher")
        logger.info("SimpleEventPublisher initialized")
    
    def publish_system_event(self, message: str, priority: EventPriority = EventPriority.NORMAL) -> None:
        """Publish a system event."""
        self.publish_event(
            event_type=EventType.SYSTEM,
            data={"message": message, "timestamp": time.time()},
            priority=priority
        )
    
    def publish_data_event(self, data_id: str, value: float, is_new: bool = False) -> None:
        """Publish a data event."""
        event_type = EventType.DATA_NEW if is_new else EventType.DATA_UPDATE
        self.publish_event(
            event_type=event_type,
            data={
                "data_id": data_id,
                "value": value,
                "timestamp": time.time()
            }
        )

def callback_handler(event: Event) -> None:
    """
    A simple callback function for handling events.
    """
    logger.info(f"Callback received event: {event.event_type.name}, {event.data}")

async def main() -> None:
    """Run the event system example."""
    try:
        # Create event bus
        event_bus = EventBus()
        event_bus.start()
        logger.info("Event bus started")
        
        # Create event handler
        handler = SimpleEventHandler(event_bus)
        
        # Create event publisher
        publisher = SimpleEventPublisher(event_bus)
        
        # Register callback for another event type
        event_bus.add_callback(EventType.FEATURE_UPDATE, callback_handler)
        logger.info("Registered callback for FEATURE_UPDATE events")
        
        # Publish events with different priorities
        logger.info("Publishing events...")
        
        # High priority system event
        publisher.publish_system_event("Critical system message", EventPriority.HIGH)
        
        # New data event
        publisher.publish_data_event("sensor1", 42.0, is_new=True)
        
        # Normal priority system event
        publisher.publish_system_event("Normal system message")
        
        # Update data event
        publisher.publish_data_event("sensor1", 43.5)
        
        # Directly publish an event
        event_bus.publish_event(
            event_type=EventType.FEATURE_UPDATE,
            data={"feature_name": "test_feature", "value": 0.75},
            source="example"
        )
        
        # Wait a bit for event processing
        await asyncio.sleep(0.5)
        
        # Get stats
        stats = handler.get_stats()
        logger.info(f"Event handler processed {stats['event_count']} events")
        
        # Clean shutdown
        logger.info("Shutting down event bus...")
        event_bus.shutdown()
        logger.info("Event bus shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Example terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)