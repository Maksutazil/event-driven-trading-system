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

print("Current Python path:", sys.path)

# Configure logging more verbosely
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure logger level is set to DEBUG

print("Importing from src.core.events...")
try:
    from src.core.events import (
        Event, EventType, EventPriority, EventBus,
        BaseEventSubscriber, BaseEventPublisher
    )
    print("Successfully imported event system components")
except Exception as e:
    print(f"Error importing event system components: {e}")
    raise

class SimpleEventHandler(BaseEventSubscriber):
    """
    A simple event subscriber that handles multiple event types.
    """
    def __init__(self, event_bus: EventBus):
        print(f"Initializing SimpleEventHandler with event_bus: {event_bus}")
        super().__init__(event_bus)
        self.event_count = 0
        
        # Register event handlers
        print("Registering event handlers...")
        self.register_handler(EventType.SYSTEM, self.handle_system)
        self.register_handler(EventType.DATA_NEW, self.handle_data_new)
        self.register_handler(EventType.DATA_UPDATE, self.handle_data_update)
        
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
        print(f"Initializing SimpleEventPublisher with event_bus: {event_bus}")
        super().__init__(event_bus)
        logger.info("SimpleEventPublisher initialized")
    
    def publish_system_event(self, message: str, priority: EventPriority = EventPriority.NORMAL) -> None:
        """Publish a system event."""
        print(f"Publishing system event: {message} with priority {priority}")
        self.publish_event(
            event_type=EventType.SYSTEM,
            data={"message": message, "timestamp": time.time()},
            priority=priority,
            source="simple_publisher"
        )
    
    def publish_data_event(self, data_id: str, value: float, is_new: bool = False) -> None:
        """Publish a data event."""
        event_type = EventType.DATA_NEW if is_new else EventType.DATA_UPDATE
        print(f"Publishing data event: {data_id}={value}, type={event_type.name}")
        self.publish_event(
            event_type=event_type,
            data={
                "data_id": data_id,
                "value": value,
                "timestamp": time.time()
            },
            source="simple_publisher"
        )

def callback_handler(event: Event) -> None:
    """
    A simple callback function for handling events.
    """
    logger.info(f"Callback received event: {event.event_type.name}, {event.data}")

async def main() -> None:
    """Run the event system example."""
    try:
        print("Creating event bus...")
        # Create event bus
        event_bus = EventBus()
        print(f"Event bus created: {event_bus}")
        
        print("Starting event bus...")
        event_bus.start()
        logger.info("Event bus started")
        
        print("Creating event handler...")
        # Create event handler
        handler = SimpleEventHandler(event_bus)
        
        print("Creating event publisher...")
        # Create event publisher
        publisher = SimpleEventPublisher(event_bus)
        
        print("Registering callback...")
        # Register callback for another event type
        event_bus.add_callback(EventType.FEATURE_UPDATE, callback_handler)
        logger.info("Registered callback for FEATURE_UPDATE events")
        
        # Publish events with different priorities
        logger.info("Publishing events...")
        
        print("Publishing high priority system event...")
        # High priority system event
        publisher.publish_system_event("Critical system message", EventPriority.HIGH)
        
        print("Publishing new data event...")
        # New data event
        publisher.publish_data_event("sensor1", 42.0, is_new=True)
        
        print("Publishing normal priority system event...")
        # Normal priority system event
        publisher.publish_system_event("Normal system message")
        
        print("Publishing update data event...")
        # Update data event
        publisher.publish_data_event("sensor1", 43.5)
        
        print("Directly publishing feature update event...")
        # Directly publish an event
        event_bus.publish_event(
            event_type=EventType.FEATURE_UPDATE,
            data={"feature_name": "test_feature", "value": 0.75},
            source="example"
        )
        
        print("Waiting for event processing...")
        # Wait a bit for event processing
        await asyncio.sleep(0.5)
        
        print("Getting stats...")
        # Get stats
        stats = handler.get_stats()
        logger.info(f"Event handler processed {stats['event_count']} events")
        
        print("Shutting down event bus...")
        # Clean shutdown
        logger.info("Shutting down event bus...")
        event_bus.shutdown()
        logger.info("Event bus shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in example: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    print("Starting event system example...")
    try:
        print("Running asyncio.run(main())...")
        asyncio.run(main())
        print("Example completed successfully")
    except KeyboardInterrupt:
        logger.info("Example terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)