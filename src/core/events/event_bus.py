#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Bus Module

This module provides the EventBus class, which is the central hub for
event distribution in the system.
"""

import logging
import threading
import time
from typing import Dict, List, Set, Optional, Callable, Any
from collections import defaultdict
import queue

from .event import Event, EventType, EventHandler, EventSubscriber

logger = logging.getLogger(__name__)


class EventBus:
    """
    Central hub for event distribution.
    
    The EventBus allows components to publish events and subscribe to
    specific event types. It handles the routing of events from publishers
    to subscribers.
    """
    
    def __init__(self, async_processing: bool = True, max_queue_size: int = 1000):
        """
        Initialize the EventBus.
        
        Args:
            async_processing: Whether to process events asynchronously (default: True)
            max_queue_size: Maximum size of the event queue (default: 1000)
        """
        # Map of event type to set of handlers
        self.subscribers: Dict[EventType, Set[EventHandler]] = defaultdict(set)
        
        # Subscribers that receive all events
        self.global_subscribers: Set[EventHandler] = set()
        
        # Map of event types to component subscriber instances
        self.component_subscribers: Dict[EventType, Set[EventSubscriber]] = defaultdict(set)
        
        # Global component subscribers
        self.global_component_subscribers: Set[EventSubscriber] = set()
        
        # Asynchronous processing
        self.async_processing = async_processing
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Start processing thread if async
        if async_processing:
            self.start_processing()
        
        logger.info(f"EventBus initialized (async={async_processing})")
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe a handler function to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event occurs
        """
        with self.lock:
            self.subscribers[event_type].add(handler)
            # Use getattr to safely get __name__ or a fallback to the class name
            handler_name = getattr(handler, '__name__', handler.__class__.__name__)
            logger.debug(f"Subscribed handler {handler_name} to {event_type.name}")
    
    def subscribe_to_all(self, handler: EventHandler) -> None:
        """
        Subscribe a handler to all event types.
        
        Args:
            handler: Function to call for all events
        """
        with self.lock:
            self.global_subscribers.add(handler)
            # Use getattr to safely get __name__ or a fallback to the class name
            handler_name = getattr(handler, '__name__', handler.__class__.__name__)
            logger.debug(f"Subscribed handler {handler_name} to all events")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to unsubscribe
        """
        with self.lock:
            if event_type in self.subscribers and handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
                # Use getattr to safely get __name__ or a fallback to the class name
                handler_name = getattr(handler, '__name__', handler.__class__.__name__)
                logger.debug(f"Unsubscribed handler {handler_name} from {event_type.name}")
    
    def unsubscribe_from_all(self, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from all event types.
        
        Args:
            handler: Handler to unsubscribe
        """
        with self.lock:
            if handler in self.global_subscribers:
                self.global_subscribers.remove(handler)
                # Use getattr to safely get __name__ or a fallback to the class name
                handler_name = getattr(handler, '__name__', handler.__class__.__name__)
                logger.debug(f"Unsubscribed handler {handler_name} from all events")
            
            # Also remove from any specific event types
            for event_type in self.subscribers:
                if handler in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(handler)
                    # Use getattr to safely get __name__ or a fallback to the class name
                    handler_name = getattr(handler, '__name__', handler.__class__.__name__)
                    logger.debug(f"Unsubscribed handler {handler_name} from {event_type.name}")
    
    def subscribe_component(self, component: EventSubscriber, 
                           event_types: Optional[List[EventType]] = None) -> None:
        """
        Subscribe a component to specific event types.
        
        Args:
            component: Component that implements EventSubscriber
            event_types: List of event types to subscribe to, or None for all
        """
        with self.lock:
            if event_types is None:
                # Subscribe to all events
                self.global_component_subscribers.add(component)
                logger.debug(f"Subscribed component {component.__class__.__name__} to all events")
            else:
                # Subscribe to specific event types
                for event_type in event_types:
                    self.component_subscribers[event_type].add(component)
                    logger.debug(f"Subscribed component {component.__class__.__name__} to {event_type.name}")
    
    def unsubscribe_component(self, component: EventSubscriber,
                             event_types: Optional[List[EventType]] = None) -> None:
        """
        Unsubscribe a component from specific event types.
        
        Args:
            component: Component to unsubscribe
            event_types: List of event types to unsubscribe from, or None for all
        """
        with self.lock:
            if event_types is None:
                # Unsubscribe from all events
                if component in self.global_component_subscribers:
                    self.global_component_subscribers.remove(component)
                
                # Also remove from specific event types
                for event_type in self.component_subscribers:
                    if component in self.component_subscribers[event_type]:
                        self.component_subscribers[event_type].remove(component)
                
                logger.debug(f"Unsubscribed component {component.__class__.__name__} from all events")
            else:
                # Unsubscribe from specific event types
                for event_type in event_types:
                    if component in self.component_subscribers[event_type]:
                        self.component_subscribers[event_type].remove(component)
                        logger.debug(f"Unsubscribed component {component.__class__.__name__} from {event_type.name}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        if self.async_processing:
            # Add to queue for async processing
            try:
                self.event_queue.put(event, block=False)
                logger.debug(f"Queued event {event.event_id} for async processing")
            except queue.Full:
                logger.warning(f"Event queue full, dropping event {event.event_id}")
        else:
            # Process immediately
            self._process_event(event)
    
    def _process_event(self, event: Event) -> None:
        """
        Process a single event by dispatching it to all handlers.
        
        Args:
            event: Event to process
        """
        handlers_called = 0
        event_type = event.event_type
        
        with self.lock:
            # Call handlers for this specific event type
            for handler in self.subscribers.get(event_type, set()):
                try:
                    # For better error reporting, get the handler name before calling it
                    handler_name = getattr(handler, '__name__', 
                                         getattr(handler, '__class__', handler).__name__)
                    
                    # Call the handler
                    handler(event)
                    handlers_called += 1
                except Exception as e:
                    # Get more detailed error information
                    handler_class = handler.__class__.__name__
                    logger.error(f"Error in event handler {handler_name} ({handler_class}) for {event_type.name}: {e}")
                    logger.debug(f"Event details: {event.event_id}, source={event.source}", exc_info=True)
            
            # Call global handlers
            for handler in self.global_subscribers:
                try:
                    # For better error reporting, get the handler name before calling it
                    handler_name = getattr(handler, '__name__', 
                                         getattr(handler, '__class__', handler).__name__)
                    
                    # Call the handler
                    handler(event)
                    handlers_called += 1
                except Exception as e:
                    # Get more detailed error information
                    handler_class = handler.__class__.__name__
                    logger.error(f"Error in global event handler {handler_name} ({handler_class}): {e}")
                    logger.debug(f"Event details: {event.event_id}, source={event.source}", exc_info=True)
            
            # Call component handlers for this specific event type
            for component in self.component_subscribers.get(event_type, set()):
                try:
                    component.handle_event(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error(f"Error in component {component.__class__.__name__}.handle_event: {e}")
                    logger.debug(f"Event details: {event.event_id}, source={event.source}", exc_info=True)
            
            # Call global component handlers
            for component in self.global_component_subscribers:
                try:
                    component.handle_event(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error(f"Error in global component {component.__class__.__name__}.handle_event: {e}")
                    logger.debug(f"Event details: {event.event_id}, source={event.source}", exc_info=True)
        
        logger.debug(f"Processed event {event.event_id} ({handlers_called} handlers called)")
    
    def _event_processing_loop(self) -> None:
        """
        Main loop for event processing thread.
        """
        logger.info("Event processing thread started")
        
        while self.running:
            try:
                # Get next event from queue with timeout
                event = self.event_queue.get(timeout=0.1)
                
                # Process the event
                self._process_event(event)
                
                # Mark as done
                self.event_queue.task_done()
            except queue.Empty:
                # No events, continue
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
        
        logger.info("Event processing thread stopped")
    
    def start_processing(self) -> None:
        """
        Start the event processing thread.
        """
        with self.lock:
            if not self.running and self.async_processing:
                self.running = True
                self.processing_thread = threading.Thread(
                    target=self._event_processing_loop,
                    daemon=True
                )
                self.processing_thread.start()
                logger.info("Started event processing thread")
    
    def stop_processing(self, wait_for_queue: bool = True, timeout: float = 5.0) -> None:
        """
        Stop the event processing thread.
        
        Args:
            wait_for_queue: Whether to wait for the queue to empty
            timeout: Maximum time to wait for the queue to empty
        """
        with self.lock:
            if self.running and self.async_processing:
                if wait_for_queue:
                    try:
                        # Wait for all events to be processed
                        self.event_queue.join()
                    except Exception:
                        pass
                
                # Stop the thread
                self.running = False
                
                # Wait for the thread to terminate
                if self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout)
                
                logger.info("Stopped event processing thread")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the event bus.
        
        Returns:
            Dictionary with event bus statistics
        """
        with self.lock:
            stats = {
                "async_processing": self.async_processing,
                "running": self.running,
                "queue_size": self.event_queue.qsize() if self.async_processing else 0,
                "queue_max_size": self.event_queue.maxsize if self.async_processing else 0,
                "subscribers_count": {},
                "global_subscribers_count": len(self.global_subscribers),
                "component_subscribers_count": {},
                "global_component_subscribers_count": len(self.global_component_subscribers)
            }
            
            # Count subscribers for each event type
            for event_type in self.subscribers:
                stats["subscribers_count"][event_type.name] = len(self.subscribers[event_type])
            
            # Count component subscribers for each event type
            for event_type in self.component_subscribers:
                stats["component_subscribers_count"][event_type.name] = len(self.component_subscribers[event_type])
            
            return stats 