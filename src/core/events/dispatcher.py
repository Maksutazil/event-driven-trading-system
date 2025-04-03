#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Dispatcher Module

This module provides the EventDispatcher class, which combines event publishing and
subscription functionality into a single component.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Type, Union, Callable
import inspect

from .event import Event, EventType, EventPriority
from .event_bus import EventBus
from .interfaces import EventSubscriber, EventCallback, EventTypeFilter


logger = logging.getLogger(__name__)


class EventDispatcher:
    """
    The EventDispatcher class combines event publishing and subscription
    functionality into a single component.
    
    It provides methods for subscribing to events, registering event handlers,
    and publishing events to the event bus.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the EventDispatcher.
        
        Args:
            event_bus: The EventBus instance to use
        """
        self.event_bus = event_bus
        
        # Track registered handlers for automatic unsubscription
        self._registered_handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
        
        # Track event types that this dispatcher is subscribed to
        self._subscribed_event_types: Set[EventType] = set()
    
    def subscribe(self, event_type: EventTypeFilter) -> None:
        """
        Subscribe this dispatcher to a specific event type.
        
        Args:
            event_type: The event type or set of event types to subscribe to
        """
        if isinstance(event_type, set):
            for e_type in event_type:
                self.event_bus.subscribe(e_type, self)
                self._subscribed_event_types.add(e_type)
        else:
            self.event_bus.subscribe(event_type, self)
            self._subscribed_event_types.add(event_type)
    
    def unsubscribe(self, event_type: EventTypeFilter) -> None:
        """
        Unsubscribe this dispatcher from a specific event type.
        
        Args:
            event_type: The event type or set of event types to unsubscribe from
        """
        if isinstance(event_type, set):
            for e_type in event_type:
                self.event_bus.unsubscribe(e_type, self)
                if e_type in self._subscribed_event_types:
                    self._subscribed_event_types.remove(e_type)
        else:
            self.event_bus.unsubscribe(event_type, self)
            if event_type in self._subscribed_event_types:
                self._subscribed_event_types.remove(event_type)
    
    def unsubscribe_all(self) -> None:
        """
        Unsubscribe this dispatcher from all event types.
        """
        for event_type in self._subscribed_event_types.copy():
            self.unsubscribe(event_type)
    
    def can_handle_event(self, event: Event) -> bool:
        """
        Check if this dispatcher can handle a specific event.
        
        A dispatcher can handle an event if it has registered handlers
        for the event type or if it's the SYSTEM event type.
        
        Args:
            event: The event to check
            
        Returns:
            True if this dispatcher can handle the event, False otherwise
        """
        return (
            event.event_type in self._registered_handlers or
            EventType.SYSTEM in self._registered_handlers
        )
    
    def on_event(self, event: Event) -> None:
        """
        Handle an event.
        
        This method is called by the EventBus when an event is published
        that this dispatcher is subscribed to.
        
        Args:
            event: The event to handle
        """
        if not self.can_handle_event(event):
            return
        
        # Try to find a handler specifically for this event type
        handlers = self._registered_handlers.get(event.event_type, [])
        
        # Also check for handlers for the SYSTEM event type (catch-all)
        system_handlers = self._registered_handlers.get(EventType.SYSTEM, [])
        
        # Combine the handlers
        all_handlers = handlers + system_handlers
        
        # Call the handlers, providing the event object
        for handler in all_handlers:
            try:
                handler(event)
            except Exception as e:
                handler_name = getattr(handler, '__name__', handler.__class__.__name__)
                logger.error(f"Error in handler {handler_name} for event {event}: {e}", exc_info=True)
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
        """
        self.event_bus.publish(event)
    
    def publish_event(self, event_type: EventType, data: Dict[str, Any],
                     priority: Optional[EventPriority] = None,
                     source: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> None:
        """
        Create and publish an event.
        
        This is a convenience method that creates an Event object from the
        provided parameters and publishes it.
        
        Args:
            event_type: The type of event
            data: The data associated with the event
            priority: Optional priority of the event (default: NORMAL)
            source: Optional source of the event
            tags: Optional tags for additional categorization
        """
        self.event_bus.publish_event(
            event_type=event_type,
            data=data,
            priority=priority,
            source=source or self.__class__.__name__,
            tags=tags
        )
    
    def register_handler(self, event_type: EventTypeFilter, handler: Callable[[Event], None]) -> None:
        """
        Register a handler function for a specific event type.
        
        Args:
            event_type: The event type or set of event types to register the handler for
            handler: The handler function to call when events of this type are published
        """
        if isinstance(event_type, set):
            for e_type in event_type:
                if e_type not in self._registered_handlers:
                    self._registered_handlers[e_type] = []
                self._registered_handlers[e_type].append(handler)
                
                # Subscribe to this event type if not already subscribed
                if e_type not in self._subscribed_event_types:
                    self.subscribe(e_type)
        else:
            if event_type not in self._registered_handlers:
                self._registered_handlers[event_type] = []
            self._registered_handlers[event_type].append(handler)
            
            # Subscribe to this event type if not already subscribed
            if event_type not in self._subscribed_event_types:
                self.subscribe(event_type)
    
    def unregister_handler(self, event_type: EventTypeFilter, handler: Callable[[Event], None]) -> None:
        """
        Unregister a handler function for a specific event type.
        
        Args:
            event_type: The event type or set of event types to unregister the handler from
            handler: The handler function to remove
        """
        if isinstance(event_type, set):
            for e_type in event_type:
                if e_type in self._registered_handlers and handler in self._registered_handlers[e_type]:
                    self._registered_handlers[e_type].remove(handler)
                    
                    # Unsubscribe from this event type if no more handlers
                    if not self._registered_handlers[e_type]:
                        self.unsubscribe(e_type)
                        del self._registered_handlers[e_type]
        else:
            if event_type in self._registered_handlers and handler in self._registered_handlers[event_type]:
                self._registered_handlers[event_type].remove(handler)
                
                # Unsubscribe from this event type if no more handlers
                if not self._registered_handlers[event_type]:
                    self.unsubscribe(event_type)
                    del self._registered_handlers[event_type]
    
    def register_method_handlers(self) -> None:
        """
        Register handler methods in this class.
        
        This method looks for methods in this class that match the pattern
        'handle_<event_type_name>' and registers them as handlers for the
        corresponding event type.
        """
        for attr_name in dir(self):
            if attr_name.startswith('handle_'):
                event_type_name = attr_name[len('handle_'):].upper()
                
                # Skip methods that don't correspond to an event type
                if not hasattr(EventType, event_type_name):
                    continue
                
                event_type = getattr(EventType, event_type_name)
                method = getattr(self, attr_name)
                
                # Register the method as a handler for this event type
                self.register_handler(event_type, method)
                logger.debug(f"Registered method {attr_name} as handler for event type {event_type.name}")
                
    def cleanup(self) -> None:
        """
        Clean up this dispatcher.
        
        This unsubscribes from all event types and clears all registered handlers.
        """
        self.unsubscribe_all()
        self._registered_handlers.clear() 