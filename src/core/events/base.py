#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Event Classes Module

This module provides base implementations for event subscribers and publishers.
These classes can be extended to create custom components that publish or subscribe to events.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Callable, Union

from .event import Event, EventType, EventPriority
from .event_bus import EventBus
from .interfaces import EventSubscriber, EventCallback, EventTypeFilter


logger = logging.getLogger(__name__)


class EventHandlerWrapper:
    """
    A wrapper for event handler callbacks that ensures consistent error handling
    and makes the handler callable for use with the EventBus.
    
    This standardized wrapper should be used for all event handlers to ensure
    consistent behavior and proper error handling.
    """
    def __init__(self, callback: Callable[[Event], None], name: Optional[str] = None):
        """
        Initialize the event handler wrapper.
        
        Args:
            callback: The callback function to wrap
            name: Optional name for the handler, defaults to callback function name
        """
        self.callback = callback
        # Add a name attribute to help with debugging
        self.__name__ = name or f"{callback.__name__}_handler"
    
    def on_event(self, event: Event) -> None:
        """
        Handle an event by calling the wrapped callback.
        
        Args:
            event: The event to handle
        """
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Error in EventHandler callback {self.__name__}: {e}", exc_info=True)
    
    def __call__(self, event: Event) -> None:
        """
        Make the handler callable for direct use with EventBus.subscribe.
        
        Args:
            event: The event to handle
        """
        # Delegate to on_event method
        self.on_event(event)


class BaseEventPublisher:
    """
    Base implementation for a component that publishes events.
    
    This class provides methods for publishing events to the event bus.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the BaseEventPublisher.
        
        Args:
            event_bus: The EventBus instance to use
        """
        self.event_bus = event_bus
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
        """
        self.event_bus.publish(event)
    
    def publish_event(self, event_type: EventType, data: Dict[str, Any],
                     source: Optional[str] = None) -> None:
        """
        Create and publish an event.
        
        This is a convenience method that creates an Event object from the
        provided parameters and publishes it.
        
        Args:
            event_type: The type of event
            data: The data associated with the event
            source: Optional source of the event
        """
        # Use class name as source if not provided
        if source is None:
            source = self.__class__.__name__
            
        # Create the event
        event = Event(
            event_type=event_type,
            data=data,
            source=source
        )
        
        # Publish the event
        self.publish(event)


class BaseEventSubscriber(EventSubscriber):
    """
    Base implementation for a component that subscribes to events.
    
    This class provides methods for subscribing to events and handling them.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the BaseEventSubscriber.
        
        Args:
            event_bus: The EventBus instance to use
        """
        self.event_bus = event_bus
        
        # Set of event types that this subscriber is subscribed to
        self._subscribed_events: Set[EventType] = set()
        
        # Map of event types to handler methods
        self._event_handlers: Dict[EventType, List[Callable[[Event], None]]] = {}
    
    def subscribe(self, event_type: EventTypeFilter) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The event type or set of event types to subscribe to
        """
        if isinstance(event_type, set):
            # Subscribe to all event types in the set
            for e_type in event_type:
                self.event_bus.subscribe(e_type, self)
                self._subscribed_events.add(e_type)
        else:
            # Subscribe to the single event type
            self.event_bus.subscribe(event_type, self)
            self._subscribed_events.add(event_type)
    
    def unsubscribe(self, event_type: EventTypeFilter) -> None:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The event type or set of event types to unsubscribe from
        """
        if isinstance(event_type, set):
            # Unsubscribe from all event types in the set
            for e_type in event_type:
                self.event_bus.unsubscribe(e_type, self)
                if e_type in self._subscribed_events:
                    self._subscribed_events.remove(e_type)
        else:
            # Unsubscribe from the single event type
            self.event_bus.unsubscribe(event_type, self)
            if event_type in self._subscribed_events:
                self._subscribed_events.remove(event_type)
    
    def unsubscribe_all(self) -> None:
        """
        Unsubscribe from all event types.
        """
        # Make a copy of the set, as we'll be modifying it while iterating
        for event_type in list(self._subscribed_events):
            self.unsubscribe(event_type)
    
    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """
        Register a handler function for a specific event type.
        
        Args:
            event_type: The event type to register the handler for
            handler: The handler function to call when events of this type are published
        """
        # Create the list for this event type if it doesn't exist
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
            
        # Add the handler to the list
        self._event_handlers[event_type].append(handler)
        
        # Subscribe to this event type if not already subscribed
        if event_type not in self._subscribed_events:
            self.subscribe(event_type)
    
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
    
    def can_handle_event(self, event: Event) -> bool:
        """
        Check if this subscriber can handle a specific event.
        
        A subscriber can handle an event if it has registered handlers
        for the event type or if it's the SYSTEM event type.
        
        Args:
            event: The event to check
            
        Returns:
            True if this subscriber can handle the event, False otherwise
        """
        return (
            event.event_type in self._event_handlers or
            EventType.SYSTEM in self._event_handlers
        )
    
    def on_event(self, event: Event) -> None:
        """
        Handle an event.
        
        This method is called by the EventBus when an event is published
        that this subscriber is subscribed to.
        
        Args:
            event: The event to handle
        """
        if not self.can_handle_event(event):
            return
        
        # Try to find a handler specifically for this event type
        handlers = self._event_handlers.get(event.event_type, [])
        
        # Also check for handlers for the SYSTEM event type (catch-all)
        system_handlers = self._event_handlers.get(EventType.SYSTEM, [])
        
        # Combine the handlers
        all_handlers = handlers + system_handlers
        
        # Call the handlers, providing the event object
        for handler in all_handlers:
            try:
                handler(event)
            except Exception as e:
                handler_name = getattr(handler, '__name__', handler.__class__.__name__)
                logger.error(f"Error in handler {handler_name} for event {event}: {e}", exc_info=True)
    
    def __call__(self, event: Event) -> None:
        """
        Make the subscriber callable so it can be used directly with event_bus.subscribe().
        
        This method delegates to on_event.
        
        Args:
            event: The event to handle
        """
        self.on_event(event)
    
    def cleanup(self) -> None:
        """
        Clean up this subscriber.
        
        This unsubscribes from all event types.
        """
        self.unsubscribe_all()
        self._event_handlers.clear()


class EventDispatcher(BaseEventPublisher, BaseEventSubscriber):
    """
    A component that can both publish and subscribe to events.
    
    This class combines the functionality of BaseEventPublisher and BaseEventSubscriber.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the EventDispatcher.
        
        Args:
            event_bus: The EventBus instance to use
        """
        # Initialize both parent classes
        BaseEventPublisher.__init__(self, event_bus)
        BaseEventSubscriber.__init__(self, event_bus)
    
    def cleanup(self) -> None:
        """
        Clean up this dispatcher.
        
        This calls the cleanup method of the BaseEventSubscriber parent class.
        """
        # Call the cleanup method of the BaseEventSubscriber class
        BaseEventSubscriber.cleanup(self) 