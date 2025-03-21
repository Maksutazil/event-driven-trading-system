#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Interfaces Module

This module provides the interfaces for components to interact with the event system.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Set, Dict, List, Union, Optional
from .event import Event, EventType


class EventSubscriber(ABC):
    """
    Interface for components that subscribe to events.
    
    Components implementing this interface can subscribe to specific event types
    and provide callback methods to handle these events.
    """
    
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """
        Handle an event.
        
        This method is called when an event the component is subscribed to is published.
        Implementing classes should override this method to handle events.
        
        Args:
            event: The event to handle
        """
        pass
    
    def can_handle_event(self, event: Event) -> bool:
        """
        Check if this subscriber can handle the given event.
        
        Default implementation returns True. Override this method to add
        custom filtering logic for events.
        
        Args:
            event: The event to check
            
        Returns:
            bool: True if this subscriber can handle the event, False otherwise
        """
        return True


class EventPublisher(ABC):
    """
    Interface for components that publish events.
    
    Components implementing this interface can publish events to the event bus.
    """
    
    @abstractmethod
    def publish_event(self, event_type: EventType, data: Dict[str, Any], 
                      priority: Optional[str] = None, 
                      source: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event_type: The type of event
            data: The data associated with the event
            priority: Optional priority of the event (LOW, NORMAL, HIGH, CRITICAL)
            source: Optional source of the event
            tags: Optional tags for additional categorization
        """
        pass


# Convenience type for event callback functions
EventCallback = Callable[[Event], None]

# Type for event filters (can be single event type or a set of them)
EventTypeFilter = Union[EventType, Set[EventType]]