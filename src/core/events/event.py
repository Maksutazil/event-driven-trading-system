#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Module

This module defines the Event class and EventType enum for the event system.
"""

import uuid
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Type, TypeVar, Union


class EventType(Enum):
    """
    Enum of all event types supported by the system.
    """
    # Generic events
    GENERIC = auto()
    SYSTEM = auto()  # General system event for catch-all handlers
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    SYSTEM_STATUS = auto()
    ERROR = auto()
    
    # Token events
    TOKEN_CREATED = auto()
    TOKEN_TRADE = auto()
    TOKEN_UPDATED = auto()
    
    # Trading events
    TRADE_SIGNAL = auto()
    TRADE_EXECUTED = auto()
    TRADE_FAILED = auto()
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()
    
    # Feature events
    FEATURE_UPDATE = auto()
    
    # Risk management events
    RISK_LIMIT_REACHED = auto()
    RISK_ALERT = auto()
    
    # Websocket events
    WEBSOCKET_CONNECTED = auto()
    WEBSOCKET_DISCONNECTED = auto()
    WEBSOCKET_ERROR = auto()
    
    # API events
    API_REQUEST = auto()
    API_RESPONSE = auto()
    API_ERROR = auto()
    
    # User interface events
    UI_UPDATE = auto()
    UI_ACTION = auto()
    
    # Machine learning events
    MODEL_LOADED = auto()
    MODEL_UPDATED = auto()
    MODEL_PREDICTION = auto()


class EventPriority(Enum):
    """
    Enum for event priority levels.
    
    Events with higher priority get processed before lower priority events.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


# Type alias for event handlers
EventHandler = Callable[['Event'], None]


class EventSubscriber:
    """
    Interface for components that receive events.
    """
    def handle_event(self, event: 'Event') -> None:
        """
        Handle an event.
        
        Args:
            event: The event to handle
        """
        raise NotImplementedError("Subclasses must implement handle_event()")


@dataclass
class Event:
    """
    Base class for all events in the system.
    
    Events are immutable data containers that carry information about
    something that happened in the system.
    """
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_id: Optional[str] = None
    
    def __post_init__(self):
        """
        Validate event data after initialization.
        """
        if not isinstance(self.event_type, EventType):
            raise TypeError(f"event_type must be an EventType, got {type(self.event_type)}")
        
        if not isinstance(self.data, dict):
            raise TypeError(f"data must be a dict, got {type(self.data)}")
        
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError(f"source must be a string, got {type(self.source)}")
            
        # If token_id is not set but is in data, extract it
        if self.token_id is None:
            self.token_id = self.data.get('token_id') or self.data.get('mint')
            
        # If token_id is set but not in data, add it to data
        if self.token_id is not None and 'token_id' not in self.data:
            self.data['token_id'] = self.token_id
    
    def with_data(self, **kwargs) -> 'Event':
        """
        Create a new event with updated data.
        
        Args:
            **kwargs: Data fields to update
            
        Returns:
            A new Event instance with updated data
        """
        new_data = self.data.copy()
        new_data.update(kwargs)
        
        # Check if token_id is being updated in kwargs
        token_id = kwargs.get('token_id') or self.token_id
        if 'mint' in kwargs and token_id is None:
            token_id = kwargs.get('mint')
        
        return Event(
            event_type=self.event_type,
            data=new_data,
            source=self.source,
            timestamp=self.timestamp,
            event_id=self.event_id,
            token_id=token_id
        )
    
    def __str__(self) -> str:
        """
        String representation of the event.
        
        Returns:
            String representation
        """
        source_str = f" from {self.source}" if self.source else ""
        return f"Event({self.event_type.name}{source_str}, id={self.event_id})"