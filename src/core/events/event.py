#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event Module

This module defines the Event class and related enumerations.
"""

import time
import uuid
from enum import Enum, auto
from typing import Dict, Any, Optional, List


class EventType(Enum):
    """
    Enumeration of all possible event types.
    """
    # General events
    SYSTEM = auto()
    
    # Data-related events
    DATA_NEW = auto()
    DATA_UPDATE = auto()
    DATA_ERROR = auto()
    
    # Token-related events
    TOKEN_NEW = auto()
    TOKEN_TRADE = auto()
    TOKEN_UPDATE = auto()
    TOKEN_ERROR = auto()
    
    # Feature-related events
    FEATURE_UPDATE = auto()
    FEATURE_ERROR = auto()
    
    # Trading-related events
    TRADE_SIGNAL = auto()
    TRADE_ENTRY = auto()
    TRADE_EXIT = auto()
    POSITION_UPDATE = auto()
    
    # Socket-related events
    SOCKET_CONNECT = auto()
    SOCKET_DISCONNECT = auto()
    SOCKET_MESSAGE = auto()
    SOCKET_ERROR = auto()
    
    # Database-related events
    DB_CONNECT = auto()
    DB_DISCONNECT = auto()
    DB_QUERY = auto()
    DB_ERROR = auto()


class EventPriority(Enum):
    """
    Enumeration of possible event priorities.
    
    Higher priority events are processed before lower priority events.
    """
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Event:
    """
    Represents a discrete event in the system.
    
    Events are used for communication between components and contain
    information about what happened and any relevant data.
    """
    
    def __init__(self, 
                 event_type: EventType, 
                 data: Dict[str, Any],
                 priority: Optional[EventPriority] = EventPriority.NORMAL,
                 source: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """
        Initialize a new Event instance.
        
        Args:
            event_type: The type of event
            data: Dictionary containing the event payload
            priority: Importance level of the event (default: NORMAL)
            source: Which component generated the event (default: None)
            tags: Optional categorization tags (default: None)
        """
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()
        self.event_id = str(uuid.uuid4())
        self.priority = priority
        self.source = source
        self.tags = tags or []
    
    def __lt__(self, other: 'Event') -> bool:
        """
        Compare events based on priority and timestamp.
        
        Higher priority events come first. If priorities are equal,
        older events come first.
        
        Args:
            other: The event to compare with
            
        Returns:
            bool: True if this event has higher priority or is older
        """
        if self.priority.value == other.priority.value:
            return self.timestamp < other.timestamp
        return self.priority.value > other.priority.value
    
    def __str__(self) -> str:
        """Return a string representation of the event."""
        return f"Event({self.event_type.name}, id={self.event_id}, priority={self.priority.name})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the event
        """
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp,
            'priority': self.priority.name,
            'source': self.source,
            'tags': self.tags,
            'data': self.data
        }