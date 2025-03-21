"""
Event System Core Module

This module provides a comprehensive event system for asynchronous communication
between components in the trading application.

Core components:
- Event: Base class for all events
- EventType: Enum of all possible event types
- EventBus: Central hub for event distribution
- EventSubscriber: Interface for components that receive events
- EventPublisher: Interface for components that produce events
- BaseEventSubscriber: Base implementation of EventSubscriber
- BaseEventPublisher: Base implementation of EventPublisher
- EventDispatcher: Combined subscriber and publisher
"""

from .event import Event, EventType, EventPriority
from .event_bus import EventBus
from .interfaces import EventSubscriber, EventPublisher, EventCallback
from .base import BaseEventSubscriber, BaseEventPublisher, EventDispatcher

__all__ = [
    'Event',
    'EventType',
    'EventPriority',
    'EventBus',
    'EventSubscriber',
    'EventPublisher',
    'EventCallback',
    'BaseEventSubscriber',
    'BaseEventPublisher',
    'EventDispatcher'
]