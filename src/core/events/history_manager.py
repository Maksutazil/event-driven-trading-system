#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event History Manager Module

This module provides the EventHistoryManager class for maintaining historical events
by token, with time-based pruning and efficient storage and retrieval capabilities.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque, defaultdict

from .event import Event, EventType, EventPriority
from .event_bus import EventBus

logger = logging.getLogger(__name__)


class EventHistoryManager:
    """
    Manager for maintaining historical events by token.
    
    The EventHistoryManager stores events in memory with efficient retrieval by
    token ID and event type. It supports time-based pruning to manage memory usage
    and provides statistics on event processing.
    
    Features:
    - Event storage by token ID and event type
    - Time-based event pruning
    - Efficient event retrieval by token and type
    - Memory usage management
    - Event statistics and analytics
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        max_events_per_token: int = 1000,
        default_retention_period: timedelta = timedelta(hours=24),
        pruning_interval: timedelta = timedelta(minutes=30),
        event_types_to_track: Optional[Set[EventType]] = None
    ):
        """
        Initialize the EventHistoryManager.
        
        Args:
            event_bus: EventBus instance for subscribing to events
            max_events_per_token: Maximum number of events to store per token
            default_retention_period: Default time period to retain events
            pruning_interval: Interval for pruning old events
            event_types_to_track: Set of event types to track (None = all events)
        """
        self._event_bus = event_bus
        self._max_events_per_token = max_events_per_token
        self._default_retention_period = default_retention_period
        self._pruning_interval = pruning_interval
        
        # Default to tracking common token-related events if not specified
        self._event_types_to_track = event_types_to_track or {
            EventType.TOKEN_CREATED,
            EventType.TOKEN_UPDATED,
            EventType.TOKEN_TRADE,
            EventType.TRADE_SIGNAL,
            EventType.TRADE_EXECUTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.POSITION_UPDATED,
            EventType.FEATURE_UPDATE,
            EventType.MODEL_PREDICTION
        }
        
        # Event storage structure: token_id -> event_type -> deque of (timestamp, event)
        self._events: Dict[str, Dict[EventType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._max_events_per_token))
        )
        
        # Custom retention periods for specific event types
        self._retention_periods: Dict[EventType, timedelta] = {}
        
        # Statistics counters
        self._stats = {
            'events_added': 0,
            'events_pruned': 0,
            'events_retrieved': 0,
            'tokens_tracked': 0
        }
        
        # Thread safety lock
        self._lock = threading.RLock()
        
        # Pruning timer
        self._pruning_timer = None
        self._is_running = False
        
        # Register with event bus if provided
        if self._event_bus:
            self._register_event_handlers()
    
    def start(self) -> None:
        """Start the event history manager and pruning timer."""
        if self._is_running:
            return
            
        self._is_running = True
        self._schedule_pruning()
        logger.info("EventHistoryManager started")
    
    def stop(self) -> None:
        """Stop the event history manager and pruning timer."""
        self._is_running = False
        if self._pruning_timer:
            self._pruning_timer.cancel()
            self._pruning_timer = None
        logger.info("EventHistoryManager stopped")
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to the history.
        
        Args:
            event: Event to add
        """
        # Skip if event type not tracked
        if event.event_type not in self._event_types_to_track:
            return
            
        # Extract token_id from event data
        token_id = event.data.get('token_id')
        if not token_id:
            # Some events might use different field names for token_id
            token_id = (
                event.data.get('symbol') or
                event.data.get('token') or 
                event.data.get('asset_id')
            )
            
        if not token_id:
            # Can't index this event by token - skip it
            return
        
        with self._lock:
            # Add event to storage
            event_timestamp = event.timestamp or time.time()
            self._events[token_id][event.event_type].append((event_timestamp, event))
            
            # Update statistics
            self._stats['events_added'] += 1
            if token_id not in self._stats:
                self._stats['tokens_tracked'] += 1
    
    def get_events(
        self,
        token_id: str,
        event_type: Optional[EventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get events for a specific token.
        
        Args:
            token_id: Token ID to get events for
            event_type: Optional event type filter
            start_time: Optional start time filter (unix timestamp)
            end_time: Optional end time filter (unix timestamp)
            limit: Optional maximum number of events to return
            
        Returns:
            List of events matching the criteria
        """
        with self._lock:
            events = []
            
            if token_id not in self._events:
                return events
                
            # If event type specified, only get events of that type
            event_types = [event_type] if event_type else list(self._events[token_id].keys())
            
            for et in event_types:
                if et not in self._events[token_id]:
                    continue
                    
                # Get events from storage
                for timestamp, event in self._events[token_id][et]:
                    # Apply time filters
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                        
                    events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            # Apply limit
            if limit and len(events) > limit:
                events = events[-limit:]
                
            # Update statistics
            self._stats['events_retrieved'] += len(events)
            
            return events
    
    def get_latest_event(
        self,
        token_id: str,
        event_type: EventType
    ) -> Optional[Event]:
        """
        Get the most recent event of a specific type for a token.
        
        Args:
            token_id: Token ID to get the event for
            event_type: Event type to get
            
        Returns:
            The most recent event or None if not found
        """
        with self._lock:
            if token_id not in self._events or event_type not in self._events[token_id]:
                return None
                
            if not self._events[token_id][event_type]:
                return None
                
            # Get the most recent event
            _, event = self._events[token_id][event_type][-1]
            
            # Update statistics
            self._stats['events_retrieved'] += 1
            
            return event
    
    def get_event_count(
        self,
        token_id: str,
        event_type: Optional[EventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> int:
        """
        Get count of events for a specific token.
        
        Args:
            token_id: Token ID to get count for
            event_type: Optional event type filter
            start_time: Optional start time filter (unix timestamp)
            end_time: Optional end time filter (unix timestamp)
            
        Returns:
            Count of events matching the criteria
        """
        with self._lock:
            if token_id not in self._events:
                return 0
                
            count = 0
            
            # If event type specified, only count events of that type
            event_types = [event_type] if event_type else list(self._events[token_id].keys())
            
            for et in event_types:
                if et not in self._events[token_id]:
                    continue
                    
                # Count events within time range
                if not start_time and not end_time:
                    # No time filtering - count all events
                    count += len(self._events[token_id][et])
                else:
                    # Time filtering - count matching events
                    for timestamp, _ in self._events[token_id][et]:
                        if start_time and timestamp < start_time:
                            continue
                        if end_time and timestamp > end_time:
                            continue
                            
                        count += 1
            
            return count
    
    def get_event_frequency(
        self,
        token_id: str,
        event_type: EventType,
        window_size: timedelta = timedelta(minutes=5)
    ) -> float:
        """
        Calculate the frequency of events per minute over a time window.
        
        Args:
            token_id: Token ID to calculate frequency for
            event_type: Event type to calculate frequency for
            window_size: Time window size
            
        Returns:
            Event frequency per minute
        """
        with self._lock:
            if token_id not in self._events or event_type not in self._events[token_id]:
                return 0.0
                
            now = time.time()
            window_start = now - window_size.total_seconds()
            
            # Count events in window
            count = 0
            for timestamp, _ in self._events[token_id][event_type]:
                if timestamp >= window_start:
                    count += 1
            
            # Calculate frequency per minute
            window_minutes = window_size.total_seconds() / 60.0
            if window_minutes > 0:
                return count / window_minutes
            else:
                return 0.0
    
    def set_retention_period(
        self,
        event_type: EventType,
        retention_period: timedelta
    ) -> None:
        """
        Set custom retention period for a specific event type.
        
        Args:
            event_type: Event type to set retention period for
            retention_period: Retention period
        """
        with self._lock:
            self._retention_periods[event_type] = retention_period
            logger.info(f"Set retention period for {event_type.name} to {retention_period}")
    
    def get_tracked_tokens(self) -> List[str]:
        """
        Get list of token IDs being tracked.
        
        Returns:
            List of token IDs
        """
        with self._lock:
            return list(self._events.keys())
    
    def get_token_event_types(self, token_id: str) -> List[EventType]:
        """
        Get list of event types tracked for a token.
        
        Args:
            token_id: Token ID to get event types for
            
        Returns:
            List of event types
        """
        with self._lock:
            if token_id not in self._events:
                return []
                
            return list(self._events[token_id].keys())
    
    def clear_token_history(self, token_id: str) -> None:
        """
        Clear all history for a specific token.
        
        Args:
            token_id: Token ID to clear history for
        """
        with self._lock:
            if token_id in self._events:
                events_cleared = sum(len(events) for events in self._events[token_id].values())
                del self._events[token_id]
                self._stats['events_pruned'] += events_cleared
                self._stats['tokens_tracked'] -= 1
                logger.info(f"Cleared history for token {token_id} ({events_cleared} events)")
    
    def prune_events(
        self,
        token_id: Optional[str] = None,
        event_type: Optional[EventType] = None
    ) -> int:
        """
        Prune old events based on retention periods.
        
        Args:
            token_id: Optional token ID to prune events for (None = all tokens)
            event_type: Optional event type to prune (None = all event types)
            
        Returns:
            Number of events pruned
        """
        pruned_count = 0
        now = time.time()
        
        with self._lock:
            tokens_to_process = [token_id] if token_id else list(self._events.keys())
            
            for tid in tokens_to_process:
                if tid not in self._events:
                    continue
                    
                event_types_to_process = [event_type] if event_type else list(self._events[tid].keys())
                
                for et in event_types_to_process:
                    if et not in self._events[tid]:
                        continue
                        
                    # Get retention period for this event type
                    retention_period = self._retention_periods.get(et, self._default_retention_period)
                    cutoff_time = now - retention_period.total_seconds()
                    
                    # Find events to prune
                    events_to_keep = deque(maxlen=self._max_events_per_token)
                    for timestamp, event in self._events[tid][et]:
                        if timestamp >= cutoff_time:
                            events_to_keep.append((timestamp, event))
                        else:
                            pruned_count += 1
                    
                    # Replace with filtered events
                    self._events[tid][et] = events_to_keep
                    
                # Clean up empty event type containers
                empty_event_types = [et for et, events in self._events[tid].items() if not events]
                for et in empty_event_types:
                    del self._events[tid][et]
                
                # Clean up tokens with no events
                if not self._events[tid]:
                    del self._events[tid]
                    self._stats['tokens_tracked'] -= 1
        
        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} old events")
            self._stats['events_pruned'] += pruned_count
        
        return pruned_count
    
    def _schedule_pruning(self) -> None:
        """Schedule the next pruning run."""
        if not self._is_running:
            return
            
        # Run pruning
        try:
            self.prune_events()
        except Exception as e:
            logger.error(f"Error during event pruning: {e}", exc_info=True)
            
        # Schedule next run
        self._pruning_timer = threading.Timer(
            self._pruning_interval.total_seconds(),
            self._schedule_pruning
        )
        self._pruning_timer.daemon = True
        self._pruning_timer.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about events and storage.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Count current events
            current_event_count = 0
            for token_events in self._events.values():
                for event_list in token_events.values():
                    current_event_count += len(event_list)
            
            stats['current_events'] = current_event_count
            stats['current_tokens'] = len(self._events)
            
            return stats
    
    def _register_event_handlers(self) -> None:
        """Register event handlers with the event bus."""
        if not self._event_bus:
            return
        
        # Use the standardized EventHandlerWrapper
        from src.core.events.base import EventHandlerWrapper
        
        # Register a handler for all tracked event types
        for event_type in self._event_types_to_track:
            self._event_bus.subscribe(
                event_type, 
                EventHandlerWrapper(self.add_event)
            )
        
        logger.info(f"Registered event handlers for {len(self._event_types_to_track)} event types")
    
    def cleanup(self) -> None:
        """Clean up resources and stop processing."""
        self.stop()
        
        # Clear event storage
        with self._lock:
            self._events.clear()
            
        logger.info("EventHistoryManager cleaned up") 