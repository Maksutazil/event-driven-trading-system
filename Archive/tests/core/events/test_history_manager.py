#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the EventHistoryManager
"""

import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.events.event import Event, EventType
from src.core.events.event_bus import EventBus
from src.core.events.history_manager import EventHistoryManager


class TestEventHistoryManager(unittest.TestCase):
    """Test suite for the EventHistoryManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.event_bus = Mock(spec=EventBus)
        self.event_bus.subscribe = Mock()
        
        self.history_manager = EventHistoryManager(
            event_bus=self.event_bus,
            max_events_per_token=100,
            default_retention_period=timedelta(hours=1),
            pruning_interval=timedelta(minutes=5)
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.history_manager.cleanup()
    
    def test_add_and_get_events(self):
        """Test adding and retrieving events."""
        # Create test events
        token_id = "TEST_TOKEN"
        events = [
            Event(
                event_type=EventType.TOKEN_CREATED,
                data={"token_id": token_id, "price": 100.0},
                timestamp=time.time() - 60
            ),
            Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id, "price": 101.0},
                timestamp=time.time() - 30
            ),
            Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id, "price": 102.0},
                timestamp=time.time()
            )
        ]
        
        # Add events
        for event in events:
            self.history_manager.add_event(event)
        
        # Get all events for the token
        retrieved_events = self.history_manager.get_events(token_id)
        
        # Check count
        self.assertEqual(len(retrieved_events), 3)
        
        # Check order (should be sorted by timestamp)
        self.assertEqual(retrieved_events[0].event_type, EventType.TOKEN_CREATED)
        self.assertEqual(retrieved_events[1].event_type, EventType.TOKEN_TRADE)
        self.assertEqual(retrieved_events[2].event_type, EventType.TOKEN_TRADE)
        
        # Get events of specific type
        trade_events = self.history_manager.get_events(token_id, EventType.TOKEN_TRADE)
        self.assertEqual(len(trade_events), 2)
        
        # Get events with time range
        recent_events = self.history_manager.get_events(
            token_id, 
            start_time=time.time() - 40,  # Last 40 seconds
            end_time=time.time()
        )
        self.assertEqual(len(recent_events), 2)
        
        # Get with limit
        limited_events = self.history_manager.get_events(token_id, limit=1)
        self.assertEqual(len(limited_events), 1)
        self.assertEqual(limited_events[0].event_type, EventType.TOKEN_TRADE)  # Latest event
    
    def test_get_latest_event(self):
        """Test getting the latest event of a specific type."""
        token_id = "TEST_TOKEN"
        
        # Add some events
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": token_id, "price": 100.0},
            timestamp=time.time() - 60
        ))
        
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": token_id, "price": 105.0},
            timestamp=time.time() - 30
        ))
        
        # Get latest event
        latest = self.history_manager.get_latest_event(token_id, EventType.TOKEN_TRADE)
        
        # Check it's the most recent one
        self.assertIsNotNone(latest)
        self.assertEqual(latest.data.get("price"), 105.0)
        
        # Test with non-existent type
        missing = self.history_manager.get_latest_event(token_id, EventType.ERROR)
        self.assertIsNone(missing)
    
    def test_get_event_count(self):
        """Test counting events."""
        token_id = "TEST_TOKEN"
        
        # Add some events
        for i in range(5):
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id, "price": 100.0 + i},
                timestamp=time.time() - (5 - i) * 10
            ))
        
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_CREATED,
            data={"token_id": token_id},
            timestamp=time.time() - 60
        ))
        
        # Count all events
        count = self.history_manager.get_event_count(token_id)
        self.assertEqual(count, 6)
        
        # Count by type
        trade_count = self.history_manager.get_event_count(token_id, EventType.TOKEN_TRADE)
        self.assertEqual(trade_count, 5)
        
        # Count with time range
        recent_count = self.history_manager.get_event_count(
            token_id,
            start_time=time.time() - 25  # Last 25 seconds
        )
        self.assertEqual(recent_count, 3)  # Should get the 3 most recent events
    
    def test_get_event_frequency(self):
        """Test calculating event frequency."""
        token_id = "TEST_TOKEN"
        now = time.time()
        
        # Add events at different times
        for i in range(10):
            # Events spread over 10 minutes
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id},
                timestamp=now - (10 - i) * 60  # One event per minute for last 10 minutes
            ))
        
        # Calculate frequency over 5-minute window (should be ~1/minute)
        freq = self.history_manager.get_event_frequency(
            token_id,
            EventType.TOKEN_TRADE,
            window_size=timedelta(minutes=5)
        )
        
        # Should be close to 1.0 events per minute
        self.assertAlmostEqual(freq, 1.0, delta=0.1)
    
    def test_prune_events(self):
        """Test pruning old events."""
        token_id = "TEST_TOKEN"
        now = time.time()
        
        # Add some old events (2 hours old)
        for i in range(5):
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id},
                timestamp=now - 7200  # 2 hours ago
            ))
        
        # Add some recent events
        for i in range(3):
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id},
                timestamp=now - 60  # 1 minute ago
            ))
        
        # Check initial count
        self.assertEqual(self.history_manager.get_event_count(token_id), 8)
        
        # Prune events (default retention is 1 hour from constructor)
        pruned = self.history_manager.prune_events()
        
        # Should have pruned the 5 old events
        self.assertEqual(pruned, 5)
        
        # Check remaining count
        self.assertEqual(self.history_manager.get_event_count(token_id), 3)
    
    def test_clear_token_history(self):
        """Test clearing all history for a token."""
        token_id = "TEST_TOKEN"
        
        # Add some events
        for i in range(5):
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id}
            ))
        
        # Check we have events
        self.assertEqual(self.history_manager.get_event_count(token_id), 5)
        
        # Clear history
        self.history_manager.clear_token_history(token_id)
        
        # Should have no events now
        self.assertEqual(self.history_manager.get_event_count(token_id), 0)
    
    def test_get_tracked_tokens(self):
        """Test getting list of tracked tokens."""
        # Add events for multiple tokens
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": "TOKEN1"}
        ))
        
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": "TOKEN2"}
        ))
        
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": "TOKEN3"}
        ))
        
        # Get token list
        tokens = self.history_manager.get_tracked_tokens()
        
        # Check contents
        self.assertEqual(len(tokens), 3)
        self.assertIn("TOKEN1", tokens)
        self.assertIn("TOKEN2", tokens)
        self.assertIn("TOKEN3", tokens)
    
    def test_custom_retention_period(self):
        """Test setting custom retention period for event types."""
        token_id = "TEST_TOKEN"
        now = time.time()
        
        # Add some events of different types with different ages
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_CREATED,
            data={"token_id": token_id},
            timestamp=now - 1800  # 30 minutes ago
        ))
        
        self.history_manager.add_event(Event(
            event_type=EventType.TOKEN_TRADE,
            data={"token_id": token_id},
            timestamp=now - 1800  # 30 minutes ago
        ))
        
        # Set custom retention period for TOKEN_TRADE (15 minutes)
        self.history_manager.set_retention_period(
            EventType.TOKEN_TRADE,
            timedelta(minutes=15)
        )
        
        # Prune events
        pruned = self.history_manager.prune_events()
        
        # Should have pruned the TOKEN_TRADE event but kept TOKEN_CREATED
        self.assertEqual(pruned, 1)
        
        # Check remaining events
        events = self.history_manager.get_events(token_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.TOKEN_CREATED)
    
    def test_event_bus_integration(self):
        """Test integration with event bus."""
        # Check that subscribe was called for each tracked event type
        expected_calls = len(self.history_manager._event_types_to_track)
        self.assertEqual(self.event_bus.subscribe.call_count, expected_calls)
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        # Add some events
        token_id = "TEST_TOKEN"
        for i in range(10):
            self.history_manager.add_event(Event(
                event_type=EventType.TOKEN_TRADE,
                data={"token_id": token_id}
            ))
        
        # Get some events to increment retrieved counter
        self.history_manager.get_events(token_id)
        
        # Clear history to increment pruned counter
        self.history_manager.clear_token_history(token_id)
        
        # Get stats
        stats = self.history_manager.get_stats()
        
        # Check stats
        self.assertEqual(stats['events_added'], 10)
        self.assertEqual(stats['events_pruned'], 10)
        self.assertGreaterEqual(stats['events_retrieved'], 10)
        self.assertEqual(stats['current_events'], 0)  # All cleared
    
    def test_start_stop(self):
        """Test starting and stopping the history manager."""
        # Start the manager
        with patch.object(self.history_manager, '_schedule_pruning') as mock_schedule:
            self.history_manager.start()
            mock_schedule.assert_called_once()
        
        # Should be running
        self.assertTrue(self.history_manager._is_running)
        
        # Stop the manager
        with patch.object(threading.Timer, 'cancel') as mock_cancel:
            self.history_manager.stop()
            # Can't directly test the timer cancellation here
        
        # Should not be running
        self.assertFalse(self.history_manager._is_running)


if __name__ == '__main__':
    unittest.main() 