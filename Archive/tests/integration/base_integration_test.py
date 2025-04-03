#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Integration Test Module

This module provides base classes and utilities for integration testing
between different components of the event-driven trading system.
"""

import unittest
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Set, Type
from datetime import datetime, timedelta

from src.core.events import EventBus, Event, EventType, BaseEventSubscriber
from src.core.features import FeatureSystem, FeatureRegistry
from src.core.ml import DefaultModelManager
from src.core.trading import DefaultTradingEngine, DefaultSignalGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EventCapture(BaseEventSubscriber):
    """
    Utility class to capture and track events for testing purposes.
    """
    
    def __init__(self, event_bus: EventBus, event_types: Optional[List[EventType]] = None):
        """
        Initialize the event capture.
        
        Args:
            event_bus: The event bus to subscribe to
            event_types: Optional list of event types to capture (defaults to all)
        """
        super().__init__(event_bus)
        self.captured_events: Dict[EventType, List[Event]] = {}
        self.event_counts: Dict[EventType, int] = {}
        self._lock = threading.RLock()
        
        # Subscribe to specified event types or all if none specified
        if event_types:
            for event_type in event_types:
                self.subscribe(event_type)
                self.captured_events[event_type] = []
                self.event_counts[event_type] = 0
        else:
            # Subscribe to all event types
            for event_type in EventType:
                self.subscribe(event_type)
                self.captured_events[event_type] = []
                self.event_counts[event_type] = 0
    
    def on_event(self, event: Event) -> None:
        """
        Process received events.
        
        Args:
            event: The event to process
        """
        with self._lock:
            if event.event_type not in self.captured_events:
                self.captured_events[event.event_type] = []
            
            self.captured_events[event.event_type].append(event)
            self.event_counts[event.event_type] = self.event_counts.get(event.event_type, 0) + 1
            
            logger.debug(f"Captured event: {event}")
    
    def get_events(self, event_type: EventType) -> List[Event]:
        """
        Get captured events of a specific type.
        
        Args:
            event_type: The event type to retrieve
            
        Returns:
            List of captured events of the specified type
        """
        with self._lock:
            return self.captured_events.get(event_type, [])
    
    def get_count(self, event_type: EventType) -> int:
        """
        Get the count of events of a specific type.
        
        Args:
            event_type: The event type to count
            
        Returns:
            Count of events of the specified type
        """
        with self._lock:
            return self.event_counts.get(event_type, 0)
    
    def wait_for_event(self, event_type: EventType, count: int = 1, timeout: float = 5.0) -> bool:
        """
        Wait for a specific number of events of a type to be captured.
        
        Args:
            event_type: The event type to wait for
            count: The number of events to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if the events were captured within the timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if self.event_counts.get(event_type, 0) >= count:
                    return True
            time.sleep(0.1)
        return False
    
    def clear(self) -> None:
        """Clear all captured events."""
        with self._lock:
            for event_type in self.captured_events:
                self.captured_events[event_type] = []
                self.event_counts[event_type] = 0


class BaseIntegrationTest(unittest.TestCase):
    """
    Base class for integration tests that provides common setup and utilities.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        # Create event bus for test
        self.event_bus = EventBus()
        
        # Create feature system with registry
        self.feature_registry = FeatureRegistry()
        self.feature_system = FeatureSystem(event_bus=self.event_bus)
        
        # Create event capture for monitoring events
        self.event_capture = EventCapture(self.event_bus)
        
        # Start event processing
        self.event_bus.start_processing()
        
        logger.info("Integration test environment set up")
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        # Stop event processing
        self.event_bus.stop_processing(wait_for_queue=True)
        
        # Unsubscribe event capture
        self.event_capture.unsubscribe_all()
        
        logger.info("Integration test environment torn down")
    
    def create_ml_components(self) -> DefaultModelManager:
        """
        Create ML components for testing.
        
        Returns:
            Initialized ModelManager
        """
        model_manager = DefaultModelManager(self.event_bus, self.feature_system)
        return model_manager
    
    def create_trading_components(self) -> DefaultTradingEngine:
        """
        Create trading components for testing.
        
        Returns:
            Initialized TradingEngine
        """
        signal_generator = DefaultSignalGenerator(self.event_bus)
        trading_engine = DefaultTradingEngine(
            event_bus=self.event_bus,
            feature_system=self.feature_system,
            position_manager=None,  # Mock this in tests
            trade_executor=None,    # Mock this in tests
            risk_manager=None,      # Mock this in tests
            signal_generator=signal_generator
        )
        return trading_engine
    
    def generate_mock_features(self, token_id: str) -> Dict[str, Any]:
        """
        Generate mock features for testing.
        
        Args:
            token_id: The token ID to generate features for
            
        Returns:
            Dictionary of mock features
        """
        return {
            "token_id": token_id,
            "current_price": 100.0,
            "volume_1h": 10000.0,
            "price_change_24h": 0.05,
            "rsi_14": 65.0,
            "macd_signal": 0.2,
            "timestamp": datetime.now()
        }
    
    def publish_mock_prediction(self, token_id: str, prediction_value: float) -> Event:
        """
        Publish a mock model prediction event.
        
        Args:
            token_id: The token ID for the prediction
            prediction_value: The prediction value
            
        Returns:
            The published event
        """
        event_data = {
            "token_id": token_id,
            "model_id": "test_model",
            "prediction": prediction_value,
            "confidence": 0.8,
            "features_used": ["current_price", "volume_1h", "rsi_14"],
            "timestamp": datetime.now().isoformat()
        }
        
        event = Event(
            event_type=EventType.MODEL_PREDICTION,
            data=event_data,
            token_id=token_id,
            source="test_model_manager"
        )
        
        self.event_bus.publish(event)
        return event
    
    def wait_for_events(self, expectations: Dict[EventType, int], timeout: float = 5.0) -> bool:
        """
        Wait for multiple expected events to be captured.
        
        Args:
            expectations: Dictionary mapping event types to expected counts
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if all expected events were captured, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_met = True
            for event_type, expected_count in expectations.items():
                if self.event_capture.get_count(event_type) < expected_count:
                    all_met = False
                    break
            
            if all_met:
                return True
            
            time.sleep(0.1)
        
        # Log which expectations weren't met
        for event_type, expected_count in expectations.items():
            actual_count = self.event_capture.get_count(event_type)
            if actual_count < expected_count:
                logger.warning(f"Expected {expected_count} {event_type.name} events, got {actual_count}")
        
        return False 