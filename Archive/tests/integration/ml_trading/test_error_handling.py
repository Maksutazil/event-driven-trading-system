#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Cross-Component Error Handling

This module tests error handling and recovery mechanisms across component
boundaries, specifically between ML and Trading components.
"""

import unittest
import logging
import time
from typing import Dict, Any
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.core.events import EventType, Event
from src.core.ml.exceptions import ModelPredictionError, ModelNotFoundError
from src.core.trading.error_handler import SignalGenerationError
from tests.integration.base_integration_test import BaseIntegrationTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestCrossComponentErrorHandling(BaseIntegrationTest):
    """
    Integration tests for error handling across ML and Trading components.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        
        # Create and initialize components
        self.model_manager = self.create_ml_components()
        self.trading_engine = self.create_trading_components()
        
        # Create a test token
        self.test_token_id = "TEST_TOKEN_123"
        self.mock_features = self.generate_mock_features(self.test_token_id)
        
        # Track error events
        self.error_events_received = 0
        
        def on_error_event(event):
            if event.event_type == EventType.ERROR:
                self.error_events_received += 1
                logger.info(f"Error event received: {event.data.get('error_type')} - {event.data.get('message')}")
        
        # Subscribe to error events
        self.event_bus.subscribe(EventType.ERROR, on_error_event)
    
    def test_model_prediction_error_handled_by_signal_generator(self):
        """
        Test that model prediction errors are properly handled by the SignalGenerator.
        
        This test validates that:
        1. ML errors are published as events
        2. The SignalGenerator continues to function despite ML errors
        3. The system gracefully degrades (falls back to feature-based signals)
        """
        # Generate signals without model prediction first (baseline)
        baseline_signals = self.trading_engine.signal_generator.generate_signals(
            self.test_token_id,
            self.mock_features,
            datetime.now()
        )
        baseline_count = len(baseline_signals)
        
        # Now inject a problematic model prediction
        self.mock_features["model_prediction"] = "not_a_number"  # This will cause an error
        
        # The system should handle this and not crash
        with self.assertLogs(level='ERROR'):  # Expect error logs
            signals_with_error = self.trading_engine.signal_generator.generate_signals(
                self.test_token_id,
                self.mock_features,
                datetime.now()
            )
        
        # Verify signals were still generated despite the error
        self.assertGreater(len(signals_with_error), 0, 
                          "No signals generated when model prediction had an error")
        
        # Wait for error events to be processed
        time.sleep(0.5)
        
        # Verify error events were published
        self.assertGreater(self.error_events_received, 0, 
                           "No error events were published for model prediction error")
        
        # Clean up features
        del self.mock_features["model_prediction"]
    
    def test_signal_generation_error_recovery(self):
        """
        Test that signal generation errors don't crash the system and can be recovered from.
        
        This test validates that:
        1. SignalGenerator errors are properly caught and published
        2. The retry mechanism works for transient errors
        3. The system continues functioning after errors
        """
        # Patch the evaluate_model_prediction method to fail on first call, then recover
        original_method = self.trading_engine.signal_generator.evaluate_model_prediction
        call_count = [0]
        
        def failing_method(token_id, prediction, features, timestamp):
            call_count[0] += 1
            # Fail on first call, succeed on retry
            if call_count[0] == 1:
                raise SignalGenerationError("Simulated transient error in signal generation")
            return original_method(token_id, prediction, features, timestamp)
        
        self.trading_engine.signal_generator.evaluate_model_prediction = failing_method
        
        try:
            # Add model prediction
            self.mock_features["model_prediction"] = 0.8
            
            # Generate signals - should retry after error
            with self.assertLogs(level='ERROR'):  # Expect error logs
                signals = self.trading_engine.signal_generator.generate_signals(
                    self.test_token_id,
                    self.mock_features,
                    datetime.now()
                )
            
            # Verify signals were generated despite the error
            self.assertGreater(len(signals), 0, 
                              "No signals generated after error recovery")
            
            # Verify retry happened
            self.assertGreater(call_count[0], 1, 
                              "evaluate_model_prediction was not retried after error")
            
        finally:
            # Restore original method
            self.trading_engine.signal_generator.evaluate_model_prediction = original_method
    
    def test_model_not_found_error_handling(self):
        """
        Test handling of ModelNotFoundError across component boundaries.
        
        This test validates that:
        1. ModelNotFoundError is properly published as an event
        2. The trading system continues functioning without the model
        """
        # Patch get_prediction to raise ModelNotFoundError
        with patch.object(
            self.model_manager, 
            'get_prediction', 
            side_effect=ModelNotFoundError("test_model", "Model not found for testing")
        ):
            # Publish a model prediction event that will trigger an error
            event_data = {
                "token_id": self.test_token_id,
                "model_id": "test_model",
                "features": self.mock_features
            }
            
            event = Event(
                event_type=EventType.MODEL_PREDICTION,
                data=event_data,
                token_id=self.test_token_id,
                source="test"
            )
            
            # Publish the event - this should trigger model_manager.get_prediction
            with self.assertLogs(level='ERROR'):  # Expect error logs
                self.event_bus.publish(event)
            
            # Wait for error events to be processed
            time.sleep(0.5)
            
            # Verify error events were published
            self.assertGreater(self.error_events_received, 0, 
                               "No error events were published for ModelNotFoundError")
    
    def test_error_recovery_across_components(self):
        """
        Test end-to-end error recovery across component boundaries.
        
        This test validates that:
        1. Errors in one component don't crash the entire system
        2. Components can recover and continue processing after errors
        3. Error events are properly published and handled
        """
        # Reset error count
        self.error_events_received = 0
        
        # First, introduce an error in model_manager.get_prediction
        with patch.object(
            self.model_manager, 
            'get_prediction', 
            side_effect=ModelPredictionError("test_model", "Simulated prediction error")
        ):
            # Publish event that would trigger the error
            self.publish_mock_prediction(self.test_token_id, 0.8)
            
            # Wait for error events
            time.sleep(0.5)
            
            # Verify error was published
            self.assertGreater(self.error_events_received, 0, 
                               "No error events published for model prediction error")
        
        # Reset error count
        self.error_events_received = 0
        
        # Now, simulate recovery by allowing normal function
        # Publish another prediction event
        self.publish_mock_prediction(self.test_token_id, 0.7)
        
        # Verify no new errors
        time.sleep(0.5)
        self.assertEqual(self.error_events_received, 0, 
                         "Errors occurred after recovery")
        
        # Verify that EVENT_MODEL_PREDICTION was processed
        model_prediction_events = self.event_capture.get_events(EventType.MODEL_PREDICTION)
        self.assertGreater(len(model_prediction_events), 0, 
                          "MODEL_PREDICTION events not captured after recovery")


if __name__ == "__main__":
    unittest.main() 