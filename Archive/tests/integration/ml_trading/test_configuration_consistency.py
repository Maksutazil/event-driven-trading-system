#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Configuration Consistency

This module tests that configuration parameters are consistently applied
across system components, ensuring proper integration between ML and Trading
components with consistent configuration.
"""

import unittest
import logging
import json
import tempfile
import os
from typing import Dict, Any
from datetime import datetime

from src.core.events import EventType, Event
from src.core.trading import TradingErrorHandler
from src.core.ml.error_handler import MLErrorHandler
from src.core.trading.trading_factory import TradingSystemFactory
from tests.integration.base_integration_test import BaseIntegrationTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestConfigurationConsistency(BaseIntegrationTest):
    """
    Integration tests for configuration consistency across system components.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        
        # Create a temporary configuration file
        self.config_file = self._create_temp_config()
        
        # Create a simple price fetcher for testing
        def price_fetcher(token_id):
            return 100.0  # Return a fixed price for testing
        
        # Create the trading system with our configuration
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        # Create feature system and trading system
        self.feature_system = TradingSystemFactory.create_feature_system(
            event_bus=self.event_bus,
            config=self.config.get("feature_system", {})
        )
        
        self.trading_system = TradingSystemFactory.create_trading_system(
            event_bus=self.event_bus,
            feature_system=self.feature_system,
            price_fetcher=price_fetcher,
            config=self.config.get("trading_system", {})
        )
        
        # Extract individual components
        self.trading_engine = self.trading_system["trading_engine"]
        self.signal_generator = self.trading_system["signal_generator"]
        self.model_manager = self.trading_system["model_manager"]
        self.position_manager = self.trading_system["position_manager"]
    
    def tearDown(self) -> None:
        """Clean up after the test."""
        super().tearDown()
        
        # Clean up the temporary file
        os.unlink(self.config_file)
    
    def _create_temp_config(self) -> str:
        """Create a temporary configuration file for testing."""
        config = {
            "feature_system": {
                "max_cache_entries": 5000,
                "max_cache_age_minutes": 30
            },
            "trading_system": {
                "signal_threshold": 0.75,
                "signal_expiry_seconds": 120.0,
                "cooldown_seconds": 1800.0,
                "max_tokens_per_timepoint": 5,
                "model_prediction_weight": 0.6,
                "risk_parameters": {
                    "max_risk_per_trade": 0.02,
                    "max_open_positions": 10,
                    "default_stop_loss_pct": 0.05,
                    "take_profit_multiplier": 2.0
                },
                "error_handling": {
                    "max_retries": 3,
                    "retry_delay_seconds": 0.5,
                    "max_errors_per_token": 10
                }
            }
        }
        
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f, indent=2)
        
        return path
    
    def test_initial_configuration_propagation(self):
        """
        Test that initial configuration is correctly propagated to all components.
        
        This test verifies that configuration values specified in the config file
        are correctly applied to each component during system initialization.
        """
        # Verify trading engine configuration
        engine_params = self.trading_engine.get_parameters()
        self.assertEqual(engine_params["signal_threshold"], 
                        self.config["trading_system"]["signal_threshold"],
                        "Signal threshold not correctly propagated to trading engine")
        
        self.assertEqual(engine_params["cooldown_seconds"], 
                        self.config["trading_system"]["cooldown_seconds"],
                        "Cooldown seconds not correctly propagated to trading engine")
        
        self.assertEqual(engine_params["model_prediction_weight"], 
                        self.config["trading_system"]["model_prediction_weight"],
                        "Model prediction weight not correctly propagated to trading engine")
        
        # Verify signal generator configuration
        signal_params = self.signal_generator.get_signal_parameters()
        self.assertAlmostEqual(signal_params["model_weight"], 
                              self.config["trading_system"]["model_prediction_weight"],
                              msg="Model weight not correctly propagated to signal generator")
        
        # Verify risk management configuration
        risk_manager = self.trading_system["risk_manager"]
        risk_params = risk_manager.get_risk_parameters()
        
        self.assertEqual(risk_params["max_risk_per_trade"], 
                        self.config["trading_system"]["risk_parameters"]["max_risk_per_trade"],
                        "Max risk per trade not correctly propagated")
        
        self.assertEqual(risk_params["max_open_positions"], 
                        self.config["trading_system"]["risk_parameters"]["max_open_positions"],
                        "Max open positions not correctly propagated")
    
    def test_model_prediction_weight_consistency(self):
        """
        Test that model prediction weight is consistent across components.
        
        This test verifies that the model_prediction_weight parameter is
        consistently applied between the trading engine and signal generator.
        """
        # Get current values
        engine_weight = self.trading_engine.get_model_prediction_weight()
        signal_weight = self.signal_generator.get_signal_parameters()["model_weight"]
        
        # Verify they match
        self.assertAlmostEqual(engine_weight, signal_weight, 
                              msg="Model prediction weights don't match between components")
        
        # Update the value in the trading engine
        new_weight = 0.8
        self.trading_engine.set_model_prediction_weight(new_weight)
        
        # Verify the change propagated to the signal generator
        updated_signal_weight = self.signal_generator.get_signal_parameters()["model_weight"]
        self.assertAlmostEqual(updated_signal_weight, new_weight, 
                              msg="Model prediction weight change not propagated to signal generator")
    
    def test_error_handling_configuration(self):
        """
        Test that error handling configuration is consistent across components.
        
        This test verifies that error handling parameters are correctly
        propagated to both ML and Trading error handlers.
        """
        # Extract the ML and Trading error handlers
        ml_error_handler = MLErrorHandler()
        trading_error_handler = TradingErrorHandler()
        
        # Verify configuration is applied consistently
        ml_stats = ml_error_handler.get_error_statistics()
        trading_stats = trading_error_handler.get_error_statistics()
        
        # Both handlers should have the same retry configuration
        self.assertEqual(
            ml_stats.get("retry_settings", {}).get("max_attempts"),
            trading_stats.get("retry_settings", {}).get("max_attempts"),
            "Retry settings not consistent between ML and Trading error handlers"
        )
    
    def test_parameter_update_propagation(self):
        """
        Test that parameter updates are properly propagated to dependent components.
        
        This test verifies that when parameters are updated in one component,
        the changes are correctly propagated to dependent components.
        """
        # Update trading engine parameters
        new_params = {
            "signal_threshold": 0.85,
            "cooldown_seconds": 900.0,
            "model_prediction_weight": 0.7
        }
        
        self.trading_engine.set_parameters(new_params)
        
        # Verify parameters were updated in the trading engine
        engine_params = self.trading_engine.get_parameters()
        self.assertEqual(engine_params["signal_threshold"], new_params["signal_threshold"])
        self.assertEqual(engine_params["cooldown_seconds"], new_params["cooldown_seconds"])
        self.assertEqual(engine_params["model_prediction_weight"], new_params["model_prediction_weight"])
        
        # Verify model_prediction_weight was propagated to the signal generator
        signal_params = self.signal_generator.get_signal_parameters()
        self.assertAlmostEqual(signal_params["model_weight"], new_params["model_prediction_weight"],
                              msg="Model weight not propagated to signal generator after update")
    
    def test_feature_configuration_consistency(self):
        """
        Test that feature configuration is consistent between providers and consumers.
        
        This test verifies that feature configuration is consistent between
        feature providers and consumers (ML and Trading components).
        """
        # Add a test token
        test_token = "TEST_TOKEN_CONFIG"
        self.trading_engine.add_token(test_token)
        
        # Get required features from the signal generator and model manager
        signal_required_features = self.signal_generator.get_required_features()
        
        # Verify that all feature providers needed by the signal generator are registered
        feature_registry = self.feature_system.get_available_features()
        
        for feature in signal_required_features:
            self.assertIn(feature, feature_registry,
                         f"Required feature {feature} not in registry")
    
    def test_factory_configuration_merging(self):
        """
        Test that TradingSystemFactory correctly merges configurations.
        
        This test verifies that the factory properly merges default configuration
        with user-provided configuration.
        """
        # Create a partial configuration
        partial_config = {
            "signal_threshold": 0.9,
            "risk_parameters": {
                "max_risk_per_trade": 0.03
            }
        }
        
        # Merge with default configuration
        merged = TradingSystemFactory.merge_config({
            "signal_threshold": 0.5,
            "cooldown_seconds": 3600.0,
            "risk_parameters": {
                "max_risk_per_trade": 0.02,
                "max_open_positions": 5
            }
        }, partial_config)
        
        # Verify merged configuration contains overridden and default values
        self.assertEqual(merged["signal_threshold"], 0.9,
                        "Overridden signal_threshold not applied")
        
        self.assertEqual(merged["cooldown_seconds"], 3600.0,
                        "Default cooldown_seconds not preserved")
        
        self.assertEqual(merged["risk_parameters"]["max_risk_per_trade"], 0.03,
                        "Overridden max_risk_per_trade not applied")
        
        self.assertEqual(merged["risk_parameters"]["max_open_positions"], 5,
                        "Default max_open_positions not preserved in nested dict")
    
    def test_configuration_event_propagation(self):
        """
        Test that configuration changes are properly propagated via events.
        
        This test verifies that configuration changes trigger appropriate
        events and are propagated to dependent components.
        """
        # Subscribe to configuration events
        config_events = []
        
        def on_config_event(event: Event):
            if event.event_type == EventType.SYSTEM_STATUS and "configuration" in event.data:
                config_events.append(event)
        
        self.event_bus.subscribe(EventType.SYSTEM_STATUS, on_config_event)
        
        # Update trading engine parameters, which should trigger a system status event
        self.trading_engine.set_parameters({
            "signal_threshold": 0.95
        })
        
        # Wait for events to be processed
        self.event_bus._process_event(Event(
            event_type=EventType.SYSTEM_STATUS, 
            data={"message": "Test event to flush queue"}
        ))
        
        # Verify that a configuration event was published
        self.assertGreater(len(config_events), 0, 
                          "No configuration events were published")
        
        # Verify the event contains the updated parameter
        found_update = False
        for event in config_events:
            config = event.data.get("configuration", {})
            if "signal_threshold" in config and config["signal_threshold"] == 0.95:
                found_update = True
                break
        
        self.assertTrue(found_update, 
                       "Configuration update event not found with correct value")


if __name__ == "__main__":
    unittest.main() 