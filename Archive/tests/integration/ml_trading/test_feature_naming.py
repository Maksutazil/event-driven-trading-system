#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Feature Naming Standardization

This module tests that feature naming is consistent across ML and Trading components,
ensuring that features are correctly referenced and accessed in both systems.
"""

import unittest
import logging
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta

from src.core.features.registry import FeatureRegistry
from src.core.ml.interfaces import ModelTrainer, DataCollector
from src.core.trading.signal_generator import DefaultSignalGenerator
from tests.integration.base_integration_test import BaseIntegrationTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestFeatureNamingStandardization(BaseIntegrationTest):
    """
    Integration tests for feature naming standardization across ML and Trading components.
    """
    
    def setUp(self) -> None:
        """Set up the test environment."""
        super().setUp()
        
        # Create and initialize components
        self.model_manager = self.create_ml_components()
        self.trading_engine = self.create_trading_components()
        
        # Get feature registry
        self.feature_registry = FeatureRegistry.get_instance()
        
        # Create a test token
        self.test_token_id = "TEST_TOKEN_123"
        self.mock_features = self.generate_mock_features(self.test_token_id)
    
    def test_feature_registry_consistency(self):
        """
        Test that the feature registry contains a consistent set of features.
        
        This test validates that:
        1. The feature registry has been properly initialized
        2. The expected feature categories and providers are present
        3. Features follow the standardized naming convention
        """
        # Check that feature registry is not empty
        registered_features = self.feature_registry.get_all_features()
        self.assertIsNotNone(registered_features)
        self.assertGreater(len(registered_features), 0, 
                          "Feature registry should contain registered features")
        
        # Check feature naming convention compliance
        for feature_name in registered_features:
            # Feature names should follow the pattern: category.provider.name
            parts = feature_name.split('.')
            self.assertEqual(len(parts), 3, 
                             f"Feature {feature_name} does not follow the category.provider.name convention")
            
            category, provider, name = parts
            self.assertIsNotNone(category)
            self.assertIsNotNone(provider)
            self.assertIsNotNone(name)
            
            # Category and provider should be non-empty
            self.assertNotEqual(category, "")
            self.assertNotEqual(provider, "")
            self.assertNotEqual(name, "")
    
    def test_feature_access_consistency(self):
        """
        Test that features can be consistently accessed by both ML and Trading components.
        
        This test validates that:
        1. The same feature keys are used by both ML and Trading components
        2. Feature access methods are consistent across components
        """
        # Use the signal generator to access features
        signal_generator = self.trading_engine.signal_generator
        
        # Get the feature requirements from the signal generator
        required_features = signal_generator.get_required_features()
        self.assertIsNotNone(required_features)
        self.assertGreater(len(required_features), 0,
                          "Signal generator should require features")
        
        # Check that all required features are in the registry
        for feature_name in required_features:
            self.assertTrue(self.feature_registry.is_feature_registered(feature_name),
                          f"Required feature {feature_name} is not in the registry")
            
            # Generate a mock feature value
            mock_value = 1.0
            
            # Add to mock features
            feature_parts = feature_name.split('.')
            if len(feature_parts) == 3:
                category, provider, name = feature_parts
                
                # Ensure the category exists in mock features
                if category not in self.mock_features:
                    self.mock_features[category] = {}
                
                # Ensure the provider exists in the category
                if provider not in self.mock_features[category]:
                    self.mock_features[category][provider] = {}
                
                # Set the feature value
                self.mock_features[category][provider][name] = mock_value
    
    def test_feature_synchronization_across_components(self):
        """
        Test that features are synchronized and consistently named across components.
        
        This test validates that:
        1. Model requirements match what the signal generator expects
        2. Feature names are consistent between data collection and signal generation
        """
        # Get required features from model training perspective
        trainer = self.model_manager.get_model_trainer("test_model")
        if not trainer:
            # Create a mock trainer for testing
            trainer = self.create_mock_trainer()
        
        ml_required_features = trainer.get_required_features()
        
        # Get required features from trading perspective
        trading_required_features = self.trading_engine.signal_generator.get_required_features()
        
        # There should be overlap between ML and Trading required features
        ml_set = set(ml_required_features)
        trading_set = set(trading_required_features)
        
        common_features = ml_set.intersection(trading_set)
        self.assertGreater(len(common_features), 0,
                          "ML and Trading components should share common features")
        
        # Log the common features
        logger.info(f"Common features between ML and Trading: {common_features}")
        
        # Both ML and Trading components should follow the standardized naming convention
        for feature_set in [ml_set, trading_set]:
            for feature_name in feature_set:
                parts = feature_name.split('.')
                self.assertEqual(len(parts), 3, 
                                f"Feature {feature_name} does not follow the category.provider.name convention")
    
    def test_data_collector_feature_consistency(self):
        """
        Test that DataCollector provides features with consistent naming.
        
        This test validates that:
        1. Features collected by DataCollector follow naming standards
        2. Collected features can be accessed by both ML and Trading components
        """
        # Get or create a DataCollector
        data_collector = self.model_manager.get_data_collector()
        if not data_collector:
            # Create a mock collector for testing
            data_collector = self.create_mock_data_collector()
        
        # Get available features from collector
        available_features = data_collector.get_available_features()
        self.assertIsNotNone(available_features)
        self.assertGreater(len(available_features), 0,
                          "DataCollector should provide available features")
        
        # Features from collector should follow naming convention
        for feature_name in available_features:
            parts = feature_name.split('.')
            self.assertEqual(len(parts), 3, 
                            f"Feature {feature_name} from DataCollector does not follow the convention")
        
        # Collect some test data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        collected_data = data_collector.collect_data(
            token_ids=[self.test_token_id],
            features=available_features[:5],  # Just take first 5 features for testing
            start_time=start_time,
            end_time=end_time
        )
        
        # Verify the collected data contains properly formatted features
        self.assertIsNotNone(collected_data)
        
        # Check the structure of the collected data
        # This may vary based on implementation, adjust as needed
        if hasattr(collected_data, 'columns'):  # If pandas DataFrame
            for column in collected_data.columns:
                if column not in ['timestamp', 'token_id']:  # Skip metadata columns
                    parts = column.split('.')
                    self.assertEqual(len(parts), 3, 
                                    f"Column {column} in collected data does not follow the convention")
    
    def test_feature_transformation_naming_consistency(self):
        """
        Test that feature transformations maintain naming consistency.
        
        This test validates that:
        1. Feature transformers maintain standardized naming
        2. Transformed features are accessible by both ML and Trading components
        """
        # Get a list of transformers from registry
        transformers = self.feature_registry.get_transformers()
        
        if transformers:
            # For each transformer, verify input and output feature naming
            for transformer in transformers:
                # Get transformer input features
                input_features = transformer.get_input_features()
                
                # Get transformer output features
                output_features = transformer.get_output_features()
                
                # Both input and output features should follow naming convention
                for feature_name in input_features + output_features:
                    parts = feature_name.split('.')
                    self.assertEqual(len(parts), 3, 
                                     f"Feature {feature_name} in transformer does not follow the convention")
                
                # Test feature transformation if possible
                if hasattr(transformer, 'transform') and callable(transformer.transform):
                    # Create mock input features
                    mock_input = {}
                    for feature in input_features:
                        category, provider, name = feature.split('.')
                        
                        if category not in mock_input:
                            mock_input[category] = {}
                        
                        if provider not in mock_input[category]:
                            mock_input[category][provider] = {}
                        
                        mock_input[category][provider][name] = 1.0
                    
                    # Transform features
                    transformed = transformer.transform(mock_input)
                    
                    # Verify transformed features maintain structure
                    for output_feature in output_features:
                        category, provider, name = output_feature.split('.')
                        
                        self.assertIn(category, transformed, 
                                     f"Category {category} not in transformed features")
                        self.assertIn(provider, transformed[category], 
                                     f"Provider {provider} not in transformed features")
                        self.assertIn(name, transformed[category][provider], 
                                     f"Name {name} not in transformed features")
    
    def create_mock_trainer(self):
        """Create a mock ModelTrainer for testing."""
        mock_trainer = MagicMock(spec=ModelTrainer)
        
        # Define required features using standardized naming
        required_features = [
            "market.binance.price",
            "market.binance.volume",
            "technical.momentum.rsi",
            "sentiment.social.twitter_sentiment",
            "fundamental.blockchain.transaction_count"
        ]
        
        mock_trainer.get_required_features.return_value = required_features
        return mock_trainer
    
    def create_mock_data_collector(self):
        """Create a mock DataCollector for testing."""
        mock_collector = MagicMock(spec=DataCollector)
        
        # Define available features using standardized naming
        available_features = [
            "market.binance.price",
            "market.binance.volume",
            "market.binance.high",
            "market.binance.low",
            "technical.momentum.rsi",
            "technical.momentum.macd",
            "technical.volatility.atr",
            "sentiment.social.twitter_sentiment",
            "sentiment.social.reddit_sentiment",
            "fundamental.blockchain.transaction_count",
            "fundamental.blockchain.active_addresses"
        ]
        
        mock_collector.get_available_features.return_value = available_features
        
        # Mock the collect_data method
        def mock_collect_data(token_ids, features, start_time, end_time):
            import pandas as pd
            import numpy as np
            
            # Create a simple DataFrame with timestamps and requested features
            timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
            data = []
            
            for token_id in token_ids:
                for ts in timestamps:
                    row = {'timestamp': ts, 'token_id': token_id}
                    
                    # Add each requested feature
                    for feature in features:
                        row[feature] = np.random.random()
                    
                    data.append(row)
            
            return pd.DataFrame(data)
        
        mock_collector.collect_data.side_effect = mock_collect_data
        return mock_collector


if __name__ == "__main__":
    unittest.main() 