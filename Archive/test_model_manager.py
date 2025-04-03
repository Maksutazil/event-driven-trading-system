#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the DefaultModelManager implementation.
"""

import os
import sys
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
import joblib
import tempfile
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core.ml.model_manager import DefaultModelManager
from src.core.ml.transformers.standard import StandardFeatureTransformer
from src.core.ml.exceptions import (
    ModelNotFoundError, ModelLoadError, ModelSaveError, ModelUpdateError, ModelPredictionError,
    TransformerNotFoundError, TransformerFitError, TransformerTransformError
)
from src.core.events.event_bus import EventBus
from src.core.features.feature_system import FeatureSystem

def create_mock_feature_manager(event_bus, price_data):
    """
    Create a mock feature manager for testing.
    
    Args:
        event_bus: Event bus
        price_data: Price data DataFrame
        
    Returns:
        FeatureSystem instance
    """
    feature_system = FeatureSystem(event_bus=event_bus)
    
    # Instead of using the actual PriceFeatureProvider, we'll use a simple mock
    # by monkey patching the FeatureSystem's get_features_for_token method
    latest = price_data.iloc[-1]
    test_features = {
        'return': latest['return'],
        'ma5_diff': latest['ma5_diff'],
        'volatility': latest['volatility']
    }
    
    # Create a mock method
    def mock_get_features_for_token(self, token_id, features=None):
        return test_features
    
    # Apply the monkey patch
    import types
    feature_system.get_features_for_token = types.MethodType(mock_get_features_for_token, feature_system)
    
    return feature_system

def create_test_price_data(n_points=100):
    """
    Create synthetic price data for testing.
    
    Returns:
        DataFrame with price data and features
    """
    np.random.seed(42)
    
    # Generate random prices with a slight upward trend
    prices = np.cumsum(np.random.normal(0.001, 0.01, n_points)) + 100
    
    df = pd.DataFrame({
        'price': prices,
    })
    
    # Calculate returns
    df['return'] = df['price'].pct_change().fillna(0)
    
    # Simple moving average 5
    df['ma5'] = df['price'].rolling(5).mean().fillna(method='bfill')
    df['ma5_diff'] = (df['price'] - df['ma5']) / df['price']
    
    # Volatility
    df['volatility'] = df['return'].rolling(10).std().fillna(0)
    
    return df

def create_and_save_test_model(features, targets, temp_dir):
    """
    Create and save a test model for testing.
    
    Args:
        features: Feature matrix
        targets: Target values
        temp_dir: Directory to save the model
        
    Returns:
        Path to the saved model
    """
    # Create a simple linear regression model
    model = LinearRegression()
    model.fit(features, targets)
    
    # Save the model
    model_path = os.path.join(temp_dir, "test_model.joblib")
    joblib.dump(model, model_path)
    
    return model_path

def test_model_manager():
    """Test the DefaultModelManager implementation."""
    
    # Create temporary directory for models
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create event bus
        event_bus = EventBus()
        
        # Create synthetic price data
        price_data = create_test_price_data(n_points=100)
        
        # Create feature manager
        feature_manager = create_mock_feature_manager(event_bus, price_data)
        
        # Prepare features and targets for model training
        X = price_data[['return', 'ma5_diff', 'volatility']].iloc[:-1].values
        y = price_data['return'].iloc[1:].values  # Predict next return
        
        # Create and save test model
        model_path = create_and_save_test_model(X, y, temp_dir)
        
        # Create model manager
        model_manager = DefaultModelManager(
            model_directory=temp_dir,
            feature_system=feature_manager,
            event_bus=event_bus
        )
        
        # Test loading a model
        model_id = "test_model"
        try:
            success = model_manager.load_model(model_id, model_path, "regression")
            assert success, "Failed to load model"
            print(f"✓ Successfully loaded model {model_id}")
        except ModelLoadError as e:
            pytest.fail(f"Model loading failed: {str(e)}")
        
        # Test loading a non-existent model
        non_existent_path = os.path.join(temp_dir, "non_existent.joblib")
        try:
            model_manager.load_model("bad_model", non_existent_path, "regression")
            pytest.fail("Should have raised ModelLoadError for non-existent model")
        except ModelLoadError:
            print("✓ Correctly raised ModelLoadError for non-existent model")
        
        # Test invalid model type
        try:
            model_manager.load_model("invalid_type", model_path, "invalid_type")
            pytest.fail("Should have raised InvalidModelTypeError for invalid model type")
        except Exception:
            print("✓ Correctly raised exception for invalid model type")
        
        # Create a transformer
        transformer = StandardFeatureTransformer(
            features=['return', 'ma5_diff', 'volatility'],
            standardize=True
        )
        
        # Fit the transformer
        transformer.fit(price_data[['return', 'ma5_diff', 'volatility']].to_dict('records'))
        
        # Register the transformer
        try:
            model_manager.register_transformer(model_id, transformer)
            print(f"✓ Successfully registered transformer for model {model_id}")
        except Exception as e:
            pytest.fail(f"Failed to register transformer: {str(e)}")
        
        # Test getting a prediction
        token_id = "test_token"
        try:
            prediction = model_manager.get_prediction(model_id, token_id)
            print(f"Prediction: {prediction}")
            assert isinstance(prediction, float), "Prediction should be a float"
            print("✓ Successfully got prediction")
        except ModelPredictionError as e:
            pytest.fail(f"Failed to get prediction: {str(e)}")
        
        # Test getting prediction for a non-existent model
        try:
            model_manager.get_prediction("non_existent_model", token_id)
            pytest.fail("Should have raised ModelNotFoundError for non-existent model")
        except ModelNotFoundError:
            print("✓ Correctly raised ModelNotFoundError for prediction with non-existent model")
        
        # Test updating the model
        target = 0.005  # Example target
        try:
            model_manager.update_model(model_id, token_id, target)
            print("✓ Successfully updated model")
        except ModelUpdateError as e:
            pytest.fail(f"Failed to update model: {str(e)}")
        
        # Test updating a non-existent model
        try:
            model_manager.update_model("non_existent_model", token_id, target)
            pytest.fail("Should have raised ModelNotFoundError for non-existent model")
        except ModelNotFoundError:
            print("✓ Correctly raised ModelNotFoundError for update with non-existent model")
        
        # Test getting performance
        try:
            performance = model_manager.get_model_performance(model_id)
            print(f"Performance: {performance}")
            assert isinstance(performance, dict), "Performance should be a dictionary"
            print("✓ Successfully got model performance")
        except Exception as e:
            pytest.fail(f"Failed to get model performance: {str(e)}")
        
        # Test getting performance for a non-existent model
        try:
            model_manager.get_model_performance("non_existent_model")
            pytest.fail("Should have raised ModelNotFoundError for non-existent model")
        except ModelNotFoundError:
            print("✓ Correctly raised ModelNotFoundError for performance with non-existent model")
        
        # Test listing models
        models = model_manager.list_models()
        print(f"Models: {models}")
        assert model_id in models, "Model ID should be in the list of models"
        print("✓ Successfully listed models")
        
        # Test getting model metadata
        try:
            metadata = model_manager.get_model_metadata(model_id)
            print(f"Metadata: {metadata}")
            assert isinstance(metadata, dict), "Metadata should be a dictionary"
            print("✓ Successfully got model metadata")
        except Exception as e:
            pytest.fail(f"Failed to get model metadata: {str(e)}")
        
        # Test getting metadata for a non-existent model
        try:
            model_manager.get_model_metadata("non_existent_model")
            pytest.fail("Should have raised ModelNotFoundError for non-existent model")
        except ModelNotFoundError:
            print("✓ Correctly raised ModelNotFoundError for metadata with non-existent model")
        
        # Test saving the model
        try:
            new_path = os.path.join(temp_dir, "test_model_updated.joblib")
            success = model_manager.save_model(model_id, new_path)
            assert success, "Failed to save model"
            assert os.path.exists(new_path), "Model file should exist"
            print("✓ Successfully saved model")
        except ModelSaveError as e:
            pytest.fail(f"Failed to save model: {str(e)}")
        
        # Test saving a non-existent model
        try:
            model_manager.save_model("non_existent_model", os.path.join(temp_dir, "bad_model.joblib"))
            pytest.fail("Should have raised ModelNotFoundError for non-existent model")
        except ModelNotFoundError:
            print("✓ Correctly raised ModelNotFoundError for saving non-existent model")
        
        # Test saving the transformer
        try:
            transformer_path = os.path.join(temp_dir, "test_transformer.pkl")
            success = model_manager.save_transformer(model_id, transformer_path)
            assert success, "Failed to save transformer"
            assert os.path.exists(transformer_path), "Transformer file should exist"
            print("✓ Successfully saved transformer")
        except Exception as e:
            pytest.fail(f"Failed to save transformer: {str(e)}")
        
        # Test saving transformer for a non-existent model
        try:
            model_manager.save_transformer("non_existent_model", os.path.join(temp_dir, "bad_transformer.pkl"))
            pytest.fail("Should have raised TransformerNotFoundError for non-existent model")
        except TransformerNotFoundError:
            print("✓ Correctly raised TransformerNotFoundError for saving transformer with non-existent model")
        
        # Test model evaluation
        try:
            # Create test features and targets
            test_features = []
            test_targets = []
            for i in range(10):
                features = {
                    'return': 0.01 * (i + 1),
                    'ma5_diff': 0.005 * (i + 1),
                    'volatility': 0.002 * (i + 1)
                }
                test_features.append(features)
                test_targets.append(0.02 * (i + 1))
            
            metrics = model_manager.evaluate_model(model_id, test_features, test_targets)
            print(f"Evaluation metrics: {metrics}")
            assert 'mse' in metrics, "MSE should be in metrics for regression model"
            assert 'mae' in metrics, "MAE should be in metrics for regression model"
            assert 'r2' in metrics, "R² should be in metrics for regression model"
            print("✓ Successfully evaluated model")
        except Exception as e:
            pytest.fail(f"Failed to evaluate model: {str(e)}")
        
        print("\nAll tests passed!")

if __name__ == "__main__":
    test_model_manager() 