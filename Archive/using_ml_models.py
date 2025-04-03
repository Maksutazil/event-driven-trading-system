#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for using the ModelManager in a trading context.

This example demonstrates how to:
1. Create and configure the ModelManager
2. Load scikit-learn models
3. Create and register feature transformers
4. Make predictions with models
5. Update models with new data
6. Evaluate model performance

The example uses a simple price prediction model for a fictional trading scenario.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score

from src.core.ml.model_manager import DefaultModelManager
from src.core.ml.transformers.standard import StandardFeatureTransformer
from src.core.events.event_bus import EventBus
from src.core.events.event import Event, EventType
from src.core.events.subscribers import EventSubscriber


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEventSubscriber(EventSubscriber):
    """
    Example subscriber for model-related events.
    """
    
    def handle_event(self, event: Event) -> None:
        """Handle an incoming event."""
        if event.event_type == EventType.MODEL_LOADED:
            logger.info(f"Model loaded: {event.data.get('model_id')}")
        elif event.event_type == EventType.MODEL_UPDATED:
            logger.info(f"Model updated: {event.data.get('model_id')}")
        elif event.event_type == EventType.MODEL_PREDICTION:
            logger.info(f"Model prediction made: {event.data.get('model_id')}")


def create_dummy_price_data(days: int = 100) -> pd.DataFrame:
    """
    Create dummy price data for testing.
    
    Args:
        days: Number of days of data to generate
        
    Returns:
        DataFrame with generated price data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days)
    
    # Generate some price data with a trend and noise
    base_price = 100
    trend = np.linspace(0, 20, days)
    volatility = np.random.normal(0, 1, days) * 2
    
    # Add day-of-week seasonality
    day_of_week = dates.dayofweek
    weekday_effect = (day_of_week < 5) * 1.0  # Higher on weekdays
    
    prices = base_price + trend + volatility + weekday_effect
    
    # Calculate some features
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # 5-day moving averages
    ma5 = pd.Series(prices).rolling(5).mean().values
    ma5_diff = pd.Series(prices - ma5).fillna(0).values
    
    # Volatility (5-day rolling standard deviation)
    volatility = pd.Series(returns).rolling(5).std().fillna(0).values
    
    # Create the target: next day return (positive or negative)
    next_day_return = np.zeros_like(returns)
    next_day_return[:-1] = returns[1:]
    next_day_direction = (next_day_return > 0).astype(int)
    
    # Create the dataframe
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'return': returns,
        'ma5': ma5,
        'ma5_diff': ma5_diff,
        'volatility': volatility,
        'next_day_return': next_day_return,
        'next_day_direction': next_day_direction
    })
    
    return df


def train_price_prediction_model(data: pd.DataFrame) -> LinearRegression:
    """
    Train a simple price prediction model.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Trained model
    """
    # Use the first 80% of data for training
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    
    # Features for predicting next day return
    X = train_data[['return', 'ma5_diff', 'volatility']].values
    y = train_data['next_day_return'].values
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    return model


def train_direction_prediction_model(data: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a model to predict price direction (up or down).
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Trained model
    """
    # Use the first 80% of data for training
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    
    # Features for predicting next day direction
    X = train_data[['return', 'ma5_diff', 'volatility']].values
    y = train_data['next_day_direction'].values
    
    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model


def generate_latest_features(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate the latest features from price data.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with latest features
    """
    latest = data.iloc[-1]
    
    features = {
        'return': latest['return'],
        'ma5_diff': latest['ma5_diff'],
        'volatility': latest['volatility']
    }
    
    return features


def main():
    """Run the example."""
    # Create event bus
    event_bus = EventBus()
    
    # Create and register model event subscriber
    subscriber = ModelEventSubscriber()
    event_bus.subscribe(EventType.MODEL_LOADED, subscriber)
    event_bus.subscribe(EventType.MODEL_UPDATED, subscriber)
    event_bus.subscribe(EventType.MODEL_PREDICTION, subscriber)
    
    # Create model manager
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_manager = DefaultModelManager(
        model_dir=models_dir,
        event_bus=event_bus
    )
    
    # Generate price data
    logger.info("Generating price data...")
    price_data = create_dummy_price_data(days=100)
    
    # Train models
    logger.info("Training price prediction model...")
    price_model = train_price_prediction_model(price_data)
    
    logger.info("Training direction prediction model...")
    direction_model = train_direction_prediction_model(price_data)
    
    # Save models
    price_model_path = os.path.join(models_dir, 'price_prediction.joblib')
    direction_model_path = os.path.join(models_dir, 'direction_prediction.joblib')
    
    logger.info(f"Saving models to {models_dir}...")
    joblib.dump(price_model, price_model_path)
    joblib.dump(direction_model, direction_model_path)
    
    # Load models into the model manager
    logger.info("Loading models into the model manager...")
    price_model_id = 'price_prediction'
    direction_model_id = 'direction_prediction'
    
    model_manager.load_model(price_model_id, price_model_path)
    model_manager.load_model(direction_model_id, direction_model_path)
    
    # Create and register feature transformers
    logger.info("Creating and registering feature transformers...")
    feature_names = ['return', 'ma5_diff', 'volatility']
    
    price_transformer = StandardFeatureTransformer(
        features=feature_names,
        standardize=True
    )
    
    direction_transformer = StandardFeatureTransformer(
        features=feature_names,
        standardize=True
    )
    
    # Fit transformers with training data
    train_features = [
        {name: row[name] for name in feature_names}
        for _, row in price_data.iloc[:-20].iterrows()
    ]
    
    price_transformer.fit(train_features)
    direction_transformer.fit(train_features)
    
    # Register transformers
    model_manager.register_transformer(price_model_id, price_transformer)
    model_manager.register_transformer(direction_model_id, direction_transformer)
    
    # Make predictions with the latest data
    logger.info("Making predictions with the latest data...")
    latest_features = generate_latest_features(price_data)
    
    price_prediction = model_manager.get_prediction(price_model_id, latest_features)
    direction_prediction = model_manager.get_prediction(direction_model_id, latest_features)
    
    logger.info(f"Predicted next day return: {price_prediction[0]:.4f}")
    logger.info(f"Predicted next day direction: {'Up' if direction_prediction[0] == 1 else 'Down'}")
    
    # Evaluate models on test data
    logger.info("Evaluating models on test data...")
    test_data = price_data.iloc[-20:]
    
    test_features = [
        {name: row[name] for name in feature_names}
        for _, row in test_data.iterrows()
    ]
    
    # Evaluate price prediction model
    price_metrics = model_manager.evaluate_model(
        price_model_id, 
        test_features,
        test_data['next_day_return'].values
    )
    
    logger.info(f"Price prediction model metrics: {price_metrics}")
    
    # Evaluate direction prediction model
    direction_metrics = model_manager.evaluate_model(
        direction_model_id,
        test_features,
        test_data['next_day_direction'].values
    )
    
    logger.info(f"Direction prediction model metrics: {direction_metrics}")
    
    # Simulate updating the model with new data
    logger.info("Simulating model update with new data...")
    new_features = generate_latest_features(price_data)
    actual_return = 0.015  # Simulated actual return
    
    # In a real scenario, this would be called after observing the actual return
    model_manager.update_model(price_model_id, new_features, actual_return)
    
    logger.info("Example completed successfully.")


if __name__ == "__main__":
    main() 