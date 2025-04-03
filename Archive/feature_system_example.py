#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature System Example

This example demonstrates how to use the feature system to compute and manage
features for trading decisions.
"""

import sys
import logging
import time
from datetime import datetime, timedelta
import random
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import feature system components
from src.core.events import Event, EventType, EventBus
from src.core.features import (
    FeatureSystem, Feature, FeatureProvider,
    PriceMomentumSignalFeature, VolumeSpikeTradingSignalFeature,
    PriceFeatureProvider
)


def generate_mock_price_data(token_id, start_price=100.0, volatility=0.02, num_points=100):
    """
    Generate mock price data for a token.
    
    Args:
        token_id: ID of the token
        start_price: Starting price
        volatility: Price volatility
        num_points: Number of data points to generate
        
    Returns:
        List of (timestamp, price) tuples
    """
    price_data = []
    current_price = start_price
    current_time = datetime.now() - timedelta(minutes=num_points)
    
    for i in range(num_points):
        # Random price change with drift
        price_change = current_price * random.normalvariate(0.0001, volatility)
        current_price += price_change
        
        # Ensure price doesn't go negative
        current_price = max(0.01, current_price)
        
        # Add data point
        price_data.append((current_time, current_price))
        
        # Increment time
        current_time += timedelta(minutes=1)
    
    logger.info(f"Generated {len(price_data)} mock price points for {token_id}")
    return price_data


def simulate_price_update(price_feature_provider, token_id, price_data):
    """
    Simulate price updates for a token.
    
    Args:
        price_feature_provider: PriceFeatureProvider instance
        token_id: ID of the token
        price_data: List of (timestamp, price) tuples
    """
    for timestamp, price in price_data:
        price_feature_provider.update_price(token_id, price, timestamp)


def main():
    """Run the feature system example."""
    try:
        # Create event bus for feature updates
        logger.info("Creating event bus...")
        event_bus = EventBus()
        
        # Create feature system
        logger.info("Creating feature system...")
        feature_system = FeatureSystem(event_bus=event_bus)
        
        # Create and register feature providers
        logger.info("Creating feature providers...")
        price_provider = PriceFeatureProvider(name="example_price_provider")
        feature_system.register_provider(price_provider)
        
        # Create and register features
        logger.info("Creating features...")
        momentum_signal = PriceMomentumSignalFeature(threshold=2.0, sensitivity=1.2)
        feature_system.register_feature(momentum_signal)
        
        volume_signal = VolumeSpikeTradingSignalFeature(volume_threshold=2.5)
        feature_system.register_feature(volume_signal)
        
        # Set up a simple event handler for feature updates
        def on_feature_update(event):
            if event.event_type == EventType.FEATURE_UPDATE:
                data = event.data
                logger.info(f"Feature update: {data['feature_name']} = {data['value']:.4f} for {data['token_id']}")
        
        event_bus.subscribe(EventType.FEATURE_UPDATE, on_feature_update)
        
        # Generate mock price data
        token_id = "EXAMPLE_TOKEN_001"
        price_data = generate_mock_price_data(token_id, start_price=100.0, volatility=0.01, num_points=50)
        
        # Load initial price history
        logger.info("Loading initial price history...")
        simulate_price_update(price_provider, token_id, price_data[:30])
        
        # Compute initial features
        logger.info("\nComputing initial features...")
        context = {
            'token_id': token_id,
            'timestamp': price_data[30][0]
        }
        initial_features = feature_system.compute_features(context)
        
        logger.info(f"Initial features for {token_id}:")
        for name, value in initial_features.items():
            logger.info(f"  {name}: {value}")
        
        # Simulate real-time price updates and feature computation
        logger.info("\nSimulating real-time updates...")
        for timestamp, price in price_data[30:]:
            # Simulate a slight delay between updates
            time.sleep(0.1)
            
            # Create context with new price
            context = {
                'token_id': token_id,
                'price': price,
                'timestamp': timestamp
            }
            
            # Compute features
            features = feature_system.compute_features(context)
            
            # Log the signal values
            signal = features.get('price_momentum_signal')
            if signal is not None:
                signal_str = "NEUTRAL"
                if signal > 0.5:
                    signal_str = "STRONG BUY"
                elif signal > 0.2:
                    signal_str = "BUY"
                elif signal < -0.5:
                    signal_str = "STRONG SELL"
                elif signal < -0.2:
                    signal_str = "SELL"
                
                logger.info(f"Time: {timestamp}, Price: {price:.2f}, Signal: {signal:.4f} - {signal_str}")
        
        # Demonstrate cache functionality
        logger.info("\nDemonstrating cache functionality...")
        cache_stats = feature_system.cache.get_stats()
        logger.info(f"Cache stats: {cache_stats}")
        
        logger.info("Feature system example completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in feature system example: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 