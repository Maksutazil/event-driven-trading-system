#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Model Training Example

This example demonstrates how to train models without a database connection,
using in-memory synthetic data instead.
"""

import logging
import json
import os
import asyncio
import random
from datetime import datetime, timedelta

from src.core.events import EventBus, EventType
from src.core.data import DataFeedManager
from src.core.features import DefaultFeatureManager
from src.core.ml.training.model_trainer import ModelTrainer
from src.core.ml.models.pump_predictor import PumpPredictorModel
from src.core.ml.models.early_pump_predictor import EarlyPumpPredictor
from src.core.features.interfaces import FeatureProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Create mock implementations of the feature providers
class MockPumpDetectionFeatureProvider(FeatureProvider):
    """Mock implementation of PumpDetectionFeatureProvider for testing."""
    
    def __init__(self, data_feed):
        self._data_feed = data_feed
        self._name = "MockPumpDetectionProvider"
    
    @property
    def name(self) -> str:
        """Get the name of the provider."""
        return self._name
    
    @property
    def provides(self) -> set:
        """Get the set of features provided."""
        return {
            'price_velocity', 'price_acceleration', 'volume_surge_ratio',
            'volume_volatility', 'buy_sell_volume_ratio', 'price_deviation',
            'pump_pattern_score', 'dump_pattern_score', 'pump_phase_detection',
            'abnormal_activity_score'
        }
    
    def get_features(self, context: dict) -> dict:
        """Get values for all provided features."""
        token_id = context.get('token_id')
        if not token_id:
            return {}
        
        # Get token data
        trades = self._data_feed.get_historical_data(token_id=token_id)
        
        if not trades:
            return {}
        
        # For this mock, generate simple synthetic feature values
        results = {}
        
        # Basic features based on the token ID
        if token_id in ['token_2', 'token_5', 'token_8']:  # Known pump tokens
            results['price_velocity'] = random.uniform(0.01, 0.05)
            results['volume_surge_ratio'] = random.uniform(3.0, 8.0)
            results['pump_pattern_score'] = random.uniform(0.7, 0.95)
            results['dump_pattern_score'] = random.uniform(0.0, 0.3)
            results['abnormal_activity_score'] = random.uniform(0.7, 0.9)
        elif token_id == 'token_10':  # Early pump token
            results['price_velocity'] = random.uniform(0.005, 0.02)
            results['volume_surge_ratio'] = random.uniform(2.0, 4.0)
            results['pump_pattern_score'] = random.uniform(0.4, 0.7)
            results['dump_pattern_score'] = random.uniform(0.0, 0.1)
            results['abnormal_activity_score'] = random.uniform(0.5, 0.7)
        else:  # Normal tokens
            results['price_velocity'] = random.uniform(-0.005, 0.01)
            results['volume_surge_ratio'] = random.uniform(0.8, 2.0)
            results['pump_pattern_score'] = random.uniform(0.0, 0.3)
            results['dump_pattern_score'] = random.uniform(0.0, 0.2)
            results['abnormal_activity_score'] = random.uniform(0.1, 0.4)
        
        return results


class MockEarlyPumpDetectionProvider(FeatureProvider):
    """Mock implementation of EarlyPumpDetectionProvider for testing."""
    
    def __init__(self, data_feed):
        self._data_feed = data_feed
        self._name = "MockEarlyPumpDetectionProvider"
    
    @property
    def name(self) -> str:
        """Get the name of the provider."""
        return self._name
    
    @property
    def provides(self) -> set:
        """Get the set of features provided."""
        return {
            'immediate_price_change', 'trade_frequency',
            'buyer_dominance', 'volume_intensity', 'early_pump_score'
        }
    
    def get_features(self, context: dict) -> dict:
        """Get values for all provided features."""
        token_id = context.get('token_id')
        if not token_id:
            return {}
        
        # Get token data
        trades = self._data_feed.get_historical_data(token_id=token_id)
        
        if not trades:
            return {}
        
        # For this mock, generate simple synthetic feature values
        results = {}
        
        # Basic features based on the token ID
        if token_id == 'token_10':  # Known early pump token
            results['immediate_price_change'] = random.uniform(0.03, 0.08)
            results['trade_frequency'] = random.uniform(0.8, 1.5)
            results['buyer_dominance'] = random.uniform(0.7, 0.9)
            results['volume_intensity'] = random.uniform(1.5, 3.0)
            results['early_pump_score'] = random.uniform(0.7, 0.9)
        elif token_id in ['token_2', 'token_5', 'token_8']:  # Later stage pump tokens
            results['immediate_price_change'] = random.uniform(0.01, 0.04)
            results['trade_frequency'] = random.uniform(0.5, 1.0)
            results['buyer_dominance'] = random.uniform(0.6, 0.8)
            results['volume_intensity'] = random.uniform(1.0, 2.0)
            results['early_pump_score'] = random.uniform(0.4, 0.6)
        else:  # Normal tokens
            results['immediate_price_change'] = random.uniform(-0.01, 0.02)
            results['trade_frequency'] = random.uniform(0.2, 0.7)
            results['buyer_dominance'] = random.uniform(0.4, 0.6)
            results['volume_intensity'] = random.uniform(0.5, 1.5)
            results['early_pump_score'] = random.uniform(0.1, 0.3)
        
        return results


class SimpleModelTrainingDemo:
    """
    A simplified demo of training machine learning models using synthetic data.
    """
    
    def __init__(self, models_dir="models"):
        """
        Initialize the Simple Model Training Demo.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize components
        self.event_bus = EventBus()
        self.data_feed_manager = DataFeedManager(event_bus=self.event_bus)
        self.feature_manager = DefaultFeatureManager(event_bus=self.event_bus)
        
        # Set up feature providers
        self.pump_provider = None
        self.early_provider = None
        
        # Set up models
        self.pump_predictor = None
        self.early_predictor = None
        
        # Set up model trainer
        self.trainer = None
        
        logger.info("Initialized SimpleModelTrainingDemo")

    def create_synthetic_data(self, num_tokens=10, days=3):
        """Create synthetic data for testing."""
        logger.info(f"Creating synthetic data for {num_tokens} tokens over {days} days")
        
        data = {}
        
        # Current timestamp at midnight
        end_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=days)
        
        for token_id in range(1, num_tokens + 1):
            token_id = f"token_{token_id}"
            trades = []
            
            # Generate trades over the time period
            current_time = start_time
            price = random.uniform(0.0001, 0.1)  # Initial price
            
            while current_time < end_time:
                # Add some randomness to trade frequency
                num_trades = random.randint(1, 10)
                
                for _ in range(num_trades):
                    # Create a synthetic trade
                    price_change = random.uniform(-0.1, 0.15)  # More likely to go up
                    price = max(0.00001, price * (1 + price_change))
                    
                    volume = random.uniform(0.1, 10.0)
                    is_buy = random.random() > 0.4  # More buys than sells
                    
                    trade = {
                        'token_id': token_id,
                        'timestamp': current_time.timestamp(),
                        'price': price,
                        'volume': volume,
                        'is_buy': is_buy,
                        'tx_hash': f"0x{random.randint(0, 0xffffffff):08x}"
                    }
                    trades.append(trade)
                
                # Move forward in time
                current_time += timedelta(minutes=random.randint(1, 30))
                
            # Create some pump patterns for some tokens
            if token_id in ['token_2', 'token_5', 'token_8']:
                # Add a pump pattern
                pump_start = start_time + timedelta(hours=random.randint(24, 48))
                pump_time = pump_start
                base_price = trades[-1]['price']
                
                # Fast rising trades
                for _ in range(20):
                    price = base_price * (1 + random.uniform(0.05, 0.2))
                    base_price = price
                    volume = random.uniform(1.0, 20.0)
                    
                    trade = {
                        'token_id': token_id,
                        'timestamp': pump_time.timestamp(),
                        'price': price,
                        'volume': volume,
                        'is_buy': True,
                        'tx_hash': f"0x{random.randint(0, 0xffffffff):08x}"
                    }
                    trades.append(trade)
                    
                    pump_time += timedelta(minutes=random.randint(1, 5))
            
            # Add an early pump pattern to one token
            if token_id == 'token_10':
                # This token just started trading
                recent_time = end_time - timedelta(hours=2)
                trades = []  # Reset trades to only have very recent ones
                
                # Initial trades
                current_price = random.uniform(0.0001, 0.001)
                for i in range(10):
                    # First few trades with minimal movement
                    trade_time = recent_time + timedelta(minutes=i*5)
                    
                    # Small initial price movements
                    price_change = random.uniform(-0.02, 0.03)
                    current_price = max(0.00001, current_price * (1 + price_change))
                    
                    trade = {
                        'token_id': token_id,
                        'timestamp': trade_time.timestamp(),
                        'price': current_price,
                        'volume': random.uniform(0.1, 1.0),
                        'is_buy': random.random() > 0.4,
                        'tx_hash': f"0x{random.randint(0, 0xffffffff):08x}"
                    }
                    trades.append(trade)
                
                # Then add the beginning of a pump pattern
                for i in range(5):
                    trade_time = recent_time + timedelta(minutes=50 + i*2)
                    current_price = current_price * (1 + random.uniform(0.03, 0.08))
                    
                    trade = {
                        'token_id': token_id,
                        'timestamp': trade_time.timestamp(),
                        'price': current_price,
                        'volume': random.uniform(0.5, 3.0),
                        'is_buy': True,  # All buys
                        'tx_hash': f"0x{random.randint(0, 0xffffffff):08x}"
                    }
                    trades.append(trade)
            
            # Sort by timestamp
            trades.sort(key=lambda x: x['timestamp'])
            data[token_id] = trades
        
        return data
    
    async def setup(self):
        """Set up feature providers, models, and trainer."""
        logger.info("Setting up components")
        
        # Implement a mock get_historical_data method in the data feed manager
        synthetic_data = self.create_synthetic_data()
        
        # Create a closure to access synthetic data
        def get_historical_data(token_id=None, start_time=None, end_time=None, limit=None):
            if token_id is None:
                return list(synthetic_data.keys())
            
            if token_id not in synthetic_data:
                return []
                
            trades = synthetic_data[token_id]
            
            if start_time:
                trades = [t for t in trades if t['timestamp'] >= start_time.timestamp()]
            
            if end_time:
                trades = [t for t in trades if t['timestamp'] <= end_time.timestamp()]
                
            if limit:
                trades = trades[:limit]
                
            return trades
        
        # Monkey patch the data_feed_manager to use our synthetic data
        self.data_feed_manager.get_historical_data = get_historical_data
        
        # Create and register feature providers
        self.pump_provider = MockPumpDetectionFeatureProvider(self.data_feed_manager)
        self.early_provider = MockEarlyPumpDetectionProvider(self.data_feed_manager)
        
        self.feature_manager.register_provider(self.pump_provider)
        self.feature_manager.register_provider(self.early_provider)
        
        # Create models
        self.pump_predictor = PumpPredictorModel(
            model_id="pump_predictor_v1",
            feature_manager=self.feature_manager
        )
        
        self.early_predictor = EarlyPumpPredictor(
            model_id="early_pump_predictor_v1",
            feature_manager=self.feature_manager
        )
        
        # Create model trainer (simplified)
        self.trainer = ModelTrainer(
            feature_manager=self.feature_manager,
            models_dir=self.models_dir,
            debug_mode=True
        )
        
        # Add mock get_token_data method to the trainer
        self.trainer.get_token_data = get_historical_data
        
        # Since we're not using a real database (postgres_manager), we need to create a mock
        # This is to handle the postgres_manager requirement in ModelTrainer
        class MockPostgresManager:
            def __init__(self):
                self.connected = True
                
            async def get_all_tokens(self):
                return [{'id': token_id} for token_id in synthetic_data.keys()]
                
            async def get_trades_for_token(self, token_id, **kwargs):
                return synthetic_data.get(token_id, [])
        
        # Set a mock postgres_manager for trainer
        self.trainer.postgres_manager = MockPostgresManager()
        
        # Register models with trainer
        self.register_models_with_trainer()
        
        logger.info("Setup completed successfully")
        return True
    
    def register_models_with_trainer(self):
        """Register models with the model trainer."""
        if not hasattr(self.trainer, 'models'):
            self.trainer.models = []
            
        # Add models to trainer's model list
        if self.pump_predictor:
            self.trainer.models.append(self.pump_predictor)
            logger.info(f"Registered PumpPredictorModel with trainer: {self.pump_predictor.model_id}")
            
        if self.early_predictor:
            self.trainer.models.append(self.early_predictor)
            logger.info(f"Registered EarlyPumpPredictor with trainer: {self.early_predictor.model_id}")
    
    async def run_training(self):
        """
        Run the model training process using synthetic data.
        """
        if self.trainer is None:
            logger.error("Trainer not set up. Call setup() first.")
            return
            
        # Define training time period (last 3 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        
        logger.info(f"Starting training session for synthetic data")
        
        # Create labeled data (in a real application, this would come from a database)
        labeled_data = self.create_labeled_data()
        
        # Get token IDs
        token_ids = self.data_feed_manager.get_historical_data()
        
        # Run training session
        logger.info(f"Starting training with {len(token_ids)} tokens and {len(labeled_data)} labeled events")
        
        results = await self.trainer.train_models(
            token_ids=token_ids,
            start_date=start_date,
            end_date=end_date,
            labeled_data=labeled_data
        )
        
        # Log results
        logger.info("Training completed!")
        logger.info(f"Processed {len(token_ids)} tokens")
            
        # Save results to JSON file
        results_file = os.path.join(self.models_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training results saved to {results_file}")
        
        return results
    
    def create_labeled_data(self):
        """
        Create synthetic labeled data for supervised learning.
        
        Returns:
            Dictionary with labeled event data
        """
        labeled_data = {
            "token_2": [
                {
                    "event_type": "PUMP",
                    "timestamp": (datetime.now() - timedelta(days=1)).timestamp(),
                    "confidence": 0.9
                }
            ],
            "token_5": [
                {
                    "event_type": "PUMP",
                    "timestamp": (datetime.now() - timedelta(hours=36)).timestamp(),
                    "confidence": 0.85
                }
            ],
            "token_8": [
                {
                    "event_type": "PUMP",
                    "timestamp": (datetime.now() - timedelta(hours=12)).timestamp(),
                    "confidence": 0.95
                }
            ],
            "token_10": [
                {
                    "event_type": "EARLY_PUMP",
                    "timestamp": (datetime.now() - timedelta(minutes=30)).timestamp(),
                    "confidence": 0.8
                }
            ]
        }
        
        logger.info(f"Created labeled data with {len(labeled_data)} tokens")
        return labeled_data


async def main():
    """Main function to run the demo."""
    # Create the demo instance
    demo = SimpleModelTrainingDemo(models_dir="models_test")
    
    try:
        # Set up the demo
        setup_success = await demo.setup()
        if not setup_success:
            logger.error("Failed to set up the demo. Exiting.")
            return
        
        # Run the training process
        await demo.run_training()
        
    except Exception as e:
        logger.exception(f"Error in demo: {e}")
    finally:
        logger.info("Demo completed")
    
    logger.info("Example finished - check the models_test directory for results")


if __name__ == "__main__":
    asyncio.run(main()) 