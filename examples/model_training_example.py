#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Example

This script demonstrates how to use the ModelTrainer to train pump detection models
using historical data from a PostgreSQL database with 2-second intervals.
"""

import asyncio
import logging
import os
import json
import random
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
from typing import Dict, List, Optional, Any, Tuple, Set
from psycopg2.extras import DictCursor
import pandas as pd

from src.core.events import EventBus, EventType
from src.core.db import PostgresDataManager
from src.core.data import DataFeedManager
from src.core.features import DefaultFeatureManager
from src.core.ml.training.model_trainer import ModelTrainer
from src.core.ml.models.pump_predictor import PumpPredictorModel
from src.core.ml.models.early_pump_predictor import EarlyPumpPredictor
from src.core.ml.models.pump_dump_detection_model import PumpDumpDetectionModel
from src.core.features.providers.pump_detection_provider import PumpDetectionFeatureProvider
from src.core.features.providers.early_pump_detection_provider import EarlyPumpDetectionProvider
from src.core.data.interfaces import DataFeedInterface


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainingDemo:
    """
    Demonstration of model training using PostgreSQL database.
    """
    
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "pumpfun_monitor",
        db_user: str = "postgres",
        db_password: str = "postgres"
    ):
        """
        Initialize the model training demo.
        
        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
        """
        self.postgres_manager = PostgresDataManager(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
        )
        self.feature_manager = DefaultFeatureManager()
        self.model_trainer = None
        self.pump_detection_provider = None
        self.early_pump_detection_provider = None
        self.data_feed = None
        
    async def setup(self) -> bool:
        """
        Set up the components for the training demo.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        # Connect to the database
        success = await self.postgres_manager.connect()
        if not success:
            logger.error("Failed to connect to the database")
            return False
            
        # Check database connectivity
        try:
            # Try to get one token to validate connection
            tokens = await self.postgres_manager.get_all_tokens(limit=1)
            if not tokens:
                logger.warning("Connected to database but no tokens found")
            else:
                logger.info(f"Successfully connected to database and found tokens")
                
            # Check for token_trades
            token_id = None
            if tokens and len(tokens) > 0:
                token_id = tokens[0].get('token_id')
                        
            if token_id:
                # Try to get a few trades to verify trade data access
                trades = await self.postgres_manager.get_trades_for_token(
                    token_id=token_id,
                    limit=5
                )
                if trades and len(trades) > 0:
                    logger.info(f"Successfully retrieved sample trades from database")
                else:
                    logger.warning(f"No trades found for token {token_id}")
            
        except Exception as e:
            logger.error(f"Error checking database connectivity: {e}", exc_info=True)
            
        # Create a simple event bus for our components
        event_bus = EventBus()
        
        # Create a data feed that can use our postgres_manager to get historical data
        class PostgresDataFeed:
            def __init__(self, postgres_manager):
                self.postgres_manager = postgres_manager
                
            def get_historical_data(self, token_id, start_time=None, end_time=None):
                # Synchronous version that doesn't create a new event loop
                try:
                    # Get the current event loop if available
                    current_loop = asyncio.get_event_loop()
                    if current_loop.is_running():
                        logger.warning(f"Event loop is already running, using synchronous fallback for token {token_id}")
                        # Use a synchronous approach instead
                        trades = []
                        if hasattr(self.postgres_manager, 'conn'):
                            cursor = self.postgres_manager.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                            try:
                                # Build query with parameters
                                query_params = [token_id]
                                
                                # Get the table name, with fallback
                                trades_table = "token_trades"  # Default table name
                                if hasattr(self.postgres_manager, 'tables') and isinstance(self.postgres_manager.tables, dict):
                                    trades_table = self.postgres_manager.tables.get('trades', 'token_trades')
                                
                                logger.debug(f"Using table '{trades_table}' for synchronous query")
                                
                                query = f"""
                                    SELECT 
                                        trade_id, token_id, timestamp, type,
                                        token_amount, sol_amount, price_sol,
                                        market_cap, trader_address
                                    FROM {trades_table}
                                    WHERE token_id = %s
                                    ORDER BY timestamp ASC
                                    LIMIT 10000
                                """
                                
                                logger.debug(f"Executing synchronous query: {query} with params {query_params}")
                                cursor.execute(query, query_params)
                                
                                # Convert results to list of dicts
                                trades = cursor.fetchall()
                                if trades:
                                    logger.info(f"Retrieved {len(trades)} trades for {token_id} from synchronous query")
                                    trades = [dict(trade) for trade in trades]
                                    
                                    # Only log details about the first trade to avoid spam
                                    if len(trades) > 0:
                                        sample_trade = trades[0]
                                        logger.debug(f"First trade keys: {list(sample_trade.keys())}")
                                        
                                        # Only log a few key fields, not the entire trade
                                        important_fields = {k: v for k, v in sample_trade.items() 
                                                           if k in ['trade_id', 'timestamp', 'price_sol']}
                                        logger.debug(f"Sample trade fields: {important_fields}")
                                    
                                    # Normalize trade data to ensure consistent field names
                                    trades = [self._normalize_trade_data(trade) for trade in trades]
                                    
                                    # After normalization, only log if there's an issue with timestamp
                                    if trades and len(trades) > 0 and 'timestamp' not in trades[0]:
                                        logger.warning(f"Normalized trade missing timestamp. Original keys: {list(sample_trade.keys())}")
                                else:
                                    logger.warning(f"No trades found for token {token_id} in synchronous query")
                            except Exception as e:
                                logger.error(f"Error executing synchronous query for {token_id}: {e}")
                                return []
                            finally:
                                cursor.close()
                        return trades
                    else:
                        # Using asyncio directly (not inside another event loop)
                        result = current_loop.run_until_complete(
                            self._get_historical_data_async(token_id, start_time, end_time)
                        )
                        return result
                except Exception as e:
                    logger.error(f"Error in get_historical_data for {token_id}: {e}")
                    return []
                    
            def _normalize_trade_data(self, trade):
                """Normalize trade data to ensure consistent field names."""
                # Convert to dict if not already
                if not isinstance(trade, dict):
                    if hasattr(trade, '_asdict'):  # Handle namedtuple-like objects
                        trade = trade._asdict()
                    elif hasattr(trade, '__dict__'):  # Handle objects with __dict__
                        trade = trade.__dict__
                    else:
                        trade = dict(trade)  # Try to convert other types
                
                normalized = dict(trade)  # Make a copy to avoid modifying the original
                
                # We'll avoid timestamp field debug logging for every trade as it's too spammy
                # Only log warnings if timestamp is missing
                if 'timestamp' not in normalized:
                    logger.warning(f"No timestamp field in original trade data. Keys: {list(normalized.keys())}")
                
                # Map price_sol to price if needed
                if 'price_sol' in normalized and 'price' not in normalized:
                    normalized['price'] = normalized['price_sol']
                
                # Map volume field
                if 'token_amount' in normalized and 'volume' not in normalized:
                    normalized['volume'] = normalized['token_amount']
                
                # Map type information if needed
                if 'type' in normalized and 'is_buy' not in normalized:
                    trade_type = normalized['type']
                    if isinstance(trade_type, str):
                        normalized['is_buy'] = trade_type.lower() == 'buy'
                
                # Ensure timestamp is a datetime object
                if 'timestamp' in normalized:
                    timestamp = normalized['timestamp']
                    if not isinstance(timestamp, datetime):
                        try:
                            if isinstance(timestamp, (int, float)):
                                # Unix timestamp
                                if timestamp > 1e12:  # Milliseconds
                                    normalized['timestamp'] = datetime.fromtimestamp(timestamp / 1000)
                                else:  # Seconds
                                    normalized['timestamp'] = datetime.fromtimestamp(timestamp)
                            elif isinstance(timestamp, str):
                                # ISO format or similar
                                normalized['timestamp'] = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except Exception as e:
                            logger.warning(f"Failed to convert timestamp: {e}")
                
                return normalized
                
            async def _get_historical_data_async(self, token_id, start_time=None, end_time=None):
                """Async method to get historical data."""
                try:
                    # Call the postgres_manager's get_trades_for_token method
                    trades = await self.postgres_manager.get_trades_for_token(
                        token_id=token_id,
                        limit=10000,  # Get enough data for analysis
                        order_by="timestamp ASC"  # Order from oldest to newest
                    )
                    
                    if trades:
                        logger.info(f"Retrieved {len(trades)} trades for {token_id} from async query")
                        
                        # Normalize the result from the async method
                        trades = [self._normalize_trade_data(trade) for trade in trades]
                        
                        # Only log details for the first trade to avoid spam
                        if len(trades) > 0:
                            sample_trade = trades[0]
                            # Only log important fields
                            important_fields = {k: v for k, v in sample_trade.items() 
                                              if k in ['trade_id', 'timestamp', 'price']}
                            logger.debug(f"First trade after normalization (sample fields): {important_fields}")
                    else:
                        logger.warning(f"No trades found for token {token_id} in async query")
                        
                    return trades
                except Exception as e:
                    logger.error(f"Error in _get_historical_data_async: {e}")
                    return []
                
        self.data_feed = PostgresDataFeed(self.postgres_manager)
        
        # Create a simple feature manager that doesn't rely on providers
        class SimpleFeatureManager:
            def __init__(self):
                self.providers = {}
                
            def register_provider(self, provider):
                self.providers[provider.name] = provider
                
            def get_feature_providers(self):
                return list(self.providers.values())
                
            def get_providers(self):
                return self.get_feature_providers()
                
            def get_provider_for_feature(self, feature_name):
                for provider in self.providers.values():
                    if feature_name in provider.provides:
                        return provider
                return None
                
        # Use our simplified feature manager instead
        self.feature_manager = SimpleFeatureManager()
        
        # Create and register feature providers
        pump_provider = PumpDetectionFeatureProvider(self.data_feed)
        early_pump_provider = EarlyPumpDetectionProvider(self.data_feed)
        
        # Register providers with the feature manager
        self.feature_manager.register_provider(pump_provider)
        self.feature_manager.register_provider(early_pump_provider)
        
        logger.info(f"Registered PumpDetectionFeatureProvider and EarlyPumpDetectionProvider with feature manager")
        
        # Initialize model trainer
        self.model_trainer = await self._create_model_trainer()
        
        return True
        
    async def create_labeled_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a labeled dataset focusing on tokens with high trade counts.
        
        This method queries tokens with the most trades and analyzes their 
        price patterns to identify pump and dump events.
        
        Returns:
            Dictionary mapping token IDs to lists of labeled events
        """
        logger.info("Creating labeled dataset focusing on tokens with high trade counts")
        try:
            # Create a cursor for database operations that returns rows as dictionaries
            cursor = self.postgres_manager.conn.cursor(cursor_factory=DictCursor)
            
            # Find tokens with the most trades (at least 20 trades)
            cursor.execute("""
                SELECT token_id, COUNT(*) as trade_count
                FROM token_trades
                GROUP BY token_id
                HAVING COUNT(*) >= 20
                ORDER BY trade_count DESC
                LIMIT 10
            """)
            
            token_rows = cursor.fetchall()
            if not token_rows:
                logger.warning("No tokens found with sufficient trades for analysis")
                return {}
                
            logger.info(f"Found {len(token_rows)} tokens with sufficient trades for analysis")
            
            labeled_data = {}
            
            # Process each token with high trade count
            for row in token_rows:
                token_id = row['token_id']
                trade_count = row['trade_count']
                logger.info(f"Processing token {token_id} with {trade_count} trades")
                
                # Get trades for this token without time filtering
                trades = await self.postgres_manager.get_trades_for_token(
                    token_id=token_id,
                    limit=5000  # Get a large number of trades
                )
                
                if not trades:
                    logger.warning(f"No trades found for token {token_id}")
                    continue
                    
                logger.info(f"Retrieved {len(trades)} trades for token {token_id}")
                
                # Analyze price patterns to create labeled events
                labeled_events = self._analyze_price_patterns(token_id)
                
                if labeled_events:
                    labeled_data[token_id] = labeled_events
                    logger.info(f"Created {len(labeled_events)} labeled events for token {token_id}")
                else:
                    logger.info(f"No labeled events created for token {token_id}")
            
            return labeled_data
            
        except Exception as e:
            logger.error(f"Error creating labeled dataset: {e}", exc_info=True)
            return {}
    
    def _analyze_price_patterns(self, token_id):
        """Analyze price patterns for a specific token to identify pump and dump events."""
        try:
            logger.info(f"Analyzing price patterns for token {token_id}")
            
            # Get all historical data for the token without time filtering
            trade_data = self.data_feed.get_historical_data(token_id=token_id)
            
            if not trade_data:
                logger.warning(f"No trade data available for token {token_id}")
                return None
            
            # Log the first few trades to verify data
            sample_size = min(3, len(trade_data))
            if sample_size > 0:
                sample_trades = trade_data[:sample_size]
                logger.debug(f"Sample trades for {token_id}: {sample_trades}")
            
            # Convert trade data to pandas DataFrame for analysis
            df = pd.DataFrame(trade_data)
            
            # Check if timestamp column exists
            if 'timestamp' not in df.columns:
                logger.warning(f"No timestamp column found in trade data for {token_id}. Available columns: {df.columns.tolist()}")
                return None
            
            # Check if we have price information
            if 'price' not in df.columns and 'price_sol' not in df.columns:
                logger.warning(f"No price information found in trade data for {token_id}")
                return None
                
            # Use price_sol if price is not available
            if 'price' not in df.columns and 'price_sol' in df.columns:
                df['price'] = df['price_sol']
            
            # Ensure timestamp is in datetime format
            if pd.api.types.is_object_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Simple implementation of pump/dump detection
            labeled_events = []
            
            if len(df) >= 10:  # Need a minimum number of trades
                # Calculate price changes
                df['price_pct_change'] = df['price'].pct_change(5)  # 5-period change
                
                # Define thresholds for pump and dump detection
                PUMP_THRESHOLD = 0.10  # 10% price increase
                DUMP_THRESHOLD = -0.10  # 10% price decrease
                
                # Identify potential pump events
                pump_indices = df[df['price_pct_change'] > PUMP_THRESHOLD].index
                for idx in pump_indices:
                    if idx > 5:  # Skip first few entries due to rolling calculation
                        start_idx = max(0, idx - 5)
                        end_idx = min(len(df) - 1, idx + 5)
                        labeled_events.append({
                            'type': 'pump',
                            'start_time': df.iloc[start_idx]['timestamp'],
                            'end_time': df.iloc[end_idx]['timestamp'],
                            'class_id': 1,
                            'token_id': token_id
                        })
                        logger.info(f"Detected PUMP event for {token_id} at {df.iloc[idx]['timestamp']}")
                
                # Identify potential dump events
                dump_indices = df[df['price_pct_change'] < DUMP_THRESHOLD].index
                for idx in dump_indices:
                    if idx > 5:  # Skip first few entries due to rolling calculation
                        start_idx = max(0, idx - 5)
                        end_idx = min(len(df) - 1, idx + 5)
                        labeled_events.append({
                            'type': 'dump',
                            'start_time': df.iloc[start_idx]['timestamp'],
                            'end_time': df.iloc[end_idx]['timestamp'],
                            'class_id': 2,
                            'token_id': token_id
                        })
                        logger.info(f"Detected DUMP event for {token_id} at {df.iloc[idx]['timestamp']}")
                
                # Add some normal periods (no significant price movement)
                normal_indices = df[(df['price_pct_change'] > -0.03) & (df['price_pct_change'] < 0.03)].index
                if len(normal_indices) > 0:
                    # Take a sample of normal periods
                    import random
                    sample_size = min(len(labeled_events), len(normal_indices))
                    if sample_size > 0:
                        normal_samples = random.sample(list(normal_indices), sample_size)
                        for idx in normal_samples:
                            if idx > 5:  # Skip first few entries
                                start_idx = max(0, idx - 5)
                                end_idx = min(len(df) - 1, idx + 5)
                                labeled_events.append({
                                    'type': 'normal',
                                    'start_time': df.iloc[start_idx]['timestamp'],
                                    'end_time': df.iloc[end_idx]['timestamp'],
                                    'class_id': 0,
                                    'token_id': token_id
                                })
            
            if labeled_events:
                logger.info(f"Found {len(labeled_events)} labeled events for {token_id}")
                return labeled_events
            else:
                logger.info(f"No significant price patterns found for {token_id}")
                return None
            
        except Exception as e:
            logger.exception(f"Error analyzing price patterns for {token_id}: {str(e)}")
            return None
    
    async def _create_model_trainer(self) -> ModelTrainer:
        """
        Create and initialize the model trainer.
        
        Returns:
            Initialized ModelTrainer object
        """
        logger.info("Creating model trainer")
        
        # Create model trainer with database connection and feature manager
        model_trainer = ModelTrainer(
            postgres_manager=self.postgres_manager,
            feature_manager=self.feature_manager,
            models_dir="./trained_models",
            time_interval_seconds=2,
            training_window_hours=24,
            validation_ratio=0.2,
            debug_mode=True
        )
        
        logger.info("Model trainer created successfully")
        return model_trainer

    async def run_training(self) -> bool:
        """
        Run the model training session with a limited set of tokens for testing.
        
        This method selects only the top 10 tokens with most trades to validate
        the training process works correctly before running full training.
        
        Returns:
            True if training was successful, False otherwise
        """
        logger.info("Starting model training process with limited token set")
        try:
            # Step 1: Use existing model trainer from setup
            if not self.model_trainer:
                logger.error("Model trainer not initialized. Call setup() first.")
                return False
                
            # Step 2: Get top 10 tokens with most trades for testing
            cursor = self.postgres_manager.conn.cursor(cursor_factory=DictCursor)
            cursor.execute("""
                SELECT token_id, COUNT(*) as trade_count
                FROM token_trades
                GROUP BY token_id
                ORDER BY trade_count DESC
                LIMIT 10
            """)
            
            # Fetch results
            token_rows = cursor.fetchall()
            test_tokens = [row['token_id'] for row in token_rows]
            
            if not test_tokens:
                logger.warning("No tokens found for testing, proceeding with full training")
            else:
                logger.info(f"Selected {len(test_tokens)} tokens for test training: {', '.join(test_tokens[:3])}...")
            
            # Step 3: Create labeled dataset
            labeled_data = await self.create_labeled_dataset()
            if not labeled_data:
                logger.warning("No labeled data was created. Creating and registering model anyway.")
            else:
                logger.info(f"Created labeled data for {len(labeled_data)} tokens")
            
            # Create and register models - this step is crucial!
            model = PumpDumpDetectionModel()
            
            # Set features required for the model
            model.set_required_features([
                'price_velocity',
                'volume_surge_ratio',
                'buy_sell_ratio',
                'price_oscillation',
                'abnormal_activity_score',
                'immediate_price_change',
                'trade_frequency',
                'buyer_dominance',
                'volume_intensity',
                'early_pump_score'
            ])
            
            # Configure model parameters
            model.set_parameters({
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'class_weight': 'balanced'
            })
            
            # Register model with trainer
            self.model_trainer.register_model(model)
            logger.info(f"Registered model: {model.model_id}")
            
            # Set time window to include February 2025 data
            # All trades are from 2025-02-26, so ensure we're covering that date
            end_time = datetime(2025, 3, 1, 0, 0, 0)  # March 1, 2025
            start_time = datetime(2025, 2, 1, 0, 0, 0)  # February 1, 2025
            
            # Log the training window
            logger.info(f"Training with data from {start_time} to {end_time} for test tokens only")
            
            # Run training session with time window and limited token list
            results = await self.model_trainer.run_training_session(
                token_ids=test_tokens,  # Only use the test tokens
                labeled_data=labeled_data,
                start_date=start_time,
                end_date=end_time
            )
            
            # Save results to JSON file
            os.makedirs('./trained_models', exist_ok=True)
            results_file = f"./trained_models/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {results_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            return False
            
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")
        if self.postgres_manager.connected:
            await self.postgres_manager.disconnect()
        logger.info("Cleanup completed")


async def main():
    # Create the model training demo
    demo = ModelTrainingDemo(
        # Configure database connection
        db_host=os.environ.get('DB_HOST', 'localhost'),
        db_port=int(os.environ.get('DB_PORT', '5432')),
        db_name=os.environ.get('DB_NAME', 'pumpfun_monitor'),
        db_user=os.environ.get('DB_USER', 'postgres'),
        db_password=os.environ.get('DB_PASSWORD', 'postgres')
    )
    
    # Set up the demo
    success = await demo.setup()
    if not success:
        logger.error("Failed to set up model training demo")
        return
        
    # Run training
    success = await demo.run_training()
    if not success:
        logger.error("Training failed")
    
    # Clean up
    await demo.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True) 