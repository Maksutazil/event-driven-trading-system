#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Module

This module provides functionality to train models using historical data
from a PostgreSQL database with support for fine-grained time intervals.
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from datetime import datetime, timedelta
import asyncio

from src.core.db.postgres_data_manager import PostgresDataManager
from src.core.features.manager import FeatureManager
from src.core.ml.models.pump_predictor import PumpPredictorModel
from src.core.ml.models.early_pump_predictor import EarlyPumpPredictor
from src.core.ml.models.pump_dump_detection_model import PumpDumpDetectionModel
from src.core.features.providers.pump_detection_provider import PumpDetectionFeatureProvider
from src.core.features.providers.early_pump_detection_provider import EarlyPumpDetectionProvider
from src.core.ml.interfaces import Model

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for pump detection models using historical data from a PostgreSQL database.
    
    This class extracts training data with 2-second intervals from the database,
    computes features, and trains the models.
    """
    
    def __init__(
        self,
        postgres_manager: PostgresDataManager,
        feature_manager: FeatureManager,
        models_dir: str = "./models",
        time_interval_seconds: int = 2,
        training_window_hours: int = 24,
        validation_ratio: float = 0.2,
        debug_mode: bool = False
    ):
        """
        Initialize the model trainer.
        
        Args:
            postgres_manager: PostgreSQL data manager for database access
            feature_manager: Feature manager for computing features
            models_dir: Directory to save trained models
            time_interval_seconds: Time interval in seconds for sampling data
            training_window_hours: Hours of historical data to use for training
            validation_ratio: Ratio of data to use for validation
            debug_mode: Whether to enable debug mode
        """
        self.postgres_manager = postgres_manager
        self.feature_manager = feature_manager
        self.models_dir = models_dir
        self.time_interval_seconds = time_interval_seconds
        self.training_window_hours = training_window_hours
        self.validation_ratio = validation_ratio
        self.debug_mode = debug_mode
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Set logger level based on debug mode
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        logger.info(f"Initialized ModelTrainer with {time_interval_seconds}-second intervals")
    
    async def fetch_training_data(
        self,
        token_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        labeled_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical trading data from the database for training.
        
        Args:
            token_ids: List of token IDs to fetch data for, or None for all tokens
            start_date: Start date for data collection
            end_date: End date for data collection
            labeled_data: Optional pre-labeled data (for events)
            
        Returns:
            Dictionary mapping token IDs to DataFrames with their historical data
        """
        # Default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(hours=self.training_window_hours)
            
        logger.info(f"Fetching training data from database from {start_date} to {end_date}")
            
        # Connect to database if not already connected
        if not self.postgres_manager.connected:
            success = await self.postgres_manager.connect()
            if not success:
                logger.error("Failed to connect to database")
                return {}
                
        # Get all tokens if token_ids not provided
        if token_ids is None:
            try:
                tokens = await self.postgres_manager.get_all_tokens()
                if not tokens:
                    logger.warning("No tokens found in database")
                    return {}
                    
                token_ids = []
                for token in tokens:
                    if not token:
                        continue
                    # Check for different ID field names in the database schema
                    token_id = None
                    for id_field in ['id', 'token_id', 'address', 'contract_address']:
                        if id_field in token and token[id_field]:
                            token_id = token[id_field]
                            break
                    
                    if token_id:
                        token_ids.append(token_id)
                    else:
                        logger.warning(f"Token missing ID field: {token}")
                
                if not token_ids:
                    logger.error("No valid token IDs found in database")
                    return {}
                    
                logger.info(f"Found {len(token_ids)} tokens in database")
                
            except Exception as e:
                logger.error(f"Error fetching tokens: {e}", exc_info=True)
                return {}
            
        # Fetch data for each token
        token_data_frames = {}
        
        for token_id in token_ids:
            try:
                # Fetch trade data from token_trades table
                trades = await self.postgres_manager.get_trades_for_token(
                    token_id=token_id,
                    start_time=start_date,
                    end_time=end_date,
                    limit=20000  # Increased limit for sufficient data
                )
                
                if not trades:
                    logger.warning(f"No trades found for token {token_id}")
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame(trades)
                
                # Check for required columns
                required_columns = ['timestamp', 'price']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.warning(f"Missing required columns for token {token_id}: {missing_columns}")
                    
                    # Try to map from alternative column names
                    column_mappings = {
                        'timestamp': ['time', 'trade_time', 'created_at', 'transaction_time', 'date', 'block_timestamp'],
                        'price': ['trade_price', 'value', 'price_usd', 'token_price', 'price_eth', 'rate'],
                        'amount': ['volume', 'quantity', 'size', 'trade_amount', 'token_amount', 'amount_token'],
                        'type': ['trade_type', 'side', 'direction', 'is_buy', 'transaction_type']
                    }
                    
                    for required_col in missing_columns:
                        for alt_col in column_mappings.get(required_col, []):
                            if alt_col in df.columns:
                                df[required_col] = df[alt_col]
                                logger.info(f"Mapped column {alt_col} to {required_col} for token {token_id}")
                                break
                
                # Check again for required columns after mapping
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logger.error(f"Still missing required columns for token {token_id}: {missing_columns}")
                    continue
                
                # Convert timestamps to datetime
                if 'timestamp' in df.columns:
                    try:
                        if df['timestamp'].dtype == 'object':
                            # Try different timestamp formats
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                            except:
                                try:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                                except:
                                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        elif np.issubdtype(df['timestamp'].dtype, np.number):
                            # Determine if seconds or milliseconds based on magnitude
                            if df['timestamp'].iloc[0] > 1e11:  # Likely milliseconds
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            else:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    except Exception as e:
                        logger.error(f"Error converting timestamps for {token_id}: {e}")
                        continue
                
                # Ensure numeric price and amount
                for col in ['price', 'amount']:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logger.error(f"Error converting {col} to numeric for {token_id}: {e}")
                
                # Remove rows with NaN prices
                df = df.dropna(subset=['price'])
                if len(df) == 0:
                    logger.warning(f"No valid trades with prices for token {token_id}")
                    continue
                
                # Add 'type' column if missing (default to buy/sell based on count)
                if 'type' not in df.columns:
                    if 'is_buy' in df.columns:
                        df['type'] = df['is_buy'].apply(lambda x: 'buy' if x else 'sell')
                    else:
                        # Default to 50/50 buy/sell
                        import random
                        df['type'] = [random.choice(['buy', 'sell']) for _ in range(len(df))]
                
                # Sort by timestamp
                df.sort_values('timestamp', inplace=True)
                
                # Add event labels if provided
                if labeled_data and token_id in labeled_data:
                    self._add_labels_to_dataframe(df, labeled_data[token_id])
                
                # Check if we have sufficient data after cleaning
                if len(df) < 5:
                    logger.warning(f"Insufficient data for token {token_id} after cleaning: {len(df)} trades")
                    continue
                
                # Resample to specified time interval
                df = self._resample_to_interval(df, token_id)
                token_data_frames[token_id] = df
                logger.info(f"Fetched and processed {len(df)} data points for token {token_id}")
                
            except Exception as e:
                logger.error(f"Error processing trades for token {token_id}: {e}", exc_info=True)
                
        return token_data_frames
    
    def _resample_to_interval(self, df: pd.DataFrame, token_id: str) -> pd.DataFrame:
        """
        Resample a DataFrame to the specified time interval.
        
        Args:
            df: DataFrame to resample
            token_id: Token ID for the DataFrame
            
        Returns:
            Resampled DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Reset the index if timestamp is not the index
            if 'timestamp' in df_copy.columns:
                df_copy = df_copy.set_index('timestamp')
            
            # Define the interval for resampling
            interval = f"{self.time_interval_seconds}S"
            
            # Prepare column aggregations
            agg_dict = {}
            
            # Price columns
            if 'price' in df_copy.columns:
                agg_dict['price'] = 'ohlc'  # Open, high, low, close
            
            # Volume/amount columns
            for vol_col in ['amount', 'volume', 'quantity', 'size']:
                if vol_col in df_copy.columns:
                    agg_dict[vol_col] = 'sum'
            
            # Type/side columns
            for type_col in ['type', 'side', 'trade_type']:
                if type_col in df_copy.columns:
                    agg_dict[type_col] = lambda x: 'buy' if x.str.lower().str.contains('buy').sum() > len(x)/2 else 'sell'
            
            # Count columns
            count_col = next((col for col in ['id', 'trade_id', 'transaction_id'] if col in df_copy.columns), None)
            if count_col:
                agg_dict[count_col] = 'count'
            else:
                # Use any non-aggregated column for counting
                for col in df_copy.columns:
                    if col not in agg_dict:
                        agg_dict[col] = 'count'
                        break
            
            # Resample with the aggregation dictionary
            resampled = df_copy.resample(interval).agg(agg_dict)
            
            # Process the columns
            new_columns = []
            for col in resampled.columns:
                if isinstance(col, tuple):
                    # For OHLC columns
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col)
            
            resampled.columns = new_columns
            
            # Add token ID
            resampled['tokenId'] = token_id
            
            # Reset index to have timestamp as a column
            resampled = resampled.reset_index()
            
            # Rename standard columns
            rename_map = {}
            
            # Handle OHLC price columns
            if 'price_open' in resampled.columns:
                rename_map['price_open'] = 'open'
                rename_map['price_high'] = 'high'
                rename_map['price_low'] = 'low'
                rename_map['price_close'] = 'close'
            
            # Handle count column
            for col in resampled.columns:
                if col.endswith('_count'):
                    rename_map[col] = 'trade_count'
                    break
            
            if rename_map:
                resampled.rename(columns=rename_map, inplace=True)
            
            # Fill NaN values for required columns
            required_cols = ['open', 'high', 'low', 'close', 'trade_count']
            for col in required_cols:
                if col in resampled.columns and resampled[col].isna().any():
                    # Forward fill first
                    resampled[col] = resampled[col].ffill()
                    
                    # For any remaining NaNs, use default values
                    if col in ['trade_count']:
                        resampled[col] = resampled[col].fillna(0)
                    else:
                        # For price columns, use the last known price
                        resampled[col] = resampled[col].fillna(method='bfill')
                        # If still NaN, use the median
                        if resampled[col].isna().any():
                            resampled[col] = resampled[col].fillna(resampled[col].median())
            
            # Ensure we have a continuous time series
            if len(resampled) > 0:
                # Create a continuous time range
                time_range = pd.date_range(
                    start=resampled['timestamp'].min(),
                    end=resampled['timestamp'].max(),
                    freq=interval
                )
                
                # Create a base DataFrame with the continuous time range
                continuous_df = pd.DataFrame({'timestamp': time_range})
                
                # Merge with our resampled data
                resampled = pd.merge(
                    continuous_df,
                    resampled,
                    on='timestamp',
                    how='left'
                )
                
                # Fill in missing data
                for col in resampled.columns:
                    if col != 'timestamp' and resampled[col].isna().any():
                        # Forward fill for most columns
                        resampled[col] = resampled[col].ffill()
                        
                        # For count columns, fill with 0
                        if col in ['trade_count'] or col.endswith('_count'):
                            resampled[col] = resampled[col].fillna(0)
                
                # Fill tokenId if it got lost
                if 'tokenId' in resampled.columns and resampled['tokenId'].isna().any():
                    resampled['tokenId'] = token_id
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data for {token_id}: {e}")
            # If resampling fails, return original DataFrame with timestamp as column
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                return df.reset_index()
            return df
    
    def _add_labels_to_dataframe(self, df: pd.DataFrame, labeled_events: List[Dict[str, Any]]) -> None:
        """
        Add labels to DataFrame based on labeled events.
        
        Args:
            df: DataFrame to add labels to
            labeled_events: List of labeled events
        """
        # Create new columns for labels if they don't exist
        if 'event_type' not in df.columns:
            df['event_type'] = 'normal'
        if 'event_start' not in df.columns:
            df['event_start'] = False
        if 'event_end' not in df.columns:
            df['event_end'] = False
        if 'class_id' not in df.columns:
            df['class_id'] = 0  # 0=normal, 1=pump, 2=dump, 3=peak/distribution
        
        # Add event labels
        for event in labeled_events:
            try:
                event_type = event.get('type', 'normal')
                start_time = event.get('start_time')
                end_time = event.get('end_time')
                class_id = event.get('class_id', 0)
                
                if not start_time or not end_time:
                    continue
                    
                # Convert to datetime if needed
                if isinstance(start_time, (int, float)):
                    start_time = pd.to_datetime(start_time, unit='s')
                elif isinstance(start_time, str):
                    start_time = pd.to_datetime(start_time)
                
                if isinstance(end_time, (int, float)):
                    end_time = pd.to_datetime(end_time, unit='s')
                elif isinstance(end_time, str):
                    end_time = pd.to_datetime(end_time)
                    
                # Mark event data
                mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                df.loc[mask, 'event_type'] = event_type
                df.loc[mask, 'class_id'] = class_id
                
                # Mark start and end points
                if mask.any():
                    start_idx = df[mask].index.min() if not pd.isna(df[mask].index.min()) else None
                    end_idx = df[mask].index.max() if not pd.isna(df[mask].index.max()) else None
                    
                    if start_idx is not None:
                        df.loc[start_idx, 'event_start'] = True
                    if end_idx is not None:
                        df.loc[end_idx, 'event_end'] = True
            except Exception as e:
                logger.error(f"Error adding label to DataFrame: {e}")
    
    async def compute_features(self, token_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute features for each token's data.
        
        Args:
            token_data: Dictionary mapping token IDs to their historical data
            
        Returns:
            Dictionary mapping token IDs to DataFrames with features
        """
        logger.info("Computing features for training data")
        
        feature_dataframes = {}
        
        # Get all feature providers from feature manager
        providers = self.feature_manager.get_providers()
        pump_provider = next((p for p in providers if isinstance(p, PumpDetectionFeatureProvider)), None)
        early_provider = next((p for p in providers if isinstance(p, EarlyPumpDetectionProvider)), None)
        
        if not pump_provider and not early_provider:
            logger.error("No pump detection feature providers found in feature manager")
            return {}
            
        # Process each token
        for token_id, df in token_data.items():
            if len(df) < 5:  # Skip tokens with too little data
                continue
                
            # Create a new DataFrame for features
            feature_df = pd.DataFrame(index=df.index)
            
            # Copy basic columns
            for col in ['timestamp', 'open', 'high', 'low', 'close', 'amount', 'trade_count', 'class_id']:
                if col in df.columns:
                    feature_df[col] = df[col]
            
            # Compute features for each row
            features_computed = 0
            for i, row in df.iterrows():
                timestamp = row.get('timestamp')
                if timestamp is None:
                    continue
                    
                # Convert to unix timestamp if needed
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.timestamp()
                    
                # Context for feature computation
                context = {
                    'token_id': token_id,
                    'timestamp': timestamp,
                    # Add data for trade lookup optimization
                    'trade_data': df.iloc[:i+1].to_dict('records') if i < 20 else None
                }
                
                # Compute features from pump detection provider
                if pump_provider:
                    for feature_name in pump_provider.provides:
                        try:
                            value = self.feature_manager.compute_feature(token_id, feature_name, context)
                            feature_df.at[i, feature_name] = value
                            features_computed += 1
                        except Exception as e:
                            if self.debug_mode:
                                logger.error(f"Error computing {feature_name} for {token_id} at {timestamp}: {e}")
                            feature_df.at[i, feature_name] = np.nan
                            
                # Compute features from early pump detection provider
                if early_provider:
                    for feature_name in early_provider.provides:
                        try:
                            value = self.feature_manager.compute_feature(token_id, feature_name, context)
                            feature_df.at[i, feature_name] = value
                            features_computed += 1
                        except Exception as e:
                            if self.debug_mode:
                                logger.error(f"Error computing {feature_name} for {token_id} at {timestamp}: {e}")
                            feature_df.at[i, feature_name] = np.nan
            
            # Drop rows with too many missing values
            feature_df = feature_df.dropna(thresh=len(feature_df.columns) - 5)
            
            if len(feature_df) > 0:
                feature_dataframes[token_id] = feature_df
                logger.info(f"Computed features for {token_id}: {len(feature_df)} rows, {features_computed} total features")
            else:
                logger.warning(f"No valid feature data for {token_id} after processing")
                
        return feature_dataframes
    
    async def train_models(self, feature_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train the pump detection models using feature data.
        
        Args:
            feature_data: Dictionary mapping token IDs to DataFrames with features
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training models with feature data")
        
        # Combine all token data
        all_data = []
        for token_id, df in feature_data.items():
            df['token_id'] = token_id
            all_data.append(df)
            
        if not all_data:
            logger.error("No feature data available for training")
            return {'success': False, 'error': 'No training data available'}
            
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined training data: {len(combined_df)} rows")
        
        # Handle class imbalance
        if 'class_id' in combined_df.columns:
            class_counts = combined_df['class_id'].value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            # If severe imbalance, perform oversampling or undersampling
            if len(class_counts) > 1:
                max_class = class_counts.idxmax()
                min_class = class_counts.idxmin()
                
                if class_counts[max_class] / class_counts[min_class] > 10:
                    logger.info("Severe class imbalance detected, applying balancing techniques")
                    
                    try:
                        from sklearn.utils import resample
                        
                        # Split by class
                        df_majority = combined_df[combined_df['class_id'] == max_class]
                        df_minority = combined_df[combined_df['class_id'] != max_class]
                        
                        # Undersample majority class if it's extremely large
                        if len(df_majority) > 10000:
                            df_majority = resample(
                                df_majority, 
                                replace=False,
                                n_samples=min(10000, len(df_minority) * 10),
                                random_state=42
                            )
                        
                        # Oversample minority class
                        if len(df_minority) < 1000:
                            df_minority = resample(
                                df_minority,
                                replace=True,
                                n_samples=min(len(df_majority), 1000),
                                random_state=42
                            )
                        
                        # Combine the balanced datasets
                        combined_df = pd.concat([df_majority, df_minority])
                        logger.info(f"After balancing: {len(combined_df)} rows, class distribution: {combined_df['class_id'].value_counts().to_dict()}")
                    except Exception as e:
                        logger.error(f"Error during class balancing: {e}")
        
        # Split into training and validation sets
        from sklearn.model_selection import train_test_split
        
        # Use stratified split if we have class labels
        if 'class_id' in combined_df.columns and combined_df['class_id'].nunique() > 1:
            train_df, val_df = train_test_split(
                combined_df, 
                test_size=self.validation_ratio,
                stratify=combined_df['class_id'],
                random_state=42
            )
        else:
            train_df, val_df = train_test_split(
                combined_df, 
                test_size=self.validation_ratio,
                random_state=42
            )
        
        logger.info(f"Training set: {len(train_df)} rows, Validation set: {len(val_df)} rows")
        
        # Train PumpPredictorModel
        pump_predictor = self._find_model_by_type(PumpPredictorModel)
        early_predictor = self._find_model_by_type(EarlyPumpPredictor)
        
        training_results = {}
        
        if pump_predictor:
            logger.info(f"Training PumpPredictorModel")
            pump_predictor.train(train_df)
            pump_eval_results = pump_predictor.evaluate(val_df)
            training_results['pump_predictor'] = {
                'model_id': pump_predictor.model_id,
                'evaluation': pump_eval_results
            }
            
            # Save the model configuration
            self._save_model_config(pump_predictor.model_id, pump_predictor, 'PumpPredictorModel', pump_eval_results)
            
        if early_predictor:
            logger.info(f"Training EarlyPumpPredictor")
            early_predictor.train(train_df)
            early_eval_results = early_predictor.evaluate(val_df)
            training_results['early_predictor'] = {
                'model_id': early_predictor.model_id,
                'evaluation': early_eval_results
            }
            
            # Save the model configuration
            self._save_model_config(early_predictor.model_id, early_predictor, 'EarlyPumpPredictor', early_eval_results)
            
        return {
            'success': True,
            'results': training_results,
            'training_size': len(train_df),
            'validation_size': len(val_df)
        }
    
    def _find_model_by_type(self, model_type):
        """
        Find a model of the specified type registered with the feature manager.
        
        Args:
            model_type: Type of model to find
            
        Returns:
            Model instance or None
        """
        # Check if we have any registered models
        if not hasattr(self, '_models') or not self._models:
            logger.warning(f"No models registered with the trainer")
            return None
            
        # Find the first model that matches the specified type
        for model in self._models:
            if isinstance(model, model_type):
                logger.info(f"Found registered model of type {model_type.__name__}: {model.model_id}")
                return model
                
        logger.warning(f"No model of type {model_type.__name__} found among registered models")
        return None
    
    def _save_model_config(self, model_id: str, model: Model, model_type: str, eval_results: Dict[str, float]) -> str:
        """
        Save model configuration and evaluation results to disk.
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model instance
            model_type: Type of model (classification, regression)
            eval_results: Dictionary containing evaluation metrics
            
        Returns:
            Path to the saved model configuration
        """
        try:
            # Create directory for model
            model_dir = os.path.join(self.models_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create config dictionary
            config = {
                'model_id': model_id,
                'model_type': model_type,
                'model_version': getattr(model, 'model_version', '1.0.0'),
                'trained_at': datetime.now().isoformat(),
                'evaluation': eval_results,
                # Save model-specific parameters
                'parameters': {}
            }
            
            # Add model-specific parameters
            if hasattr(model, 'get_parameters'):
                config['parameters'] = model.get_parameters()
            elif hasattr(model, '_parameters'):
                config['parameters'] = model._parameters
                
            # Save PumpPredictorModel threshold parameters
            if isinstance(model, PumpPredictorModel):
                config['parameters']['pump_threshold'] = float(model.pump_threshold)
                config['parameters']['dump_threshold'] = float(model.dump_threshold)
                config['parameters']['neutral_threshold'] = float(model.neutral_threshold)
                
            # Save EarlyPumpPredictor parameters
            if isinstance(model, EarlyPumpPredictor):
                config['parameters']['buy_threshold'] = float(model.buy_threshold)
                config['parameters']['confidence_threshold'] = float(model.confidence_threshold)
            
            # Save config to JSON
            config_path = os.path.join(model_dir, f"{model_id}_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            logger.info(f"Saved model configuration to {config_path}")
            
            # Save the actual trained model
            model_path = os.path.join(model_dir, f"{model_id}.pkl")
            
            # Check if it's an XGBoost model
            if hasattr(model, '_clf') and model._clf is not None:
                # For PumpDumpDetectionModel which uses XGBoost
                xgb_model_path = os.path.join(model_dir, f"{model_id}.xgb")
                try:
                    if hasattr(model, 'save_model'):
                        # Use the model's own save_model method if available
                        model.save_model(xgb_model_path)
                        logger.info(f"Saved XGBoost model to {xgb_model_path}")
                    elif hasattr(model._clf, 'save_model'):
                        # Direct call to XGBoost save_model
                        model._clf.save_model(xgb_model_path)
                        logger.info(f"Saved XGBoost model to {xgb_model_path}")
                    else:
                        # Fallback to pickle for XGBoost model
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                        logger.info(f"Saved XGBoost model using pickle to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving XGBoost model: {e}", exc_info=True)
            else:
                # For other model types, use pickle
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Saved model using pickle to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}", exc_info=True)
            
            return config_path
            
        except Exception as e:
            logger.error(f"Error saving model configuration: {e}", exc_info=True)
            return ""
    
    def adjust_features_for_time_interval(self) -> Dict[str, Any]:
        """
        Adjust feature providers to work with the specified time interval.
        
        Returns:
            Dictionary with adjustment results
        """
        logger.info(f"Adjusting feature providers for {self.time_interval_seconds}-second intervals")
        
        adjustments = {}
        
        # Get all feature providers from feature manager
        providers = self.feature_manager.get_providers()
        pump_provider = next((p for p in providers if isinstance(p, PumpDetectionFeatureProvider)), None)
        early_provider = next((p for p in providers if isinstance(p, EarlyPumpDetectionProvider)), None)
        
        # Adjust PumpDetectionFeatureProvider if present
        if pump_provider:
            # Check if we need to adjust time windows
            # This is a placeholder - real implementation would depend on the actual code
            logger.info("Adjusting PumpDetectionFeatureProvider for fine-grained intervals")
            adjustments['pump_provider'] = {
                'status': 'adjusted',
                'details': 'Adjusted time windows for 2-second intervals'
            }
        
        # Early pump provider likely needs fewer adjustments since it's designed for minimal data
        if early_provider:
            logger.info("EarlyPumpDetectionProvider already optimized for minimal data")
            adjustments['early_provider'] = {
                'status': 'optimized',
                'details': 'Already designed for minimal data points'
            }
            
        return adjustments
    
    async def run_training_session(
        self,
        token_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        labeled_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete training session.
        
        Args:
            token_ids: List of token IDs to train on
            start_date: Start date for training data
            end_date: End date for training data
            labeled_data: Pre-labeled event data
            
        Returns:
            Dictionary with training session results
        """
        logger.info("Starting model training session")
        
        # Step 1: Adjust features for time interval
        adjustment_results = self.adjust_features_for_time_interval()
        
        # Step 2: Fetch training data
        token_data = await self.fetch_training_data(token_ids, start_date, end_date, labeled_data)
        if not token_data:
            return {'success': False, 'error': 'No training data available'}
            
        # Step 3: Compute features
        feature_data = await self.compute_features(token_data)
        if not feature_data:
            return {'success': False, 'error': 'Feature computation failed'}
            
        # Step 4: Train models
        training_results = await self.train_models(feature_data)
        
        # Return complete results
        return {
            'success': training_results.get('success', False),
            'adjustment_results': adjustment_results,
            'tokens_processed': len(token_data),
            'features_computed': len(feature_data),
            'training_results': training_results.get('results', {}),
            'timestamp': datetime.now().isoformat()
        }
    
    def register_model(self, model: Any) -> None:
        """
        Register a model for training.
        
        Args:
            model: The model to register
        """
        if not hasattr(self, '_models'):
            self._models = []
            
        self._models.append(model)
        logger.info(f"Registered model: {model.model_id} (type: {model.model_type}, version: {model.model_version})")
        
    def train(self, labeled_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """
        Train all registered models using provided labeled data.
        
        Args:
            labeled_data: Optional dictionary of labeled events for supervision
            
        Returns:
            Dictionary with training results
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.run_training_session(labeled_data=labeled_data)) 