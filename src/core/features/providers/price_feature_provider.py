#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Price Feature Provider Module

This module provides the PriceFeatureProvider class that computes price-related
features for tokens.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.core.data import DataFeedInterface
from src.core.features.providers.base_feature_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class PriceFeatureProvider(BaseFeatureProvider):
    """
    Feature provider for price-related features.
    
    This provider computes features related to token prices, including
    current price, price changes, volatility, and momentum.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the price feature provider.
        
        Args:
            data_feed: Data feed to use for retrieving historical price data
        """
        # Define all price-related features
        feature_names = [
            'price_current',                # Current price
            'price_change_1m',              # Price change in last 1 minute
            'price_change_5m',              # Price change in last 5 minutes
            'price_change_15m',             # Price change in last 15 minutes
            'price_change_1h',              # Price change in last 1 hour
            'price_change_24h',             # Price change in last 24 hours
            'price_volatility_1h',          # Price volatility in last 1 hour
            'price_volatility_24h',         # Price volatility in last 24 hours
            'price_momentum_short',         # Short-term price momentum (5-15 min)
            'price_momentum_medium',        # Medium-term price momentum (15-60 min)
            'price_velocity',               # Rate of price change
            'price_acceleration',           # Change in rate of price change
            'price_mean_1h',                # Mean price over last 1 hour
            'price_median_1h',              # Median price over last 1 hour
            'highest_price_24h',            # Highest price in last 24 hours
            'lowest_price_24h',             # Lowest price in last 24 hours
        ]
        
        # Define dependencies (if any features depend on others)
        dependencies = {
            'price_velocity': ['price_current', 'price_change_5m'],
            'price_acceleration': ['price_velocity'],
            'price_momentum_short': ['price_change_5m'],
            'price_momentum_medium': ['price_change_15m', 'price_change_1h'],
        }
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
        
    def _get_price_data(self, token_id: str, hours: int = 24) -> pd.DataFrame:
        """
        Get historical price data for the specified token.
        
        Args:
            token_id: The ID of the token
            hours: Number of hours of historical data to retrieve
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get historical data from data feed
            historical_data = self.data_feed.get_historical_data(
                token_id=token_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Convert to DataFrame if necessary
            if not isinstance(historical_data, pd.DataFrame):
                df = pd.DataFrame(historical_data)
            else:
                df = historical_data
            
            # Ensure timestamp and price columns exist
            if 'timestamp' not in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = df['time']
                else:
                    logger.warning(f"No timestamp column found in price data for {token_id}")
                    return pd.DataFrame()
            
            if 'price' not in df.columns:
                if 'last_price' in df.columns:
                    df['price'] = df['last_price']
                else:
                    logger.warning(f"No price column found in price data for {token_id}")
                    return pd.DataFrame()
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting price data for {token_id}: {e}")
            return pd.DataFrame()
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified price feature for the given token.
        
        Args:
            feature_name: The name of the feature to compute
            token_id: The ID of the token
            data: Dictionary containing required data for computation
            
        Returns:
            The computed feature value
        """
        try:
            # Check if feature is provided by this provider
            if feature_name not in self.feature_names:
                logger.warning(f"Feature '{feature_name}' is not provided by PriceFeatureProvider")
                return None
            
            # Get or extract price data
            price_data = None
            trades_data = data.get('trade_data')
            
            # For current price, we can use the latest trade data if available
            if feature_name == 'price_current' and trades_data:
                if isinstance(trades_data, dict):
                    return trades_data.get('price', 0)
                elif isinstance(trades_data, list) and trades_data:
                    return trades_data[-1].get('price', 0)
            
            # For other features, we need historical price data
            if not price_data:
                hours_required = 24  # Default to 24 hours of data
                if feature_name in ['price_change_1m', 'price_change_5m', 'price_change_15m']:
                    hours_required = 1
                elif feature_name in ['price_change_1h', 'price_volatility_1h', 'price_mean_1h', 'price_median_1h']:
                    hours_required = 2  # Get a bit more than needed
                
                price_data = self._get_price_data(token_id, hours=hours_required)
                
                if price_data.empty:
                    logger.warning(f"No price data available for token {token_id}")
                    return None
            
            # Compute the requested feature
            if feature_name == 'price_current':
                return price_data['price'].iloc[-1] if not price_data.empty else None
                
            elif feature_name.startswith('price_change_'):
                # Extract timeframe from feature name
                timeframe = feature_name.split('_')[-1]
                seconds = self._timeframe_to_seconds(timeframe)
                
                # Filter data for the timeframe
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=seconds)
                
                # If data has timestamps
                if 'timestamp' in price_data.columns:
                    # Filter recent data
                    if isinstance(price_data['timestamp'].iloc[0], (int, float)):
                        # Convert unix timestamp to datetime if needed
                        cutoff_timestamp = cutoff_time.timestamp()
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_timestamp]
                    else:
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_time]
                else:
                    # If no timestamp, use the last N rows
                    # This is a fallback and less accurate
                    n_rows = min(100, len(price_data))  # Limit to avoid memory issues
                    filtered_data = price_data.iloc[-n_rows:]
                
                if len(filtered_data) >= 2:
                    first_price = filtered_data['price'].iloc[0]
                    last_price = filtered_data['price'].iloc[-1]
                    if first_price > 0:
                        return (last_price / first_price) - 1
                
                return 0
                
            elif feature_name.startswith('price_volatility_'):
                # Extract timeframe from feature name
                timeframe = feature_name.split('_')[-1]
                seconds = self._timeframe_to_seconds(timeframe)
                
                # Filter data for the timeframe
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=seconds)
                
                # If data has timestamps
                if 'timestamp' in price_data.columns:
                    # Filter recent data
                    if isinstance(price_data['timestamp'].iloc[0], (int, float)):
                        # Convert unix timestamp to datetime if needed
                        cutoff_timestamp = cutoff_time.timestamp()
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_timestamp]
                    else:
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_time]
                else:
                    # If no timestamp, use the last N rows
                    n_rows = min(100, len(price_data))  # Limit to avoid memory issues
                    filtered_data = price_data.iloc[-n_rows:]
                
                if len(filtered_data) >= 2:
                    # Calculate standard deviation of price
                    std_dev = filtered_data['price'].std()
                    mean_price = filtered_data['price'].mean()
                    if mean_price > 0:
                        return std_dev / mean_price  # Normalized volatility
                
                return 0
                
            elif feature_name == 'price_velocity':
                # Rate of price change (first derivative)
                if 'price_change_5m' in data:
                    change_5m = data['price_change_5m']
                    return change_5m / 300  # Change per second
                
                # Fallback calculation
                if not price_data.empty and len(price_data) >= 2:
                    time_diff = (price_data['timestamp'].iloc[-1] - price_data['timestamp'].iloc[0])
                    if isinstance(time_diff, timedelta):
                        time_diff = time_diff.total_seconds()
                    price_diff = price_data['price'].iloc[-1] - price_data['price'].iloc[0]
                    if time_diff > 0:
                        return price_diff / time_diff
                
                return 0
                
            elif feature_name == 'price_acceleration':
                # Change in rate of price change (second derivative)
                if 'price_velocity' in data and len(price_data) >= 3:
                    # Need at least 3 data points for acceleration
                    # Calculate velocity for first half and second half
                    mid_point = len(price_data) // 2
                    
                    first_half = price_data.iloc[:mid_point]
                    second_half = price_data.iloc[mid_point:]
                    
                    if not first_half.empty and not second_half.empty:
                        time_diff1 = (first_half['timestamp'].iloc[-1] - first_half['timestamp'].iloc[0])
                        if isinstance(time_diff1, timedelta):
                            time_diff1 = time_diff1.total_seconds()
                        
                        time_diff2 = (second_half['timestamp'].iloc[-1] - second_half['timestamp'].iloc[0])
                        if isinstance(time_diff2, timedelta):
                            time_diff2 = time_diff2.total_seconds()
                        
                        price_diff1 = first_half['price'].iloc[-1] - first_half['price'].iloc[0]
                        price_diff2 = second_half['price'].iloc[-1] - second_half['price'].iloc[0]
                        
                        if time_diff1 > 0 and time_diff2 > 0:
                            velocity1 = price_diff1 / time_diff1
                            velocity2 = price_diff2 / time_diff2
                            time_between = (time_diff1 + time_diff2) / 2
                            return (velocity2 - velocity1) / time_between
                
                return 0
                
            elif feature_name.startswith('price_momentum_'):
                # Momentum indicators
                if feature_name == 'price_momentum_short':
                    if 'price_change_5m' in data:
                        return data['price_change_5m'] * 12  # Annualized momentum
                elif feature_name == 'price_momentum_medium':
                    if 'price_change_15m' in data and 'price_change_1h' in data:
                        # Weighted average of different timeframes
                        return data['price_change_15m'] * 0.7 + data['price_change_1h'] * 0.3
                
                return 0
                
            elif feature_name == 'price_mean_1h':
                # Mean price over the last hour
                now = datetime.now()
                cutoff_time = now - timedelta(hours=1)
                
                if 'timestamp' in price_data.columns:
                    if isinstance(price_data['timestamp'].iloc[0], (int, float)):
                        cutoff_timestamp = cutoff_time.timestamp()
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_timestamp]
                    else:
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_time]
                    
                    if not filtered_data.empty:
                        return filtered_data['price'].mean()
                
                return price_data['price'].mean() if not price_data.empty else None
                
            elif feature_name == 'price_median_1h':
                # Median price over the last hour
                now = datetime.now()
                cutoff_time = now - timedelta(hours=1)
                
                if 'timestamp' in price_data.columns:
                    if isinstance(price_data['timestamp'].iloc[0], (int, float)):
                        cutoff_timestamp = cutoff_time.timestamp()
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_timestamp]
                    else:
                        filtered_data = price_data[price_data['timestamp'] >= cutoff_time]
                    
                    if not filtered_data.empty:
                        return filtered_data['price'].median()
                
                return price_data['price'].median() if not price_data.empty else None
                
            elif feature_name == 'highest_price_24h':
                # Highest price in the last 24 hours
                return price_data['price'].max() if not price_data.empty else None
                
            elif feature_name == 'lowest_price_24h':
                # Lowest price in the last 24 hours
                return price_data['price'].min() if not price_data.empty else None
                
            else:
                logger.warning(f"Unknown price feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing price feature '{feature_name}' for token {token_id}: {e}")
            return None
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g. '1m', '5m', '1h', '24h')
            
        Returns:
            Number of seconds
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            logger.warning(f"Unknown timeframe unit: {unit}")
            return 0 