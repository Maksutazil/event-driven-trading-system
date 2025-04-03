#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volume Feature Provider Module

This module provides the VolumeFeatureProvider class that computes volume-related
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


class VolumeFeatureProvider(BaseFeatureProvider):
    """
    Feature provider for volume-related features.
    
    This provider computes features related to token trading volumes, including
    current volume, volume changes, volume patterns, and liquidity metrics.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the volume feature provider.
        
        Args:
            data_feed: Data feed to use for retrieving historical volume data
        """
        # Define all volume-related features
        feature_names = [
            'volume_24h',                # Trading volume in last 24 hours
            'volume_1h',                 # Trading volume in last 1 hour
            'volume_15m',                # Trading volume in last 15 minutes
            'volume_5m',                 # Trading volume in last 5 minutes
            'volume_1m',                 # Trading volume in last 1 minute
            'volume_change_1h',          # Volume change in last 1 hour
            'volume_change_24h',         # Volume change in last 24 hours
            'avg_trade_size',            # Average trade size
            'volume_buy_ratio',          # Ratio of buy volume to total volume
            'volume_sell_ratio',         # Ratio of sell volume to total volume
            'volume_volatility',         # Volatility of volume
            'volume_acceleration',       # Volume change acceleration
            'volume_price_ratio',        # Volume to price ratio
            'volume_per_second',         # Volume per second
            'trade_frequency',           # Number of trades per minute
            'liquidity_estimate',        # Estimate of token liquidity
        ]
        
        # Define dependencies
        dependencies = {
            'volume_change_1h': ['volume_1h'],
            'volume_change_24h': ['volume_24h'],
            'volume_acceleration': ['volume_change_1h'],
            'volume_price_ratio': ['volume_24h', 'price_current'],
        }
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
    
    def _get_trade_data(self, token_id: str, hours: int = 24) -> pd.DataFrame:
        """
        Get historical trade data for the specified token.
        
        Args:
            token_id: The ID of the token
            hours: Number of hours of historical data to retrieve
            
        Returns:
            DataFrame with historical trade data
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
            
            # Ensure required columns exist
            if 'timestamp' not in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = df['time']
                else:
                    logger.warning(f"No timestamp column found in trade data for {token_id}")
                    return pd.DataFrame()
            
            # Ensure volume column exists
            if 'volume' not in df.columns:
                if 'size' in df.columns:
                    df['volume'] = df['size']
                elif 'amount' in df.columns:
                    df['volume'] = df['amount']
                else:
                    logger.warning(f"No volume column found in trade data for {token_id}")
                    return pd.DataFrame()
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trade data for {token_id}: {e}")
            return pd.DataFrame()
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified volume feature for the given token.
        
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
                logger.warning(f"Feature '{feature_name}' is not provided by VolumeFeatureProvider")
                return None
            
            # Get trade data
            trade_data = None
            trades_data = data.get('trade_data')
            
            # For volume_1m and volume_5m, we can sometimes use the provided trade data
            if feature_name in ['volume_1m', 'volume_5m'] and trades_data:
                if isinstance(trades_data, list) and trades_data:
                    now = datetime.now()
                    if feature_name == 'volume_1m':
                        cutoff_time = now - timedelta(minutes=1)
                    else:  # volume_5m
                        cutoff_time = now - timedelta(minutes=5)
                    
                    # Filter recent trades
                    recent_trades = [
                        t for t in trades_data 
                        if 'timestamp' in t and t['timestamp'] >= cutoff_time.timestamp()
                    ]
                    
                    if recent_trades:
                        # Calculate volume
                        volume = sum(t.get('volume', 0) for t in recent_trades)
                        return volume
            
            # For other features or if we couldn't use provided trade data, we need historical data
            if not trade_data:
                hours_required = 24  # Default to 24 hours of data
                if feature_name in ['volume_1m', 'volume_5m', 'volume_15m']:
                    hours_required = 1
                elif feature_name in ['volume_1h', 'volume_change_1h']:
                    hours_required = 2  # Get a bit more than needed
                
                trade_data = self._get_trade_data(token_id, hours=hours_required)
                
                if trade_data.empty:
                    logger.warning(f"No trade data available for token {token_id}")
                    return None
            
            # Compute the requested feature
            if feature_name.startswith('volume_') and not feature_name.startswith('volume_change_'):
                # Extract timeframe from feature name
                parts = feature_name.split('_')
                if len(parts) >= 2:
                    timeframe = parts[1]
                    seconds = self._timeframe_to_seconds(timeframe)
                    
                    # Filter data for the timeframe
                    now = datetime.now()
                    cutoff_time = now - timedelta(seconds=seconds)
                    
                    # Filter by timestamp
                    if 'timestamp' in trade_data.columns:
                        # Handle numeric timestamps (Unix timestamps)
                        if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                            cutoff_timestamp = cutoff_time.timestamp()
                            filtered_data = trade_data[trade_data['timestamp'] >= cutoff_timestamp]
                        else:
                            filtered_data = trade_data[trade_data['timestamp'] >= cutoff_time]
                        
                        # Calculate total volume
                        return filtered_data['volume'].sum() if not filtered_data.empty else 0
                    
                    # Fallback: use all available data
                    return trade_data['volume'].sum() if not trade_data.empty else 0
            
            elif feature_name == 'volume_change_1h':
                # Calculate volume change in the last hour vs previous hour
                now = datetime.now()
                hour_ago = now - timedelta(hours=1)
                two_hours_ago = now - timedelta(hours=2)
                
                if 'timestamp' in trade_data.columns:
                    # Filter data for current hour and previous hour
                    if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                        hour_ago_ts = hour_ago.timestamp()
                        two_hours_ago_ts = two_hours_ago.timestamp()
                        
                        current_hour = trade_data[trade_data['timestamp'] >= hour_ago_ts]
                        previous_hour = trade_data[
                            (trade_data['timestamp'] >= two_hours_ago_ts) & 
                            (trade_data['timestamp'] < hour_ago_ts)
                        ]
                    else:
                        current_hour = trade_data[trade_data['timestamp'] >= hour_ago]
                        previous_hour = trade_data[
                            (trade_data['timestamp'] >= two_hours_ago) & 
                            (trade_data['timestamp'] < hour_ago)
                        ]
                    
                    current_volume = current_hour['volume'].sum() if not current_hour.empty else 0
                    previous_volume = previous_hour['volume'].sum() if not previous_hour.empty else 0
                    
                    if previous_volume > 0:
                        return (current_volume / previous_volume) - 1
                    elif current_volume > 0:
                        return 1.0  # If previous was 0 but current is positive, that's a 100% increase
                    else:
                        return 0.0  # No change if both are 0
                
                return 0.0
            
            elif feature_name == 'volume_change_24h':
                # Calculate volume change in the last 24 hours vs previous 24 hours
                now = datetime.now()
                day_ago = now - timedelta(days=1)
                two_days_ago = now - timedelta(days=2)
                
                if 'timestamp' in trade_data.columns:
                    # Filter data for current day and previous day
                    if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                        day_ago_ts = day_ago.timestamp()
                        two_days_ago_ts = two_days_ago.timestamp()
                        
                        current_day = trade_data[trade_data['timestamp'] >= day_ago_ts]
                        previous_day = trade_data[
                            (trade_data['timestamp'] >= two_days_ago_ts) & 
                            (trade_data['timestamp'] < day_ago_ts)
                        ]
                    else:
                        current_day = trade_data[trade_data['timestamp'] >= day_ago]
                        previous_day = trade_data[
                            (trade_data['timestamp'] >= two_days_ago) & 
                            (trade_data['timestamp'] < day_ago)
                        ]
                    
                    current_volume = current_day['volume'].sum() if not current_day.empty else 0
                    previous_volume = previous_day['volume'].sum() if not previous_day.empty else 0
                    
                    if previous_volume > 0:
                        return (current_volume / previous_volume) - 1
                    elif current_volume > 0:
                        return 1.0  # If previous was 0 but current is positive, that's a 100% increase
                    else:
                        return 0.0  # No change if both are 0
                
                return 0.0
            
            elif feature_name == 'avg_trade_size':
                # Calculate average trade size from available data
                if not trade_data.empty and 'volume' in trade_data.columns:
                    return trade_data['volume'].mean()
                return 0.0
            
            elif feature_name == 'volume_buy_ratio':
                # Calculate ratio of buy volume to total volume
                if not trade_data.empty and 'volume' in trade_data.columns:
                    if 'trade_type' in trade_data.columns:
                        # Filter buy trades
                        buy_trades = trade_data[trade_data['trade_type'] == 'buy']
                        buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
                        total_volume = trade_data['volume'].sum()
                        
                        return buy_volume / total_volume if total_volume > 0 else 0.5
                    elif 'is_buy' in trade_data.columns:
                        # Alternative column name
                        buy_trades = trade_data[trade_data['is_buy'] == True]
                        buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
                        total_volume = trade_data['volume'].sum()
                        
                        return buy_volume / total_volume if total_volume > 0 else 0.5
                
                return 0.5  # Default to neutral if we can't determine
            
            elif feature_name == 'volume_sell_ratio':
                # Calculate ratio of sell volume to total volume
                if not trade_data.empty and 'volume' in trade_data.columns:
                    if 'trade_type' in trade_data.columns:
                        # Filter sell trades
                        sell_trades = trade_data[trade_data['trade_type'] == 'sell']
                        sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
                        total_volume = trade_data['volume'].sum()
                        
                        return sell_volume / total_volume if total_volume > 0 else 0.5
                    elif 'is_sell' in trade_data.columns:
                        # Alternative column name
                        sell_trades = trade_data[trade_data['is_sell'] == True]
                        sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
                        total_volume = trade_data['volume'].sum()
                        
                        return sell_volume / total_volume if total_volume > 0 else 0.5
                
                return 0.5  # Default to neutral if we can't determine
            
            elif feature_name == 'volume_volatility':
                # Calculate volatility of volume
                if not trade_data.empty and 'volume' in trade_data.columns and 'timestamp' in trade_data.columns:
                    # Group trades by minute to get volume per minute
                    if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                        # Convert unix timestamp to datetime
                        trade_data['datetime'] = pd.to_datetime(trade_data['timestamp'], unit='s')
                    else:
                        trade_data['datetime'] = trade_data['timestamp']
                    
                    # Group by minute
                    trade_data['minute'] = trade_data['datetime'].dt.floor('min')
                    volume_per_minute = trade_data.groupby('minute')['volume'].sum()
                    
                    if len(volume_per_minute) >= 5:
                        # Calculate coefficient of variation
                        std_dev = volume_per_minute.std()
                        mean_volume = volume_per_minute.mean()
                        
                        if mean_volume > 0:
                            return std_dev / mean_volume
                
                return 0.5  # Default moderate volatility
            
            elif feature_name == 'volume_acceleration':
                # Calculate volume acceleration (change in volume change rate)
                if not trade_data.empty and 'volume' in trade_data.columns and 'timestamp' in trade_data.columns:
                    now = datetime.now()
                    hour_ago = now - timedelta(hours=1)
                    two_hours_ago = now - timedelta(hours=2)
                    three_hours_ago = now - timedelta(hours=3)
                    
                    if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                        hour_ago_ts = hour_ago.timestamp()
                        two_hours_ago_ts = two_hours_ago.timestamp()
                        three_hours_ago_ts = three_hours_ago.timestamp()
                        
                        current_hour = trade_data[trade_data['timestamp'] >= hour_ago_ts]
                        previous_hour = trade_data[
                            (trade_data['timestamp'] >= two_hours_ago_ts) & 
                            (trade_data['timestamp'] < hour_ago_ts)
                        ]
                        two_hours_prior = trade_data[
                            (trade_data['timestamp'] >= three_hours_ago_ts) & 
                            (trade_data['timestamp'] < two_hours_ago_ts)
                        ]
                    else:
                        current_hour = trade_data[trade_data['timestamp'] >= hour_ago]
                        previous_hour = trade_data[
                            (trade_data['timestamp'] >= two_hours_ago) & 
                            (trade_data['timestamp'] < hour_ago)
                        ]
                        two_hours_prior = trade_data[
                            (trade_data['timestamp'] >= three_hours_ago) & 
                            (trade_data['timestamp'] < two_hours_ago)
                        ]
                    
                    current_volume = current_hour['volume'].sum() if not current_hour.empty else 0
                    previous_volume = previous_hour['volume'].sum() if not previous_hour.empty else 0
                    prior_volume = two_hours_prior['volume'].sum() if not two_hours_prior.empty else 0
                    
                    # Calculate change rates
                    change_rate_current = ((current_volume / max(1e-10, previous_volume)) - 1) if previous_volume > 0 else 0
                    change_rate_previous = ((previous_volume / max(1e-10, prior_volume)) - 1) if prior_volume > 0 else 0
                    
                    # Calculate acceleration (change in change rate)
                    return change_rate_current - change_rate_previous
                
                return 0.0
            
            elif feature_name == 'volume_price_ratio':
                # Calculate volume to price ratio (normalized by dividing by 24h average)
                current_volume = data.get('volume_24h')
                current_price = data.get('price_current')
                
                if current_volume is not None and current_price is not None and current_price > 0:
                    return current_volume / current_price
                
                return 0.0
            
            elif feature_name == 'volume_per_second':
                # Calculate average volume per second over the past hour
                if not trade_data.empty and 'volume' in trade_data.columns:
                    now = datetime.now()
                    hour_ago = now - timedelta(hours=1)
                    
                    if 'timestamp' in trade_data.columns:
                        if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                            hour_ago_ts = hour_ago.timestamp()
                            recent_data = trade_data[trade_data['timestamp'] >= hour_ago_ts]
                        else:
                            recent_data = trade_data[trade_data['timestamp'] >= hour_ago]
                        
                        if not recent_data.empty:
                            total_volume = recent_data['volume'].sum()
                            time_span = 3600  # 1 hour in seconds
                            return total_volume / time_span
                
                return 0.0
            
            elif feature_name == 'trade_frequency':
                # Calculate number of trades per minute over the past hour
                if not trade_data.empty:
                    now = datetime.now()
                    hour_ago = now - timedelta(hours=1)
                    
                    if 'timestamp' in trade_data.columns:
                        if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
                            hour_ago_ts = hour_ago.timestamp()
                            recent_data = trade_data[trade_data['timestamp'] >= hour_ago_ts]
                        else:
                            recent_data = trade_data[trade_data['timestamp'] >= hour_ago]
                        
                        if not recent_data.empty:
                            num_trades = len(recent_data)
                            minutes = 60  # 1 hour in minutes
                            return num_trades / minutes
                
                return 0.0
            
            elif feature_name == 'liquidity_estimate':
                # Estimate liquidity based on volume and trade frequency
                volume_24h = data.get('volume_24h')
                trade_frequency = data.get('trade_frequency')
                
                if volume_24h is not None and trade_frequency is not None:
                    # Simple liquidity score combining volume and frequency
                    # Normalized to 0-1 range
                    normalized_volume = min(1.0, volume_24h / 1000)  # Assume 1000 is high volume
                    normalized_frequency = min(1.0, trade_frequency / 10)  # Assume 10 trades/min is high
                    
                    return (normalized_volume * 0.7) + (normalized_frequency * 0.3)
                
                return 0.5  # Default to medium liquidity
            
            else:
                logger.warning(f"Unknown volume feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing volume feature '{feature_name}' for token {token_id}: {e}")
            return None
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert a timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g. '1m', '5m', '1h', '24h')
            
        Returns:
            Number of seconds
        """
        try:
            if timeframe == '24h':
                return 86400  # 24 hours in seconds
            elif timeframe == '1h':
                return 3600  # 1 hour in seconds
            elif timeframe == '15m':
                return 900  # 15 minutes in seconds
            elif timeframe == '5m':
                return 300  # 5 minutes in seconds
            elif timeframe == '1m':
                return 60  # 1 minute in seconds
            else:
                # Try to parse the timeframe
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
                    return 3600  # Default to 1 hour
        except Exception as e:
            logger.error(f"Error parsing timeframe '{timeframe}': {e}")
            return 3600  # Default to 1 hour 