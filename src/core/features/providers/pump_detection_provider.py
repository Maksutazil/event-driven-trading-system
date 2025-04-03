#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump Detection Feature Provider Module

This module provides the PumpDetectionFeatureProvider class that computes features
specifically designed for detecting pump and dump patterns in token trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.core.data import DataFeedInterface
from src.core.features.providers.base_feature_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class PumpDetectionFeatureProvider(BaseFeatureProvider):
    """
    Feature provider for pump and dump detection.
    
    This provider computes features that are useful for detecting pump and dump patterns
    in token trading activity, including volatility metrics, price surge indicators,
    and abnormal volume patterns.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the pump detection feature provider.
        
        Args:
            data_feed: Data feed to use for retrieving historical data
        """
        # Define all pump detection features
        feature_names = [
            # Price velocity and acceleration metrics
            'price_velocity',              # Rate of price change (first derivative)
            'price_acceleration',          # Change in rate of price change (second derivative)
            
            # Volume anomaly metrics
            'volume_surge_ratio',          # Ratio of current volume to historical average
            'volume_volatility',           # Volatility of trading volume
            'buy_sell_volume_ratio',       # Ratio of buy to sell volume
            
            # Order book metrics
            'order_imbalance',             # Imbalance between buy and sell orders
            'std_rush_order',              # Standard deviation of rush orders
            'avg_rush_order',              # Average rush order size
            
            # Price anomaly metrics
            'price_deviation',             # Deviation from moving average
            'price_volatility_short',      # Short-term price volatility
            'price_volatility_ratio',      # Ratio of short-term to long-term volatility
            
            # Pattern detection
            'pump_pattern_score',          # Score indicating presence of pump pattern
            'dump_pattern_score',          # Score indicating presence of dump pattern
            'pump_phase_detection',        # Detected phase of pump (0-4, where 0=none, 4=dump)
            
            # Time-based features
            'minute_since_volume_peak',    # Minutes since volume peaked
            'minute_since_price_peak',     # Minutes since price peaked
            
            # Combined metrics
            'abnormal_activity_score',     # Overall score of abnormal trading activity
        ]
        
        # Define dependencies
        dependencies = {
            'price_acceleration': ['price_velocity'],
            'volume_surge_ratio': ['volume_1m', 'volume_1h'],
            'price_deviation': ['price_current', 'sma_50'],
            'price_volatility_ratio': ['price_volatility_short', 'price_volatility_24h'],
            'pump_pattern_score': ['price_velocity', 'volume_surge_ratio', 'price_deviation'],
            'dump_pattern_score': ['price_velocity', 'volume_surge_ratio', 'price_deviation'],
            'abnormal_activity_score': ['pump_pattern_score', 'volume_volatility', 'price_volatility_ratio'],
        }
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
        self._name = "PumpDetectionProvider"
        
    @property
    def name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name
        """
        return self._name
    
    @property
    def provides(self) -> set:
        """
        Get the set of features provided.
        
        Returns:
            Set of feature names
        """
        return set(self.feature_names)
    
    def get_features(self, context: dict) -> dict:
        """
        Get values for all provided features.
        
        Args:
            context: Dictionary containing context data, such as token_id
            
        Returns:
            Dictionary mapping feature names to values
        """
        token_id = context.get('token_id')
        if not token_id:
            logger.warning("No token_id in context for get_features")
            return {}
        
        # Create a data dictionary for feature computation
        data = {}
        
        # Compute and return all features
        features = {}
        for feature_name in self.feature_names:
            features[feature_name] = self.compute_feature(feature_name, token_id, data)
        
        return features
    
    def _get_trade_data(self, token_id: str, hours: int = 24) -> pd.DataFrame:
        """
        Get historical trade data for a token.
        
        Args:
            token_id: The ID of the token
            hours: Number of hours of data to retrieve
            
        Returns:
            DataFrame with trade data
        """
        try:
            # Get start and end time
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
                trades_df = pd.DataFrame(historical_data)
            else:
                trades_df = historical_data
            
            # Ensure required columns exist
            if 'timestamp' not in trades_df.columns:
                # Try alternate field names for timestamp
                timestamp_alternatives = ['time', 'created_at', 'trade_time', 'date', 'block_time', 'transaction_time']
                for alt_col in timestamp_alternatives:
                    if alt_col in trades_df.columns:
                        trades_df['timestamp'] = trades_df[alt_col]
                        logger.info(f"Using {alt_col} as timestamp for token {token_id}")
                        break
                
                # Still no timestamp column?
                if 'timestamp' not in trades_df.columns:
                    logger.warning(f"No timestamp column found in trade data for {token_id}")
                    if trades_df.columns.tolist():
                        logger.debug(f"Available columns: {trades_df.columns.tolist()}")
                    return pd.DataFrame()
            
            # Ensure price column exists
            if 'price' not in trades_df.columns:
                if 'last_price' in trades_df.columns:
                    trades_df['price'] = trades_df['last_price']
                else:
                    logger.warning(f"No price column found in trade data for {token_id}")
                    return pd.DataFrame()
            
            # Ensure volume column exists
            if 'volume' not in trades_df.columns:
                if 'amount' in trades_df.columns:
                    trades_df['volume'] = trades_df['amount']
                else:
                    logger.warning(f"No volume column found in trade data for {token_id}")
                    return pd.DataFrame()
            
            # Sort by timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            return trades_df
            
        except Exception as e:
            logger.error(f"Error getting trade data for {token_id}: {e}")
            return pd.DataFrame()
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified pump detection feature for the given token.
        
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
                logger.warning(f"Feature '{feature_name}' is not provided by PumpDetectionFeatureProvider")
                return None
            
            # Get trade data
            trade_data = data.get('trade_data')
            
            # If trade data not provided in data, fetch it
            if not trade_data or not isinstance(trade_data, pd.DataFrame) or trade_data.empty:
                hours_required = 24
                if feature_name in ['price_velocity', 'volume_surge_ratio', 'pump_pattern_score']:
                    hours_required = 2
                
                trade_data = self._get_trade_data(token_id, hours=hours_required)
                
                if trade_data.empty:
                    logger.warning(f"No trade data available for token {token_id}")
                    return None
                
                # Cache for future use
                data['trade_data'] = trade_data
            
            # Compute price velocity
            if feature_name == 'price_velocity':
                # Calculate price change per second over the last 5 minutes
                recent_data = self._get_recent_data(trade_data, minutes=5)
                if recent_data.empty or len(recent_data) < 2:
                    return 0.0
                
                time_diff = (recent_data['timestamp'].iloc[-1] - recent_data['timestamp'].iloc[0])
                if isinstance(time_diff, timedelta):
                    time_diff = time_diff.total_seconds()
                
                if time_diff <= 0:
                    return 0.0
                
                price_diff = recent_data['price'].iloc[-1] - recent_data['price'].iloc[0]
                return price_diff / time_diff
            
            # Compute price acceleration
            elif feature_name == 'price_acceleration':
                velocity = data.get('price_velocity', 0)
                
                # Calculate previous velocity (from 10-5 minutes ago)
                recent_data = self._get_recent_data(trade_data, minutes=10)
                if recent_data.empty or len(recent_data) < 3:
                    return 0.0
                
                # Split the data in half
                mid_point = len(recent_data) // 2
                prev_data = recent_data.iloc[:mid_point]
                
                if len(prev_data) < 2:
                    return 0.0
                
                time_diff = (prev_data['timestamp'].iloc[-1] - prev_data['timestamp'].iloc[0])
                if isinstance(time_diff, timedelta):
                    time_diff = time_diff.total_seconds()
                
                if time_diff <= 0:
                    return 0.0
                
                price_diff = prev_data['price'].iloc[-1] - prev_data['price'].iloc[0]
                prev_velocity = price_diff / time_diff
                
                # Acceleration is change in velocity
                time_between = (recent_data['timestamp'].iloc[-1] - prev_data['timestamp'].iloc[-1])
                if isinstance(time_between, timedelta):
                    time_between = time_between.total_seconds()
                
                if time_between <= 0:
                    return 0.0
                
                return (velocity - prev_velocity) / time_between
            
            # Compute volume surge ratio
            elif feature_name == 'volume_surge_ratio':
                # Get volumes
                volume_1m = data.get('volume_1m')
                volume_1h = data.get('volume_1h')
                
                # If not available in data, calculate
                if volume_1m is None or volume_1h is None:
                    recent_data = self._get_recent_data(trade_data, minutes=60)
                    if recent_data.empty:
                        return 1.0
                    
                    # Get 1-minute volume
                    last_minute_data = recent_data[recent_data['timestamp'] >= (datetime.now() - timedelta(minutes=1))]
                    volume_1m = last_minute_data['volume'].sum() if not last_minute_data.empty else 0
                    
                    # Get hourly average (per minute)
                    volume_1h = recent_data['volume'].sum() / 60.0
                
                # Avoid division by zero
                if volume_1h <= 0:
                    return 1.0
                
                return volume_1m / volume_1h
            
            # Compute volume volatility
            elif feature_name == 'volume_volatility':
                recent_data = self._get_recent_data(trade_data, minutes=60)
                if recent_data.empty or len(recent_data) < 5:
                    return 0.5  # Default moderate volatility
                
                # Group by minute and sum volumes
                if isinstance(recent_data['timestamp'].iloc[0], (int, float)):
                    recent_data['datetime'] = pd.to_datetime(recent_data['timestamp'], unit='s')
                else:
                    recent_data['datetime'] = recent_data['timestamp']
                
                # Group by minute
                recent_data['minute'] = recent_data['datetime'].dt.floor('min')
                volume_per_minute = recent_data.groupby('minute')['volume'].sum()
                
                if len(volume_per_minute) < 5:
                    return 0.5  # Default moderate volatility
                
                # Calculate coefficient of variation
                std_dev = volume_per_minute.std()
                mean_volume = volume_per_minute.mean()
                
                if mean_volume > 0:
                    return std_dev / mean_volume
                
                return 0.5  # Default moderate volatility
            
            # Compute buy-sell volume ratio
            elif feature_name == 'buy_sell_volume_ratio':
                recent_data = self._get_recent_data(trade_data, minutes=30)
                if recent_data.empty:
                    return 1.0  # Default neutral ratio
                
                # Check if we have trade type information
                if 'trade_type' in recent_data.columns:
                    # Filter buy and sell trades
                    buy_trades = recent_data[recent_data['trade_type'] == 'buy']
                    sell_trades = recent_data[recent_data['trade_type'] == 'sell']
                    
                    buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
                    sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
                    
                    # Avoid division by zero
                    if sell_volume <= 0:
                        return 10.0 if buy_volume > 0 else 1.0
                    
                    return buy_volume / sell_volume
                
                # Alternative column names
                elif 'is_buy' in recent_data.columns:
                    buy_trades = recent_data[recent_data['is_buy'] == True]
                    sell_trades = recent_data[recent_data['is_buy'] == False]
                    
                    buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
                    sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
                    
                    # Avoid division by zero
                    if sell_volume <= 0:
                        return 10.0 if buy_volume > 0 else 1.0
                    
                    return buy_volume / sell_volume
                
                return 1.0  # Default neutral ratio if we can't determine
            
            # Compute order imbalance
            elif feature_name == 'order_imbalance':
                # This requires order book data
                order_book = data.get('order_book')
                if not order_book:
                    return 0.0
                
                bids = order_book.get('bids', [])
                asks = order_book.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                
                # Calculate total volume on each side
                bid_volume = sum(bid[1] for bid in bids)
                ask_volume = sum(ask[1] for ask in asks)
                
                # Avoid division by zero
                total_volume = bid_volume + ask_volume
                if total_volume <= 0:
                    return 0.0
                
                # Normalized imbalance (-1 to 1)
                return (bid_volume - ask_volume) / total_volume
            
            # Standard deviation of rush orders
            elif feature_name == 'std_rush_order':
                recent_data = self._get_recent_data(trade_data, minutes=30)
                if recent_data.empty or len(recent_data) < 5:
                    return 0.0
                
                # Define rush orders as orders with volume > 2x average
                avg_volume = recent_data['volume'].mean()
                rush_orders = recent_data[recent_data['volume'] > (2 * avg_volume)]
                
                if rush_orders.empty:
                    return 0.0
                
                return rush_orders['volume'].std() / avg_volume
            
            # Average rush order size
            elif feature_name == 'avg_rush_order':
                recent_data = self._get_recent_data(trade_data, minutes=30)
                if recent_data.empty or len(recent_data) < 5:
                    return 0.0
                
                # Define rush orders as orders with volume > 2x average
                avg_volume = recent_data['volume'].mean()
                rush_orders = recent_data[recent_data['volume'] > (2 * avg_volume)]
                
                if rush_orders.empty:
                    return 0.0
                
                return rush_orders['volume'].mean() / avg_volume
            
            # Compute price deviation
            elif feature_name == 'price_deviation':
                current_price = data.get('price_current')
                sma_50 = data.get('sma_50')
                
                if current_price is None or sma_50 is None:
                    # Try to calculate from trade data
                    if trade_data.empty:
                        return 0.0
                    
                    current_price = trade_data['price'].iloc[-1]
                    
                    # Calculate SMA50
                    if len(trade_data) >= 50:
                        sma_50 = trade_data['price'].rolling(window=50).mean().iloc[-1]
                    else:
                        return 0.0
                
                # Avoid division by zero
                if sma_50 <= 0:
                    return 0.0
                
                # Return deviation as percentage
                return ((current_price - sma_50) / sma_50) * 100
            
            # Compute short-term price volatility
            elif feature_name == 'price_volatility_short':
                recent_data = self._get_recent_data(trade_data, minutes=15)
                if recent_data.empty or len(recent_data) < 5:
                    return 0.0
                
                # Calculate using price standard deviation / mean
                mean_price = recent_data['price'].mean()
                if mean_price <= 0:
                    return 0.0
                
                return (recent_data['price'].std() / mean_price) * 100
            
            # Compute price volatility ratio (short-term to long-term)
            elif feature_name == 'price_volatility_ratio':
                volatility_short = data.get('price_volatility_short')
                volatility_24h = data.get('price_volatility_24h')
                
                if volatility_short is None or volatility_24h is None:
                    # Try to calculate from trade data
                    if trade_data.empty:
                        return 1.0
                    
                    # Short-term volatility (15 min)
                    short_data = self._get_recent_data(trade_data, minutes=15)
                    if short_data.empty or len(short_data) < 5:
                        return 1.0
                    
                    mean_price_short = short_data['price'].mean()
                    if mean_price_short <= 0:
                        return 1.0
                    
                    volatility_short = (short_data['price'].std() / mean_price_short) * 100
                    
                    # Long-term volatility (24h)
                    if len(trade_data) < 5:
                        return 1.0
                    
                    mean_price_long = trade_data['price'].mean()
                    if mean_price_long <= 0:
                        return 1.0
                    
                    volatility_24h = (trade_data['price'].std() / mean_price_long) * 100
                
                # Avoid division by zero
                if volatility_24h <= 0:
                    return 5.0 if volatility_short > 0 else 1.0
                
                return volatility_short / volatility_24h
            
            # Compute pump pattern score
            elif feature_name == 'pump_pattern_score':
                # Get required features
                price_velocity = data.get('price_velocity', 0)
                volume_surge_ratio = data.get('volume_surge_ratio', 1)
                price_deviation = data.get('price_deviation', 0)
                
                # Normalize components to 0-1 range
                norm_velocity = min(1.0, max(0.0, price_velocity / 0.01))  # Assuming 0.01 is a high velocity
                norm_volume_surge = min(1.0, max(0.0, (volume_surge_ratio - 1) / 5))  # Surge ratio > 1
                norm_deviation = min(1.0, max(0.0, price_deviation / 30))  # 30% deviation is high
                
                # Weighted score (0-1 range)
                score = (norm_velocity * 0.4) + (norm_volume_surge * 0.4) + (norm_deviation * 0.2)
                
                return score
            
            # Compute dump pattern score
            elif feature_name == 'dump_pattern_score':
                # Get required features
                price_velocity = data.get('price_velocity', 0)
                volume_surge_ratio = data.get('volume_surge_ratio', 1)
                price_deviation = data.get('price_deviation', 0)
                
                # For dump, we look for negative price velocity with high volume
                norm_velocity = min(1.0, max(0.0, -price_velocity / 0.01))  # Negative velocity
                norm_volume_surge = min(1.0, max(0.0, (volume_surge_ratio - 1) / 5))
                norm_deviation = min(1.0, max(0.0, -price_deviation / 30))  # Negative deviation
                
                # Weighted score (0-1 range)
                score = (norm_velocity * 0.4) + (norm_volume_surge * 0.4) + (norm_deviation * 0.2)
                
                return score
            
            # Compute pump phase detection
            elif feature_name == 'pump_phase_detection':
                # Get pump and dump scores
                pump_score = data.get('pump_pattern_score')
                dump_score = data.get('dump_pattern_score')
                
                if pump_score is None or dump_score is None:
                    # Try to compute
                    pump_score = self.compute_feature('pump_pattern_score', token_id, data)
                    dump_score = self.compute_feature('dump_pattern_score', token_id, data)
                
                # Phase detection
                # 0 = No pump/dump
                # 1 = Early accumulation
                # 2 = Pump in progress
                # 3 = Peak/distribution
                # 4 = Dump in progress
                
                if pump_score < 0.2 and dump_score < 0.2:
                    return 0  # No pump/dump
                elif pump_score > 0.6 and dump_score < 0.2:
                    return 2  # Pump in progress
                elif dump_score > 0.6:
                    return 4  # Dump in progress
                elif pump_score > 0.3 and dump_score < 0.3:
                    return 1  # Early accumulation
                else:
                    return 3  # Peak/distribution
            
            # Compute minutes since volume peak
            elif feature_name == 'minute_since_volume_peak':
                recent_data = self._get_recent_data(trade_data, minutes=60)
                if recent_data.empty:
                    return 60  # Default: assume peak was an hour ago
                
                # Group by minute
                if isinstance(recent_data['timestamp'].iloc[0], (int, float)):
                    recent_data['datetime'] = pd.to_datetime(recent_data['timestamp'], unit='s')
                else:
                    recent_data['datetime'] = recent_data['timestamp']
                
                recent_data['minute'] = recent_data['datetime'].dt.floor('min')
                volume_per_minute = recent_data.groupby('minute')['volume'].sum()
                
                if volume_per_minute.empty:
                    return 60
                
                # Find the peak volume minute
                peak_minute = volume_per_minute.idxmax()
                
                # Calculate minutes since peak
                now = pd.Timestamp.now().floor('min')
                if isinstance(peak_minute, pd.Timestamp):
                    minutes_since_peak = (now - peak_minute).total_seconds() / 60
                    return max(0, int(minutes_since_peak))
                
                return 60  # Default
            
            # Compute minutes since price peak
            elif feature_name == 'minute_since_price_peak':
                recent_data = self._get_recent_data(trade_data, minutes=60)
                if recent_data.empty:
                    return 60  # Default: assume peak was an hour ago
                
                # Group by minute and get max price
                if isinstance(recent_data['timestamp'].iloc[0], (int, float)):
                    recent_data['datetime'] = pd.to_datetime(recent_data['timestamp'], unit='s')
                else:
                    recent_data['datetime'] = recent_data['timestamp']
                
                recent_data['minute'] = recent_data['datetime'].dt.floor('min')
                price_per_minute = recent_data.groupby('minute')['price'].max()
                
                if price_per_minute.empty:
                    return 60
                
                # Find the peak price minute
                peak_minute = price_per_minute.idxmax()
                
                # Calculate minutes since peak
                now = pd.Timestamp.now().floor('min')
                if isinstance(peak_minute, pd.Timestamp):
                    minutes_since_peak = (now - peak_minute).total_seconds() / 60
                    return max(0, int(minutes_since_peak))
                
                return 60  # Default
            
            # Compute abnormal activity score
            elif feature_name == 'abnormal_activity_score':
                # Get component features
                pump_score = data.get('pump_pattern_score')
                dump_score = data.get('dump_pattern_score')
                volume_volatility = data.get('volume_volatility')
                price_volatility_ratio = data.get('price_volatility_ratio')
                
                # Calculate if missing
                if pump_score is None:
                    pump_score = self.compute_feature('pump_pattern_score', token_id, data)
                if dump_score is None:
                    dump_score = self.compute_feature('dump_pattern_score', token_id, data)
                if volume_volatility is None:
                    volume_volatility = self.compute_feature('volume_volatility', token_id, data)
                if price_volatility_ratio is None:
                    price_volatility_ratio = self.compute_feature('price_volatility_ratio', token_id, data)
                
                # Normalize factors
                event_score = max(pump_score or 0, dump_score or 0)
                vol_vol_norm = min(1.0, (volume_volatility or 0) / 2.0)
                price_vol_ratio_norm = min(1.0, (price_volatility_ratio or 1.0) / 5.0)
                
                # Weighted combination
                score = (event_score * 0.5) + (vol_vol_norm * 0.3) + (price_vol_ratio_norm * 0.2)
                
                return score
            
            else:
                logger.warning(f"Unknown pump detection feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing pump detection feature '{feature_name}' for token {token_id}: {e}")
            return None
    
    def _get_recent_data(self, trade_data: pd.DataFrame, minutes: int = 30) -> pd.DataFrame:
        """
        Get recent trade data filtered by time.
        
        Args:
            trade_data: DataFrame with all trade data
            minutes: Number of minutes of recent data to return
            
        Returns:
            DataFrame with filtered recent data
        """
        if trade_data.empty:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        # Handle different timestamp formats
        if isinstance(trade_data['timestamp'].iloc[0], (int, float)):
            cutoff_timestamp = cutoff_time.timestamp()
            return trade_data[trade_data['timestamp'] >= cutoff_timestamp]
        else:
            return trade_data[trade_data['timestamp'] >= cutoff_time] 