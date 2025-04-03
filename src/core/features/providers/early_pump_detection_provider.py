#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Early Pump Detection Feature Provider

This module provides features designed specifically for detecting pump patterns
in newly created tokens with minimal trading history.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.core.data import DataFeedInterface
from src.core.features.providers.base_feature_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class EarlyPumpDetectionProvider(BaseFeatureProvider):
    """
    Feature provider for early-stage pump detection with minimal history requirements.
    
    This provider is specifically designed for newly created tokens with limited 
    trading history, focusing on real-time signals that can be calculated with 
    minimal data points.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the early pump detection feature provider.
        
        Args:
            data_feed: Data feed to use for retrieving trade data
        """
        # Define features that work with minimal data
        feature_names = [
            # Immediate price metrics
            'immediate_price_change',     # Percentage price change in very short term
            'trade_frequency',            # Number of trades per minute
            'buyer_dominance',            # Ratio of buys to total trades
            'volume_intensity',           # Volume relative to token age
            'early_pump_score',           # Combined score for early pump detection
        ]
        
        # Simple dependencies (minimal for early detection)
        dependencies = {
            'early_pump_score': ['immediate_price_change', 'trade_frequency', 
                                 'buyer_dominance', 'volume_intensity'],
        }
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
        self._name = "EarlyPumpDetectionProvider"
    
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
    
    def _get_minimal_trade_data(self, token_id: str, data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get minimal trade data needed for early detection.
        
        Args:
            token_id: ID of the token
            data: Optional data containing trade history
            
        Returns:
            List of trade records, even if very few
        """
        # Try to get trades from provided data
        if data and 'trade_data' in data and isinstance(data['trade_data'], list):
            return data['trade_data']
            
        # Try to get from data feed (last 5 minutes max)
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
            
            trades = self.data_feed.get_historical_data(
                token_id=token_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # If we got a DataFrame, convert to list of dicts
            if hasattr(trades, 'to_dict'):
                return trades.to_dict('records')
                
            return trades if isinstance(trades, list) else []
        except Exception as e:
            logger.debug(f"Error getting minimal trade data for {token_id}: {e}")
            return []
    
    def _get_default_value(self, feature_name: str) -> Any:
        """
        Get default value for a feature when insufficient data is available.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Default value appropriate for the feature
        """
        defaults = {
            'immediate_price_change': 0.0,
            'trade_frequency': 0.0,
            'buyer_dominance': 0.5,
            'volume_intensity': 0.0,
            'early_pump_score': 0.0,
        }
        return defaults.get(feature_name, 0.0)
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute early-stage pump detection features with minimal data requirements.
        
        Args:
            feature_name: Name of feature to compute
            token_id: ID of the token
            data: Dictionary containing any available data
            
        Returns:
            Computed feature value or default if insufficient data
        """
        try:
            # Get available trade data (even if minimal)
            trades = self._get_minimal_trade_data(token_id, data)
            if not trades or len(trades) < 2:
                logger.debug(f"Insufficient trade data for {token_id} to calculate {feature_name}")
                return self._get_default_value(feature_name)
                
            # Calculate immediate price change (works with just 2 trades)
            if feature_name == 'immediate_price_change':
                first_price = trades[0].get('price', 0)
                last_price = trades[-1].get('price', 0)
                if first_price <= 0:
                    return 0.0
                return (last_price / first_price - 1) * 100
                
            # Calculate trade frequency (trades per minute)
            elif feature_name == 'trade_frequency':
                first_time = trades[0].get('timestamp', 0)
                last_time = trades[-1].get('timestamp', 0)
                # Convert to numeric timestamps if needed
                if isinstance(first_time, str) or isinstance(first_time, datetime):
                    first_time = datetime.fromisoformat(first_time).timestamp() if isinstance(first_time, str) else first_time.timestamp()
                if isinstance(last_time, str) or isinstance(last_time, datetime):
                    last_time = datetime.fromisoformat(last_time).timestamp() if isinstance(last_time, str) else last_time.timestamp()
                    
                time_diff = last_time - first_time
                if time_diff <= 0:
                    return len(trades) # If all trades at same time, use count as frequency
                minutes = time_diff / 60.0
                return len(trades) / max(minutes, 0.1)  # Avoid division by zero
                
            # Calculate buyer dominance (ratio of buys to total)
            elif feature_name == 'buyer_dominance':
                # Look for trade_type, side, or is_buy fields
                buy_trades = 0
                for trade in trades:
                    if 'trade_type' in trade and trade['trade_type'] == 'buy':
                        buy_trades += 1
                    elif 'side' in trade and trade['side'] == 'buy':
                        buy_trades += 1
                    elif 'is_buy' in trade and trade['is_buy']:
                        buy_trades += 1
                
                return buy_trades / len(trades)
                
            # Calculate volume intensity (volume normalized by age)
            elif feature_name == 'volume_intensity':
                # Get token age in minutes
                first_time = trades[0].get('timestamp', 0)
                current_time = time.time()
                
                # Convert to numeric timestamp if needed
                if isinstance(first_time, str) or isinstance(first_time, datetime):
                    first_time = datetime.fromisoformat(first_time).timestamp() if isinstance(first_time, str) else first_time.timestamp()
                
                age_minutes = max(1, (current_time - first_time) / 60)
                
                # Calculate total volume
                total_volume = sum(trade.get('volume', 0) for trade in trades)
                
                # Weight more heavily for newer tokens
                age_factor = max(0.2, min(1.0, 5 / age_minutes))  # 0.2-1.0 range
                return total_volume * age_factor / len(trades)
                
            # Calculate the early pump score
            elif feature_name == 'early_pump_score':
                # Get component features
                price_change = data.get('immediate_price_change', 
                                  self.compute_feature('immediate_price_change', token_id, data))
                trade_freq = data.get('trade_frequency', 
                                self.compute_feature('trade_frequency', token_id, data))
                buyer_dom = data.get('buyer_dominance', 
                               self.compute_feature('buyer_dominance', token_id, data))
                volume_int = data.get('volume_intensity', 
                                 self.compute_feature('volume_intensity', token_id, data))
                
                # Normalize each component to 0-1 range
                norm_price = min(1.0, max(0, price_change / 10.0))  # 10% price increase → 1.0
                norm_freq = min(1.0, trade_freq / 10.0)  # 10 trades per minute → 1.0
                norm_buyers = min(1.0, max(0, (buyer_dom - 0.5) * 2))  # 100% buyers → 1.0
                norm_volume = min(1.0, volume_int / 100)  # Normalize volume intensity
                
                # Combined score with weighted components
                return (norm_price * 0.35 + 
                        norm_freq * 0.25 + 
                        norm_buyers * 0.25 + 
                        norm_volume * 0.15)
            
            else:
                logger.warning(f"Unknown feature: {feature_name}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error computing early pump feature '{feature_name}' for {token_id}: {e}")
            return self._get_default_value(feature_name) 