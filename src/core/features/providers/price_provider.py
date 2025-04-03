#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Price Feature Provider Module

This module provides a feature provider that computes price-related features
for tokens based on market data.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

import numpy as np

from .base_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class PriceFeatureProvider(BaseFeatureProvider):
    """
    Feature provider for price-related features.
    
    This provider computes basic price features such as price change percentage,
    moving averages, volatility, and other price-derived metrics.
    """
    
    def __init__(self, name: str = "price_provider", max_history: int = 100):
        """
        Initialize the price feature provider.
        
        Args:
            name: Provider name
            max_history: Maximum number of price points to keep in history
        """
        super().__init__(name)
        self.max_history = max_history
        
        # Store price history for each token
        # {token_id: {timestamp: price}}
        self.price_history: Dict[str, Dict[datetime, float]] = {}
        
        # Define provided features
        self._provides = {
            "current_price",
            "price_change_pct_1m",
            "price_change_pct_5m",
            "price_change_pct_15m",
            "price_change_pct_1h",
            "ma_5m",
            "ma_15m",
            "ma_1h",
            "volatility_5m",
            "volatility_15m",
            "rsi_14"
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self._provides)} features")
    
    def get_dependencies(self, feature_name: str) -> Set[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature_name: Name of the feature to get dependencies for
            
        Returns:
            Set[str]: Set of feature names this feature depends on
        """
        # This is a base provider, no feature depends on other features
        return set()
    
    def update_price(self, token_id: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update the price history for a token.
        
        Args:
            token_id: ID of the token
            price: Current price of the token
            timestamp: Optional timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize price history for token if not exists
        if token_id not in self.price_history:
            self.price_history[token_id] = {}
        
        # Add price to history
        self.price_history[token_id][timestamp] = price
        
        # Limit history size
        if len(self.price_history[token_id]) > self.max_history:
            # Remove oldest entries
            sorted_timestamps = sorted(self.price_history[token_id].keys())
            for ts in sorted_timestamps[:len(sorted_timestamps) - self.max_history]:
                del self.price_history[token_id][ts]
        
        logger.debug(f"Updated price for {token_id}: {price} at {timestamp}")
    
    def get_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute and return price features for the given context.
        
        Args:
            context: Context data for computation
            
        Returns:
            Dictionary of computed price feature values
        """
        token_id = context.get('token_id')
        current_price = context.get('price')
        timestamp = context.get('timestamp', datetime.now())
        
        # Initialize features dictionary with default values
        features = {
            "current_price": 0.0,
            "price_change_pct_1m": 0.0,
            "price_change_pct_5m": 0.0,
            "price_change_pct_15m": 0.0,
            "price_change_pct_1h": 0.0,
            "ma_5m": 0.0,
            "ma_15m": 0.0,
            "ma_1h": 0.0,
            "volatility_5m": 0.0,
            "volatility_15m": 0.0,
            "rsi_14": 50.0  # Neutral default
        }
        
        # Can't compute features without token_id
        if token_id is None:
            logger.warning("Cannot compute price features: token_id not provided in context")
            return features
        
        # Update price history if current price is provided
        if current_price is not None:
            self.update_price(token_id, current_price, timestamp)
        
        # Return default features if we don't have price history
        if token_id not in self.price_history or not self.price_history[token_id]:
            return features
        
        # Get sorted price history (newest first)
        sorted_history = sorted(
            [(ts, price) for ts, price in self.price_history[token_id].items()],
            key=lambda x: x[0],
            reverse=True
        )
        
        # Current price (most recent)
        current_price = sorted_history[0][1]
        features["current_price"] = current_price
        
        # Price change percentages
        for minutes, feature_name in [(1, "price_change_pct_1m"), 
                                    (5, "price_change_pct_5m"),
                                    (15, "price_change_pct_15m"),
                                    (60, "price_change_pct_1h")]:
            # Find oldest price within the interval
            cutoff_time = sorted_history[0][0] - timedelta(minutes=minutes)
            older_prices = [(ts, price) for ts, price in sorted_history if ts <= cutoff_time]
            
            if older_prices:
                old_price = older_prices[0][1]  # First price just before cutoff
                if old_price > 0:
                    change_pct = ((current_price - old_price) / old_price) * 100
                    features[feature_name] = change_pct
        
        # Moving averages
        for minutes, feature_name in [(5, "ma_5m"), (15, "ma_15m"), (60, "ma_1h")]:
            cutoff_time = sorted_history[0][0] - timedelta(minutes=minutes)
            recent_prices = [price for ts, price in sorted_history if ts >= cutoff_time]
            
            if recent_prices:
                features[feature_name] = sum(recent_prices) / len(recent_prices)
            else:
                features[feature_name] = current_price
        
        # Volatility (standard deviation of price over period)
        for minutes, feature_name in [(5, "volatility_5m"), (15, "volatility_15m")]:
            cutoff_time = sorted_history[0][0] - timedelta(minutes=minutes)
            recent_prices = [price for ts, price in sorted_history if ts >= cutoff_time]
            
            if len(recent_prices) >= 2:
                mean_price = sum(recent_prices) / len(recent_prices)
                squared_diffs = [(price - mean_price) ** 2 for price in recent_prices]
                variance = sum(squared_diffs) / len(recent_prices)
                std_dev = variance ** 0.5
                features[feature_name] = (std_dev / mean_price) * 100 if mean_price > 0 else 0
        
        # RSI calculation (14 periods)
        if len(sorted_history) >= 15:  # Need at least 15 points for 14-period RSI
            prices = [price for _, price in sorted_history[:15]]  # Get last 15 prices
            gains = []
            losses = []
            
            # Calculate gains and losses
            for i in range(len(prices)-1):
                change = prices[i] - prices[i+1]  # Remember prices are newest first
                if change >= 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            # Calculate average gain and loss
            if gains and losses:  # Ensure we have data
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses) / 14
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features["rsi_14"] = 100 - (100 / (1 + rs))
                else:
                    features["rsi_14"] = 100  # Full strength if no losses
        
        return features 