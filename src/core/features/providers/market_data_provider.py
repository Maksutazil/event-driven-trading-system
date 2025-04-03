#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Data Feature Provider

This module provides a feature provider implementation for market data features,
demonstrating integration with the feature registry for standardized naming.
"""

import logging
import time
from typing import Dict, Any, Set, Optional
from datetime import datetime, timedelta

from ...data import MarketDataClient
from ..interfaces import FeatureProvider

logger = logging.getLogger(__name__)


class MarketDataFeatureProvider(FeatureProvider):
    """
    Feature provider for market data features.
    
    This provider extracts and computes features from market data,
    with standardized feature naming for integration with the ML system.
    """
    
    def __init__(self, market_data_client: MarketDataClient, cache_ttl: int = 60):
        """
        Initialize the market data feature provider.
        
        Args:
            market_data_client: Client for accessing market data
            cache_ttl: Cache time-to-live in seconds
        """
        self._client = market_data_client
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Define the features this provider provides
        self._provided_features = {
            # Price features
            "current_price",
            "price_change_pct_5m",
            "price_change_pct_15m",
            
            # Volume features
            "volume_5m",
            "volume_15m",
            
            # Technical indicators
            "ma_5m",
            "ma_15m",
            "rsi_14",
            "macd_histogram",
            "volatility_5m",
        }
        
        logger.info("MarketDataFeatureProvider initialized")
    
    @property
    def name(self) -> str:
        """Get the name of the provider."""
        return "MarketDataFeatureProvider"
    
    @property
    def provides(self) -> Set[str]:
        """Get the set of features provided by this provider."""
        return self._provided_features
    
    def get_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get market data features.
        
        Args:
            context: Context dictionary containing token_id
            
        Returns:
            Dictionary of feature values
        """
        if 'token_id' not in context:
            logger.error("token_id not found in context")
            return {}
        
        token_id = context['token_id']
        
        # Check cache
        if self._is_cache_valid(token_id):
            logger.debug(f"Using cached market data for {token_id}")
            return self._cache[token_id]
        
        # Get market data and compute features
        try:
            market_data = self._get_market_data(token_id)
            features = self._compute_features(token_id, market_data)
            
            # Update cache
            self._cache[token_id] = features
            self._cache_timestamps[token_id] = datetime.now()
            
            return features
        except Exception as e:
            logger.error(f"Error computing market data features for {token_id}: {e}", exc_info=True)
            return {}
    
    def _is_cache_valid(self, token_id: str) -> bool:
        """
        Check if the cache for a token is valid.
        
        Args:
            token_id: ID of the token
            
        Returns:
            True if cache is valid, False otherwise
        """
        if token_id not in self._cache or token_id not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[token_id]
        return cache_age.total_seconds() < self._cache_ttl
    
    def _get_market_data(self, token_id: str) -> Dict[str, Any]:
        """
        Get market data for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary of market data
        """
        # Get current market data
        current_price = self._client.get_current_price(token_id)
        
        # Get historical data for different time periods
        candles_5m = self._client.get_historical_data(
            token_id, 
            interval="5m", 
            limit=30  # For 5-minute period
        )
        
        candles_15m = self._client.get_historical_data(
            token_id, 
            interval="15m", 
            limit=30  # For 15-minute period
        )
        
        # Get volume data
        volume_data = self._client.get_volume_data(token_id, intervals=["5m", "15m"])
        
        # Get technical indicators
        indicators = self._client.get_technical_indicators(
            token_id,
            indicators=["rsi", "macd", "volatility"]
        )
        
        # Combine all data
        return {
            "current_price": current_price,
            "candles_5m": candles_5m,
            "candles_15m": candles_15m,
            "volume_data": volume_data,
            "indicators": indicators
        }
    
    def _compute_features(self, token_id: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute features from market data.
        
        Args:
            token_id: ID of the token
            market_data: Dictionary of market data
            
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        # Extract price features
        if "current_price" in market_data:
            features["current_price"] = market_data["current_price"]
        
        # Compute price change percentages
        if "candles_5m" in market_data and market_data["candles_5m"]:
            candles_5m = market_data["candles_5m"]
            if len(candles_5m) > 1:
                first_price = candles_5m[0]["close"]
                last_price = candles_5m[-1]["close"]
                if first_price > 0:
                    features["price_change_pct_5m"] = ((last_price - first_price) / first_price) * 100.0
                else:
                    features["price_change_pct_5m"] = 0.0
            else:
                features["price_change_pct_5m"] = 0.0
        
        if "candles_15m" in market_data and market_data["candles_15m"]:
            candles_15m = market_data["candles_15m"]
            if len(candles_15m) > 1:
                first_price = candles_15m[0]["close"]
                last_price = candles_15m[-1]["close"]
                if first_price > 0:
                    features["price_change_pct_15m"] = ((last_price - first_price) / first_price) * 100.0
                else:
                    features["price_change_pct_15m"] = 0.0
            else:
                features["price_change_pct_15m"] = 0.0
        
        # Extract volume features
        if "volume_data" in market_data:
            volume_data = market_data["volume_data"]
            if "5m" in volume_data:
                features["volume_5m"] = volume_data["5m"]
            if "15m" in volume_data:
                features["volume_15m"] = volume_data["15m"]
        
        # Compute moving averages
        if "candles_5m" in market_data and market_data["candles_5m"]:
            prices = [candle["close"] for candle in market_data["candles_5m"]]
            if prices:
                features["ma_5m"] = sum(prices) / len(prices)
        
        if "candles_15m" in market_data and market_data["candles_15m"]:
            prices = [candle["close"] for candle in market_data["candles_15m"]]
            if prices:
                features["ma_15m"] = sum(prices) / len(prices)
        
        # Extract technical indicators
        if "indicators" in market_data:
            indicators = market_data["indicators"]
            if "rsi" in indicators:
                features["rsi_14"] = indicators["rsi"]
            if "macd" in indicators and "histogram" in indicators["macd"]:
                features["macd_histogram"] = indicators["macd"]["histogram"]
            if "volatility" in indicators:
                features["volatility_5m"] = indicators["volatility"]
        
        return features
    
    def invalidate_cache(self, token_id: Optional[str] = None) -> None:
        """
        Invalidate the cache for a token or all tokens.
        
        Args:
            token_id: Optional token ID. If None, invalidate for all tokens.
        """
        if token_id is None:
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.debug("Cleared all cached market data")
        elif token_id in self._cache:
            del self._cache[token_id]
            if token_id in self._cache_timestamps:
                del self._cache_timestamps[token_id]
            logger.debug(f"Cleared cached market data for {token_id}") 