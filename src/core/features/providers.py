#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Providers Module

This module provides implementations of the FeatureProvider interface
for common token features.
"""

import time
import logging
import threading
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta

from .interfaces import FeatureProvider
from ..data.interfaces import DataFeedInterface

logger = logging.getLogger(__name__)


class BaseFeatureProvider(FeatureProvider):
    """
    Base implementation of the FeatureProvider interface.
    
    This class provides common functionality for feature providers.
    """
    
    def __init__(self, feature_names: List[str], dependencies: Dict[str, List[str]] = None):
        """
        Initialize the base feature provider.
        
        Args:
            feature_names: List of features provided by this provider
            dependencies: Optional dictionary mapping feature names to their dependencies
        """
        self._feature_names = feature_names
        self._dependencies = dependencies or {}
        
        # Ensure all provided features have an entry in dependencies
        for feature_name in self._feature_names:
            if feature_name not in self._dependencies:
                self._dependencies[feature_name] = []
    
    def get_provided_features(self) -> List[str]:
        """
        Get a list of features provided by this provider.
        
        Returns:
            List[str]: List of feature names
        """
        return self._feature_names
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            List[str]: List of feature dependencies
            
        Raises:
            ValueError: If the feature is not provided by this provider
        """
        if feature_name not in self._feature_names:
            raise ValueError(f"Feature {feature_name} is not provided by this provider")
        
        return self._dependencies.get(feature_name, [])
    
    def compute_feature(self, token_id: str, feature_name: str, dependencies: Dict[str, Any]) -> Any:
        """
        Compute a feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            dependencies: Dictionary mapping dependency names to values
            
        Returns:
            Any: Computed feature value
            
        Raises:
            ValueError: If the feature is not provided by this provider
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError(f"Subclasses must implement compute_feature()")


class PriceFeatureProvider(BaseFeatureProvider):
    """
    Provider for price-related token features.
    
    This provider computes features based on token price data.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the price feature provider.
        
        Args:
            data_feed: Data feed interface for retrieving token price data
        """
        # Define provided features
        features = [
            "price_current",
            "price_change_24h",
            "price_change_pct_24h",
            "volume_24h",
            "market_cap",
            "price_high_24h",
            "price_low_24h",
            "price_volatility_24h"
        ]
        
        # Define dependencies
        dependencies = {
            "price_change_24h": ["price_current"],
            "price_change_pct_24h": ["price_current", "price_change_24h"],
            "price_volatility_24h": ["price_high_24h", "price_low_24h"]
        }
        
        super().__init__(features, dependencies)
        self._data_feed = data_feed
        self._cache_lock = threading.RLock()
        self._price_cache = {}  # token_id -> {timestamp -> price}
    
    def _get_price_data(self, token_id: str, hours: int = 24) -> pd.DataFrame:
        """
        Get price data for a token.
        
        Args:
            token_id: ID of the token
            hours: Number of hours of historical data to retrieve
            
        Returns:
            pd.DataFrame: Dataframe with price data
        """
        with self._cache_lock:
            if token_id in self._price_cache:
                # Check if cache has recent data
                cache = self._price_cache[token_id]
                
                if cache and max(cache.keys()) >= datetime.now() - timedelta(minutes=5):
                    # Convert cache to dataframe
                    df = pd.DataFrame({
                        "timestamp": list(cache.keys()),
                        "price": list(cache.values())
                    })
                    df = df.sort_values("timestamp")
                    return df
        
        # Retrieve data from data feed
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        df = self._data_feed.get_historical_data(
            token_id=token_id,
            start_time=start_time,
            end_time=end_time,
            interval="1m"
        )
        
        # Update cache with new data
        if df is not None and not df.empty and "timestamp" in df.columns and "price" in df.columns:
            with self._cache_lock:
                self._price_cache[token_id] = {
                    row["timestamp"]: row["price"]
                    for _, row in df.iterrows()
                }
        
        return df
    
    def compute_feature(self, token_id: str, feature_name: str, dependencies: Dict[str, Any]) -> Any:
        """
        Compute a price-related feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            dependencies: Dictionary mapping dependency names to values
            
        Returns:
            Any: Computed feature value
        """
        if feature_name not in self.get_provided_features():
            raise ValueError(f"Feature {feature_name} is not provided by this provider")
        
        # Get price data for the token
        price_data = self._get_price_data(token_id)
        
        if price_data is None or price_data.empty:
            logger.warning(f"No price data available for token {token_id}")
            return None
        
        # Compute specific feature
        if feature_name == "price_current":
            # Current price is the most recent price
            return price_data["price"].iloc[-1]
        
        elif feature_name == "price_high_24h":
            # Highest price in the last 24 hours
            return price_data["price"].max()
        
        elif feature_name == "price_low_24h":
            # Lowest price in the last 24 hours
            return price_data["price"].min()
        
        elif feature_name == "volume_24h":
            # Trading volume in the last 24 hours
            if "volume" in price_data.columns:
                return price_data["volume"].sum()
            else:
                return None
        
        elif feature_name == "market_cap":
            # Current market cap (price * supply)
            current_price = dependencies.get("price_current") or price_data["price"].iloc[-1]
            
            if "supply" in price_data.columns:
                return current_price * price_data["supply"].iloc[-1]
            else:
                return None
        
        elif feature_name == "price_change_24h":
            # Price change in the last 24 hours
            if len(price_data) >= 2:
                return price_data["price"].iloc[-1] - price_data["price"].iloc[0]
            else:
                return None
        
        elif feature_name == "price_change_pct_24h":
            # Percentage price change in the last 24 hours
            price_change = dependencies.get("price_change_24h")
            current_price = dependencies.get("price_current")
            
            if price_change is None or current_price is None:
                if len(price_data) >= 2:
                    old_price = price_data["price"].iloc[0]
                    new_price = price_data["price"].iloc[-1]
                    
                    if old_price > 0:
                        return ((new_price - old_price) / old_price) * 100
                
                return None
            
            old_price = current_price - price_change
            if old_price > 0:
                return (price_change / old_price) * 100
            
            return None
        
        elif feature_name == "price_volatility_24h":
            # Price volatility in the last 24 hours (high-low range as percentage)
            high = dependencies.get("price_high_24h")
            low = dependencies.get("price_low_24h")
            
            if high is None or low is None:
                high = price_data["price"].max()
                low = price_data["price"].min()
            
            if low > 0:
                return ((high - low) / low) * 100
            
            return None
        
        # Unknown feature
        return None


class TokenMetadataProvider(BaseFeatureProvider):
    """
    Provider for token metadata features.
    
    This provider computes features based on token metadata.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the token metadata provider.
        
        Args:
            data_feed: Data feed interface for retrieving token metadata
        """
        # Define provided features
        features = [
            "token_name",
            "token_symbol",
            "token_decimals",
            "token_supply",
            "token_creation_time",
            "token_age_days",
            "token_holder_count",
            "token_transaction_count"
        ]
        
        # Define dependencies
        dependencies = {
            "token_age_days": ["token_creation_time"]
        }
        
        super().__init__(features, dependencies)
        self._data_feed = data_feed
        self._cache_lock = threading.RLock()
        self._metadata_cache = {}  # token_id -> {feature_name -> value}
    
    def _get_token_metadata(self, token_id: str) -> Dict[str, Any]:
        """
        Get metadata for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dict[str, Any]: Dictionary of token metadata
        """
        with self._cache_lock:
            if token_id in self._metadata_cache:
                # Return cached data
                return self._metadata_cache[token_id]
        
        # Query for token metadata
        query = f"""
        SELECT
            name, symbol, decimals, total_supply,
            creation_timestamp, holder_count, transaction_count
        FROM tokens
        WHERE token_id = :token_id
        """
        
        params = {"token_id": token_id}
        
        try:
            df = self._data_feed.execute_query(query, params)
            
            if df is not None and not df.empty:
                # Convert to dictionary
                metadata = df.iloc[0].to_dict()
                
                # Cache the metadata
                with self._cache_lock:
                    self._metadata_cache[token_id] = metadata
                
                return metadata
            
        except Exception as e:
            logger.error(f"Error retrieving metadata for token {token_id}: {e}")
        
        return {}
    
    def compute_feature(self, token_id: str, feature_name: str, dependencies: Dict[str, Any]) -> Any:
        """
        Compute a metadata-related feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            dependencies: Dictionary mapping dependency names to values
            
        Returns:
            Any: Computed feature value
        """
        if feature_name not in self.get_provided_features():
            raise ValueError(f"Feature {feature_name} is not provided by this provider")
        
        # Get metadata for the token
        metadata = self._get_token_metadata(token_id)
        
        if not metadata:
            logger.warning(f"No metadata available for token {token_id}")
            return None
        
        # Compute specific feature
        if feature_name == "token_name":
            return metadata.get("name")
        
        elif feature_name == "token_symbol":
            return metadata.get("symbol")
        
        elif feature_name == "token_decimals":
            return metadata.get("decimals")
        
        elif feature_name == "token_supply":
            return metadata.get("total_supply")
        
        elif feature_name == "token_creation_time":
            return metadata.get("creation_timestamp")
        
        elif feature_name == "token_age_days":
            creation_time = dependencies.get("token_creation_time")
            
            if creation_time is None:
                creation_time = metadata.get("creation_timestamp")
            
            if creation_time is not None:
                if isinstance(creation_time, str):
                    try:
                        creation_time = datetime.fromisoformat(creation_time)
                    except ValueError:
                        return None
                
                age = datetime.now() - creation_time
                return age.total_seconds() / (24 * 3600)  # Convert to days
            
            return None
        
        elif feature_name == "token_holder_count":
            return metadata.get("holder_count")
        
        elif feature_name == "token_transaction_count":
            return metadata.get("transaction_count")
        
        # Unknown feature
        return None


class TechnicalIndicatorProvider(BaseFeatureProvider):
    """
    Provider for technical indicator features.
    
    This provider computes technical analysis indicators based on price data.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the technical indicator provider.
        
        Args:
            data_feed: Data feed interface for retrieving price data
        """
        # Define provided features
        features = [
            "rsi_14",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower"
        ]
        
        # Define dependencies (empty as these are computed directly from raw data)
        dependencies = {}
        
        super().__init__(features, dependencies)
        self._data_feed = data_feed
        self._price_provider = PriceFeatureProvider(data_feed)
    
    def compute_feature(self, token_id: str, feature_name: str, dependencies: Dict[str, Any]) -> Any:
        """
        Compute a technical indicator feature value for a token.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            dependencies: Dictionary mapping dependency names to values
            
        Returns:
            Any: Computed feature value
        """
        if feature_name not in self.get_provided_features():
            raise ValueError(f"Feature {feature_name} is not provided by this provider")
        
        # Use PriceFeatureProvider to get price data
        price_data = self._price_provider._get_price_data(token_id, hours=72)  # Get more data for indicators
        
        if price_data is None or price_data.empty or len(price_data) < 30:
            logger.warning(f"Insufficient price data for technical indicators for token {token_id}")
            return None
        
        try:
            # Compute specific indicator
            if feature_name == "rsi_14":
                return self._compute_rsi(price_data, 14)
            
            elif feature_name == "sma_20":
                return self._compute_sma(price_data, 20)
            
            elif feature_name == "sma_50":
                return self._compute_sma(price_data, 50)
            
            elif feature_name == "ema_12":
                return self._compute_ema(price_data, 12)
            
            elif feature_name == "ema_26":
                return self._compute_ema(price_data, 26)
            
            elif feature_name == "macd":
                return self._compute_macd(price_data)[0]
            
            elif feature_name == "macd_signal":
                return self._compute_macd(price_data)[1]
            
            elif feature_name == "macd_histogram":
                macd, signal = self._compute_macd(price_data)
                return macd - signal
            
            elif feature_name.startswith("bollinger_"):
                upper, middle, lower = self._compute_bollinger_bands(price_data)
                
                if feature_name == "bollinger_upper":
                    return upper
                elif feature_name == "bollinger_middle":
                    return middle
                elif feature_name == "bollinger_lower":
                    return lower
            
            # Unknown feature
            return None
            
        except Exception as e:
            logger.error(f"Error computing technical indicator {feature_name} for token {token_id}: {e}")
            return None
    
    def _compute_rsi(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """
        Compute the Relative Strength Index (RSI) for a price series.
        
        Args:
            price_data: Dataframe with price data
            period: Period for RSI calculation
            
        Returns:
            float: RSI value
        """
        # Calculate price changes
        delta = price_data["price"].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Return the most recent RSI value
        return rsi.iloc[-1]
    
    def _compute_sma(self, price_data: pd.DataFrame, period: int) -> float:
        """
        Compute the Simple Moving Average (SMA) for a price series.
        
        Args:
            price_data: Dataframe with price data
            period: Period for SMA calculation
            
        Returns:
            float: SMA value
        """
        sma = price_data["price"].rolling(window=period).mean()
        return sma.iloc[-1]
    
    def _compute_ema(self, price_data: pd.DataFrame, period: int) -> float:
        """
        Compute the Exponential Moving Average (EMA) for a price series.
        
        Args:
            price_data: Dataframe with price data
            period: Period for EMA calculation
            
        Returns:
            float: EMA value
        """
        ema = price_data["price"].ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    
    def _compute_macd(self, price_data: pd.DataFrame) -> tuple:
        """
        Compute the Moving Average Convergence Divergence (MACD) for a price series.
        
        Args:
            price_data: Dataframe with price data
            
        Returns:
            tuple: (MACD line, Signal line)
        """
        # Calculate EMA-12 and EMA-26
        ema12 = price_data["price"].ewm(span=12, adjust=False).mean()
        ema26 = price_data["price"].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema12 - ema26
        
        # Calculate Signal line (9-day EMA of MACD line)
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Return the most recent values
        return macd_line.iloc[-1], signal_line.iloc[-1]
    
    def _compute_bollinger_bands(self, price_data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple:
        """
        Compute Bollinger Bands for a price series.
        
        Args:
            price_data: Dataframe with price data
            period: Period for SMA calculation
            std_dev: Number of standard deviations
            
        Returns:
            tuple: (Upper band, Middle band, Lower band)
        """
        # Calculate middle band (SMA)
        middle_band = price_data["price"].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = price_data["price"].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Return the most recent values
        return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1] 