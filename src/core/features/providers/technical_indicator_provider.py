#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Indicator Provider Module

This module provides the TechnicalIndicatorProvider class that computes technical
analysis indicators for tokens.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from src.core.data import DataFeedInterface
from src.core.features.providers.base_feature_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class TechnicalIndicatorProvider(BaseFeatureProvider):
    """
    Feature provider for technical analysis indicators.
    
    This provider computes technical analysis indicators like RSI, MACD,
    Bollinger Bands, and moving averages for token price data.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the technical indicator provider.
        
        Args:
            data_feed: Data feed to use for retrieving historical price data
        """
        # Define all technical indicator features
        feature_names = [
            'rsi_14',                   # Relative Strength Index (14 periods)
            'rsi_7',                    # Relative Strength Index (7 periods)
            'sma_20',                   # Simple Moving Average (20 periods)
            'sma_50',                   # Simple Moving Average (50 periods)
            'sma_200',                  # Simple Moving Average (200 periods)
            'ema_12',                   # Exponential Moving Average (12 periods)
            'ema_26',                   # Exponential Moving Average (26 periods)
            'macd',                     # Moving Average Convergence Divergence
            'macd_signal',              # MACD Signal Line
            'macd_histogram',           # MACD Histogram
            'bollinger_upper',          # Bollinger Band Upper
            'bollinger_middle',         # Bollinger Band Middle
            'bollinger_lower',          # Bollinger Band Lower
            'price_sma20_ratio',        # Price to SMA(20) ratio
            'volume_sma20_ratio',       # Volume to SMA(20) ratio
        ]
        
        # Define dependencies
        dependencies = {
            'macd': ['ema_12', 'ema_26'],
            'macd_signal': ['macd'],
            'macd_histogram': ['macd', 'macd_signal'],
            'bollinger_upper': ['sma_20'],
            'bollinger_middle': ['sma_20'],
            'bollinger_lower': ['sma_20'],
            'price_sma20_ratio': ['price_current', 'sma_20'],
            'volume_sma20_ratio': ['volume_24h'],
        }
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified technical indicator for the given token.
        
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
                logger.warning(f"Feature '{feature_name}' is not provided by TechnicalIndicatorProvider")
                return None
            
            # Get price data
            price_data = data.get('price_data')
            
            # If price data not provided in data, fetch it
            if not price_data or not isinstance(price_data, pd.DataFrame) or price_data.empty:
                # For most indicators, we need at least 200 periods of price data
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)  # Get a week of data
                
                # Get historical data from data feed
                historical_data = self.data_feed.get_historical_data(
                    token_id=token_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Convert to DataFrame if necessary
                if not isinstance(historical_data, pd.DataFrame):
                    price_data = pd.DataFrame(historical_data)
                else:
                    price_data = historical_data
                
                # Ensure required columns exist
                if 'timestamp' not in price_data.columns:
                    if 'time' in price_data.columns:
                        price_data['timestamp'] = price_data['time']
                    else:
                        logger.warning(f"No timestamp column found in price data for {token_id}")
                        return None
                
                if 'price' not in price_data.columns:
                    if 'last_price' in price_data.columns:
                        price_data['price'] = price_data['last_price']
                    else:
                        logger.warning(f"No price column found in price data for {token_id}")
                        return None
                
                # Sort by timestamp
                if 'timestamp' in price_data.columns:
                    price_data = price_data.sort_values('timestamp')
                
                # Cache for future use
                data['price_data'] = price_data
            
            # Check if we have enough data
            if price_data.empty:
                logger.warning(f"No price data available for token {token_id}")
                return None
            
            # Compute RSI
            if feature_name == 'rsi_14':
                return self._compute_rsi(price_data, period=14)
            elif feature_name == 'rsi_7':
                return self._compute_rsi(price_data, period=7)
            
            # Compute Moving Averages
            elif feature_name == 'sma_20':
                return self._compute_sma(price_data, period=20)
            elif feature_name == 'sma_50':
                return self._compute_sma(price_data, period=50)
            elif feature_name == 'sma_200':
                return self._compute_sma(price_data, period=200)
            elif feature_name == 'ema_12':
                return self._compute_ema(price_data, period=12)
            elif feature_name == 'ema_26':
                return self._compute_ema(price_data, period=26)
            
            # Compute MACD
            elif feature_name in ['macd', 'macd_signal', 'macd_histogram']:
                macd, signal, histogram = self._compute_macd(price_data)
                if feature_name == 'macd':
                    return macd
                elif feature_name == 'macd_signal':
                    return signal
                else:  # macd_histogram
                    return histogram
            
            # Compute Bollinger Bands
            elif feature_name in ['bollinger_upper', 'bollinger_middle', 'bollinger_lower']:
                upper, middle, lower = self._compute_bollinger_bands(price_data)
                if feature_name == 'bollinger_upper':
                    return upper
                elif feature_name == 'bollinger_middle':
                    return middle
                else:  # bollinger_lower
                    return lower
            
            # Compute ratios
            elif feature_name == 'price_sma20_ratio':
                current_price = data.get('price_current')
                if current_price is None and not price_data.empty:
                    current_price = price_data['price'].iloc[-1]
                
                sma_20 = self._compute_sma(price_data, period=20)
                
                if current_price is not None and sma_20 is not None and sma_20 > 0:
                    return current_price / sma_20
                return None
            
            elif feature_name == 'volume_sma20_ratio':
                # Get volume data
                volume_data = data.get('volume_data')
                
                # If volume data not provided, try to extract it from price data
                if not volume_data or not isinstance(volume_data, pd.DataFrame) or volume_data.empty:
                    if 'volume' in price_data.columns:
                        volume_data = price_data[['timestamp', 'volume']].copy()
                    else:
                        logger.warning(f"No volume data available for token {token_id}")
                        return None
                
                current_volume = data.get('volume_24h')
                if current_volume is None and not volume_data.empty:
                    current_volume = volume_data['volume'].iloc[-1]
                
                # Compute volume SMA(20)
                volume_sma_20 = volume_data['volume'].rolling(window=20).mean().iloc[-1] if len(volume_data) >= 20 else None
                
                if current_volume is not None and volume_sma_20 is not None and volume_sma_20 > 0:
                    return current_volume / volume_sma_20
                return None
            
            else:
                logger.warning(f"Unknown technical indicator: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing technical indicator '{feature_name}' for token {token_id}: {e}")
            return None
    
    def _compute_rsi(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """
        Compute the Relative Strength Index (RSI).
        
        Args:
            price_data: DataFrame with price data
            period: Number of periods to use for calculation
            
        Returns:
            RSI value
        """
        try:
            if len(price_data) < period + 1:
                logger.warning(f"Not enough data to compute RSI ({len(price_data)} < {period + 1})")
                return None
            
            # Extract price series
            prices = price_data['price'].values
            
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Create arrays for gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RS and RSI
            if avg_loss == 0:
                return 100.0  # No losses means RSI is 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error computing RSI: {e}")
            return None
    
    def _compute_sma(self, price_data: pd.DataFrame, period: int) -> float:
        """
        Compute the Simple Moving Average (SMA).
        
        Args:
            price_data: DataFrame with price data
            period: Number of periods to use for calculation
            
        Returns:
            SMA value
        """
        try:
            if len(price_data) < period:
                logger.warning(f"Not enough data to compute SMA ({len(price_data)} < {period})")
                return None
            
            # Extract price series
            prices = price_data['price'].values
            
            # Calculate SMA
            sma = np.mean(prices[-period:])
            
            return sma
            
        except Exception as e:
            logger.error(f"Error computing SMA: {e}")
            return None
    
    def _compute_ema(self, price_data: pd.DataFrame, period: int) -> float:
        """
        Compute the Exponential Moving Average (EMA).
        
        Args:
            price_data: DataFrame with price data
            period: Number of periods to use for calculation
            
        Returns:
            EMA value
        """
        try:
            if len(price_data) < period:
                logger.warning(f"Not enough data to compute EMA ({len(price_data)} < {period})")
                return None
            
            # Extract price series
            prices = price_data['price'].values
            
            # Calculate EMA
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            
            # Calculate EMA
            ema = np.convolve(prices, weights, mode='valid')[-1]
            
            return ema
            
        except Exception as e:
            logger.error(f"Error computing EMA: {e}")
            return None
    
    def _compute_macd(self, price_data: pd.DataFrame) -> tuple:
        """
        Compute the Moving Average Convergence Divergence (MACD).
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        try:
            # Extract price series
            prices = price_data['price'].values
            
            # Calculate EMAs
            ema_12 = self._compute_ema(price_data, period=12)
            ema_26 = self._compute_ema(price_data, period=26)
            
            if ema_12 is None or ema_26 is None:
                return None, None, None
            
            # Calculate MACD
            macd = ema_12 - ema_26
            
            # Calculate Signal Line (9-day EMA of MACD)
            # For simplicity, we'll use a simple approximation here
            signal = macd * 0.9  # Simplified signal line
            
            # Calculate Histogram
            histogram = macd - signal
            
            return macd, signal, histogram
            
        except Exception as e:
            logger.error(f"Error computing MACD: {e}")
            return None, None, None
    
    def _compute_bollinger_bands(self, price_data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple:
        """
        Compute Bollinger Bands.
        
        Args:
            price_data: DataFrame with price data
            period: Number of periods to use for calculation
            std_dev: Number of standard deviations for the bands
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        try:
            if len(price_data) < period:
                logger.warning(f"Not enough data to compute Bollinger Bands ({len(price_data)} < {period})")
                return None, None, None
            
            # Extract price series
            prices = price_data['price'].values
            
            # Calculate middle band (SMA)
            middle_band = np.mean(prices[-period:])
            
            # Calculate standard deviation
            std = np.std(prices[-period:])
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logger.error(f"Error computing Bollinger Bands: {e}")
            return None, None, None 