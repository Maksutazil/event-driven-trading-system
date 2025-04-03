#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Feature Module

This module provides features for generating trading signals based on
price and other metrics.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np

from .interfaces import Feature

logger = logging.getLogger(__name__)


class PriceMomentumSignalFeature(Feature):
    """
    Price momentum trading signal feature.
    
    This feature calculates a trading signal based on price momentum,
    using a combination of price change, moving averages, and volatility.
    """
    
    def __init__(self, threshold: float = 3.0, sensitivity: float = 1.0):
        """
        Initialize the signal feature.
        
        Args:
            threshold: Threshold for signal generation (default: 3.0)
            sensitivity: Sensitivity multiplier for signal strength (default: 1.0)
        """
        self._name = "price_momentum_signal"
        self._threshold = threshold
        self._sensitivity = sensitivity
        logger.info(f"Initialized {self.__class__.__name__} with threshold={threshold}, sensitivity={sensitivity}")
    
    @property
    def name(self) -> str:
        """
        Get the name of the feature.
        
        Returns:
            Feature name
        """
        return self._name
    
    @property
    def dependencies(self) -> List[str]:
        """
        Get the list of features this feature depends on.
        
        Returns:
            List of dependency names
        """
        return [
            "current_price",
            "price_change_pct_5m",
            "price_change_pct_15m",
            "ma_5m",
            "ma_15m",
            "volatility_5m",
            "rsi_14"
        ]
    
    def compute(self, context: Dict[str, Any]) -> float:
        """
        Compute the price momentum signal from the context.
        
        Args:
            context: Dictionary of context data, including price features
            
        Returns:
            Signal value (-1.0 to 1.0), where:
              - Positive values indicate buy signals (stronger as it approaches 1.0)
              - Negative values indicate sell signals (stronger as it approaches -1.0)
              - Values near zero indicate neutral signals
        """
        # Check if all dependencies are available
        for dep in self.dependencies:
            if dep not in context:
                logger.warning(f"Cannot compute {self.name}: missing dependency {dep}")
                return 0.0
        
        # Extract relevant features
        current_price = context["current_price"]
        price_change_5m = context["price_change_pct_5m"]
        price_change_15m = context["price_change_pct_15m"]
        ma_5m = context["ma_5m"]
        ma_15m = context["ma_15m"]
        volatility_5m = context["volatility_5m"]
        rsi = context["rsi_14"]
        
        # Compute signal components
        
        # 1. Price change component: stronger signals for larger price changes
        change_component = (price_change_5m + price_change_15m * 0.5) / (volatility_5m + 1.0)
        
        # 2. Moving average component: positive when price above MAs, negative when below
        ma_component = 0.0
        if ma_5m > 0 and ma_15m > 0:  # Valid MAs available
            ma_component = ((current_price / ma_5m - 1.0) + 
                           (current_price / ma_15m - 1.0) * 2.0)
        
        # 3. RSI component: buy when RSI low, sell when RSI high
        rsi_component = 0.0
        if 0 <= rsi <= 100:  # Valid RSI
            # Transform RSI from 0-100 scale to -1 to 1 scale
            # 50 is neutral, below 30 is strong buy, above 70 is strong sell
            rsi_component = (50 - rsi) / 50.0
        
        # 4. Volatility adjustment - reduce signal strength in high volatility
        volatility_factor = 1.0 / (1.0 + volatility_5m / 10.0)
        
        # Combine components into final signal
        raw_signal = (
            change_component * 0.4 + 
            ma_component * 0.3 + 
            rsi_component * 0.3
        ) * volatility_factor * self._sensitivity
        
        # Apply threshold and normalize to -1.0 to 1.0 range
        if abs(raw_signal) < self._threshold:
            signal = 0.0
        else:
            # Clip and normalize signal
            signal = np.clip(raw_signal / (self._threshold * 2.0), -1.0, 1.0)
        
        logger.debug(f"Computed signal {self.name}: {signal:.4f} (raw: {raw_signal:.4f})")
        return signal


class VolumeSpikeTradingSignalFeature(Feature):
    """
    Volume spike trading signal feature.
    
    This feature detects unusual trading activity based on volume spikes,
    and generates trading signals based on volume and price movements.
    """
    
    def __init__(self, volume_threshold: float = 3.0, price_threshold: float = 1.0):
        """
        Initialize the volume spike signal feature.
        
        Args:
            volume_threshold: Threshold for volume spike detection (default: 3.0)
            price_threshold: Min price change to register a signal (default: 1.0%)
        """
        self._name = "volume_spike_signal"
        self._volume_threshold = volume_threshold
        self._price_threshold = price_threshold
        logger.info(f"Initialized {self.__class__.__name__} with volume_threshold={volume_threshold}")
    
    @property
    def name(self) -> str:
        """
        Get the name of the feature.
        
        Returns:
            Feature name
        """
        return self._name
    
    @property
    def dependencies(self) -> List[str]:
        """
        Get the list of features this feature depends on.
        
        Returns:
            List of dependency names
        """
        return [
            "current_price",
            "price_change_pct_5m",
            "volume_5m",
            "volume_15m"
        ]
    
    def compute(self, context: Dict[str, Any]) -> float:
        """
        Compute the volume spike signal from the context.
        
        Args:
            context: Dictionary of context data, including price and volume features
            
        Returns:
            Signal value (-1.0 to 1.0), where:
              - Positive values indicate buy signals (stronger as it approaches 1.0)
              - Negative values indicate sell signals (stronger as it approaches -1.0)
              - Values near zero indicate neutral signals
        """
        # For demonstration - this would be implemented in a real system
        # Instead, return a neutral signal for now
        return 0.0 