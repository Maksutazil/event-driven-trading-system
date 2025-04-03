#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Price Provider Module

This module provides an enhanced version of the PriceFeatureProvider that
implements the FeatureConsumer interface and handles token events directly.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from src.core.events import EventBus, Event, EventType
from src.core.features.interfaces import FeatureConsumer
from src.core.features.providers.price_provider import PriceFeatureProvider

logger = logging.getLogger(__name__)


class EnhancedPriceProvider(PriceFeatureProvider, FeatureConsumer):
    """
    Enhanced price provider that implements FeatureConsumer to receive token events.
    
    This provider automatically subscribes to token trade and creation events,
    updating price data without requiring manual event handlers.
    """
    
    def __init__(self, name: str = "price_provider", max_history: int = 100):
        """
        Initialize the enhanced price provider.
        
        Args:
            name: Name of the provider
            max_history: Maximum number of price points to store per token
        """
        PriceFeatureProvider.__init__(self, name=name, max_history=max_history)
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    def get_required_features(self) -> List[str]:
        """
        Get features required by this consumer.
        
        Returns:
            List[str]: List of required feature names
        """
        # We don't require any features, as we're a base provider
        return []
    
    def get_dependencies(self, feature_name: str) -> Set[str]:
        """
        Get dependencies for a feature.
        
        Args:
            feature_name: Name of the feature to get dependencies for
            
        Returns:
            Set[str]: Set of feature names this feature depends on
        """
        # Since we're a base provider, we don't have dependencies
        return set()
    
    def on_feature_update(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Handle feature updates - not used since we're a base provider.
        
        We update our data through event handlers instead.
        
        Args:
            token_id: ID of the token the feature is for
            feature_name: Name of the feature
            value: New feature value
        """
        pass
    
    def handle_token_trade(self, event: Event) -> None:
        """
        Handle token trade events to update price data.
        
        Args:
            event: Token trade event
        """
        try:
            if event.event_type != EventType.TOKEN_TRADE:
                return
            
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Token trade event missing token_id")
                return
            
            # Extract trade data
            price = data.get('price')
            if price is None:
                logger.warning(f"Token trade event missing price: {data}")
                return
                
            timestamp = data.get('timestamp')
            
            # Convert timestamp to datetime if it's a unix timestamp
            if timestamp and isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
                
            # Update price provider with latest price
            self.update_price(token_id, float(price), timestamp)
            logger.debug(f"Updated price provider for {token_id}: price={price}")
            
        except Exception as e:
            logger.error(f"Error updating price provider from trade event: {e}", exc_info=True)
    
    def handle_token_created(self, event: Event) -> None:
        """
        Handle token creation events to initialize price data.
        
        Args:
            event: Token creation event
        """
        try:
            if event.event_type != EventType.TOKEN_CREATED:
                return
            
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Token creation event missing token_id")
                return
            
            # Extract token data
            initial_price = data.get('initial_price')
            if initial_price is None:
                # Try alternative field names that might contain the price
                initial_price = data.get('price', 0.0)
            
            if initial_price is None or float(initial_price) == 0.0:
                logger.warning(f"Token creation event missing valid initial price: {data}")
                return
                
            timestamp = data.get('timestamp')
            
            # Convert timestamp to datetime if it's a unix timestamp
            if timestamp and isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
                
            # Initialize price provider with token's initial price
            self.update_price(token_id, float(initial_price), timestamp)
            logger.info(f"Initialized price provider for new token {token_id}: initial_price={initial_price}")
            
        except Exception as e:
            logger.error(f"Error updating price provider from creation event: {e}", exc_info=True)
    
    def register_with_event_bus(self, event_bus: EventBus) -> None:
        """
        Register this provider with an event bus to receive events.
        
        Args:
            event_bus: EventBus to register with
        """
        if not event_bus:
            return
        
        # Create simple event handlers
        class EventHandler:
            def __init__(self, callback):
                self.callback = callback
                self.__name__ = f"{callback.__name__}_handler"
                
            def __call__(self, event):
                self.callback(event)
        
        # Register handlers
        event_bus.subscribe(EventType.TOKEN_TRADE, EventHandler(self.handle_token_trade))
        event_bus.subscribe(EventType.TOKEN_CREATED, EventHandler(self.handle_token_created))
        logger.info(f"Price provider {self.name} registered for token events") 