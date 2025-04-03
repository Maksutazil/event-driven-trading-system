#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Early Pump Trading Strategy

This module provides a trading strategy optimized for detecting and trading
pump events in newly created tokens with minimal trading history.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from src.core.events import EventType, Event
from src.core.trading.interfaces import TradingSignal, SignalType, Strategy
from src.core.features.manager import FeatureManager
from src.core.ml.models.early_pump_predictor import EarlyPumpPredictor

logger = logging.getLogger(__name__)


class EarlyPumpStrategy(Strategy):
    """
    Trading strategy for early-stage pump detection.
    
    This strategy is designed to quickly identify potential pump events in
    newly created tokens with minimal trading history.
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        pump_predictor: EarlyPumpPredictor,
        confidence_threshold: float = 0.6,
        cooldown_seconds: int = 60,  # Short cooldown for rapid response
        max_active_tokens: int = 10
    ):
        """
        Initialize the early pump trading strategy.
        
        Args:
            feature_manager: Feature manager for retrieving features
            pump_predictor: Model for predicting early pump events
            confidence_threshold: Minimum confidence to generate a signal
            cooldown_seconds: Seconds to wait between signals for same token
            max_active_tokens: Maximum number of tokens to monitor simultaneously
        """
        self._feature_manager = feature_manager
        self._pump_predictor = pump_predictor
        self._confidence_threshold = confidence_threshold
        self._cooldown_seconds = cooldown_seconds
        self._max_active_tokens = max_active_tokens
        
        # Track token states and last signal time
        self._last_signal_time: Dict[str, float] = {}
        
        # Track active positions to avoid duplicate signals
        self._active_positions: Dict[str, str] = {}  # token_id -> position_id
        
        # Track monitored tokens with their first seen time
        self._monitored_tokens: Dict[str, float] = {}  # token_id -> first_seen_time
        
        logger.info(f"Initialized EarlyPumpStrategy (confidence_threshold={confidence_threshold})")
    
    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        return "early_pump_strategy"
    
    @property
    def description(self) -> str:
        """
        Get the description of the strategy.
        
        Returns:
            Strategy description
        """
        return "Generates trading signals for early pump detection in new tokens"
    
    def process_token_trade(self, token_id: str, trade_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Process a token trade event and generate signals for early pump detection.
        
        Args:
            token_id: ID of the token
            trade_data: Trade data
            
        Returns:
            Trading signal or None
        """
        current_time = time.time()
        
        # Track the token if not already monitored
        if token_id not in self._monitored_tokens:
            # Check if we're at capacity
            if len(self._monitored_tokens) >= self._max_active_tokens:
                # Find oldest token to replace
                oldest_token = min(self._monitored_tokens.items(), key=lambda x: x[1])[0]
                if oldest_token not in self._active_positions:
                    del self._monitored_tokens[oldest_token]
                    logger.debug(f"Replaced oldest monitored token {oldest_token} with {token_id}")
                
            # Add new token if under capacity
            if len(self._monitored_tokens) < self._max_active_tokens:
                self._monitored_tokens[token_id] = current_time
                logger.info(f"Now monitoring token {token_id} for early pump signals")
        
        # Skip processing if not monitored
        if token_id not in self._monitored_tokens:
            return None
        
        # Check if we're in cooldown period
        last_signal_time = self._last_signal_time.get(token_id, 0)
        if current_time - last_signal_time < self._cooldown_seconds:
            return None
        
        # Check if we already have an active position
        if token_id in self._active_positions:
            return None
        
        # Get features and make prediction
        features = self._get_token_features(token_id)
        prediction = self._pump_predictor.predict(features)
        
        # Generate signal based on prediction
        signal = self._check_for_signal(token_id, prediction, trade_data)
        
        if signal:
            # Update last signal time
            self._last_signal_time[token_id] = current_time
            
            if signal.signal_type == SignalType.OPEN:
                # Track active position
                self._active_positions[token_id] = signal.data.get('position_id', '')
                logger.info(f"Generated OPEN signal for {token_id} with confidence {signal.confidence:.2f}")
            elif signal.signal_type == SignalType.CLOSE:
                # Remove from active positions
                if token_id in self._active_positions:
                    del self._active_positions[token_id]
                logger.info(f"Generated CLOSE signal for {token_id}")
        
        return signal
    
    def process_event(self, event: Event) -> Optional[TradingSignal]:
        """
        Process an event.
        
        Args:
            event: The event to process
            
        Returns:
            Trading signal or None
        """
        if event.event_type == EventType.TOKEN_TRADE:
            # Process token trade event
            token_id = event.data.get('token_id')
            if token_id:
                return self.process_token_trade(token_id, event.data)
        
        return None
    
    def _get_token_features(self, token_id: str) -> Dict[str, Any]:
        """
        Get features for token analysis.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary of feature values
        """
        # Context for feature computation
        context = {
            'token_id': token_id,
            'timestamp': time.time()
        }
        
        # Get all required features
        features = {}
        for feature_name in self._pump_predictor.get_required_features():
            features[feature_name] = self._feature_manager.compute_feature(token_id, feature_name, context)
        
        return features
    
    def _check_for_signal(
        self,
        token_id: str,
        prediction: Dict[str, Any],
        trade_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Check if we should generate a trading signal based on the prediction.
        
        Args:
            token_id: ID of the token
            prediction: Current prediction
            trade_data: Latest trade data
            
        Returns:
            Trading signal or None
        """
        # Extract prediction details
        class_label = prediction.get('class_label', 'NORMAL')
        confidence = prediction.get('confidence', 0.0)
        evidence = prediction.get('evidence', {})
        
        # Only generate signals if confidence is high enough
        if confidence < self._confidence_threshold:
            return None
        
        # Get current price
        current_price = trade_data.get('price', 0.0)
        if current_price <= 0:
            return None
        
        # Generate signal based on prediction class
        if class_label == "STRONG_PUMP_SIGNAL":
            # Strong pump signal - high confidence LONG signal
            position_id = f"early_pump_{token_id}_{int(time.time())}"
            
            return TradingSignal(
                token_id=token_id,
                signal_type=SignalType.OPEN,
                direction=1,  # LONG
                confidence=confidence,
                data={
                    "strategy": self.name,
                    "reason": "Strong early pump signal detected",
                    "class_label": class_label,
                    "price": current_price,
                    "price_change": evidence.get('price_change', 0.0),
                    "trade_frequency": evidence.get('trade_frequency', 0.0),
                    "buyer_dominance": evidence.get('buyer_dominance', 0.0),
                    "timestamp": time.time(),
                    "position_id": position_id
                }
            )
            
        elif class_label == "EARLY_PUMP":
            # Potential early pump - moderate confidence LONG signal
            position_id = f"early_pump_{token_id}_{int(time.time())}"
            
            # Only generate signal if confidence is sufficiently high
            if confidence >= self._confidence_threshold + 0.05:  # Add a small buffer
                return TradingSignal(
                    token_id=token_id,
                    signal_type=SignalType.OPEN,
                    direction=1,  # LONG
                    confidence=confidence,
                    data={
                        "strategy": self.name,
                        "reason": "Early pump pattern detected",
                        "class_label": class_label,
                        "price": current_price,
                        "price_change": evidence.get('price_change', 0.0),
                        "trade_frequency": evidence.get('trade_frequency', 0.0),
                        "buyer_dominance": evidence.get('buyer_dominance', 0.0),
                        "timestamp": time.time(),
                        "position_id": position_id
                    }
                )
        
        return None 