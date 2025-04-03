#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump Detection Trading Strategy

This module provides a trading strategy that uses pump and dump detection
to generate trading signals.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.core.events import EventType, Event
from src.core.trading.interfaces import TradingSignal, SignalType, Strategy
from src.core.features.manager import FeatureManager
from src.core.ml.models.pump_predictor import PumpPredictorModel

logger = logging.getLogger(__name__)


class PumpDetectionStrategy(Strategy):
    """
    Trading strategy based on pump and dump detection.
    
    This strategy uses the PumpPredictorModel to detect pump and dump patterns
    and generates trading signals accordingly.
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        pump_predictor: PumpPredictorModel,
        confidence_threshold: float = 0.75,
        cooldown_period: int = 120,  # 2 hours in minutes
        enable_shorts: bool = True,
        enable_pump_entry: bool = False,
        max_tokens: int = 10
    ):
        """
        Initialize the pump detection trading strategy.
        
        Args:
            feature_manager: Feature manager for retrieving features
            pump_predictor: Model for predicting pump and dump events
            confidence_threshold: Minimum confidence to generate a signal
            cooldown_period: Minutes to wait between signals for same token
            enable_shorts: Whether to generate SHORT signals for dumps
            enable_pump_entry: Whether to enter during pump phase (risky)
            max_tokens: Maximum number of tokens to track simultaneously
        """
        self._feature_manager = feature_manager
        self._pump_predictor = pump_predictor
        self._confidence_threshold = confidence_threshold
        self._cooldown_period = cooldown_period
        self._enable_shorts = enable_shorts
        self._enable_pump_entry = enable_pump_entry
        self._max_tokens = max_tokens
        
        # Track token states and signal history
        self._token_states: Dict[str, Dict[str, Any]] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        
        # Track active positions to avoid duplicate signals
        self._active_positions: Dict[str, str] = {}  # token_id -> position_id
        
        # Track tokens by interest level
        self._monitored_tokens: List[str] = []
        
        logger.info(f"Initialized PumpDetectionStrategy (confidence_threshold={confidence_threshold}, enable_shorts={enable_shorts})")
    
    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name
        """
        return "pump_detection"
    
    @property
    def description(self) -> str:
        """
        Get the description of the strategy.
        
        Returns:
            Strategy description
        """
        return "Generates trading signals based on pump and dump pattern detection"
    
    def process_token_trade(self, token_id: str, trade_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Process a trade event for a token.
        
        Args:
            token_id: ID of the token
            trade_data: Trade data
            
        Returns:
            Trading signal or None
        """
        if token_id not in self._monitored_tokens:
            # Add to monitored tokens if we're under the limit
            if len(self._monitored_tokens) < self._max_tokens:
                self._monitored_tokens.append(token_id)
                logger.info(f"Added {token_id} to monitored tokens (total: {len(self._monitored_tokens)})")
            else:
                # Skip tokens that aren't being monitored
                return None
        
        # Check if we're in a cooldown period for this token
        current_time = datetime.now()
        last_signal_time = self._last_signal_time.get(token_id)
        
        if last_signal_time and (current_time - last_signal_time).total_seconds() < self._cooldown_period * 60:
            # Still in cooldown period, skip analysis
            return None
        
        # Get features and make a prediction
        features = self._get_token_features(token_id)
        prediction = self._pump_predictor.predict(features)
        
        # Store token state
        previous_state = self._token_states.get(token_id, {})
        self._token_states[token_id] = prediction
        
        # Check if we should generate a signal
        signal = self._check_for_signal(token_id, prediction, previous_state, trade_data)
        
        if signal:
            # Update last signal time
            self._last_signal_time[token_id] = current_time
            
            if signal.signal_type == SignalType.OPEN:
                # Track active position
                self._active_positions[token_id] = signal.data.get('position_id', '')
            elif signal.signal_type == SignalType.CLOSE:
                # Remove from active positions
                if token_id in self._active_positions:
                    del self._active_positions[token_id]
        
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
            # Process trade event
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
            'timestamp': datetime.now().timestamp()
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
        previous_state: Dict[str, Any],
        trade_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Check if we should generate a trading signal based on prediction.
        
        Args:
            token_id: ID of the token
            prediction: Current prediction
            previous_state: Previous prediction
            trade_data: Latest trade data
            
        Returns:
            Trading signal or None
        """
        # Extract state information
        state = prediction['class_label']
        phase = prediction['phase']
        confidence = prediction['confidence']
        
        # Only generate signals if confidence is high enough
        if confidence < self._confidence_threshold:
            return None
        
        # Check if we already have an active position for this token
        has_active_position = token_id in self._active_positions
        
        # Get the current price
        current_price = trade_data.get('price', 0.0)
        if current_price <= 0:
            return None
        
        # Generate signal based on state and phase
        if state == "PUMP" and phase == 1 and self._enable_pump_entry and not has_active_position:
            # Early pump detection - LONG signal
            return TradingSignal(
                token_id=token_id,
                signal_type=SignalType.OPEN,
                direction=1,  # LONG
                confidence=confidence,
                data={
                    "strategy": self.name,
                    "reason": "Early pump detected",
                    "phase": phase,
                    "state": state,
                    "price": current_price,
                    "timestamp": datetime.now().timestamp(),
                    "position_id": f"pump_{token_id}_{int(time.time())}"
                }
            )
        
        elif state == "PEAK" and phase == 3 and has_active_position:
            # Peak detected - CLOSE signal for pump trades
            position_id = self._active_positions[token_id]
            
            return TradingSignal(
                token_id=token_id,
                signal_type=SignalType.CLOSE,
                direction=0,  # Close
                confidence=confidence,
                data={
                    "strategy": self.name,
                    "reason": "Distribution phase detected",
                    "phase": phase,
                    "state": state,
                    "price": current_price,
                    "timestamp": datetime.now().timestamp(),
                    "position_id": position_id
                }
            )
        
        elif state == "DUMP" and phase == 4:
            if has_active_position:
                # Dump detected - CLOSE signal (emergency exit)
                position_id = self._active_positions[token_id]
                
                return TradingSignal(
                    token_id=token_id,
                    signal_type=SignalType.CLOSE,
                    direction=0,  # Close
                    confidence=confidence * 1.2,  # Higher confidence for exit signals
                    data={
                        "strategy": self.name,
                        "reason": "Dump detected - emergency exit",
                        "phase": phase,
                        "state": state,
                        "price": current_price,
                        "timestamp": datetime.now().timestamp(),
                        "position_id": position_id
                    }
                )
            elif self._enable_shorts and not has_active_position:
                # Dump detected - SHORT signal
                return TradingSignal(
                    token_id=token_id,
                    signal_type=SignalType.OPEN,
                    direction=-1,  # SHORT
                    confidence=confidence,
                    data={
                        "strategy": self.name,
                        "reason": "Dump detected",
                        "phase": phase,
                        "state": state,
                        "price": current_price,
                        "timestamp": datetime.now().timestamp(),
                        "position_id": f"dump_{token_id}_{int(time.time())}"
                    }
                )
        
        elif state == "NORMAL" and has_active_position:
            # Check if we should close an existing position
            # Only close if we were previously in an abnormal state
            prev_state = previous_state.get('class_label', 'NORMAL')
            
            if prev_state != 'NORMAL' and confidence > 0.8:
                position_id = self._active_positions[token_id]
                
                return TradingSignal(
                    token_id=token_id,
                    signal_type=SignalType.CLOSE,
                    direction=0,  # Close
                    confidence=confidence,
                    data={
                        "strategy": self.name,
                        "reason": "Returned to normal trading",
                        "phase": phase,
                        "state": state,
                        "price": current_price,
                        "timestamp": datetime.now().timestamp(),
                        "position_id": position_id
                    }
                )
        
        return None 