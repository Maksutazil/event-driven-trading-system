#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Engine Module

This module provides the DefaultTradingEngine implementation for coordinating
trading activities, processing data, and making trading decisions.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
import uuid
import json
import threading
import numpy as np

from src.core.events import EventBus, Event, EventType, EventDispatcher
from src.core.trading.interfaces import (
    TradingEngine, PositionManager, TradeExecutor, RiskManager
)
from src.core.features.feature_system import FeatureSystem
from src.core.features.interfaces import FeatureConsumer
from src.core.ml import ModelManager
from src.core.events.base import EventHandlerWrapper

logger = logging.getLogger(__name__)


class DefaultTradingEngine(TradingEngine, FeatureConsumer):
    """
    Default implementation of the TradingEngine interface.
    
    This implementation coordinates the trading process, including strategy evaluation,
    signal generation, and trade execution.
    
    Implements FeatureConsumer to receive feature updates directly.
    """
    
    def __init__(self,
                event_bus: EventBus,
                feature_system: FeatureSystem,
                position_manager: PositionManager,
                trade_executor: TradeExecutor,
                risk_manager: RiskManager,
                signal_threshold: float = 0.7,
                signal_expiry_seconds: float = 60.0,
                cooldown_seconds: float = 3600.0,
                max_tokens_per_timepoint: int = 3,
                model_manager: Optional[ModelManager] = None,
                model_prediction_weight: float = 0.5,
                signal_generator = None):
        """
        Initialize the DefaultTradingEngine.
        
        Args:
            event_bus: Event bus for publishing and subscribing to events
            feature_system: Feature system for computing features
            position_manager: Position manager for tracking positions
            trade_executor: Trade executor for executing trades
            risk_manager: Risk manager for risk calculations
            signal_threshold: Threshold for trade signals (0.0 to 1.0)
            signal_expiry_seconds: Time in seconds before signals expire
            cooldown_seconds: Cooldown period between trades for the same token
            max_tokens_per_timepoint: Maximum tokens to trade per timepoint
            model_manager: Optional model manager for ML predictions
            model_prediction_weight: Weight to give ML predictions in decision making
            signal_generator: Optional signal generator for generating signals
        """
        # Validate inputs
        if signal_threshold < 0 or signal_threshold > 1:
            raise ValueError("signal_threshold must be between 0 and 1")
        if signal_expiry_seconds <= 0:
            raise ValueError("signal_expiry_seconds must be greater than 0")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if max_tokens_per_timepoint <= 0:
            raise ValueError("max_tokens_per_timepoint must be greater than 0")
            
        self.event_bus = event_bus
        self.feature_system = feature_system
        self.position_manager = position_manager
        self.trade_executor = trade_executor
        self.risk_manager = risk_manager
        
        # Engine parameters
        self.signal_threshold = signal_threshold
        self.signal_expiry_seconds = signal_expiry_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_tokens_per_timepoint = max_tokens_per_timepoint
        
        # Internal state
        self.active_tokens: Set[str] = set()
        self.token_metadata: Dict[str, Dict[str, Any]] = {}
        self.token_last_update: Dict[str, datetime] = {}
        self.token_cooldown: Dict[str, float] = {}
        self.pending_signals: Dict[str, Dict[str, Any]] = {}
        self.timepoint_signals: Dict[str, List[Dict[str, Any]]] = {}
        self.engine_state: str = "initialized"
        self.last_processed_time: Optional[datetime] = None
        
        # Feature cache to store the latest feature values
        self.token_features: Dict[str, Dict[str, Any]] = {}
        
        # Strategy parameters (can be updated)
        self.strategy_parameters: Dict[str, Any] = {
            'entry_params': {
                'min_price_change_pct': 0.05,
                'min_volume_usd': 10000,
                'max_market_cap': 1000000000,
                'min_token_age_days': 1,
                'min_holder_count': 100,
                'max_wallet_concentration': 0.3
            },
            'exit_params': {
                'exit_after_days': 3,
                'trailing_stop_pct': 0.15,
                'exit_on_trend_reversal': True
            }
        }
        
        # Add model manager and prediction weight
        self.model_manager = model_manager
        self.model_prediction_weight = model_prediction_weight
        self.model_predictions = {}  # Store recent model predictions
        
        # Store signal generator
        self.signal_generator = signal_generator
        
        # Register as feature consumer
        if self.feature_system:
            self.feature_system.register_consumer(self)
            
        # Register event handlers
        if self.event_bus:
            self._register_event_handlers()
            
        logger.info(f"Initialized DefaultTradingEngine with signal_threshold={signal_threshold}, "
                    f"cooldown_seconds={cooldown_seconds}")
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of features this consumer requires.
        
        Returns:
            List[str]: List of required feature names
        """
        # Return all features needed for trading decisions
        return [
            "current_price",
            "price_change_pct_5m",
            "price_change_pct_15m",
            "ma_5m",
            "ma_15m",
            "volume_5m",
            "volatility_5m",
            "rsi_14",
            "price_momentum_signal",
            "volume_spike_signal"
        ]
    
    def on_feature_update(self, token_id: str, feature_name: str, value: Any) -> None:
        """
        Handle a feature update.
        
        This method is called when a feature value is updated.
        
        Args:
            token_id: ID of the token the feature is for
            feature_name: Name of the feature
            value: New feature value
        """
        # Only care about active tokens
        if token_id not in self.active_tokens:
            return
            
        # Initialize token features dict if needed
        if token_id not in self.token_features:
            self.token_features[token_id] = {}
            
        # Update feature value
        self.token_features[token_id][feature_name] = value
        
        # Update last update time
        self.token_last_update[token_id] = datetime.now()
        
        # Check for trading signals on significant feature updates
        if feature_name.endswith('_signal') and isinstance(value, (int, float)):
            # If it's a strong enough signal, process it
            if abs(value) >= self.signal_threshold:
                logger.info(f"Significant feature update: {feature_name}={value:.4f} for {token_id}")
                
                # Process the timepoint with all available features
                self.process_timepoint_with_features(token_id, self.token_features[token_id])
    
    def process_timepoint_with_features(self, token_id: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a timepoint with pre-computed features.
        
        This is an internal method called from on_feature_update or directly
        when features are already available.
        
        Args:
            token_id: ID of the token
            features: Dictionary of already computed features
            
        Returns:
            List of trading actions
        """
        return self.process_timepoint(token_id, features)
    
    def process_timepoint(self, token_id: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a timepoint for a token and return trading actions.
        
        Args:
            token_id: ID of the token
            features: Dictionary of feature values
            
        Returns:
            List of trading actions (e.g., enter, exit)
        """
        # Check if token is being tracked
        if token_id not in self.active_tokens:
            logger.debug(f"Token {token_id} is not being tracked, ignoring")
            return []
            
        # Check if token is in cooldown
        if self._is_in_cooldown(token_id):
            logger.debug(f"Token {token_id} is in cooldown, ignoring")
            return []
            
        # Get current timestamp
        timepoint = datetime.now()
        
        # Process timepoint data
        signals = []
        
        # Check if we have an open position for this token
        open_positions = self.position_manager.get_open_positions_for_token(token_id)
        
        if open_positions:
            # We have an open position, evaluate exit signals
            for position in open_positions:
                exit_signal = self._evaluate_exit(token_id, position, timepoint, features)
                if exit_signal:
                    signals.append(exit_signal)
                    
                    # Execute exit immediately if configured
                    if not self.signal_expiry_seconds:
                        self._handle_exit_signal(token_id, position['id'], exit_signal, timepoint)
                    else:
                        # Add to pending signals
                        self.pending_signals[token_id] = exit_signal
        else:
            # No open position, evaluate entry signals
            entry_signal = self._evaluate_entry(token_id, timepoint, features)
            if entry_signal:
                signals.append(entry_signal)
                
                # Execute entry immediately if configured
                if not self.signal_expiry_seconds:
                    self._handle_entry_signal(token_id, entry_signal, timepoint)
                else:
                    # Add to pending signals
                    self.pending_signals[token_id] = entry_signal
                    
        # Store signals for this timepoint
        if signals:
            if token_id not in self.timepoint_signals:
                self.timepoint_signals[token_id] = []
            self.timepoint_signals[token_id].extend(signals)
            
        # If model manager is available, incorporate model predictions
        if self.model_manager and token_id in self.model_predictions:
            model_prediction = self.model_predictions[token_id]
            prediction_age = time.time() - model_prediction.get('timestamp', 0)
            
            # Only use predictions that are recent (within signal expiry time)
            if prediction_age < self.signal_expiry_seconds:
                # Get the prediction value
                prediction_value = model_prediction.get('prediction')
                model_id = model_prediction.get('model_id')
                
                logger.info(f"Incorporating model {model_id} prediction {prediction_value} for {token_id}")
                
                # Apply model prediction to the signal strength
                if 'signals' in results:
                    for signal in results['signals']:
                        # For classification models (predicting direction)
                        if isinstance(prediction_value, (int, bool)) or (
                            isinstance(prediction_value, (float, np.number)) and (prediction_value == 0 or prediction_value == 1)
                        ):
                            # If model predicts same direction, boost signal
                            signal_direction = 1 if signal['type'] == 'entry' else -1
                            model_direction = 1 if prediction_value > 0.5 else -1
                            
                            if signal_direction == model_direction:
                                signal['strength'] = min(1.0, signal['strength'] * (1 + self.model_prediction_weight))
                            else:
                                signal['strength'] = max(0.0, signal['strength'] * (1 - self.model_prediction_weight))
                                
                            logger.debug(f"Adjusted signal strength to {signal['strength']} based on model prediction")
                        
                        # For regression models (predicting a value like price movement)
                        elif isinstance(prediction_value, (float, np.number)):
                            # Scale the prediction to a factor between 0.5 and 1.5
                            prediction_factor = 1.0 + (prediction_value * self.model_prediction_weight)
                            signal['strength'] = min(1.0, signal['strength'] * prediction_factor)
                            logger.debug(f"Adjusted signal strength to {signal['strength']} based on model prediction value {prediction_value}")
                
        return signals
    
    def get_active_tokens(self) -> Set[str]:
        """
        Get the set of tokens currently being tracked by the trading engine.
        
        Returns:
            Set of token IDs
        """
        return set(self.active_tokens)
    
    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add multiple tokens to be tracked by the trading engine.
        
        Args:
            tokens: List of token IDs to add
        """
        for token_id in tokens:
            self.add_token(token_id)
    
    def add_token(self, token_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a token to be tracked by the trading engine.
        
        Args:
            token_id: ID of the token to add
            metadata: Optional metadata for the token
            
        Returns:
            True if the token was added successfully, False otherwise
        """
        if token_id in self.active_tokens:
            logger.debug(f"Token {token_id} is already being tracked")
            return False
            
        self.active_tokens.add(token_id)
        self.token_metadata[token_id] = metadata or {}
        self.token_last_update[token_id] = datetime.now()
        self.token_features[token_id] = {}
        
        logger.info(f"Added token {token_id} to tracking")
        return True
    
    def remove_tokens(self, tokens: List[str]) -> None:
        """
        Remove multiple tokens from tracking by the trading engine.
        
        Args:
            tokens: List of token IDs to remove
        """
        for token_id in tokens:
            self.remove_token(token_id)
    
    def remove_token(self, token_id: str) -> bool:
        """
        Remove a token from tracking by the trading engine.
        
        Args:
            token_id: ID of the token to remove
            
        Returns:
            True if the token was removed, False otherwise
        """
        if token_id not in self.active_tokens:
            logger.debug(f"Token {token_id} is not being tracked")
            return False
            
        self.active_tokens.remove(token_id)
        
        # Clean up other token data
        if token_id in self.token_metadata:
            del self.token_metadata[token_id]
        if token_id in self.token_last_update:
            del self.token_last_update[token_id]
        if token_id in self.token_cooldown:
            del self.token_cooldown[token_id]
        if token_id in self.pending_signals:
            del self.pending_signals[token_id]
        if token_id in self.timepoint_signals:
            del self.timepoint_signals[token_id]
        if token_id in self.token_features:
            del self.token_features[token_id]
            
        logger.info(f"Removed token {token_id} from tracking")
        return True
    
    def set_parameter(self, parameter_name: str, value: Any) -> None:
        """
        Set a trading engine parameter.
        
        Args:
            parameter_name: Name of the parameter
            value: New value for the parameter
        """
        if parameter_name == 'signal_threshold':
            if 0 <= value <= 1:
                self.signal_threshold = value
            else:
                raise ValueError("signal_threshold must be between 0 and 1")
                
        elif parameter_name == 'signal_expiry_seconds':
            if value > 0:
                self.signal_expiry_seconds = value
            else:
                raise ValueError("signal_expiry_seconds must be greater than 0")
                
        elif parameter_name == 'cooldown_seconds':
            if value >= 0:
                self.cooldown_seconds = value
            else:
                raise ValueError("cooldown_seconds must be non-negative")
                
        elif parameter_name == 'max_tokens_per_timepoint':
            if value > 0:
                self.max_tokens_per_timepoint = value
            else:
                raise ValueError("max_tokens_per_timepoint must be greater than 0")
                
        elif parameter_name == 'entry_params':
            if isinstance(value, dict):
                self.strategy_parameters['entry_params'].update(value)
            else:
                raise ValueError("entry_params must be a dictionary")
                
        elif parameter_name == 'exit_params':
            if isinstance(value, dict):
                self.strategy_parameters['exit_params'].update(value)
            else:
                raise ValueError("exit_params must be a dictionary")
                
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")
            
        logger.info(f"Updated parameter {parameter_name} to {value}")
    
    def get_parameter(self, parameter_name: str) -> Any:
        """
        Get a trading engine parameter.
        
        Args:
            parameter_name: Name of the parameter
            
        Returns:
            Current value of the parameter
        """
        if parameter_name == 'signal_threshold':
            return self.signal_threshold
            
        elif parameter_name == 'signal_expiry_seconds':
            return self.signal_expiry_seconds
            
        elif parameter_name == 'cooldown_seconds':
            return self.cooldown_seconds
            
        elif parameter_name == 'max_tokens_per_timepoint':
            return self.max_tokens_per_timepoint
            
        elif parameter_name == 'entry_params':
            return self.strategy_parameters['entry_params'].copy()
            
        elif parameter_name == 'exit_params':
            return self.strategy_parameters['exit_params'].copy()
            
        elif parameter_name == 'engine_state':
            return self.engine_state
            
        elif parameter_name == 'last_processed_time':
            return self.last_processed_time
            
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters for the trading engine.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            "signal_threshold": self.signal_threshold,
            "signal_expiry_seconds": self.signal_expiry_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "max_tokens_per_timepoint": self.max_tokens_per_timepoint,
            "model_prediction_weight": self.model_prediction_weight
        }
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set multiple parameters for the trading engine.
        
        Args:
            params: Dictionary of parameter values to update
        """
        if "signal_threshold" in params:
            threshold = params["signal_threshold"]
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError("signal_threshold must be between 0.0 and 1.0")
            self.signal_threshold = threshold
            
        if "signal_expiry_seconds" in params:
            expiry = params["signal_expiry_seconds"]
            if expiry <= 0:
                raise ValueError("signal_expiry_seconds must be greater than 0")
            self.signal_expiry_seconds = expiry
            
        if "cooldown_seconds" in params:
            cooldown = params["cooldown_seconds"]
            if cooldown < 0:
                raise ValueError("cooldown_seconds must be non-negative")
            self.cooldown_seconds = cooldown
            
        if "max_tokens_per_timepoint" in params:
            max_tokens = params["max_tokens_per_timepoint"]
            if max_tokens <= 0:
                raise ValueError("max_tokens_per_timepoint must be greater than 0")
            self.max_tokens_per_timepoint = max_tokens
            
        if "model_prediction_weight" in params:
            self.set_model_prediction_weight(params["model_prediction_weight"])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading engine.
        
        Returns:
            Dictionary with status information
        """
        # Get current state of positions
        open_positions = self.position_manager.get_open_positions()
        performance_metrics = self.position_manager.get_performance_metrics()
        
        # Calculate additional metrics
        execution_stats = self.trade_executor.get_execution_statistics()
        
        return {
            'engine_state': self.engine_state,
            'last_processed_time': self.last_processed_time,
            'active_tokens_count': len(self.active_tokens),
            'open_positions_count': len(open_positions),
            'pending_signals_count': len(self.pending_signals),
            'performance_metrics': performance_metrics,
            'execution_stats': execution_stats
        }
    
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens based on active tokens and other constraints.
        
        Args:
            tokens: List of token IDs to filter
            
        Returns:
            Filtered list of token IDs
        """
        # If active_tokens is empty, process all tokens
        if not self.active_tokens:
            return tokens
            
        # Otherwise, only process active tokens
        return [token for token in tokens if token in self.active_tokens]
    
    def update_token_metadata(self, token_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a tracked token.
        
        This method is called when new token metadata is received, typically
        from a TOKEN_NEW event from the socket connection.
        
        Args:
            token_id: ID of the token to update
            metadata: Metadata dictionary for the token
            
        Returns:
            True if the token metadata was updated successfully, False otherwise
        """
        if token_id not in self.active_tokens:
            logger.debug(f"Token {token_id} is not being tracked, cannot update metadata")
            return False
            
        if token_id not in self.token_metadata:
            self.token_metadata[token_id] = {}
            
        # Update metadata
        self.token_metadata[token_id].update(metadata)
        
        # Update last update time
        self.token_last_update[token_id] = datetime.now()
        
        logger.debug(f"Updated metadata for token {token_id}")
        return True
    
    def update_token_trade(self, token_id: str, trade_data: Dict[str, Any]) -> bool:
        """
        Update trade data for a tracked token.
        
        This method is called when new trade data is received, typically
        from a TOKEN_TRADE event from the socket connection.
        
        Args:
            token_id: ID of the token to update
            trade_data: Trade data dictionary for the token
            
        Returns:
            True if the token trade data was processed successfully, False otherwise
        """
        start_time = time.time()  # Track execution time
        
        if token_id not in self.active_tokens:
            logger.debug(f"Token {token_id} is not being tracked, adding to active tokens")
            self.add_token(token_id, metadata=trade_data)
            
        # Update token's last update time
        self.token_last_update[token_id] = datetime.now()
        
        # Store trade data in token metadata if not already present
        if token_id not in self.token_metadata:
            self.token_metadata[token_id] = {}
            
        if 'trades' not in self.token_metadata[token_id]:
            self.token_metadata[token_id]['trades'] = []
            
        # Add to trade history (limiting to last 100 trades to avoid memory issues)
        trades = self.token_metadata[token_id]['trades']
        trades.append(trade_data)
        self.token_metadata[token_id]['trades'] = trades[-100:]
        
        # Update latest price in metadata
        if 'price' in trade_data and trade_data['price'] > 0:
            self.token_metadata[token_id]['latest_price'] = trade_data['price']
            
            # Also update last price change time and price
            current_price = trade_data['price']
            previous_price = self.token_metadata[token_id].get('previous_price')
            
            if previous_price is not None and previous_price > 0:
                price_change_pct = ((current_price - previous_price) / previous_price) * 100
                self.token_metadata[token_id]['price_change_pct'] = price_change_pct
                
                # Log significant price changes
                if abs(price_change_pct) >= 5.0:  # 5% or more change
                    logger.info(f"Significant price change for {token_id}: {price_change_pct:.2f}% (from {previous_price} to {current_price})")
            
            self.token_metadata[token_id]['previous_price'] = current_price
            self.token_metadata[token_id]['last_price_update_time'] = datetime.now()
        
        # Process the token with this new trade data
        processed = False
        try:
            # Extract timestamp from trade data or use current time
            timestamp = trade_data.get('timestamp', time.time())
            if isinstance(timestamp, (int, float)) and timestamp > 1000000000000:  # If in milliseconds
                timestamp = timestamp / 1000  # Convert to seconds
                
            timepoint = datetime.fromtimestamp(timestamp)
            
            # Get features for this token
            features = self._get_token_features(token_id, timepoint, self.token_metadata[token_id])
            
            if features:
                # Update token metadata with the computed features
                if 'features' not in self.token_metadata[token_id]:
                    self.token_metadata[token_id]['features'] = {}
                
                self.token_metadata[token_id]['features'].update(features)
                
                # Process timepoint to generate trading signals
                signals = self.process_timepoint(token_id, features)
                processed = True
                
                if signals:
                    logger.info(f"Generated {len(signals)} trading signals for {token_id}")
                    
                    # Execute signals if they meet threshold
                    for signal in signals:
                        signal_strength = signal.get('strength', 0)
                        if signal_strength >= self.signal_threshold:
                            signal_type = signal.get('type')
                            
                            if signal_type == 'entry':
                                self._handle_entry_signal(token_id, signal, timepoint)
                            elif signal_type == 'exit':
                                position_id = signal.get('position_id')
                                if position_id:
                                    self._handle_exit_signal(token_id, position_id, signal, timepoint)
                                    
                # Log execution time for performance monitoring
                elapsed = time.time() - start_time
                if elapsed > 0.1:  # Log if processing took more than 100ms
                    logger.debug(f"Token {token_id} processing took {elapsed:.3f} seconds")
                    
        except Exception as e:
            logger.error(f"Error processing trade data for token {token_id}: {str(e)}", exc_info=True)
            # Record but don't fail completely
            self.token_metadata[token_id]['last_error'] = str(e)
            self.token_metadata[token_id]['last_error_time'] = datetime.now()
        
        # Even if processing fails, we did update the trade history
        return True
    
    def _get_token_features(self, token_id: str, timepoint: datetime,
                          token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute features for a token.
        
        Args:
            token_id: ID of the token
            timepoint: Timepoint being processed
            token_data: Raw data for the token
            
        Returns:
            Dictionary of computed features or None if computation fails
        """
        try:
            # Get context for feature computation
            context = {
                'token_id': token_id,
                'timepoint': timepoint,
                **token_data
            }
            
            # Compute features using feature system
            features = self.feature_system.compute_features(context)
            return features
            
        except Exception as e:
            logger.error(f"Error computing features for token {token_id}: {str(e)}")
            return None
    
    def _evaluate_entry(self, token_id: str, timepoint: datetime,
                      features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate token for entry signals.
        
        Args:
            token_id: ID of the token
            timepoint: Timepoint being processed
            features: Computed features for the token
            
        Returns:
            Dictionary with entry signal if generated, None otherwise
        """
        # Check if we already have open positions for this token
        open_positions = self.position_manager.get_open_positions_for_token(token_id)
        if open_positions:
            logger.debug(f"Skipping entry evaluation for {token_id}, already have {len(open_positions)} open positions")
            return None
            
        # Get entry parameters
        entry_params = self.strategy_parameters['entry_params']
        
        # Check minimum requirements
        current_price = features.get('price.close', 0)
        volume_usd = features.get('volume.volume_usd_24h', 0)
        market_cap = features.get('token.market_cap', 0)
        token_age_days = features.get('token.age_days', 0)
        holder_count = features.get('token.holder_count', 0)
        wallet_concentration = features.get('token.wallet_concentration', 1.0)
        
        # Skip tokens that don't meet basic criteria
        if volume_usd < entry_params['min_volume_usd']:
            return None
            
        if market_cap > entry_params['max_market_cap']:
            return None
            
        if token_age_days < entry_params['min_token_age_days']:
            return None
            
        if holder_count < entry_params['min_holder_count']:
            return None
            
        if wallet_concentration > entry_params['max_wallet_concentration']:
            return None
            
        # Compute score from predictive features
        price_momentum = features.get('price.momentum_1h', 0)
        volume_change = features.get('volume.change_24h', 0)
        rsi = features.get('indicator.rsi_14', 50)
        macd_hist = features.get('indicator.macd_histogram', 0)
        
        # Simple scoring algorithm (replace with your actual strategy)
        score = 0.0
        
        # Price momentum contributes positively (0 to 0.3)
        score += max(0, min(0.3, price_momentum * 3))
        
        # Volume change contributes positively (0 to 0.3)
        score += max(0, min(0.3, volume_change * 0.5))
        
        # RSI contributes if in a good range (0 to 0.2)
        if 30 <= rsi <= 70:
            # Favor RSI closer to 50 (equilibrium)
            rsi_score = 0.2 * (1 - abs(rsi - 50) / 20)
            score += rsi_score
            
        # MACD histogram contributes positively if positive (0 to 0.2)
        if macd_hist > 0:
            score += min(0.2, macd_hist * 10)
            
        # Check if score exceeds threshold
        if score < self.signal_threshold:
            return None
            
        # Generate entry signal
        signal = {
            'token_id': token_id,
            'timepoint': timepoint,
            'type': 'entry',
            'score': score,
            'price': current_price,
            'features': features,
            'expiry_time': datetime.now().timestamp() + self.signal_expiry_seconds
        }
        
        logger.info(f"Generated entry signal for {token_id} with score {score:.2f} at price {current_price}")
        return signal
    
    def _evaluate_exit(self, token_id: str, position: Dict[str, Any], 
                     timepoint: datetime, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate position for exit signals.
        
        Args:
            token_id: ID of the token
            position: Position details
            timepoint: Timepoint being processed
            features: Computed features for the token
            
        Returns:
            Dictionary with exit signal if generated, None otherwise
        """
        # Get exit parameters
        exit_params = self.strategy_parameters['exit_params']
        
        # Get position details
        position_id = position['id']
        entry_price = position['entry_price']
        current_price = features.get('price.close', position['current_price'])
        entry_time = position['entry_time']
        
        # Check time-based exit
        exit_after_days = exit_params['exit_after_days']
        position_age_days = (timepoint - entry_time).total_seconds() / (24 * 3600)
        
        if position_age_days >= exit_after_days:
            return {
                'token_id': token_id,
                'position_id': position_id,
                'timepoint': timepoint,
                'type': 'exit',
                'reason': 'time_exit',
                'price': current_price,
                'features': features,
                'expiry_time': datetime.now().timestamp() + self.signal_expiry_seconds
            }
            
        # Check for trend reversal exit
        if exit_params['exit_on_trend_reversal']:
            price_momentum = features.get('price.momentum_1h', 0)
            rsi = features.get('indicator.rsi_14', 50)
            macd_hist = features.get('indicator.macd_histogram', 0)
            
            # Check for reversal signals
            if ((price_momentum < -0.02 and macd_hist < 0) or  # Strong downward momentum
                (rsi > 70) or  # Overbought
                (position_age_days > 1 and price_momentum < -0.01)):  # Downward momentum after holding
                
                return {
                    'token_id': token_id,
                    'position_id': position_id,
                    'timepoint': timepoint,
                    'type': 'exit',
                    'reason': 'trend_reversal',
                    'price': current_price,
                    'features': features,
                    'expiry_time': datetime.now().timestamp() + self.signal_expiry_seconds
                }
                
        # Trailing stop loss
        # Update high water mark if price is higher
        high_water_mark = position.get('high_water_mark', entry_price)
        if current_price > high_water_mark:
            position['high_water_mark'] = current_price
            high_water_mark = current_price
            
        # Check trailing stop
        trailing_stop_pct = exit_params['trailing_stop_pct']
        trailing_stop_price = high_water_mark * (1 - trailing_stop_pct)
        
        if current_price <= trailing_stop_price:
            return {
                'token_id': token_id,
                'position_id': position_id,
                'timepoint': timepoint,
                'type': 'exit',
                'reason': 'trailing_stop',
                'price': current_price,
                'features': features,
                'expiry_time': datetime.now().timestamp() + self.signal_expiry_seconds
            }
            
        # No exit signal
        return None
    
    def _handle_entry_signal(self, token_id: str, signal: Dict[str, Any], 
                           timepoint: datetime) -> None:
        """
        Handle a generated entry signal.
        
        Args:
            token_id: ID of the token
            signal: Entry signal details
            timepoint: Timepoint being processed
        """
        # Generate signal ID
        signal_id = f"entry_{token_id}_{timepoint.timestamp()}"
        
        # Store in pending signals
        self.pending_signals[signal_id] = signal
        
        # Add to timepoint signals
        if timepoint.isoformat() not in self.timepoint_signals:
            self.timepoint_signals[timepoint.isoformat()] = []
            
        self.timepoint_signals[timepoint.isoformat()].append(signal)
        
        # Publish signal event
        self._publish_signal_event(signal_id, signal)
    
    def _handle_exit_signal(self, token_id: str, position_id: str, 
                          signal: Dict[str, Any], timepoint: datetime) -> None:
        """
        Handle a generated exit signal.
        
        Args:
            token_id: ID of the token
            position_id: ID of the position
            signal: Exit signal details
            timepoint: Timepoint being processed
        """
        # Generate signal ID
        signal_id = f"exit_{position_id}_{timepoint.timestamp()}"
        
        # Store in pending signals
        self.pending_signals[signal_id] = signal
        
        # Add to timepoint signals
        if timepoint.isoformat() not in self.timepoint_signals:
            self.timepoint_signals[timepoint.isoformat()] = []
            
        self.timepoint_signals[timepoint.isoformat()].append(signal)
        
        # Publish signal event
        self._publish_signal_event(signal_id, signal)
    
    def _execute_pending_signals(self, timepoint: datetime) -> List[Dict[str, Any]]:
        """
        Execute pending signals.
        
        Args:
            timepoint: Timepoint being processed
            
        Returns:
            List of executed signals
        """
        # Remove expired signals
        current_time = datetime.now().timestamp()
        expired_signals = [
            signal_id for signal_id, signal in self.pending_signals.items()
            if signal.get('expiry_time', 0) < current_time
        ]
        
        for signal_id in expired_signals:
            del self.pending_signals[signal_id]
            
        if expired_signals:
            logger.debug(f"Removed {len(expired_signals)} expired signals")
            
        # Get available capital
        available_capital = self.position_manager.get_available_capital()
        
        # Get max positions allowed
        max_positions = self.risk_manager.get_max_positions(available_capital)
        current_positions = len(self.position_manager.get_open_positions())
        positions_available = max(0, max_positions - current_positions)
        
        # If no positions available, skip execution
        if positions_available <= 0:
            logger.info("No positions available, skipping signal execution")
            return []
            
        # Sort entry signals by score
        entry_signals = [
            (signal_id, signal) for signal_id, signal in self.pending_signals.items()
            if signal['type'] == 'entry'
        ]
        
        entry_signals.sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Sort exit signals by priority (exits always execute first)
        exit_signals = [
            (signal_id, signal) for signal_id, signal in self.pending_signals.items()
            if signal['type'] == 'exit'
        ]
        
        # Execute exit signals first
        executed_signals = []
        for signal_id, signal in exit_signals:
            # Execute exit
            success = self.trade_executor.execute_exit(
                position_id=signal['position_id'],
                signal_price=signal['price'],
                reason=signal['reason']
            )
            
            if success:
                executed_signals.append(signal)
                
                # Update token last trade time (for cooldown)
                self.token_last_update[signal['token_id']] = datetime.now()
                
                # Remove from pending signals
                if signal_id in self.pending_signals:
                    del self.pending_signals[signal_id]
        
        # Execute entry signals (limited by max_tokens_per_timepoint)
        entries_to_execute = min(positions_available, self.max_tokens_per_timepoint)
        entries_executed = 0
        
        for signal_id, signal in entry_signals:
            if entries_executed >= entries_to_execute:
                break
                
            # Execute entry
            position_id = self.trade_executor.execute_entry(
                token_id=signal['token_id'],
                signal_price=signal['price'],
                risk_score=signal['score'],
                metadata={'signal': signal}
            )
            
            if position_id:
                executed_signals.append(signal)
                entries_executed += 1
                
                # Update token last trade time (for cooldown)
                self.token_last_update[signal['token_id']] = datetime.now()
                
                # Remove from pending signals
                if signal_id in self.pending_signals:
                    del self.pending_signals[signal_id]
        
        return executed_signals
    
    def _is_in_cooldown(self, token_id: str) -> bool:
        """
        Check if a token is in cooldown period.
        
        Args:
            token_id: ID of the token
            
        Returns:
            True if in cooldown, False otherwise
        """
        if token_id not in self.token_last_update:
            return False
            
        last_trade_time = self.token_last_update[token_id]
        cooldown_end_time = last_trade_time.timestamp() + self.cooldown_seconds
        current_time = datetime.now().timestamp()
        
        return current_time < cooldown_end_time
    
    def _publish_signal_event(self, signal_id: str, signal: Dict[str, Any]) -> None:
        """
        Publish a trade signal event.
        
        Args:
            signal_id: ID of the signal
            signal: Signal details
        """
        if not self.event_bus:
            return
            
        event_data = {
            'signal_id': signal_id,
            'token_id': signal['token_id'],
            'type': signal['type'],
            'timestamp': datetime.now(),
            **signal
        }
        
        event = Event(
            event_type=EventType.TRADE_SIGNAL,
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _register_event_handlers(self) -> None:
        """
        Register event handlers.
        """
        # Register for position update events
        self.event_bus.subscribe(
            EventType.POSITION_UPDATED, 
            EventHandlerWrapper(self._handle_position_updated_event)
        )
        
        # Register for trade executed events (replaces both entry and exit handlers)
        self.event_bus.subscribe(
            EventType.TRADE_EXECUTED, 
            EventHandlerWrapper(self._handle_trade_executed_event)
        )
        
        # Register for token update events
        self.event_bus.subscribe(
            EventType.TOKEN_UPDATED, 
            EventHandlerWrapper(self._handle_token_updated_event)
        )
        
        # Register for token trade events
        self.event_bus.subscribe(
            EventType.TOKEN_TRADE, 
            EventHandlerWrapper(self._handle_token_trade_event)
        )
        
        # Register for model prediction events
        self.event_bus.subscribe(
            EventType.MODEL_PREDICTION, 
            EventHandlerWrapper(self._handle_model_prediction_event)
        )
    
    def _handle_position_updated_event(self, event: Event) -> None:
        """
        Handle position updated events.
        
        Args:
            event: Position updated event
        """
        # Process position updates as needed
        pass
    
    def _handle_trade_executed_event(self, event: Event) -> None:
        """
        Handle trade executed events.
        
        Args:
            event: Trade executed event
        """
        # Process trade executed events
        # This replaces the separate entry and exit handlers
        trade_data = event.data
        trade_type = trade_data.get('trade_type')
        
        if trade_type == 'entry':
            # Handle entry trade
            pass
        elif trade_type == 'exit':
            # Handle exit trade
            pass
    
    def _handle_token_updated_event(self, event: Event) -> None:
        """
        Handle token updated events from the monitor thread.
        
        Args:
            event: Token updated event
        """
        try:
            # Extract event data
            data = event.data
            token_id = data.get('token_id')
            features = data.get('features', {})
            price = data.get('price')
            timestamp = data.get('timestamp')
            
            if not token_id:
                logger.warning("Received TOKEN_UPDATE event without token_id")
                return
                
            # Update token metadata
            if token_id not in self.active_tokens:
                logger.info(f"Adding token {token_id} from update event")
                self.add_token(token_id)
                
            # Update last update time
            self.token_last_update[token_id] = datetime.now()
            
            # Store features in token metadata
            if token_id not in self.token_metadata:
                self.token_metadata[token_id] = {}
                
            if 'features' not in self.token_metadata[token_id]:
                self.token_metadata[token_id]['features'] = {}
                
            # Update with new features
            self.token_metadata[token_id]['features'].update(features)
            
            # Update price information
            if price is not None and price > 0:
                self.token_metadata[token_id]['latest_price'] = price
                
                # Also track previous price for change calculation
                previous_price = self.token_metadata[token_id].get('previous_price')
                if previous_price is not None and previous_price > 0:
                    price_change_pct = ((price - previous_price) / previous_price) * 100
                    self.token_metadata[token_id]['price_change_pct'] = price_change_pct
                    
                self.token_metadata[token_id]['previous_price'] = price
                
            # Process timepoint with new features if available
            if features:
                timepoint = datetime.now()
                if timestamp:
                    if isinstance(timestamp, (int, float)) and timestamp > 1000000000000:
                        timestamp = timestamp / 1000  # Convert from milliseconds to seconds
                    timepoint = datetime.fromtimestamp(timestamp)
                    
                # Generate trading signals based on features
                signals = self.process_timepoint(token_id, features)
                
                if signals:
                    logger.info(f"Generated {len(signals)} trading signals for {token_id}")
                    
                    # Execute signals if they meet threshold
                    for signal in signals:
                        signal_strength = signal.get('strength', 0)
                        if signal_strength >= self.signal_threshold:
                            signal_type = signal.get('type')
                            
                            if signal_type == 'entry':
                                self._handle_entry_signal(token_id, signal, timepoint)
                            elif signal_type == 'exit':
                                position_id = signal.get('position_id')
                                if position_id:
                                    self._handle_exit_signal(token_id, position_id, signal, timepoint)
                
        except Exception as e:
            logger.error(f"Error processing token update for {token_id if 'token_id' in locals() else 'unknown token'}: {e}", exc_info=True)
            
    def _handle_token_trade_event(self, event: Event) -> None:
        """
        Handle token trade events.
        
        Args:
            event: Token trade event
        """
        try:
            # Extract event data
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Received TOKEN_TRADE event without token_id")
                return
                
            # Process the trade data
            self.update_token_trade(token_id, data)
                
        except Exception as e:
            logger.error(f"Error processing token trade for {token_id if 'token_id' in locals() else 'unknown token'}: {e}", exc_info=True)

    def _handle_model_prediction_event(self, event: Event) -> None:
        """
        Handle model prediction events.
        
        Args:
            event: Model prediction event
        """
        try:
            # Extract event data
            data = event.data
            token_id = data.get('token_id')
            model_id = data.get('model_id')
            prediction = data.get('prediction')
            features = data.get('features', {})
            
            if not token_id or prediction is None:
                logger.warning(f"Received MODEL_PREDICTION event with missing token_id or prediction: {data}")
                return
                
            # Store the prediction for use in decision making
            self.model_predictions[token_id] = {
                'model_id': model_id,
                'prediction': prediction,
                'features': features,
                'timestamp': time.time()
            }
            
            logger.info(f"Received prediction {prediction} from model {model_id} for token {token_id}")
            
            # If token is already being monitored, trigger a signal evaluation
            if token_id in self.active_tokens:
                # Get current features for the token
                try:
                    current_features = self.feature_system.get_features_for_token(token_id)
                    if current_features:
                        logger.info(f"Processing timepoint for {token_id} after model prediction")
                        self.process_timepoint(token_id, current_features)
                except Exception as e:
                    logger.error(f"Error getting features after model prediction for {token_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling model prediction event: {e}", exc_info=True)

    def set_model_prediction_weight(self, weight: float) -> None:
        """
        Set the weight to give model predictions in trading decisions.
        
        Args:
            weight: Weight value between 0.0 and 1.0
        """
        if weight < 0.0 or weight > 1.0:
            raise ValueError("model_prediction_weight must be between 0.0 and 1.0")
            
        self.model_prediction_weight = weight
        
        # Propagate to signal generator if available
        if self.signal_generator and hasattr(self.signal_generator, 'set_signal_parameters'):
            self.signal_generator.set_signal_parameters({"model_weight": weight})
            
    def get_model_prediction_weight(self) -> float:
        """
        Get the current weight given to model predictions in trading decisions.
        
        Returns:
            The model prediction weight value
        """
        return self.model_prediction_weight

    def cleanup(self):
        """Clean up resources used by the trading engine."""
        # Unregister from feature system
        if self.feature_system:
            try:
                self.feature_system.unregister_consumer(self)
                logger.info("Unregistered trading engine from feature system")
            except Exception as e:
                logger.error(f"Error unregistering from feature system: {e}")
                
        # Clean up other resources
        self.active_tokens.clear()
        self.token_metadata.clear()
        self.token_last_update.clear()
        self.token_cooldown.clear()
        self.pending_signals.clear()
        self.timepoint_signals.clear()
        self.token_features.clear()
        self.model_predictions.clear() 