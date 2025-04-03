#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Generator Module

This module provides the DefaultSignalGenerator implementation for analyzing features
and generating trading signals based on configured strategies and model predictions.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import uuid
import random

from src.core.events import EventBus, Event, EventType
from src.core.trading.interfaces import SignalGenerator, TradingSignal, Position
from src.core.trading.error_handler import (
    TradingErrorHandler, SignalGenerationError, InvalidParameterError
)

logger = logging.getLogger(__name__)


class DefaultSignalGenerator(SignalGenerator):
    """
    Default implementation of the SignalGenerator interface.
    
    This implementation analyzes features and market data to generate trading signals,
    with support for multiple strategies and model prediction integration.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None,
                entry_threshold: float = 0.6,
                exit_threshold: float = 0.5,
                model_weight: float = 0.5,
                signal_expiry_seconds: float = 300.0):
        """
        Initialize the signal generator.
        
        Args:
            event_bus: EventBus for publishing signal-related events
            entry_threshold: Threshold for entry signal generation (0.0 to 1.0)
            exit_threshold: Threshold for exit signal generation (0.0 to 1.0)
            model_weight: Weight to give model predictions (0.0 to 1.0)
            signal_expiry_seconds: Time in seconds until a signal expires
        """
        self.event_bus = event_bus
        
        # Signal generation parameters
        self._params = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'model_weight': model_weight,
            'signal_expiry_seconds': signal_expiry_seconds,
            'feature_weights': {
                'price_momentum_signal': 1.0,
                'volume_spike_signal': 0.7,
                'rsi_signal': 0.5,
                'macd_signal': 0.8
            },
            'stop_loss_pct': 0.05,  # Default 5% stop loss
            'take_profit_pct': 0.15  # Default 15% take profit
        }
        
        # Error handling
        self._error_handler = TradingErrorHandler(event_bus=event_bus)
        
        # Signal tracking
        self._recent_signals: Dict[str, List[TradingSignal]] = {}
        self._max_signals_per_token = 10
        
        logger.info(f"Initialized DefaultSignalGenerator with entry_threshold={entry_threshold}, "
                   f"exit_threshold={exit_threshold}, model_weight={model_weight}")
    
    def generate_signals(self, token_id: str, features: Dict[str, Any], 
                       timestamp: datetime) -> List[TradingSignal]:
        """
        Generate trading signals for a token based on features.
        
        Args:
            token_id: ID of the token
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated trading signals
        """
        try:
            # Generate entry signals
            entry_signals = self.generate_entry_signals(token_id, features, timestamp)
            
            # Check for existing positions to generate exit signals
            exit_signals = []
            position_data = features.get('position_data')
            if position_data:
                # Create Position object from position data
                try:
                    position = Position(
                        position_id=position_data.get('position_id', ''),
                        token_id=token_id,
                        entry_price=position_data.get('entry_price', 0.0),
                        position_size=position_data.get('position_size', 0.0),
                        stop_loss=position_data.get('stop_loss', 0.0),
                        take_profit=position_data.get('take_profit', 0.0),
                        entry_time=position_data.get('entry_time', timestamp),
                        metadata=position_data.get('metadata', {}),
                        current_price=features.get('current_price'),
                        unrealized_pnl=position_data.get('unrealized_pnl'),
                        unrealized_pnl_pct=position_data.get('unrealized_pnl_pct')
                    )
                    exit_signals = self.generate_exit_signals(token_id, position, features, timestamp)
                except Exception as e:
                    err = SignalGenerationError(f"Error creating Position object: {str(e)}")
                    self._error_handler.handle_error(
                        err,
                        context={
                            'component': 'DefaultSignalGenerator',
                            'operation': 'generate_signals.position',
                            'token_id': token_id,
                            'features': {k: v for k, v in features.items() if k != 'position_data'}
                        }
                    )
            
            # Combine all signals
            signals = entry_signals + exit_signals
            
            # Store signals in recent history
            if token_id not in self._recent_signals:
                self._recent_signals[token_id] = []
            
            self._recent_signals[token_id].extend(signals)
            
            # Limit size of recent signals list
            if len(self._recent_signals[token_id]) > self._max_signals_per_token:
                self._recent_signals[token_id] = self._recent_signals[token_id][-self._max_signals_per_token:]
            
            # Publish signal events
            if self.event_bus is not None:
                for signal in signals:
                    self._publish_signal_event(signal)
            
            return signals
            
        except Exception as e:
            err = SignalGenerationError(f"Error generating signals: {str(e)}")
            self._error_handler.handle_error(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'generate_signals',
                    'token_id': token_id,
                    'timestamp': timestamp.isoformat()
                }
            )
            
            # Try to recover by returning an empty list
            recovery = self._error_handler.try_recover(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'generate_signals',
                    'token_id': token_id
                }
            )
            
            if recovery is not None:
                return recovery
            return []
    
    @TradingErrorHandler.retry_static(max_attempts=2, delay=0.1, exceptions=[SignalGenerationError])
    def generate_entry_signals(self, token_id: str, features: Dict[str, Any], 
                             timestamp: datetime) -> List[TradingSignal]:
        """
        Generate entry signals for a token based on features.
        
        Args:
            token_id: ID of the token
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated entry signals
        """
        try:
            signals = []
            
            # 1. Basic signal from feature indicators
            feature_score = self._calculate_feature_score(features)
            
            # 2. Get model prediction signal if available
            model_signal = None
            if 'model_prediction' in features:
                try:
                    model_prediction = features['model_prediction']
                    model_signal = self.evaluate_model_prediction(
                        token_id, model_prediction, features, timestamp
                    )
                except Exception as e:
                    logger.warning(f"Error evaluating model prediction for {token_id}: {e}")
            
            # 3. Combine feature score and model signal
            combined_score = feature_score
            if model_signal is not None:
                # Weight the model prediction according to configuration
                model_weight = self._params['model_weight']
                combined_score = (1 - model_weight) * feature_score + model_weight * model_signal.score
            
            # 4. Apply threshold and generate signal if strong enough
            if abs(combined_score) >= self._params['entry_threshold']:
                # Calculate expiry time
                expiry = timestamp + timedelta(seconds=self._params['signal_expiry_seconds'])
                
                # Create entry signal
                current_price = features.get('current_price', 0.0)
                if current_price <= 0:
                    raise SignalGenerationError(f"Invalid current price for {token_id}: {current_price}")
                
                entry_signal = TradingSignal(
                    token_id=token_id,
                    signal_type='entry',
                    score=combined_score,
                    price=current_price,
                    timestamp=timestamp,
                    expiry=expiry,
                    metadata={
                        'feature_score': feature_score,
                        'model_score': model_signal.score if model_signal else None,
                        'indicators': {
                            k: features.get(k) for k in [
                                'price_momentum_signal', 'volume_spike_signal', 'rsi_14', 'macd_histogram'
                            ] if k in features
                        }
                    }
                )
                
                signals.append(entry_signal)
                logger.info(f"Generated entry signal for {token_id} with score {combined_score:.2f}")
            
            return signals
            
        except Exception as e:
            err = SignalGenerationError(f"Error generating entry signals: {str(e)}")
            self._error_handler.handle_error(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'generate_entry_signals',
                    'token_id': token_id,
                    'timestamp': timestamp.isoformat(),
                    'feature_keys': list(features.keys())
                }
            )
            raise err
    
    def generate_exit_signals(self, token_id: str, position: Position, 
                            features: Dict[str, Any], timestamp: datetime) -> List[TradingSignal]:
        """
        Generate exit signals for a position based on features.
        
        Args:
            token_id: ID of the token
            position: Current position
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated exit signals
        """
        try:
            signals = []
            
            # Check if position data is valid
            if not position.position_id:
                return []
            
            current_price = features.get('current_price', 0.0)
            if current_price <= 0:
                raise SignalGenerationError(f"Invalid current price for {token_id}: {current_price}")
            
            # 1. Check stop-loss
            if position.stop_loss > 0 and current_price <= position.stop_loss:
                exit_signal = TradingSignal(
                    token_id=token_id,
                    signal_type='exit',
                    score=-1.0,  # Strong sell signal
                    price=current_price,
                    timestamp=timestamp,
                    expiry=None,  # No expiry for stop-loss signals
                    metadata={
                        'position_id': position.position_id,
                        'reason': 'stop_loss',
                        'entry_price': position.entry_price,
                        'stop_loss': position.stop_loss,
                        'pnl_pct': (current_price / position.entry_price - 1) * 100
                    }
                )
                signals.append(exit_signal)
                logger.info(f"Generated stop-loss exit signal for {token_id}, position {position.position_id}")
                return signals  # Return immediately for stop-loss
            
            # 2. Check take-profit
            if position.take_profit > 0 and current_price >= position.take_profit:
                exit_signal = TradingSignal(
                    token_id=token_id,
                    signal_type='exit',
                    score=-0.8,  # Strong sell signal but not as strong as stop-loss
                    price=current_price,
                    timestamp=timestamp,
                    expiry=None,  # No expiry for take-profit signals
                    metadata={
                        'position_id': position.position_id,
                        'reason': 'take_profit',
                        'entry_price': position.entry_price,
                        'take_profit': position.take_profit,
                        'pnl_pct': (current_price / position.entry_price - 1) * 100
                    }
                )
                signals.append(exit_signal)
                logger.info(f"Generated take-profit exit signal for {token_id}, position {position.position_id}")
                return signals  # Return immediately for take-profit
            
            # 3. Check technical indicators for exit signals
            feature_score = self._calculate_feature_score(features)
            
            # If score becomes strongly negative, consider exit
            if feature_score <= -self._params['exit_threshold']:
                exit_signal = TradingSignal(
                    token_id=token_id,
                    signal_type='exit',
                    score=feature_score,
                    price=current_price,
                    timestamp=timestamp,
                    expiry=timestamp + timedelta(seconds=self._params['signal_expiry_seconds']),
                    metadata={
                        'position_id': position.position_id,
                        'reason': 'technical',
                        'entry_price': position.entry_price,
                        'indicators': {
                            k: features.get(k) for k in [
                                'price_momentum_signal', 'volume_spike_signal', 'rsi_14', 'macd_histogram'
                            ] if k in features
                        },
                        'pnl_pct': (current_price / position.entry_price - 1) * 100
                    }
                )
                signals.append(exit_signal)
                logger.info(f"Generated technical exit signal for {token_id}, position {position.position_id} "
                           f"with score {feature_score:.2f}")
            
            # 4. Check model prediction for exit signals
            if 'model_prediction' in features:
                try:
                    model_prediction = features['model_prediction']
                    model_signal = self.evaluate_model_prediction(
                        token_id, model_prediction, features, timestamp
                    )
                    
                    # If model strongly suggests exit
                    if model_signal.score <= -self._params['exit_threshold']:
                        exit_signal = TradingSignal(
                            token_id=token_id,
                            signal_type='exit',
                            score=model_signal.score,
                            price=current_price,
                            timestamp=timestamp,
                            expiry=timestamp + timedelta(seconds=self._params['signal_expiry_seconds']),
                            metadata={
                                'position_id': position.position_id,
                                'reason': 'model_prediction',
                                'entry_price': position.entry_price,
                                'model_score': model_signal.score,
                                'pnl_pct': (current_price / position.entry_price - 1) * 100
                            }
                        )
                        signals.append(exit_signal)
                        logger.info(f"Generated model-based exit signal for {token_id}, position {position.position_id} "
                                   f"with score {model_signal.score:.2f}")
                except Exception as e:
                    logger.warning(f"Error evaluating model prediction for exit signal: {e}")
                
            return signals
            
        except Exception as e:
            err = SignalGenerationError(f"Error generating exit signals: {str(e)}")
            self._error_handler.handle_error(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'generate_exit_signals',
                    'token_id': token_id,
                    'position_id': position.position_id,
                    'timestamp': timestamp.isoformat()
                }
            )
            
            # For exit signals, we should be more cautious - don't recover automatically
            # but require manual review
            return []
    
    def evaluate_model_prediction(self, token_id: str, prediction: Any, 
                                features: Dict[str, Any], timestamp: datetime) -> TradingSignal:
        """
        Evaluate a model prediction to generate a trading signal.
        
        Args:
            token_id: ID of the token
            prediction: Model prediction
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            Generated trading signal
        """
        try:
            # Interpret the prediction based on its type
            score = 0.0
            
            # If prediction is already a normalized score (-1 to 1)
            if isinstance(prediction, (int, float)):
                score = float(prediction)
                if score < -1.0 or score > 1.0:
                    # Normalize to -1 to 1 range
                    score = max(-1.0, min(1.0, score))
            
            # If prediction is a classification (0 or 1)
            elif isinstance(prediction, (bool, int)) and prediction in (0, 1, True, False):
                score = 1.0 if prediction else -1.0
            
            # If prediction is a dictionary with a score or probability
            elif isinstance(prediction, dict):
                if 'score' in prediction:
                    score = prediction['score']
                elif 'probability' in prediction:
                    # Convert probability (0-1) to score (-1 to 1)
                    probability = prediction['probability']
                    score = (probability * 2) - 1.0
                elif 'class' in prediction and prediction['class'] in (0, 1, '0', '1'):
                    # Convert class to score
                    score = 1.0 if prediction['class'] in (1, '1') else -1.0
            
            # Fallback to extracting a score from the raw prediction
            else:
                # Try to extract a score from the prediction
                try:
                    # If it's a string, try to parse as float
                    if isinstance(prediction, str):
                        score = float(prediction)
                    # If it's a list, use the first element
                    elif isinstance(prediction, (list, tuple)) and prediction:
                        score = float(prediction[0])
                    else:
                        raise ValueError(f"Unsupported prediction format: {type(prediction)}")
                    
                    # Normalize to -1 to 1 range
                    score = max(-1.0, min(1.0, score))
                    
                except Exception as e:
                    logger.warning(f"Could not extract score from prediction {prediction}: {e}")
                    # Default to neutral score
                    score = 0.0
            
            # Get current price
            current_price = features.get('current_price', 0.0)
            if current_price <= 0:
                raise SignalGenerationError(f"Invalid current price for {token_id}: {current_price}")
            
            # Create a signal based on the model score
            signal = TradingSignal(
                token_id=token_id,
                signal_type='model',  # Indicate this is from a model
                score=score,
                price=current_price,
                timestamp=timestamp,
                expiry=timestamp + timedelta(seconds=self._params['signal_expiry_seconds']),
                metadata={
                    'source': 'model_prediction',
                    'raw_prediction': prediction,
                    'confidence': abs(score)  # Use absolute score as confidence
                }
            )
            
            return signal
            
        except Exception as e:
            err = SignalGenerationError(f"Error evaluating model prediction: {str(e)}")
            self._error_handler.handle_error(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'evaluate_model_prediction',
                    'token_id': token_id,
                    'prediction_type': type(prediction).__name__,
                    'timestamp': timestamp.isoformat()
                }
            )
            
            # Default to a neutral signal
            return TradingSignal(
                token_id=token_id,
                signal_type='model',
                score=0.0,  # Neutral score
                price=features.get('current_price', 0.0),
                timestamp=timestamp,
                expiry=timestamp + timedelta(seconds=30),  # Short expiry for fallback signal
                metadata={
                    'source': 'model_prediction',
                    'is_fallback': True,
                    'error': str(e)
                }
            )
    
    def set_signal_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for signal generation.
        
        Args:
            params: Dictionary of parameter values
        """
        try:
            # Validate parameters
            for key, value in params.items():
                if key in ('entry_threshold', 'exit_threshold', 'model_weight') and (value < 0.0 or value > 1.0):
                    raise InvalidParameterError(f"Parameter {key} must be between 0.0 and 1.0")
                    
            # Update parameters
            self._params.update(params)
            
            logger.info(f"Updated signal parameters: {params}")
            
        except Exception as e:
            err = InvalidParameterError(f"Error setting signal parameters: {str(e)}")
            self._error_handler.handle_error(
                err,
                context={
                    'component': 'DefaultSignalGenerator',
                    'operation': 'set_signal_parameters',
                    'params': params
                }
            )
            raise err
    
    def get_signal_parameters(self) -> Dict[str, Any]:
        """
        Get current signal generation parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return dict(self._params)
    
    def _calculate_feature_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate a signal score from feature values.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Signal score between -1.0 (strong sell) and 1.0 (strong buy)
        """
        score = 0.0
        feature_weights = self._params['feature_weights']
        weight_sum = 0.0
        
        # Price momentum signal
        if 'price_momentum_signal' in features:
            momentum_signal = features['price_momentum_signal']
            weight = feature_weights.get('price_momentum_signal', 1.0)
            score += momentum_signal * weight
            weight_sum += weight
        
        # Volume spike signal
        if 'volume_spike_signal' in features:
            volume_signal = features['volume_spike_signal']
            weight = feature_weights.get('volume_spike_signal', 0.7)
            score += volume_signal * weight
            weight_sum += weight
        
        # RSI signal (convert 0-100 to -1 to 1)
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            # RSI over 70 is overbought (bearish), under 30 is oversold (bullish)
            rsi_signal = 0.0
            if rsi < 30:
                rsi_signal = (30 - rsi) / 30  # 0 to 1 (bullish)
            elif rsi > 70:
                rsi_signal = -1 * (rsi - 70) / 30  # -1 to 0 (bearish)
                
            weight = feature_weights.get('rsi_signal', 0.5)
            score += rsi_signal * weight
            weight_sum += weight
        
        # MACD signal
        if 'macd_histogram' in features:
            macd_hist = features['macd_histogram']
            # Normalize MACD histogram to -1 to 1 range
            macd_signal = max(-1.0, min(1.0, macd_hist * 10))
            
            weight = feature_weights.get('macd_signal', 0.8)
            score += macd_signal * weight
            weight_sum += weight
        
        # Normalize final score
        if weight_sum > 0:
            score /= weight_sum
        
        # Ensure score is in -1 to 1 range
        return max(-1.0, min(1.0, score))
    
    def _publish_signal_event(self, signal: TradingSignal) -> None:
        """
        Publish a signal event to the event bus.
        
        Args:
            signal: The trading signal to publish
        """
        if not self.event_bus:
            return
            
        try:
            # Convert signal to dict for event data
            signal_dict = signal._asdict()
            
            # Convert datetime objects to timestamps
            if signal_dict['timestamp']:
                signal_dict['timestamp'] = signal_dict['timestamp'].timestamp()
            if signal_dict['expiry']:
                signal_dict['expiry'] = signal_dict['expiry'].timestamp()
            
            # Publish event
            self.event_bus.publish(Event(
                event_type=EventType.TRADING_SIGNAL,
                data=signal_dict,
                token_id=signal.token_id,
                source="signal_generator"
            ))
            
        except Exception as e:
            logger.error(f"Error publishing signal event: {e}", exc_info=True)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about errors handled by the signal generator.
        
        Returns:
            Dictionary with error statistics
        """
        return self._error_handler.get_error_statistics()
    
    def clear_error_history(self) -> None:
        """Clear the error history and counts."""
        self._error_handler.clear_error_history() 