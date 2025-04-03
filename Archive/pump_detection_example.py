#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump Detection Example

This example demonstrates how to use the PumpDetectionFeatureProvider and
PumpPredictorModel to detect pump and dump events in token trading.
"""

import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.events import EventBus, Event, EventType
from src.core.data import DataFeedInterface, MockDataFeed
from src.core.features.manager import DefaultFeatureManager
from src.core.features.providers.pump_detection_provider import PumpDetectionFeatureProvider
from src.core.features.registry import DefaultFeatureRegistry
from src.core.ml.models.pump_predictor import PumpPredictorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PumpDetectionDemo:
    """Demo class for pump and dump detection."""
    
    def __init__(self):
        """Initialize the pump detection demo."""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create data feed
        self.data_feed = MockDataFeed()
        
        # Create feature registry
        self.feature_registry = DefaultFeatureRegistry()
        
        # Create feature manager
        self.feature_manager = DefaultFeatureManager(
            registry=self.feature_registry,
            event_bus=self.event_bus
        )
        
        # Create and register pump detection feature provider
        self.pump_detection_provider = PumpDetectionFeatureProvider(self.data_feed)
        self.feature_manager.register_provider(self.pump_detection_provider)
        
        # Create pump predictor model
        self.pump_predictor = PumpPredictorModel(
            model_id="pump_predictor_v1",
            feature_manager=self.feature_manager
        )
        
        # List of tokens to monitor
        self.tokens_to_monitor = [
            "BTC-USD",
            "ETH-USD",
            "SOL-USD",
            "DOGE-USD",
            "SHIB-USD",
        ]
        
        # Store token predictions
        self.token_predictions = {}
        
        # Subscribe to token trade events
        self.event_bus.subscribe(EventType.TOKEN_TRADE, self.on_token_trade)
        
        logger.info("Pump Detection Demo initialized")
    
    def on_token_trade(self, event: Event) -> None:
        """
        Handle token trade events.
        
        Args:
            event: The token trade event
        """
        data = event.data
        token_id = data.get('token_id')
        
        if token_id in self.tokens_to_monitor:
            # Update token trade data
            logger.debug(f"Received trade for {token_id}: {data}")
            
            # Check for pump or dump patterns
            self.analyze_token(token_id)
    
    def analyze_token(self, token_id: str) -> None:
        """
        Analyze a token for pump and dump patterns.
        
        Args:
            token_id: ID of the token to analyze
        """
        # Get pump detection features
        features = self.get_token_features(token_id)
        
        # Make prediction using the model
        prediction = self.pump_predictor.predict(features)
        
        # Store prediction
        self.token_predictions[token_id] = prediction
        
        # Log significant state changes
        prev_prediction = self.token_predictions.get(token_id, {})
        prev_class = prev_prediction.get('class_id', 0)
        curr_class = prediction['class_id']
        
        if prev_class != curr_class or prediction['confidence'] > 0.8:
            logger.info(f"[{token_id}] Detected state: {prediction['class_label']} (Phase: {prediction['phase']}) with {prediction['confidence']:.2f} confidence")
            
            # Emit an event for significant predictions
            if prediction['confidence'] > 0.7:
                self.event_bus.emit(Event(
                    event_type=EventType.MODEL_PREDICTION,
                    data={
                        'token_id': token_id,
                        'model_id': self.pump_predictor.model_id,
                        'prediction': prediction,
                        'timestamp': datetime.now().timestamp()
                    }
                ))
        
        # Print detailed analysis periodically
        curr_time = datetime.now()
        last_analysis_time = getattr(self, 'last_analysis_time', datetime.min)
        
        if (curr_time - last_analysis_time).total_seconds() > 60:  # Every minute
            self.print_detailed_analysis()
            self.last_analysis_time = curr_time
    
    def get_token_features(self, token_id: str) -> Dict[str, Any]:
        """
        Get pump detection features for a token.
        
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
        
        # Get features from the pump detection provider
        features = {}
        
        # Get pump velocity and acceleration
        features['price_velocity'] = self.feature_manager.compute_feature(token_id, 'price_velocity', context)
        features['price_acceleration'] = self.feature_manager.compute_feature(token_id, 'price_acceleration', context)
        
        # Get volume metrics
        features['volume_surge_ratio'] = self.feature_manager.compute_feature(token_id, 'volume_surge_ratio', context)
        features['volume_volatility'] = self.feature_manager.compute_feature(token_id, 'volume_volatility', context)
        features['buy_sell_volume_ratio'] = self.feature_manager.compute_feature(token_id, 'buy_sell_volume_ratio', context)
        
        # Get price metrics
        features['price_deviation'] = self.feature_manager.compute_feature(token_id, 'price_deviation', context)
        features['price_volatility_short'] = self.feature_manager.compute_feature(token_id, 'price_volatility_short', context)
        features['price_volatility_ratio'] = self.feature_manager.compute_feature(token_id, 'price_volatility_ratio', context)
        
        # Get pattern scores
        features['pump_pattern_score'] = self.feature_manager.compute_feature(token_id, 'pump_pattern_score', context)
        features['dump_pattern_score'] = self.feature_manager.compute_feature(token_id, 'dump_pattern_score', context)
        features['abnormal_activity_score'] = self.feature_manager.compute_feature(token_id, 'abnormal_activity_score', context)
        
        return features
    
    def print_detailed_analysis(self) -> None:
        """Print detailed analysis of all monitored tokens."""
        logger.info("\n===== DETAILED TOKEN ANALYSIS =====")
        
        for token_id in self.tokens_to_monitor:
            if token_id not in self.token_predictions:
                continue
                
            prediction = self.token_predictions[token_id]
            features = self.get_token_features(token_id)
            
            logger.info(f"\n--- {token_id} Analysis ---")
            logger.info(f"State: {prediction['class_label']} (Phase: {prediction['phase']}) with {prediction['confidence']:.2f} confidence")
            
            # Feature values
            logger.info("Key Features:")
            logger.info(f"  Price Velocity: {features.get('price_velocity', 0):.6f}")
            logger.info(f"  Price Acceleration: {features.get('price_acceleration', 0):.6f}")
            logger.info(f"  Volume Surge Ratio: {features.get('volume_surge_ratio', 0):.2f}")
            logger.info(f"  Volume Volatility: {features.get('volume_volatility', 0):.2f}")
            logger.info(f"  Buy/Sell Ratio: {features.get('buy_sell_volume_ratio', 1):.2f}")
            logger.info(f"  Price Deviation: {features.get('price_deviation', 0):.2f}%")
            
            # Pattern scores
            logger.info("Pattern Scores:")
            logger.info(f"  Pump Pattern: {features.get('pump_pattern_score', 0):.2f}")
            logger.info(f"  Dump Pattern: {features.get('dump_pattern_score', 0):.2f}")
            logger.info(f"  Abnormal Activity: {features.get('abnormal_activity_score', 0):.2f}")
            
            # Probabilities
            logger.info("State Probabilities:")
            for state, prob in prediction.get('probabilities', {}).items():
                logger.info(f"  {state}: {prob:.2f}")
    
    def simulate_pump_dump_scenario(self, token_id: str) -> None:
        """
        Simulate a pump and dump scenario for testing.
        
        Args:
            token_id: ID of the token to simulate
        """
        logger.info(f"Simulating pump and dump scenario for {token_id}")
        
        # Current time
        curr_time = datetime.now()
        
        # Create price and volume data for different phases
        phases = [
            # Phase 1: Accumulation (gradual increase, moderate volume)
            {'duration': 5, 'price_change': 0.05, 'volume_factor': 1.2},
            
            # Phase 2: Pump (sharp increase, high volume)
            {'duration': 3, 'price_change': 0.3, 'volume_factor': 5.0},
            
            # Phase 3: Peak/Distribution (fluctuation, high volume)
            {'duration': 2, 'price_change': 0.05, 'volume_factor': 3.0},
            
            # Phase 4: Dump (sharp decrease, high volume)
            {'duration': 3, 'price_change': -0.4, 'volume_factor': 4.0},
            
            # Phase 5: Return to normal (gradual decrease, low volume)
            {'duration': 5, 'price_change': -0.1, 'volume_factor': 0.8},
        ]
        
        # Base price and volume
        base_price = 1.0
        base_volume = 1000.0
        
        # Starting price and time
        curr_price = base_price
        curr_volume = base_volume
        
        # Generate trade events for each phase
        for phase_idx, phase in enumerate(phases):
            phase_start_price = curr_price
            phase_duration = phase['duration']  # minutes
            phase_price_change = phase['price_change']
            phase_volume_factor = phase['volume_factor']
            
            # Target price at end of phase
            target_price = phase_start_price * (1 + phase_price_change)
            
            # Generate events for this phase
            for minute in range(phase_duration):
                # Progress within the phase (0 to 1)
                progress = minute / (phase_duration - 1) if phase_duration > 1 else 1
                
                # Calculate price for this minute
                if phase_idx == 2:  # Peak phase - add fluctuation
                    fluctuation = (0.5 - progress) * 0.1  # Fluctuate around peak
                    minute_price = phase_start_price + (phase_price_change * progress * phase_start_price) + (fluctuation * phase_start_price)
                else:
                    minute_price = phase_start_price + (phase_price_change * progress * phase_start_price)
                
                # Calculate volume for this minute
                if phase_idx == 1:  # Pump phase - volume builds up
                    minute_volume = base_volume * (1 + (phase_volume_factor - 1) * (progress ** 2))
                elif phase_idx == 3:  # Dump phase - volume starts high and decreases
                    minute_volume = base_volume * (phase_volume_factor - (phase_volume_factor - 1) * progress)
                else:
                    minute_volume = base_volume * phase_volume_factor
                
                # Generate several trades per minute
                trades_per_minute = max(1, int(minute_volume / 100))
                for _ in range(trades_per_minute):
                    # Add some randomness to price and volume
                    trade_price = minute_price * (1 + (np.random.random() - 0.5) * 0.02)
                    trade_volume = minute_volume / trades_per_minute * (0.5 + np.random.random())
                    
                    # Create trade event
                    trade_time = curr_time + timedelta(minutes=minute)
                    trade_data = {
                        'token_id': token_id,
                        'price': trade_price,
                        'volume': trade_volume,
                        'timestamp': trade_time.timestamp(),
                        'trade_type': 'buy' if phase_idx < 3 else 'sell'
                    }
                    
                    # Emit trade event
                    self.event_bus.emit(Event(
                        event_type=EventType.TOKEN_TRADE,
                        data=trade_data
                    ))
                    
                    # Analyze after each significant trade
                    if _ % 5 == 0:
                        self.analyze_token(token_id)
                    
                    # Small delay to simulate real-time trading
                    time.sleep(0.01)
                
                # Update current price
                curr_price = minute_price
                
                # Log phase progress
                logger.debug(f"Phase {phase_idx+1}: {minute+1}/{phase_duration} - Price: {curr_price:.4f}, Volume: {minute_volume:.0f}")
            
            # Log phase completion
            logger.info(f"Completed phase {phase_idx+1} for {token_id} - Price: {curr_price:.4f}")
        
        logger.info(f"Completed pump and dump simulation for {token_id}")
    
    def run(self) -> None:
        """Run the pump detection demo."""
        logger.info("Starting Pump Detection Demo")
        
        try:
            # Run simulation for a test token
            self.simulate_pump_dump_scenario("TEST-PUMP-TOKEN")
            
            # Monitor tokens for a period
            start_time = time.time()
            duration = 300  # 5 minutes
            
            while time.time() - start_time < duration:
                # Analyze all monitored tokens
                for token_id in self.tokens_to_monitor:
                    self.analyze_token(token_id)
                
                # Wait before next analysis
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        except Exception as e:
            logger.error(f"Error during demo: {e}", exc_info=True)
        
        logger.info("Pump Detection Demo completed")


if __name__ == "__main__":
    # Import numpy here to avoid moving it to the top
    import numpy as np
    
    # Run the demo
    demo = PumpDetectionDemo()
    demo.run() 