#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Early Pump Detection Example

This example demonstrates how to use the EarlyPumpDetectionProvider, EarlyPumpPredictor,
and EarlyPumpStrategy to detect pump events in newly created tokens with minimal history.
"""

import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.events import EventBus, Event, EventType
from src.core.data import MockDataFeed
from src.core.features.manager import DefaultFeatureManager
from src.core.features.providers.early_pump_detection_provider import EarlyPumpDetectionProvider
from src.core.features.registry import DefaultFeatureRegistry
from src.core.ml.models.early_pump_predictor import EarlyPumpPredictor
from src.core.trading.strategies.early_pump_strategy import EarlyPumpStrategy
from src.core.trading.signal_generator import DefaultSignalGenerator
from src.core.trading.interfaces import TradingSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyPumpDemo:
    """
    Demonstration class for early-stage pump detection.
    
    Shows how to set up and use the early pump detection system with newly 
    created tokens that have minimal trading history.
    """
    
    def __init__(self):
        """Initialize the early pump detection demo."""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create data feed
        self.data_feed = MockDataFeed()
        
        # Create feature registry and manager
        self.feature_registry = DefaultFeatureRegistry()
        self.feature_manager = DefaultFeatureManager(
            registry=self.feature_registry,
            event_bus=self.event_bus
        )
        
        # Create and register early pump detection feature provider
        self.pump_detection_provider = EarlyPumpDetectionProvider(self.data_feed)
        self.feature_manager.register_provider(self.pump_detection_provider)
        
        # Create pump predictor model
        self.pump_predictor = EarlyPumpPredictor(
            model_id="early_pump_predictor_v1",
            feature_manager=self.feature_manager
        )
        
        # Create early pump strategy
        self.pump_strategy = EarlyPumpStrategy(
            feature_manager=self.feature_manager,
            pump_predictor=self.pump_predictor,
            confidence_threshold=0.6,
            cooldown_seconds=30,  # Short cooldown for demo
            max_active_tokens=5
        )
        
        # Create signal generator
        self.signal_generator = DefaultSignalGenerator(event_bus=self.event_bus)
        
        # Register strategy with signal generator
        self.signal_generator.register_strategy(self.pump_strategy)
        
        # Subscribe to trading signals
        self.event_bus.subscribe(EventType.TRADING_SIGNAL, self.on_trading_signal)
        
        # Store received signals
        self.signals = []
        
        logger.info("Early Pump Detection Demo initialized")
    
    def on_trading_signal(self, event: Event) -> None:
        """
        Handle trading signal events.
        
        Args:
            event: Trading signal event
        """
        signal_data = event.data
        signal = TradingSignal(**signal_data)
        
        # Store the signal
        self.signals.append(signal)
        
        # Log the signal with evidence
        direction_str = "LONG" if signal.direction == 1 else "SHORT" if signal.direction == -1 else "CLOSE"
        signal_type_str = signal.signal_type.name
        
        logger.info(f"SIGNAL: {signal_type_str} {direction_str} for {signal.token_id}")
        logger.info(f"  Reason: {signal.data.get('reason')}")
        logger.info(f"  Confidence: {signal.confidence:.2f}")
        logger.info(f"  Price: {signal.data.get('price')}")
        logger.info(f"  Price change: {signal.data.get('price_change', 0):.2f}%")
        logger.info(f"  Trade frequency: {signal.data.get('trade_frequency', 0):.1f} trades/min")
        logger.info(f"  Buyer dominance: {signal.data.get('buyer_dominance', 0):.2f}")
    
    def simulate_new_token(self, token_id: str, seed: int = None) -> None:
        """
        Simulate a newly created token and its initial trading activity.
        
        Unlike full pump & dump simulations, this focuses only on the early 
        phase with minimal data points, simulating just minutes of activity.
        
        Args:
            token_id: ID to use for the token
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        logger.info(f"Simulating new token: {token_id}")
        
        # Current time
        now = datetime.now()
        
        # Generate a creation event
        self.event_bus.emit(Event(
            event_type=EventType.TOKEN_CREATED,
            data={
                'token_id': token_id,
                'token_name': f"Token {token_id}",
                'creator': f"Creator_{random.randint(1000, 9999)}",
                'timestamp': now.timestamp(),
                'price': 0.0001,  # Initial price
                'supply': 1000000  # Supply
            }
        ))
        
        # Simulate behavior types with different parameters
        behaviors = {
            'organic': {
                'price_trend': 0.5,      # Slow price growth (0.5% per trade)
                'volume_trend': 1.1,     # Slightly increasing volume
                'trade_frequency': 1.0,  # Average trade frequency
                'buyer_ratio': 0.55      # Slightly more buys than sells
            },
            'promising': {
                'price_trend': 1.0,      # Moderate price growth
                'volume_trend': 1.3,     # Increasing volume
                'trade_frequency': 1.5,  # Higher trade frequency
                'buyer_ratio': 0.65      # More buys than sells
            },
            'early_pump': {
                'price_trend': 2.0,      # Faster price growth
                'volume_trend': 2.0,     # Strongly increasing volume
                'trade_frequency': 2.0,  # High trade frequency
                'buyer_ratio': 0.8       # Heavily buy-dominated
            }
        }
        
        # Pick a behavior
        behavior_type = random.choice(['organic', 'promising', 'early_pump'])
        behavior = behaviors[behavior_type]
        
        logger.info(f"Token {token_id} behavior type: {behavior_type}")
        
        # Initial conditions
        initial_price = 0.0001
        current_price = initial_price
        current_volume = 100
        base_frequency = 3  # Base trades per minute
        
        # Simulate trading activity for up to 10 minutes
        trade_count = 0
        max_trades = 20
        
        # Generate trades
        while trade_count < max_trades:
            # Timestamp for this trade
            minutes_offset = trade_count / (base_frequency * behavior['trade_frequency']) 
            trade_time = now + timedelta(minutes=minutes_offset)
            
            # Determine if buy or sell
            is_buy = random.random() < behavior['buyer_ratio']
            
            # Update price based on behavior and trade type
            price_change = behavior['price_trend'] * (1 if is_buy else -0.5)
            # Add some randomness
            price_change += (random.random() - 0.5) * behavior['price_trend']
            # Apply change
            current_price *= (1 + price_change/100)
            
            # Update volume
            volume_change = behavior['volume_trend'] * (1 if is_buy else 0.8)
            # Add some randomness
            volume_change *= 0.8 + 0.4 * random.random()
            # Apply change
            current_volume *= volume_change
            
            # Create trade data
            trade_volume = max(10, current_volume * (0.5 + 0.5 * random.random()))
            trade_data = {
                'token_id': token_id,
                'price': current_price,
                'volume': trade_volume,
                'timestamp': trade_time.timestamp(),
                'trade_type': 'buy' if is_buy else 'sell',
                'buyer': f"Buyer_{random.randint(1000, 9999)}" if is_buy else None,
                'seller': f"Seller_{random.randint(1000, 9999)}" if not is_buy else None
            }
            
            # Emit trade event
            self.event_bus.emit(Event(
                event_type=EventType.TOKEN_TRADE,
                data=trade_data
            ))
            
            # Increment trade count
            trade_count += 1
            
            # Sleep to simulate real-time processing
            time.sleep(0.1)
        
        logger.info(f"Completed simulation for {token_id}: {trade_count} trades generated")
        logger.info(f"Final price: {current_price:.8f} (change: {(current_price/initial_price - 1)*100:.2f}%)")
    
    def run(self) -> None:
        """Run the early pump detection demo."""
        logger.info("Starting Early Pump Detection Demo")
        
        try:
            # Start the signal generator
            self.signal_generator.start()
            
            # Simulate several new tokens with different behaviors
            for i in range(5):
                token_id = f"NEW-TOKEN-{i+1}"
                self.simulate_new_token(token_id, seed=i)
                # Brief pause between tokens
                time.sleep(1)
            
            # Wait for all events to be processed
            time.sleep(2)
            
            # Summarize results
            self.summarize_results()
            
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        except Exception as e:
            logger.error(f"Error during demo: {e}", exc_info=True)
        finally:
            # Stop the signal generator
            self.signal_generator.stop()
        
        logger.info("Early Pump Detection Demo completed")
    
    def summarize_results(self) -> None:
        """Summarize the trading signals generated during the demo."""
        logger.info("\n===== SIGNAL SUMMARY =====")
        
        if not self.signals:
            logger.info("No trading signals were generated")
            return
        
        # Count signals by token
        signals_by_token = {}
        for signal in self.signals:
            token_id = signal.token_id
            if token_id not in signals_by_token:
                signals_by_token[token_id] = []
            signals_by_token[token_id].append(signal)
        
        logger.info(f"Total signals: {len(self.signals)}")
        logger.info(f"Tokens with signals: {len(signals_by_token)}")
        
        # List tokens with signals
        for token_id, signals in signals_by_token.items():
            signal_types = [f"{s.signal_type.name} {s.direction}" for s in signals]
            logger.info(f"{token_id}: {len(signals)} signals - {', '.join(signal_types)}")
        
        # Analyze confidence levels
        confidences = [signal.confidence for signal in self.signals]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            logger.info(f"Average signal confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    # Run the demo
    demo = EarlyPumpDemo()
    demo.run() 