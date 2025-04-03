#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pump Detection Strategy Example

This example demonstrates how to use the PumpDetectionStrategy with the trading system
to automatically generate trading signals based on pump and dump patterns.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.core.events import EventBus, Event, EventType
from src.core.data import MockDataFeed
from src.core.features.manager import DefaultFeatureManager
from src.core.features.providers.pump_detection_provider import PumpDetectionFeatureProvider
from src.core.features.registry import DefaultFeatureRegistry
from src.core.ml.models.pump_predictor import PumpPredictorModel
from src.core.trading.strategies.pump_detection_strategy import PumpDetectionStrategy
from src.core.trading.signal_generator import DefaultSignalGenerator
from src.core.trading.interfaces import TradingSignal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PumpDetectionStrategyDemo:
    """Demo class for using the pump detection strategy."""
    
    def __init__(self):
        """Initialize the strategy demo."""
        # Create event bus
        self.event_bus = EventBus()
        
        # Create data feed
        self.data_feed = MockDataFeed()
        self.data_feed.connect()
        
        # Create feature registry and manager
        self.feature_registry = DefaultFeatureRegistry()
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
        
        # Create pump detection strategy
        self.pump_strategy = PumpDetectionStrategy(
            feature_manager=self.feature_manager,
            pump_predictor=self.pump_predictor,
            confidence_threshold=0.7,
            enable_shorts=True,
            enable_pump_entry=False
        )
        
        # Create signal generator
        self.signal_generator = DefaultSignalGenerator(event_bus=self.event_bus)
        
        # Register strategy with signal generator
        self.signal_generator.register_strategy(self.pump_strategy)
        
        # Subscribe to trading signals
        self.event_bus.subscribe(EventType.TRADING_SIGNAL, self.on_trading_signal)
        
        # Store received signals
        self.signals: List[TradingSignal] = []
        
        logger.info("Pump Detection Strategy Demo initialized")
    
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
        
        # Log the signal
        direction_str = "LONG" if signal.direction == 1 else "SHORT" if signal.direction == -1 else "CLOSE"
        signal_type_str = signal.signal_type.name
        
        logger.info(f"Received {signal_type_str} {direction_str} signal for {signal.token_id}")
        logger.info(f"  Reason: {signal.data.get('reason')}")
        logger.info(f"  Confidence: {signal.confidence:.2f}")
        logger.info(f"  Price: {signal.data.get('price')}")
        logger.info(f"  State: {signal.data.get('state')} (Phase: {signal.data.get('phase')})")
    
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
                    
                    # Small delay to simulate real-time trading
                    time.sleep(0.01)
                
                # Update current price
                curr_price = minute_price
                
                # Log phase progress
                if minute == phase_duration // 2:
                    logger.info(f"Phase {phase_idx+1} progress: {minute+1}/{phase_duration} - Price: {curr_price:.4f}, Volume: {minute_volume:.0f}")
            
            # Log phase completion
            logger.info(f"Completed phase {phase_idx+1} for {token_id} - Price: {curr_price:.4f}")
            
            # Wait between phases to allow the system to process the data
            time.sleep(0.5)
        
        logger.info(f"Completed pump and dump simulation for {token_id}")
    
    def analyze_signals(self) -> None:
        """Analyze the trading signals generated during the simulation."""
        logger.info("\n===== SIGNAL ANALYSIS =====")
        
        if not self.signals:
            logger.info("No signals were generated")
            return
        
        # Count signals by type and direction
        open_long = sum(1 for s in self.signals if s.signal_type.name == "OPEN" and s.direction == 1)
        open_short = sum(1 for s in self.signals if s.signal_type.name == "OPEN" and s.direction == -1)
        close_signals = sum(1 for s in self.signals if s.signal_type.name == "CLOSE")
        
        logger.info(f"Total signals: {len(self.signals)}")
        logger.info(f"OPEN LONG signals: {open_long}")
        logger.info(f"OPEN SHORT signals: {open_short}")
        logger.info(f"CLOSE signals: {close_signals}")
        
        # Analyze signals by phase
        signals_by_phase = {}
        for signal in self.signals:
            phase = signal.data.get('phase', 'unknown')
            if phase not in signals_by_phase:
                signals_by_phase[phase] = []
            signals_by_phase[phase].append(signal)
        
        logger.info("\nSignals by phase:")
        for phase, signals in signals_by_phase.items():
            logger.info(f"Phase {phase}: {len(signals)} signals")
        
        # Analyze confidence levels
        confidences = [signal.confidence for signal in self.signals]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.info(f"\nAverage signal confidence: {avg_confidence:.2f}")
        logger.info(f"Min confidence: {min(confidences):.2f}")
        logger.info(f"Max confidence: {max(confidences):.2f}")
    
    def run(self) -> None:
        """Run the strategy demo."""
        logger.info("Starting Pump Detection Strategy Demo")
        
        try:
            # Start the signal generator
            self.signal_generator.start()
            
            # Run simulation for a test token
            self.simulate_pump_dump_scenario("TEST-PUMP-TOKEN")
            
            # Wait for all events to be processed
            time.sleep(1)
            
            # Analyze the generated signals
            self.analyze_signals()
            
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
        except Exception as e:
            logger.error(f"Error during demo: {e}", exc_info=True)
        finally:
            # Stop the signal generator
            self.signal_generator.stop()
        
        logger.info("Pump Detection Strategy Demo completed")


if __name__ == "__main__":
    # Import numpy here to avoid moving it to the top
    import numpy as np
    
    # Run the demo
    demo = PumpDetectionStrategyDemo()
    demo.run() 