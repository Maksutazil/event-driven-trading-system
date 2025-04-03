#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Event History Example

This example demonstrates how to use the EventHistoryManager to track and analyze events
for different tokens in the trading system.
"""

import sys
import time
import logging
import random
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import event system components
from src.core.events import (
    Event, EventType, EventBus, EventHistoryManager,
    BaseEventPublisher
)


class MockTokenPublisher(BaseEventPublisher):
    """
    Mock publisher that simulates token-related events.
    """
    
    def __init__(self, event_bus, token_ids=None):
        """
        Initialize the mock publisher.
        
        Args:
            event_bus: EventBus instance
            token_ids: List of token IDs to simulate events for
        """
        super().__init__(event_bus)
        self.token_ids = token_ids or [f"TOKEN_{i}" for i in range(1, 6)]
        self.running = False
        logger.info(f"MockTokenPublisher initialized with {len(self.token_ids)} tokens")
    
    def publish_token_created(self, token_id):
        """Publish a token creation event."""
        self.publish_event(
            EventType.TOKEN_CREATED,
            {
                "token_id": token_id,
                "name": f"Mock {token_id}",
                "symbol": token_id,
                "initial_price": 100.0,
                "timestamp": time.time()
            }
        )
        logger.info(f"Published TOKEN_CREATED for {token_id}")
    
    def publish_token_trade(self, token_id, price, volume):
        """Publish a token trade event."""
        self.publish_event(
            EventType.TOKEN_TRADE,
            {
                "token_id": token_id,
                "price": price,
                "volume": volume,
                "timestamp": time.time()
            }
        )
        logger.debug(f"Published TOKEN_TRADE for {token_id} at ${price:.2f} (volume: {volume:.2f})")
    
    def publish_position_opened(self, token_id, price, size):
        """Publish a position opened event."""
        position_id = f"POS_{token_id}_{int(time.time())}"
        self.publish_event(
            EventType.POSITION_OPENED,
            {
                "token_id": token_id,
                "position_id": position_id,
                "entry_price": price,
                "size": size,
                "timestamp": time.time()
            }
        )
        logger.info(f"Published POSITION_OPENED for {token_id} at ${price:.2f} (size: {size:.2f})")
        return position_id
    
    def publish_position_closed(self, token_id, position_id, price):
        """Publish a position closed event."""
        self.publish_event(
            EventType.POSITION_CLOSED,
            {
                "token_id": token_id,
                "position_id": position_id,
                "exit_price": price,
                "timestamp": time.time()
            }
        )
        logger.info(f"Published POSITION_CLOSED for {token_id} at ${price:.2f}")
    
    async def start_simulation(self, duration_seconds=30):
        """
        Start simulating token events.
        
        Args:
            duration_seconds: How long to run the simulation
        """
        self.running = True
        start_time = time.time()
        
        # Create all tokens
        for token_id in self.token_ids:
            self.publish_token_created(token_id)
            await asyncio.sleep(0.1)
        
        # Current price for each token
        prices = {token_id: 100.0 for token_id in self.token_ids}
        
        # Active positions
        positions = {}
        
        logger.info(f"Starting event simulation for {duration_seconds} seconds...")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            # Pick a random token
            token_id = random.choice(self.token_ids)
            
            # Update price (random walk with drift)
            current_price = prices[token_id]
            change_pct = random.normalvariate(0.0001, 0.005)  # Small upward drift
            new_price = max(0.01, current_price * (1 + change_pct))
            prices[token_id] = new_price
            
            # Generate trade event
            volume = random.uniform(10, 1000)
            self.publish_token_trade(token_id, new_price, volume)
            
            # Randomly open or close positions (5% chance each)
            if token_id not in positions and random.random() < 0.05:
                # Open position
                size = random.uniform(1, 10)
                position_id = self.publish_position_opened(token_id, new_price, size)
                positions[token_id] = position_id
            elif token_id in positions and random.random() < 0.05:
                # Close position
                position_id = positions[token_id]
                self.publish_position_closed(token_id, position_id, new_price)
                del positions[token_id]
            
            # Random sleep interval
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        self.running = False
        logger.info("Event simulation completed")


def print_token_statistics(history_manager, token_id):
    """Print statistics about a token's events."""
    logger.info(f"\n--- Statistics for {token_id} ---")
    
    # Get event counts by type
    event_types = history_manager.get_token_event_types(token_id)
    for event_type in event_types:
        count = history_manager.get_event_count(token_id, event_type)
        logger.info(f"{event_type.name}: {count} events")
    
    # Calculate trade frequency (events per minute)
    if EventType.TOKEN_TRADE in event_types:
        freq = history_manager.get_event_frequency(
            token_id, 
            EventType.TOKEN_TRADE,
            window_size=timedelta(minutes=1)
        )
        logger.info(f"Trade frequency: {freq:.2f} trades/minute")
    
    # Get latest price
    if EventType.TOKEN_TRADE in event_types:
        latest_trade = history_manager.get_latest_event(token_id, EventType.TOKEN_TRADE)
        if latest_trade:
            price = latest_trade.data.get('price', 'N/A')
            logger.info(f"Latest price: ${price:.2f}")
    
    # Position history
    position_events = history_manager.get_events(
        token_id,
        event_type=EventType.POSITION_OPENED
    )
    for event in position_events:
        position_id = event.data.get('position_id')
        entry_price = event.data.get('entry_price')
        logger.info(f"Position opened: {position_id} at ${entry_price:.2f}")
    
    closed_events = history_manager.get_events(
        token_id,
        event_type=EventType.POSITION_CLOSED
    )
    for event in closed_events:
        position_id = event.data.get('position_id')
        exit_price = event.data.get('exit_price')
        logger.info(f"Position closed: {position_id} at ${exit_price:.2f}")


async def main():
    """Run the event history example."""
    try:
        # Create event bus
        event_bus = EventBus()
        event_bus.start_processing()
        
        # Create event history manager
        history_manager = EventHistoryManager(
            event_bus=event_bus,
            max_events_per_token=1000,
            default_retention_period=timedelta(hours=1),
            pruning_interval=timedelta(minutes=5)
        )
        history_manager.start()
        
        # Create mock publisher
        token_ids = [f"TOKEN_{i}" for i in range(1, 4)]  # Just 3 tokens for the example
        publisher = MockTokenPublisher(event_bus, token_ids)
        
        # Run simulation
        simulation_duration = 20  # seconds
        await publisher.start_simulation(duration_seconds=simulation_duration)
        
        # Wait for all events to be processed
        logger.info("Waiting for event processing to complete...")
        await asyncio.sleep(1)
        
        # Print statistics for each token
        for token_id in token_ids:
            print_token_statistics(history_manager, token_id)
        
        # Print overall stats
        logger.info("\n--- Overall Event Statistics ---")
        stats = history_manager.get_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Demonstrate pruning
        logger.info("\nTesting event pruning...")
        # Set a very short retention period for TOKEN_TRADE events
        history_manager.set_retention_period(
            EventType.TOKEN_TRADE,
            timedelta(seconds=1)  # Short period for demonstration
        )
        
        # Wait briefly to make trades "old"
        await asyncio.sleep(2)
        
        # Prune events
        pruned_count = history_manager.prune_events()
        logger.info(f"Pruned {pruned_count} events")
        
        # Show stats after pruning
        stats = history_manager.get_stats()
        logger.info(f"Events after pruning: {stats['current_events']}")
        
        # Clean up
        history_manager.stop()
        event_bus.stop_processing()
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Error in event history example: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main()) 