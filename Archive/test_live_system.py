#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Live System Test

This script runs a test of the trading system with live (or mock) data to verify
that each component works correctly. It sets up the complete trading system
and monitors interactions between components, verifying that data flows properly
through the system.
"""

import asyncio
import logging
import os
import sys
import time
import json
import random
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
import signal
import threading

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import core components
from src.core.events import (
    Event, EventType, EventPriority, EventBus, 
    BaseEventSubscriber, BaseEventPublisher, EventDispatcher,
    EventHistoryManager
)
from src.core.features import FeatureSystem, DefaultFeatureManager
from src.core.features.providers import PriceFeatureProvider, EnhancedPriceProvider
from src.core.features.signal_feature import PriceMomentumSignalFeature, VolumeSpikeTradingSignalFeature
from src.core.trading.trading_factory import TradingSystemFactory
from src.core.trading.token_monitor import TokenMonitorThreadPool
from src.core.ml import DefaultModelManager

# Import math module required for ActivityMetrics
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("live_system_test")

# Global flag for graceful shutdown
shutdown_requested = False

class ComponentVerifier(BaseEventSubscriber):
    """
    Verifies that components are working correctly by monitoring events.
    
    This class subscribes to various events and tracks which components
    have been verified to be working.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize component verifier.
        
        Args:
            event_bus: Event bus to subscribe to
        """
        super().__init__(event_bus)
        
        # Track components that have been verified
        self.verified_components = {
            "EventHistoryManager": False,
            "FeatureManager": False,
            "ModelManager": False,
            "GracefulExitManager": False,
            "ActivityAnalyzer": False
        }
        
        # Register event handlers
        self.register_handler(EventType.TOKEN_CREATED, self.handle_token_created)
        self.register_handler(EventType.TOKEN_TRADE, self.handle_token_trade)
        self.register_handler(EventType.FEATURE_UPDATE, self.handle_feature_update)
        self.register_handler(EventType.MODEL_PREDICTION, self.handle_model_prediction)
        self.register_handler(EventType.SYSTEM_STATUS, self.handle_system_status)
        
        # Track events by component
        self.events_by_component = {
            "EventHistoryManager": 0,
            "FeatureManager": 0,
            "ModelManager": 0,
            "GracefulExitManager": 0,
            "ActivityAnalyzer": 0
        }
        
        self.token_id_by_component = {}
        self.last_verification_time = time.time()
        
        logger.info("ComponentVerifier initialized")
    
    def __call__(self, event: Event) -> None:
        """
        Make the verifier callable directly by the event bus.
        
        Args:
            event: The event to process
        """
        # Delegate to the appropriate handler based on event type
        if event.event_type == EventType.TOKEN_CREATED:
            self.handle_token_created(event)
        elif event.event_type == EventType.TOKEN_TRADE:
            self.handle_token_trade(event)
        elif event.event_type == EventType.FEATURE_UPDATE:
            self.handle_feature_update(event)
        elif event.event_type == EventType.MODEL_PREDICTION:
            self.handle_model_prediction(event)
        elif event.event_type == EventType.SYSTEM_STATUS:
            self.handle_system_status(event)
    
    def handle_token_created(self, event: Event) -> None:
        """
        Handle token created event.
        
        This verifies that the EventHistoryManager is capturing events.
        """
        # Try to get the token_id from different possible field names
        token_id = event.data.get('token_id') or event.data.get('mint')
        if not token_id:
            return
        
        self.token_id_by_component["EventHistoryManager"] = token_id
        self.events_by_component["EventHistoryManager"] += 1
    
    def handle_token_trade(self, event: Event) -> None:
        """
        Handle token trade event.
        
        This verifies that tokens are trading and events are being captured.
        """
        # Try to get the token_id from different possible field names
        token_id = event.data.get('token_id') or event.data.get('mint')
        if not token_id:
            return
        
        # This is relevant for multiple components
        self.token_id_by_component["EventHistoryManager"] = token_id
        self.events_by_component["EventHistoryManager"] += 1
    
    def handle_feature_update(self, event: Event) -> None:
        """
        Handle feature update event.
        
        This verifies that the FeatureManager is calculating features.
        """
        # Try to get the token_id from different possible field names
        token_id = event.data.get('token_id') or event.data.get('mint')
        if not token_id:
            return
        
        self.token_id_by_component["FeatureManager"] = token_id
        self.events_by_component["FeatureManager"] += 1
    
    def handle_model_prediction(self, event: Event) -> None:
        """
        Handle model prediction event.
        
        This verifies that the ModelManager is making predictions.
        """
        # Try to get the token_id from different possible field names
        token_id = event.data.get('token_id') or event.data.get('mint')
        if not token_id:
            return
        
        self.token_id_by_component["ModelManager"] = token_id
        self.events_by_component["ModelManager"] += 1
    
    def handle_system_status(self, event: Event) -> None:
        """
        Handle system status event.
        
        This verifies that system components are publishing status events.
        """
        action = event.data.get('action', '')
        
        if 'graceful_exit' in action:
            self.events_by_component["GracefulExitManager"] += 1
        
        if action == 'activity_level_changed':
            token_id = event.data.get('token_id')
            if token_id:
                self.token_id_by_component["ActivityAnalyzer"] = token_id
                self.events_by_component["ActivityAnalyzer"] += 1
    
    def verify_components(self, event_history_manager, feature_manager, model_manager) -> Dict[str, bool]:
        """
        Verify that components are working correctly.
        
        This method checks various conditions to verify that each component
        is working as expected.
        
        Args:
            event_history_manager: EventHistoryManager instance
            feature_manager: FeatureManager instance
            model_manager: ModelManager instance
            
        Returns:
            Dictionary mapping component names to verification status
        """
        now = time.time()
        
        # Only run verification once per 5 seconds to avoid spam
        if now - self.last_verification_time < 5:
            return self.verified_components
            
        self.last_verification_time = now
        
        # Verify EventHistoryManager
        token_id = self.token_id_by_component.get("EventHistoryManager")
        if token_id and not self.verified_components["EventHistoryManager"]:
            # Check if events are being recorded
            if event_history_manager:
                try:
                    events = event_history_manager.get_events(token_id)
                    if events:
                        self.verified_components["EventHistoryManager"] = True
                        logger.info("✅ EventHistoryManager verified - successfully storing events")
                except Exception as e:
                    logger.error(f"Error verifying EventHistoryManager: {e}")
        
        # Verify FeatureManager
        token_id = self.token_id_by_component.get("FeatureManager")
        if token_id and not self.verified_components["FeatureManager"]:
            if feature_manager:
                try:
                    features = feature_manager.get_all_features(token_id)
                    if features:
                        self.verified_components["FeatureManager"] = True
                        logger.info(f"✅ FeatureManager verified - calculated features: {list(features.keys())}")
                except Exception as e:
                    logger.error(f"Error verifying FeatureManager: {e}")
        
        # Verify ModelManager
        if self.events_by_component["ModelManager"] > 0 and not self.verified_components["ModelManager"]:
            if model_manager:
                try:
                    models = model_manager.list_models()
                    if models:
                        performance = model_manager.get_model_performance(models[0])
                        if performance:
                            self.verified_components["ModelManager"] = True
                            logger.info(f"✅ ModelManager verified - found models: {models}")
                except Exception as e:
                    logger.error(f"Error verifying ModelManager: {e}")
        
        # For components that are harder to directly verify, we rely on event counts
        if self.events_by_component["GracefulExitManager"] > 0 and not self.verified_components["GracefulExitManager"]:
            self.verified_components["GracefulExitManager"] = True
            logger.info("✅ GracefulExitManager verified - received exit events")
        
        if self.events_by_component["ActivityAnalyzer"] > 0 and not self.verified_components["ActivityAnalyzer"]:
            self.verified_components["ActivityAnalyzer"] = True
            logger.info("✅ ActivityAnalyzer verified - tracking token activity")
        
        return self.verified_components
    
    def all_verified(self) -> bool:
        """Check if all components have been verified."""
        return all(self.verified_components.values())
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get a summary of component verification status."""
        return {
            "verified": self.verified_components.copy(),
            "events": self.events_by_component.copy(),
            "all_verified": self.all_verified()
        }


class SystemMonitor(threading.Thread):
    """
    Monitor thread that periodically checks system status.
    """
    
    def __init__(self, 
                 verifier: ComponentVerifier,
                 event_history_manager: Optional[EventHistoryManager] = None,
                 feature_manager: Optional[DefaultFeatureManager] = None,
                 model_manager: Optional[DefaultModelManager] = None,
                 check_interval: float = 5.0):
        """
        Initialize system monitor.
        
        Args:
            verifier: ComponentVerifier instance
            event_history_manager: EventHistoryManager to monitor
            feature_manager: FeatureManager to monitor
            model_manager: ModelManager to monitor
            check_interval: Interval in seconds between status checks
        """
        super().__init__(name="SystemMonitor")
        self.verifier = verifier
        self.event_history_manager = event_history_manager
        self.feature_manager = feature_manager
        self.model_manager = model_manager
        self.check_interval = check_interval
        self.daemon = True
        self.stop_event = threading.Event()
    
    def run(self) -> None:
        """Run the monitoring thread."""
        logger.info("System monitor started")
        
        start_time = time.time()
        
        while not self.stop_event.is_set():
            # Check component status
            status = self.verifier.verify_components(
                self.event_history_manager,
                self.feature_manager, 
                self.model_manager
            )
            
            # Check if all components are verified
            if self.verifier.all_verified():
                logger.info("🎉 All components verified successfully!")
                # Wait longer between checks once all verified
                self.check_interval = 30.0
            
            # Print summary periodically
            elapsed = time.time() - start_time
            logger.info(f"System running for {elapsed:.1f}s - Component verification status:")
            for component, verified in status.items():
                logger.info(f"  {'✅' if verified else '❌'} {component}")
            
            # Sleep until next check
            self.stop_event.wait(self.check_interval)
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.stop_event.set()
        self.join(timeout=5.0)
        logger.info("System monitor stopped")


# Setup signal handlers for graceful shutdown
def signal_handler(sig, frame):
    """Handle signal interrupts."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown signal received. Cleaning up...")


def patch_activity_analyzer():
    """Patch the ActivityAnalyzer class to add a __call__ method for direct event handling."""
    from src.core.trading.activity_analyzer import ActivityAnalyzer
    
    # Only add the method if it doesn't already exist
    if not hasattr(ActivityAnalyzer, '__call__'):
        def __call__(self, event):
            """Make the analyzer callable directly by the event bus."""
            try:
                if event.event_type == EventType.TOKEN_CREATED:
                    self._handle_token_created(event)
                    logger.debug(f"ActivityAnalyzer processed TOKEN_CREATED event for {event.data.get('token_id') or event.data.get('mint')}")
                elif event.event_type == EventType.TOKEN_TRADE:
                    self._handle_token_trade(event)
                    logger.debug(f"ActivityAnalyzer processed TOKEN_TRADE event for {event.data.get('token_id') or event.data.get('mint')}")
            except Exception as e:
                logger.error(f"Error in ActivityAnalyzer.__call__: {e}")
        
        # Add the method to the class
        ActivityAnalyzer.__call__ = __call__
        logger.info("Patched ActivityAnalyzer with __call__ method")


async def main():
    """Main function to run the live system test."""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Patch ActivityAnalyzer with __call__ method
        patch_activity_analyzer()
        
        # Create event bus
        event_bus = EventBus(async_processing=True)
        event_bus.start_processing()
        logger.info("Event bus started")
        
        # Create event history manager
        event_history_manager = EventHistoryManager(
            event_bus=event_bus,
            max_events_per_token=1000,
            default_retention_period=timedelta(hours=1),
            pruning_interval=timedelta(minutes=5)
        )
        event_history_manager.start()
        logger.info("EventHistoryManager started")
        
        # Create component verifier
        verifier = ComponentVerifier(event_bus)
        
        # Create data source client - using MockWebSocketClient for testing
        # Import from the actual example file path
        import sys
        import os
        from pathlib import Path
        
        # Add the example directory to the path to ensure we can import from run_sol_trading
        example_path = Path(__file__).resolve().parent
        if str(example_path) not in sys.path:
            sys.path.insert(0, str(example_path))
            
        from run_sol_trading import DataFeedManager, MockWebSocketClient
        
        # Create a data feed manager
        data_feed_manager = DataFeedManager(event_bus)
        
        # Create mock client and connect it to data feed manager
        data_client = MockWebSocketClient(event_bus, data_feed_manager)
        
        # Define a price fetcher function
        async def price_fetcher(token_id: str) -> float:
            """Fetch the current price for a token."""
            token_data = data_feed_manager.get_token_data(token_id)
            if token_data and 'price' in token_data:
                return token_data['price']
            return 0.0
        
        # Prepare configuration for the factory
        factory_config = {
            'initial_capital': 1000.0,
            'price_history_size': 100,
            'momentum_threshold': 0.05,
            'momentum_sensitivity': 1.0,
            'use_volume_spike_signal': True
        }
        
        # Build the complete trading system using the factory
        logger.info("Creating complete trading system with factory")
        trading_system = TradingSystemFactory.create_complete_trading_system(
            event_bus=event_bus,
            data_feed_manager=data_feed_manager,
            price_fetcher=price_fetcher,
            config=factory_config,
            subscription_keys=None
        )
        
        # Extract components from the trading system
        feature_manager = trading_system.get('feature_manager')
        model_manager = trading_system.get('model_manager')
        activity_analyzer = trading_system.get('activity_analyzer')
        
        # Manually register the activity analyzer to handle events through the __call__ method
        if activity_analyzer:
            event_bus.unsubscribe_component(activity_analyzer, [EventType.TOKEN_CREATED, EventType.TOKEN_TRADE])
            event_bus.subscribe(EventType.TOKEN_CREATED, activity_analyzer)
            event_bus.subscribe(EventType.TOKEN_TRADE, activity_analyzer)
            logger.info("Re-registered ActivityAnalyzer with direct event subscriptions")
        
        # Start the system monitor
        monitor = SystemMonitor(
            verifier=verifier,
            event_history_manager=event_history_manager,
            feature_manager=feature_manager,
            model_manager=model_manager
        )
        monitor.start()
        
        # Connect to data source
        logger.info("Starting data streaming")
        success = await data_client.start_streaming()
        if not success:
            logger.error("Failed to start data streaming. Exiting.")
            return 1
        
        # Run for a set duration or until shutdown requested
        logger.info("System running. Press Ctrl+C to exit.")
        test_duration = 120  # Run for 2 minutes
        start_time = time.time()
        
        while not shutdown_requested and (time.time() - start_time) < test_duration:
            await asyncio.sleep(1)
            
            # Check components verification status every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                verification = verifier.get_verification_summary()
                if verification["all_verified"]:
                    logger.info("All components verified, test successful!")
        
        # If we have a graceful exit manager, test it
        if 'graceful_exit_manager' in trading_system:
            logger.info("Testing GracefulExitManager...")
            trading_system['graceful_exit_manager'].begin_graceful_exit()
            
            # Wait for exit to complete
            await asyncio.sleep(5)
            
            # Get exit stats
            exit_stats = trading_system['graceful_exit_manager'].get_exit_stats()
            logger.info(f"GracefulExitManager stats: {exit_stats}")
            
        # Final verification check
        verification = verifier.get_verification_summary()
        verified_count = sum(1 for v in verification["verified"].values() if v)
        total_count = len(verification["verified"])
        
        logger.info(f"Test completed. Verified {verified_count}/{total_count} components.")
        for component, verified in verification["verified"].items():
            logger.info(f"  {'✅' if verified else '❌'} {component}")
        
        # Test result
        if verified_count == total_count:
            logger.info("🎉 All tests passed! Trading system components are working correctly.")
        else:
            missing = [c for c, v in verification["verified"].items() if not v]
            logger.warning(f"⚠️ Some components not verified: {', '.join(missing)}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error during system test: {e}", exc_info=True)
        return 1
        
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        
        if 'data_client' in locals() and data_client:
            await data_client.stop_streaming()
            
        if 'event_history_manager' in locals() and event_history_manager:
            event_history_manager.stop()
            event_history_manager.cleanup()
            
        if 'trading_system' in locals() and trading_system:
            # Shutdown all components
            TradingSystemFactory.shutdown_trading_system(trading_system)
            
        if 'event_bus' in locals() and event_bus:
            event_bus.stop_processing()
            
        if 'monitor' in locals() and monitor:
            monitor.stop()
            
        logger.info("Cleanup complete")


if __name__ == "__main__":
    try:
        # Run the async main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 