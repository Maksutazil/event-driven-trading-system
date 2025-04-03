#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token Monitor Module

This module provides thread-based monitoring of tokens for the trading system.
It allows for concurrent monitoring of multiple tokens with thread prioritization,
health monitoring, and performance metrics.
"""

import logging
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

from src.core.events import EventBus, EventType, BaseEventPublisher
from src.core.features.interfaces import FeatureConsumer
from src.core.events.base import EventHandlerWrapper

# Note: DataFeedManager and FeatureSystem are passed as parameters to the classes
# rather than imported directly to avoid circular imports

logger = logging.getLogger(__name__)


class ThreadStatus(Enum):
    """Status of a token monitoring thread."""
    STARTING = 0
    RUNNING = 1
    PAUSED = 2
    STOPPING = 3
    STOPPED = 4
    FAILED = 5


class TokenMonitorThread(threading.Thread, BaseEventPublisher, FeatureConsumer):
    """
    Thread responsible for monitoring a specific token.
    
    Each thread is responsible for:
    1. Getting token data from the data feed
    2. Computing features for the token
    3. Publishing events for significant changes
    4. Reporting its health status
    
    Implements FeatureConsumer to receive feature updates directly.
    """
    
    def __init__(self, 
                 token_id: str,
                 event_bus: EventBus,
                 data_feed_manager: Any,  # DataFeedManager
                 feature_system: Any,     # FeatureSystem (avoiding circular import)
                 interval: float = 1.0,
                 priority: int = 1):
        """
        Initialize a token monitoring thread.
        
        Args:
            token_id: ID of the token to monitor
            event_bus: EventBus for communication
            data_feed_manager: Manager for data access
            feature_system: System for computing features
            interval: Monitoring interval in seconds
            priority: Thread priority (higher = more important)
        """
        threading.Thread.__init__(self)
        BaseEventPublisher.__init__(self, event_bus)  # Remove source parameter
        
        self.token_id = token_id
        self.data_feed_manager = data_feed_manager
        self.feature_system = feature_system
        self.interval = interval
        self.priority = priority
        self.source = f"monitor_{token_id}"  # Store source as an instance variable instead
        
        # Thread control
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self.daemon = True  # Thread will exit when main program exits
        
        # State tracking
        self.status = ThreadStatus.STARTING
        self.last_run_time = None
        self.last_data_time = None
        self.run_count = 0
        self.error_count = 0
        self.last_error = None
        
        # Feature cache for latest values
        self._feature_cache = {}
        self._feature_lock = threading.RLock()
        
        # Register as a feature consumer
        if self.feature_system:
            self.feature_system.register_consumer(self)
        
        logger.info(f"Initialized monitoring thread for token: {token_id}, priority: {priority}")
    
    def get_required_features(self) -> List[str]:
        """
        Get the list of features this consumer requires.
        
        Returns:
            List[str]: List of required feature names
        """
        # Return all features that might be needed for monitoring
        # This can be customized based on actual needs
        return [
            "current_price",
            "price_change_pct_5m",
            "price_change_pct_15m",
            "volume_5m",
            "ma_5m",
            "ma_15m",
            "volatility_5m",
            "rsi_14",
            "price_momentum_signal"
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
        # Only handle updates for our token
        if token_id != self.token_id:
            return
        
        # Update our feature cache
        with self._feature_lock:
            self._feature_cache[feature_name] = value
            
        # Special handling for signal features that might trigger immediate actions
        if feature_name.endswith('_signal') and isinstance(value, (int, float)):
            # If it's a trading signal with significant value, publish an event
            if abs(value) >= 0.7:  # Threshold can be adjusted
                signal_type = "BUY" if value > 0 else "SELL"
                logger.info(f"Significant {signal_type} signal ({value:.4f}) detected for {token_id}")
                
                # Publish signal event
                self.publish_event(
                    EventType.TRADE_SIGNAL,
                    {
                        'token_id': token_id,
                        'signal_name': feature_name,
                        'signal_value': value,
                        'signal_type': signal_type,
                        'timestamp': time.time()
                    }
                )
    
    def publish_event(self, event_type: EventType, data: Dict[str, Any], **kwargs) -> None:
        """
        Override of BaseEventPublisher.publish_event to include the source parameter.
        
        Args:
            event_type: Type of event to publish
            data: Event data payload
            **kwargs: Additional keyword arguments
        """
        # Include the source parameter from our instance variable
        kwargs['source'] = self.source
        super().publish_event(event_type, data, **kwargs)
    
    def run(self):
        """Main thread execution loop."""
        try:
            # Subscribe to token data
            subscription_results = self.data_feed_manager.subscribe_token(self.token_id)
            if not any(subscription_results.values()):
                logger.error(f"Failed to subscribe to token {self.token_id} in any data feed")
                self.status = ThreadStatus.FAILED
                self._publish_status_update()
                return
            
            logger.info(f"Starting monitoring for token {self.token_id}")
            self.status = ThreadStatus.RUNNING
            self._publish_status_update()
            
            while not self._stop_event.is_set():
                start_time = time.time()
                
                # Check if paused
                if self._paused.is_set():
                    if self.status != ThreadStatus.PAUSED:
                        self.status = ThreadStatus.PAUSED
                        self._publish_status_update()
                    time.sleep(self.interval)
                    continue
                
                if self.status != ThreadStatus.RUNNING:
                    self.status = ThreadStatus.RUNNING
                    self._publish_status_update()
                
                try:
                    # Process token data
                    self._process_token()
                    self.run_count += 1
                    self.last_run_time = time.time()
                except Exception as e:
                    self.error_count += 1
                    self.last_error = str(e)
                    logger.error(f"Error monitoring token {self.token_id}: {e}", exc_info=True)
                    
                    # If too many errors, mark thread as failed
                    if self.error_count > 10:
                        self.status = ThreadStatus.FAILED
                        self._publish_status_update()
                        break
                
                # Sleep for the interval, accounting for processing time
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.interval - elapsed)
                time.sleep(sleep_time)
            
            # Clean up when stopped
            self._cleanup()
            
        except Exception as e:
            logger.error(f"Fatal error in monitor thread for {self.token_id}: {e}", exc_info=True)
            self.status = ThreadStatus.FAILED
            self.last_error = str(e)
            self._publish_status_update()
        finally:
            if self.status != ThreadStatus.STOPPED:
                self.status = ThreadStatus.STOPPED
                self._publish_status_update()
            logger.info(f"Monitor thread for token {self.token_id} has exited")
    
    def _process_token(self):
        """Process the token data and compute features."""
        # Get latest data and compute features
        logger.debug(f"Processing token {self.token_id}")
        
        try:
            # 1. Get latest data from data feeds
            token_data = self.data_feed_manager.get_token_data(self.token_id)
            if not token_data:
                logger.warning(f"No data available for token {self.token_id}")
                return
                
            # Extract the current price and other relevant data
            current_price = token_data.get('price')
            if current_price is None or current_price <= 0:
                logger.warning(f"Invalid price data for token {self.token_id}: {current_price}")
                return
                
            # Set up context for feature computation
            timestamp = token_data.get('timestamp', time.time())
            if isinstance(timestamp, (int, float)) and timestamp > 1000000000000:  # If in milliseconds
                timestamp = timestamp / 1000  # Convert to seconds
                
            timepoint = datetime.fromtimestamp(timestamp)
            
            context = {
                'token_id': self.token_id,
                'price': current_price,
                'timestamp': timepoint,
                'volume': token_data.get('volume', 0),
                'market_cap': token_data.get('market_cap', 0),
                **token_data  # Include all other token data
            }
            
            # 2. Get features from feature system (will trigger FeatureConsumer callbacks)
            features = self.feature_system.get_features_for_token(self.token_id)
            
            # If no features from system, try to compute them directly
            if not features:
                logger.debug(f"No cached features for {self.token_id}, computing directly")
                features = self.feature_system.compute_features(context)
            
            if not features:
                logger.warning(f"No features computed for token {self.token_id}")
                return
                
            # 3. Publish events with computed features
            event_data = {
                'token_id': self.token_id,
                'timestamp': timestamp,
                'price': current_price,
                'monitor_run_count': self.run_count,
                'priority': self.priority,
                'features': features
            }
            
            self.publish_event(EventType.TOKEN_UPDATED, event_data)
            logger.debug(f"Published TOKEN_UPDATED event for {self.token_id} with {len(features)} features")
            
            # Also publish TOKEN_TRADE event if this is a new trade
            if token_data.get('is_trade', False):
                trade_data = {
                    'token_id': self.token_id,
                    'timestamp': timestamp,
                    'price': current_price,
                    'volume': token_data.get('volume', 0),
                    'side': token_data.get('side', 'unknown'),
                    'raw_data': token_data
                }
                self.publish_event(EventType.TOKEN_TRADE, trade_data)
                logger.debug(f"Published TOKEN_TRADE event for {self.token_id}")
                
            self.last_data_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing token {self.token_id}: {e}", exc_info=True)
            self.error_count += 1
    
    def _cleanup(self):
        """Clean up resources when thread is stopping."""
        try:
            # Unregister as feature consumer
            if self.feature_system:
                self.feature_system.unregister_consumer(self)
                
            # Unsubscribe from token data
            self.data_feed_manager.unsubscribe_token(self.token_id)
            logger.info(f"Unsubscribed from token {self.token_id}")
        except Exception as e:
            logger.error(f"Error during cleanup for token {self.token_id}: {e}")
    
    def _publish_status_update(self):
        """Publish thread status update event."""
        self.publish_event(
            EventType.GENERIC,
            {
                "action": "thread_status",
                "token_id": self.token_id,
                "status": self.status.name,
                "run_count": self.run_count,
                "error_count": self.error_count,
                "last_error": self.last_error,
                "last_run_time": self.last_run_time,
                "priority": self.priority
            }
        )
    
    def stop(self):
        """Signal the thread to stop."""
        logger.info(f"Stopping monitor thread for token {self.token_id}")
        self.status = ThreadStatus.STOPPING
        self._stop_event.set()
        self._paused.clear()  # Unpause if paused
        self._publish_status_update()
    
    def pause(self):
        """Pause the thread processing."""
        logger.info(f"Pausing monitor thread for token {self.token_id}")
        self._paused.set()
    
    def resume(self):
        """Resume the thread processing if paused."""
        logger.info(f"Resuming monitor thread for token {self.token_id}")
        self._paused.clear()
    
    def is_running(self) -> bool:
        """Check if the thread is currently running."""
        return self.is_alive() and self.status == ThreadStatus.RUNNING and not self._paused.is_set()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the thread."""
        return {
            "token_id": self.token_id,
            "status": self.status.name,
            "is_alive": self.is_alive(),
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_run_time": self.last_run_time,
            "last_data_time": self.last_data_time,
            "priority": self.priority
        }


class TokenMonitorThreadPool(BaseEventPublisher):
    """
    Manages a pool of threads for monitoring multiple tokens.
    
    This class is responsible for:
    1. Creating and managing threads for token monitoring
    2. Ensuring thread safety for shared resources
    3. Monitoring thread health and restarting failed threads
    4. Prioritizing tokens based on importance
    5. Tracking performance metrics
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 data_feed_manager: Any,  # DataFeedManager
                 feature_system: Any,     # FeatureSystem (avoiding circular import)
                 max_threads: int = 10,
                 monitor_interval: float = 1.0,
                 health_check_interval: float = 30.0,
                 trading_engine = None,   # Add trading engine parameter
                 activity_analyzer = None): # ActivityAnalyzer for token activity tracking
        """
        Initialize the token monitor thread pool.
        
        Args:
            event_bus: EventBus for communication
            data_feed_manager: Manager for data feeds
            feature_system: System for computing features
            max_threads: Maximum number of concurrent threads
            monitor_interval: Base interval for token monitoring in seconds
            health_check_interval: Interval for health checking threads in seconds
            trading_engine: Optional trading engine for processing trades
            activity_analyzer: Optional activity analyzer for adjusting priorities
        """
        super().__init__(event_bus)  # BaseEventPublisher init without source
        
        self.data_feed_manager = data_feed_manager
        self.feature_system = feature_system
        self.max_threads = max_threads
        self.monitor_interval = monitor_interval
        self.health_check_interval = health_check_interval
        self.trading_engine = trading_engine  # Store trading engine reference
        self.activity_analyzer = activity_analyzer  # Store activity analyzer reference
        
        # Thread management
        self.active_threads: Dict[str, TokenMonitorThread] = {}
        self.thread_lock = threading.RLock()
        
        # Priority queue for tokens waiting to be monitored
        self.token_queue = queue.PriorityQueue()
        
        # Performance metrics
        self.stats = {
            "total_tokens_added": 0,
            "total_tokens_removed": 0,
            "total_thread_restarts": 0,
            "total_priority_adjustments": 0,
            "failed_restarts": 0,
            "start_time": time.time()
        }
        
        # Health monitoring thread
        self._health_monitor = None
        self._priority_adjuster = None
        self._stop_event = threading.Event()
        
        logger.info(f"Initialized TokenMonitorThreadPool with max_threads={max_threads}")
        
        # Register event handlers
        if self.event_bus:
            self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register event handlers for token events."""
        # Use the standardized EventHandlerWrapper
        from src.core.events.base import EventHandlerWrapper
        
        # Register for token events
        token_created_handler = EventHandlerWrapper(self._handle_token_created_event)
        self.event_bus.subscribe(EventType.TOKEN_CREATED, token_created_handler)
        
        token_trade_handler = EventHandlerWrapper(self._handle_token_trade_event)
        self.event_bus.subscribe(EventType.TOKEN_TRADE, token_trade_handler)
        
        token_updated_handler = EventHandlerWrapper(self._handle_token_updated_event)
        self.event_bus.subscribe(EventType.TOKEN_UPDATED, token_updated_handler)
    
    def _handle_token_created_event(self, event):
        """Handle token created events."""
        try:
            token_id = event.data.get('token_id')
            if not token_id:
                logger.warning("Received TOKEN_CREATED event without token_id")
                return
            
            # Add token to monitoring if not already monitored
            if token_id not in self.active_threads:
                metadata = {
                    'token_name': event.data.get('token_name'),
                    'token_symbol': event.data.get('token_symbol'),
                    'raw_data': event.data.get('raw_data', {})
                }
                self.add_token(token_id, priority=1, metadata=metadata)
            
            # Forward to trading engine if available
            if self.trading_engine:
                self.trading_engine.add_token(token_id, metadata=event.data)
                logger.info(f"Added token {token_id} to trading engine")
            else:
                logger.warning(f"Trading engine not available for token {token_id}")
        
        except Exception as e:
            logger.error(f"Error handling TOKEN_CREATED event: {e}", exc_info=True)
    
    def _handle_token_trade_event(self, event):
        """Handle token trade events."""
        try:
            token_id = event.data.get('token_id')
            if not token_id:
                logger.warning("Received TOKEN_TRADE event without token_id")
                return
            
            # Forward to trading engine if available
            if self.trading_engine:
                self.trading_engine.update_token_trade(token_id, event.data)
            else:
                logger.warning(f"Trading engine not available for token {token_id}")
        
        except Exception as e:
            logger.error(f"Error handling TOKEN_TRADE event: {e}", exc_info=True)
    
    def _handle_token_updated_event(self, event):
        """Handle token updated events."""
        try:
            token_id = event.data.get('token_id')
            if not token_id:
                logger.warning("Received TOKEN_UPDATED event without token_id")
                return
            
            # Forward to trading engine if available
            if self.trading_engine:
                self.trading_engine.update_token_metadata(token_id, event.data)
            else:
                logger.warning(f"Trading engine not available for token {token_id}")
        
        except Exception as e:
            logger.error(f"Error handling TOKEN_UPDATED event: {e}", exc_info=True)
    
    def start(self):
        """Start the thread pool and health monitoring."""
        logger.info("Starting TokenMonitorThreadPool")
        
        # Start health monitoring thread
        self._health_monitor = threading.Thread(
            target=self._monitor_thread_health,
            daemon=True,
            name="HealthMonitor"
        )
        self._health_monitor.start()
        
        # Publish started event
        self.publish_event(
            EventType.GENERIC,
            {
                "action": "thread_pool_started",
                "max_threads": self.max_threads,
                "monitor_interval": self.monitor_interval
            },
            source="TokenMonitorThreadPool"
        )
    
    def stop(self):
        """Stop all monitoring threads and the thread pool."""
        logger.info("Stopping TokenMonitorThreadPool")
        
        # Signal health monitor to stop
        self._stop_event.set()
        
        # Stop all active threads
        with self.thread_lock:
            for token_id, thread in list(self.active_threads.items()):
                self._stop_thread(token_id)
        
        # Wait for health monitor to exit
        if self._health_monitor and self._health_monitor.is_alive():
            self._health_monitor.join(timeout=5.0)
        
        # Empty the token queue
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
                self.token_queue.task_done()
            except queue.Empty:
                break
        
        # Publish stopped event
        self.publish_event(
            EventType.GENERIC,
            {
                "action": "thread_pool_stopped",
                "stats": self.get_stats()
            },
            source="TokenMonitorThreadPool"
        )
        
        logger.info("TokenMonitorThreadPool stopped")
    
    def add_token(self, token_id: str, priority: int = 1, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a token to be monitored.
        
        Args:
            token_id: ID of the token to monitor
            priority: Priority of the token (higher = more important)
            metadata: Additional metadata for the token
            
        Returns:
            bool: True if the token was added, False if it was already being monitored
        """
        logger.info(f"Adding token {token_id} with priority {priority}")
        
        with self.thread_lock:
            # Check if already monitoring this token
            if token_id in self.active_threads:
                logger.warning(f"Token {token_id} is already being monitored")
                return False
            
            self.stats["total_tokens_added"] += 1
            
            # Add to the thread pool if space available
            if len(self.active_threads) < self.max_threads:
                return self._start_monitoring_thread(token_id, priority, metadata)
            
            # Otherwise add to the priority queue
            logger.info(f"Thread pool full, adding token {token_id} to queue")
            self.token_queue.put((-priority, token_id, metadata))  # Negative priority for highest-first
            
            return True
    
    def remove_token(self, token_id: str) -> bool:
        """
        Remove a token from monitoring.
        
        Args:
            token_id: ID of the token to stop monitoring
            
        Returns:
            bool: True if the token was removed, False if it wasn't being monitored
        """
        logger.info(f"Removing token {token_id} from monitoring")
        
        with self.thread_lock:
            # Remove from active threads
            if token_id in self.active_threads:
                self._stop_thread(token_id)
                self.stats["total_tokens_removed"] += 1
                
                # Check if any tokens are waiting in the queue
                self._process_queue()
                return True
            
            # Check if in the queue
            queue_items = []
            found = False
            
            # Extract all items from the queue
            while not self.token_queue.empty():
                try:
                    item = self.token_queue.get_nowait()
                    if item[1] != token_id:  # Not the one we're looking for
                        queue_items.append(item)
                    else:
                        found = True
                    self.token_queue.task_done()
                except queue.Empty:
                    break
            
            # Put other items back
            for item in queue_items:
                self.token_queue.put(item)
            
            if found:
                self.stats["total_tokens_removed"] += 1
                return True
            
            logger.warning(f"Token {token_id} was not being monitored")
            return False
    
    def _start_monitoring_thread(self, token_id: str, priority: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start a new thread to monitor a token.
        
        Args:
            token_id: ID of the token to monitor
            priority: Priority of the token
            metadata: Additional metadata for the token
            
        Returns:
            bool: True if the thread was started successfully
        """
        try:
            # Calculate interval based on priority
            interval = self._calculate_interval(priority)
            
            # Create and start the thread
            thread = TokenMonitorThread(
                token_id=token_id,
                event_bus=self.event_bus,
                data_feed_manager=self.data_feed_manager,
                feature_system=self.feature_system,
                interval=interval,
                priority=priority
            )
            thread.start()
            
            # Store in active threads
            self.active_threads[token_id] = thread
            
            # Publish event
            self.publish_event(
                EventType.GENERIC,
                {
                    "action": "token_monitoring_started",
                    "token_id": token_id,
                    "priority": priority,
                    "interval": interval,
                    "metadata": metadata
                }
            )
            
            logger.info(f"Started monitoring thread for token {token_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring thread for token {token_id}: {e}", exc_info=True)
            return False
    
    def _stop_thread(self, token_id: str):
        """
        Stop a monitoring thread.
        
        Args:
            token_id: ID of the token to stop monitoring
        """
        if token_id in self.active_threads:
            thread = self.active_threads[token_id]
            thread.stop()
            
            # Wait for thread to exit (with timeout)
            if thread.is_alive():
                thread.join(timeout=2.0)
            
            # Remove from active threads
            del self.active_threads[token_id]
            
            # Publish event
            self.publish_event(
                EventType.GENERIC,
                {
                    "action": "token_monitoring_stopped",
                    "token_id": token_id
                }
            )
            
            logger.info(f"Stopped monitoring thread for token {token_id}")
    
    def _process_queue(self):
        """Process the token queue to start monitoring highest priority tokens."""
        while len(self.active_threads) < self.max_threads and not self.token_queue.empty():
            try:
                # Get highest priority token
                neg_priority, token_id, metadata = self.token_queue.get_nowait()
                priority = -neg_priority  # Convert back to positive
                
                # Start monitoring
                self._start_monitoring_thread(token_id, priority, metadata)
                self.token_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing token queue: {e}", exc_info=True)
                break
    
    def _monitor_thread_health(self):
        """Monitor the health of active threads and restart failed ones."""
        logger.info("Thread health monitoring started")
        
        while not self._stop_event.is_set():
            try:
                # Check and manage thread health
                with self.thread_lock:
                    for token_id, thread in list(self.active_threads.items()):
                        # Check if thread is still alive
                        if not thread.is_alive():
                            logger.warning(f"Thread for token {token_id} is not alive, restarting")
                            self._restart_thread(token_id, thread.priority)
                            continue
                        
                        # Check if thread has too many errors
                        if thread.status == ThreadStatus.FAILED:
                            logger.warning(f"Thread for token {token_id} has failed, restarting")
                            self._restart_thread(token_id, thread.priority)
                            continue
                        
                        # Check if thread hasn't run recently
                        if thread.last_run_time > 0:
                            time_since_last_run = time.time() - thread.last_run_time
                            expected_interval = thread.interval * 5  # Allow 5x interval as buffer
                            
                            if time_since_last_run > max(30.0, expected_interval):
                                logger.warning(f"Thread for token {token_id} hasn't run in {time_since_last_run:.1f}s, restarting")
                                self._restart_thread(token_id, thread.priority)
                
                # Adjust priorities based on activity analyzer if available
                if self.activity_analyzer:
                    self._adjust_thread_priorities()
                
                # Process queue if space available
                self._process_queue()
                
                # Publish health status periodically
                self.publish_event(
                    EventType.GENERIC,
                    {
                        "action": "thread_pool_health",
                        "active_threads": len(self.active_threads),
                        "queued_tokens": self.token_queue.qsize(),
                        "stats": self.get_stats()
                    },
                    source="TokenMonitorThreadPool"
                )
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}", exc_info=True)
            
            # Sleep until next check
            for _ in range(int(self.health_check_interval / 0.1)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
        
        logger.info("Thread health monitoring stopped")
    
    def _restart_thread(self, token_id: str, priority: int) -> bool:
        """
        Restart a monitoring thread.
        
        Args:
            token_id: ID of the token to restart
            priority: Priority of the token
            
        Returns:
            bool: True if restart was successful
        """
        try:
            # Stop the existing thread if it exists
            if token_id in self.active_threads:
                self._stop_thread(token_id)
            
            # Start a new thread
            success = self._start_monitoring_thread(token_id, priority)
            
            if success:
                self.stats["total_thread_restarts"] += 1
                logger.info(f"Successfully restarted thread for token {token_id}")
            else:
                self.stats["failed_restarts"] += 1
                logger.error(f"Failed to restart thread for token {token_id}")
            
            return success
            
        except Exception as e:
            self.stats["failed_restarts"] += 1
            logger.error(f"Error restarting thread for token {token_id}: {e}", exc_info=True)
            return False
    
    def _adjust_thread_priorities(self) -> None:
        """
        Adjust thread priorities based on token activity levels.
        
        This method uses the ActivityAnalyzer to get recommended priorities
        for tokens and updates thread intervals accordingly.
        """
        if not self.activity_analyzer:
            return
        
        with self.thread_lock:
            adjustments_made = 0
            
            # Adjust priorities for active threads
            for token_id, thread in list(self.active_threads.items()):
                try:
                    # Get recommended priority from activity analyzer
                    recommended_priority = self.activity_analyzer.recommend_priority(token_id)
                    
                    # If priority is significantly different, adjust it
                    if abs(recommended_priority - thread.priority) > 0.5:
                        # Calculate new interval based on recommended priority
                        old_priority = thread.priority
                        old_interval = thread.interval
                        thread.priority = recommended_priority
                        thread.interval = self._calculate_interval(recommended_priority)
                        
                        logger.debug(f"Adjusted priority for {token_id}: {old_priority:.1f} -> {recommended_priority:.1f} "
                                   f"(interval: {old_interval:.2f}s -> {thread.interval:.2f}s)")
                        adjustments_made += 1
                
                    # Check if token has become inactive or abandoned and should be removed
                    lifecycle_state = self.activity_analyzer.get_token_lifecycle_state(token_id)
                    if lifecycle_state and lifecycle_state.name in ('INACTIVE', 'ABANDONED'):
                        inactive_time = self.activity_analyzer.token_activity[token_id].seconds_since_last_trade()
                        
                        # If inactive for more than 15 minutes (900 seconds), remove from monitoring
                        if inactive_time > 900:
                            logger.info(f"Removing inactive token {token_id} after {inactive_time:.1f}s of inactivity")
                            self._stop_thread(token_id)
                            self.stats["total_tokens_removed"] += 1
                
                except Exception as e:
                    logger.error(f"Error adjusting priority for token {token_id}: {e}", exc_info=True)
            
            # If we made adjustments, update stats
            if adjustments_made > 0:
                self.stats["total_priority_adjustments"] += adjustments_made
                logger.info(f"Adjusted priorities for {adjustments_made} tokens")
                
                # Process queue to start monitoring new tokens if space available
                self._process_queue()
    
    def _calculate_interval(self, priority: int) -> float:
        """
        Calculate monitoring interval based on priority.
        
        Higher priority tokens are monitored more frequently.
        
        Args:
            priority: Token priority (higher = more important)
            
        Returns:
            float: Monitoring interval in seconds
        """
        # Ensure priority is at least 1
        priority = max(1, priority)
        
        # Higher priority gets shorter interval, but never less than 0.1s
        interval = max(0.1, self.monitor_interval / priority)
        return interval
    
    def pause_token(self, token_id: str) -> bool:
        """
        Pause monitoring for a specific token.
        
        Args:
            token_id: ID of the token to pause
            
        Returns:
            bool: True if paused, False if not found
        """
        with self.thread_lock:
            if token_id in self.active_threads:
                self.active_threads[token_id].pause()
                return True
            return False
    
    def resume_token(self, token_id: str) -> bool:
        """
        Resume monitoring for a specific token.
        
        Args:
            token_id: ID of the token to resume
            
        Returns:
            bool: True if resumed, False if not found
        """
        with self.thread_lock:
            if token_id in self.active_threads:
                self.active_threads[token_id].resume()
                return True
            return False
    
    def update_priority(self, token_id: str, priority: int) -> bool:
        """
        Update the priority of a token.
        
        This may involve stopping and restarting the monitoring thread
        to adjust the monitoring interval.
        
        Args:
            token_id: ID of the token
            priority: New priority value
            
        Returns:
            bool: True if updated, False if not found
        """
        with self.thread_lock:
            # If actively monitoring, restart with new priority
            if token_id in self.active_threads:
                old_thread = self.active_threads[token_id]
                old_thread.stop()
                
                # Wait for thread to exit (with timeout)
                if old_thread.is_alive():
                    old_thread.join(timeout=2.0)
                
                # Remove from active threads
                del self.active_threads[token_id]
                
                # Start with new priority
                return self._start_monitoring_thread(token_id, priority)
            
            # If in queue, needs to be requeued with new priority
            queue_items = []
            found = False
            
            # Extract all items from the queue
            while not self.token_queue.empty():
                try:
                    item = self.token_queue.get_nowait()
                    if item[1] != token_id:  # Not the one we're updating
                        queue_items.append(item)
                    else:
                        # Update priority and re-add
                        found = True
                        queue_items.append((-priority, token_id, item[2]))
                    self.token_queue.task_done()
                except queue.Empty:
                    break
            
            # Put all items back
            for item in queue_items:
                self.token_queue.put(item)
            
            return found
    
    def get_token_status(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dict with status or None if not found
        """
        with self.thread_lock:
            if token_id in self.active_threads:
                return self.active_threads[token_id].get_status()
            
            # Check if in queue
            in_queue = any(
                item[1] == token_id 
                for item in list(self.token_queue.queue)
            )
            
            if in_queue:
                return {
                    "token_id": token_id,
                    "status": "QUEUED",
                    "is_alive": False
                }
            
            return None
    
    def get_all_token_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all monitored tokens.
        
        Returns:
            Dict mapping token IDs to their status
        """
        result = {}
        
        with self.thread_lock:
            # Get active thread statuses
            for token_id, thread in self.active_threads.items():
                result[token_id] = thread.get_status()
            
            # Add queued tokens
            for item in list(self.token_queue.queue):
                token_id = item[1]
                priority = -item[0]  # Convert back to positive
                result[token_id] = {
                    "token_id": token_id,
                    "status": "QUEUED",
                    "priority": priority,
                    "is_alive": False
                }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the thread pool.
        
        Returns:
            Dict with performance statistics
        """
        uptime = time.time() - self.stats["start_time"]
        
        with self.thread_lock:
            stats = {
                **self.stats,
                "uptime_seconds": uptime,
                "active_threads": len(self.active_threads),
                "queued_tokens": self.token_queue.qsize(),
                "max_threads": self.max_threads
            }
            
            # Calculate aggregate thread stats
            if self.active_threads:
                total_runs = sum(t.run_count for t in self.active_threads.values())
                total_errors = sum(t.error_count for t in self.active_threads.values())
                stats.update({
                    "total_monitoring_runs": total_runs,
                    "total_monitoring_errors": total_errors,
                    "runs_per_second": total_runs / max(1, uptime),
                    "priority_adjustments_per_hour": self.stats.get("total_priority_adjustments", 0) / (uptime / 3600) if uptime > 0 else 0
                })
            
            # Add activity stats if available
            if self.activity_analyzer:
                active_count = len(self.activity_analyzer.get_active_tokens())
                declining_count = len(self.activity_analyzer.get_declining_tokens())
                inactive_count = len(self.activity_analyzer.get_inactive_tokens())
                
                stats.update({
                    "active_tokens": active_count,
                    "declining_tokens": declining_count,
                    "inactive_tokens": inactive_count,
                    "total_tracked_tokens": active_count + declining_count + inactive_count
                })
        
        return stats 