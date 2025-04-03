#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token Activity Analyzer Module

This module provides functionality for analyzing token activity patterns,
detecting when tokens become inactive, and optimizing resource allocation
based on token lifecycle stages.
"""

import logging
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import threading
from collections import deque

from src.core.events import EventBus, Event, EventType, BaseEventSubscriber

logger = logging.getLogger(__name__)


class TokenLifecycleState(Enum):
    """
    Enum representing the lifecycle states of a token.
    """
    NEW = 0          # Newly created token, limited data available
    ACTIVE = 1       # Active trading, significant volume and price changes
    DECLINING = 2    # Decreasing activity, but still some trades
    INACTIVE = 3     # Very low or no activity
    ABANDONED = 4    # No activity for an extended period


class ActivityMetrics:
    """
    Container for token activity metrics.
    """
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize activity metrics.
        
        Args:
            window_sizes: List of window sizes (in seconds) for tracking metrics
        """
        self.window_sizes = window_sizes or [60, 300, 900]  # 1min, 5min, 15min
        
        # Timestamps of recent trades, used for frequency calculation
        self.trade_times = deque(maxlen=100)
        
        # Store trade data for each window
        self.trades_by_window = {
            window: deque(maxlen=1000) for window in self.window_sizes
        }
        
        # Activity metrics
        self.last_trade_time = 0
        self.last_price = 0.0
        self.creation_time = time.time()
        self.total_trades = 0
        self.total_volume = 0.0
        
        # State information
        self.lifecycle_state = TokenLifecycleState.NEW
        self.last_state_change = time.time()
        self.activity_score = 1.0  # Higher is more active, range 0-1
        self.priority_score = 1.0  # Higher priority gets more resources
        
        # Inactivity detection
        self.consecutive_inactive_checks = 0
        self.peak_volume_per_minute = 0.0
        self.peak_trades_per_minute = 0
        
        # Trend analysis
        self.volume_trend = 0.0  # Positive means increasing
        self.price_trend = 0.0   # Positive means increasing
        self.frequency_trend = 0.0  # Positive means increasing
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Add a trade event to the activity metrics.
        
        Args:
            trade_data: Trade event data
        """
        current_time = time.time()
        
        # Extract core trade information
        timestamp = trade_data.get('timestamp', current_time)
        price = trade_data.get('price', 0.0)
        volume = trade_data.get('volume', 0.0)
        
        # Update basic metrics
        self.last_trade_time = timestamp
        self.last_price = price
        self.total_trades += 1
        self.total_volume += volume
        self.trade_times.append(timestamp)
        
        # Add to window-based metrics
        for window in self.window_sizes:
            self.trades_by_window[window].append({
                'timestamp': timestamp,
                'price': price,
                'volume': volume
            })
        
        # Update peak metrics
        trades_per_minute = self.get_trades_per_minute()
        volume_per_minute = self.get_volume_per_minute()
        
        if trades_per_minute > self.peak_trades_per_minute:
            self.peak_trades_per_minute = trades_per_minute
        
        if volume_per_minute > self.peak_volume_per_minute:
            self.peak_volume_per_minute = volume_per_minute
        
        # Reset inactivity counter since we got a trade
        self.consecutive_inactive_checks = 0
    
    def update_trends(self) -> None:
        """Update the trend metrics based on recent activity."""
        # Calculate volume trend
        volumes = self._get_volumes_by_window(60, 6)  # Last 6 minutes in 1-minute chunks
        self.volume_trend = self._calculate_trend(volumes)
        
        # Calculate price trend
        prices = self._get_prices_by_window(60, 6)  # Last 6 minutes in 1-minute chunks
        self.price_trend = self._calculate_trend(prices)
        
        # Calculate frequency trend
        frequencies = self._get_frequencies_by_window(60, 6)  # Last 6 minutes in 1-minute chunks
        self.frequency_trend = self._calculate_trend(frequencies)
    
    def _get_volumes_by_window(self, window_size: int, num_windows: int) -> List[float]:
        """
        Get volume metrics for multiple windows.
        
        Args:
            window_size: Size of each window in seconds
            num_windows: Number of windows to calculate
            
        Returns:
            List of volumes for each window, oldest first
        """
        now = time.time()
        result = []
        
        for i in range(num_windows, 0, -1):
            start_time = now - (i * window_size)
            end_time = start_time + window_size
            
            # Sum volumes in this window
            window_volume = 0.0
            for window in self.window_sizes:
                for trade in self.trades_by_window[window]:
                    if start_time <= trade['timestamp'] < end_time:
                        window_volume += trade['volume']
            
            result.append(window_volume)
        
        return result
    
    def _get_prices_by_window(self, window_size: int, num_windows: int) -> List[float]:
        """
        Get average prices for multiple windows.
        
        Args:
            window_size: Size of each window in seconds
            num_windows: Number of windows to calculate
            
        Returns:
            List of average prices for each window, oldest first
        """
        now = time.time()
        result = []
        
        for i in range(num_windows, 0, -1):
            start_time = now - (i * window_size)
            end_time = start_time + window_size
            
            # Calculate average price in this window
            prices = []
            for window in self.window_sizes:
                for trade in self.trades_by_window[window]:
                    if start_time <= trade['timestamp'] < end_time:
                        prices.append(trade['price'])
            
            avg_price = sum(prices) / max(1, len(prices)) if prices else 0.0
            result.append(avg_price)
        
        return result
    
    def _get_frequencies_by_window(self, window_size: int, num_windows: int) -> List[int]:
        """
        Get trade frequencies for multiple windows.
        
        Args:
            window_size: Size of each window in seconds
            num_windows: Number of windows to calculate
            
        Returns:
            List of trade counts for each window, oldest first
        """
        now = time.time()
        result = []
        
        for i in range(num_windows, 0, -1):
            start_time = now - (i * window_size)
            end_time = start_time + window_size
            
            # Count trades in this window
            count = 0
            for window in self.window_sizes:
                for trade in self.trades_by_window[window]:
                    if start_time <= trade['timestamp'] < end_time:
                        count += 1
            
            result.append(count)
        
        return result
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend from a list of values.
        
        Args:
            values: List of values, oldest first
            
        Returns:
            Trend value between -1 and 1
        """
        if not values or len(values) < 2:
            return 0.0
        
        # Simple trend calculation: normalize the difference between latest and first
        first, *_, latest = values
        
        # Avoid division by zero
        if first == 0:
            return 1.0 if latest > 0 else 0.0
        
        # Calculate percentage change
        change = (latest - first) / abs(first)
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, change))
    
    def get_trades_per_minute(self) -> float:
        """
        Calculate the number of trades per minute based on recent activity.
        
        Returns:
            Trades per minute
        """
        if len(self.trade_times) < 2:
            return 0.0
        
        # Get time span in minutes
        now = time.time()
        oldest = self.trade_times[0]
        time_span_minutes = (now - oldest) / 60.0
        
        # Avoid division by zero
        if time_span_minutes < 0.1:
            return len(self.trade_times) * 10.0  # Extrapolate to a minute
        
        return len(self.trade_times) / time_span_minutes
    
    def get_volume_per_minute(self) -> float:
        """
        Calculate the trading volume per minute based on recent activity.
        
        Returns:
            Volume per minute
        """
        # Get the volume in the 1-minute window
        window = self.window_sizes[0]  # 60 seconds
        window_volume = sum(trade['volume'] for trade in self.trades_by_window[window])
        
        return window_volume
    
    def get_price_volatility(self) -> float:
        """
        Calculate price volatility based on recent trades.
        
        Returns:
            Volatility measure (standard deviation / mean)
        """
        window = self.window_sizes[0]  # 60 seconds
        prices = [trade['price'] for trade in self.trades_by_window[window]]
        
        if not prices:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        
        # Avoid division by zero
        if mean_price == 0:
            return 0.0
        
        # Calculate standard deviation
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        # Return coefficient of variation
        return std_dev / mean_price
    
    def seconds_since_last_trade(self) -> float:
        """
        Calculate seconds since the last trade.
        
        Returns:
            Seconds since last trade
        """
        return time.time() - self.last_trade_time
    
    def calculate_activity_score(self) -> float:
        """
        Calculate an overall activity score based on recent metrics.
        
        Returns:
            Activity score between 0 and 1
        """
        # We'll use a weighted combination of several factors:
        
        # 1. Recency: exponential decay based on time since last trade
        seconds_since = self.seconds_since_last_trade()
        recency_score = max(0.0, min(1.0, math.exp(-seconds_since / 300.0)))  # 5-minute half-life
        
        # 2. Trade frequency relative to peak
        current_frequency = self.get_trades_per_minute()
        frequency_ratio = current_frequency / max(1.0, self.peak_trades_per_minute)
        frequency_score = max(0.0, min(1.0, frequency_ratio))
        
        # 3. Volume relative to peak
        current_volume = self.get_volume_per_minute()
        volume_ratio = current_volume / max(0.01, self.peak_volume_per_minute)
        volume_score = max(0.0, min(1.0, volume_ratio))
        
        # 4. Trend factors
        volume_trend_factor = (self.volume_trend + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        frequency_trend_factor = (self.frequency_trend + 1.0) / 2.0
        
        # Combine factors with weights
        combined_score = (
            0.4 * recency_score +
            0.3 * frequency_score +
            0.2 * volume_score +
            0.05 * volume_trend_factor +
            0.05 * frequency_trend_factor
        )
        
        # Ensure the score is in [0,1] range
        self.activity_score = max(0.0, min(1.0, combined_score))
        return self.activity_score


class ActivityAnalyzer(BaseEventSubscriber):
    """
    Analyzes token activity patterns to detect changing lifecycle states.
    
    This component tracks trade activity for tokens, detects when tokens
    become inactive, and recommends resource allocation based on token activity.
    """
    
    def __init__(self, event_bus: EventBus, 
                 activity_check_interval: float = 30.0,
                 inactivity_threshold: float = 0.2,
                 inactive_checks_before_state_change: int = 3):
        """
        Initialize the activity analyzer.
        
        Args:
            event_bus: Event bus for publishing activity state changes
            activity_check_interval: Interval between activity checks in seconds
            inactivity_threshold: Activity score threshold for considering a token inactive
            inactive_checks_before_state_change: Number of consecutive checks below threshold
                                                before changing state to inactive
        """
        super().__init__(event_bus)
        
        # Settings
        self.activity_check_interval = activity_check_interval
        self.inactivity_threshold = inactivity_threshold
        self.inactive_checks_before_state_change = inactive_checks_before_state_change
        
        # Metrics storage
        self.token_activity: Dict[str, ActivityMetrics] = {}
        self.token_lock = threading.RLock()
        
        # Thread for periodic activity checks
        self._stop_event = threading.Event()
        self._activity_check_thread = None
        
        # Register event handlers
        self.register_handler(EventType.TOKEN_CREATED, self._handle_token_created)
        self.register_handler(EventType.TOKEN_TRADE, self._handle_token_trade)
        self.register_handler(EventType.SYSTEM, self._handle_system_event)
        
        logger.info(f"Initialized ActivityAnalyzer with check interval {activity_check_interval}s")
    
    def start(self) -> None:
        """Start the activity analyzer thread."""
        if self._activity_check_thread and self._activity_check_thread.is_alive():
            logger.warning("Activity analyzer thread already running")
            return
        
        self._stop_event.clear()
        self._activity_check_thread = threading.Thread(
            target=self._activity_check_loop,
            daemon=True,
            name="ActivityAnalyzer"
        )
        self._activity_check_thread.start()
        logger.info("Started activity analyzer thread")
    
    def stop(self) -> None:
        """Stop the activity analyzer thread."""
        self._stop_event.set()
        
        if self._activity_check_thread and self._activity_check_thread.is_alive():
            self._activity_check_thread.join(timeout=5.0)
            logger.info("Stopped activity analyzer thread")
    
    def _activity_check_loop(self) -> None:
        """Main loop for checking token activity."""
        logger.info("Activity check loop starting")
        
        while not self._stop_event.is_set():
            try:
                self._check_all_token_activity()
            except Exception as e:
                logger.error(f"Error in activity check loop: {e}", exc_info=True)
            
            # Sleep until next check
            for _ in range(int(self.activity_check_interval / 0.1)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
        
        logger.info("Activity check loop stopped")
    
    def _check_all_token_activity(self) -> None:
        """Check activity for all tracked tokens."""
        with self.token_lock:
            for token_id, metrics in list(self.token_activity.items()):
                try:
                    self._check_token_activity(token_id, metrics)
                except Exception as e:
                    logger.error(f"Error checking activity for token {token_id}: {e}", exc_info=True)
    
    def _check_token_activity(self, token_id: str, metrics: ActivityMetrics) -> None:
        """
        Check activity for a specific token.
        
        Args:
            token_id: ID of the token to check
            metrics: Activity metrics for the token
        """
        # Update trend metrics
        metrics.update_trends()
        
        # Calculate activity score
        activity_score = metrics.calculate_activity_score()
        
        # Check for state transitions
        current_state = metrics.lifecycle_state
        new_state = current_state
        
        # Determine new state based on activity score and current state
        if current_state == TokenLifecycleState.NEW:
            # New token becomes active after a few trades
            if metrics.total_trades >= 5:
                new_state = TokenLifecycleState.ACTIVE
        
        elif current_state == TokenLifecycleState.ACTIVE:
            # Active token becomes declining if activity drops significantly
            if activity_score < self.inactivity_threshold * 2:
                metrics.consecutive_inactive_checks += 1
                if metrics.consecutive_inactive_checks >= self.inactive_checks_before_state_change:
                    new_state = TokenLifecycleState.DECLINING
            else:
                metrics.consecutive_inactive_checks = 0
        
        elif current_state == TokenLifecycleState.DECLINING:
            # Declining token becomes inactive if activity continues to drop
            if activity_score < self.inactivity_threshold:
                metrics.consecutive_inactive_checks += 1
                if metrics.consecutive_inactive_checks >= self.inactive_checks_before_state_change:
                    new_state = TokenLifecycleState.INACTIVE
            else:
                # Can go back to active if activity picks up
                if activity_score > self.inactivity_threshold * 3:
                    new_state = TokenLifecycleState.ACTIVE
                metrics.consecutive_inactive_checks = 0
        
        elif current_state == TokenLifecycleState.INACTIVE:
            # Inactive token becomes abandoned after extended inactivity
            seconds_since = metrics.seconds_since_last_trade()
            if seconds_since > 900:  # 15 minutes
                new_state = TokenLifecycleState.ABANDONED
            # Can go back to declining if activity picks up
            elif activity_score > self.inactivity_threshold * 2:
                new_state = TokenLifecycleState.DECLINING
        
        # If state changed, update and notify
        if new_state != current_state:
            self._handle_state_change(token_id, metrics, current_state, new_state)
    
    def _handle_state_change(self, token_id: str, metrics: ActivityMetrics, 
                           old_state: TokenLifecycleState, new_state: TokenLifecycleState) -> None:
        """
        Handle a token lifecycle state change.
        
        Args:
            token_id: ID of the token
            metrics: Activity metrics for the token
            old_state: Previous lifecycle state
            new_state: New lifecycle state
        """
        # Update state in metrics
        metrics.lifecycle_state = new_state
        metrics.last_state_change = time.time()
        
        # Calculate new priority based on state
        if new_state == TokenLifecycleState.ACTIVE:
            # Active tokens get high priority
            new_priority = max(1.0, metrics.activity_score * 10)
        elif new_state == TokenLifecycleState.DECLINING:
            # Declining tokens get medium priority
            new_priority = metrics.activity_score * 5
        elif new_state == TokenLifecycleState.INACTIVE:
            # Inactive tokens get low priority
            new_priority = metrics.activity_score * 2
        elif new_state == TokenLifecycleState.ABANDONED:
            # Abandoned tokens get minimal priority
            new_priority = 0.1
        else:
            # New tokens start with medium-high priority
            new_priority = 7.0
        
        # Update priority score
        metrics.priority_score = new_priority
        
        # Publish state change event
        self.publish_event(
            EventType.GENERIC,
            {
                "action": "token_lifecycle_changed",
                "token_id": token_id,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "activity_score": metrics.activity_score,
                "priority_score": metrics.priority_score,
                "total_trades": metrics.total_trades,
                "total_volume": metrics.total_volume,
                "seconds_since_last_trade": metrics.seconds_since_last_trade(),
                "trades_per_minute": metrics.get_trades_per_minute(),
                "volume_trend": metrics.volume_trend,
                "frequency_trend": metrics.frequency_trend
            }
        )
        
        logger.info(f"Token {token_id} state changed: {old_state.name} -> {new_state.name} "
                   f"(activity: {metrics.activity_score:.2f}, priority: {metrics.priority_score:.2f})")
    
    def _handle_token_created(self, event: Event) -> None:
        """
        Handle token creation events.
        
        Args:
            event: Token created event
        """
        data = event.data
        token_id = data.get('token_id')
        
        if not token_id:
            logger.warning("Token creation event missing token_id")
            return
        
        with self.token_lock:
            # Initialize activity metrics for new token
            if token_id not in self.token_activity:
                self.token_activity[token_id] = ActivityMetrics()
                logger.info(f"Started tracking activity for new token: {token_id}")
    
    def _handle_token_trade(self, event: Event) -> None:
        """
        Handle token trade events.
        
        Args:
            event: Token trade event
        """
        data = event.data
        token_id = data.get('token_id')
        
        if not token_id:
            logger.warning("Token trade event missing token_id")
            return
        
        with self.token_lock:
            # Ensure token is being tracked
            if token_id not in self.token_activity:
                self.token_activity[token_id] = ActivityMetrics()
                logger.info(f"Started tracking activity for token: {token_id} (from trade event)")
            
            # Update metrics with trade data
            self.token_activity[token_id].add_trade(data)
    
    def _handle_system_event(self, event: Event) -> None:
        """
        Handle system events.
        
        Args:
            event: System event
        """
        # Extract the system event data
        data = event.data
        action = data.get('action', '')
        
        if action == 'shutdown':
            logger.info("Received shutdown system event, stopping ActivityAnalyzer")
            self.stop()
        elif action == 'pause':
            logger.info("Received pause system event, pausing ActivityAnalyzer")
            # Implement pause behavior if needed
        elif action == 'resume':
            logger.info("Received resume system event, resuming ActivityAnalyzer")
            # Implement resume behavior if needed
        else:
            # Log unknown system events for debugging
            logger.debug(f"Received unknown system event: {action}")
    
    def update_activity_metrics(self, token_id: str, event: Event) -> None:
        """
        Update activity metrics for a token based on an event.
        
        Args:
            token_id: ID of the token
            event: Event with token data
        """
        # This is a public version of the internal event handlers
        event_type = event.event_type
        
        if event_type == EventType.TOKEN_CREATED:
            self._handle_token_created(event)
        elif event_type == EventType.TOKEN_TRADE:
            self._handle_token_trade(event)
    
    def get_activity_level(self, token_id: str) -> float:
        """
        Get the current activity level for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Activity score between 0 and 1
        """
        with self.token_lock:
            if token_id not in self.token_activity:
                return 0.0
            
            return self.token_activity[token_id].activity_score
    
    def detect_activity_change(self, token_id: str) -> bool:
        """
        Detect if there has been a significant change in token activity recently.
        
        Args:
            token_id: ID of the token
            
        Returns:
            True if significant activity change detected
        """
        with self.token_lock:
            if token_id not in self.token_activity:
                return False
            
            metrics = self.token_activity[token_id]
            
            # Check for significant changes in trend
            return abs(metrics.volume_trend) > 0.3 or abs(metrics.frequency_trend) > 0.3
    
    def get_token_lifecycle_state(self, token_id: str) -> Optional[TokenLifecycleState]:
        """
        Get the current lifecycle state of a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Current lifecycle state or None if token not found
        """
        with self.token_lock:
            if token_id not in self.token_activity:
                return None
            
            return self.token_activity[token_id].lifecycle_state
    
    def recommend_priority(self, token_id: str) -> float:
        """
        Recommend a monitoring priority for the token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Recommended priority level (higher = more important)
        """
        with self.token_lock:
            if token_id not in self.token_activity:
                # New tokens start with medium-high priority
                return 7.0
            
            return self.token_activity[token_id].priority_score
    
    def get_active_tokens(self) -> Set[str]:
        """
        Get the set of currently active tokens.
        
        Returns:
            Set of token IDs that are in the ACTIVE state
        """
        active_tokens = set()
        
        with self.token_lock:
            for token_id, metrics in self.token_activity.items():
                if metrics.lifecycle_state == TokenLifecycleState.ACTIVE:
                    active_tokens.add(token_id)
        
        return active_tokens
    
    def get_declining_tokens(self) -> Set[str]:
        """
        Get the set of currently declining tokens.
        
        Returns:
            Set of token IDs that are in the DECLINING state
        """
        declining_tokens = set()
        
        with self.token_lock:
            for token_id, metrics in self.token_activity.items():
                if metrics.lifecycle_state == TokenLifecycleState.DECLINING:
                    declining_tokens.add(token_id)
        
        return declining_tokens
    
    def get_inactive_tokens(self) -> Set[str]:
        """
        Get the set of inactive tokens.
        
        Returns:
            Set of token IDs that are in the INACTIVE or ABANDONED state
        """
        inactive_tokens = set()
        
        with self.token_lock:
            for token_id, metrics in self.token_activity.items():
                if metrics.lifecycle_state in (TokenLifecycleState.INACTIVE, TokenLifecycleState.ABANDONED):
                    inactive_tokens.add(token_id)
        
        return inactive_tokens
    
    def get_token_stats(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary with token statistics or None if token not found
        """
        with self.token_lock:
            if token_id not in self.token_activity:
                return None
            
            metrics = self.token_activity[token_id]
            
            return {
                'token_id': token_id,
                'lifecycle_state': metrics.lifecycle_state.name,
                'activity_score': metrics.activity_score,
                'priority_score': metrics.priority_score,
                'total_trades': metrics.total_trades,
                'total_volume': metrics.total_volume,
                'creation_time': metrics.creation_time,
                'last_trade_time': metrics.last_trade_time,
                'last_price': metrics.last_price,
                'seconds_since_last_trade': metrics.seconds_since_last_trade(),
                'trades_per_minute': metrics.get_trades_per_minute(),
                'volume_per_minute': metrics.get_volume_per_minute(),
                'price_volatility': metrics.get_price_volatility(),
                'volume_trend': metrics.volume_trend,
                'price_trend': metrics.price_trend,
                'frequency_trend': metrics.frequency_trend,
                'peak_trades_per_minute': metrics.peak_trades_per_minute,
                'peak_volume_per_minute': metrics.peak_volume_per_minute
            }
    
    def get_all_token_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tracked tokens.
        
        Returns:
            Dictionary mapping token IDs to their statistics
        """
        result = {}
        
        with self.token_lock:
            for token_id in self.token_activity:
                result[token_id] = self.get_token_stats(token_id)
        
        return result
    
    def cleanup(self) -> None:
        """Clean up resources used by the activity analyzer."""
        self.stop()
        
        with self.token_lock:
            self.token_activity.clear()
        
        super().cleanup()


import math  # Import at the end to avoid unresolved reference in ActivityMetrics 