#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graceful Exit Manager Module

This module provides the GracefulExitManager implementation for managing
strategic exits of positions during system shutdown or when exiting tokens.
It prioritizes positions based on activity levels, profit/loss, and market conditions.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from queue import PriorityQueue

from src.core.events import EventBus, Event, EventType
from src.core.trading.interfaces import PositionManager, TradeExecutor
from src.core.trading.activity_analyzer import ActivityAnalyzer, TokenLifecycleState

logger = logging.getLogger(__name__)


class ExitPriority:
    """Constants for exit prioritization."""
    URGENT = 0       # Immediate exit needed (e.g., inactive token, large loss)
    HIGH = 1         # High priority (e.g., declining activity, moderate loss)
    NORMAL = 2       # Normal priority (e.g., normal activity, small loss)
    OPPORTUNITY = 3  # Wait for opportunity (e.g., active token, in profit)
    HOLD = 4         # Hold as long as possible (e.g., very active, high profit)


class GracefulExitManager:
    """
    Manager for strategic exits during system shutdown or token offboarding.
    
    The GracefulExitManager prioritizes and manages exits for positions to
    maximize profit and minimize losses when exiting tokens or shutting down.
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 position_manager: PositionManager,
                 trade_executor: TradeExecutor,
                 activity_analyzer: Optional[ActivityAnalyzer] = None,
                 exit_timeout_seconds: float = 60.0,
                 price_check_interval: float = 0.5,
                 max_exit_attempts: int = 3,
                 emergency_exit_threshold: float = -0.15):
        """
        Initialize the GracefulExitManager.
        
        Args:
            event_bus: Event bus for communication
            position_manager: Position manager for tracking positions
            trade_executor: Trade executor for executing exits
            activity_analyzer: Optional activity analyzer for token activity tracking
            exit_timeout_seconds: Maximum seconds to wait for all exits to complete
            price_check_interval: Interval in seconds to check prices for exit conditions
            max_exit_attempts: Maximum number of exit attempts per position
            emergency_exit_threshold: PnL threshold for emergency exits (-0.15 = -15%)
        """
        self.event_bus = event_bus
        self.position_manager = position_manager
        self.trade_executor = trade_executor
        self.activity_analyzer = activity_analyzer
        
        self.exit_timeout_seconds = exit_timeout_seconds
        self.price_check_interval = price_check_interval
        self.max_exit_attempts = max_exit_attempts
        self.emergency_exit_threshold = emergency_exit_threshold
        
        self.exit_queue = PriorityQueue()
        self.exit_in_progress = False
        self.exit_thread = None
        self.stop_event = threading.Event()
        
        # Statistics about exit performance
        self.stats = {
            'total_exits_attempted': 0,
            'successful_exits': 0,
            'failed_exits': 0,
            'total_pnl_saved': 0.0,
            'exit_processing_time': 0.0,
            'avg_exit_slippage': 0.0,
            'emergency_exits': 0,
            'positions_remaining': 0
        }
        
        logger.info(f"Initialized GracefulExitManager with exit_timeout={exit_timeout_seconds}s")
    
    def begin_graceful_exit(self, token_ids: Optional[List[str]] = None) -> bool:
        """
        Begin graceful exit process for all positions or specific tokens.
        
        Args:
            token_ids: Optional list of token IDs to exit. If None, exit all positions.
            
        Returns:
            True if exit process started, False if already in progress
        """
        if self.exit_in_progress:
            logger.warning("Graceful exit already in progress")
            return False
            
        self.exit_in_progress = True
        start_time = time.time()
        
        # Get all open positions
        open_positions = self.position_manager.get_open_positions()
        
        # Filter by token_ids if provided
        if token_ids:
            filtered_positions = {
                pos_id: pos for pos_id, pos in open_positions.items()
                if pos['token_id'] in token_ids
            }
        else:
            filtered_positions = open_positions
            
        if not filtered_positions:
            logger.info("No positions to exit")
            self.exit_in_progress = False
            return True
            
        logger.info(f"Beginning graceful exit for {len(filtered_positions)} positions")
        
        # Prioritize all positions for exit
        self.prioritize_exits(filtered_positions)
        
        # Start exit processing thread
        self.stop_event.clear()
        self.exit_thread = threading.Thread(
            target=self._process_exits,
            name="GracefulExitProcessor"
        )
        self.exit_thread.daemon = True
        self.exit_thread.start()
        
        # Publish exit started event
        self._publish_exit_event(
            action="graceful_exit_started",
            positions_count=len(filtered_positions),
            token_ids=token_ids
        )
        
        self.stats['exit_processing_time'] = time.time() - start_time
        return True
    
    def prioritize_exits(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """
        Prioritize positions for exit and add to exit queue.
        
        Args:
            positions: Dictionary of positions to prioritize
        """
        logger.info(f"Prioritizing {len(positions)} positions for exit")
        
        # Clear existing queue
        with self.exit_queue.mutex:
            self.exit_queue.queue.clear()
            
        # Add each position to the queue with priority
        for position_id, position in positions.items():
            priority = self._calculate_exit_priority(position)
            self.exit_queue.put((priority, position_id, position))
            
        logger.info(f"Exit queue prepared with {self.exit_queue.qsize()} positions")
    
    def abort_graceful_exit(self) -> None:
        """
        Abort the current graceful exit process.
        """
        if not self.exit_in_progress:
            logger.warning("No graceful exit in progress to abort")
            return
            
        logger.info("Aborting graceful exit process")
        self.stop_event.set()
        
        if self.exit_thread and self.exit_thread.is_alive():
            self.exit_thread.join(timeout=5.0)
            
        # Empty the queue
        with self.exit_queue.mutex:
            remaining = self.exit_queue.qsize()
            self.exit_queue.queue.clear()
            
        # Update stats
        self.stats['positions_remaining'] = remaining
            
        self.exit_in_progress = False
        
        # Publish exit aborted event
        self._publish_exit_event(
            action="graceful_exit_aborted",
            remaining_positions=remaining
        )
        
        logger.info(f"Graceful exit aborted with {remaining} positions remaining")
    
    def is_exit_in_progress(self) -> bool:
        """
        Check if a graceful exit is currently in progress.
        
        Returns:
            True if exit is in progress, False otherwise
        """
        return self.exit_in_progress
    
    def wait_for_exit_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all exits to complete.
        
        Args:
            timeout: Maximum seconds to wait (None = use exit_timeout_seconds)
            
        Returns:
            True if all exits completed, False if timeout reached
        """
        if not self.exit_in_progress:
            return True
            
        timeout = timeout or self.exit_timeout_seconds
        start_time = time.time()
        
        logger.info(f"Waiting up to {timeout}s for graceful exit completion")
        
        while self.exit_in_progress and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if self.exit_in_progress:
            logger.warning(f"Timeout reached after {timeout}s waiting for exit completion")
            return False
            
        return True
    
    def get_exit_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the exit process.
        
        Returns:
            Dictionary of exit statistics
        """
        # Add current queue size if in progress
        if self.exit_in_progress:
            self.stats['positions_remaining'] = self.exit_queue.qsize()
            
        return self.stats.copy()
    
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        This method must be called when the manager is no longer needed.
        It ensures that any running exit processes are properly terminated.
        """
        logger.info("Cleaning up GracefulExitManager")
        
        if self.exit_in_progress:
            self.abort_graceful_exit()
            
        # Clear queue
        with self.exit_queue.mutex:
            self.exit_queue.queue.clear()
            
        self.stop_event.set()
            
        logger.info("GracefulExitManager cleaned up")
    
    def _calculate_exit_priority(self, position: Dict[str, Any]) -> int:
        """
        Calculate exit priority for a position.
        
        This method determines how urgently a position should be exited based on:
        1. Token activity level (if activity_analyzer is available)
        2. Position profit/loss
        3. Position age
        4. Market conditions
        
        Args:
            position: Position details
            
        Returns:
            Priority level (lower number = higher priority)
        """
        token_id = position['token_id']
        current_price = position['current_price']
        entry_price = position['entry_price']
        unrealized_pnl_pct = position['unrealized_pnl_pct']
        entry_time = position['entry_time']
        position_age = (datetime.now() - entry_time).total_seconds() / 60.0  # In minutes
        
        # Default to NORMAL priority
        priority = ExitPriority.NORMAL
        
        # Check for emergency exit conditions (large loss)
        if unrealized_pnl_pct <= self.emergency_exit_threshold:
            logger.warning(f"Emergency exit condition for {token_id}: PnL {unrealized_pnl_pct:.2%}")
            return ExitPriority.URGENT
        
        # Assess token activity if analyzer is available
        if self.activity_analyzer:
            try:
                lifecycle_state = self.activity_analyzer.get_token_lifecycle_state(token_id)
                activity_level = self.activity_analyzer.get_activity_level(token_id)
                
                # Adjust priority based on token lifecycle state
                if lifecycle_state == TokenLifecycleState.INACTIVE:
                    priority = min(priority, ExitPriority.URGENT)
                elif lifecycle_state == TokenLifecycleState.DECLINING:
                    priority = min(priority, ExitPriority.HIGH)
                elif lifecycle_state == TokenLifecycleState.ACTIVE:
                    if unrealized_pnl_pct > 0:
                        priority = max(priority, ExitPriority.OPPORTUNITY)
                    else:
                        priority = min(priority, ExitPriority.NORMAL)
                        
                # Further adjust based on numeric activity level
                if activity_level < 0.2:  # Very low activity
                    priority = min(priority, ExitPriority.URGENT)
                elif activity_level < 0.5:  # Low activity
                    priority = min(priority, ExitPriority.HIGH)
                
            except Exception as e:
                logger.error(f"Error getting activity data for {token_id}: {e}")
        
        # Adjust priority based on profit/loss
        if unrealized_pnl_pct >= 0.1:  # +10% profit
            priority = max(priority, ExitPriority.HOLD)
        elif unrealized_pnl_pct >= 0.05:  # +5% profit
            priority = max(priority, ExitPriority.OPPORTUNITY)
        elif unrealized_pnl_pct <= -0.1:  # -10% loss
            priority = min(priority, ExitPriority.HIGH)
        elif unrealized_pnl_pct <= -0.05:  # -5% loss
            priority = min(priority, ExitPriority.NORMAL)
            
        # Adjust priority based on position age
        if position_age > 60:  # Position open for > 1 hour
            priority = min(priority, ExitPriority.HIGH)
        elif position_age > 15:  # Position open for > 15 minutes
            priority = min(priority, ExitPriority.NORMAL)
            
        return priority
    
    def _calculate_optimal_exit_price(self, token_id: str, position: Dict[str, Any]) -> float:
        """
        Calculate optimal exit price for a position.
        
        Args:
            token_id: ID of the token
            position: Position details
            
        Returns:
            Optimal exit price
        """
        # Get current price
        try:
            current_price = self.trade_executor.get_current_price(token_id)
        except Exception as e:
            logger.error(f"Error getting current price for {token_id}: {e}")
            return position['current_price']
            
        entry_price = position['entry_price']
        unrealized_pnl_pct = position['unrealized_pnl_pct']
        
        # If in profit, use current price
        if unrealized_pnl_pct >= 0:
            return current_price
            
        # If in loss but not an emergency, consider market trends
        if unrealized_pnl_pct > self.emergency_exit_threshold:
            # In the future, this could use market trend analysis to
            # determine if waiting might improve the exit price
            return current_price
            
        # For emergency exits, just use current price
        return current_price
    
    def _process_exits(self) -> None:
        """
        Process the exit queue until empty or stopped.
        
        This method runs in a separate thread to handle position exits.
        """
        logger.info("Starting exit processing thread")
        
        start_time = time.time()
        successful_exits = 0
        failed_exits = 0
        total_pnl_saved = 0.0
        
        try:
            while not self.stop_event.is_set() and not self.exit_queue.empty():
                try:
                    # Get the highest priority position
                    priority, position_id, position = self.exit_queue.get(block=False)
                    token_id = position['token_id']
                    
                    logger.info(f"Processing exit for {token_id} (position {position_id}, " 
                               f"priority {priority})")
                    
                    # Calculate optimal exit price
                    exit_price = self._calculate_optimal_exit_price(token_id, position)
                    
                    # Execute the exit
                    success = self._execute_exit(position_id, token_id, exit_price)
                    
                    if success:
                        # Get updated position for PnL calculation
                        try:
                            closed_position = self.position_manager.get_position(position_id)
                            realized_pnl = closed_position.get('realized_pnl', 0.0)
                            total_pnl_saved += realized_pnl
                            successful_exits += 1
                            
                            logger.info(f"Successfully exited {token_id} with PnL {realized_pnl:.2f}")
                        except Exception as e:
                            logger.error(f"Error getting closed position data: {e}")
                            successful_exits += 1
                    else:
                        failed_exits += 1
                        
                        # Requeue if not too many attempts
                        attempt = position.get('exit_attempts', 0) + 1
                        if attempt < self.max_exit_attempts:
                            position['exit_attempts'] = attempt
                            # Bump up priority for retry
                            retry_priority = max(0, priority - 1)
                            self.exit_queue.put((retry_priority, position_id, position))
                            logger.info(f"Requeued {token_id} for retry (attempt {attempt}/{self.max_exit_attempts})")
                    
                    # Mark task as done
                    self.exit_queue.task_done()
                    
                    # Check if we've exceeded the timeout
                    if (time.time() - start_time) > self.exit_timeout_seconds:
                        logger.warning(f"Exit timeout reached after {self.exit_timeout_seconds}s")
                        break
                        
                    # Small delay between exits
                    time.sleep(self.price_check_interval)
                    
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing exit: {e}", exc_info=True)
                    time.sleep(1.0)  # Avoid tight loop on error
        
        finally:
            # Update statistics
            self.stats['total_exits_attempted'] += successful_exits + failed_exits
            self.stats['successful_exits'] += successful_exits
            self.stats['failed_exits'] += failed_exits
            self.stats['total_pnl_saved'] += total_pnl_saved
            self.stats['exit_processing_time'] = time.time() - start_time
            
            # If there are remaining positions, record them
            remaining = self.exit_queue.qsize()
            self.stats['positions_remaining'] = remaining
            
            # Mark exit as complete
            self.exit_in_progress = False
            
            # Publish exit completed event
            self._publish_exit_event(
                action="graceful_exit_completed",
                successful_exits=successful_exits,
                failed_exits=failed_exits,
                remaining_positions=remaining,
                total_pnl_saved=total_pnl_saved,
                processing_time=self.stats['exit_processing_time']
            )
            
            logger.info(f"Exit processing complete: {successful_exits} successful, "
                       f"{failed_exits} failed, {remaining} remaining, "
                       f"PnL saved: {total_pnl_saved:.2f}, "
                       f"time: {self.stats['exit_processing_time']:.2f}s")
    
    def _execute_exit(self, position_id: str, token_id: str, exit_price: float) -> bool:
        """
        Execute a position exit.
        
        Args:
            position_id: ID of the position
            token_id: ID of the token
            exit_price: Calculated exit price
            
        Returns:
            True if exit was successful, False otherwise
        """
        try:
            # Check if position is still open
            position = self.position_manager.get_position(position_id)
            if position.get('status', '') != 'open':
                logger.info(f"Position {position_id} for {token_id} is already closed")
                return True
                
            # Execute the exit through the trade executor
            success = self.trade_executor.execute_exit(
                position_id=position_id,
                signal_price=exit_price,
                reason='graceful_exit'
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing exit for {token_id} (position {position_id}): {e}")
            return False
    
    def _publish_exit_event(self, action: str, **kwargs) -> None:
        """
        Publish exit event.
        
        Args:
            action: Action performed
            **kwargs: Additional event data
        """
        if not self.event_bus:
            return
            
        event_data = {
            'action': action,
            'timestamp': datetime.now(),
            **kwargs
        }
        
        event = Event(
            event_type=EventType.SYSTEM_STATUS,
            data=event_data
        )
        
        self.event_bus.publish(event) 