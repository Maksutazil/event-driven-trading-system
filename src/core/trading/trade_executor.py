#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade Executor Module

This module provides the DefaultTradeExecutor implementation for executing
trade signals by interacting with exchange APIs or simulated execution.
"""

import logging
import time
import random
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime

from src.core.events import EventBus, Event, EventType
from src.core.trading.interfaces import TradeExecutor, RiskManager, PositionManager

logger = logging.getLogger(__name__)


class DefaultTradeExecutor(TradeExecutor):
    """
    Default implementation of the TradeExecutor interface.
    
    This implementation handles trade execution based on signals,
    with support for simulation or real trading via adapters.
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 position_manager: PositionManager,
                 risk_manager: RiskManager,
                 price_fetcher: Callable[[str], float],
                 slippage_pct: float = 0.005,
                 execution_delay: float = 0.0,
                 simulate: bool = True):
        """
        Initialize the DefaultTradeExecutor.
        
        Args:
            event_bus: Event bus for publishing trade events
            position_manager: Position manager for tracking positions
            risk_manager: Risk manager for calculating position sizes
            price_fetcher: Function that returns current price for a token
            slippage_pct: Slippage percentage for simulated execution (default: 0.5%)
            execution_delay: Simulated execution delay in seconds (default: 0)
            simulate: Whether to simulate execution (default: True)
        """
        # Validate inputs
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if execution_delay < 0:
            raise ValueError("execution_delay must be non-negative")
            
        self.event_bus = event_bus
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.price_fetcher = price_fetcher
        self.slippage_pct = slippage_pct
        self.execution_delay = execution_delay
        self.simulate = simulate
        
        # Dictionary to store execution statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_slippage': 0.0,
            'avg_execution_time': 0.0
        }
        
        logger.info(f"Initialized DefaultTradeExecutor with simulate={simulate}, "
                   f"slippage_pct={slippage_pct}, execution_delay={execution_delay}")
        
    def execute_entry(self, token_id: str, signal_price: float, 
                     risk_score: float = 0.5, 
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Execute a trade entry based on a signal.
        
        Args:
            token_id: ID of the token to trade
            signal_price: Price at which the signal was generated
            risk_score: Risk score for the trade (0.0 to 1.0)
            metadata: Additional metadata about the trade
            
        Returns:
            Position ID if successful, None otherwise
        """
        # Start tracking execution time
        start_time = time.time()
        
        # Log the trade signal
        logger.info(f"Executing entry for {token_id} at signal price {signal_price} with risk score {risk_score}")
        
        try:
            # Get current price
            current_price = self._get_execution_price(token_id, signal_price, 'entry')
            
            # Check if price has moved too much from signal price
            price_diff_pct = abs(current_price - signal_price) / signal_price
            if price_diff_pct > 0.05:  # 5% threshold for price movement
                logger.warning(f"Price moved too much since signal: {price_diff_pct:.2%}, aborting entry")
                self.stats['failed_executions'] += 1
                
                # Publish execution failed event
                self._publish_trade_event(
                    token_id=token_id,
                    action='entry_failed',
                    reason=f"Price moved too much: {price_diff_pct:.2%}",
                    price=current_price,
                    signal_price=signal_price
                )
                
                return None
                
            # Get available capital from position manager
            available_capital = self.position_manager.get_available_capital()
            
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                token_id=token_id,
                entry_price=current_price,
                risk_score=risk_score,
                available_capital=available_capital
            )
            
            # Check if position size is valid
            if position_size <= 0:
                logger.warning(f"Invalid position size: {position_size}, aborting entry")
                self.stats['failed_executions'] += 1
                
                # Publish execution failed event
                self._publish_trade_event(
                    token_id=token_id,
                    action='entry_failed',
                    reason="Invalid position size",
                    price=current_price,
                    signal_price=signal_price
                )
                
                return None
                
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss(
                token_id=token_id,
                entry_price=current_price,
                position_size=position_size,
                risk_score=risk_score
            )
            
            take_profit = self.risk_manager.calculate_take_profit(
                token_id=token_id,
                entry_price=current_price,
                stop_loss_price=stop_loss
            )
            
            # Add execution details to metadata
            exec_metadata = metadata or {}
            exec_metadata.update({
                'signal_price': signal_price,
                'execution_price': current_price,
                'price_slippage_pct': (current_price - signal_price) / signal_price,
                'execution_time': datetime.now(),
                'risk_score': risk_score,
                'simulated': self.simulate
            })
            
            # Simulate execution delay if enabled
            if self.execution_delay > 0:
                time.sleep(self.execution_delay)
                
            # Open position through position manager
            position_id = self.position_manager.open_position(
                token_id=token_id,
                entry_price=current_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=exec_metadata
            )
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['total_executions'] += 1
            self.stats['successful_executions'] += 1
            self.stats['avg_execution_time'] = self._update_average(
                self.stats['avg_execution_time'],
                execution_time,
                self.stats['successful_executions']
            )
            self.stats['avg_slippage'] = self._update_average(
                self.stats['avg_slippage'],
                abs((current_price - signal_price) / signal_price),
                self.stats['successful_executions']
            )
            
            # Publish trade entry event
            self._publish_trade_event(
                token_id=token_id,
                action='entry',
                position_id=position_id,
                price=current_price,
                signal_price=signal_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"Successfully opened position {position_id} for {token_id} at {current_price}")
            return position_id
            
        except Exception as e:
            logger.error(f"Error executing entry for {token_id}: {str(e)}")
            self.stats['failed_executions'] += 1
            
            # Publish execution failed event
            self._publish_trade_event(
                token_id=token_id,
                action='entry_failed',
                reason=str(e),
                price=signal_price,
                signal_price=signal_price
            )
            
            return None
    
    def execute_exit(self, position_id: str, signal_price: float,
                    reason: str = 'signal') -> bool:
        """
        Execute a trade exit based on a signal.
        
        Args:
            position_id: ID of the position to exit
            signal_price: Price at which the exit signal was generated
            reason: Reason for exiting the position
            
        Returns:
            True if successful, False otherwise
        """
        # Start tracking execution time
        start_time = time.time()
        
        try:
            # Get position details
            position = self.position_manager.get_position(position_id)
            if not position or position['status'] != 'open':
                logger.warning(f"Position {position_id} not found or not open")
                self.stats['failed_executions'] += 1
                return False
                
            token_id = position['token_id']
            logger.info(f"Executing exit for position {position_id} ({token_id}) at signal price {signal_price}")
            
            # Get current price
            current_price = self._get_execution_price(token_id, signal_price, 'exit')
            
            # Simulate execution delay if enabled
            if self.execution_delay > 0:
                time.sleep(self.execution_delay)
                
            # Close position through position manager
            closed_position = self.position_manager.close_position(
                position_id=position_id,
                exit_price=current_price,
                exit_reason=reason
            )
            
            # Update statistics
            execution_time = time.time() - start_time
            self.stats['total_executions'] += 1
            self.stats['successful_executions'] += 1
            self.stats['avg_execution_time'] = self._update_average(
                self.stats['avg_execution_time'],
                execution_time,
                self.stats['successful_executions']
            )
            self.stats['avg_slippage'] = self._update_average(
                self.stats['avg_slippage'],
                abs((current_price - signal_price) / signal_price),
                self.stats['successful_executions']
            )
            
            # Publish trade exit event
            self._publish_trade_event(
                token_id=token_id,
                action='exit',
                position_id=position_id,
                price=current_price,
                signal_price=signal_price,
                realized_pnl=closed_position.get('realized_pnl', 0.0),
                exit_reason=reason
            )
            
            logger.info(f"Successfully closed position {position_id} for {token_id} at {current_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing exit for position {position_id}: {str(e)}")
            self.stats['failed_executions'] += 1
            
            # Publish execution failed event
            self._publish_trade_event(
                token_id=position.get('token_id', 'unknown'),
                action='exit_failed',
                position_id=position_id,
                reason=str(e),
                price=signal_price,
                signal_price=signal_price
            )
            
            return False
            
    def get_current_price(self, token_id: str) -> float:
        """
        Get the current price for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Current price
        """
        try:
            return self.price_fetcher(token_id)
        except Exception as e:
            logger.error(f"Error fetching price for {token_id}: {str(e)}")
            raise
            
    def update_execution_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update execution parameters.
        
        Args:
            params: Dictionary of parameter values
        """
        if 'slippage_pct' in params:
            value = params['slippage_pct']
            if value >= 0:
                self.slippage_pct = value
                
        if 'execution_delay' in params:
            value = params['execution_delay']
            if value >= 0:
                self.execution_delay = value
                
        if 'simulate' in params:
            self.simulate = bool(params['simulate'])
            
        logger.info(f"Updated execution parameters: {params}")
        
    def get_execution_parameters(self) -> Dict[str, Any]:
        """
        Get current execution parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            'slippage_pct': self.slippage_pct,
            'execution_delay': self.execution_delay,
            'simulate': self.simulate
        }
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about trade executions.
        
        Returns:
            Dictionary containing execution statistics
        """
        return self.stats.copy()
        
    def set_execution_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for trade execution.
        
        Args:
            params: Dictionary of parameter values
        """
        if 'slippage_pct' in params:
            value = params['slippage_pct']
            if value >= 0:
                self.slippage_pct = value
                
        if 'execution_delay' in params:
            value = params['execution_delay']
            if value >= 0:
                self.execution_delay = value
                
        if 'simulate' in params:
            self.simulate = bool(params['simulate'])
            
        logger.info(f"Updated execution parameters: {params}")
        
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
        
    def _get_execution_price(self, token_id: str, signal_price: float, action: str) -> float:
        """
        Get execution price with simulated slippage.
        
        Args:
            token_id: ID of the token
            signal_price: Price at which the signal was generated
            action: 'entry' or 'exit'
            
        Returns:
            Execution price with slippage
        """
        if not self.simulate:
            # Use current market price from price fetcher
            return self.price_fetcher(token_id)
            
        # Generate random slippage
        slippage_factor = random.uniform(-self.slippage_pct, self.slippage_pct)
        
        # Apply slippage (always negative for entries, random for exits)
        if action == 'entry':
            # For entries, slippage is always negative (buying at higher price)
            slippage_factor = abs(slippage_factor)
        
        execution_price = signal_price * (1 + slippage_factor)
        
        logger.debug(f"Simulated {action} price for {token_id}: {execution_price} " 
                    f"(signal: {signal_price}, slippage: {slippage_factor:.2%})")
                    
        return execution_price
        
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """
        Update running average with new value.
        
        Args:
            current_avg: Current average value
            new_value: New value to include
            count: Number of values (including the new one)
            
        Returns:
            Updated average
        """
        if count <= 1:
            return new_value
        return current_avg + (new_value - current_avg) / count
        
    def _publish_trade_event(self, token_id: str, action: str, **kwargs) -> None:
        """
        Publish trade event.
        
        Args:
            token_id: ID of the token
            action: Action performed
            **kwargs: Additional event data
        """
        if not self.event_bus:
            return
            
        event_data = {
            'token_id': token_id,
            'action': action,
            'timestamp': datetime.now(),
            **kwargs
        }
        
        # Determine event type based on action
        if action in ('entry', 'entry_failed'):
            event_type = EventType.TRADE_ENTRY
        elif action in ('exit', 'exit_failed'):
            event_type = EventType.TRADE_EXIT
        else:
            event_type = EventType.TRADE_SIGNAL
            
        event = Event(
            event_type=event_type,
            data=event_data
        )
        
        self.event_bus.publish(event) 