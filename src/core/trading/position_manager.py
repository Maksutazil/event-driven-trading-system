#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Position Manager Module

This module provides the DefaultPositionManager implementation for managing
trading positions, including opening, tracking, and closing positions.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timezone

from src.core.events import EventBus, Event, EventType
from src.core.trading.interfaces import PositionManager

logger = logging.getLogger(__name__)


class DefaultPositionManager(PositionManager):
    """
    Default implementation of the PositionManager interface.
    
    This implementation provides position management functionality with
    event-based notifications for position changes.
    """
    
    def __init__(self, event_bus: EventBus, initial_capital: float = 10000.0):
        """
        Initialize the DefaultPositionManager.
        
        Args:
            event_bus: Event bus for publishing position updates
            initial_capital: Initial trading capital
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be greater than 0")
            
        self.event_bus = event_bus
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.overall_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"Initialized DefaultPositionManager with initial_capital={initial_capital}")
    
    def open_position(self, token_id: str, entry_price: float, position_size: float, 
                     stop_loss: float, take_profit: float, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Open a new position.
        
        Args:
            token_id: ID of the token
            entry_price: Entry price of the position
            position_size: Size of the position in base currency
            stop_loss: Stop-loss price
            take_profit: Take-profit price
            metadata: Additional metadata about the position
            
        Returns:
            Position ID
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError("entry_price must be greater than 0")
        if position_size <= 0:
            raise ValueError("position_size must be greater than 0")
        if stop_loss <= 0 or stop_loss >= entry_price:
            raise ValueError("stop_loss must be greater than 0 and less than entry_price")
        if take_profit <= entry_price:
            raise ValueError("take_profit must be greater than entry_price")
        if position_size > self.available_capital:
            raise ValueError(f"Insufficient capital: {self.available_capital} < {position_size}")
            
        # Generate position ID
        position_id = str(uuid.uuid4())
        
        # Create position object
        timestamp = datetime.now()
        position = {
            'id': position_id,
            'token_id': token_id,
            'entry_price': entry_price,
            'current_price': entry_price,
            'position_size': position_size,
            'stop_loss_price': stop_loss,
            'take_profit_price': take_profit,
            'entry_time': timestamp,
            'status': 'open',
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0,
            'metadata': metadata or {}
        }
        
        # Update available capital
        self.available_capital -= position_size
        
        # Store position
        self.positions[position_id] = position
        
        # Publish position update event
        self._publish_position_update(position_id, position, 'open')
        
        logger.info(f"Opened position {position_id} for {token_id} at {entry_price} with size {position_size}")
        return position_id
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_reason: str = 'manual') -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            position_id: ID of the position
            exit_price: Exit price of the position
            exit_reason: Reason for closing the position
            
        Returns:
            Position details with realized P&L
        """
        # Validate position exists
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} does not exist")
            
        # Get position details
        position = self.positions[position_id]
        if position['status'] != 'open':
            raise ValueError(f"Position {position_id} is already {position['status']}")
            
        # Calculate realized P&L
        entry_price = position['entry_price']
        position_size = position['position_size']
        price_change = exit_price - entry_price
        realized_pnl = position_size * (price_change / entry_price)
        realized_pnl_pct = price_change / entry_price
        
        # Update position
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['exit_reason'] = exit_reason
        position['status'] = 'closed'
        position['realized_pnl'] = realized_pnl
        position['realized_pnl_pct'] = realized_pnl_pct
        
        # Update available capital
        self.available_capital += position_size + realized_pnl
        
        # Update overall performance metrics
        self.overall_pnl += realized_pnl
        if realized_pnl > 0:
            self.win_count += 1
        elif realized_pnl < 0:
            self.loss_count += 1
            
        # Move to history
        self.position_history.append(position.copy())
        del self.positions[position_id]
        
        # Publish position update event
        self._publish_position_update(position_id, position, 'close')
        
        logger.info(f"Closed position {position_id} at {exit_price} with P&L {realized_pnl:.2f} ({realized_pnl_pct:.2%})")
        return position
    
    def update_position(self, position_id: str, current_price: float) -> Dict[str, Any]:
        """
        Update an existing position with the current price.
        
        Args:
            position_id: ID of the position
            current_price: Current price of the token
            
        Returns:
            Updated position details
        """
        # Validate position exists
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} does not exist")
            
        # Get position details
        position = self.positions[position_id]
        if position['status'] != 'open':
            return position  # Already closed, no update needed
            
        # Update current price
        position['current_price'] = current_price
        
        # Calculate unrealized P&L
        entry_price = position['entry_price']
        position_size = position['position_size']
        price_change = current_price - entry_price
        unrealized_pnl = position_size * (price_change / entry_price)
        unrealized_pnl_pct = price_change / entry_price
        
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pnl_pct'] = unrealized_pnl_pct
        
        # Check for stop-loss or take-profit
        should_close = False
        exit_reason = ''
        
        if current_price <= position['stop_loss_price']:
            should_close = True
            exit_reason = 'stop_loss'
        elif current_price >= position['take_profit_price']:
            should_close = True
            exit_reason = 'take_profit'
            
        # If should close, close the position
        if should_close:
            return self.close_position(position_id, current_price, exit_reason)
            
        # Publish position update event
        self._publish_position_update(position_id, position, 'update')
            
        return position
    
    def get_position(self, position_id: str) -> Dict[str, Any]:
        """
        Get details of a specific position.
        
        Args:
            position_id: ID of the position
            
        Returns:
            Position details
        """
        # Check if in active positions
        if position_id in self.positions:
            return self.positions[position_id].copy()
            
        # Check if in position history
        for position in self.position_history:
            if position['id'] == position_id:
                return position.copy()
                
        raise ValueError(f"Position {position_id} not found")
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open positions
        """
        return {pid: pos.copy() for pid, pos in self.positions.items()}
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get history of closed positions.
        
        Returns:
            List of closed position details
        """
        return [pos.copy() for pos in self.position_history]
    
    def get_positions_for_token(self, token_id: str) -> List[Dict[str, Any]]:
        """
        Get all positions (open and closed) for a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            List of position details
        """
        open_positions = [pos for pos in self.positions.values() if pos['token_id'] == token_id]
        closed_positions = [pos for pos in self.position_history if pos['token_id'] == token_id]
        return open_positions + closed_positions
    
    def get_open_positions_for_token(self, token_id: str) -> List[Dict[str, Any]]:
        """
        Get open positions for a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            List of open position details
        """
        return [pos.copy() for pos in self.positions.values() if pos['token_id'] == token_id]
    
    def get_position_by_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of an open position for a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary containing position details, or None if not found
        """
        open_positions = self.get_open_positions_for_token(token_id)
        if open_positions:
            return open_positions[0]  # Return first open position for this token
        return None
    
    def get_closed_positions(self, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get closed positions.
        
        Args:
            limit: Optional maximum number of positions to return
            
        Returns:
            Dictionary mapping position IDs to position details
        """
        if limit is not None:
            history = self.position_history[-limit:]
        else:
            history = self.position_history
            
        return {pos['id']: pos.copy() for pos in history}
    
    def get_position_count(self) -> Dict[str, int]:
        """
        Get count of open and closed positions.
        
        Returns:
            Dictionary with keys 'open' and 'closed' and their counts
        """
        return {
            'open': len(self.positions),
            'closed': len(self.position_history)
        }
    
    def get_position_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for positions.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.get_performance_metrics()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        # Calculate average win and loss
        avg_win = 0.0
        avg_loss = 0.0
        total_win_amount = 0.0
        total_loss_amount = 0.0
        
        for position in self.position_history:
            if 'realized_pnl' in position:
                pnl = position['realized_pnl']
                if pnl > 0:
                    total_win_amount += pnl
                elif pnl < 0:
                    total_loss_amount += abs(pnl)
        
        if self.win_count > 0:
            avg_win = total_win_amount / self.win_count
        if self.loss_count > 0:
            avg_loss = total_loss_amount / self.loss_count
            
        # Calculate profit factor
        profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else float('inf')
        
        # Calculate return metrics
        roi = self.overall_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': self.overall_pnl,
            'roi': roi,
            'initial_capital': self.initial_capital,
            'current_capital': self.get_current_capital()
        }
    
    def get_current_capital(self) -> float:
        """
        Get current total capital.
        
        Returns:
            Current capital including unrealized P&L
        """
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        return self.available_capital + unrealized_pnl
    
    def get_available_capital(self) -> float:
        """
        Get available capital for new positions.
        
        Returns:
            Available capital
        """
        return self.available_capital
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """
        Reset the position manager.
        
        Args:
            initial_capital: New initial capital (optional)
        """
        if initial_capital is not None:
            if initial_capital <= 0:
                raise ValueError("initial_capital must be greater than 0")
            self.initial_capital = initial_capital
            
        self.available_capital = self.initial_capital
        self.positions = {}
        self.position_history = []
        self.overall_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info(f"Reset PositionManager with initial_capital={self.initial_capital}")
    
    def _publish_position_update(self, position_id: str, position: Dict[str, Any], 
                                action: str) -> None:
        """
        Publish position update event.
        
        Args:
            position_id: ID of the position
            position: Position details
            action: Action performed (open, update, close)
        """
        if not self.event_bus:
            return
            
        event_data = {
            'position_id': position_id,
            'token_id': position['token_id'],
            'action': action,
            'position': position.copy()
        }
        
        event = Event(
            event_type=EventType.POSITION_UPDATE,
            data=event_data
        )
        
        self.event_bus.publish(event) 