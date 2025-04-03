#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Component Interfaces

This module provides the interfaces for the trading components:
- TradingEngine: Responsible for making trading decisions
- PositionManager: Responsible for managing open positions
- TradeExecutor: Responsible for executing trades
- RiskManager: Responsible for risk calculations and position sizing
- SignalGenerator: Responsible for generating trading signals
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union, NamedTuple
from datetime import datetime

from src.core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class TradingSignal(NamedTuple):
    """
    Data class representing a trading signal.
    
    Attributes:
        token_id: ID of the token
        signal_type: Type of signal (entry, exit, etc.)
        score: Signal score/strength (-1.0 to 1.0)
        price: Price at which the signal was generated
        timestamp: Time the signal was generated
        expiry: Time when the signal expires
        metadata: Additional signal metadata
    """
    token_id: str
    signal_type: str  # 'entry', 'exit', 'adjust', etc.
    score: float  # -1.0 (strong sell) to 1.0 (strong buy)
    price: float
    timestamp: datetime
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    @property
    def is_valid(self) -> bool:
        """Check if the signal is still valid (not expired)."""
        if self.expiry is None:
            return True
        return datetime.now() < self.expiry
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy signal."""
        return self.score > 0 and self.signal_type == 'entry'
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell signal."""
        return self.score < 0 or self.signal_type == 'exit'


class Position(NamedTuple):
    """
    Data class representing a trading position.
    
    Attributes:
        position_id: Unique ID for the position
        token_id: ID of the token
        entry_price: Entry price of the position
        position_size: Size of the position in base currency
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        entry_time: Time the position was entered
        metadata: Additional position metadata
        current_price: Current price (optional, for position updates)
        unrealized_pnl: Unrealized profit/loss (optional)
        unrealized_pnl_pct: Unrealized profit/loss as a percentage (optional)
    """
    position_id: str
    token_id: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    metadata: Dict[str, Any] = {}
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    
    @property
    def is_profitable(self) -> bool:
        """Check if the position is currently profitable."""
        if self.current_price is None:
            return False
        return self.current_price > self.entry_price
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate the risk/reward ratio of the position."""
        if self.stop_loss == self.entry_price:
            return 0.0  # Avoid division by zero
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0.0


class TradingEngine(ABC):
    """
    Interface for trading engines.
    
    Trading engines are responsible for making trading decisions based on
    features and market data.
    """
    
    @abstractmethod
    def process_timepoint(self, token_id: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a timepoint for a token.
        
        Args:
            token_id: Identifier for the token
            features: Dictionary of features
            
        Returns:
            List of generated signals
        """
        pass
    
    @abstractmethod
    def get_active_tokens(self) -> Set[str]:
        """
        Get the set of active tokens.
        
        Returns:
            Set of token identifiers
        """
        pass
    
    @abstractmethod
    def add_token(self, token_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a token to the trading engine.
        
        Args:
            token_id: Identifier for the token
            metadata: Optional metadata for the token
            
        Returns:
            Whether the token was added successfully
        """
        pass
    
    @abstractmethod
    def remove_token(self, token_id: str) -> bool:
        """
        Remove a token from the trading engine.
        
        Args:
            token_id: Identifier for the token
            
        Returns:
            Whether the token was removed successfully
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for the trading engine.
        
        Args:
            params: Dictionary of parameter values
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for the trading engine.
        
        Returns:
            Dictionary of parameter values
        """
        pass
        
    def set_model_prediction_weight(self, weight: float) -> None:
        """
        Set the weight to give model predictions in trading decisions.
        
        Args:
            weight: Weight between 0.0 (ignore predictions) and 1.0 (rely heavily on predictions)
        """
        params = {'model_prediction_weight': weight}
        self.set_parameters(params)
        
    def get_model_prediction_weight(self) -> float:
        """
        Get the current weight given to model predictions.
        
        Returns:
            Current weight value
        """
        return self.get_parameters().get('model_prediction_weight', 0.0)


class PositionManager(ABC):
    """
    Interface for the position manager component.
    
    The position manager is responsible for tracking and managing open positions.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def update_position(self, position_id: str, current_price: float) -> Dict[str, Any]:
        """
        Update an existing position with the current price.
        
        Args:
            position_id: ID of the position
            current_price: Current price of the token
            
        Returns:
            Updated position details
        """
        pass
    
    @abstractmethod
    def get_position(self, position_id: str) -> Dict[str, Any]:
        """
        Get details of a specific position.
        
        Args:
            position_id: ID of the position
            
        Returns:
            Position details
        """
        pass
    
    @abstractmethod
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open positions
        """
        pass
    
    @abstractmethod
    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get history of closed positions.
        
        Returns:
            List of closed position details
        """
        pass
    
    @abstractmethod
    def get_position_by_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of an open position for a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary containing position details, or None if not found
        """
        pass
    
    @abstractmethod
    def get_open_positions_for_token(self, token_id: str) -> List[Dict[str, Any]]:
        """
        Get open positions for a specific token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            List of open position details
        """
        pass
    
    @abstractmethod
    def get_available_capital(self) -> float:
        """
        Get available capital for new positions.
        
        Returns:
            Available capital
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get overall performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        pass


class TradeExecutor(ABC):
    """
    Interface for the trade executor component.
    
    The trade executor is responsible for executing trades based on signals
    from the trading engine.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_current_price(self, token_id: str) -> float:
        """
        Get the current price for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Current price
        """
        pass
    
    @abstractmethod
    def set_execution_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for trade execution.
        
        Args:
            params: Dictionary of parameter values
        """
        pass
    
    @abstractmethod
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about trade executions.
        
        Returns:
            Dictionary containing execution statistics
        """
        pass


class RiskManager(ABC):
    """
    Interface for the risk manager component.
    
    The risk manager is responsible for calculating position sizes,
    stop-loss levels, and managing overall risk exposure.
    """
    
    @abstractmethod
    def calculate_position_size(self, token_id: str, entry_price: float,
                               risk_score: float, available_capital: float) -> float:
        """
        Calculate the appropriate position size for a trade.
        
        Args:
            token_id: ID of the token
            entry_price: Entry price of the position
            risk_score: Risk score for the trade (0.0 to 1.0)
            available_capital: Available capital for trading
            
        Returns:
            Position size in base currency
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, token_id: str, entry_price: float,
                           position_size: float, risk_score: float) -> float:
        """
        Calculate the appropriate stop-loss price for a position.
        
        Args:
            token_id: ID of the token
            entry_price: Entry price of the position
            position_size: Size of the position in base currency
            risk_score: Risk score for the trade (0.0 to 1.0)
            
        Returns:
            Stop-loss price
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, token_id: str, entry_price: float,
                             stop_loss_price: float) -> float:
        """
        Calculate the appropriate take-profit price for a position.
        
        Args:
            token_id: ID of the token
            entry_price: Entry price of the position
            stop_loss_price: Stop-loss price of the position
            
        Returns:
            Take-profit price
        """
        pass
    
    @abstractmethod
    def get_max_positions(self, available_capital: float) -> int:
        """
        Calculate the maximum number of concurrent positions.
        
        Args:
            available_capital: Available capital for trading
            
        Returns:
            Maximum number of concurrent positions
        """
        pass
    
    @abstractmethod
    def get_max_risk_per_trade(self, available_capital: float) -> float:
        """
        Calculate the maximum risk per trade.
        
        Args:
            available_capital: Available capital for trading
            
        Returns:
            Maximum risk per trade in base currency
        """
        pass
    
    @abstractmethod
    def get_current_risk_exposure(self, open_positions: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate the current risk exposure across all open positions.
        
        Args:
            open_positions: Dictionary of open positions
            
        Returns:
            Current risk exposure as a percentage of total capital
        """
        pass
    
    @abstractmethod
    def set_risk_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for risk management.
        
        Args:
            params: Dictionary of parameter values
        """
        pass
    
    @abstractmethod
    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Get current risk management parameters.
        
        Returns:
            Dictionary of parameter values
        """
        pass


class SignalGenerator(ABC):
    """
    Interface for the signal generator component.
    
    The signal generator is responsible for analyzing features and market data
    to generate trading signals (entry, exit, etc.).
    """
    
    @abstractmethod
    def generate_signals(self, token_id: str, features: Dict[str, Any], 
                       timestamp: datetime) -> List[TradingSignal]:
        """
        Generate trading signals for a token based on features.
        
        Args:
            token_id: ID of the token
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated trading signals
        """
        pass
    
    @abstractmethod
    def generate_entry_signals(self, token_id: str, features: Dict[str, Any], 
                             timestamp: datetime) -> List[TradingSignal]:
        """
        Generate entry signals for a token based on features.
        
        Args:
            token_id: ID of the token
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated entry signals
        """
        pass
    
    @abstractmethod
    def generate_exit_signals(self, token_id: str, position: Position, 
                            features: Dict[str, Any], timestamp: datetime) -> List[TradingSignal]:
        """
        Generate exit signals for a position based on features.
        
        Args:
            token_id: ID of the token
            position: Current position
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            List of generated exit signals
        """
        pass
    
    @abstractmethod
    def evaluate_model_prediction(self, token_id: str, prediction: Any, 
                                features: Dict[str, Any], timestamp: datetime) -> TradingSignal:
        """
        Evaluate a model prediction to generate a trading signal.
        
        Args:
            token_id: ID of the token
            prediction: Model prediction
            features: Dictionary of features
            timestamp: Current timestamp
            
        Returns:
            Generated trading signal
        """
        pass
    
    @abstractmethod
    def set_signal_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for signal generation.
        
        Args:
            params: Dictionary of parameter values
        """
        pass
    
    @abstractmethod
    def get_signal_parameters(self) -> Dict[str, Any]:
        """
        Get current signal generation parameters.
        
        Returns:
            Dictionary of parameter values
        """
        pass 