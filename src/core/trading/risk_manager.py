#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Risk Manager Module

This module provides the DefaultRiskManager implementation for managing
risk in trading operations, including position sizing and stop-loss calculations.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from src.core.trading.interfaces import RiskManager

logger = logging.getLogger(__name__)


class DefaultRiskManager(RiskManager):
    """
    Default implementation of the RiskManager interface.
    
    This implementation provides standard risk management functionality
    with configurable risk parameters.
    """
    
    def __init__(self, 
                risk_per_trade: float = 0.05,
                max_position_pct: float = 0.2,
                min_trade_size: float = 0.01,
                max_concurrent_positions: int = 5,
                default_stop_loss_pct: float = 0.20,
                default_take_profit_pct: float = 0.30,
                risk_reward_ratio: float = 1.5,
                max_risk_exposure: float = 0.5):
        """
        Initialize the DefaultRiskManager.
        
        Args:
            risk_per_trade: Maximum risk per trade as a percentage of capital (default: 5%)
            max_position_pct: Maximum position size as a percentage of capital (default: 20%)
            min_trade_size: Minimum trade size in base currency (default: 0.01)
            max_concurrent_positions: Maximum number of concurrent positions (default: 5)
            default_stop_loss_pct: Default stop-loss percentage (default: 20%)
            default_take_profit_pct: Default take-profit percentage (default: 30%)
            risk_reward_ratio: Target risk-reward ratio (default: 1.5)
            max_risk_exposure: Maximum risk exposure as percentage of capital (default: 50%)
        """
        # Validate inputs
        if risk_per_trade <= 0 or risk_per_trade > 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        if max_position_pct <= 0 or max_position_pct > 1:
            raise ValueError("max_position_pct must be between 0 and 1")
        if min_trade_size <= 0:
            raise ValueError("min_trade_size must be greater than 0")
        if max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be greater than 0")
        if default_stop_loss_pct <= 0 or default_stop_loss_pct > 1:
            raise ValueError("default_stop_loss_pct must be between 0 and 1")
        if default_take_profit_pct <= 0:
            raise ValueError("default_take_profit_pct must be greater than 0")
        if risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio must be greater than 0")
        if max_risk_exposure <= 0 or max_risk_exposure > 1:
            raise ValueError("max_risk_exposure must be between 0 and 1")
            
        # Store parameters
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_trade_size = min_trade_size
        self.max_concurrent_positions = max_concurrent_positions
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.max_risk_exposure = max_risk_exposure
        
        # Additional token-specific risk adjustments (can be updated)
        self.token_risk_adjustments: Dict[str, float] = {}
        
        logger.info(f"Initialized DefaultRiskManager with risk_per_trade={risk_per_trade}, "
                    f"max_position_pct={max_position_pct}, min_trade_size={min_trade_size}")
    
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
        # Apply risk score to adjust risk per trade (higher risk score = higher risk)
        # Scale between 50% and 100% of max risk per trade
        adjusted_risk_per_trade = self.risk_per_trade * (0.5 + (risk_score * 0.5))
        
        # Apply token-specific risk adjustment if available
        token_adjustment = self.token_risk_adjustments.get(token_id, 1.0)
        adjusted_risk_per_trade *= token_adjustment
        
        # Calculate base position size
        position_size = available_capital * adjusted_risk_per_trade
        
        # Limit position size to max_position_pct
        max_position = available_capital * self.max_position_pct
        if position_size > max_position:
            position_size = max_position
            logger.debug(f"Position size limited to max_position_pct ({self.max_position_pct*100}%): {position_size}")
        
        # Ensure position size is at least min_trade_size
        if position_size < self.min_trade_size:
            # Only allow trade if we have enough capital
            if available_capital >= self.min_trade_size:
                position_size = self.min_trade_size
                logger.debug(f"Position size increased to min_trade_size: {self.min_trade_size}")
            else:
                logger.warning(f"Insufficient capital ({available_capital}) for min_trade_size ({self.min_trade_size})")
                return 0.0  # Cannot trade with insufficient capital
        
        logger.debug(f"Calculated position size for {token_id}: {position_size} (risk_score={risk_score}, "
                     f"adjusted_risk={adjusted_risk_per_trade}, capital={available_capital})")
        
        return position_size
    
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
        # Adjust stop-loss percentage based on risk score
        # Higher risk score = tighter stop loss (closer to entry price)
        adjusted_stop_loss_pct = self.default_stop_loss_pct * (0.7 + (risk_score * 0.6))
        
        # Cap at sensible limits (5% to 30%)
        adjusted_stop_loss_pct = max(0.05, min(0.30, adjusted_stop_loss_pct))
        
        # Apply token-specific adjustment if available
        token_adjustment = self.token_risk_adjustments.get(token_id, 1.0)
        adjusted_stop_loss_pct *= token_adjustment
        
        # Calculate stop-loss price
        stop_loss_price = entry_price * (1 - adjusted_stop_loss_pct)
        
        logger.debug(f"Calculated stop-loss for {token_id}: {stop_loss_price} (entry={entry_price}, "
                     f"percentage={adjusted_stop_loss_pct*100}%)")
        
        return stop_loss_price
    
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
        # Calculate risk amount
        risk_amount = entry_price - stop_loss_price
        
        # Calculate take-profit price based on risk-reward ratio
        reward_amount = risk_amount * self.risk_reward_ratio
        take_profit_price = entry_price + reward_amount
        
        # Ensure the take-profit price is at least the default percentage above entry
        min_take_profit = entry_price * (1 + self.default_take_profit_pct)
        take_profit_price = max(take_profit_price, min_take_profit)
        
        logger.debug(f"Calculated take-profit for {token_id}: {take_profit_price} (entry={entry_price}, "
                     f"stop_loss={stop_loss_price}, reward_ratio={self.risk_reward_ratio})")
        
        return take_profit_price
    
    def get_max_positions(self, available_capital: float) -> int:
        """
        Calculate the maximum number of concurrent positions.
        
        Args:
            available_capital: Available capital for trading
            
        Returns:
            Maximum number of concurrent positions
        """
        # Base maximum on capital - one position per minimum trade size up to max_concurrent_positions
        if self.min_trade_size <= 0:
            return self.max_concurrent_positions
            
        capital_based_max = math.floor(available_capital / self.min_trade_size)
        return min(capital_based_max, self.max_concurrent_positions)
    
    def get_max_risk_per_trade(self, available_capital: float) -> float:
        """
        Calculate the maximum risk per trade.
        
        Args:
            available_capital: Available capital for trading
            
        Returns:
            Maximum risk per trade in base currency
        """
        return available_capital * self.risk_per_trade
    
    def get_current_risk_exposure(self, open_positions: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate the current risk exposure across all open positions.
        
        Args:
            open_positions: Dictionary of open positions
            
        Returns:
            Current risk exposure as a percentage of total capital
        """
        if not open_positions:
            return 0.0
            
        # Calculate total position size
        total_position_size = sum(position.get('position_size', 0) for position in open_positions.values())
        
        # Calculate total potential loss based on stop-loss levels
        total_potential_loss = 0.0
        for position in open_positions.values():
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss_price', 0)
            position_size = position.get('position_size', 0)
            
            if entry_price > 0 and stop_loss > 0:
                loss_percentage = (entry_price - stop_loss) / entry_price
                potential_loss = position_size * loss_percentage
                total_potential_loss += potential_loss
        
        # Calculate total capital (sum of positions and remaining capital)
        remaining_capital = 0.0  # This should ideally be passed in, but we can estimate
        for position in open_positions.values():
            if 'initial_capital' in position:
                remaining_capital = position.get('initial_capital', 0) - total_position_size
                break
                
        total_capital = total_position_size + remaining_capital
        
        # Calculate risk exposure
        risk_exposure = 0.0
        if total_capital > 0:
            risk_exposure = total_potential_loss / total_capital
            
        return risk_exposure
    
    def set_risk_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters for risk management.
        
        Args:
            params: Dictionary of parameter values
        """
        if 'risk_per_trade' in params:
            value = params['risk_per_trade']
            if 0 < value <= 1:
                self.risk_per_trade = value
                
        if 'max_position_pct' in params:
            value = params['max_position_pct']
            if 0 < value <= 1:
                self.max_position_pct = value
                
        if 'min_trade_size' in params:
            value = params['min_trade_size']
            if value > 0:
                self.min_trade_size = value
                
        if 'max_concurrent_positions' in params:
            value = params['max_concurrent_positions']
            if value > 0:
                self.max_concurrent_positions = value
                
        if 'default_stop_loss_pct' in params:
            value = params['default_stop_loss_pct']
            if 0 < value <= 1:
                self.default_stop_loss_pct = value
                
        if 'default_take_profit_pct' in params:
            value = params['default_take_profit_pct']
            if value > 0:
                self.default_take_profit_pct = value
                
        if 'risk_reward_ratio' in params:
            value = params['risk_reward_ratio']
            if value > 0:
                self.risk_reward_ratio = value
                
        if 'max_risk_exposure' in params:
            value = params['max_risk_exposure']
            if 0 < value <= 1:
                self.max_risk_exposure = value
                
        if 'token_risk_adjustments' in params:
            adjustments = params['token_risk_adjustments']
            if isinstance(adjustments, dict):
                for token_id, adjustment in adjustments.items():
                    if adjustment > 0:
                        self.token_risk_adjustments[token_id] = adjustment
        
        logger.info(f"Updated risk parameters: {params}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Get current risk management parameters.
        
        Returns:
            Dictionary of parameter values
        """
        return {
            'risk_per_trade': self.risk_per_trade,
            'max_position_pct': self.max_position_pct,
            'min_trade_size': self.min_trade_size,
            'max_concurrent_positions': self.max_concurrent_positions,
            'default_stop_loss_pct': self.default_stop_loss_pct,
            'default_take_profit_pct': self.default_take_profit_pct,
            'risk_reward_ratio': self.risk_reward_ratio,
            'max_risk_exposure': self.max_risk_exposure,
            'token_risk_adjustments': self.token_risk_adjustments.copy()
        } 