#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Module for the Event-Driven Trading System.

This module provides the necessary components for trading operations,
including signal generation, trade execution, position management, and risk handling.

Core components:
- TradingEngine: Interface for trading operations
- DefaultTradingEngine: Implementation of the TradingEngine interface
- PositionManager: Interface for managing trading positions
- DefaultPositionManager: Implementation of the PositionManager interface
- SignalGenerator: Interface for generating trading signals
- DefaultSignalGenerator: Implementation of the SignalGenerator interface
- RiskManager: Interface for risk management
- DefaultRiskManager: Implementation of the RiskManager interface
- TradingErrorHandler: Centralized error handling for trading components
- TradingSystemFactory: Factory class for creating trading system components
"""

# Import interfaces
from src.core.trading.interfaces import (
    TradingEngine, PositionManager, SignalGenerator, RiskManager,
    TradingSignal, Position
)

# Import implementations
from src.core.trading.trading_engine import DefaultTradingEngine
from src.core.trading.position_manager import DefaultPositionManager
from src.core.trading.signal_generator import DefaultSignalGenerator
from src.core.trading.risk_manager import DefaultRiskManager
from src.core.trading.trading_factory import TradingSystemFactory

# Import error handling
from src.core.trading.error_handler import (
    TradingErrorHandler,
    TradingError, SignalGenerationError, TradeExecutionError, 
    PositionManagementError, RiskCalculationError, TokenMonitoringError,
    PriceFetchError, InvalidParameterError, TradingConfigError
)

__all__ = [
    # Interfaces
    'TradingEngine',
    'PositionManager',
    'SignalGenerator',
    'RiskManager',
    'TradingSignal',
    'Position',
    
    # Implementations
    'DefaultTradingEngine',
    'DefaultPositionManager',
    'DefaultSignalGenerator',
    'DefaultRiskManager',
    'TradingSystemFactory',
    
    # Error handling
    'TradingErrorHandler',
    'TradingError',
    'SignalGenerationError',
    'TradeExecutionError',
    'PositionManagementError',
    'RiskCalculationError',
    'TokenMonitoringError',
    'PriceFetchError',
    'InvalidParameterError',
    'TradingConfigError'
] 