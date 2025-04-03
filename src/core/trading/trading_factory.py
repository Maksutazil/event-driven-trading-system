#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading System Factory Module

This module provides a factory for creating different configurations
of trading systems, from simple to complex, with various components.
"""

import logging
import os
from typing import Dict, List, Set, Tuple, Any, Callable, Optional, Union

from src.core.events import EventBus
from src.core.ml import DefaultModelManager
from src.core.features import FeatureSystem
from src.core.features.providers.price_provider import PriceFeatureProvider
from src.core.trading.position_manager import DefaultPositionManager
from src.core.trading.trade_executor import DefaultTradeExecutor
from src.core.trading.risk_manager import DefaultRiskManager
from src.core.trading.trading_engine import DefaultTradingEngine
from src.core.trading.signal_generator import DefaultSignalGenerator
from src.core.trading.token_monitor import TokenMonitorThreadPool
from src.core.trading.interfaces import PositionManager, TradeExecutor, RiskManager, SignalGenerator
from src.core.trading.activity_analyzer import ActivityAnalyzer, TokenLifecycleState
from src.core.trading.graceful_exit_manager import GracefulExitManager, ExitPriority

logger = logging.getLogger(__name__)


class TradingSystemFactory:
    """
    Factory class for building and wiring together trading system components.
    """

    @staticmethod
    def merge_config(default_config: Dict[str, Any], user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge default configuration with user-provided configuration.
        User configuration overrides default values.
        
        Args:
            default_config: Default configuration dictionary
            user_config: User-provided configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        if not user_config:
            return default_config.copy()
            
        merged = default_config.copy()
        
        # Perform a shallow merge - for nested dicts, consider a deeper merge approach
        for key, value in user_config.items():
            merged[key] = value
            
        return merged
    
    @staticmethod
    def initialize_system_with_tokens(
        trading_system: Dict[str, Any], 
        tokens: List[str], 
        priority: int = 1
    ) -> None:
        """
        Initialize the trading system with a list of tokens to monitor.
        
        Args:
            trading_system: Trading system components created by the factory
            tokens: List of token IDs to monitor
            priority: Priority for token monitoring (lower number = higher priority)
        """
        trading_engine = trading_system.get('trading_engine')
        token_monitor = trading_system.get('token_monitor')
        
        if not trading_engine:
            logger.warning("Cannot initialize tokens: trading_engine not found in system")
            return
            
        logger.info(f"Adding {len(tokens)} tokens to trading system")
        for token_id in tokens:
            # Add token to trading engine
            trading_engine.add_token(token_id)
            
            # If token monitor exists, add token to it with the specified priority
            if token_monitor:
                token_monitor.add_token(token_id, priority=priority)
    
    @staticmethod
    def create_feature_system(
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None
    ) -> FeatureSystem:
        """
        Create and configure a feature system with standard features and providers.
        
        Args:
            event_bus: Event bus for communication
            config: Optional configuration parameters
            
        Returns:
            Configured FeatureSystem
        """
        config = config or {}
        
        # Create feature system
        logger.info("Creating feature system")
        feature_system = FeatureSystem(event_bus=event_bus)
        
        # Create and register price provider
        logger.info("Creating enhanced price provider")
        price_provider = EnhancedPriceProvider(
            name=config.get('price_provider_name', 'price_provider'),
            max_history=config.get('price_history_size', 100)
        )
        price_provider.register_with_event_bus(event_bus)
        feature_system.register_provider(price_provider)
        
        # Register signal features
        logger.info("Registering signal features")
        
        # Price momentum signal
        momentum_threshold = config.get('momentum_threshold', 0.05)
        momentum_sensitivity = config.get('momentum_sensitivity', 1.0)
        price_momentum_signal = PriceMomentumSignalFeature(
            threshold=momentum_threshold, 
            sensitivity=momentum_sensitivity
        )
        feature_system.register_feature(price_momentum_signal)
        
        # Volume spike signal if enabled
        if config.get('use_volume_spike_signal', True):
            volume_threshold = config.get('volume_threshold', 3.0)
            price_threshold = config.get('price_threshold', 1.0)
            volume_spike_signal = VolumeSpikeTradingSignalFeature(
                volume_threshold=volume_threshold,
                price_threshold=price_threshold
            )
            feature_system.register_feature(volume_spike_signal)
        
        logger.info("Feature system created and configured")
        return feature_system
    
    @staticmethod
    def create_trading_system(
        event_bus: EventBus,
        feature_system: FeatureSystem,
        price_fetcher: Callable[[str], Union[float, Tuple[float, float]]],
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a complete trading system with all components.
        
        Args:
            event_bus: The event bus for communication
            feature_system: The feature system for computing features
            price_fetcher: A function to fetch token prices
            config: Optional configuration parameters
            
        Returns:
            Dict with the created components
        """
        logger.info("Creating trading system with factory")
        config = config or {}
        
        # CONFIGURATION PARAMETERS WITH DEFAULTS
        
        # Position Manager configuration
        initial_capital = config.get("initial_capital", 10000.0)
        
        # Risk Manager configuration
        risk_per_trade = config.get("risk_per_trade", 0.02)
        max_position_pct = config.get("max_position_pct", 0.1)
        min_trade_size = config.get("min_trade_size", 0.01)
        max_concurrent_positions = config.get("max_concurrent_positions", 5)
        default_stop_loss_pct = config.get("default_stop_loss_pct", 0.05)
        default_take_profit_pct = config.get("default_take_profit_pct", 0.15)
        risk_reward_ratio = config.get("risk_reward_ratio", 3.0)
        max_risk_exposure = config.get("max_risk_exposure", 0.5)
        
        # Trade Executor configuration
        slippage_pct = config.get("slippage_pct", 0.005)
        execution_delay = config.get("execution_delay", 0.0)
        simulate = config.get("simulate", True)
        
        # Trading Engine configuration
        signal_threshold = config.get("signal_threshold", 0.7)
        signal_expiry_seconds = config.get("signal_expiry_seconds", 60.0)
        cooldown_seconds = config.get("cooldown_seconds", 3600.0)
        max_tokens_per_timepoint = config.get("max_tokens_per_timepoint", 3)
        model_prediction_weight = config.get("model_prediction_weight", 0.5)
        
        # Token Monitoring configuration
        enable_token_monitoring = config.get("enable_token_monitoring", True)
        max_monitor_threads = config.get("max_monitor_threads", 10)
        monitor_interval = config.get("monitor_interval", 1.0)
        health_check_interval = config.get("health_check_interval", 30.0)
        
        # Activity Analyzer reference
        activity_analyzer = config.get("activity_analyzer", None)
        
        # CREATE CORE COMPONENTS
        
        # Initialize components in order of dependency
        logger.info("Creating RiskManager")
        risk_manager = DefaultRiskManager(
            risk_per_trade=risk_per_trade,
            max_position_pct=max_position_pct,
            min_trade_size=min_trade_size,
            max_concurrent_positions=max_concurrent_positions,
            default_stop_loss_pct=default_stop_loss_pct,
            default_take_profit_pct=default_take_profit_pct,
            risk_reward_ratio=risk_reward_ratio,
            max_risk_exposure=max_risk_exposure
        )
        
        logger.info("Creating PositionManager")
        position_manager = DefaultPositionManager(
            event_bus=event_bus,
            initial_capital=initial_capital
        )
        
        logger.info("Creating TradeExecutor")
        trade_executor = DefaultTradeExecutor(
            event_bus=event_bus,
            position_manager=position_manager,
            risk_manager=risk_manager,
            price_fetcher=price_fetcher,
            slippage_pct=slippage_pct,
            execution_delay=execution_delay,
            simulate=simulate
        )
        
        # Create a ModelManager for machine learning integration
        logger.info("Creating ModelManager")
        model_manager = DefaultModelManager(
            event_bus=event_bus,
            feature_manager=feature_system  # FeatureSystem implements FeatureManager interface
        )
        
        # Create a SignalGenerator
        logger.info("Creating SignalGenerator")
        signal_generator = DefaultSignalGenerator(
            event_bus=event_bus,
            entry_threshold=signal_threshold,
            model_weight=model_prediction_weight,
            signal_expiry_seconds=signal_expiry_seconds
        )
        
        logger.info("Creating TradingEngine")
        trading_engine = DefaultTradingEngine(
            event_bus=event_bus,
            feature_system=feature_system,
            position_manager=position_manager,
            trade_executor=trade_executor,
            risk_manager=risk_manager,
            signal_threshold=signal_threshold,
            signal_expiry_seconds=signal_expiry_seconds,
            cooldown_seconds=cooldown_seconds,
            max_tokens_per_timepoint=max_tokens_per_timepoint,
            model_manager=model_manager,
            model_prediction_weight=model_prediction_weight,
            signal_generator=signal_generator
        )
        
        # Create token monitor if enabled
        token_monitor = None
        if enable_token_monitoring:
            data_feed_manager = config.get("data_feed_manager")
            if data_feed_manager:
                logger.info("Creating TokenMonitorThreadPool")
                token_monitor = TokenMonitorThreadPool(
                    event_bus=event_bus,
                    data_feed_manager=data_feed_manager,
                    feature_system=feature_system,
                    max_threads=max_monitor_threads,
                    monitor_interval=monitor_interval,
                    health_check_interval=health_check_interval,
                    trading_engine=trading_engine,
                    activity_analyzer=activity_analyzer  # Pass the activity analyzer
                )
                
                # Log whether the activity analyzer is connected
                if activity_analyzer:
                    logger.info("TokenMonitorThreadPool connected with ActivityAnalyzer")
                else:
                    logger.info("TokenMonitorThreadPool created without ActivityAnalyzer")
            else:
                logger.warning("Token monitoring enabled but no data_feed_manager provided")
        
        # Return all components
        components = {
            "trading_engine": trading_engine,
            "position_manager": position_manager,
            "trade_executor": trade_executor,
            "risk_manager": risk_manager,
            "event_bus": event_bus,
            "feature_system": feature_system,
            "model_manager": model_manager,  # Add model_manager to returned components
            "signal_generator": signal_generator,  # Add signal_generator to returned components
        }
        
        # Add token monitor if created
        if token_monitor:
            components["token_monitor"] = token_monitor
            
        logger.info("Trading system created successfully")
        return components
    
    @staticmethod
    def create_paper_trading_system(
        event_bus: EventBus,
        feature_system: FeatureSystem,
        price_fetcher: Callable[[str], Union[float, Tuple[float, float]]],
        data_feed_manager = None,
        initial_capital: float = 10000.0,
        config: Optional[Dict[str, Any]] = None,
        activity_analyzer = None  # Add activity_analyzer parameter
    ) -> Dict[str, Any]:
        """
        Create a paper trading system with default settings.
        
        Args:
            event_bus: The event bus for communication
            feature_system: The feature system for computing features
            price_fetcher: A function to fetch token prices
            data_feed_manager: Optional data feed manager for token monitoring
            initial_capital: Initial capital for paper trading
            config: Optional additional configuration
            activity_analyzer: Optional activity analyzer for token activity tracking
            
        Returns:
            Dict with the created components
        """
        logger.info(f"Creating paper trading system with initial capital: {initial_capital}")
        
        # Paper trading default configuration
        default_config = {
            "initial_capital": initial_capital,
            "simulate": True,
            "slippage_pct": 0.005,  # 0.5% slippage for paper trading
            "risk_per_trade": 0.02,  # 2% risk per trade
            "max_position_pct": 0.1,  # Max 10% of capital per position
            "data_feed_manager": data_feed_manager,
            "enable_token_monitoring": data_feed_manager is not None,
            "activity_analyzer": activity_analyzer,  # Include activity_analyzer in config
            "exit_timeout_seconds": 60.0,  # For GracefulExitManager
            "price_check_interval": 0.5,  # For GracefulExitManager
            "max_exit_attempts": 3,      # For GracefulExitManager
            "emergency_exit_threshold": -0.15,  # For GracefulExitManager
            "model_prediction_weight": 0.5,  # Default weight for model predictions
        }
        
        # Merge with user-provided config
        merged_config = TradingSystemFactory.merge_config(default_config, config)
        
        # Create the trading system
        components = TradingSystemFactory.create_trading_system(
            event_bus=event_bus,
            feature_system=feature_system,
            price_fetcher=price_fetcher,
            config=merged_config
        )
        
        # If activity_analyzer was provided but not included in the components, add it
        if activity_analyzer and "activity_analyzer" not in components:
            components["activity_analyzer"] = activity_analyzer
            
        # Create GracefulExitManager if not already present
        if "graceful_exit_manager" not in components and "position_manager" in components and "trade_executor" in components:
            logger.info("Creating GracefulExitManager for paper trading system")
            
            graceful_exit_manager = GracefulExitManager(
                event_bus=event_bus,
                position_manager=components["position_manager"],
                trade_executor=components["trade_executor"],
                activity_analyzer=components.get("activity_analyzer"),
                exit_timeout_seconds=merged_config.get("exit_timeout_seconds", 60.0),
                price_check_interval=merged_config.get("price_check_interval", 0.5),
                max_exit_attempts=merged_config.get("max_exit_attempts", 3),
                emergency_exit_threshold=merged_config.get("emergency_exit_threshold", -0.15)
            )
            
            components["graceful_exit_manager"] = graceful_exit_manager
            logger.info("GracefulExitManager added to paper trading system")
        
        # Start token monitoring if available
        token_monitor = components.get("token_monitor")
        auto_start_monitoring = merged_config.get("auto_start_monitoring", True)
        
        if token_monitor and auto_start_monitoring:
            logger.info("Auto-starting token monitoring for paper trading system")
            token_monitor.start()
        
        logger.info("Paper trading system created successfully")
        return components
    
    @staticmethod
    def create_backtest_trading_system(
        event_bus: EventBus,
        feature_system: FeatureSystem,
        price_fetcher: Callable[[str], Union[float, Tuple[float, float]]],
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.005,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a backtest trading system with specific configs for backtesting.
        
        This method is intended for historical data testing and does not include
        components like TokenMonitorThreadPool, ActivityAnalyzer, or GracefulExitManager
        which are designed for real-time monitoring and management. Since backtesting
        processes historical data in a controlled manner, real-time exit management
        is not needed.
        
        Args:
            event_bus: The event bus for communication
            feature_system: The feature system for computing features
            price_fetcher: A function to fetch token prices
            initial_capital: Initial capital for backtest
            slippage_pct: Slippage percentage to simulate real-world conditions
            config: Additional configuration options
            
        Returns:
            Dict with the created components
        """
        logger.info(f"Creating backtest trading system with initial capital: {initial_capital}")
        
        # Backtest specific configuration
        default_config = {
            "initial_capital": initial_capital,
            "simulate": True,
            "slippage_pct": slippage_pct,
            "signal_expiry_seconds": 0,  # No signal expiry in backtesting
            "cooldown_seconds": 0,       # No cooldown in backtesting
            "enable_token_monitoring": False,  # No token monitoring needed for backtesting
        }
        
        # Merge with user-provided config
        merged_config = TradingSystemFactory.merge_config(default_config, config)
        
        # Create the trading system
        components = TradingSystemFactory.create_trading_system(
            event_bus=event_bus,
            feature_system=feature_system,
            price_fetcher=price_fetcher,
            config=merged_config
        )
        
        logger.info("Backtest trading system created successfully")
        return components
    
    @staticmethod
    def create_complete_trading_system(
        event_bus: EventBus,
        feature_system: Optional[FeatureSystem] = None,
        data_feed_manager: Optional[Any] = None,
        price_fetcher: Optional[Callable[[str], Union[float, Tuple[float, float]]]] = None,
        config: Optional[Dict[str, Any]] = None,
        subscription_keys: Optional[Dict[str, List[str]]] = None,
        model_paths: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a complete trading system with all components.
        
        Args:
            event_bus: The event bus for communication
            feature_system: The feature system for computing features (optional)
            data_feed_manager: Data feed manager for accessing market data (optional)
            price_fetcher: A function to fetch token prices (optional)
            config: Optional configuration parameters
            subscription_keys: Optional subscription keys for data sources
            model_paths: Optional dictionary mapping model IDs to file paths
            
        Returns:
            Dict with the created components
        """
        logger.info("Creating trading system with factory")
        config = config or {}
        
        # Add data feed manager to config
        config["data_feed_manager"] = data_feed_manager
        
        # Add model prediction weight to config if not already present
        if "model_prediction_weight" not in config:
            config["model_prediction_weight"] = 0.5
        
        # Create feature system
        if feature_system is None:
            logger.info("Creating FeatureSystem")
            feature_system = FeatureSystem(event_bus=event_bus)
            
            # Create and register providers
            logger.info("Creating feature providers")
            price_provider = PriceFeatureProvider(name="price_provider", max_history=200)
            feature_system.register_provider(price_provider)
        else:
            logger.info("Using provided FeatureSystem")
        
        # Create activity analyzer
        logger.info("Creating ActivityAnalyzer")
        activity_check_interval = config.get('activity_check_interval', 30.0)
        inactivity_threshold = config.get('inactivity_threshold', 0.2)
        inactive_checks = config.get('inactive_checks_before_state_change', 3)
        
        activity_analyzer = ActivityAnalyzer(
            event_bus=event_bus,
            activity_check_interval=activity_check_interval,
            inactivity_threshold=inactivity_threshold,
            inactive_checks_before_state_change=inactive_checks
        )
        
        # Start the activity analyzer
        activity_analyzer.start()
        logger.info(f"ActivityAnalyzer created and started with check_interval={activity_check_interval}s")
        
        # Add activity analyzer to config
        config["activity_analyzer"] = activity_analyzer
        
        # Create the core trading system
        components = TradingSystemFactory.create_trading_system(
            event_bus=event_bus,
            feature_system=feature_system,
            price_fetcher=price_fetcher,
            config=config
        )
        
        # Add feature system and activity analyzer to components
        components["feature_system"] = feature_system
        components["activity_analyzer"] = activity_analyzer
        
        # Ensure token_monitor is always in the components dictionary
        if "token_monitor" not in components:
            components["token_monitor"] = None
        
        # Create GracefulExitManager
        if "graceful_exit_manager" not in components and "position_manager" in components and "trade_executor" in components:
            logger.info("Creating GracefulExitManager")
            
            graceful_exit_manager = GracefulExitManager(
                event_bus=event_bus,
                position_manager=components["position_manager"],
                trade_executor=components["trade_executor"],
                exit_timeout_seconds=config.get("exit_timeout_seconds", 60.0)
            )
            
            components["graceful_exit_manager"] = graceful_exit_manager
            logger.info(f"GracefulExitManager created with exit_timeout={config.get('exit_timeout_seconds', 60.0)}s")
        
        # Initialize the system with tokens if provided
        if subscription_keys and 'tokens' in subscription_keys:
            tokens = subscription_keys['tokens']
            TradingSystemFactory.initialize_system_with_tokens(
                trading_system=components,
                tokens=tokens,
                priority=config.get('token_priority', 1)
            )
        
        # Load ML models if paths are provided
        if model_paths and "model_manager" in components:
            model_manager = components["model_manager"]
            for model_id, model_path in model_paths.items():
                model_type = config.get(f"model_{model_id}_type", "classification")
                try:
                    logger.info(f"Loading model {model_id} from {model_path}")
                    model_manager.load_model(model_id, model_path, model_type)
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")
            
            # Connect model manager to trading engine if not already connected
            if "trading_engine" in components and components["trading_engine"].model_manager is None:
                components["trading_engine"].model_manager = model_manager
                logger.info("Connected ModelManager to TradingEngine")
        
        logger.info("Complete trading system created successfully")
        return components
    
    @staticmethod
    def shutdown_trading_system(trading_system: Dict[str, Any]) -> None:
        """
        Shutdown all components of a trading system in the proper order.
        
        This method ensures that all components are properly stopped and resources
        are cleaned up to prevent memory leaks and ensure graceful shutdown.
        
        Args:
            trading_system: Dictionary of trading system components created by the factory
        """
        if not trading_system:
            logger.warning("No trading system provided for shutdown")
            return
            
        logger.info("Beginning systematic shutdown of trading system components")
        
        # Step 1: Process graceful exit of positions if GracefulExitManager exists
        if 'graceful_exit_manager' in trading_system and 'position_manager' in trading_system:
            graceful_exit_manager = trading_system['graceful_exit_manager']
            position_manager = trading_system['position_manager']
            
            # Check if there are open positions that need to be exited
            open_positions = position_manager.get_open_positions()
            if open_positions:
                logger.info(f"Found {len(open_positions)} open positions to exit gracefully")
                
                try:
                    # Begin graceful exit process
                    graceful_exit_manager.begin_graceful_exit()
                    
                    # Wait for exits to complete with timeout
                    completed = graceful_exit_manager.wait_for_exit_completion()
                    
                    if completed:
                        # Get exit stats
                        stats = graceful_exit_manager.get_exit_stats()
                        logger.info(f"Graceful exit completed: {stats['successful_exits']} successful, "
                                   f"{stats['failed_exits']} failed, PnL saved: {stats['total_pnl_saved']:.2f}")
                    else:
                        logger.warning("Graceful exit timed out, some positions may remain open")
                except Exception as e:
                    logger.error(f"Error during graceful exit: {e}", exc_info=True)
            else:
                logger.info("No open positions to exit")
                
            # Clean up graceful exit manager
            try:
                graceful_exit_manager.cleanup()
                logger.info("GracefulExitManager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up graceful exit manager: {e}", exc_info=True)
        
        # Step 2: Stop token monitoring threads
        if 'token_monitor' in trading_system and trading_system['token_monitor'] is not None:
            logger.info("Stopping token monitor thread pool")
            try:
                trading_system['token_monitor'].stop()
                logger.info("Token monitor thread pool stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping token monitor thread pool: {e}", exc_info=True)
        elif 'token_monitor' in trading_system:
            logger.warning("Token monitor exists in trading system but is None, skipping stop")
        
        # Step 3: Stop activity analyzer
        if 'activity_analyzer' in trading_system:
            logger.info("Cleaning up activity analyzer")
            try:
                trading_system['activity_analyzer'].cleanup()
                logger.info("Activity analyzer cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up activity analyzer: {e}", exc_info=True)
        
        # Step 4: Clean up trading engine
        if 'trading_engine' in trading_system and hasattr(trading_system['trading_engine'], 'cleanup'):
            logger.info("Cleaning up trading engine")
            try:
                trading_system['trading_engine'].cleanup()
                logger.info("Trading engine cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up trading engine: {e}", exc_info=True)
                
        # Step 5: Clean up position manager if needed
        if 'position_manager' in trading_system and hasattr(trading_system['position_manager'], 'cleanup'):
            logger.info("Cleaning up position manager")
            try:
                trading_system['position_manager'].cleanup()
                logger.info("Position manager cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up position manager: {e}", exc_info=True)
                
        # Step 6: Clean up trade executor if needed
        if 'trade_executor' in trading_system and hasattr(trading_system['trade_executor'], 'cleanup'):
            logger.info("Cleaning up trade executor")
            try:
                trading_system['trade_executor'].cleanup()
                logger.info("Trade executor cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up trade executor: {e}", exc_info=True)
        
        logger.info("Trading system components shutdown complete") 