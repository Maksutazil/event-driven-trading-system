import logging
import time
from typing import Dict, Any, List, Optional

from src.core.events import Event, EventType, EventBus, BaseEventSubscriber

logger = logging.getLogger(__name__)

class PortfolioManager(BaseEventSubscriber):
    """
    Portfolio manager component that tracks positions and performance.
    
    This component subscribes to trade execution and position update events,
    and maintains a record of the portfolio.
    """
    
    def __init__(self, event_bus: EventBus, initial_capital: float = 1000.0):
        super().__init__(event_bus)
        
        # Portfolio state
        self.positions: Dict[str, Dict[str, Any]] = {} # symbol -> position details
        self.trades: List[Dict[str, Any]] = []
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Performance metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Register event handlers
        # It's often better to register handlers externally or use decorators,
        # but this works for self-contained example.
        self.register_handler(EventType.TRADE_EXECUTION, self.handle_trade_execution)
        self.register_handler(EventType.POSITION_UPDATE, self.handle_position_update)
        
        logger.info(f"Portfolio manager initialized with initial capital: {initial_capital:.2f}")
    
    def handle_trade_execution(self, event: Event) -> None:
        """
        Handle trade execution events to update cash and PnL.
        
        Args:
            event: The trade execution event
        """
        trade_data = event.data
        symbol = trade_data.get("symbol")
        side = trade_data.get("side")
        timestamp = trade_data.get("timestamp", time.time())
        
        if not symbol or not side:
            logger.warning(f"Received incomplete trade execution event: {trade_data}")
            return
            
        # Record the trade
        self.trades.append(trade_data)
        self.total_trades += 1
        
        # Update cash and track PnL
        if side == "buy":
            cost = trade_data.get("cost", 0.0)
            if cost == 0.0:
                 cost = trade_data.get("size", 0.0) * trade_data.get("price", 0.0)
            self.cash -= cost
            logger.debug(f"Trade Exec (BUY {symbol}): Cost={cost:.4f}, New Cash={self.cash:.4f}")
        elif side == "sell":
            proceeds = trade_data.get("proceeds", 0.0)
            pnl = trade_data.get("pnl", 0.0) # PnL should ideally be calculated by the position manager/executor
            self.cash += proceeds
            self.total_pnl += pnl
            
            # Track win/loss rate
            if pnl > 0:
                self.winning_trades += 1
            elif pnl < 0:
                self.losing_trades += 1
            logger.debug(f"Trade Exec (SELL {symbol}): Proceeds={proceeds:.4f}, PnL={pnl:.4f}, New Cash={self.cash:.4f}")
        else:
             logger.warning(f"Unrecognized trade side '{side}' in execution event: {trade_data}")
             return # Don't log portfolio update for invalid side
        
        # Log summary portfolio update
        # Consider logging less frequently or only on significant changes
        self.log_portfolio_summary()
            
    def handle_position_update(self, event: Event) -> None:
        """
        Handle position update events to maintain the local position state.
        
        Args:
            event: The position update event
        """
        position_data = event.data
        symbol = position_data.get("symbol")
        size = position_data.get("size") # Assume size is the amount of the token held
        
        if symbol is None or size is None:
             logger.warning(f"Received incomplete position update event: {position_data}")
             return
             
        if size > 1e-9: # Use a small threshold to account for float precision
            # Update or add position
            self.positions[symbol] = position_data
            logger.debug(f"Position Update ({symbol}): Size={size:.6f}, Details={position_data}")
        else:
            # Remove position if size is effectively zero
            if symbol in self.positions:
                logger.info(f"Position closed for {symbol}. Previous state: {self.positions[symbol]}")
                del self.positions[symbol]
            else:
                 # Log if we receive a zero-size update for a non-existent position
                 logger.debug(f"Received zero-size position update for non-held symbol {symbol}")

        # Optional: Log portfolio summary after position change if desired
        # self.log_portfolio_summary() 
    
    def calculate_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate the total portfolio value using current prices if available.
        
        Args:
            current_prices: Optional dictionary mapping symbol -> current price.
                            If None, uses the last known price or cost basis from position data.
        Returns:
            The total portfolio value (cash + market value of positions).
        """
        position_market_value = 0.0
        for symbol, pos in self.positions.items():
            size = pos.get('size', 0.0)
            # Use provided current price, fallback to last known price, fallback to cost basis
            price = current_prices.get(symbol) if current_prices else None
            if price is None:
                 price = pos.get("last_price", pos.get("price", pos.get("cost_basis", 0.0)))
                 
            position_market_value += size * price
            
        return self.cash + position_market_value
    
    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get a summary dictionary of the current portfolio state and performance.
        
        Args:
            current_prices: Optional dictionary mapping symbol -> current price for valuation.

        Returns:
            A dictionary with portfolio summary information.
        """
        portfolio_value = self.calculate_portfolio_value(current_prices)
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0.0
        return {
            "timestamp": time.time(),
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "portfolio_value": portfolio_value,
            "position_count": len(self.positions),
            "positions": self.positions, # Contains detailed position info
            "total_pnl": self.total_pnl,
            "total_pnl_pct": (self.total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": win_rate
        }

    def log_portfolio_summary(self, level=logging.INFO):
        """Logs a concise summary of the portfolio."""
        summary = self.get_portfolio_summary()
        logger.log(level, f"Portfolio Summary: Value={summary['portfolio_value']:.2f}, Cash={summary['cash']:.2f}, "
                      f"Positions={summary['position_count']}, PnL={summary['total_pnl']:.2f} ({summary['total_pnl_pct']:.2f}%), "
                      f"Trades={summary['total_trades']}, WinRate={summary['win_rate_pct']:.1f}%") 