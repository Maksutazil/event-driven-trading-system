import logging
import time
from src.core.events import EventType, EventHandlerWrapper
from src.core.data.interfaces import DataFeedInterface  # Import directly from interfaces file

logger = logging.getLogger(__name__)

class DataFeedManager(DataFeedInterface):  # Implement the interface
    """
    Manager for data feeds and token data.
    
    This class manages data feeds and provides access to token data.
    It subscribes to token events and keeps track of token data.
    """
    
    def __init__(self, event_bus):
        """
        Initialize the data feed manager.
        
        Args:
            event_bus: Event bus for communication
        """
        self.event_bus = event_bus
        self.token_data = {}  # {token_id: {data}}
        self.data_sources = set()  # Set of registered data sources
        self.subscribers = {}  # {token_id: set(data_source_ids)}
        
        # Register for token update events
        token_update_handler = EventHandlerWrapper(self._handle_token_update)
        self.event_bus.subscribe(EventType.TOKEN_UPDATED, token_update_handler)
        
        # Register for token trade events
        token_trade_handler = EventHandlerWrapper(self._handle_token_trade)
        self.event_bus.subscribe(EventType.TOKEN_TRADE, token_trade_handler)
        
        # Register for token creation events
        token_created_handler = EventHandlerWrapper(self._handle_token_created)
        self.event_bus.subscribe(EventType.TOKEN_CREATED, token_created_handler)
        
        logger.info("DataFeedManager initialized")
    
    def subscribe_token(self, token_id):
        """
        Subscribe to data for a specific token.
        
        Args:
            token_id: ID of the token to subscribe to
            
        Returns:
            Dictionary with subscription results for each data source
        """
        results = {}
        
        if token_id not in self.subscribers:
            self.subscribers[token_id] = set()
        
        # Try to subscribe using available data sources
        for source_id in self.data_sources:
            # In a real implementation, we'd call the actual data source
            # For the example, we'll just pretend it worked
            success = True
            self.subscribers[token_id].add(source_id)
            results[source_id] = success
        
        # Initialize empty data for token if not exists
        if token_id not in self.token_data:
            self.token_data[token_id] = {}
        
        logger.info(f"Subscribed to token {token_id}")
        return results
    
    def unsubscribe_token(self, token_id):
        """
        Unsubscribe from data for a specific token.
        
        Args:
            token_id: ID of the token to unsubscribe from
        """
        if token_id in self.subscribers:
            del self.subscribers[token_id]
        
        logger.info(f"Unsubscribed from token {token_id}")
    
    def get_token_data(self, token_id):
        """
        Get the latest data for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary with token data, or empty dict if not available
        """
        return self.token_data.get(token_id, {})
    
    def update_token_data(self, token_id, data):
        """
        Update data for a token.
        
        Args:
            token_id: ID of the token
            data: New data to merge with existing data
        """
        if token_id not in self.token_data:
            self.token_data[token_id] = {}
        
        self.token_data[token_id].update(data)
    
    def register_data_source(self, source_id):
        """
        Register a data source.
        
        Args:
            source_id: ID of the data source
        """
        self.data_sources.add(source_id)
        logger.info(f"Registered data source: {source_id}")
    
    def _handle_token_created(self, event):
        """Handle token creation events."""
        try:
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Token creation event missing token_id")
                return
                
            # Extract token data
            token_name = data.get('token_name', 'Unknown')
            token_symbol = data.get('token_symbol', 'UNKNOWN')
            initial_price = data.get('initial_price', 0.0)
            market_cap = data.get('market_cap', 0.0)
            
            # Update token data in the manager
            token_data = {
                'token_id': token_id,
                'name': token_name,
                'symbol': token_symbol,
                'price': float(initial_price),
                'market_cap': float(market_cap),
                'creation_time': data.get('timestamp', time.time()),
                'last_updated': time.time(),
                'is_new_token': True
            }
            
            self.update_token_data(token_id, token_data)
            logger.info(f"Updated data for new token: {token_symbol} ({token_id})")
            
        except Exception as e:
            logger.error(f"Error handling token creation event: {e}", exc_info=True)
    
    def _handle_token_update(self, event):
        """Handle token update events."""
        try:
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Token update event missing token_id")
                return
            
            # Update token data
            update_data = {
                'last_updated': time.time()
            }
            
            # Copy relevant fields
            for key in ['price', 'volume', 'market_cap', 'features']:
                if key in data:
                    update_data[key] = data[key]
            
            self.update_token_data(token_id, update_data)
            logger.debug(f"Updated token data for {token_id}")
            
        except Exception as e:
            logger.error(f"Error processing token update: {e}", exc_info=True)
    
    def _handle_token_trade(self, event):
        """Handle token trade events."""
        try:
            data = event.data
            token_id = data.get('token_id')
            
            if not token_id:
                logger.warning("Token trade event missing token_id")
                return
            
            # Extract trade data
            price = data.get('price')
            if price is None:
                logger.warning(f"Token trade event missing price: {data}")
                return
                
            # Update token data with trade information
            trade_data = {
                'price': float(price),
                'last_trade_amount': data.get('amount', 0),
                'last_trade_side': 'buy' if data.get('is_buy', False) else 'sell',
                'last_trade_timestamp': data.get('timestamp', time.time()),
                'last_updated': time.time(),
                'is_trade': True
            }
            
            self.update_token_data(token_id, trade_data)
            logger.debug(f"Updated token price for {token_id}: {price}")
            
        except Exception as e:
            logger.error(f"Error processing token trade: {e}", exc_info=True)
    
    def get_historical_data(self, token_id, start_time=None, end_time=None):
        """
        Get historical data for a token.
        
        Args:
            token_id: The token ID
            start_time: Start time for the data (optional)
            end_time: End time for the data (optional)
            
        Returns:
            List of trade records or empty list if no data
        """
        logger.info(f"Getting historical data for {token_id}")
        
        # In a real implementation, this would query historical data
        # For now, return an empty list as a placeholder
        # Actual implementations should override this
        return [] 