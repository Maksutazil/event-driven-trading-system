import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

from src.core.events import EventBus, BaseEventPublisher, Event, EventType
from src.core.data import DataFeedManager

logger = logging.getLogger(__name__)

class PostgresDataManager(BaseEventPublisher):
    """
    PostgreSQL database manager for loading historical trade data.
    
    This class connects to a PostgreSQL database and loads historical trade data,
    simulating an event stream similar to the websocket client.
    """
    def __init__(
        self, 
        event_bus: Optional[EventBus] = None,
        data_feed_manager: Optional[DataFeedManager] = None,
        connection_params: Optional[Dict[str, Any]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        batch_size: int = 100,
        streaming_delay_ms: int = 100,
        debug_mode: bool = False
    ):
        """
        Initialize the PostgreSQL data manager.
        
        Args:
            event_bus: Event bus for publishing events
            data_feed_manager: Data feed manager
            connection_params: Dictionary with connection parameters
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            batch_size: Batch size for retrieving data
            streaming_delay_ms: Delay between batches when streaming
            debug_mode: Whether to enable debug logging
        """
        if event_bus:
            super().__init__(event_bus)
        else:
            self.event_bus = None
            
        self.data_feed_manager = data_feed_manager
        self.batch_size = batch_size
        self.streaming_delay_ms = streaming_delay_ms
        self.debug_mode = debug_mode
        
        # Connection parameters
        self.connection_params = connection_params or {}
        if host:
            self.connection_params['host'] = host
        if port:
            self.connection_params['port'] = port
        if database:
            self.connection_params['dbname'] = database  # psycopg2 uses 'dbname', not 'database'
        if user:
            self.connection_params['user'] = user
        if password:
            self.connection_params['password'] = password
            
        # Set defaults if not provided
        if 'host' not in self.connection_params:
            self.connection_params['host'] = 'localhost'
        if 'port' not in self.connection_params:
            self.connection_params['port'] = 5432
        if 'dbname' not in self.connection_params:
            self.connection_params['dbname'] = 'pumpfun_monitor'
        if 'user' not in self.connection_params:
            self.connection_params['user'] = 'postgres'
        if 'password' not in self.connection_params:
            self.connection_params['password'] = 'postgres'
        
        # Connection state
        self.conn = None
        self.connected = False
        self.streaming = False
        self.streaming_task = None
        
        # Cache of token information
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        
        # Database table and column mapping
        self.tables = {
            'tokens': 'tokens',
            'trades': 'token_trades'
        }
        
        logger.info(f"PostgresDataManager initialized for {self.connection_params['dbname']}")
        
        # Register as data source if data_feed_manager is provided
        if self.data_feed_manager:
            self.data_feed_manager.register_data_source("postgres_historical")
            logger.info("Registered PostgreSQL data source with DataFeedManager")
            
        # Set logger level based on debug mode
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
    async def connect(self) -> bool:
        """Connect to the PostgreSQL database."""
        if not self.connection_params:
            logger.warning("Cannot connect to database: Connection parameters not provided.")
            return False
            
        if self.connected:
            logger.info("Database already connected.")
            return True
            
        try:
            # Connection params should include host, port, dbname, user, password
            logger.info(f"Connecting to PostgreSQL database at {self.connection_params.get('host')}:{self.connection_params.get('port')}/{self.connection_params.get('dbname')}...")
            
            # Create a synchronous connection - will be used for queries
            self.conn = psycopg2.connect(
                **self.connection_params,
                cursor_factory=RealDictCursor  # Returns results as dictionaries
            )
            
            self.connected = True
            logger.info("Connected to PostgreSQL database")
            
            # Verify database structure
            if await self._verify_database_structure():
                logger.info("Database structure verified")
                # Preload tokens into memory
                await self._preload_tokens()
                return True
            else:
                logger.error("Database structure verification failed")
                await self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}", exc_info=True)
            self.connected = False
            return False
    
    async def _verify_database_structure(self) -> bool:
        """
        Verify that the database has the expected structure.
        
        Returns:
            True if the database structure is valid, False otherwise
        """
        try:
            # Check if the database has the expected tables
            cursor = self.conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row['table_name'] for row in cursor.fetchall()]
            
            logger.info(f"Found tables in database: {tables}")
            
            # Check if our required tables exist
            if 'tokens' not in tables and 'token' not in tables:
                logger.error(f"Required table 'tokens' not found in database")
                return False
                
            if 'token_trades' not in tables and 'trades' not in tables:
                logger.error(f"Required table 'token_trades' not found in database")
                return False
                
            # Store the actual table names we'll use
            self.tables['tokens'] = 'tokens' if 'tokens' in tables else 'token'
            self.tables['trades'] = 'token_trades' if 'token_trades' in tables else 'trades'
            
            # Check if the tables have the expected columns
            tokens_table = self.tables['tokens']
            trades_table = self.tables['trades']
            
            # Get token table columns
            cursor.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{tokens_table}'
            """)
            token_columns = [row['column_name'].lower() for row in cursor.fetchall()]
            logger.debug(f"Token table columns: {token_columns}")
            
            # Get trade table columns
            cursor.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{trades_table}'
            """)
            trade_columns = [row['column_name'].lower() for row in cursor.fetchall()]
            logger.debug(f"Trade table columns: {trade_columns}")
            
            # Check for essential columns in token table
            required_token_columns = ['token_id']
            for col in required_token_columns:
                if col.lower() not in token_columns:
                    logger.error(f"Required column '{col}' not found in table '{tokens_table}'")
                    return False
                    
            # Check for essential columns in trade table
            required_trade_columns = ['token_id', 'timestamp', 'type']
            for col in required_trade_columns:
                if col.lower() not in trade_columns:
                    logger.error(f"Required column '{col}' not found in table '{trades_table}'")
                    return False
            
            # Store the column mappings for later use
            self.token_columns = token_columns
            self.trade_columns = trade_columns
            
            logger.info(f"Database structure verified: tokens={tokens_table}, trades={trades_table}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying database structure: {e}")
            # Reset any aborted transaction
            self.conn.rollback()
            return False
            
    async def _preload_tokens(self) -> None:
        """Preload tokens from the database."""
        try:
            logger.info("Preloading tokens from database...")
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            tokens_table = self.tables['tokens']
            
            query = f"""
                SELECT 
                    token_id, mint, symbol, name, created_at, 
                    initial_market_cap, last_market_cap, last_price, last_volume
                FROM {tokens_table}
            """
            
            cursor.execute(query)
            tokens = cursor.fetchall()
            
            logger.info(f"Preloaded {len(tokens)} tokens from database")
            
            # Cache tokens
            for token in tokens:
                token_id = token['token_id']
                self.token_cache[token_id] = dict(token)
                
                # Publish token created event if we have an event bus
                self._publish_token_created_event(token)
                
        except Exception as e:
            logger.error(f"Error preloading tokens: {e}")
            self.conn.rollback()
    
    async def get_token_by_id(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get token details by ID."""
        if token_id in self.token_cache:
            return self.token_cache[token_id]
            
        try:
            with self.conn.cursor() as cursor:
                # Find the appropriate ID column
                id_column = next((col for col in self.token_columns if col in ['id', 'token_id']), 'id')
                
                cursor.execute(f'SELECT * FROM "{self.token_table}" WHERE "{id_column}" = %s', (token_id,))
                token = cursor.fetchone()
                
                if token:
                    # Extract token data
                    token_data = {'id': token_id}
                    for field in ['address', 'contract_address', 'name', 'symbol', 'metadata']:
                        if field in token:
                            token_data[field] = token[field]
                            
                    self.token_cache[token_id] = token_data
                    return token_data
                    
                return None
                
        except Exception as e:
            logger.error(f"Error getting token by ID {token_id}: {e}", exc_info=True)
            return None
    
    async def get_trades_for_token(
        self, 
        token_id: str, 
        limit: int = 100, 
        offset: int = 0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get trades for a token.
        
        Args:
            token_id: Token ID
            limit: Maximum number of trades to return
            offset: Offset for pagination
            start_time: Start time for filtering trades
            end_time: End time for filtering trades
            
        Returns:
            List of trade records
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database")
                return []
                
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            trades_table = self.tables['trades']
            
            # Build query with parameters
            query_params = [token_id]
            query = f"""
                SELECT 
                    trade_id, token_id, timestamp, type, 
                    token_amount, sol_amount, price_sol as price, 
                    market_cap, trader_address
                FROM {trades_table}
                WHERE token_id = %s
            """
            
            # Add time filtering if specified
            if start_time:
                query += " AND timestamp >= %s"
                query_params.append(start_time)
            if end_time:
                query += " AND timestamp <= %s"
                query_params.append(end_time)
                
            # Add ordering and limit
            query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            query_params.extend([limit, offset])
            
            cursor.execute(query, query_params)
            trades = cursor.fetchall()
            
            # Convert to list of dicts
            result = []
            for trade in trades:
                trade_dict = dict(trade)
                # Rename fields to match expected format if needed
                trade_dict['is_buy'] = trade_dict['type'] == 'buy'
                result.append(trade_dict)
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting trades for token {token_id}: {e}")
            # Reset any aborted transaction
            self.conn.rollback()
            return []
    
    async def get_all_tokens(self, limit: int = 1000, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all tokens from the database.
        
        Args:
            limit: Maximum number of tokens to return
            offset: Offset for pagination
            
        Returns:
            List of token records
        """
        try:
            if not self.connected:
                logger.warning("Not connected to database")
                return []
                
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            tokens_table = self.tables['tokens']
            
            query = f"""
                SELECT 
                    token_id, mint, symbol, name, created_at, 
                    initial_market_cap, last_market_cap, last_price, last_volume,
                    creator_address, holder_count, monitoring_status
                FROM {tokens_table}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """
            
            cursor.execute(query, (limit, offset))
            tokens = cursor.fetchall()
            
            # Convert to list of dicts
            result = []
            for token in tokens:
                result.append(dict(token))
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting all tokens: {e}")
            # Reset any aborted transaction
            self.conn.rollback()
            return []
    
    async def start_streaming(
        self,
        token_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Start streaming historical data as events.
        
        Args:
            token_ids: Optional list of token IDs to stream (all tokens if None)
            start_time: Optional start time for filtering trades
            end_time: Optional end time for filtering trades
            
        Returns:
            Success status
        """
        if not self.connected:
            logger.warning("Database not connected. Connecting first...")
            success = await self.connect()
            if not success:
                return False
        
        if self.streaming:
            logger.warning("Streaming already active, stopping first...")
            await self.stop_streaming()
            
        self.streaming = True
        
        # Start the streaming task
        self.streaming_task = asyncio.create_task(
            self._stream_historical_data(token_ids, start_time, end_time)
        )
        
        logger.info(f"Started historical data streaming task")
        return True
    
    async def stop_streaming(self) -> None:
        """Stop streaming historical data."""
        if not self.streaming:
            logger.info("Streaming not active.")
            return
            
        self.streaming = False
        
        if self.streaming_task and not self.streaming_task.done():
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Stopped historical data streaming")
    
    async def _stream_historical_data(
        self,
        token_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Stream historical data as events.
        
        Args:
            token_ids: Optional list of token IDs to stream (all tokens if None)
            start_time: Optional start time for filtering trades
            end_time: Optional end time for filtering trades
        """
        try:
            # Get all tokens if not specified
            if token_ids is None:
                tokens = await self.get_all_tokens()
                token_ids = [token['id'] for token in tokens]
                
            logger.info(f"Streaming historical data for {len(token_ids)} tokens...")
            
            # First emit events for tokens
            for token_id in token_ids:
                token = await self.get_token_by_id(token_id)
                if token:
                    # Publish token created event
                    self._publish_token_created_event(token)
                    await asyncio.sleep(self.streaming_delay_ms / 1000)
                    
            # Now emit trades for all tokens
            for token_id in token_ids:
                offset = 0
                while self.streaming:
                    trades = await self.get_trades_for_token(
                        token_id, 
                        limit=self.batch_size, 
                        offset=offset,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if not trades:
                        break  # No more trades for this token
                        
                    for trade in trades:
                        if not self.streaming:
                            break
                            
                        # Process and emit trade event
                        self._publish_trade_event(trade)
                        
                        # Small delay between events
                        await asyncio.sleep(self.streaming_delay_ms / 1000)
                        
                    # Move to next batch
                    offset += self.batch_size
                    
                    # Small delay between batches
                    await asyncio.sleep(0.5)
                    
            logger.info("Finished streaming all historical data")
            
        except asyncio.CancelledError:
            logger.info("Historical data streaming task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in historical data streaming: {e}", exc_info=True)
            
        finally:
            self.streaming = False
    
    def _publish_token_created_event(self, token: Dict[str, Any]) -> None:
        """
        Publish a token created event.
        
        Args:
            token: Token data
        """
        token_id = token['id']
        token_name = token['name']
        token_symbol = token['symbol']
        
        # Create event data for the event bus
        event_data = {
            'token_id': token_id,
            'token_name': token_name,
            'token_symbol': token_symbol,
            'address': token['address'],
            'initial_price': 0.0,  # Not available in token data
            'market_cap': 0.0,     # Not available in token data
            'creator': 'Unknown',  # Not available in token data
            'timestamp': time.time(),  # Use current time
            'metadata': token.get('metadata', {})
        }
        
        # Update data feed manager if available
        if self.data_feed_manager:
            dfm_update = {
                'name': token_name,
                'symbol': token_symbol,
                'address': token['address'],
                'last_updated': time.time()
            }
            self.data_feed_manager.update_token_data(token_id, dfm_update)
            
        # Publish the TOKEN_CREATED event
        self.event_bus.publish(Event(
            event_type=EventType.TOKEN_CREATED,
            data=event_data,
            source="postgres_historical"
        ))
        logger.info(f"Published TOKEN_CREATED event for {token_name} ({token_symbol})")
    
    def _publish_trade_event(self, trade: Dict[str, Any]) -> None:
        """
        Publish a trade event.
        
        Args:
            trade: Trade data
        """
        try:
            token_id = trade['tokenId']
            price = float(trade['price'])
            amount = float(trade['amount'])
            trade_type = trade['type'].lower()
            
            # Convert timestamp if needed
            if isinstance(trade['timestamp'], datetime):
                timestamp = trade['timestamp'].timestamp()
            else:
                timestamp = time.time()  # Fallback to current time
                
            is_buy = trade_type == 'buy'
            
            # Get token info if available
            token_info = self.token_cache.get(token_id, {})
            token_name = token_info.get('name', 'Unknown')
            token_symbol = token_info.get('symbol', 'UNKNOWN')
                
            if self.debug_mode:
                logger.debug(f"Processing Trade: {'BUY' if is_buy else 'SELL'} {amount:.4f} {token_id} @ {price:.6f}")
                
            # Create event data for the event bus
            event_data = {
                'token_id': token_id,
                'is_buy': is_buy,
                'is_sell': not is_buy,
                'price': price,
                'amount': amount,
                'trader': trade.get('walletId', 'Unknown'),
                'timestamp': timestamp,
                'signature': trade.get('id', ''),  # Use trade ID as signature
                'raw_data': dict(trade)
            }
            
            # Update data feed manager if available
            if self.data_feed_manager:
                dfm_update = {
                    'price': price,
                    'last_trade_amount': amount,
                    'last_trade_side': 'buy' if is_buy else 'sell',
                    'last_trade_timestamp': timestamp,
                    'last_updated': time.time(),  # Use current time for last update
                    'is_trade': True,
                    'name': token_name,
                    'symbol': token_symbol
                }
                self.data_feed_manager.update_token_data(token_id, dfm_update)
                
            # Publish the TOKEN_TRADE event
            self.event_bus.publish(Event(
                event_type=EventType.TOKEN_TRADE,
                data=event_data,
                source="postgres_historical"
            ))
            logger.info(f"Published TOKEN_TRADE event for {token_id} ({'BUY' if is_buy else 'SELL'} @ {price:.6f})")
            
        except Exception as e:
            logger.error(f"Error publishing trade event: {e}", exc_info=True)
            if self.debug_mode:
                logger.error(f"Problematic trade data: {trade}")
    
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        # Stop streaming if active
        if self.streaming:
            await self.stop_streaming()
            
        if not self.connected:
            logger.info("Database already disconnected.")
            return
            
        try:
            if self.conn:
                self.conn.close()
                logger.info("Disconnected from PostgreSQL database")
                
            self.connected = False
            
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}", exc_info=True)
            
        finally:
            self.connected = False 