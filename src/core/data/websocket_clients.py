import asyncio
import logging
import os
import sys
import time
import json
import random
import websockets
import ssl
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List, Set, Tuple

# Assuming numpy is installed if generate_mock_price_history uses it heavily,
# but using random.normalvariate as a fallback if numpy isn't strictly needed.
# import numpy as np 

from src.core.events import Event, EventType, EventBus, BaseEventPublisher
# Assuming DataFeedManager is in src/core/data/data_feed_manager.py:
from src.core.data.data_feed_manager import DataFeedManager
# Assuming FeatureSystem might be needed by MockWebSocketClient
# from src.core.features import FeatureSystem # Add if needed

# Placeholder for global shutdown signal used by RealWebSocketClient
# TODO: Refactor RealWebSocketClient to accept a shutdown event/signal
shutdown_requested = False 

logger = logging.getLogger(__name__) # Module-level logger

class RealWebSocketClient:
    """Real WebSocket client for connecting to a trading data source."""
    
    def __init__(self, 
                event_bus: EventBus, 
                websocket_uri: str, 
                data_feed_manager: Optional[DataFeedManager] = None,
                api_key: Optional[str] = None, 
                subscription_keys: Optional[Dict[str, list]] = None, 
                enable_dynamic_token_discovery: bool = False,
                debug_mode: bool = False, 
                heartbeat_interval: int = 30):
        """
        Initialize the WebSocket client.
        Args:
            event_bus: Event bus for publishing events
            websocket_uri: URI of the WebSocket server
            data_feed_manager: Manager for token data
            api_key: Optional API key for authentication
            subscription_keys: Optional keys for subscription endpoints
            enable_dynamic_token_discovery: Whether to automatically discover and subscribe to new tokens
            debug_mode: Whether to enable debug mode
            heartbeat_interval: Interval in seconds for sending heartbeat messages
        """
        self.event_bus = event_bus
        self.websocket_uri = websocket_uri
        self.data_feed_manager = data_feed_manager
        self.api_key = api_key
        self.subscription_keys = subscription_keys or {}
        self.enable_dynamic_token_discovery = enable_dynamic_token_discovery
        self.debug_mode = debug_mode
        self.heartbeat_interval = heartbeat_interval
        
        # WebSocket state
        self.websocket: Optional[websockets.client.WebSocketClientProtocol] = None
        self.connected = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self.token_subscription_queue: List[str] = []
        self.subscribed_tokens: Set[str] = set()
        
        # Register as data source if data_feed_manager is provided
        if self.data_feed_manager:
            self.data_feed_manager.register_data_source("pumpportal_websocket")
            # Use module logger here, as self.logger is set later
            logger.info("Registered PumpPortal WebSocket as data source with DataFeedManager")
        
        # Set up WebSocket connection parameters
        self.connection_task: Optional[asyncio.Task] = None
        self.token_queue_processor_task: Optional[asyncio.Task] = None
        self.message_count = 0
        
        # Track raw messages for debugging
        self.raw_message_file = None
        if self.debug_mode:
            try:
                # Ensure the log file path is appropriate
                log_dir = Path(".") # Or specify a dedicated logs directory
                log_dir.mkdir(exist_ok=True)
                self.raw_message_file = open(log_dir / 'raw_websocket_data.log', 'a')
            except Exception as e:
                logger.error(f"Failed to open raw_websocket_data.log for appending: {e}")
        
        # Logger specific to this client instance
        self.logger = logging.getLogger("pumpportal_websocket")
        self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        if self.connected:
            self.logger.info("Already connected.")
            return True
        try:
            self.logger.info(f"Attempting to connect to PumpPortal WebSocket at {self.websocket_uri}")
            
            # Disable SSL certificate verification for testing (consider security implications)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            self.logger.debug("SSL context created with certificate verification disabled")
            
            self.logger.debug(f"Connection parameters: Debug={self.debug_mode}, DynDiscovery={self.enable_dynamic_token_discovery}")
            
            try:
                # Explicit connect timeout (using asyncio.wait_for)
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.websocket_uri, 
                        ssl=ssl_context,
                        ping_interval=None,  # Disable built-in ping
                        close_timeout=10,    # Timeout for close handshake
                        max_size=10 * 1024 * 1024, # 10MB max message size
                        open_timeout=10 # Timeout for connection opening
                    ),
                    timeout=15.0 # Overall connection attempt timeout
                )
                self.logger.debug("WebSocket connection established successfully")
            except asyncio.TimeoutError:
                 self.logger.error(f"Connection attempt timed out after 15 seconds.")
                 return False
            except websockets.exceptions.InvalidStatusCode as status_error:
                self.logger.error(f"Invalid status code from server: {status_error}")
                return False
            except websockets.exceptions.InvalidURI as uri_error:
                self.logger.error(f"Invalid WebSocket URI format: {uri_error}")
                return False
            except websockets.exceptions.InvalidHandshake as handshake_error:
                self.logger.error(f"WebSocket handshake failed: {handshake_error}")
                return False
            except ConnectionRefusedError:
                 self.logger.error(f"Connection refused by server at {self.websocket_uri}")
                 return False
            except OSError as os_err:
                self.logger.error(f"OS error during connection: {os_err}")
                return False
            except Exception as conn_error: # Catch other potential errors
                self.logger.error(f"Unexpected connection error: {conn_error}", exc_info=True)
                return False
            
            self.connected = True
            self.logger.info("Successfully connected to PumpPortal WebSocket")
            self._reconnect_attempts = 0 # Reset on successful connection
            
            # Start token subscription queue processor if not running
            if self.token_queue_processor_task is None or self.token_queue_processor_task.done():
                self.logger.debug("Starting token subscription queue processor...")
                self.token_queue_processor_task = asyncio.create_task(self._process_token_subscription_queue())
            
            return True
            
        except Exception as e:
            self.connected = False # Ensure state is false on any error
            self.logger.error(f"Unhandled exception during WebSocket connect method: {e}", exc_info=True)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the WebSocket server and clean up tasks."""
        self.connected = False # Signal disconnection immediately
        self.logger.info("Disconnecting WebSocket client...")
        
        # Cancel background tasks
        tasks_to_cancel = [self.connection_task, self.token_queue_processor_task, self.heartbeat_task]
        for i, task in enumerate(tasks_to_cancel):
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0) # Wait briefly for cancellation
                except asyncio.CancelledError:
                    self.logger.debug(f"Task {i+1} cancelled successfully.")
                except asyncio.TimeoutError:
                    self.logger.warning(f"Task {i+1} did not cancel within timeout.")
                except Exception as e:
                     self.logger.error(f"Error waiting for task {i+1} cancellation: {e}")
        self.connection_task = self.token_queue_processor_task = self.heartbeat_task = None
            
        # Close WebSocket connection
        ws = self.websocket
        if ws and ws.open:
            try:
                await asyncio.wait_for(ws.close(), timeout=5.0)
                self.logger.info("WebSocket connection closed gracefully.")
            except asyncio.TimeoutError:
                 self.logger.warning("WebSocket close handshake timed out.")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
        self.websocket = None
            
        # Close debug file
        if self.raw_message_file:
            try:
                self.raw_message_file.close()
            except Exception as e:
                 self.logger.error(f"Error closing raw message log file: {e}")
            self.raw_message_file = None
        
        self.logger.info("WebSocket client disconnected.")
        return True

    async def _setup_subscriptions(self) -> bool:
        """Set up initial subscriptions after connection."""
        if not self.connected or not self.websocket or not self.websocket.open:
             self.logger.error("Cannot set up subscriptions: WebSocket not connected.")
             return False
        try:
            self.logger.info("Setting up initial subscriptions...")
            success = True
            # Subscribe to new token events
            payload_new_token = {"method": "subscribeNewToken"}
            if not await self._send_websocket_message(payload_new_token):
                 self.logger.error("Failed to send subscribeNewToken request.")
                 success = False # Mark as failure but continue trying others
            else:
                 self.logger.info("Sent subscribeNewToken request.")
            
            # Subscribe to migration events
            payload_migration = {"method": "subscribeMigration"}
            if not await self._send_websocket_message(payload_migration):
                 self.logger.warning("Failed to send subscribeMigration request.") # Warning may be sufficient
            else:
                self.logger.info("Sent subscribeMigration request.")
            
            # Subscribe to initial tokens if provided
            initial_tokens = self.subscription_keys.get('tokens', [])
            valid_initial_tokens = [t for t in initial_tokens if isinstance(t, str) and t]
            if valid_initial_tokens:
                payload_tokens = {"method": "subscribeTokenTrade", "keys": valid_initial_tokens}
                if await self._send_websocket_message(payload_tokens):
                    self.logger.info(f"Sent subscription request for initial token trades: {valid_initial_tokens}")
                    self.subscribed_tokens.update(valid_initial_tokens)
                else:
                    self.logger.error(f"Failed to send initial token trade subscription: {valid_initial_tokens}")
                    success = False
            
            # Subscribe to account trades if provided
            initial_accounts = self.subscription_keys.get('accounts', [])
            valid_accounts = [a for a in initial_accounts if isinstance(a, str) and a]
            if valid_accounts:
                payload_accounts = {"method": "subscribeAccountTrade", "keys": valid_accounts}
                if await self._send_websocket_message(payload_accounts):
                    self.logger.info(f"Sent subscription request for account trades: {valid_accounts}")
                else:
                    self.logger.warning(f"Failed to send initial account trade subscription: {valid_accounts}")
            
            if success:
                 self.logger.info("Initial subscriptions setup completed (some may have failed)." if not success else "Initial subscriptions setup completed successfully.")
            return success
            
        except Exception as e:
            self.logger.error(f"Error during initial subscription setup: {e}", exc_info=True)
            return False
    
    async def _subscribe_to_token(self, token_mint: str):
        """Subscribe to trades for a specific token."""
        # Basic validation
        if not token_mint or not isinstance(token_mint, str) or len(token_mint) < 10: # Adjust min length if needed
            self.logger.warning(f"Attempted to subscribe to an invalid or short token mint: '{token_mint}'")
            return
        if token_mint in self.subscribed_tokens:
             self.logger.debug(f"Token {token_mint} is already in the subscribed set.")
             return
            
        try:
            payload = {"method": "subscribeTokenTrade", "keys": [token_mint]}
            success = await self._send_websocket_message(payload)

            if success:
                self.subscribed_tokens.add(token_mint)
                self.logger.info(f"Successfully sent token trade subscription request for: {token_mint}")
            else:
                self.logger.error(f"Failed to send subscription request for token {token_mint}. Will not mark as subscribed.")

        except Exception as e:
            self.logger.error(f"Exception during token subscription for {token_mint}: {e}", exc_info=True)
    
    async def _queue_token_subscription(self, token_mint: str):
        """Add a token to the subscription queue if valid and not already handled."""
        if (token_mint and 
            isinstance(token_mint, str) and 
            len(token_mint) > 10 and 
            token_mint not in self.subscribed_tokens and 
            token_mint not in self.token_subscription_queue):
            self.token_subscription_queue.append(token_mint)
            self.logger.debug(f"Queued token for subscription: {token_mint}")
        else:
            if self.debug_mode:
                 self.logger.debug(f"Did not queue token {token_mint} (invalid, already subscribed, or already queued).")
    
    async def _process_token_subscription_queue(self):
        """Process the token subscription queue periodically."""
        self.logger.info("Token subscription queue processor task started.")
        try:
            while True:
                if not self.connected:
                    await asyncio.sleep(2) # Wait longer if not connected
                    continue
                
                # Process a few tokens from the queue if available
                processed_count = 0
                # Check queue has items before trying to pop
                while self.token_subscription_queue and self.connected and processed_count < 5:
                    try:
                         token = self.token_subscription_queue.pop(0)
                         await self._subscribe_to_token(token)
                         processed_count += 1
                         await asyncio.sleep(0.2) # Small delay between subscription sends
                    except IndexError:
                         break # Queue became empty during processing
                    except Exception as e:
                         self.logger.error(f"Error processing item from subscription queue: {e}", exc_info=True)
                         # Optionally put the token back in the queue or handle differently
                         break # Stop processing batch on error
                
                # Wait before checking queue again
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Token subscription queue processor task stopped.")
        except Exception as e:
            self.logger.error(f"Fatal error in token subscription queue processor: {e}", exc_info=True)
            # Consider signaling failure or attempting restart depending on the error
        finally:
             self.logger.info("Token subscription queue processor task finished.")
    
    async def _send_websocket_message(self, message: Dict[str, Any]) -> bool:
        """Send a JSON message to the WebSocket server."""
        if not self.connected or not self.websocket or not self.websocket.open:
            self.logger.warning(f"Cannot send message: WebSocket not connected/open. Message: {message}")
            return False
            
        try:
            message_str = json.dumps(message)
            await asyncio.wait_for(self.websocket.send(message_str), timeout=10.0) # Add timeout to send
            
            if self.debug_mode:
                self.logger.debug(f"Sent message: {message_str}")
            return True
            
        except asyncio.TimeoutError:
             self.logger.error(f"Timeout sending WebSocket message: {message}")
             # Consider connection potentially broken after send timeout
             # await self.disconnect() # Or trigger reconnect logic
             return False
        except websockets.exceptions.ConnectionClosed as closed_err:
            self.logger.warning(f"Cannot send message: Connection closed. Error: {closed_err}. Message: {message}")
            await self.disconnect() # Disconnect if sending on closed connection
            return False
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {e}. Message: {message}", exc_info=True)
            return False
    
    async def _setup_heartbeat(self):
        """Run periodic heartbeat (ping/pong) to keep connection alive."""
        self.logger.info(f"Starting heartbeat task (interval: {self.heartbeat_interval}s)")
        try:
            while self.connected:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.connected or not self.websocket or not self.websocket.open:
                    self.logger.debug("Heartbeat stopping: WebSocket not connected/open.")
                    break
                    
                ping_success = False
                try:
                    ping_waiter = await self.websocket.ping()
                    # Wait for the pong response with a timeout
                    await asyncio.wait_for(ping_waiter, timeout=10.0)
                    if self.debug_mode:
                        self.logger.debug(f"Ping successful at {time.time():.1f}")
                    ping_success = True
                except asyncio.TimeoutError:
                    self.logger.warning("Pong response not received in time. Connection may be stale.")
                    # Aggressively disconnect if pong times out
                    await self.disconnect()
                    break
                except websockets.exceptions.ConnectionClosed as closed_err:
                    self.logger.warning(f"Heartbeat failed: Connection closed. Error: {closed_err}")
                    await self.disconnect()
                    break
                except Exception as ping_error:
                    self.logger.error(f"Error sending WebSocket ping frame: {ping_error}")
                    # Consider a fallback ping mechanism if needed, but often indicates a problem
                    # If ping fails consistently, consider disconnecting
                    # await self.disconnect() # Optional: disconnect on ping error
                    # break 

                # Optional: Add fallback ping using send_message if native ping fails                
                            
        except asyncio.CancelledError:
            self.logger.debug("Heartbeat task cancelled.")
        except Exception as e:
            # Avoid heartbeat task failure bringing down the whole client if possible
            self.logger.error(f"Error in heartbeat task: {e}", exc_info=True)
        finally:    
            self.logger.info("Heartbeat task stopped.")
    
    async def start_streaming(self) -> bool:
        """Connect, subscribe, and start listening for data.
        Returns True if streaming started successfully, False otherwise.
        """
        if self.connection_task and not self.connection_task.done():
            self.logger.warning("Streaming task already seems to be running.")
            return True
            
        self.logger.info("Attempting to start data streaming...")
        # Ensure disconnected state before starting
        await self.disconnect() 
        self._reconnect_attempts = 0 # Reset reconnect attempts for a fresh start

        # 1. Connect
        connection_success = await self.connect()
        if not connection_success:
            self.logger.error("Failed to connect to data source. Cannot start streaming.")
            return False
            
        # 2. Setup Heartbeat
        if self.heartbeat_task is None or self.heartbeat_task.done():
            self.heartbeat_task = asyncio.create_task(self._setup_heartbeat())
        
        # 3. Setup Subscriptions
        subscribed_ok = await self._setup_subscriptions()
        if not subscribed_ok:
            self.logger.error("Failed to set up initial subscriptions. Stopping.")
            await self.disconnect()
            return False
            
        # 4. Start the main data receiving task
        if self.connection_task is None or self.connection_task.done():
            self.logger.info("Starting main data streaming task...")
            self.connection_task = asyncio.create_task(self._stream_data())
        else:
             # This case should ideally not happen due to disconnect logic above
             self.logger.warning("Streaming task reference already exists and is running?")
        
        self.logger.info("Data streaming started successfully.")
        return True
    
    async def stop_streaming(self):
        """Stop streaming and disconnect gracefully."""
        self.logger.info("Stopping data streaming requested.")
        # The disconnect method now handles task cancellation and connection closing
        await self.disconnect()
        self.logger.info("Data streaming stopped.")
    
    async def _stream_data(self):
        """Main loop to receive and process WebSocket messages."""
        if not self.connected or not self.websocket or not self.websocket.open:
            self.logger.error("Cannot start streaming data: WebSocket not connected/open.")
            return
            
        self.logger.info(f"Listening for messages on {self.websocket_uri}...")
        last_message_time = time.time()
        connection_healthy = True
            
        try:
            while self.connected:
                try:
                    # Wait for a message with a timeout
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=15.0)
                    last_message_time = time.time() # Reset timer on message receipt
                    self.message_count += 1
                    
                    if self.debug_mode:
                         self.logger.debug(f"Received WS msg #{self.message_count}")
                    
                    # Log raw message if enabled
                    if self.raw_message_file:
                        try:
                            log_line = f"{datetime.now().isoformat()} - {message}\n"
                            self.raw_message_file.write(log_line)
                            self.raw_message_file.flush()
                        except Exception as log_err:
                            self.logger.error(f"Failed to write to raw log file: {log_err}")
                    
                    # Process the received message asynchronously
                    await self._process_data(message)
                        
                except asyncio.TimeoutError:
                    # No message received in timeout period. Check connection status.
                    self.logger.debug(f"No message received in 15s. Checking connection...")
                    if not self.connected or not self.websocket or not self.websocket.open:
                        self.logger.warning("WebSocket disconnected during receive timeout.")
                        connection_healthy = False
                        break
                    else:
                        # If still connected, maybe log less frequently or rely on heartbeat
                        self.logger.debug("Still connected, continuing receive loop.")
                        # Optionally trigger an explicit ping here if heartbeat seems insufficient
                        # await self.websocket.ping() 
                        pass 
                except websockets.exceptions.ConnectionClosedOK:
                    self.logger.info("WebSocket connection closed gracefully by server.")
                    connection_healthy = False
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    self.logger.warning(f"WebSocket connection closed unexpectedly: {e}")
                    connection_healthy = False
                    break
                except asyncio.CancelledError:
                    self.logger.info("Streaming data task cancelled.")
                    connection_healthy = False
                    break    
                except Exception as e:
                    self.logger.error(f"Error during WebSocket receive/process loop: {e}", exc_info=True)
                    connection_healthy = False
                    # Add a small delay to prevent rapid looping on persistent errors
                    await asyncio.sleep(1)
                    break # Exit loop on unexpected error
            
            # After loop finishes (due to break or self.connected becoming False)
            self.logger.info(f"Exited receive loop. Connection healthy: {connection_healthy}. Shutdown requested: {shutdown_requested}")

            # Attempt reconnect if connection was lost unexpectedly and not during shutdown
            if not connection_healthy and not shutdown_requested:
                self.logger.warning("Connection lost unexpectedly. Attempting to reconnect...")
                # Ensure full disconnect before reconnecting
                await self.disconnect()
                reconnect_success = await self.reconnect() # Reconnect handles backoff
                
                if reconnect_success:
                    self.logger.info("Reconnection successful, restarting streaming...")
                    # start_streaming handles connect, subscribe, heartbeat, stream tasks
                    await self.start_streaming() 
                else:
                    self.logger.error("Failed to reconnect after connection loss. Giving up.")
                    # TODO: Signal application failure or trigger alternative logic
                    
        except asyncio.CancelledError:
            self.logger.info("Streaming task was cancelled externally.")
        except Exception as e:
            self.logger.error(f"Unhandled exception in _stream_data task: {e}", exc_info=True)
        finally:
            # Final cleanup check - ensure disconnect is called if needed
            if self.connected:
                 self.logger.warning("_stream_data finished but client still marked connected. Forcing disconnect.")
                 await self.disconnect()
            self.logger.info("Streaming data task (_stream_data) has finished.")
    
    async def _process_data(self, raw_data: Any):
        """Parse and route incoming WebSocket messages."""
        try:
            if self.debug_mode:
                self.logger.debug(f"Raw data received: {raw_data}")
            
            # Parse JSON if necessary
            data = None
            if isinstance(raw_data, str):
                try:
                    data = json.loads(raw_data)
                    if self.debug_mode:
                         self.logger.debug(f"Parsed JSON data: {data}")
                except json.JSONDecodeError:
                    # Handle non-JSON text messages (e.g., simple confirmations)
                    if "subscribed successfully" in raw_data.lower():
                         self.logger.info(f"Received subscription confirmation text: {raw_data}")
                         return 
                    else:
                        self.logger.warning(f"Received non-JSON message: {raw_data[:200]}...")
                        return
            elif isinstance(raw_data, bytes):
                 try:
                      decoded_data = raw_data.decode('utf-8')
                      data = json.loads(decoded_data)
                      if self.debug_mode:
                           self.logger.debug(f"Parsed bytes->JSON data: {data}")
                 except (UnicodeDecodeError, json.JSONDecodeError) as e:
                      self.logger.warning(f"Failed to decode/parse bytes message: {raw_data[:200]}... Error: {e}")
                      return
            elif isinstance(raw_data, dict):
                 data = raw_data # Already a dictionary
            else:
                self.logger.warning(f"Received data of unexpected type: {type(raw_data)}")
                return
            
            if not isinstance(data, dict):
                self.logger.warning(f"Processed data is not a dictionary: {type(data)}")
                return
            
            # --- Message Routing --- 
            # Use .get() for safer access
            method = data.get('method')
            tx_type = data.get('txType')
            mint_address = data.get('mint')

            # 1. Handle Method Responses (Pong, Subscription ACKs)
            if method:
                if method.lower() == 'pong':
                    self.logger.debug("Received pong response (method: pong)")
                    return
                # Check for generic success indicators
                elif data.get('result') == True or data.get('status') == 'success':
                    self.logger.info(f"Received successful response for method: {method}")
                    # TODO: Potentially link to specific sent requests if needed
                    return
                else:
                     result_info = data.get('result', data.get('status', data.get('error', 'N/A')))
                     self.logger.warning(f"Received potentially unsuccessful response for method {method}: {result_info}")
                     return

            # 2. Handle Token Creation Events
            # Be specific with checks: txType='create', mint is present, name is present
            if tx_type == 'create' and mint_address and data.get('name'):
                token_name = data.get('name', 'Unknown')
                token_symbol = data.get('symbol', 'UNKNOWN')
                self.logger.info(f"Routing token creation event: {token_name} ({token_symbol})")
                self._handle_token_creation_event(data)
                return
            
            # 3. Handle Token Trade Events
            # Be specific: txType is 'buy' or 'sell', mint is present, amount fields are present
            if tx_type in ['buy', 'sell'] and mint_address and data.get('solAmount') is not None:
                # Check if we are subscribed *before* logging noisily
                if mint_address in self.subscribed_tokens:
                    if self.debug_mode:
                         self.logger.debug(f"Routing {tx_type} trade event for subscribed token {mint_address}")
                    self._handle_trade_data(data)
                elif self.debug_mode: # Log if debugging, even if not subscribed
                     self.logger.debug(f"Ignoring trade event for UNSUBSCRIBED token {mint_address}")
                return # Handled (processed or intentionally ignored)

            # 4. Handle other potential message types if known (e.g., migrations, errors)
            # Example: if data.get('event_type') == 'migration': ... handle migration ...
            
            # Fallback for messages not matching known patterns
            self.logger.warning(f"Unhandled message structure: {json.dumps(data)[:300]}...")
            
        except Exception as e:
            # Corrected indentation for the except block
            self.logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
            if self.debug_mode:
                # Log problematic raw data safely
                try: 
                    raw_log_data = raw_data if isinstance(raw_data, str) else str(raw_data)
                except: 
                    raw_log_data = "[Unloggable Raw Data]"
                self.logger.error(f"Problematic raw message data: {raw_log_data[:500]}...")

    def _handle_token_creation_event(self, data: Dict[str, Any]):
        """Process data confirmed to be a token creation event."""
        try:
            token_id = data.get('mint')
            if not token_id: # Should be checked before calling, but double-check
                self.logger.error("Assertion failed: _handle_token_creation_event called without 'mint' field.")
                return
            
            token_name = data.get('name', 'Unknown')
            token_symbol = data.get('symbol', 'UNKNOWN')
            creator = data.get('traderPublicKey', 'Unknown')

            # Safely convert numeric fields
            try: initial_price = float(data.get('solAmount', 0.0))
            except (ValueError, TypeError): initial_price = 0.0
            try: market_cap = float(data.get('marketCapSol', 0.0))
            except (ValueError, TypeError): market_cap = 0.0
            
            self.logger.info(f"Processing details for token creation: {token_name} ({token_symbol}), ID: {token_id}")
            self.logger.debug(f"  -> Price: {initial_price}, MCAP: {market_cap}, Creator: {creator}")
            
            # Create event data for the event bus
            event_data = {
                'token_id': token_id,
                'token_name': token_name,
                'token_symbol': token_symbol,
                'initial_price': initial_price,
                'market_cap': market_cap,
                'creator': creator,
                'timestamp': time.time(), # Use reception time
                'raw_data': data
            }
            
            # Update data feed manager if available
            if self.data_feed_manager:
                dfm_data = {
                    'token_id': token_id,
                    'name': token_name,
                    'symbol': token_symbol,
                    'price': initial_price, 
                    'market_cap': market_cap,
                    'last_updated': event_data['timestamp'],
                    'is_new_token': True
                }
                self.data_feed_manager.update_token_data(token_id, dfm_data)
                self.logger.debug(f"Updated DataFeedManager for new token {token_symbol}")
            
            # Publish event to internal event bus
            self.event_bus.publish(Event(
                event_type=EventType.TOKEN_CREATED,
                data=event_data,
                source="pumpportal_websocket"
            ))
            self.logger.info(f"Published TOKEN_CREATED event for {token_symbol} ({token_id})")
            
            # Queue for subscription if dynamic discovery is enabled
            if self.enable_dynamic_token_discovery:
                 # Use create_task to avoid blocking the processing loop
                 # Validation happens inside _queue_token_subscription
                 asyncio.create_task(self._queue_token_subscription(token_id))
                
        except Exception as e:
            self.logger.error(f"Error processing token creation event details: {e}", exc_info=True)

    def _handle_trade_data(self, data: Dict[str, Any]):
        """Process data confirmed to be a token trade event for a subscribed token."""
        try:
            token_id = data.get('mint')
            tx_type = data.get('txType', '').lower()
            # Should be pre-validated, but check again
            if not token_id or tx_type not in ['buy', 'sell']:
                 self.logger.error(f"Assertion failed: _handle_trade_data called with invalid data. Mint: {token_id}, Type: {tx_type}")
                 return

            is_buy = tx_type == 'buy'
            trader = data.get('traderPublicKey', 'Unknown')
            signature = data.get('signature', '')

            # Safely convert numeric fields
            try: price = float(data.get('solAmount', 0.0))
            except (ValueError, TypeError): price = 0.0
            try: amount = float(data.get('tokenAmount', 0.0))
            except (ValueError, TypeError): amount = 0.0

            # Timestamp conversion (assuming milliseconds from server)
            try: 
                timestamp_ms = int(data.get('timestamp', time.time() * 1000))
                timestamp_sec = timestamp_ms / 1000.0
            except (ValueError, TypeError): 
                timestamp_sec = time.time()
            
            if self.debug_mode:
                 self.logger.debug(f"Processing Trade Details: {'BUY' if is_buy else 'SELL'} {amount:.4f} {token_id} @ {price:.6f} SOL by ...{trader[-6:]}")
            
            # Create event data for the event bus
            event_data = {
                'token_id': token_id,
                'is_buy': is_buy,
                'is_sell': not is_buy,
                'price': price,
                'amount': amount,
                'trader': trader,
                'timestamp': timestamp_sec,
                'signature': signature,
                'raw_data': data
            }
            
            # Update data feed manager if available
            if self.data_feed_manager:
                dfm_update = {
                    'price': price,
                    'last_trade_amount': amount,
                    'last_trade_side': 'buy' if is_buy else 'sell',
                    'last_trade_timestamp': event_data['timestamp'],
                    'last_updated': time.time(), # Use current time for last update
                    'is_trade': True
                }
                self.data_feed_manager.update_token_data(token_id, dfm_update)
                if self.debug_mode:
                    self.logger.debug(f"Updated DataFeedManager for trade on {token_id}")
            
            # Publish the TOKEN_TRADE event
            self.event_bus.publish(Event(
                event_type=EventType.TOKEN_TRADE,
                data=event_data,
                source="pumpportal_websocket"
            ))
            self.logger.info(f"Published TOKEN_TRADE event for {token_id} ({'BUY' if is_buy else 'SELL'} @ {price:.6f})")
            
        except Exception as e:
            self.logger.error(f"Error processing trade event details: {e}", exc_info=True)
            if self.debug_mode:
                 self.logger.error(f"Problematic trade data dict: {data}")

    async def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff after ensuring disconnection."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error(f"Maximum reconnect attempts ({self._max_reconnect_attempts}) reached. Giving up.")
            return False
            
        # Calculate backoff time
        self._reconnect_attempts += 1
        backoff_time = min(30, 2 ** (self._reconnect_attempts - 1)) # Start with 1s, 2s, 4s...
        self.logger.info(f"Waiting {backoff_time}s before reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}...")
        await asyncio.sleep(backoff_time)
        
        # Ensure fully disconnected before trying again
        await self.disconnect()
            
        # Try to connect (connect method resets attempts on success)
        self.logger.info(f"Attempting reconnection #{self._reconnect_attempts}...")
        connected = await self.connect()
        
        if connected:
            self.logger.info(f"Reconnection attempt {self._reconnect_attempts} successful.")
            # Re-setup subscriptions and restart heartbeat/streaming externally if needed
            # Or modify start_streaming to handle this state
            return True
        else:
            self.logger.error(f"Reconnection attempt {self._reconnect_attempts} failed.")
            return False

# --- Mock Client and Helper Function --- 

def generate_mock_price_history(token_id: str, start_price: float = 0.001, num_points: int = 120, volatility: float = 0.03) -> List[Tuple[datetime, float]]:
    """Generate mock price history as (datetime, price) tuples."""
    price_history: List[Tuple[datetime, float]] = []
    current_price = start_price
    # Start timestamps in the past
    current_time_sec = time.time() - (num_points * 60) 
    
    for _ in range(num_points):
        price_change = current_price * random.normalvariate(0.0005, volatility) # Slight upward bias
        current_price = max(0.000001, current_price + price_change) # Avoid zero/negative
        price_history.append((datetime.fromtimestamp(current_time_sec), current_price))
        current_time_sec += 60 # Increment by 1 minute
    
    return price_history

class MockWebSocketClient(BaseEventPublisher):
    """Simulates PumpPortal WebSocket for testing.
    Generates TOKEN_CREATED and TOKEN_TRADE events.
    """
    def __init__(self, event_bus: EventBus, 
                 data_feed_manager: Optional[DataFeedManager] = None, 
                 feature_system: Optional[Any] = None): # Use Any if FeatureSystem type isn't imported
        super().__init__(event_bus)
        self.data_feed_manager = data_feed_manager
        self.feature_system = feature_system
        self.logger = logging.getLogger("MockWebSocketClient") # Use module logger
        
        self.connected = False
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._trade_sim_task: Optional[asyncio.Task] = None
        
        # Mock token storage
        self.tokens: List[Dict[str, Any]] = []
        self.token_mints: Set[str] = set()

        # Initialize mock state
        self._initialize_mock_tokens(5)
        
        # Register as a data source if manager is provided
        if self.data_feed_manager:
            self.data_feed_manager.register_data_source("mock_websocket")
            self.logger.info("Registered mock WebSocket as data source.")

    def _initialize_mock_tokens(self, num_tokens: int):
        """Create initial set of mock tokens."""
        self.logger.debug(f"Initializing {num_tokens} mock tokens...")
        token_prefixes = ["PUMP", "MEME", "SOL", "BONK", "PEPE", "DEGEN"]
        token_suffixes = ["COIN", "TOKEN", "MOON", "MARS", "INU", "FLOKI"]
        
        for i in range(num_tokens):
            prefix = random.choice(token_prefixes)
            suffix = random.choice(token_suffixes)
            token_symbol = f"{prefix}{suffix}"
            mint_address = ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=43))
            base_price = random.uniform(0.0001, 0.01)
            
            token_data = {
                "symbol": token_symbol,
                "mint": mint_address,
                "name": f"{prefix} {suffix} Mock",
                "price": base_price, # Current simulated price
                "solAmount": base_price, # Initial price for creation event
                "marketCapSol": random.uniform(1, 50),
                "creator": f"mock_creator_{i}",
                "traderPublicKey": f"mock_creator_{i}"
            }
            self.tokens.append(token_data)
            self.token_mints.add(mint_address)
            
            # Populate data feed manager and feature system if they exist
            self._populate_external_systems(token_data)
        
        self.logger.info(f"Initialized with {len(self.tokens)} mock tokens.")

    def _populate_external_systems(self, token_data: Dict[str, Any]):
        """Update DFM and Feature System for a given mock token."""
        mint_address = token_data['mint']
        base_price = token_data['solAmount']
        
        if self.data_feed_manager:
             dfm_data = {
                'token_id': mint_address,
                'name': token_data['name'], 'symbol': token_data['symbol'],
                'price': base_price, 'market_cap': token_data['marketCapSol'],
                'creation_time': time.time(), 'last_updated': time.time(),
                'is_new_token': True
             }
             self.data_feed_manager.update_token_data(mint_address, dfm_data)
        
        if self.feature_system:
            # Assumes feature system has a get_provider method
            price_provider = self.feature_system.get_provider("price_provider") 
            if price_provider:
                price_history = generate_mock_price_history(mint_address, start_price=base_price)
                for timestamp_dt, price in price_history:
                     # Assumes provider expects (token_id, price, timestamp_sec)
                     if hasattr(price_provider, 'update_price'):
                          price_provider.update_price(mint_address, price, timestamp_dt.timestamp())
                     else:
                           self.logger.warning("Price provider lacks 'update_price' method.")
                           break # Stop trying if method missing
                self.logger.debug(f"Generated {len(price_history)} history points for {token_data['symbol']}")
            else:
                self.logger.warning("'price_provider' not found in feature system for mock history.")

    async def connect(self) -> bool:
        """Simulate connecting."""
        if self.connected:
             return True
        self.logger.info("Connecting mock client...")
        await asyncio.sleep(0.1) # Simulate tiny delay
        self.connected = True
        self.logger.info("Mock client connected.")
        return True
    
    async def start_streaming(self) -> bool:
        """Start mock data generation tasks."""
        if self._running:
            self.logger.warning("Mock streaming already running.")
            return True
            
        if not self.connected:
            await self.connect()
            
        self._running = True
        # Ensure tasks are only created if not already running
        if self._stream_task is None or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_mock_data(), name="MockTokenCreationStream")
        if self._trade_sim_task is None or self._trade_sim_task.done():
             self._trade_sim_task = asyncio.create_task(self._simulate_token_trade(), name="MockTradeSimStream")

        self.logger.info("Started mock data streaming tasks.")
        return True
    
    async def stop_streaming(self):
        """Stop mock data generation tasks."""
        self.logger.info("Stopping mock data streaming...")
        self._running = False
        self.connected = False
        
        tasks_to_stop = [self._stream_task, self._trade_sim_task]
        for task in tasks_to_stop:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.CancelledError:
                    self.logger.debug(f"Mock task {task.get_name()} cancelled.")
                except asyncio.TimeoutError:
                     self.logger.warning(f"Mock task {task.get_name()} did not cancel quickly.")
                except Exception as e:
                     self.logger.error(f"Error cancelling mock task {task.get_name()}: {e}")
        
        self._stream_task = self._trade_sim_task = None
        self.logger.info("Stopped mock data streaming.")
    
    async def _stream_mock_data(self):
        """Task to periodically create new mock tokens."""
        self.logger.debug("Mock token creation stream started.")
        try:
            # Publish initial tokens
            for token_data in self.tokens:
                if not self._running: break
                await self._publish_token_creation(token_data)
                await asyncio.sleep(random.uniform(0.2, 0.8))
            
            # Create new tokens periodically
            while self._running:
                await asyncio.sleep(random.uniform(10, 25)) # Interval for new tokens
                if not self._running: break
                await self._publish_new_token()
                # Optional: Prune old tokens
                if len(self.tokens) > 30:
                     oldest = self.tokens.pop(0)
                     self.token_mints.discard(oldest['mint'])
                     self.logger.debug(f"Pruned oldest mock token: {oldest['symbol']}")
                
        except asyncio.CancelledError:
            self.logger.debug("Mock token creation stream cancelled.")
        except Exception as e:
            self.logger.error(f"Error in mock token stream: {e}", exc_info=True)
        finally:
             self.logger.debug("Mock token creation stream finished.")
    
    async def _publish_token_creation(self, token_data: Dict[str, Any]):
        """Publish a TOKEN_CREATED event for a mock token."""
        event_payload = {
            'mint': token_data['mint'], 'token_id': token_data['mint'],
            'name': token_data.get('name', 'Mock Token'),
            'symbol': token_data.get('symbol', 'MOCK'),
            'initial_price': token_data.get('solAmount', 0.0),
            'market_cap': token_data.get('marketCapSol', 0.0),
            'creator': token_data.get('creator', 'mock_creator'),
            'timestamp': time.time(),
            'raw_data': { # Mimic structure often used by handlers
                'txType': 'create',
                'mint': token_data['mint'],
                'name': token_data.get('name', 'Mock Token'),
                'symbol': token_data.get('symbol', 'MOCK'),
                'solAmount': token_data.get('solAmount', 0.0),
                'marketCapSol': token_data.get('marketCapSol', 0.0),
                'traderPublicKey': token_data.get('creator', 'mock_creator'),
                'signature': f"mocksig_{random.randint(1000,9999)}",
                'timestamp': int(time.time() * 1000)
            }
        }
        self.publish_event(EventType.TOKEN_CREATED, event_payload, source="mock_websocket")
        self.logger.info(f"Published mock TOKEN_CREATED: {token_data['symbol']}")
    
    async def _publish_new_token(self):
        """Generate data for a new mock token and publish its creation."""
        prefix = random.choice(["NEW", "FRESH", "HOT", "ALPHA", "BETA"])
        suffix = random.choice(["GEM", "ROCKET", "X", "AI", "DAO"])
        token_symbol = f"{prefix}{suffix}"
        mint_address = ''.join(random.choices('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz', k=43))
        base_price = random.uniform(0.00005, 0.005)
        
        new_token_data = {
            "symbol": token_symbol, "mint": mint_address,
            "name": f"{prefix} {suffix} Mock Project",
            "price": base_price, "solAmount": base_price,
            "marketCapSol": random.uniform(0.5, 20),
            "creator": f"new_creator_{random.randint(1,99)}",
            "traderPublicKey": f"new_creator_{random.randint(1,99)}"
        }
        
        self.tokens.append(new_token_data)
        self.token_mints.add(mint_address)
        self._populate_external_systems(new_token_data)
        await self._publish_token_creation(new_token_data)
        self.logger.debug(f"Created new mock token: {token_symbol}")
    
    async def _simulate_token_trade(self):
        """Task to simulate trades for existing mock tokens."""
        self.logger.debug("Mock trade simulation stream started.")
        try:
            while self._running:
                await asyncio.sleep(random.uniform(0.1, 1.0)) # Trade frequency
                if not self._running or not self.tokens: continue
                    
                try:
                    token_to_trade = random.choice(self.tokens)
                    token_id = token_to_trade['mint']
                except IndexError: continue

                # Simulate price change
                current_price = token_to_trade["price"]
                momentum = random.uniform(-0.01, 0.015) # Small momentum factor
                price_change_pct = random.normalvariate(momentum, 0.04)
                new_price = max(0.000001, current_price * (1 + price_change_pct))
                token_to_trade["price"] = new_price
                
                # Simulate trade
                sol_amount = random.uniform(0.01, 1.0)
                token_amount = sol_amount / new_price if new_price > 0 else 0
                is_buy = price_change_pct >= -0.005 # Bias towards buys slightly
                tx_type = 'buy' if is_buy else 'sell'
                trader = f"mock_trader_{random.randint(1,99)}"
                timestamp_ms = int(time.time() * 1000)
                signature = f"mocksig_trade_{random.randint(1000,9999)}"

                # Payload mimicking raw data
                trade_payload = {
                    'mint': token_id, 'txType': tx_type, 'traderPublicKey': trader,
                    'solAmount': sol_amount, 'tokenAmount': token_amount,
                    'timestamp': timestamp_ms, 'signature': signature
                }

                # Event data for the bus
                event_data_for_bus = {
                    'token_id': token_id, 'is_buy': is_buy, 'is_sell': not is_buy,
                    'price': new_price, 'amount': token_amount, 'trader': trader,
                    'timestamp': timestamp_ms / 1000.0, 'signature': signature,
                    'raw_data': trade_payload
                }

                # Update DFM if present
                if self.data_feed_manager:
                     dfm_update = {
                         'price': new_price, 'last_trade_amount': token_amount,
                         'last_trade_side': tx_type, 'last_trade_timestamp': event_data_for_bus['timestamp'],
                         'last_updated': time.time(), 'is_trade': True
                     }
                     self.data_feed_manager.update_token_data(token_id, dfm_update)
                
                # Publish event
                self.publish_event(EventType.TOKEN_TRADE, event_data_for_bus, source="mock_websocket")
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Published mock TRADE: {token_to_trade['symbol']} {tx_type.upper()} {token_amount:.2f} @ {new_price:.6f}")
                                
        except asyncio.CancelledError:
            self.logger.debug("Mock trade simulation stream cancelled.")
        except Exception as e:
            self.logger.error(f"Error in mock trade simulation: {e}", exc_info=True)
        finally:
             self.logger.debug("Mock trade simulation stream finished.") 