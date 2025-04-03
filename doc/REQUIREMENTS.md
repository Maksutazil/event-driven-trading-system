# PumpFun Python Refactoring Requirements

## Background
The current implementation has a working socket connection in token_monitor.py and backtester, along with proper database connectivity. The system needs refactoring to handle both data sources effectively through a single interface, handle events from the socket efficiently, and properly monitor multiple tokens concurrently.

***IMPORTANT: The current socket and database connections are fully functional and working correctly. The refactoring MUST maintain compatibility with these existing connections, using exactly the same interfaces, names, and data types to ensure continued operation.***

## Current Architecture Issues

1. **No unified data interface**: Socket connections in TokenMonitor and database connections are handled separately.
2. **Overlapping responsibilities**: PaperTrader and TradingEngine have some duplicated functionality.
3. **Feature engineering is distributed**: Feature generation happens in multiple places without a single source of truth.
4. **Limited event handling**: System needs to react better to events from socket connections.
5. **Thread management**: Need better handling for monitoring multiple tokens concurrently.

## Refactoring Requirements

### 1. Unified Data Interface
- Create a `DataFeedInterface` abstract class for all data sources
- Implement concrete classes:
  - `SocketDataFeed`: For real-time WebSocket data
  - `DatabaseDataFeed`: For historical DB data
- Ensure consistent data formats and event-driven architecture
- Data feeds must handle connection disruptions gracefully
- Implement proper error handling and reconnection logic
- **MUST maintain exact compatibility with existing socket and DB interfaces, preserving all method names, parameters, and data types**

### 2. Feature Engineering as Single Source of Truth
- Create a centralized `FeatureManager` class
- Implement caching and performance optimizations
- Ensure all components use this for both trading and model training
- Feature history should be maintained efficiently
- Implement feature versioning to track changes
- Provide efficient batch and real-time feature generation
- **Maintain compatibility with existing feature generation functionality**

### 3. Trading Component Refactoring
- Split the system into components with single responsibilities:
  - `TradingEngine`: Decision-making logic
  - `PositionManager`: Manages open positions and their state
  - `TradeExecutor`: Handles the actual execution of trades
  - `RiskManager`: Handles position sizing and risk calculations
- Ensure clear interfaces between components
- Use dependency injection to improve testability
- Maintain state persistence for crash recovery

### 4. Event-Driven Architecture
- Implement an `EventBus` for communication between components
- Use a publish-subscribe pattern for loosely coupled components
- Define clear event types (new token, new trade, feature update, etc.)
- Ensure thread-safe event processing
- Add event prioritization and queuing mechanisms
- Implement event logging for debugging and auditing

### 5. Thread-Based Token Monitoring
- Implement a thread pool for monitoring multiple tokens
- Ensure thread safety for shared resources
- Add proper cleanup and shutdown mechanisms
- Monitor thread health and restart failed threads
- Implement thread prioritization for important tokens
- Add performance metrics for thread utilization

### 6. Code Quality Requirements
- **Continuously check for code duplication**: During implementation, regularly review to ensure no redundant code is created
- **Adhere to Single Responsibility Principle**: Each class and method should have exactly one reason to change
- **Clean Code Principles**: Follow clean code practices for naming, function size, commenting, and organization
- **Error Handling**: Implement proper error handling and reporting throughout the system
- **Logging**: Add detailed logging for monitoring and debugging

### 7. Tools and Resources
- **Always use all available MCP servers** (GitHub, Sequential Thinking, Brave Search) during implementation
- If any MCP servers are unavailable, notify the user immediately
- Use GitHub for code versioning and issue tracking
- Utilize sequential thinking for complex problem solving
- Leverage Brave Search for finding relevant documentation and solutions

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create the event system (EventBus, Event classes)
2. Implement the data feed interfaces and concrete implementations
3. Develop the centralized feature engineering component

### Phase 2: Trading Components
1. Refactor the TradingEngine to use the new event system
2. Create separate PositionManager and TradeExecutor components
3. Implement a dedicated RiskManager

### Phase 3: Integration
1. Connect all components through the EventBus
2. Implement thread pool for token monitoring
3. Add error handling and recovery mechanisms

### Phase 4: Testing and Optimization
1. Create comprehensive unit and integration tests
2. Profile and optimize performance bottlenecks
3. Enhance logging and monitoring

## Key Interfaces

```python
# Data Feed Interface
class DataFeedInterface:
    def connect(self): pass
    def disconnect(self): pass
    def subscribe(self, token_id): pass
    def unsubscribe(self, token_id): pass
    async def get_historical_data(self, token_id, start_time, end_time): pass
    def register_callback(self, event_type, callback): pass

# Socket-based implementation
class SocketDataFeed(DataFeedInterface, EventPublisher):
    # Implementation for real-time data via WebSocket
    pass

# Database-based implementation
class DatabaseDataFeed(DataFeedInterface, EventPublisher):
    # Implementation for historical data from database
    pass

# Manager that coordinates multiple data feeds
class DataFeedManager:
    def __init__(self, event_bus):
        self.data_feeds = []
        self.event_bus = event_bus
    
    def add_data_feed(self, data_feed): pass
    def subscribe_to_token(self, token_id): pass
    def unsubscribe_from_token(self, token_id): pass

# Feature Management
class FeatureManager:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.feature_cache = FeatureCache()
    
    def register_features(self, feature_set): pass
    def generate_features(self, token_id, raw_data): pass
    def get_cached_features(self, token_id): pass
    def clear_cache(self): pass
    def update_feature_history(self, token_id, timestamp, features): pass

class FeatureCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def add(self, token_id, timestamp, features): pass
    def get(self, token_id, timestamp=None): pass
    def clear(self, token_id=None): pass

# Trading Components
class TradingEngine(EventSubscriber):
    def __init__(self, event_bus, position_manager, risk_manager):
        self.event_bus = event_bus
        self.position_manager = position_manager
        self.risk_manager = risk_manager
    
    def on_feature_event(self, event): pass
    def analyze_token(self, token_id, features): pass
    def make_decision(self, token_id, features): pass

class PositionManager(EventPublisher, EventSubscriber):
    def __init__(self, event_bus, trade_executor):
        self.event_bus = event_bus
        self.trade_executor = trade_executor
        self.open_positions = {}
    
    def open_position(self, token_id, size, entry_price): pass
    def close_position(self, token_id, exit_price, reason): pass
    def update_positions(self): pass

class TradeExecutor(EventSubscriber):
    def __init__(self, event_bus, trading_db):
        self.event_bus = event_bus
        self.trading_db = trading_db
    
    def execute_entry(self, token_id, size, price): pass
    def execute_exit(self, token_id, price, reason): pass

class RiskManager:
    def __init__(self, capital, max_risk_per_trade, max_concurrent_positions):
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_concurrent_positions = max_concurrent_positions
    
    def calculate_position_size(self, token_id, entry_price, stop_loss): pass
    def check_risk_limits(self, token_id, position_size): pass
    def update_capital(self, new_capital): pass

# Event System
class EventBus:
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type, subscriber): pass
    def unsubscribe(self, event_type, subscriber): pass
    def publish(self, event): pass

class Event:
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

class EventPublisher:
    def __init__(self, event_bus):
        self.event_bus = event_bus
    
    def publish_event(self, event_type, data): pass

class EventSubscriber:
    def on_event(self, event): pass

# Thread Management
class TokenMonitorThreadPool:
    def __init__(self, max_threads=10):
        self.max_threads = max_threads
        self.active_threads = {}
        self.thread_queue = Queue()
    
    def add_token(self, token_id, monitor_function): pass
    def remove_token(self, token_id): pass
    def start_all(self): pass
    def stop_all(self): pass
    def monitor_threads(self): pass
```

## Performance Considerations

1. **Memory Management**: 
   - Implement efficient memory usage for feature history
   - Clean up stale data regularly
   - Monitor memory usage and implement safeguards

2. **CPU Utilization**:
   - Balance thread usage based on system capabilities
   - Implement appropriate locking mechanisms to minimize contention
   - Profile and optimize CPU-intensive operations

3. **I/O Efficiency**:
   - Minimize database queries through caching
   - Batch database operations where possible
   - Optimize socket data handling

## Success Criteria

The refactored system should:

1. Handle multiple data sources transparently
2. Process events from sockets in real-time
3. Monitor multiple tokens concurrently with proper threading
4. Maintain a single source of truth for feature engineering
5. Have clean separation of responsibilities between components
6. Be easily testable and maintainable
7. Perform efficiently with minimal resource usage
8. Recover gracefully from errors and connection issues

## Implementation Guidelines

Throughout the implementation process:

1. **Prioritize interfaces first**: Define interfaces before implementations
2. **Incremental changes**: Refactor one component at a time while maintaining functionality
3. **Test thoroughly**: Create unit tests for each component before moving to the next
4. **Document as you go**: Add documentation for interfaces and key functions
5. **Regular reviews**: Continuously review for code duplication and SRP violations
6. **Use available tools**: Leverage all MCP servers for implementation assistance 

## Compatibility Requirements

### Socket Connection Compatibility
- The existing socket connection in token_monitor.py is working correctly and MUST be preserved
- All socket connection parameters, event handlers, and data structures must remain the same
- The WebSocket URI and authentication methods must not be changed
- Data parsing and event handling logic must maintain the same behavior

### Database Connection Compatibility
- Current database connections are functional and must not be disrupted
- Database schema, table names, and column definitions must remain the same
- Query structures and transaction handling must be preserved
- Connection pooling and authentication methods must be maintained

### API and Interface Consistency
- All public methods that interact with external systems must keep the same signatures
- Data types passed between components must not be changed
- Event naming and structure must be consistent with the current implementation
- Error handling patterns must remain compatible

The refactoring should create more modular components with cleaner separation of concerns, but externally visible interfaces MUST remain identical to ensure no disruption to the working functionality. 

# PumpFun Meme Coin Trading System Requirements

## Background
The system is designed to trade newly created meme coin tokens on pump.fun portal, characterized by extreme volatility and short lifespans. About 95% of newly created meme coins lose activity within 10-15 minutes or see volume completely drop to zero, requiring very time-sensitive decision making.

The current implementation has a working socket connection to pump.fun and database connectivity. The system connects to pump.fun via WebSocket to receive two types of events: token creation events and trade events. After receiving a token creation event with the token's mint address, the system subscribes to trade events for that token.

***IMPORTANT: The current socket and database connections are fully functional and working correctly. The refactoring MUST maintain compatibility with these existing connections, using exactly the same interfaces, names, and data types to ensure continued operation.***

## Market Characteristics and Trading Requirements

1. **Extreme Volatility**: Meme coins experience rapid price fluctuations requiring quick decision making
2. **Short Lifespans**: 95% of new tokens lose activity within 10-15 minutes
3. **Time Sensitivity**: Trading decisions must be made rapidly to capture opportunities
4. **Multi-Token Monitoring**: System must monitor and analyze multiple tokens simultaneously
5. **Adaptive Resource Allocation**: Focus computational resources on the most promising tokens

## Current Architecture Issues

1. **No unified data interface**: Socket connections and database connections are handled separately
2. **Overlapping responsibilities**: Components have some duplicated functionality
3. **Feature engineering is distributed**: Feature generation happens in multiple places
4. **Limited event handling**: System needs to react better to events from socket connections
5. **Thread management**: Need better handling for monitoring multiple tokens concurrently
6. **No machine learning integration**: Current decisions are rule-based rather than ML-driven
7. **Token activity analysis**: No clear mechanism to detect and deprioritize inactive tokens
8. **Position exit strategy**: No optimized exit strategy during system shutdown

## Enhanced Refactoring Requirements

### 1. Unified Data Interface
- Create a `DataFeedInterface` abstract class for all data sources
- Implement concrete classes:
  - `SocketDataFeed`: For real-time WebSocket data
  - `DatabaseDataFeed`: For historical DB data
- Ensure consistent data formats and event-driven architecture
- Data feeds must handle connection disruptions gracefully
- Implement proper error handling and reconnection logic
- **MUST maintain exact compatibility with existing socket and DB interfaces**

### 2. Feature Engineering as Single Source of Truth
- Create a centralized `FeatureManager` class
- Implement caching and performance optimizations
- Ensure all components use this for both trading and model training
- Feature history should be maintained efficiently
- Implement feature versioning to track changes
- Provide efficient batch and real-time feature generation
- **Optimize for extremely time-sensitive calculations**

### 3. Machine Learning Integration
- Create a `ModelManager` to load and serve ML models
- Implement feature transformers to prepare data for models
- Support multiple model types (classification, regression)
- Enable A/B testing between different models
- Provide standardized interfaces for model predictions
- Support real-time model updating based on new data
- Include monitoring for model drift and performance

### 4. Token Activity Analysis
- Implement an `ActivityAnalyzer` component
- Track and analyze trading volume, frequency, and price patterns
- Detect when tokens are becoming inactive
- Implement automatic deprioritization of inactive tokens
- Create a token lifecycle state machine (new, active, declining, inactive)
- Optimize monitoring resources based on token activity levels

### 5. Trading Component Refactoring
- Split the system into components with single responsibilities:
  - `TradingEngine`: Decision-making logic
  - `PositionManager`: Manages open positions and their state
  - `TradeExecutor`: Handles the actual execution of trades
  - `RiskManager`: Handles position sizing and risk calculations
- Ensure clear interfaces between components
- Use dependency injection to improve testability
- Maintain state persistence for crash recovery

### 6. Strategic Exit Management
- Implement a `GracefulExitManager` component
- Create strategies for maximizing profit during system shutdown
- Implement prioritized exit queue for positions
- Support partial position closing based on market conditions
- Add profit-taking algorithms optimized for volatile assets
- Include monitoring and reporting for exit performance

### 7. Event History Management
- Create an `EventHistoryManager` to maintain historical events by token
- Implement efficient storage and retrieval of analyzed events
- Support time-based pruning of old events to manage memory
- Provide analysis capabilities for historical events
- Ensure event history is accessible for both trading and analysis
- Implement efficient indexing for fast event retrieval

### 8. Event-Driven Architecture
- Implement an `EventBus` for communication between components
- Use a publish-subscribe pattern for loosely coupled components
- Define clear event types (new token, new trade, feature update, etc.)
- Ensure thread-safe event processing
- Add event prioritization and queuing mechanisms
- Implement event logging for debugging and auditing

### 9. Thread-Based Token Monitoring
- Implement a thread pool for monitoring multiple tokens
- Ensure thread safety for shared resources
- Add proper cleanup and shutdown mechanisms
- Monitor thread health and restart failed threads
- Implement thread prioritization for important tokens
- Add performance metrics for thread utilization
- **Optimize for monitoring many tokens simultaneously**

### 10. Code Quality Requirements
- **Continuously check for code duplication**: During implementation, regularly review to ensure no redundant code is created
- **Adhere to Single Responsibility Principle**: Each class and method should have exactly one reason to change
- **Clean Code Principles**: Follow clean code practices for naming, function size, commenting, and organization
- **Error Handling**: Implement proper error handling and reporting throughout the system
- **Logging**: Add detailed logging for monitoring and debugging

### 11. Performance Optimization
- Implement aggressive caching for frequently accessed data
- Optimize thread prioritization based on token activity
- Use non-blocking pipelines for processing token events
- Add performance benchmarking and monitoring tools
- Implement adaptive processing that adjusts to system load
- Profile and optimize critical code paths
- Use memory-efficient data structures for high-volume event processing

### 12. Error Handling and System Robustness
- Implement circuit breakers to prevent trading on erroneous data
- Add comprehensive data validation for external inputs
- Develop state persistence for crash recovery
- Implement component-level health monitoring and self-healing
- Create fallback strategies for system degradation
- Add comprehensive exception handling and reporting

## Implementation Plan

### Phase 1: Core Infrastructure Enhancements
1. Update the event system with improved prioritization
2. Implement the unified data feed interface
3. Enhance the feature engineering component for ML readiness
4. Develop the token activity analyzer

### Phase 2: Machine Learning and Trading Logic
1. Implement the ModelManager and ML integration
2. Create the EventHistoryManager for analyzed events
3. Develop the GracefulExitManager for strategic exits
4. Enhance the TradingEngine with ML-based decision making

### Phase 3: Performance and Monitoring
1. Optimize the thread pool for better resource allocation
2. Implement advanced performance monitoring
3. Add circuit breakers and validation systems
4. Develop benchmarking tools for system performance

### Phase 4: Testing and Validation
1. Create comprehensive unit and integration tests
2. Develop simulation framework for token lifecycles
3. Implement A/B testing for trading strategies
4. Add validation metrics for trading performance

## Key Interfaces

```python
# Existing interfaces remain as documented previously...

# New interfaces for enhanced requirements:

# Machine Learning Integration
class ModelManager:
    def __init__(self, event_bus, feature_manager):
        self.event_bus = event_bus
        self.feature_manager = feature_manager
        self.models = {}
    
    def load_model(self, model_id, model_path, model_type): pass
    def get_prediction(self, model_id, features): pass
    def update_model(self, model_id, new_data): pass
    def get_model_performance(self, model_id): pass

# Token Activity Analysis
class ActivityAnalyzer:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.token_activity = {}
    
    def update_activity_metrics(self, token_id, event): pass
    def get_activity_level(self, token_id): pass
    def detect_activity_change(self, token_id): pass
    def get_token_lifecycle_state(self, token_id): pass
    def recommend_priority(self, token_id): pass

# Strategic Exit Management
class GracefulExitManager:
    def __init__(self, event_bus, position_manager, trade_executor):
        self.event_bus = event_bus
        self.position_manager = position_manager
        self.trade_executor = trade_executor
        self.exit_queue = PriorityQueue()
    
    def begin_graceful_exit(self): pass
    def prioritize_exits(self): pass
    def execute_next_exit(self): pass
    def calculate_optimal_exit_price(self, token_id): pass
    def monitor_exit_progress(self): pass

# Event History Management
class EventHistoryManager:
    def __init__(self, max_events_per_token=1000, pruning_interval=300):
        self.event_history = {}
        self.max_events_per_token = max_events_per_token
        self.pruning_interval = pruning_interval
    
    def add_event(self, token_id, event_type, event_data): pass
    def get_events(self, token_id, event_type=None, start_time=None, end_time=None): pass
    def get_latest_events(self, token_id, event_type=None, count=10): pass
    def prune_old_events(self, max_age=3600): pass
    def clear_token_history(self, token_id): pass
```

## Performance Considerations

1. **Memory Management**: 
   - Implement efficient memory usage for feature history
   - Clean up stale data regularly
   - Monitor memory usage and implement safeguards
   - Use memory-efficient data structures for high-volume events

2. **CPU Utilization**:
   - Balance thread usage based on system capabilities
   - Implement appropriate locking mechanisms to minimize contention
   - Profile and optimize CPU-intensive operations
   - Prioritize computation for the most active tokens

3. **I/O Efficiency**:
   - Minimize database queries through caching
   - Batch database operations where possible
   - Optimize socket data handling
   - Implement connection pooling for database access

4. **Latency Optimization**:
   - Minimize processing pipeline latency for trade events
   - Implement fast-path processing for critical events
   - Use lock-free algorithms where appropriate
   - Optimize critical decision paths for minimum latency

## Success Criteria

The enhanced system should:

1. Handle multiple data sources transparently (live and historical)
2. Process events from sockets in real-time with minimal latency
3. Monitor multiple tokens concurrently with efficient resource allocation
4. Maintain a single source of truth for feature engineering
5. Integrate with machine learning models for trading decisions
6. Detect and adapt to token activity patterns
7. Exit positions strategically during shutdown to maximize profit
8. Maintain a comprehensive history of analyzed events by token
9. Have clean separation of responsibilities between components
10. Perform efficiently even with many tokens being monitored
11. Recover gracefully from errors and connection issues
12. Provide comprehensive monitoring and diagnostics

## Compatibility Requirements

The enhanced system must maintain compatibility with existing socket and database connections, preserving method signatures and data formats where required. The implementation should create more modular components with cleaner separation of concerns, while ensuring no disruption to working functionality. 