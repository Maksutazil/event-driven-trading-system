# Event-Driven Trading System

This repository contains a flexible, loosely-coupled trading system architecture based on an event-driven design pattern. It provides a robust foundation for building algorithmic trading applications with high modularity and testability.

## System Architecture

The core of the system is an event-driven architecture that allows components to communicate asynchronously through events. This design enables:

- **Loose coupling**: Components interact through events, not direct method calls
- **Flexibility**: Easy to add, remove, or replace components
- **Testability**: Components can be tested in isolation
- **Concurrency**: Asynchronous event processing
- **Scalability**: Components can be distributed across processes/machines

### Core Components

#### Event System

- **Event**: Discrete messages with a type, data payload, priority, and metadata
- **EventBus**: Central message broker that handles event subscription and publishing
- **EventSubscriber**: Interface for components that receive events
- **EventPublisher**: Interface for components that produce events
- **EventDispatcher**: Combined subscriber and publisher

#### Trading Components

The system includes several example components for trading:

- **WebSocket Client**: Connects to data sources and publishes market data events
- **TradingEngine**: Processes market data, generates signals, and executes trades
- **PortfolioManager**: Tracks positions, trades, and performance metrics

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`)

### Configuration

You can configure the trading system in three ways:

1. **Command-line arguments** - Pass options directly when running the script
2. **Configuration file** - Create a `config.json` file in the root directory
3. **Environment variables** - Set variables in a `.env` file or system environment

#### PumpFun Configuration

For PumpFun integration, specify the following settings:

- `websocket_url`: Set to `wss://pumpportal.fun/api/data`
- `watched_accounts`: Array of account addresses to track trades for
- `watched_tokens`: Array of token addresses to track trades for

Example `.env` file:
```
TRADING_WEBSOCKET_URL=wss://pumpportal.fun/api/data
TRADING_API_KEY=your_api_key_here
TRADING_WATCHED_ACCOUNTS=AArPXm8JatJiuyEffuC1un2Sc835SULa4uQqDcaGpAjV
TRADING_WATCHED_TOKENS=91WNez8D22NwBssQbkzjy4s2ipFrzpmn5hfvWVe2aY5p
TRADING_LOG_LEVEL=INFO
```

Example `config.json`:
```json
{
    "websocket_url": "wss://pumpportal.fun/api/data",
    "log_level": "INFO",
    "watched_accounts": [
        "AArPXm8JatJiuyEffuC1un2Sc835SULa4uQqDcaGpAjV"
    ],
    "watched_tokens": [
        "91WNez8D22NwBssQbkzjy4s2ipFrzpmn5hfvWVe2aY5p"
    ]
}
```

### Running the Examples

The repository includes example implementations to demonstrate the event system:

1. Basic Event System Example:
   ```bash
   python examples/event_system_basic.py
   ```

2. SOL Trading Example with mock data:
   ```bash
   python examples/run_sol_trading.py --use-mock
   ```

3. SOL Trading Example with PumpFun data:
   ```bash
   python examples/run_sol_trading.py --websocket-url wss://pumpportal.fun/api/data
   ```

   Or using a config file:
   ```bash
   python examples/run_sol_trading.py --config my_config.json
   ```

## Implementation Details

### Event Types

The system defines multiple event types for different aspects of trading:

- System events (connection status, errors)
- Data events (market data updates)
- Token events (new tokens, price updates)
- Trading events (signals, executions, position updates)
- Feature events (indicator updates)

### Event Processing Model

The EventBus supports both synchronous and asynchronous event processing:

- **Synchronous**: Events are processed immediately when published
- **Asynchronous**: Events are queued and processed in a background thread

### Event Prioritization

Events have priority levels that determine processing order:

- CRITICAL: Highest priority (processed first)
- HIGH: Important events
- NORMAL: Default priority
- LOW: Background/non-urgent events

### Data Source Formats

The system supports multiple data source formats:

- **default**: The default format expected by the trading system
- **binance**: Format used by Binance WebSocket API
- **pumpfun**: Format used by PumpFun WebSocket API
- **custom**: Custom format that can be implemented by extending the client

## Supported Data Sources

### PumpFun WebSocket API

The system includes native support for the PumpFun WebSocket API, which provides:

- New token creation events
- Trade events for specific accounts
- Trade events for specific tokens

The WebSocket client automatically handles the subscription process based on your configuration.

### Binance WebSocket API

The system includes support for Binance WebSocket API, which provides:

- Trade data for cryptocurrency pairs
- Real-time price updates
- Market information

### PostgreSQL Historical Data

The system includes support for loading historical trade data from a PostgreSQL database:

- Token data from the `Token` table
- Trade data from the `Trade` table
- Simulated event streaming with configurable delay

This allows you to run the trading system with historical data for backtesting and analysis. The `PostgresDataManager` class handles loading data from the database and publishing it as events with the same format as the real-time data sources.

#### Database Schema

The system expects the following schema in your PostgreSQL database:

```sql
CREATE TABLE IF NOT EXISTS "Token" (
    id text NOT NULL,
    address text NOT NULL,
    name text NOT NULL,
    symbol text NOT NULL,
    "createdAt" timestamp(3) without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata jsonb,
    CONSTRAINT "Token_pkey" PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS "Trade" (
    id text NOT NULL,
    "tokenId" text NOT NULL,
    price double precision NOT NULL,
    amount double precision NOT NULL,
    type text NOT NULL,
    "timestamp" timestamp(3) without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "walletId" text NOT NULL,
    CONSTRAINT "Trade_pkey" PRIMARY KEY (id),
    CONSTRAINT "Trade_tokenId_fkey" FOREIGN KEY ("tokenId")
        REFERENCES "Token" (id)
);
```

#### Running with Historical Data

To run the trading system with historical data from PostgreSQL:

```bash
python examples/run_db_historical_trading.py --debug-mode --days 7 --stream-delay 100
```

Options:
- `--debug-mode`: Enable debug logging
- `--days`: Number of days of historical data to load (default: 7)
- `--stream-delay`: Delay between events in milliseconds (default: 100, lower = faster simulation)
- `--token-ids`: Comma-separated list of token IDs to stream (optional, all tokens if omitted)

Database connection parameters can be specified in a `.env` file:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password_here
```

## Extending the System

### Creating a New Component

To create a new component that interacts with the event system:

1. For components that only publish events, extend `BaseEventPublisher`
2. For components that only receive events, extend `BaseEventSubscriber`
3. For components that do both, extend `EventDispatcher`

Example:

```python
from src.core.events import EventDispatcher, Event, EventType, EventBus

class MyComponent(EventDispatcher):
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self.register_handler(EventType.SYSTEM, self.handle_system_event)
    
    def handle_system_event(self, event: Event) -> None:
        # Process system event
        print(f"Received system event: {event.data}")
        
        # Publish a response event
        self.publish_event(
            event_type=EventType.DATA_UPDATE,
            data={"message": "Response to system event"}
        )
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Testing Framework

The system includes a comprehensive testing framework with unit tests, integration tests, and CI/CD pipeline integration.

### Running Tests

You can run the tests using the provided scripts:

1. Run all tests:
   ```bash
   python run_integration_tests.py
   ```

2. Run a specific integration test:
   ```bash
   python run_specific_test.py tests.integration.ml_trading.test_model_prediction_flow
   ```

3. Run tests with coverage:
   ```bash
   pytest --cov=src tests/
   ```

### Integration Tests

The system includes several key integration tests:

1. **Model Prediction Flow Test**: Validates that model predictions properly influence trading signals
2. **Error Handling Test**: Ensures errors are properly propagated and handled across component boundaries
3. **Feature Naming Standardization Test**: Verifies feature naming consistency across components
4. **Model Training Workflow Test**: Tests the model training pipeline and workflow
5. **Configuration Consistency Test**: Ensures configuration is properly propagated across components

### Continuous Integration

The project includes GitHub Actions workflows in `.github/workflows/` that automatically run tests, linting, and security checks on every push and pull request to main branches.

To set up the development environment with all testing tools:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=src --cov-report=html tests/
```

This will create an HTML report in the `htmlcov` directory that you can open in a browser.