# Event-Driven Trading System

This repository contains an event-driven trading system that utilizes WebSocket connections to receive real-time market data and execute trades.

## Structure

```
.
├── config/                  # Configuration files
│   ├── .env.example         # Example environment file
│   ├── schema.sql           # Database schema
│   └── config_sol_mock.json # Mock configuration for SOL trading
│
├── src/                     # Source code
│   ├── core/                # Core components
│   │   ├── events/          # Event system
│   │   ├── data/            # Data feeds
│   │   ├── features/        # Feature calculation
│   │   └── trading/         # Trading logic
│   │
│   └── run_sol_trading.py   # Main script to run the trading system
│
├── tests/                   # Test files
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
│
├── examples/                # Example scripts
├── scripts/                 # Utility scripts
└── README.md                # This file
```

## Event System

The event system is the heart of this application, providing a pub/sub architecture that allows components to communicate asynchronously. Key components include:

- **EventBus**: Central message hub for event distribution
- **Event**: Base class for all events with type, data, and priority
- **EventSubscriber**: Interface for components that receive events
- **EventPublisher**: Interface for components that produce events

## Data Feeds

The system supports multiple data sources:

- **SocketDataFeed**: Real-time data from WebSocket connections
- **DatabaseDataFeed**: Historical data from PostgreSQL database
- **DataFeedManager**: Manages multiple data feeds

## Feature System

Features are calculated from raw data and used to make trading decisions:

- **FeatureSystem**: Orchestrates feature calculation
- **FeatureProvider**: Interface for components that calculate features
- **FeatureConsumer**: Interface for components that use features

## Trading System

Trading logic is separated into multiple components:

- **TradingEngine**: Main component for making trade decisions
- **PositionManager**: Manages open and closed positions
- **RiskManager**: Handles risk calculations and limits
- **TradeExecutor**: Executes trades and handles orders

## Getting Started

1. Copy the example environment file and adjust it to your needs:
   ```
   copy config\.env.example .env
   ```

2. Set up your database using the schema:
   ```
   psql -U postgres -d your_database -f config/schema.sql
   ```

3. Run the trading system:
   ```
   python src/run_sol_trading.py --config config/config_sol_mock.json
   ```