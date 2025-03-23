# Event-Driven Trading System

This repository contains an implementation of an event-driven trading system for cryptocurrency markets. The system is designed to be extensible, maintainable, and scalable, utilizing an event-driven architecture for loose coupling between components.

## System Architecture

The system consists of several core components:

1. **Event System**: The backbone of the application, responsible for communication between components
2. **Data Feeds**: Components that connect to various data sources like WebSockets and databases
3. **Feature System**: Calculates and manages derived features from raw data
4. **Trading Engine**: Makes trading decisions based on signals from the feature system
5. **Position Manager**: Keeps track of open and closed positions
6. **Risk Manager**: Manages risk per trade and overall portfolio risk

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository
```bash
git clone https://github.com/Maksutazil/event-driven-trading-system.git
cd event-driven-trading-system
```

2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

The system can be used in different modes:

- **Live Trading**: Connect to real exchanges and execute trades
- **Paper Trading**: Simulate trades without real money
- **Backtesting**: Test strategies against historical data

### Basic Example

```python
from src.core.events import EventBus
from src.core.data import DataFeedManager, SocketDataFeed
from src.core.features import FeatureSystem
from src.core.trading import TradingSystemFactory

# Create the event bus
event_bus = EventBus()

# Setup data feeds
data_feed_manager = DataFeedManager(event_bus)
socket_feed = SocketDataFeed(event_bus, websocket_uri="wss://example.com/stream")
data_feed_manager.register_feed("main", socket_feed)

# Setup feature system
feature_system = FeatureSystem(event_bus)

# Create trading system components
trading_system = TradingSystemFactory.create_paper_trading_system(
    event_bus=event_bus,
    feature_system=feature_system,
    price_fetcher=socket_feed.get_price,
    initial_capital=10000.0
)

# Start the system
event_bus.start()
data_feed_manager.connect_all()

# Subscribe to tokens
data_feed_manager.subscribe_token("BTC")
data_feed_manager.subscribe_token("ETH")

# Trading system runs automatically, processing events as they come in
```

## Handling Large Files and Git Repository Maintenance

If you encounter issues with large files in the Git repository, especially with the `trading_venv` folder or other large files that were accidentally committed, you can use the provided scripts to clean up the repository:

### Option 1: Using Shell Script (requires BFG or git filter-branch)

```bash
# Make the script executable
chmod +x clean_large_files.sh

# Run the script
./clean_large_files.sh
```

### Option 2: Using Python Script (requires git-filter-repo)

```bash
# Install git-filter-repo if not already installed
pip install git-filter-repo

# Run the script
python clean_large_files_with_filter_repo.py
```

After running either script, you'll need to force push the changes to GitHub:

```bash
git push origin --force --all
```

**Warning**: This will rewrite the Git history. All collaborators will need to re-clone the repository or take other steps to synchronize their local repositories.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.