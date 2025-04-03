# Model Training Module

This document describes the model training system that allows training pump detection models using historical data from a PostgreSQL database.

## Overview

The model training module is designed to extract historical trade data from a PostgreSQL database, compute features at fine-grained time intervals (as small as 2 seconds), and train models to detect pump and dump patterns. The module supports both the standard `PumpPredictorModel` and the `EarlyPumpPredictor` for newly created tokens.

## Components

The model training system consists of:

1. **ModelTrainer**: Core class that handles data extraction, feature computation, and model training
2. **PostgresDataManager**: Connects to the PostgreSQL database and retrieves historical data
3. **Feature Providers**: Compute features for model training
4. **Models**: ML models that will be trained with the data

## Using with 2-Second Intervals

The model training system is specifically designed to work with very fine-grained time intervals, as small as 2 seconds. This allows for more precise detection of pump and dump patterns, especially in the fast-moving crypto market.

### Data Resampling

When working with 2-second intervals, the system:

1. Extracts raw trade data from the database
2. Resamples the data to 2-second intervals using OHLC (Open, High, Low, Close) aggregation
3. Creates additional features for each interval
4. Transforms the data for model training

### Feature Adjustments

When using 2-second intervals, some features need adjustment:

1. **Price velocity features** need to be scaled appropriately for the shorter time frame
2. **Volume-based features** are normalized for the shorter intervals
3. **Pattern detection** becomes more sensitive to micro-patterns

The `adjust_features_for_time_interval()` method in the ModelTrainer handles these adjustments automatically.

## Setup and Usage

### Required Database Schema

The PostgreSQL database should have the following tables:

- **Token**: Contains token information (id, address, name, symbol)
- **Trade**: Contains trade records (id, tokenId, price, amount, type, timestamp)

### Basic Usage Example

```python
from src.core.ml.training.model_trainer import ModelTrainer
from src.core.db.postgres_data_manager import PostgresDataManager

# Set up database connection
db_config = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'trading_data',
    'user': 'postgres',
    'password': 'password'
}

# Create PostgresDataManager
postgres_manager = PostgresDataManager(
    event_bus=event_bus,
    data_feed_manager=data_feed_manager,
    connection_params=db_config
)

# Connect to database
await postgres_manager.connect()

# Create model trainer
trainer = ModelTrainer(
    postgres_manager=postgres_manager,
    feature_manager=feature_manager,
    time_interval_seconds=2,  # Using 2-second intervals
    training_window_hours=24   # Using 24 hours of data
)

# Run training session
results = await trainer.run_training_session()
```

See the file `examples/model_training_example.py` for a complete example.

## Training with Labeled Data

For supervised learning, the module supports training with labeled data:

```python
# Create labeled data
labeled_data = {
    'token_id_1': [
        {
            'type': 'pump',
            'start_time': timestamp1,
            'end_time': timestamp2,
            'class_id': 1
        },
        {
            'type': 'dump',
            'start_time': timestamp3,
            'end_time': timestamp4,
            'class_id': 2
        }
    ]
}

# Train with labeled data
results = await trainer.run_training_session(labeled_data=labeled_data)
```

## Model Evaluation and Saving

The training process includes:

1. Splitting data into training and validation sets
2. Training models on the training set
3. Evaluating models on the validation set
4. Saving model configurations and evaluation results

Trained models are saved to the specified `models_dir` directory.

## Performance Considerations

Training with 2-second intervals requires:

1. More computational resources than longer intervals
2. Sufficient historical data for meaningful patterns
3. Appropriate feature scaling for the shorter time frame

For optimal performance, consider:

- Using a database with good indexing on token IDs and timestamps
- Preprocessing data to remove outliers
- Caching feature calculations where appropriate
- Using a subset of tokens for initial training runs

## References

For more information on the underlying models:

- `src/core/ml/models/pump_predictor.py`: Standard pump detection model
- `src/core/ml/models/early_pump_predictor.py`: Model for newly created tokens
- `doc/pump_detection.md`: Overview of pump detection features and models
- `doc/early_pump_detection.md`: Documentation for early-stage pump detection 