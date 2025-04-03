# Machine Learning Integration Guide

## Overview

This document provides comprehensive guidance on integrating machine learning (ML) models with the event-driven trading system. It covers how ML components connect with other system modules, feature naming conventions, model training workflows, and deployment practices.

## System Architecture

The ML integration in our trading system follows these key principles:

1. **Event-Driven Communication**: ML components communicate with other system modules via the central event bus.
2. **Standardized Feature Naming**: A central feature registry ensures consistent feature naming across all components.
3. **Modular Design**: ML components are designed to be modular and replaceable.
4. **Error Resilience**: The system includes robust error handling to manage ML prediction failures gracefully.

### Core ML Components

The ML integration involves these key components:

#### 1. Model Manager

The `ModelManager` is the central coordinator for ML functionality, responsible for:

- Loading and managing models
- Preprocessing features for predictions
- Publishing prediction events to the event bus
- Handling prediction errors and recovery strategies

```python
# Key interactions with ModelManager
model_manager = ModelManager(event_bus, feature_manager)
model_manager.load_model('price_prediction', '/path/to/model')
model_manager.enable_model('price_prediction')
```

#### 2. Feature Registry

The `FeatureRegistry` maintains a centralized registry of all features used by the system, ensuring consistent naming between data providers, transformers, and ML models.

```python
# Using the feature registry
registry = FeatureRegistry('config/features.json')
standard_name = registry.get_standard_name('price')  # Returns 'current_price'
```

#### 3. Feature Transformer

The `FeatureTransformer` prepares raw features for model input, handling:

- Feature scaling and normalization
- Categorical encoding
- Feature selection and grouping

```python
# Transforming features for model input
transformer = FeatureTransformer(feature_manager, registry)
transformer.set_required_features(['current_price', 'volume_5m', 'rsi_14'])
model_input = transformer.prepare_model_input(token_id)
```

#### 4. ML Models

Models implemented as Python classes with a standard interface:

```python
class PricePredictionModel:
    def __init__(self, config):
        # Initialize model
        
    def predict(self, features):
        # Make predictions
        
    def get_required_features(self):
        # Return list of required features
```

## Event Flow

### ML-Related Events

The system defines these ML-related events:

1. **`FEATURE_UPDATE`**: Published when feature values are updated, triggering model predictions.
2. **`MODEL_PREDICTION`**: Published when a model makes a prediction, consumed by the trading engine.
3. **`ML_ERROR`**: Published when an ML component encounters an error.

### Prediction Flow

The complete flow from raw data to trading decisions:

1. Market data is received by data providers
2. Feature providers compute features from raw data
3. `FeatureManager` standardizes and publishes feature updates
4. `ModelManager` receives feature updates via event subscription
5. ML models make predictions using standardized features
6. `ModelManager` publishes prediction events to the event bus
7. Trading components consume prediction events to inform decisions

## Feature Naming Standards

### Feature Registry

The feature registry in `config/features.json` defines the canonical feature names, including:

- Standard name
- Description
- Group categorization
- Aliases for compatibility
- Metadata (units, data types, etc.)

Example feature definition:

```json
{
  "name": "current_price",
  "description": "Current market price of the token",
  "group": "price",
  "aliases": ["price", "latest_price", "market_price"],
  "metadata": {
    "unit": "USD",
    "data_type": "float"
  }
}
```

### Feature Groups

Features are organized into these standard groups:

1. **Price**: Features related to price levels and changes
2. **Volume**: Features related to trading volume
3. **Indicator**: Technical indicators (RSI, MACD, etc.)
4. **Signal**: Trading signals derived from indicators
5. **ML**: Model predictions and confidence scores

### Standardization Benefits

Using standardized feature names provides:

1. Consistent interpretation across components
2. Easier integration of new data sources and models
3. Simplified debugging and monitoring
4. Better system maintainability

## Model Training Workflow

### Training Data Preparation

1. Historical feature data is collected through the `FeatureManager`
2. The `FeatureTransformer` applies consistent standardization
3. Training datasets are created with proper train/validation splits
4. Feature scaling parameters are saved for inference time

### Training Process

1. Models are trained using the standardized training datasets
2. Hyperparameter tuning is performed (optional)
3. Models are evaluated on validation data
4. Final models are exported for deployment

### Model Storage Format

1. Serialized model files (joblib, pickle, ONNX)
2. Model configuration YAML files
3. Feature transformer configuration (scaling parameters)

### Deployment Process

1. Models are deployed to the model registry location
2. The `ModelManager` loads models during system initialization
3. Feature transformers are configured with saved parameters
4. System verifies model compatibility with available features

## Integration Example

Here's an example of using the ML integration in the trading system:

```python
# 1. Initialize components
feature_registry = FeatureRegistry('config/features.json')
feature_manager = DefaultFeatureManager(event_bus, feature_registry)
model_manager = ModelManager(event_bus, feature_manager)

# 2. Register feature providers
market_data_provider = MarketDataFeatureProvider(market_data_client)
feature_manager.register_provider(market_data_provider)

# 3. Load ML models
model_manager.load_model('price_prediction', '/models/price_prediction_v1.pkl')
model_manager.load_model('volatility_prediction', '/models/volatility_v1.pkl')

# 4. Subscribe to ML predictions in trading engine
trading_engine = DefaultTradingEngine(event_bus)
trading_engine.register_event_handler(EventType.MODEL_PREDICTION, trading_engine.on_model_prediction)

# 5. Start the system
event_bus.start()
```

## Error Handling and Recovery

The ML integration includes comprehensive error handling:

1. **Central Error Handler**: The `MLErrorHandler` manages errors from ML components
2. **Prediction Timeouts**: Enforces time limits on prediction operations
3. **Fallback Strategies**:
   - Using the previous valid prediction
   - Using a simple rule-based fallback
   - Skipping the prediction step in the pipeline
4. **Error Reporting**: All ML errors are published to the event bus and logged

## Performance Considerations

Optimizing ML integration performance:

1. **Batched Predictions**: Process multiple tokens in a single prediction call
2. **Feature Caching**: Avoid recomputing features when possible
3. **Model Complexity**: Balance accuracy against prediction latency
4. **Resource Limits**: Set memory and CPU limits for ML operations

## Monitoring and Metrics

The ML integration provides these metrics:

1. **Prediction Counts**: Total predictions made by each model
2. **Prediction Latency**: Time taken for predictions
3. **Error Rates**: Prediction errors by type and model
4. **Feature Availability**: Tracking of missing or invalid features

## Best Practices

### For Feature Providers

1. Always use standardized feature names from the registry
2. Provide accurate documentation for new features
3. Use appropriate data types and units for features
4. Follow error handling protocols for data issues

### For Model Developers

1. Specify all required features using standard names
2. Document model inputs, outputs, and limitations
3. Include validation metrics in model metadata
4. Provide fallback strategies for prediction failures

### For System Integrators

1. Initialize the feature registry before other components
2. Verify feature compatibility before enabling models
3. Implement proper error handling for ML components
4. Monitor prediction performance in production

## Conclusion

This integration guide provides the foundation for effective ML integration in the trading system. By following these standards and best practices, we ensure that ML capabilities enhance the system's trading performance while maintaining reliability and maintainability. 