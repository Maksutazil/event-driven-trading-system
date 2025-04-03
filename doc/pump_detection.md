# Pump and Dump Detection Module

This document describes the pump and dump detection module in the event-driven trading system, which provides features and models to detect potential pump and dump events in token trading.

## Overview

Pump and dump schemes are manipulative practices where the price of a token is artificially inflated (pumped) through misleading recommendations and then sold (dumped) when other investors buy in at the inflated price. This module helps identify such patterns by analyzing trading behavior and price movements.

## Components

The pump and dump detection module consists of two main components:

1. **PumpDetectionFeatureProvider**: Calculates features that help identify pump and dump patterns
2. **PumpPredictorModel**: Uses these features to predict the current state and phase of a token

## Features

The `PumpDetectionFeatureProvider` provides the following features:

### Price Dynamics Features
- `price_velocity`: Rate of price change (first derivative)
- `price_acceleration`: Change in rate of price change (second derivative)

### Volume Features
- `volume_surge_ratio`: Ratio of current volume to historical average
- `volume_volatility`: Volatility of trading volume
- `buy_sell_volume_ratio`: Ratio of buy to sell volume

### Order Book Features
- `order_imbalance`: Imbalance between buy and sell orders
- `std_rush_order`: Standard deviation of rush orders
- `avg_rush_order`: Average rush order size

### Price Anomaly Features
- `price_deviation`: Deviation from moving average
- `price_volatility_short`: Short-term price volatility
- `price_volatility_ratio`: Ratio of short-term to long-term volatility

### Pattern Detection Features
- `pump_pattern_score`: Score indicating presence of pump pattern
- `dump_pattern_score`: Score indicating presence of dump pattern
- `pump_phase_detection`: Detected phase of pump (0-4, where 0=none, 4=dump)

### Time-based Features
- `minute_since_volume_peak`: Minutes since volume peaked
- `minute_since_price_peak`: Minutes since price peaked

### Combined Metrics
- `abnormal_activity_score`: Overall score of abnormal trading activity

## Model Predictions

The `PumpPredictorModel` provides predictions with the following information:

- `class_id`: Numeric class ID (0=normal, 1=pump, 2=dump, 3=peak/distribution)
- `class_label`: Text label for the class (NORMAL, PUMP, DUMP, PEAK)
- `probabilities`: Probabilities for each possible class
- `phase`: Detailed pump/dump phase (0-4)
- `confidence`: Confidence score for the prediction

## Phases of Pump and Dump Schemes

The model recognizes several phases in a pump and dump cycle:

1. **Phase 0 - Normal**: No pump or dump activity detected
2. **Phase 1 - Early Accumulation**: Initial accumulation phase with subtle price increases
3. **Phase 2 - Pump in Progress**: Active pump phase with rapid price increases
4. **Phase 3 - Peak/Distribution**: Price peaks and early distribution begins
5. **Phase 4 - Dump in Progress**: Active dump phase with rapid price decreases

## Integration

### Adding the Feature Provider

```python
from src.core.features.providers.pump_detection_provider import PumpDetectionFeatureProvider

# Create and register the provider
pump_detection_provider = PumpDetectionFeatureProvider(data_feed)
feature_manager.register_provider(pump_detection_provider)
```

### Creating the Model

```python
from src.core.ml.models.pump_predictor import PumpPredictorModel

# Create the pump predictor model
pump_predictor = PumpPredictorModel(
    model_id="pump_predictor_v1",
    feature_manager=feature_manager
)
```

### Getting Features

```python
# Context for feature computation
context = {
    'token_id': token_id,
    'timestamp': datetime.now().timestamp()
}

# Get pump pattern score
pump_score = feature_manager.compute_feature(
    token_id, 
    'pump_pattern_score', 
    context
)

# Get all pump detection features
features = {}
for feature_name in pump_predictor.get_required_features():
    features[feature_name] = feature_manager.compute_feature(
        token_id, 
        feature_name, 
        context
    )
```

### Making Predictions

```python
# Make prediction with the model
prediction = pump_predictor.predict(features)

# Extract prediction information
state = prediction['class_label']  # NORMAL, PUMP, DUMP, or PEAK
phase = prediction['phase']        # 0-4
confidence = prediction['confidence']
probabilities = prediction['probabilities']

print(f"Token state: {state} (Phase {phase}) with {confidence:.2f} confidence")
```

## Example Usage

See the `examples/pump_detection_example.py` file for a complete example of using the pump detection module, including:

1. Setting up the feature provider and model
2. Analyzing tokens for pump and dump patterns
3. Simulating a complete pump and dump scenario for testing
4. Detailed analysis and visualization of results

## Performance Considerations

- The features require historical trade data, which may require database access
- For optimal performance, calculated features can be cached
- Consider using a sliding window approach for continuous monitoring

## References

- For academic background on pump and dump detection, see papers by La Morgia et al., 2020 and Chadalapaka et al., 2022
- Our features are inspired by research on crypto pump and dump detection using deep learning techniques 