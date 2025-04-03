# Early Pump Detection System

This document describes the Early Pump Detection system, which is designed to detect potential pump events in newly created tokens with minimal trading history.

## Overview

The Early Pump Detection system is optimized for detecting pump events in the earliest stages of a token's lifecycle, when only minimal trading data is available. Unlike traditional pump detection systems that require substantial historical data, this system can work with just a few trades.

## Components

The system consists of three main components:

1. **EarlyPumpDetectionProvider**: Provides features that can be calculated with minimal data
2. **EarlyPumpPredictor**: Makes predictions about potential pump events with minimal history
3. **EarlyPumpStrategy**: Generates trading signals based on early pump predictions

## Features

The `EarlyPumpDetectionProvider` calculates these key features:

- **immediate_price_change**: Percentage price change over the available trading history
- **trade_frequency**: Number of trades per minute
- **buyer_dominance**: Ratio of buys to total trades
- **volume_intensity**: Volume normalized by token age
- **early_pump_score**: Combined score indicating the likelihood of an early pump

These features are specifically designed to work with as few as 2-3 trades, making them suitable for newly created tokens.

## Prediction Model

The `EarlyPumpPredictor` uses a rule-based approach to classify tokens into three categories:

- **NORMAL**: No pump activity detected
- **EARLY_PUMP**: Potential early-stage pump
- **STRONG_PUMP_SIGNAL**: High-confidence pump signal

Each prediction includes confidence scores and supporting evidence.

## Trading Strategy

The `EarlyPumpStrategy` generates trading signals based on the pump predictions:

- Generates OPEN signals for EARLY_PUMP and STRONG_PUMP_SIGNAL predictions
- Uses higher confidence thresholds for EARLY_PUMP signals
- Includes supporting evidence in the signal data

## Usage Example

```python
# Create and register the provider
early_pump_provider = EarlyPumpDetectionProvider(data_feed)
feature_manager.register_provider(early_pump_provider)

# Create the predictor
early_pump_predictor = EarlyPumpPredictor(
    model_id="early_pump_predictor_v1",
    feature_manager=feature_manager
)

# Create the strategy
early_pump_strategy = EarlyPumpStrategy(
    feature_manager=feature_manager,
    pump_predictor=early_pump_predictor,
    confidence_threshold=0.6
)

# Register the strategy with a signal generator
signal_generator.register_strategy(early_pump_strategy)
```

See the complete example in `examples/early_pump_detection_example.py`.

## Key Benefits

- Works with extremely limited trading history (as few as 2-3 trades)
- Optimized for newly created tokens
- Fast response to emerging patterns
- Simple and efficient implementation

## Limitations

- Less accurate than systems with more historical data
- May generate false positives in very volatile markets
- Should be combined with risk management to limit exposure

## Customization

The system can be customized by adjusting these parameters:

- **confidence_threshold**: Controls sensitivity (default: 0.6)
- **cooldown_seconds**: Controls signal frequency (default: 60)
- **max_active_tokens**: Limits the number of monitored tokens (default: 10)

## Integration with Event-Driven Architecture

The early pump detection system fits smoothly into the event-driven architecture:

1. Token trade events trigger feature calculation
2. Features are fed to the prediction model
3. Predictions are evaluated for signal generation
4. Trading signals are emitted as events

This allows for real-time monitoring and rapid response to emerging pump patterns. 