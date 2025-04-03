# Machine Learning Module

This directory contains the machine learning components of the event-driven trading system. The module is designed to provide a flexible and extensible framework for incorporating machine learning models into the trading system.

## Architecture

The ML module is built around the following core components:

1. **Interfaces** - Abstract base classes that define the expected behavior of models, transformers, and the model manager.
2. **Model Adapters** - Classes that adapt different types of machine learning models to the system's unified interface.
3. **Feature Transformers** - Components that transform raw features into a format suitable for model input.
4. **ModelManager** - The central component that manages models, transformers, and predictions.

## Components

### Interfaces

The module defines three main interfaces:

- **Model** - Defines the interface for working with machine learning models (prediction, updating, metadata).
- **FeatureTransformer** - Defines the interface for transforming raw features for model consumption.
- **ModelManager** - Defines the interface for the component that manages models and transformers.

### Model Adapters

Adapters allow different types of machine learning models to be used with the system:

- **ScikitLearnModel** - Adapts scikit-learn models to the system's Model interface.

### Feature Transformers

Transformers prepare raw features for use with machine learning models:

- **StandardFeatureTransformer** - A general-purpose transformer that handles feature selection, standardization, missing values, and encoding of categorical features.

### Model Manager

The model manager is the central component that orchestrates all machine learning operations:

- **DefaultModelManager** - The default implementation of the ModelManager interface.

## Error Handling

The module uses custom exceptions to provide better error handling and diagnostics:

- **MLModuleError** - Base exception for all ML module errors
- **ModelError** - Base exception for model-related errors
  - **ModelNotFoundError** - Raised when a model is not found
  - **ModelLoadError** - Raised when loading a model fails
  - **ModelSaveError** - Raised when saving a model fails
  - **ModelUpdateError** - Raised when updating a model fails
  - **ModelPredictionError** - Raised when making a prediction fails
- **TransformerError** - Base exception for transformer-related errors
  - **TransformerNotFoundError** - Raised when a transformer is not found
  - **TransformerFitError** - Raised when fitting a transformer fails
  - **TransformerTransformError** - Raised when transforming features fails
- **InvalidFeatureError** - Base exception for feature-related errors
  - **MissingFeatureError** - Raised when required features are missing
- **InvalidModelTypeError** - Raised when an invalid model type is specified

These exceptions help provide meaningful error messages and allow for more specific error handling.

## Usage Examples

### Loading a Model

```python
from src.core.ml.model_manager import DefaultModelManager
from src.core.ml.exceptions import ModelLoadError

# Create model manager
model_manager = DefaultModelManager(
    model_directory="/path/to/models",
    feature_system=feature_system,
    event_bus=event_bus
)

# Load a model
try:
    model_id = "price_prediction"
    model_path = "/path/to/models/price_prediction.joblib"
    model_manager.load_model(model_id, model_path, "regression")
except ModelLoadError as e:
    print(f"Failed to load model: {str(e)}")
```

### Creating and Registering a Transformer

```python
from src.core.ml.transformers.standard import StandardFeatureTransformer
from src.core.ml.exceptions import TransformerFitError

# Create a transformer
transformer = StandardFeatureTransformer(
    features=["return", "ma5_diff", "volatility"],
    standardize=True
)

# Fit the transformer
try:
    transformer.fit(feature_dicts)
except TransformerFitError as e:
    print(f"Failed to fit transformer: {str(e)}")

# Register the transformer with a model
model_manager.register_transformer("price_prediction", transformer)
```

### Making Predictions

```python
from src.core.ml.exceptions import ModelNotFoundError, ModelPredictionError

# Get a prediction for a token
token_id = "BTC-USD"
try:
    prediction = model_manager.get_prediction("price_prediction", token_id)
except ModelNotFoundError as e:
    print(f"Model not found: {str(e)}")
except ModelPredictionError as e:
    print(f"Prediction error: {str(e)}")

# The prediction is automatically based on the latest features for the token
# from the feature system, transformed by the registered transformer
```

### Updating Models

```python
from src.core.ml.exceptions import ModelUpdateError

# Update a model with new data
try:
    model_manager.update_model("price_prediction", token_id, actual_value)
except ModelUpdateError as e:
    print(f"Failed to update model: {str(e)}")
```

## Performance Tracking

The model manager automatically tracks performance metrics for each model:

- For regression models: Mean Absolute Error (MAE), Mean Squared Error (MSE), R-squared (R²)
- For classification models: Accuracy, Precision, Recall, F1 Score

Performance metrics can be retrieved with:

```python
performance = model_manager.get_model_performance("model_id")
```

## Integration with Event System

The model manager integrates with the event system to:

1. Respond to relevant events that might trigger model updates or predictions.
2. Publish events when models are updated or predictions are made.

## Threading and Performance

The model manager is designed to be thread-safe and can handle concurrent requests for predictions or updates. It uses locks to ensure that critical operations are properly synchronized. 