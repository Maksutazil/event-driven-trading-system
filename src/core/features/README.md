# Feature System

The Feature System is a core component of the event-driven trading platform that allows computing, managing, and caching features used for trading decisions. Features represent calculated metrics and indicators derived from market data that can be used to generate trading signals.

## Overview

The Feature System consists of the following core components:

- **FeatureSystem**: Central manager for features and providers that handles feature computation with dependency resolution and caching
- **Feature**: Interface for individual features that can be computed
- **FeatureProvider**: Interface for components that supply base features from data sources
- **FeatureCache**: Caching mechanism for efficient feature reuse

## Usage

### Basic Usage

```python
from src.core.events import EventBus
from src.core.features import FeatureSystem, PriceFeatureProvider, PriceMomentumSignalFeature

# Create an event bus for feature updates
event_bus = EventBus()

# Create the feature system
feature_system = FeatureSystem(event_bus=event_bus)

# Register feature providers
price_provider = PriceFeatureProvider()
feature_system.register_provider(price_provider)

# Register features
momentum_signal = PriceMomentumSignalFeature(threshold=2.0, sensitivity=1.5)
feature_system.register_feature(momentum_signal)

# Compute features for a token
context = {
    'token_id': 'TOKEN_XYZ', 
    'timestamp': datetime.now()
}
features = feature_system.compute_features(context)

# Use the computed features
signal_value = features.get('price_momentum_signal')
if signal_value > 0.5:
    print("Strong buy signal detected!")
```

### Creating a Custom Feature

To create a custom feature, inherit from the `Feature` interface and implement the required methods:

```python
from src.core.features import Feature

class MyCustomFeature(Feature):
    """A custom feature that calculates something useful for trading."""
    
    def __init__(self, parameter1=1.0, parameter2=2.0):
        """Initialize with custom parameters."""
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    @property
    def name(self) -> str:
        """Get the name of this feature."""
        return 'my_custom_feature'
    
    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of this feature."""
        return ['price', 'volume', 'another_feature']
    
    def compute(self, context: Dict[str, Any]) -> Any:
        """
        Compute the feature value based on the context.
        
        Args:
            context: Dictionary containing required input data and dependencies
            
        Returns:
            Computed feature value
        """
        # Extract dependencies from context
        price = context.get('price', 0.0)
        volume = context.get('volume', 0.0)
        another_value = context.get('another_feature', 0.0)
        
        # Perform calculations
        result = (price * self.parameter1 + volume * self.parameter2) / max(1.0, another_value)
        
        return result
```

### Creating a Custom Feature Provider

To create a custom feature provider, inherit from `BaseFeatureProvider` and implement the required methods:

```python
from src.core.features.providers import BaseFeatureProvider

class MyDataProvider(BaseFeatureProvider):
    """Provider that supplies features from an external data source."""
    
    def __init__(self, name='my_data_provider'):
        """Initialize the provider."""
        super().__init__(name=name)
        
        # Register the features this provider can compute
        self._provided_features.update({
            'external_metric_1',
            'external_metric_2',
            'market_sentiment'
        })
    
    def get_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all available features for the given context.
        
        Args:
            context: Dictionary containing required input data
            
        Returns:
            Dictionary of feature names to computed values
        """
        token_id = context.get('token_id')
        if not token_id:
            return {}
        
        # In a real implementation, this might retrieve data from an API
        # or database, but for example purposes we'll use mock data
        
        result = {
            'external_metric_1': 42.0,
            'external_metric_2': 123.5,
            'market_sentiment': 0.75
        }
        
        return result
```

## Event Integration

The feature system integrates with the event system by publishing `FEATURE_UPDATE` events when features are computed. To listen for feature updates:

```python
from src.core.events import EventType

def on_feature_update(event):
    if event.event_type == EventType.FEATURE_UPDATE:
        data = event.data
        feature_name = data['feature_name']
        feature_value = data['value']
        token_id = data['token_id']
        
        print(f"Feature update: {feature_name} = {feature_value} for {token_id}")

# Subscribe to feature updates
event_bus.subscribe(EventType.FEATURE_UPDATE, on_feature_update)
```

## Caching

Features are automatically cached to improve performance. The cache operates based on `token_id` and feature name, with support for time-based invalidation. Cache statistics can be retrieved:

```python
cache_stats = feature_system.cache.get_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Cache misses: {cache_stats['misses']}")
print(f"Cache hit rate: {cache_stats['hit_rate']:.2f}")
```

## Best Practices

1. **Dependency Management**: Carefully define dependencies between features to avoid circular dependencies
2. **Performance Optimization**: Use caching for features that are expensive to compute
3. **Error Handling**: Implement proper error handling in feature computation
4. **Testing**: Unit test features with mock data to ensure correct calculations
5. **Documentation**: Document what each feature represents and how it's calculated

For a working example, see `examples/feature_system_example.py`. 