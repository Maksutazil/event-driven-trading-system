# Event System Architecture

The event system is the heart of the trading application, providing a pub/sub architecture that allows components to communicate asynchronously. This document explains the design, components, and usage of the event system.

## Core Components

### Event Class

The `Event` class represents a discrete event in the system. Each event has:

- `event_type`: Categorizes the event (e.g., trade, market data, system)
- `data`: Dictionary containing the event payload
- `timestamp`: When the event was created
- `event_id`: Unique identifier
- `priority`: Importance level of the event
- `source`: Which component generated the event
- `tags`: Optional categorization tags

Events are compared based on priority (higher priority first) and then timestamp (older first).

### EventType Enum

The `EventType` enum defines all possible event categories:

```python
class EventType(Enum):
    # General events
    SYSTEM = auto()
    
    # Data-related events
    DATA_NEW = auto()
    DATA_UPDATE = auto()
    DATA_ERROR = auto()
    
    # Token-related events
    TOKEN_NEW = auto()
    TOKEN_TRADE = auto()
    TOKEN_UPDATE = auto()
    TOKEN_ERROR = auto()
    
    # Feature-related events
    FEATURE_UPDATE = auto()
    FEATURE_ERROR = auto()
    
    # Trading-related events
    TRADE_SIGNAL = auto()
    TRADE_ENTRY = auto()
    TRADE_EXIT = auto()
    POSITION_UPDATE = auto()
    
    # Socket-related events
    SOCKET_CONNECT = auto()
    SOCKET_DISCONNECT = auto()
    SOCKET_MESSAGE = auto()
    SOCKET_ERROR = auto()
    
    # Database-related events
    DB_CONNECT = auto()
    DB_DISCONNECT = auto()
    DB_QUERY = auto()
    DB_ERROR = auto()
```

### EventBus Class

The `EventBus` is the central hub that manages event distribution. It maintains subscriber lists for each event type and handles the publishing and distribution of events. Features include:

- Asynchronous event processing using a priority queue
- Support for both callback functions and subscriber objects
- Filtering of events by type
- Event prioritization
- Thread-safe event handling

## Interfaces

### EventSubscriber

The `EventSubscriber` interface represents components that can receive events:

```python
class EventSubscriber(ABC):
    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Handle an event."""
        pass
        
    def can_handle_event(self, event: Event) -> bool:
        """Determine if this subscriber can handle the event."""
        return True
```

### EventPublisher

The `EventPublisher` interface represents components that can publish events:

```python
class EventPublisher(ABC):
    @abstractmethod
    def publish_event(self, event_type: EventType, data: Dict[str, Any], 
                     priority: Optional[EventPriority] = None, 
                     source: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> None:
        """Publish an event to the event bus."""
        pass
```

## Base Implementations

### BaseEventPublisher

The `BaseEventPublisher` provides a base implementation of the `EventPublisher` interface. It:

- Maintains a reference to the event bus
- Implements the `publish_event` method to forward events to the bus
- Stores and uses a source identifier for published events

### BaseEventSubscriber

The `BaseEventSubscriber` provides a base implementation of the `EventSubscriber` interface. It:

- Maintains a reference to the event bus
- Handles subscription and unsubscription to event types
- Provides a template method pattern for handling different event types

### EventDispatcher

The `EventDispatcher` combines both `BaseEventPublisher` and `BaseEventSubscriber`, allowing a component to both receive and publish events. This is useful for components that transform or process events.

## Usage Examples

### Creating and Publishing Events

```python
from src.core.events import EventBus, EventType

# Create event bus
event_bus = EventBus()
event_bus.start()

# Publish an event
event_bus.publish_event(
    event_type=EventType.TRADE_SIGNAL,
    data={"token_id": "ABC123", "action": "BUY", "price": 100.0},
    source="trading_engine"
)
```

### Subscribing to Events

```python
from src.core.events import EventBus, EventType, BaseEventSubscriber, Event

class TradeHandler(BaseEventSubscriber):
    def __init__(self, event_bus):
        super().__init__(event_bus, {EventType.TRADE_SIGNAL})
        
    def handle_trade_signal(self, event: Event):
        data = event.data
        print(f"Trade signal: {data['action']} {data['token_id']} at {data['price']}")

# Create handler and subscribe
handler = TradeHandler(event_bus)
```

### Using Callback Functions

```python
def handle_system_event(event: Event):
    print(f"System event: {event.data}")

# Register callback
event_bus.add_callback(EventType.SYSTEM, handle_system_event)
```

## Best Practices

1. Use appropriate event types for different domains
2. Keep event data lightweight and serializable
3. Handle event processing failures gracefully
4. Set appropriate priorities for time-sensitive events
5. Use tags for additional filtering when needed
6. Always unsubscribe components when they're no longer needed
7. Avoid circular dependencies between event handlers