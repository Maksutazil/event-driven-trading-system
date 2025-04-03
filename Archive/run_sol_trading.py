import time
from event_bus import Event, EventType

class MockWebSocketClient:
    async def _publish_token_creation(self, token):
        """
        Publish a token creation event on the event bus.
        
        Args:
            token: Token data dictionary
        """
        event_data = {
            'mint': token['mint'],  # Keep mint for backward compatibility
            'token_id': token['mint'],  # Add token_id as the standardized field
            'symbol': token['symbol'],
            'creator': token.get('creator', 'unknown'),
            'signature': token.get('signature', 'mock_signature'),
            'timestamp': time.time()
        }
        
        # Add extra data if available
        for key in ['metadata', 'price', 'volume', 'liquidity']:
            if key in token:
                event_data[key] = token[key]
        
        await self.event_bus.publish(Event(
            event_type=EventType.TOKEN_CREATED,
            data=event_data,
            source="MockWebSocketClient"
        ))
        
        self.logger.info(f"Published TOKEN_CREATED event for {token['symbol']} ({token['mint']})") 