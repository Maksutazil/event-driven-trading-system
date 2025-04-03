"""
Data module for handling data feeds and sources.

This module provides clients for connecting to data sources and
managing data feeds.
"""

from .data_feed_manager import DataFeedManager
from .websocket_clients import RealWebSocketClient, MockWebSocketClient
from .interfaces import DataFeedInterface

__all__ = ['DataFeedManager', 'RealWebSocketClient', 'MockWebSocketClient', 'DataFeedInterface'] 