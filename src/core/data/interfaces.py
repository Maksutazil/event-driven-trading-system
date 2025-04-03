#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Module Interfaces

This module defines interfaces for data components.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class DataFeedInterface:
    """Interface for data feed implementations."""
    
    def get_historical_data(
        self, 
        token_id: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical data for a token.
        
        Args:
            token_id: The token ID
            start_time: Start time for the data (optional)
            end_time: End time for the data (optional)
            
        Returns:
            List of trade records or empty list if no data
        """
        raise NotImplementedError("Subclasses must implement get_historical_data") 