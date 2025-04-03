#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token Metadata Provider Module

This module provides the TokenMetadataProvider class that computes metadata-related
features for tokens.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.core.data import DataFeedInterface
from src.core.features.providers.base_feature_provider import BaseFeatureProvider

logger = logging.getLogger(__name__)


class TokenMetadataProvider(BaseFeatureProvider):
    """
    Feature provider for token metadata features.
    
    This provider computes features related to token metadata, including
    token information, creator details, and creation-related metrics.
    """
    
    def __init__(self, data_feed: DataFeedInterface):
        """
        Initialize the token metadata provider.
        
        Args:
            data_feed: Data feed to use for retrieving token metadata
        """
        # Define all metadata-related features
        feature_names = [
            'token_name',               # Name of the token
            'token_symbol',             # Symbol of the token
            'token_address',            # Address/ID of the token
            'token_creator',            # Creator address of the token
            'token_creation_time',      # Creation timestamp of the token
            'token_age_seconds',        # Age of the token in seconds
            'token_age_minutes',        # Age of the token in minutes
            'token_age_hours',          # Age of the token in hours
            'token_age_days',           # Age of the token in days
            'has_verified_creator',     # Whether the creator is verified
            'creator_token_count',      # Number of tokens created by this creator
            'creator_average_volume',   # Average volume of tokens by this creator
            'token_description',        # Description of the token
            'token_website',            # Website of the token
            'token_social_links',       # Social media links of the token
            'token_market_cap',         # Market cap of the token
        ]
        
        # Define dependencies
        dependencies = {}  # Metadata features typically don't depend on other features
        
        super().__init__(feature_names, dependencies)
        self.data_feed = data_feed
        
        # Cache for creator data to avoid repeated lookups
        self.creator_cache = {}
    
    def _get_token_metadata(self, token_id: str) -> Dict[str, Any]:
        """
        Get metadata for the specified token.
        
        Args:
            token_id: The ID of the token
            
        Returns:
            Dictionary with token metadata
        """
        try:
            # Try to get token information from the data feed
            if hasattr(self.data_feed, 'get_token_info'):
                token_info = self.data_feed.get_token_info(token_id)
                if token_info:
                    return token_info
            
            # Fallback: try to get token information from the historical data
            historical_data = self.data_feed.get_historical_data(
                token_id=token_id,
                limit=1  # Just need one record to get metadata
            )
            
            if historical_data:
                # Convert to DataFrame if necessary
                if not isinstance(historical_data, pd.DataFrame):
                    df = pd.DataFrame(historical_data)
                else:
                    df = historical_data
                
                # Extract token metadata
                if not df.empty:
                    row = df.iloc[0]
                    metadata = {}
                    
                    # Map common column names to metadata fields
                    column_mapping = {
                        'token_id': 'token_address',
                        'token_name': 'token_name',
                        'name': 'token_name',
                        'symbol': 'token_symbol',
                        'creator_address': 'token_creator',
                        'created_at': 'token_creation_time',
                        'description': 'token_description',
                        'website': 'token_website',
                        'social_links': 'token_social_links',
                        'market_cap': 'token_market_cap',
                        'marketCapSol': 'token_market_cap',
                    }
                    
                    # Extract available metadata
                    for col, meta_field in column_mapping.items():
                        if col in row:
                            metadata[meta_field] = row[col]
                    
                    return metadata
            
            # If no metadata found, return minimal information
            return {
                'token_address': token_id,
                'token_name': 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting token metadata for {token_id}: {e}")
            return {
                'token_address': token_id,
                'token_name': 'Unknown',
                'error': str(e)
            }
    
    def compute_feature(self, feature_name: str, token_id: str, data: Dict[str, Any]) -> Any:
        """
        Compute the specified metadata feature for the given token.
        
        Args:
            feature_name: The name of the feature to compute
            token_id: The ID of the token
            data: Dictionary containing required data for computation
            
        Returns:
            The computed feature value
        """
        try:
            # Check if feature is provided by this provider
            if feature_name not in self.feature_names:
                logger.warning(f"Feature '{feature_name}' is not provided by TokenMetadataProvider")
                return None
            
            # Get token metadata
            token_metadata = data.get('token_metadata')
            
            # If metadata not provided in data, fetch it
            if not token_metadata:
                token_metadata = self._get_token_metadata(token_id)
                data['token_metadata'] = token_metadata  # Cache for future use
            
            # Basic token information features
            if feature_name == 'token_name':
                return token_metadata.get('token_name', 'Unknown')
                
            elif feature_name == 'token_symbol':
                return token_metadata.get('token_symbol', '')
                
            elif feature_name == 'token_address':
                return token_metadata.get('token_address', token_id)
                
            elif feature_name == 'token_creator':
                return token_metadata.get('token_creator', '')
                
            elif feature_name == 'token_description':
                return token_metadata.get('token_description', '')
                
            elif feature_name == 'token_website':
                return token_metadata.get('token_website', '')
                
            elif feature_name == 'token_social_links':
                return token_metadata.get('token_social_links', {})
                
            elif feature_name == 'token_market_cap':
                return token_metadata.get('token_market_cap', 0)
                
            # Time-related features
            elif feature_name == 'token_creation_time':
                creation_time = token_metadata.get('token_creation_time')
                if creation_time:
                    # Convert to epoch timestamp if it's a datetime object
                    if isinstance(creation_time, datetime):
                        return creation_time.timestamp()
                    return creation_time
                return None
                
            elif feature_name.startswith('token_age_'):
                creation_time = token_metadata.get('token_creation_time')
                if not creation_time:
                    return None
                    
                # Convert to datetime if it's a timestamp
                if isinstance(creation_time, (int, float)):
                    creation_datetime = datetime.fromtimestamp(creation_time)
                elif isinstance(creation_time, str):
                    try:
                        creation_datetime = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                    except:
                        try:
                            creation_datetime = datetime.strptime(creation_time, '%Y-%m-%dT%H:%M:%S.%fZ')
                        except:
                            logger.warning(f"Could not parse creation time: {creation_time}")
                            return None
                else:
                    creation_datetime = creation_time
                
                # Calculate age
                now = datetime.now()
                age_seconds = (now - creation_datetime).total_seconds()
                
                if feature_name == 'token_age_seconds':
                    return age_seconds
                elif feature_name == 'token_age_minutes':
                    return age_seconds / 60
                elif feature_name == 'token_age_hours':
                    return age_seconds / 3600
                elif feature_name == 'token_age_days':
                    return age_seconds / 86400
                
            # Creator-related features
            elif feature_name == 'has_verified_creator':
                creator = token_metadata.get('token_creator', '')
                if not creator:
                    return False
                
                # This would require integration with a verification service
                # For now, return a placeholder value
                return False  # Default to not verified
                
            elif feature_name == 'creator_token_count':
                creator = token_metadata.get('token_creator', '')
                if not creator:
                    return 0
                
                # Check cache first
                if creator in self.creator_cache and 'token_count' in self.creator_cache[creator]:
                    return self.creator_cache[creator]['token_count']
                
                # Query database or other data source
                try:
                    if hasattr(self.data_feed, 'execute_query'):
                        query = f"SELECT COUNT(*) as token_count FROM tokens WHERE creator_address = '{creator}'"
                        result = self.data_feed.execute_query(query)
                        if result and len(result) > 0:
                            token_count = result[0].get('token_count', 0)
                            
                            # Cache the result
                            if creator not in self.creator_cache:
                                self.creator_cache[creator] = {}
                            self.creator_cache[creator]['token_count'] = token_count
                            
                            return token_count
                except Exception as e:
                    logger.error(f"Error querying creator token count: {e}")
                
                return 1  # Default to 1 (at least this token)
                
            elif feature_name == 'creator_average_volume':
                creator = token_metadata.get('token_creator', '')
                if not creator:
                    return 0
                
                # Check cache first
                if creator in self.creator_cache and 'avg_volume' in self.creator_cache[creator]:
                    return self.creator_cache[creator]['avg_volume']
                
                # Query database or other data source
                try:
                    if hasattr(self.data_feed, 'execute_query'):
                        query = f"""
                        SELECT AVG(t.volume_24h) as avg_volume
                        FROM tokens as t
                        WHERE t.creator_address = '{creator}'
                        """
                        result = self.data_feed.execute_query(query)
                        if result and len(result) > 0:
                            avg_volume = result[0].get('avg_volume', 0)
                            
                            # Cache the result
                            if creator not in self.creator_cache:
                                self.creator_cache[creator] = {}
                            self.creator_cache[creator]['avg_volume'] = avg_volume
                            
                            return avg_volume
                except Exception as e:
                    logger.error(f"Error querying creator average volume: {e}")
                
                return 0  # Default to 0
                
            else:
                logger.warning(f"Unknown metadata feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error computing metadata feature '{feature_name}' for token {token_id}: {e}")
            return None
    
    def get_required_data_types(self, feature_name: str) -> List[str]:
        """
        Get the required data types for computing the specified feature.
        
        Args:
            feature_name: The name of the feature
            
        Returns:
            List of required data type names
        """
        # Most metadata features need token_metadata
        return ["token_metadata"] 