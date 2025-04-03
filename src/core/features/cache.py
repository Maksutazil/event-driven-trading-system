#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Cache Module

This module provides a caching mechanism for storing and retrieving feature values.
The FeatureCache improves performance by preventing redundant computations.
"""

import logging
import time
from typing import Dict, Optional, Any, Tuple, DefaultDict, List
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheEntry:
    """
    A cache entry storing a feature value with metadata.
    """
    
    def __init__(self, value: Any, timestamp: datetime = None) -> None:
        """
        Initialize a new cache entry.
        
        Args:
            value: The feature value to cache
            timestamp: Optional timestamp of when the value was computed
        """
        self.value = value
        self.timestamp = timestamp or datetime.now()
        self.last_accessed = time.time()
        self.access_count = 0
    
    def access(self) -> None:
        """Record an access to this cache entry."""
        self.last_accessed = time.time()
        self.access_count += 1


class FeatureCache:
    """
    Cache for storing computed feature values.
    
    The cache operates with token_id as the primary key, and feature_name 
    as the secondary key. It supports time-based invalidation and LRU eviction.
    """
    
    def __init__(
        self, 
        max_entries: int = 10000,
        max_age: Optional[timedelta] = timedelta(minutes=30)
    ) -> None:
        """
        Initialize a new feature cache.
        
        Args:
            max_entries: Maximum number of entries in the cache
            max_age: Maximum age of cached entries before they're considered stale
        """
        self._cache: DefaultDict[str, Dict[str, CacheEntry]] = defaultdict(dict)
        self._max_entries = max_entries
        self._max_age = max_age
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Initialized FeatureCache with max_entries={max_entries}, max_age={max_age}")
    
    def set(self, token_id: str, feature_name: str, value: Any, timestamp: datetime = None) -> None:
        """
        Store a feature value in the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            value: Value to cache
            timestamp: Optional timestamp of when the value was computed
        """
        # Check if we need to evict entries
        total_entries = sum(len(features) for features in self._cache.values())
        if total_entries >= self._max_entries:
            self._evict_entries()
        
        # Store the new entry
        self._cache[token_id][feature_name] = CacheEntry(value, timestamp)
        logger.debug(f"Cached feature {feature_name} for token {token_id}")
    
    def get(self, token_id: str, feature_name: str) -> Optional[Any]:
        """
        Retrieve a feature value from the cache.
        
        Args:
            token_id: ID of the token
            feature_name: Name of the feature
            
        Returns:
            The cached value, or None if not found or expired
        """
        # Return None if token or feature not in cache
        if token_id not in self._cache or feature_name not in self._cache[token_id]:
            self._misses += 1
            return None
        
        # Get the cache entry
        entry = self._cache[token_id][feature_name]
        
        # Check if entry is expired
        if self._max_age is not None:
            age = datetime.now() - entry.timestamp
            if age > self._max_age:
                # Remove expired entry
                del self._cache[token_id][feature_name]
                if not self._cache[token_id]:  # Clean up empty dicts
                    del self._cache[token_id]
                
                self._misses += 1
                logger.debug(f"Cache miss (expired) for {feature_name} of token {token_id}")
                return None
        
        # Entry found and not expired
        entry.access()
        self._hits += 1
        logger.debug(f"Cache hit for {feature_name} of token {token_id}")
        return entry.value
    
    def invalidate(self, token_id: str = None, feature_name: str = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            token_id: Optional token ID to invalidate entries for
            feature_name: Optional feature name to invalidate entries for
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        if token_id is None and feature_name is None:
            # Invalidate all entries
            count = sum(len(features) for features in self._cache.values())
            self._cache.clear()
            logger.info(f"Invalidated entire cache ({count} entries)")
            
        elif token_id is not None and feature_name is None:
            # Invalidate all entries for a specific token
            if token_id in self._cache:
                count = len(self._cache[token_id])
                del self._cache[token_id]
                logger.info(f"Invalidated all features for token {token_id} ({count} entries)")
                
        elif token_id is None and feature_name is not None:
            # Invalidate a specific feature for all tokens
            for token, features in list(self._cache.items()):
                if feature_name in features:
                    del features[feature_name]
                    count += 1
                    
                    # Clean up empty dicts
                    if not features:
                        del self._cache[token]
            
            logger.info(f"Invalidated feature {feature_name} for all tokens ({count} entries)")
            
        else:
            # Invalidate a specific feature for a specific token
            if token_id in self._cache and feature_name in self._cache[token_id]:
                del self._cache[token_id][feature_name]
                count = 1
                
                # Clean up empty dicts
                if not self._cache[token_id]:
                    del self._cache[token_id]
                
                logger.info(f"Invalidated feature {feature_name} for token {token_id}")
        
        return count
    
    def _evict_entries(self, count: int = None) -> int:
        """
        Evict entries from the cache using LRU policy.
        
        Args:
            count: Number of entries to evict, defaults to 10% of max_entries
            
        Returns:
            Number of entries evicted
        """
        if count is None:
            count = max(1, self._max_entries // 10)
        
        evicted = 0
        all_entries = []
        
        # Collect all entries with their token_id and feature_name
        for token_id, features in self._cache.items():
            for feature_name, entry in features.items():
                all_entries.append((token_id, feature_name, entry))
        
        # Sort by last accessed time (oldest first)
        all_entries.sort(key=lambda x: x[2].last_accessed)
        
        # Evict the oldest entries
        for token_id, feature_name, _ in all_entries[:count]:
            if token_id in self._cache and feature_name in self._cache[token_id]:
                del self._cache[token_id][feature_name]
                evicted += 1
                
                # Clean up empty dicts
                if not self._cache[token_id]:
                    del self._cache[token_id]
        
        self._evictions += evicted
        logger.debug(f"Evicted {evicted} cache entries")
        return evicted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_entries = sum(len(features) for features in self._cache.values())
        token_count = len(self._cache)
        
        if self._hits + self._misses > 0:
            hit_rate = self._hits / (self._hits + self._misses)
        else:
            hit_rate = 0.0
        
        return {
            'entries': total_entries,
            'tokens': token_count,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': hit_rate,
            'capacity': self._max_entries,
            'utilization': total_entries / self._max_entries if self._max_entries > 0 else 0.0
        }


class InMemoryFeatureCache(FeatureCache):
    """
    In-memory implementation of FeatureCache.
    
    This class extends the base FeatureCache with methods specific to in-memory caching,
    such as getting all features for a token and clearing the cache.
    """
    
    def __init__(
        self, 
        max_entries: int = 10000,
        max_age: Optional[timedelta] = timedelta(minutes=30)
    ) -> None:
        """
        Initialize a new in-memory feature cache.
        
        Args:
            max_entries: Maximum number of entries in the cache
            max_age: Maximum age of cached entries before they're considered stale
        """
        super().__init__(max_entries, max_age)
        logger.info("Initialized InMemoryFeatureCache")
    
    def get_all_features(self, token_id: str) -> Dict[str, Any]:
        """
        Get all cached features for a token.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Dictionary mapping feature names to values
        """
        result = {}
        
        if token_id not in self._cache:
            return result
        
        # Collect all non-expired features
        now = datetime.now()
        for feature_name, entry in list(self._cache[token_id].items()):
            # Check if entry is expired
            if self._max_age is not None:
                age = now - entry.timestamp
                if age > self._max_age:
                    # Remove expired entry
                    del self._cache[token_id][feature_name]
                    continue
            
            # Add to result and mark as accessed
            entry.access()
            result[feature_name] = entry.value
            self._hits += 1
        
        # Clean up empty dicts
        if not self._cache[token_id]:
            del self._cache[token_id]
        
        return result
    
    def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        self._cache.clear()
        logger.info("Cleared InMemoryFeatureCache") 