#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Model Trainer

This module contains tests for the model training functionality.
"""

import asyncio
import unittest
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

from src.core.ml.training.model_trainer import ModelTrainer
from src.core.db.postgres_data_manager import PostgresDataManager
from src.core.features.manager import FeatureManager


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mocks
        self.postgres_manager = AsyncMock(spec=PostgresDataManager)
        self.feature_manager = MagicMock(spec=FeatureManager)
        
        # Set the connected property on postgres_manager mock
        self.postgres_manager.connected = True
        
        # Create trainer
        self.trainer = ModelTrainer(
            postgres_manager=self.postgres_manager,
            feature_manager=self.feature_manager,
            models_dir="./test_models",
            time_interval_seconds=2,
            training_window_hours=24,
            validation_ratio=0.2,
            debug_mode=True
        )
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.trainer.time_interval_seconds, 2)
        self.assertEqual(self.trainer.training_window_hours, 24)
        self.assertEqual(self.trainer.validation_ratio, 0.2)
        self.assertEqual(self.trainer.models_dir, "./test_models")
        self.assertTrue(self.trainer.debug_mode)
    
    async def test_fetch_training_data(self):
        """Test fetching training data."""
        # Mock token data
        token1 = {'id': 'token1', 'name': 'Token 1', 'symbol': 'TKN1', 'address': '0x123'}
        token2 = {'id': 'token2', 'name': 'Token 2', 'symbol': 'TKN2', 'address': '0x456'}
        
        # Mock token list
        self.postgres_manager.get_all_tokens.return_value = [token1, token2]
        
        # Mock trade data for token1
        trades1 = [
            {'id': 't1', 'tokenId': 'token1', 'price': 1.0, 'amount': 10.0, 'type': 'buy', 
             'timestamp': datetime.now() - timedelta(minutes=10)},
            {'id': 't2', 'tokenId': 'token1', 'price': 1.1, 'amount': 5.0, 'type': 'buy', 
             'timestamp': datetime.now() - timedelta(minutes=5)},
            {'id': 't3', 'tokenId': 'token1', 'price': 1.2, 'amount': 8.0, 'type': 'sell', 
             'timestamp': datetime.now()}
        ]
        
        # Mock trade data for token2
        trades2 = [
            {'id': 't4', 'tokenId': 'token2', 'price': 2.0, 'amount': 20.0, 'type': 'buy', 
             'timestamp': datetime.now() - timedelta(minutes=15)},
            {'id': 't5', 'tokenId': 'token2', 'price': 2.1, 'amount': 15.0, 'type': 'sell', 
             'timestamp': datetime.now() - timedelta(minutes=10)}
        ]
        
        # Mock get_trades_for_token to return different data for different tokens
        async def mock_get_trades(token_id, **kwargs):
            if token_id == 'token1':
                return trades1
            elif token_id == 'token2':
                return trades2
            return []
            
        self.postgres_manager.get_trades_for_token.side_effect = mock_get_trades
        
        # Test fetch_training_data
        token_data = await self.trainer.fetch_training_data()
        
        # Assertions
        self.postgres_manager.get_all_tokens.assert_called_once()
        self.assertEqual(self.postgres_manager.get_trades_for_token.call_count, 2)
        
        # Check result
        self.assertIn('token1', token_data)
        self.assertIn('token2', token_data)
        self.assertTrue(isinstance(token_data['token1'], pd.DataFrame))
        self.assertTrue(isinstance(token_data['token2'], pd.DataFrame))
    
    def test_resample_to_interval(self):
        """Test resampling data to 2-second intervals."""
        # Create test DataFrame
        df = pd.DataFrame({
            'timestamp': [
                datetime.now() - timedelta(seconds=10),
                datetime.now() - timedelta(seconds=8),
                datetime.now() - timedelta(seconds=6),
                datetime.now() - timedelta(seconds=4),
                datetime.now() - timedelta(seconds=2)
            ],
            'price': [1.0, 1.1, 1.2, 1.3, 1.4],
            'amount': [10.0, 5.0, 8.0, 12.0, 7.0],
            'type': ['buy', 'buy', 'sell', 'buy', 'sell'],
            'id': ['t1', 't2', 't3', 't4', 't5']
        })
        
        # Test resampling
        resampled = self.trainer._resample_to_interval(df, 'test_token')
        
        # Assertions
        self.assertIn('open', resampled.columns)
        self.assertIn('high', resampled.columns)
        self.assertIn('low', resampled.columns)
        self.assertIn('close', resampled.columns)
        self.assertIn('amount', resampled.columns)
        self.assertIn('trade_count', resampled.columns)
        self.assertIn('tokenId', resampled.columns)
        
        # Check values
        self.assertEqual(resampled['tokenId'].iloc[0], 'test_token')
    
    async def test_compute_features(self):
        """Test computing features."""
        # Create mock token data
        token_data = {
            'token1': pd.DataFrame({
                'timestamp': [datetime.now() - timedelta(seconds=i) for i in range(5)],
                'open': [1.0, 1.1, 1.2, 1.3, 1.4],
                'high': [1.1, 1.2, 1.3, 1.4, 1.5],
                'low': [0.9, 1.0, 1.1, 1.2, 1.3],
                'close': [1.1, 1.2, 1.3, 1.4, 1.5],
                'amount': [10.0, 5.0, 8.0, 12.0, 7.0],
                'trade_count': [1, 2, 1, 3, 1]
            })
        }
        
        # Mock feature providers
        mock_provider1 = MagicMock()
        mock_provider1.get_feature_names.return_value = ['feature1', 'feature2']
        
        mock_provider2 = MagicMock()
        mock_provider2.get_feature_names.return_value = ['feature3', 'feature4']
        
        # Mock feature manager to return providers
        self.feature_manager.get_providers.return_value = [mock_provider1, mock_provider2]
        
        # Mock compute_feature to return different values
        def mock_compute_feature(token_id, feature_name, context):
            if feature_name == 'feature1':
                return 0.1
            elif feature_name == 'feature2':
                return 0.2
            elif feature_name == 'feature3':
                return 0.3
            elif feature_name == 'feature4':
                return 0.4
            return 0.0
            
        self.feature_manager.compute_feature.side_effect = mock_compute_feature
        
        # Test compute_features
        feature_data = await self.trainer.compute_features(token_data)
        
        # Assertions
        self.assertIn('token1', feature_data)
        self.assertTrue(isinstance(feature_data['token1'], pd.DataFrame))
        
        # Check features
        self.assertIn('feature1', feature_data['token1'].columns)
        self.assertIn('feature2', feature_data['token1'].columns)
        self.assertIn('feature3', feature_data['token1'].columns)
        self.assertIn('feature4', feature_data['token1'].columns)


def run_async_test(coro):
    """Run an async test coroutine."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main() 