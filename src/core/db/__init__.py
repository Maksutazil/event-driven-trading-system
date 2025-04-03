"""
Database module for interacting with databases.

This module provides clients for connecting to databases and retrieving
historical trade data.
"""

from .postgres_data_manager import PostgresDataManager

__all__ = ['PostgresDataManager'] 