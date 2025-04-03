#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the PostgreSQL database.

This script creates the required tables for the trading system and
populates them with sample data.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup_database")

def generate_token_id() -> str:
    """Generate a unique token ID."""
    return str(uuid.uuid4())

def generate_wallet_id() -> str:
    """Generate a unique wallet ID."""
    return str(uuid.uuid4())

def generate_trade_id() -> str:
    """Generate a unique trade ID."""
    return str(uuid.uuid4())

def generate_sample_tokens(num_tokens: int = 5) -> List[Dict[str, Any]]:
    """Generate sample token data."""
    tokens = []
    symbols = ["BTC", "ETH", "SOL", "DOGE", "SHIB", "APE", "UNI", "MATIC", "LINK", "AVAX"]
    names = ["Bitcoin", "Ethereum", "Solana", "Dogecoin", "Shiba Inu", "ApeCoin", "Uniswap", "Polygon", "Chainlink", "Avalanche"]
    
    for i in range(min(num_tokens, len(symbols))):
        token_id = generate_token_id()
        token = {
            "id": token_id,
            "address": f"0x{os.urandom(20).hex()}",
            "name": names[i],
            "symbol": symbols[i],
            "created_at": datetime.now() - timedelta(days=30),
            "metadata": {
                "creator": f"0x{os.urandom(20).hex()}",
                "initial_price": 0.01 + (i * 0.005),
                "decimals": 9,
                "total_supply": 1000000000
            }
        }
        tokens.append(token)
    
    return tokens

def generate_sample_wallet(num_wallets: int = 3) -> List[Dict[str, Any]]:
    """Generate sample wallet data."""
    wallets = []
    for i in range(num_wallets):
        wallet_id = generate_wallet_id()
        wallet = {
            "id": wallet_id,
            "address": f"0x{os.urandom(20).hex()}",
            "name": f"Wallet {i+1}"
        }
        wallets.append(wallet)
    
    return wallets

def generate_sample_trades(tokens: List[Dict[str, Any]], wallets: List[Dict[str, Any]], num_trades_per_token: int = 10) -> List[Dict[str, Any]]:
    """Generate sample trade data."""
    trades = []
    
    for token in tokens:
        token_id = token["id"]
        initial_price = token["metadata"]["initial_price"]
        
        # Generate a price series with some volatility
        prices = [initial_price]
        for _ in range(num_trades_per_token - 1):
            # Random price change, biased towards small moves
            change_pct = (0.5 - (0.5 * (0.5 + 0.5 * (0.5 - (0.5 * (0.5 - 0.5 * 0.5))))))
            price_change = prices[-1] * change_pct
            new_price = max(0.001, prices[-1] + price_change)
            prices.append(new_price)
        
        # Generate trades over the past few days
        start_time = datetime.now() - timedelta(days=7)
        time_increment = timedelta(hours=24) / num_trades_per_token
        
        for i in range(num_trades_per_token):
            trade_id = generate_trade_id()
            price = prices[i]
            wallet = wallets[i % len(wallets)]
            
            # Alternate buy/sell
            trade_type = "buy" if i % 2 == 0 else "sell"
            
            # Random amount
            amount = 10 + (i * 2)
            
            trade = {
                "id": trade_id,
                "token_id": token_id,
                "price": price,
                "amount": amount,
                "type": trade_type,
                "timestamp": start_time + (i * time_increment),
                "wallet_id": wallet["id"]
            }
            trades.append(trade)
    
    return trades

def create_database_schema(conn) -> bool:
    """Create the database schema."""
    try:
        with conn.cursor() as cursor:
            # Create Wallet table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Wallet" (
                id text NOT NULL,
                address text NOT NULL,
                name text,
                CONSTRAINT "Wallet_pkey" PRIMARY KEY (id)
            )
            """)
            
            # Create Token table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Token" (
                id text NOT NULL,
                address text NOT NULL,
                name text NOT NULL,
                symbol text NOT NULL,
                "createdAt" timestamp(3) without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata jsonb,
                CONSTRAINT "Token_pkey" PRIMARY KEY (id),
                CONSTRAINT "Token_address_key" UNIQUE (address)
            )
            """)
            
            # Create Trade table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS "Trade" (
                id text NOT NULL,
                "tokenId" text NOT NULL,
                price double precision NOT NULL,
                amount double precision NOT NULL,
                type text NOT NULL,
                "timestamp" timestamp(3) without time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
                "walletId" text NOT NULL,
                CONSTRAINT "Trade_pkey" PRIMARY KEY (id),
                CONSTRAINT "Trade_tokenId_fkey" FOREIGN KEY ("tokenId")
                    REFERENCES "Token" (id) MATCH SIMPLE
                    ON UPDATE CASCADE
                    ON DELETE RESTRICT,
                CONSTRAINT "Trade_walletId_fkey" FOREIGN KEY ("walletId")
                    REFERENCES "Wallet" (id) MATCH SIMPLE
                    ON UPDATE CASCADE
                    ON DELETE RESTRICT
            )
            """)
            
            conn.commit()
            logger.info("Database schema created successfully")
            return True
    
    except Exception as e:
        logger.error(f"Error creating database schema: {e}")
        conn.rollback()
        return False

def insert_sample_data(conn, tokens: List[Dict[str, Any]], wallets: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> bool:
    """Insert sample data into the database."""
    try:
        with conn.cursor() as cursor:
            # Insert wallets
            for wallet in wallets:
                cursor.execute(
                    'INSERT INTO "Wallet" (id, address, name) VALUES (%s, %s, %s)',
                    (wallet["id"], wallet["address"], wallet["name"])
                )
            
            # Insert tokens
            for token in tokens:
                cursor.execute(
                    'INSERT INTO "Token" (id, address, name, symbol, "createdAt", metadata) VALUES (%s, %s, %s, %s, %s, %s)',
                    (token["id"], token["address"], token["name"], token["symbol"], token["created_at"], 
                     psycopg2.extras.Json(token["metadata"]))
                )
            
            # Insert trades
            for trade in trades:
                cursor.execute(
                    'INSERT INTO "Trade" (id, "tokenId", price, amount, type, "timestamp", "walletId") VALUES (%s, %s, %s, %s, %s, %s, %s)',
                    (trade["id"], trade["token_id"], trade["price"], trade["amount"], trade["type"], 
                     trade["timestamp"], trade["wallet_id"])
                )
            
            conn.commit()
            logger.info(f"Inserted {len(wallets)} wallets, {len(tokens)} tokens, and {len(trades)} trades")
            return True
    
    except Exception as e:
        logger.error(f"Error inserting sample data: {e}")
        conn.rollback()
        return False

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description='Set up the PostgreSQL database for the trading system.')
    parser.add_argument('--num-tokens', type=int, default=5, help='Number of tokens to generate')
    parser.add_argument('--num-wallets', type=int, default=3, help='Number of wallets to generate')
    parser.add_argument('--num-trades-per-token', type=int, default=20, help='Number of trades per token')
    parser.add_argument('--drop-tables', action='store_true', help='Drop existing tables before creating new ones')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get database connection parameters from environment variables
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'postgres')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    
    # Connect to the database
    conn = None
    try:
        logger.info(f"Connecting to PostgreSQL database at {db_host}:{db_port}")
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            cursor_factory=RealDictCursor
        )
        
        # Drop tables if requested
        if args.drop_tables:
            with conn.cursor() as cursor:
                logger.info("Dropping existing tables...")
                cursor.execute('DROP TABLE IF EXISTS "Trade" CASCADE')
                cursor.execute('DROP TABLE IF EXISTS "Token" CASCADE')
                cursor.execute('DROP TABLE IF EXISTS "Wallet" CASCADE')
                conn.commit()
        
        # Create schema
        if not create_database_schema(conn):
            logger.error("Failed to create database schema")
            return 1
        
        # Generate sample data
        logger.info("Generating sample data...")
        wallets = generate_sample_wallet(args.num_wallets)
        tokens = generate_sample_tokens(args.num_tokens)
        trades = generate_sample_trades(tokens, wallets, args.num_trades_per_token)
        
        # Insert sample data
        if not insert_sample_data(conn, tokens, wallets, trades):
            logger.error("Failed to insert sample data")
            return 1
        
        logger.info("Database setup completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Unhandled exception in database setup: {e}")
        return 1
    
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    sys.exit(main()) 