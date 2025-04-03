#!/usr/bin/env python3
"""
check_tokens_with_trades.py
Script to identify tokens with the most trades in the database,
which can be used for model training.
"""

import os
import sys
import json
import logging
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    load_dotenv()
    try:
        # Explicitly set the database name to pumpfun_monitor
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname="pumpfun_monitor",  # Explicitly set database name
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def get_tokens_with_most_trades(conn, limit=10):
    """
    Get tokens with the most trades in the database.
    
    Args:
        conn: PostgreSQL connection
        limit: Number of tokens to return
    
    Returns:
        List of tuples (token_id, trade_count)
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT token_id, COUNT(*) as trade_count
            FROM token_trades
            GROUP BY token_id
            ORDER BY trade_count DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        return results
    except Exception as e:
        logger.error(f"Error querying tokens with most trades: {e}")
        return []
    finally:
        cursor.close()

def get_token_trade_time_range(conn, token_id):
    """
    Get the time range of trades for a specific token.
    
    Args:
        conn: PostgreSQL connection
        token_id: Token ID to check
    
    Returns:
        Tuple of (first_trade_time, last_trade_time, trade_count)
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                MIN(timestamp) as first_trade,
                MAX(timestamp) as last_trade,
                COUNT(*) as trade_count
            FROM token_trades
            WHERE token_id = %s
        """, (token_id,))
        
        result = cursor.fetchone()
        return result
    except Exception as e:
        logger.error(f"Error querying token trade time range: {e}")
        return None, None, 0
    finally:
        cursor.close()

def get_trade_samples(conn, token_id, limit=5):
    """
    Get sample trades for a specific token.
    
    Args:
        conn: PostgreSQL connection
        token_id: Token ID to check
        limit: Number of trades to return
    
    Returns:
        List of trade records
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                trade_id, 
                token_id, 
                timestamp,
                type,
                token_amount,
                sol_amount,
                price_sol,
                market_cap
            FROM token_trades
            WHERE token_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (token_id, limit))
        
        results = cursor.fetchall()
        trades = []
        for trade in results:
            trades.append({
                "trade_id": trade[0],
                "token_id": trade[1],
                "timestamp": trade[2].isoformat(),
                "type": trade[3],
                "token_amount": float(trade[4]),
                "sol_amount": float(trade[5]),
                "price_sol": float(trade[6]),
                "market_cap": float(trade[7])
            })
        return trades
    except Exception as e:
        logger.error(f"Error querying sample trades: {e}")
        return []
    finally:
        cursor.close()

def main():
    # Connect to the database
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        sys.exit(1)
    
    try:
        print("Connected to pumpfun_monitor database")
        
        # Get total trades and tokens in database
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM token_trades")
        total_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tokens")
        total_tokens = cursor.fetchone()[0]
        
        print(f"Total trades in database: {total_trades}")
        print(f"Total tokens in database: {total_tokens}")
        print()
        
        # Get tokens with most trades
        top_tokens = get_tokens_with_most_trades(conn, 20)
        
        print(f"Top {len(top_tokens)} tokens with the most trades:")
        print("-" * 60)
        
        token_details = []
        
        for token_id, trade_count in top_tokens:
            first_trade, last_trade, _ = get_token_trade_time_range(conn, token_id)
            
            time_span = "N/A"
            if first_trade and last_trade:
                days_span = (last_trade - first_trade).days
                time_span = f"{days_span} days"
            
            print(f"Token {token_id}: {trade_count} trades, Time span: {time_span}")
            print(f"  First trade: {first_trade}")
            print(f"  Last trade: {last_trade}")
            
            trade_samples = get_trade_samples(conn, token_id, 3)
            if trade_samples:
                print("  Sample trades:")
                for trade in trade_samples:
                    print(f"    Type: {trade['type']}, Price: {trade['price_sol']}, Amount: {trade['token_amount']}, Time: {trade['timestamp']}")
            
            print("-" * 60)
            
            token_details.append({
                "token_id": token_id,
                "trade_count": trade_count,
                "first_trade": first_trade.isoformat() if first_trade else None,
                "last_trade": last_trade.isoformat() if last_trade else None,
                "time_span_days": (last_trade - first_trade).days if first_trade and last_trade else None,
                "sample_trades": trade_samples
            })
        
        # Save token details to a JSON file
        with open("token_trade_details.json", "w") as f:
            json.dump(token_details, f, indent=2)
        
        print(f"Token trade details saved to token_trade_details.json")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        conn.close()
        logger.info("Disconnected from PostgreSQL database")

if __name__ == "__main__":
    main() 