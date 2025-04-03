#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psycopg2
from datetime import datetime, timedelta

# Connect to the database
conn = psycopg2.connect(
    dbname='pumpfun_monitor',
    user='postgres',
    password='postgres',
    host='localhost'
)

try:
    cursor = conn.cursor()
    
    # Get the total number of trades
    cursor.execute('SELECT COUNT(*) FROM token_trades')
    total_trades = cursor.fetchone()[0]
    print(f"Total trades in database: {total_trades}")
    
    # Get the total number of tokens
    cursor.execute('SELECT COUNT(*) FROM tokens')
    total_tokens = cursor.fetchone()[0]
    print(f"Total tokens in database: {total_tokens}")
    
    # Calculate the time window (last 30 days)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    # Get tokens with at least 5 trades in the time window
    cursor.execute("""
    SELECT COUNT(DISTINCT t.token_id)
    FROM tokens t
    INNER JOIN token_trades tr ON t.token_id = tr.token_id
    WHERE tr.timestamp BETWEEN %s AND %s
    GROUP BY t.token_id
    HAVING COUNT(tr.trade_id) >= 5
    """, (start_time, end_time))
    
    tokens_with_trades = cursor.rowcount
    print(f"Tokens with at least 5 trades in the last 30 days: {tokens_with_trades}")
    
    # Get the top 5 tokens by trade count
    cursor.execute("""
    SELECT t.token_id, COUNT(tr.trade_id) as trade_count
    FROM tokens t
    INNER JOIN token_trades tr ON t.token_id = tr.token_id
    GROUP BY t.token_id
    ORDER BY trade_count DESC
    LIMIT 5
    """)
    
    print("\nTop 5 tokens by trade count:")
    for row in cursor.fetchall():
        print(f"Token {row[0]}: {row[1]} trades")
    
    # Check if there are recent trades
    cursor.execute("""
    SELECT COUNT(*) 
    FROM token_trades 
    WHERE timestamp > %s
    """, (end_time - timedelta(days=7),))
    
    recent_trades = cursor.fetchone()[0]
    print(f"\nTrades in the last 7 days: {recent_trades}")
    
finally:
    conn.close() 