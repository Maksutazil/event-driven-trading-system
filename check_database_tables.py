#!/usr/bin/env python3
"""
Script to list database tables and their column names
"""

import os
import psycopg2
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Connect to the database
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname="pumpfun_monitor",  # Explicitly set database name
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )
    
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema='public' 
        AND table_type='BASE TABLE';
    """)
    
    tables = cursor.fetchall()
    
    print("Tables in database (with exact case):")
    for table in tables:
        table_name = table[0]
        print(f"- {table_name}")
        
        # Count rows in each table
        try:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            print(f"  Row count: {count}")
        except Exception as e:
            print(f"  Error counting rows: {e}")
    
    # Get column names for tokens and token_trades tables
    print("\nColumn names for 'tokens' table:")
    try:
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'tokens';
        """)
        columns = cursor.fetchall()
        for column in columns:
            print(f"  - {column[0]} ({column[1]})")
    except Exception as e:
        print(f"  Error getting column names: {e}")
    
    print("\nColumn names for 'token_trades' table:")
    try:
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'token_trades';
        """)
        columns = cursor.fetchall()
        for column in columns:
            print(f"  - {column[0]} ({column[1]})")
    except Exception as e:
        print(f"  Error getting column names: {e}")
    
    # Check for sample data
    print("\nSample data from 'token_trades' (first row):")
    try:
        cursor.execute("SELECT * FROM token_trades LIMIT 1")
        row = cursor.fetchone()
        column_names = [desc[0] for desc in cursor.description]
        if row:
            for i, value in enumerate(row):
                print(f"  {column_names[i]}: {value}")
    except Exception as e:
        print(f"  Error getting sample data: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main() 