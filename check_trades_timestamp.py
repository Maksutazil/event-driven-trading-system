import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    dbname='pumpfun_monitor',
    user='postgres',
    password='postgres'
)

# Get timestamp range for trades
token_id = 'Bv5disHHAHwDhEbRMN4c329teTdR9aneFQ4EmeJb54Sx'
cursor = conn.cursor(cursor_factory=RealDictCursor)
cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM token_trades WHERE token_id = %s', [token_id])
result = cursor.fetchone()
print(f'Timestamp range for token {token_id}:')
print(f'- Min timestamp: {result["min"]}')
print(f'- Max timestamp: {result["max"]}')

# Get the number of trades within a specific time range (2025-03-31 to 2025-04-01)
cursor.execute('''
    SELECT COUNT(*) 
    FROM token_trades 
    WHERE token_id = %s 
    AND timestamp BETWEEN '2025-03-31 00:00:00' AND '2025-04-01 23:59:59'
''', [token_id])
count_recent = cursor.fetchone()['count']
print(f'Number of trades between 2025-03-31 and 2025-04-01: {count_recent}')

# Get the number of trades within a specific time range (2025-02-01 to 2025-03-01)
cursor.execute('''
    SELECT COUNT(*) 
    FROM token_trades 
    WHERE token_id = %s 
    AND timestamp BETWEEN '2025-02-01 00:00:00' AND '2025-03-01 23:59:59'
''', [token_id])
count_feb = cursor.fetchone()['count']
print(f'Number of trades between 2025-02-01 and 2025-03-01: {count_feb}')

# Get all trades for this token to check their timestamps
cursor.execute('''
    SELECT timestamp 
    FROM token_trades 
    WHERE token_id = %s 
    ORDER BY timestamp DESC 
    LIMIT 5
''', [token_id])
recent_trades = cursor.fetchall()
print("\nMost recent 5 trades:")
for idx, trade in enumerate(recent_trades):
    print(f"{idx+1}. {trade['timestamp']}")

conn.close() 