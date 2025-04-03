import psycopg2
from psycopg2.extras import RealDictCursor

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    dbname='pumpfun_monitor',
    user='postgres',
    password='postgres'
)

# Get table schema
cursor = conn.cursor(cursor_factory=RealDictCursor)
cursor.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'token_trades'")
columns = cursor.fetchall()

print("Columns in token_trades table:")
for col in columns:
    print(f"- {col['column_name']} ({col['data_type']})")

# Check if there are trades for the token
token_id = 'Bv5disHHAHwDhEbRMN4c329teTdR9aneFQ4EmeJb54Sx'
cursor.execute("SELECT COUNT(*) FROM token_trades WHERE token_id = %s", [token_id])
count = cursor.fetchone()['count']
print(f"\nNumber of trades for token {token_id}: {count}")

# Get a sample trade
if count > 0:
    cursor.execute("SELECT * FROM token_trades WHERE token_id = %s LIMIT 1", [token_id])
    sample = cursor.fetchone()
    print("\nSample trade:")
    for key, value in sample.items():
        print(f"- {key}: {value}")

conn.close() 