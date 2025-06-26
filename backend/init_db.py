import sqlite3
import os

def init_db():
    # Create the database file if it doesn't exist
    db_path = 'purogene.db'
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create query_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                question TEXT NOT NULL,
                sql_query TEXT,
                status TEXT NOT NULL,
                execution_time REAL,
                error_message TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print("Database initialized successfully!")
    else:
        print("Database already exists.")

if __name__ == "__main__":
    init_db() 