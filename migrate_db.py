import sqlite3
from agents.common.config import settings
import os

def migrate_db():
    db_path_str = settings.DATABASE_URL.split('///')[1]
    db_path = os.path.abspath(db_path_str)
    
    print(f"Migrating database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("ALTER TABLE videos ADD COLUMN processing_config TEXT")
        conn.commit()
        print("Successfully added 'processing_config' column to 'videos' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column" in str(e):
            print("Column 'processing_config' already exists.")
        else:
            print(f"Error adding column: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_db()
