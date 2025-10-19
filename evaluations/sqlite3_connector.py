import sqlite3
from pathlib import Path

db_path = Path("/var/lib/sqlite/main.db")
conn = sqlite3.connect(db_path)

conn.close()
