import sqlite3
from pathlib import Path

db_path = Path("/var/lib/sqlite/main.db")
conn = sqlite3.connect(db_path)
conn.execute(
    "CREATE TABLE IF NOT EXISTS heroes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, level INTEGER NOT NULL DEFAULT 1);"
)
conn.execute("INSERT INTO heroes (name, level) VALUES (?, ?);", ("Grace Hopper", 99))
conn.commit()
for row in conn.execute("SELECT id, name, level FROM heroes;"):
    print(row)
conn.close()
