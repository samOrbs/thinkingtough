"""
Basic Memory: SQLite conversation persistence.
Stores conversation turns for session continuity.
"""

import sqlite3
import uuid
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_active DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn


def create_session() -> str:
    session_id = str(uuid.uuid4())
    db = get_db()
    db.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    db.commit()
    db.close()
    return session_id


def save_turn(session_id: str, role: str, content: str):
    db = get_db()
    db.execute(
        "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    db.execute(
        "UPDATE sessions SET last_active = ? WHERE id = ?",
        (datetime.now().isoformat(), session_id)
    )
    db.commit()
    db.close()


def get_history(session_id: str, limit: int = 20) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    db.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
