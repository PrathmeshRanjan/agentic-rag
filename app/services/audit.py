import json
import sqlite3
from datetime import datetime, timezone

def init_db() -> None:
    con = sqlite3.connect('data/audit.db')
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect('data/audit.db')
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()