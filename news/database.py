"""SQLite database for A2A message logging and CMS storage."""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "newsroom.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            receiver TEXT NOT NULL,
            msg_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            fact_check_notes TEXT,
            status TEXT DEFAULT 'draft',
            published_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS agent_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            agent TEXT NOT NULL,
            step TEXT NOT NULL,
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
    """)
    conn.commit()
    conn.close()


def log_message(run_id, sender, receiver, msg_type, payload):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (run_id, sender, receiver, msg_type, payload, created_at) VALUES (?,?,?,?,?,?)",
        (run_id, sender, receiver, msg_type, json.dumps(payload), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def log_step(run_id, agent, step, result=None, error=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO agent_steps (run_id, agent, step, result, error, created_at) VALUES (?,?,?,?,?,?)",
        (run_id, agent, step, result, error, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def create_run(run_id, topic):
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        "INSERT INTO runs (run_id, topic, status, created_at, updated_at) VALUES (?,?,?,?,?)",
        (run_id, topic, "running", now, now)
    )
    conn.commit()
    conn.close()


def update_run_status(run_id, status):
    conn = get_connection()
    conn.execute(
        "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
        (status, datetime.utcnow().isoformat(), run_id)
    )
    conn.commit()
    conn.close()


def save_article(run_id, topic, content, fact_check_notes, status="published"):
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute(
        "INSERT INTO articles (run_id, topic, content, fact_check_notes, status, published_at) VALUES (?,?,?,?,?,?)",
        (run_id, topic, content, fact_check_notes, status, now)
    )
    conn.commit()
    conn.close()


def get_run_messages(run_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE run_id=? ORDER BY id ASC", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_steps(run_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM agent_steps WHERE run_id=? ORDER BY id ASC", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_article(run_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM articles WHERE run_id=? ORDER BY id DESC LIMIT 1", (run_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_runs():
    conn = get_connection()
    rows = conn.execute(
        "SELECT r.run_id, r.topic, r.status, r.created_at, "
        "(SELECT COUNT(*) FROM messages m WHERE m.run_id=r.run_id) as msg_count "
        "FROM runs r ORDER BY r.created_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
