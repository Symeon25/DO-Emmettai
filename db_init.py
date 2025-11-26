# db_init.py
from db import get_conn

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # Create app_users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS app_users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        );
        """)

        # Create conversations table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            messages JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT fk_conv_user FOREIGN KEY (username)
                REFERENCES app_users (username)
                ON DELETE CASCADE
        );
        """)

        print("✔ Database tables ensured.")
