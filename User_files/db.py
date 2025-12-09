# db.py
import os
import psycopg
from contextlib import contextmanager

DATABASE_URL = os.environ["DATABASE_URL"]  # set this in DO App Platform

@contextmanager
def get_conn():
    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        sslmode="require"
    )
    yield conn


