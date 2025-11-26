# db.py
import os
import psycopg
from contextlib import contextmanager

DATABASE_URL = os.environ["DATABASE_URL"].strip()  # 👈 strip spaces

@contextmanager
def get_conn():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        yield conn
