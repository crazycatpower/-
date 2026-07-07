import hashlib
import os
from contextlib import contextmanager

DB_RUN_MIGRATIONS = os.getenv("DB_RUN_MIGRATIONS", "1").strip().lower() not in ("0", "false", "no")
DB_RUN_BACKFILL = os.getenv("DB_RUN_BACKFILL", "0").strip().lower() in ("1", "true", "yes")

_DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()


@contextmanager
def db_conn():
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(_DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def build_where(clauses: list) -> object:
    from psycopg.sql import SQL
    parts = [SQL(c) if isinstance(c, str) else c for c in clauses]
    return SQL(" AND ").join(parts) if parts else SQL("TRUE")


def _lock_key(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:15], 16) % (2 ** 63)


def with_advisory_lock(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (_lock_key(name),))


def release_advisory_lock(conn, name: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s)", (_lock_key(name),))
