import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from loguru import logger
from config import DB_CONFIG


@contextmanager
def get_conn():
    """Yield a psycopg2 connection, commit on success, rollback on error."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB error, rolling back: {e}")
        raise
    finally:
        conn.close()


def test_connection() -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                logger.info(f"PostgreSQL connected: {version}")
        return True
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return False


def log_ingestion(source: str, asset: str, status: str,
                  rows_saved: int = 0, error_msg: str = None):
    sql = """
        INSERT INTO ingestion_log (source, asset, status, rows_saved, error_msg)
        VALUES (%s, %s, %s, %s, %s)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (source, asset, status, rows_saved, error_msg))
