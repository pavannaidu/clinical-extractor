"""Lakebase (PostgreSQL) connection and preset CRUD operations."""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from databricks.sdk import WorkspaceClient

from .schemas import Preset, PresetCreate, PresetUpdate
from .presets import PRESETS

logger = logging.getLogger(__name__)

APP_SCHEMA = "clinical_extractor"
_TOKEN_REFRESH_SECONDS = 45 * 60
_token_lock = threading.Lock()
_token_cache: tuple[str, float] | None = None


def is_configured() -> bool:
    """Return whether the minimum Lakebase connection settings are present."""
    host = os.environ.get("PGHOST") or os.environ.get("LAKEBASE_DB_HOST")
    database = os.environ.get("PGDATABASE") or os.environ.get("LAKEBASE_DB_NAME")
    endpoint = os.environ.get("LAKEBASE_ENDPOINT")
    legacy_password = os.environ.get("LAKEBASE_DB_PASSWORD")
    return bool(host and database and (endpoint or legacy_password))


def _database_token() -> str:
    """Return a cached Lakebase OAuth token, refreshing it before expiry."""
    global _token_cache
    endpoint = os.environ.get("LAKEBASE_ENDPOINT")
    legacy_password = os.environ.get("LAKEBASE_DB_PASSWORD")
    if not endpoint:
        if legacy_password:
            return legacy_password
        raise RuntimeError("LAKEBASE_ENDPOINT is not set")

    now = time.monotonic()
    with _token_lock:
        if _token_cache is not None and now < _token_cache[1]:
            return _token_cache[0]
        token = WorkspaceClient().postgres.generate_database_credential(
            endpoint=endpoint
        ).token
        _token_cache = (token, now + _TOKEN_REFRESH_SECONDS)
        return token


def get_connection_kwargs() -> dict[str, str | int]:
    """Build connection arguments from Databricks Apps' postgres binding."""
    host = os.environ.get("PGHOST") or os.environ.get("LAKEBASE_DB_HOST", "")
    port = os.environ.get("PGPORT") or os.environ.get("LAKEBASE_DB_PORT", "5432")
    dbname = os.environ.get("PGDATABASE") or os.environ.get("LAKEBASE_DB_NAME", "")
    user = os.environ.get("PGUSER") or os.environ.get("LAKEBASE_DB_USER", "")
    if not is_configured():
        raise RuntimeError("Lakebase environment variables not set")
    return {
        "host": host,
        "port": int(port),
        "dbname": dbname,
        "user": user,
        "password": _database_token(),
        "sslmode": os.environ.get("PGSSLMODE", "require"),
        "connect_timeout": 10,
    }


@contextmanager
def get_db():
    """Context manager yielding a psycopg2 connection with auto commit/rollback."""
    conn = psycopg2.connect(**get_connection_kwargs())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the presets and app_settings tables if they don't exist."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {APP_SCHEMA}")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_extractor.presets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    schema_json TEXT NOT NULL,
                    example_prompt TEXT NOT NULL DEFAULT '',
                    system_prompt TEXT NOT NULL DEFAULT '',
                    is_builtin BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                ALTER TABLE clinical_extractor.presets
                ADD COLUMN IF NOT EXISTS system_prompt TEXT NOT NULL DEFAULT ''
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_extractor.app_settings (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)


def seed_defaults():
    """Upsert built-in presets from the hardcoded PRESETS list."""
    with get_db() as conn:
        with conn.cursor() as cur:
            for p in PRESETS:
                cur.execute(
                    """
                    INSERT INTO clinical_extractor.presets (id, name, description, schema_json, example_prompt, system_prompt, is_builtin, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        schema_json = EXCLUDED.schema_json,
                        example_prompt = EXCLUDED.example_prompt,
                        system_prompt = EXCLUDED.system_prompt,
                        is_builtin = TRUE,
                        updated_at = NOW()
                    """,
                    (
                        p["id"],
                        p["name"],
                        p["description"],
                        p["schema_json"],
                        p["example_prompt"],
                        p["system_prompt"],
                    ),
                )


def get_setting(key: str) -> dict | list | None:
    """Return the JSON value stored under key, or None if absent."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT value FROM clinical_extractor.app_settings WHERE key = %s",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                return None
            value = row["value"]
            # psycopg2 returns JSONB as a parsed object, but tolerate text too.
            return json.loads(value) if isinstance(value, str) else value


def set_setting(key: str, value: dict | list) -> None:
    """Upsert a JSON value under key."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clinical_extractor.app_settings (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
                """,
                (key, json.dumps(value)),
            )


def _row_to_preset(row: dict) -> Preset:
    return Preset(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        schema_json=row["schema_json"],
        example_prompt=row["example_prompt"],
        system_prompt=row.get("system_prompt", ""),
        is_builtin=row["is_builtin"],
    )


def list_presets() -> list[Preset]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM clinical_extractor.presets "
                "ORDER BY is_builtin DESC, name ASC"
            )
            return [_row_to_preset(row) for row in cur.fetchall()]


def create_preset(req: PresetCreate) -> Preset:
    preset_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc)
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO clinical_extractor.presets (id, name, description, schema_json, example_prompt, system_prompt, is_builtin, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, FALSE, %s, %s)
                RETURNING *
                """,
                (
                    preset_id,
                    req.name,
                    req.description,
                    req.json_schema,
                    req.example_prompt,
                    req.system_prompt,
                    now,
                    now,
                ),
            )
            row = cur.fetchone()
            return _row_to_preset(row)


def update_preset(preset_id: str, req: PresetUpdate) -> Preset | None:
    updates = {k: v for k, v in req.model_dump(by_alias=True).items() if v is not None}
    if not updates:
        # No fields to update, just return existing
        with get_db() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM clinical_extractor.presets WHERE id = %s",
                    (preset_id,),
                )
                row = cur.fetchone()
                return _row_to_preset(row) if row else None

    set_clauses = [f"{k} = %s" for k in updates]
    set_clauses.append("updated_at = NOW()")
    values = list(updates.values())
    values.append(preset_id)

    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"UPDATE {APP_SCHEMA}.presets SET {', '.join(set_clauses)} WHERE id = %s RETURNING *",
                values,
            )
            row = cur.fetchone()
            return _row_to_preset(row) if row else None


def delete_preset(preset_id: str) -> str:
    """Returns 'deleted', 'builtin', or 'not_found'."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT is_builtin FROM clinical_extractor.presets WHERE id = %s",
                (preset_id,),
            )
            row = cur.fetchone()
            if not row:
                return "not_found"
            if row["is_builtin"]:
                return "builtin"
            cur.execute(
                "DELETE FROM clinical_extractor.presets WHERE id = %s", (preset_id,)
            )
            return "deleted"
