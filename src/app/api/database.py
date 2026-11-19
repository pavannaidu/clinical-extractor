"""Lakebase connections and persistence for presets, settings, and evaluations."""

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
    """Create application tables if they don't exist."""
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_extractor.evaluation_runs (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    input_column TEXT NOT NULL,
                    ground_truth_column TEXT NOT NULL,
                    identifier_column TEXT,
                    model TEXT NOT NULL,
                    preset_id TEXT NOT NULL,
                    schema_json TEXT NOT NULL,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    row_limit INTEGER NOT NULL DEFAULT 100,
                    enable_groundedness BOOLEAN NOT NULL DEFAULT FALSE,
                    enable_completeness BOOLEAN NOT NULL DEFAULT FALSE,
                    status TEXT NOT NULL DEFAULT 'queued',
                    total_rows INTEGER NOT NULL DEFAULT 0,
                    completed_rows INTEGER NOT NULL DEFAULT 0,
                    succeeded_rows INTEGER NOT NULL DEFAULT 0,
                    failed_rows INTEGER NOT NULL DEFAULT 0,
                    mlflow_experiment_id TEXT,
                    mlflow_run_id TEXT,
                    aggregate_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error_code TEXT,
                    error_message TEXT,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS evaluation_runs_owner_created_idx
                ON clinical_extractor.evaluation_runs (owner_id, created_at DESC)
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS evaluation_runs_one_active_per_owner_idx
                ON clinical_extractor.evaluation_runs (owner_id)
                WHERE status IN ('queued', 'running')
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_extractor.evaluation_results (
                    run_id TEXT NOT NULL REFERENCES clinical_extractor.evaluation_runs(id) ON DELETE CASCADE,
                    record_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION,
                    matched_fields INTEGER NOT NULL DEFAULT 0,
                    total_fields INTEGER NOT NULL DEFAULT 0,
                    mismatched_paths JSONB NOT NULL DEFAULT '[]'::jsonb,
                    schema_valid BOOLEAN NOT NULL DEFAULT FALSE,
                    mlflow_trace_id TEXT,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (run_id, record_id)
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


def mark_evaluation_runs_interrupted() -> int:
    """Mark work that cannot survive an app process restart as interrupted."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE clinical_extractor.evaluation_runs
                SET status = 'interrupted',
                    error_code = 'APP_RESTARTED',
                    error_message = 'The app restarted before this evaluation finished.',
                    completed_at = NOW(),
                    updated_at = NOW()
                WHERE status IN ('queued', 'running')
                """
            )
            return cur.rowcount


def _decode_json(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _evaluation_run_row(row: dict) -> dict:
    result = dict(row)
    result["aggregate_metrics"] = _decode_json(
        result.get("aggregate_metrics"), {}
    )
    return result


def create_evaluation_run(values: dict) -> dict:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO clinical_extractor.evaluation_runs (
                    id, owner_id, name, table_name, input_column,
                    ground_truth_column, identifier_column, model, preset_id,
                    schema_json, system_prompt, row_limit, enable_groundedness,
                    enable_completeness, total_rows, mlflow_experiment_id,
                    mlflow_run_id
                ) VALUES (
                    %(id)s, %(owner_id)s, %(name)s, %(table_name)s,
                    %(input_column)s, %(ground_truth_column)s,
                    %(identifier_column)s, %(model)s, %(preset_id)s,
                    %(schema_json)s, %(system_prompt)s, %(row_limit)s,
                    %(enable_groundedness)s, %(enable_completeness)s,
                    %(total_rows)s, %(mlflow_experiment_id)s, %(mlflow_run_id)s
                )
                RETURNING *
                """,
                values,
            )
            return _evaluation_run_row(cur.fetchone())


def has_active_evaluation_run(owner_id: str) -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM clinical_extractor.evaluation_runs
                WHERE owner_id = %s AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (owner_id,),
            )
            return cur.fetchone() is not None


def list_evaluation_runs(owner_id: str, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM clinical_extractor.evaluation_runs
                WHERE owner_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (owner_id, limit),
            )
            return [_evaluation_run_row(row) for row in cur.fetchall()]


def get_evaluation_run(run_id: str, owner_id: str) -> dict | None:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM clinical_extractor.evaluation_runs
                WHERE id = %s AND owner_id = %s
                """,
                (run_id, owner_id),
            )
            row = cur.fetchone()
            return _evaluation_run_row(row) if row else None


def update_evaluation_run(run_id: str, **values) -> dict | None:
    allowed = {
        "status",
        "completed_rows",
        "succeeded_rows",
        "failed_rows",
        "aggregate_metrics",
        "error_code",
        "error_message",
        "cancel_requested",
        "started_at",
        "completed_at",
        "mlflow_run_id",
        "mlflow_experiment_id",
    }
    updates = {key: value for key, value in values.items() if key in allowed}
    if not updates:
        raise ValueError("No evaluation run fields to update")
    assignments = []
    params = []
    for key, value in updates.items():
        assignments.append(f"{key} = %s")
        if key == "aggregate_metrics":
            value = json.dumps(value)
        params.append(value)
    assignments.append("updated_at = NOW()")
    params.append(run_id)
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                UPDATE clinical_extractor.evaluation_runs
                SET {', '.join(assignments)}
                WHERE id = %s
                RETURNING *
                """,
                params,
            )
            row = cur.fetchone()
            return _evaluation_run_row(row) if row else None


def evaluation_cancel_requested(run_id: str) -> bool:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT cancel_requested FROM clinical_extractor.evaluation_runs WHERE id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            return bool(row and row[0])


def upsert_evaluation_result(values: dict) -> None:
    params = dict(values)
    params["mismatched_paths"] = json.dumps(values.get("mismatched_paths", []))
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clinical_extractor.evaluation_results (
                    run_id, record_id, status, latency_ms, matched_fields,
                    total_fields, mismatched_paths, schema_valid,
                    mlflow_trace_id, error_message
                ) VALUES (
                    %(run_id)s, %(record_id)s, %(status)s, %(latency_ms)s,
                    %(matched_fields)s, %(total_fields)s,
                    %(mismatched_paths)s, %(schema_valid)s,
                    %(mlflow_trace_id)s, %(error_message)s
                )
                ON CONFLICT (run_id, record_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    latency_ms = EXCLUDED.latency_ms,
                    matched_fields = EXCLUDED.matched_fields,
                    total_fields = EXCLUDED.total_fields,
                    mismatched_paths = EXCLUDED.mismatched_paths,
                    schema_valid = EXCLUDED.schema_valid,
                    mlflow_trace_id = EXCLUDED.mlflow_trace_id,
                    error_message = EXCLUDED.error_message
                """,
                params,
            )


def list_evaluation_results(run_id: str) -> list[dict]:
    with get_db() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM clinical_extractor.evaluation_results
                WHERE run_id = %s
                ORDER BY created_at ASC, record_id ASC
                """,
                (run_id,),
            )
            rows = []
            for row in cur.fetchall():
                value = dict(row)
                value["mismatched_paths"] = _decode_json(
                    value.get("mismatched_paths"), []
                )
                rows.append(value)
            return rows
