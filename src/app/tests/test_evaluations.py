import asyncio
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi.responses import JSONResponse
from starlette.requests import Request

from api import database, evaluations
from api.schemas import (
    EvaluationColumn,
    EvaluationCreateRequest,
    EvaluationDatasetRequest,
    InferenceResponse,
)


SCHEMA = {
    "type": "object",
    "properties": {
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "negated": {"type": "boolean"},
                },
                "required": ["name", "negated"],
            },
        }
    },
    "required": ["diagnoses"],
}


def _request(**headers: str) -> Request:
    encoded_headers = [(name.encode(), value.encode()) for name, value in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": encoded_headers})


def _run_row(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    row = {
        "id": "run-1",
        "owner_id": "clinician@example.com",
        "name": "Evaluation",
        "status": "completed",
        "table_name": "catalog.schema.dataset",
        "input_column": "note",
        "ground_truth_column": "truth",
        "identifier_column": "id",
        "model": "qwen",
        "preset_id": "diagnosis_extraction",
        "schema_json": json.dumps(SCHEMA),
        "system_prompt": "Extract diagnoses.",
        "row_limit": 10,
        "enable_groundedness": False,
        "enable_completeness": False,
        "total_rows": 1,
        "completed_rows": 1,
        "succeeded_rows": 1,
        "failed_rows": 0,
        "mlflow_experiment_id": "123",
        "mlflow_run_id": "abc",
        "aggregate_metrics": {},
        "error_code": None,
        "error_message": None,
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "updated_at": now,
    }
    row.update(overrides)
    return row


def _create_body() -> EvaluationCreateRequest:
    return EvaluationCreateRequest(
        name="Evaluation",
        table_name="catalog.schema.dataset",
        input_column="note",
        ground_truth_column="truth",
        identifier_column="id",
        model="qwen",
        preset_id="diagnosis_extraction",
        schema_json=json.dumps(SCHEMA),
        system_prompt="Extract diagnoses.",
        row_limit=10,
    )


def test_identifier_quoting_requires_three_part_table_name():
    assert evaluations._quote_identifier("clinical`notes") == "`clinical``notes`"
    assert evaluations._quote_table_name("catalog.schema.notes") == "`catalog`.`schema`.`notes`"

    try:
        evaluations._quote_table_name("schema.notes")
    except ValueError as exc:
        assert "three-part" in str(exc)
    else:
        raise AssertionError("Expected a three-part table validation error")


def test_user_query_uses_only_forwarded_user_token(monkeypatch):
    captured: dict[str, str] = {}

    class FakeStatementExecution:
        def execute_statement(self, **_kwargs):
            return SimpleNamespace(
                statement_id="statement-1",
                status=SimpleNamespace(state=evaluations.StatementState.SUCCEEDED),
                manifest=SimpleNamespace(
                    schema=SimpleNamespace(
                        columns=[SimpleNamespace(name="catalog")]
                    )
                ),
                result=SimpleNamespace(
                    data_array=[["demo_catalog"]],
                    next_chunk_index=None,
                ),
            )

    class FakeWorkspaceClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.statement_execution = FakeStatementExecution()

    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "warehouse-1")
    monkeypatch.setattr(evaluations, "_workspace_host", lambda: "https://workspace")
    monkeypatch.setattr(evaluations, "WorkspaceClient", FakeWorkspaceClient)

    rows = evaluations._execute_user_query("forwarded-token", "SHOW CATALOGS")

    assert rows == [{"catalog": "demo_catalog"}]
    assert captured == {
        "host": "https://workspace",
        "token": "forwarded-token",
        "auth_type": "pat",
    }


def test_user_query_sanitizes_missing_sql_scope(monkeypatch):
    class FakeStatementExecution:
        def execute_statement(self, **_kwargs):
            raise evaluations.PermissionDenied(
                "403 Forbidden: Invalid scope, required scopes: sql. Request log: secret"
            )

    class FakeWorkspaceClient:
        def __init__(self, **_kwargs):
            self.statement_execution = FakeStatementExecution()

    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "warehouse-1")
    monkeypatch.setattr(evaluations, "_workspace_host", lambda: "https://workspace")
    monkeypatch.setattr(evaluations, "WorkspaceClient", FakeWorkspaceClient)

    try:
        evaluations._execute_user_query("forwarded-token", "SHOW CATALOGS")
    except PermissionError as exc:
        assert "session predates SQL access" in str(exc)
        assert "secret" not in str(exc)
    else:
        raise AssertionError("Expected a sanitized authorization error")


def test_workspace_host_normalizes_url_scheme(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "workspace.cloud.databricks.com/")
    assert evaluations._workspace_host() == "https://workspace.cloud.databricks.com"


def test_compare_json_normalizes_array_order_and_reports_nested_paths():
    expected = {
        "diagnoses": [
            {"name": "pneumonia", "negated": False},
            {"name": "pneumothorax", "negated": True},
        ]
    }
    reordered = {
        "diagnoses": [
            {"negated": True, "name": "pneumothorax"},
            {"negated": False, "name": "pneumonia"},
        ]
    }
    matched, total, mismatches = evaluations.compare_json(reordered, expected)
    assert matched == total
    assert mismatches == []

    reordered["diagnoses"][0]["negated"] = False
    matched, total, mismatches = evaluations.compare_json(reordered, expected)
    assert matched == total - 1
    assert mismatches == ["$/diagnoses/1/negated"]


def test_preflight_parses_and_validates_ground_truth(monkeypatch):
    async def fake_columns(_request, _table_name):
        return [
            EvaluationColumn(name="id", data_type="string"),
            EvaluationColumn(name="note", data_type="string"),
            EvaluationColumn(name="truth", data_type="string"),
        ]

    async def fake_query(_request, _statement, **_kwargs):
        return [
            {
                "record_id": "patient-1",
                "input_text": "No pneumothorax.",
                "ground_truth_json": json.dumps(
                    {"diagnoses": [{"name": "pneumothorax", "negated": True}]}
                ),
            },
            {
                "record_id": "patient-2",
                "input_text": "Pneumonia.",
                "ground_truth_json": "{}",
            },
        ]

    monkeypatch.setattr(evaluations, "_table_columns", fake_columns)
    monkeypatch.setattr(evaluations, "_user_query", fake_query)
    body = EvaluationDatasetRequest(
        table_name="catalog.schema.dataset",
        input_column="note",
        ground_truth_column="truth",
        identifier_column="id",
        schema_json=json.dumps(SCHEMA),
        row_limit=10,
    )

    rows, warnings = asyncio.run(evaluations._read_evaluation_rows(_request(), body))

    assert rows[0]["record_id"] == "patient-1"
    assert rows[0]["ground_truth"]["diagnoses"][0]["negated"] is True
    assert len(warnings) == 1
    assert "diagnoses" in warnings[0]


def test_evaluation_execution_updates_progress_and_metrics(monkeypatch):
    updates: list[dict[str, Any]] = []
    saved_results: list[dict[str, Any]] = []

    monkeypatch.setattr(database, "evaluation_cancel_requested", lambda _run_id: False)
    monkeypatch.setattr(database, "update_evaluation_run", lambda _run_id, **values: updates.append(values))
    monkeypatch.setattr(database, "upsert_evaluation_result", lambda values: saved_results.append(values))

    async def fake_inference(_request):
        return InferenceResponse(
            prediction={"diagnoses": [{"name": "pneumonia", "negated": False}]},
            latency_ms=125,
            raw_output="{}",
            model="qwen",
        )

    monkeypatch.setattr(evaluations, "run_inference", fake_inference)
    monkeypatch.setattr(
        evaluations,
        "_run_mlflow_evaluation",
        lambda *_args: ({"groundedness/mean": 1.0}, {"patient-1": "trace-1"}),
    )
    rows = [
        {
            "record_id": "patient-1",
            "input_text": "Pneumonia.",
            "ground_truth": {"diagnoses": [{"name": "pneumonia", "negated": False}]},
        }
    ]

    asyncio.run(evaluations._execute_evaluation("run-1", _create_body(), rows, "mlflow-1"))

    assert saved_results[0]["matched_fields"] == saved_results[0]["total_fields"]
    assert saved_results[-1]["mlflow_trace_id"] == "trace-1"
    assert any(update.get("completed_rows") == 1 for update in updates)
    completed = next(update for update in updates if update.get("status") == "completed")
    assert completed["aggregate_metrics"]["overall_field_accuracy"] == 1.0
    assert completed["aggregate_metrics"]["judge_metrics"] == {"groundedness/mean": 1.0}


def test_evaluation_execution_honors_cancellation(monkeypatch):
    updates: list[dict[str, Any]] = []
    inference_called = False

    monkeypatch.setattr(database, "evaluation_cancel_requested", lambda _run_id: True)
    monkeypatch.setattr(database, "update_evaluation_run", lambda _run_id, **values: updates.append(values))
    monkeypatch.setattr(evaluations, "_terminate_mlflow_run", lambda *_args: None)

    async def fake_inference(_request):
        nonlocal inference_called
        inference_called = True
        return JSONResponse(status_code=500, content={})

    monkeypatch.setattr(evaluations, "run_inference", fake_inference)
    rows = [{"record_id": "1", "input_text": "note", "ground_truth": {"diagnoses": []}}]

    asyncio.run(evaluations._execute_evaluation("run-1", _create_body(), rows, "mlflow-1"))

    assert inference_called is False
    assert any(update.get("status") == "cancelled" for update in updates)


def test_run_lookup_is_scoped_to_signed_in_owner(monkeypatch):
    captured: dict[str, str] = {}
    monkeypatch.setattr(database, "is_configured", lambda: True)

    def fake_get(run_id: str, owner_id: str):
        captured.update(run_id=run_id, owner_id=owner_id)
        return _run_row(id=run_id, owner_id=owner_id)

    monkeypatch.setattr(database, "get_evaluation_run", fake_get)
    response = asyncio.run(
        evaluations.get_run(
            _request(**{"x-forwarded-email": "Clinician@Example.com"}),
            "run-42",
        )
    )

    assert response.run_id == "run-42"
    assert captured == {"run_id": "run-42", "owner_id": "clinician@example.com"}


class _RecoveryCursor:
    def __init__(self):
        self.calls: list[str] = []
        self.rowcount = 3

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement: str, _params=None):
        self.calls.append(statement)


class _RecoveryConnection:
    def __init__(self, cursor: _RecoveryCursor):
        self._cursor = cursor

    def cursor(self, **_kwargs):
        return self._cursor


def test_lakebase_migration_and_restart_recovery(monkeypatch):
    cursor = _RecoveryCursor()

    @contextmanager
    def fake_db():
        yield _RecoveryConnection(cursor)

    monkeypatch.setattr(database, "get_db", fake_db)
    database.init_db()
    interrupted = database.mark_evaluation_runs_interrupted()
    statements = "\n".join(cursor.calls)

    assert "CREATE TABLE IF NOT EXISTS clinical_extractor.evaluation_runs" in statements
    assert "CREATE TABLE IF NOT EXISTS clinical_extractor.evaluation_results" in statements
    assert "status = 'interrupted'" in statements
    assert interrupted == 3
