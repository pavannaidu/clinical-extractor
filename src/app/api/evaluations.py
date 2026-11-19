import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.sql import StatementState
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from jsonschema import Draft202012Validator

from . import database
from .routes import _load_allowed_endpoints, run_inference
from .schemas import (
    ErrorResponse,
    EvaluationColumn,
    EvaluationColumnsResponse,
    EvaluationCreateRequest,
    EvaluationDatasetRequest,
    EvaluationDiscoveryResponse,
    EvaluationMetrics,
    EvaluationPreflightResponse,
    EvaluationPreflightSample,
    EvaluationReportResponse,
    EvaluationResultSummary,
    EvaluationRunResponse,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])

EVALUATION_MAX_ROWS = 100
EVALUATION_MAX_ACTIVE_GLOBAL = 2
EVALUATION_SQL_TIMEOUT_S = 105
EVALUATION_EXPERIMENT_NAME = "/Shared/clinical-extractor-evaluations"
EVALUATION_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "interrupted",
}

_evaluation_tasks: dict[str, asyncio.Task] = {}
_evaluation_semaphore = asyncio.Semaphore(EVALUATION_MAX_ACTIVE_GLOBAL)


def _api_error(status_code: int, error_code: str, message: str, detail: str | None = None):
    payload = ErrorResponse(
        error_code=error_code,
        message=message,
        detail=detail,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _owner_id(request: Request) -> str:
    for header_name in (
        "x-forwarded-email",
        "x-databricks-user-id",
        "x-forwarded-user",
    ):
        value = request.headers.get(header_name)
        if value and value.strip():
            return value.strip().casefold()
    return "anonymous"


def _user_token(request: Request) -> str | None:
    value = request.headers.get("x-forwarded-access-token")
    return value.strip() if value and value.strip() else None


def _quote_identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 255:
        raise ValueError("Invalid Unity Catalog identifier")
    return f"`{value.strip().replace('`', '``')}`"


def _quote_table_name(value: str) -> str:
    parts = value.split(".")
    if len(parts) != 3:
        raise ValueError("Use a three-part Unity Catalog table name")
    return ".".join(_quote_identifier(part) for part in parts)


def _row_value(row: dict, *names: str):
    values_by_name = {str(key).casefold(): value for key, value in row.items()}
    for name in names:
        value = values_by_name.get(name.casefold())
        if value is not None:
            return value
    return None


def _workspace_host() -> str:
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        host = WorkspaceClient().config.host
    host = host.rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"


def _execute_user_query(token: str, statement: str, *, row_limit: int = 1000) -> list[dict]:
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("The evaluation SQL warehouse is not configured")

    client = WorkspaceClient(host=_workspace_host(), token=token, auth_type="pat")
    try:
        response = client.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=warehouse_id,
            row_limit=row_limit,
            wait_timeout="30s",
        )
    except PermissionDenied as exc:
        message = str(exc)
        if "required scopes: sql" in message.casefold():
            raise PermissionError(
                "Your app session predates SQL access. Sign out and back in to "
                "Databricks, then reopen Clinical Extractor."
            ) from exc
        raise PermissionError(
            "You do not have permission to query this SQL warehouse or Unity Catalog data."
        ) from exc
    deadline = time.monotonic() + EVALUATION_SQL_TIMEOUT_S
    while response.status and response.status.state in {
        StatementState.PENDING,
        StatementState.RUNNING,
    }:
        if time.monotonic() >= deadline:
            if response.statement_id:
                client.statement_execution.cancel_execution(response.statement_id)
            raise TimeoutError("The Unity Catalog query exceeded the app time limit")
        time.sleep(0.5)
        response = client.statement_execution.get_statement(response.statement_id)

    if not response.status or response.status.state != StatementState.SUCCEEDED:
        error_message = "The Unity Catalog query failed"
        if response.status and response.status.error and response.status.error.message:
            error_message = response.status.error.message
        raise RuntimeError(error_message)

    columns = []
    if response.manifest and response.manifest.schema:
        columns = [
            column.name or f"column_{index}"
            for index, column in enumerate(response.manifest.schema.columns or [])
        ]

    data_rows = list(response.result.data_array or []) if response.result else []
    next_chunk_index = response.result.next_chunk_index if response.result else None
    while response.statement_id and next_chunk_index is not None:
        chunk = client.statement_execution.get_statement_result_chunk_n(
            response.statement_id,
            next_chunk_index,
        )
        data_rows.extend(chunk.data_array or [])
        next_chunk_index = chunk.next_chunk_index

    return [dict(zip(columns, row, strict=False)) for row in data_rows]


async def _user_query(request: Request, statement: str, *, row_limit: int = 1000):
    token = _user_token(request)
    if not token:
        raise PermissionError("Signed-in user authorization is unavailable")
    return await asyncio.to_thread(
        _execute_user_query,
        token,
        statement,
        row_limit=row_limit,
    )


def _decode_ground_truth(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Ground truth must be a JSON object")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Ground truth must be a JSON object")
    return parsed


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        canonical = [_canonicalize(item) for item in value]
        return sorted(
            canonical,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    return value


def _flatten(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {path: {}}
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            flattened.update(_flatten(child, f"{path}/{escaped}"))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path: []}
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten(child, f"{path}/{index}"))
        return flattened
    return {path: value}


def compare_json(prediction: Any, ground_truth: dict[str, Any]) -> tuple[int, int, list[str]]:
    actual = _flatten(_canonicalize(prediction))
    expected = _flatten(_canonicalize(ground_truth))
    paths = sorted(set(actual) | set(expected))
    mismatched = [path for path in paths if actual.get(path) != expected.get(path)]
    return len(paths) - len(mismatched), len(paths), mismatched


async def _table_columns(request: Request, table_name: str) -> list[EvaluationColumn]:
    rows = await _user_query(
        request,
        f"DESCRIBE TABLE {_quote_table_name(table_name)}",
        row_limit=1000,
    )
    columns = []
    for row in rows:
        name = _row_value(row, "col_name", "column_name")
        data_type = _row_value(row, "data_type", "type_text") or ""
        if name and not name.startswith("#"):
            columns.append(EvaluationColumn(name=name, data_type=data_type))
    return columns


async def _read_evaluation_rows(
    request: Request,
    body: EvaluationDatasetRequest,
) -> tuple[list[dict], list[str]]:
    table_name = _quote_table_name(body.table_name)
    input_column = _quote_identifier(body.input_column)
    truth_column = _quote_identifier(body.ground_truth_column)
    id_column = (
        f"CAST({_quote_identifier(body.identifier_column)} AS STRING)"
        if body.identifier_column
        else f"CAST(ROW_NUMBER() OVER (ORDER BY CAST({input_column} AS STRING)) AS STRING)"
    )

    columns = await _table_columns(request, body.table_name)
    type_by_name = {column.name: column.data_type.casefold() for column in columns}
    truth_type = type_by_name.get(body.ground_truth_column, "")
    if body.input_column not in type_by_name or body.ground_truth_column not in type_by_name:
        raise ValueError("The selected input or ground-truth column no longer exists")
    if body.identifier_column and body.identifier_column not in type_by_name:
        raise ValueError("The selected identifier column no longer exists")

    truth_expression = f"CAST({truth_column} AS STRING)"
    if truth_type.startswith(("struct", "map", "array", "variant")):
        truth_expression = f"to_json({truth_column})"

    statement = f"""
        SELECT
            {id_column} AS record_id,
            CAST({input_column} AS STRING) AS input_text,
            {truth_expression} AS ground_truth_json
        FROM {table_name}
        WHERE {input_column} IS NOT NULL AND {truth_column} IS NOT NULL
        LIMIT {min(body.row_limit, EVALUATION_MAX_ROWS)}
    """
    raw_rows = await _user_query(request, statement, row_limit=body.row_limit)
    validator = Draft202012Validator(json.loads(body.json_schema))
    rows = []
    warnings = []
    for index, row in enumerate(raw_rows, start=1):
        try:
            input_text = _row_value(row, "input_text")
            if not isinstance(input_text, str) or not input_text.strip():
                raise ValueError("Input is empty")
            ground_truth = _decode_ground_truth(_row_value(row, "ground_truth_json"))
            validation_errors = sorted(
                validator.iter_errors(ground_truth),
                key=lambda error: list(error.absolute_path),
            )
            if validation_errors:
                raise ValueError(validation_errors[0].message)
            rows.append(
                {
                    "record_id": str(_row_value(row, "record_id") or index),
                    "input_text": input_text,
                    "ground_truth": ground_truth,
                }
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Row {index}: {str(exc)[:180]}")
    return rows, warnings


def _row_to_run(row: dict) -> EvaluationRunResponse:
    metrics = row.get("aggregate_metrics") or {}
    return EvaluationRunResponse(
        run_id=row["id"],
        name=row["name"],
        status=row["status"],
        table_name=row["table_name"],
        input_column=row["input_column"],
        ground_truth_column=row["ground_truth_column"],
        identifier_column=row.get("identifier_column"),
        model=row["model"],
        preset_id=row["preset_id"],
        json_schema=row["schema_json"],
        system_prompt=row.get("system_prompt", ""),
        row_limit=row["row_limit"],
        enable_groundedness=row.get("enable_groundedness", False),
        enable_completeness=row.get("enable_completeness", False),
        total_rows=row.get("total_rows", 0),
        completed_rows=row.get("completed_rows", 0),
        succeeded_rows=row.get("succeeded_rows", 0),
        failed_rows=row.get("failed_rows", 0),
        mlflow_experiment_id=row.get("mlflow_experiment_id"),
        mlflow_run_id=row.get("mlflow_run_id"),
        metrics=EvaluationMetrics(**metrics),
        error_code=row.get("error_code"),
        error_message=row.get("error_message"),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        updated_at=row["updated_at"],
    )


def _row_to_result(row: dict) -> EvaluationResultSummary:
    return EvaluationResultSummary(
        record_id=row["record_id"],
        status=row["status"],
        latency_ms=row.get("latency_ms"),
        matched_fields=row.get("matched_fields", 0),
        total_fields=row.get("total_fields", 0),
        mismatched_paths=row.get("mismatched_paths") or [],
        schema_valid=row.get("schema_valid", False),
        mlflow_trace_id=row.get("mlflow_trace_id"),
        error_message=row.get("error_message"),
    )


def _mlflow_url(run: EvaluationRunResponse) -> str | None:
    if not run.mlflow_experiment_id or not run.mlflow_run_id:
        return None
    return (
        f"{_workspace_host()}/ml/experiments/{run.mlflow_experiment_id}"
        f"/runs/{run.mlflow_run_id}"
    )


def _create_mlflow_run(
    run_id: str,
    owner_id: str,
    body: EvaluationCreateRequest,
    rows: list[dict],
) -> tuple[str, str]:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if not experiment_id:
        experiment = mlflow.set_experiment(EVALUATION_EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id

    client = MlflowClient()
    mlflow_run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": body.name,
            "clinical_extractor.evaluation_id": run_id,
            "clinical_extractor.owner": owner_id,
            "clinical_extractor.model": body.model,
            "clinical_extractor.table": body.table_name,
            "clinical_extractor.preset": body.preset_id,
        },
    )
    mlflow_run_id = mlflow_run.info.run_id

    config = body.model_dump(by_alias=True)
    snapshot = [
        {
            "record_id": row["record_id"],
            "input_text": row["input_text"],
            "ground_truth": row["ground_truth"],
        }
        for row in rows
    ]
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        config_path = directory_path / "configuration.json"
        snapshot_path = directory_path / "input_snapshot.jsonl"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        snapshot_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in snapshot),
            encoding="utf-8",
        )
        client.log_artifact(mlflow_run_id, str(config_path), "evaluation")
        client.log_artifact(mlflow_run_id, str(snapshot_path), "evaluation")
    return experiment_id, mlflow_run_id


def _mlflow_scorers(schema: dict, body: EvaluationCreateRequest):
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    validator = Draft202012Validator(schema)

    @scorer
    def schema_validity(outputs):
        valid = not list(validator.iter_errors(outputs))
        return Feedback(value=valid, rationale="Output matches the configured JSON Schema." if valid else "Output does not match the configured JSON Schema.")

    @scorer
    def field_accuracy(outputs, expectations):
        matched, total, mismatches = compare_json(
            outputs,
            expectations.get("ground_truth", {}),
        )
        return Feedback(
            value=matched / total if total else 1.0,
            rationale=(
                "All fields match ground truth."
                if not mismatches
                else f"Mismatched paths: {', '.join(mismatches[:12])}"
            ),
        )

    @scorer
    def fully_correct(outputs, expectations):
        matched, total, _ = compare_json(
            outputs,
            expectations.get("ground_truth", {}),
        )
        return matched == total

    scorers = [schema_validity, field_accuracy, fully_correct]
    if body.enable_groundedness or body.enable_completeness:
        from mlflow.genai.judges import make_judge

        judge_model = os.environ.get(
            "EVALUATION_JUDGE_MODEL",
            "databricks:/system.ai.gpt-5-4-mini",
        )
        if body.enable_groundedness:
            scorers.append(
                make_judge(
                    name="groundedness",
                    instructions="""
                    Determine whether every non-null value in {{ outputs }} is directly
                    supported by the source text in {{ inputs }}. Respond with exactly
                    true when all values are supported, otherwise false.
                    """,
                    model=judge_model,
                    feedback_value_type=bool,
                )
            )
        if body.enable_completeness:
            scorers.append(
                make_judge(
                    name="completeness",
                    instructions="""
                    Compare {{ outputs }} with {{ expectations }}. Respond with exactly
                    true when the prediction captures all material non-null ground-truth
                    facts, otherwise false.
                    """,
                    model=judge_model,
                    feedback_value_type=bool,
                )
            )
    return scorers


def _trace_record_id(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            return _trace_record_id(json.loads(value))
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        record_id = value.get("record_id")
        if record_id is not None:
            return str(record_id)
        for key in ("inputs", "request"):
            if key in value and (record_id := _trace_record_id(value[key])):
                return record_id
    return None


def _mlflow_trace_ids(run_id: str) -> dict[str, str]:
    import mlflow

    try:
        traces = mlflow.search_traces(run_id=run_id)
    except Exception:
        logger.exception("Unable to load trace IDs for MLflow run %s", run_id)
        return {}

    trace_ids: dict[str, str] = {}
    for _, trace in traces.iterrows():
        record_id = _trace_record_id(trace.get("request"))
        trace_id = trace.get("trace_id")
        if record_id and trace_id:
            trace_ids[record_id] = str(trace_id)
    return trace_ids


def _run_mlflow_evaluation(
    mlflow_run_id: str,
    body: EvaluationCreateRequest,
    records: list[dict],
    metrics: dict,
    output_rows: list[dict],
) -> tuple[dict, dict[str, str]]:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    scorers = _mlflow_scorers(json.loads(body.json_schema), body)
    with mlflow.start_run(run_id=mlflow_run_id):
        evaluation_result = mlflow.genai.evaluate(
            data=records,
            scorers=scorers,
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            report_path = directory_path / "report.json"
            outputs_path = directory_path / "outputs.jsonl"
            report_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            outputs_path.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False) for row in output_rows
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path), "evaluation")
            mlflow.log_artifact(str(outputs_path), "evaluation")

    judge_metrics = {
        key: value
        for key, value in (evaluation_result.metrics or {}).items()
        if "groundedness" in key or "completeness" in key
    }
    return judge_metrics, _mlflow_trace_ids(mlflow_run_id)


def _terminate_mlflow_run(run_id: str, status: str) -> None:
    from mlflow import MlflowClient

    MlflowClient().set_terminated(run_id, status=status)


def _aggregate_metrics(results: list[dict], field_counts: dict[str, dict[str, int]]) -> dict:
    total = len(results)
    succeeded = [result for result in results if result["status"] == "completed"]
    total_fields = sum(result["total_fields"] for result in results)
    matched_fields = sum(result["matched_fields"] for result in results)
    latencies = [
        result["latency_ms"]
        for result in succeeded
        if isinstance(result.get("latency_ms"), (int, float))
    ]
    return {
        "overall_field_accuracy": matched_fields / total_fields if total_fields else 0.0,
        "fully_correct_percentage": (
            sum(result["matched_fields"] == result["total_fields"] for result in results)
            / total
            if total
            else 0.0
        ),
        "schema_valid_percentage": (
            sum(bool(result["schema_valid"]) for result in results) / total
            if total
            else 0.0
        ),
        "inference_success_rate": len(succeeded) / total if total else 0.0,
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "field_scores": {
            path: counts["matched"] / counts["total"]
            for path, counts in sorted(field_counts.items())
            if counts["total"]
        },
        "judge_metrics": {},
    }


async def _execute_evaluation(
    run_id: str,
    body: EvaluationCreateRequest,
    rows: list[dict],
    mlflow_run_id: str,
):
    try:
        async with _evaluation_semaphore:
            database.update_evaluation_run(
                run_id,
                status="running",
                started_at=datetime.now(timezone.utc),
            )
            validator = Draft202012Validator(json.loads(body.json_schema))
            results = []
            records = []
            output_rows = []
            field_counts: dict[str, dict[str, int]] = {}
            succeeded_rows = 0
            failed_rows = 0

            for row in rows:
                if database.evaluation_cancel_requested(run_id):
                    database.update_evaluation_run(
                        run_id,
                        status="cancelled",
                        completed_at=datetime.now(timezone.utc),
                    )
                    await asyncio.to_thread(
                        _terminate_mlflow_run,
                        mlflow_run_id,
                        "KILLED",
                    )
                    return

                inference = await run_inference(
                    InferenceRequest(
                        prompt=row["input_text"],
                        schema_json=body.json_schema,
                        system_prompt=body.system_prompt,
                        model=body.model,
                    )
                )
                if isinstance(inference, InferenceResponse):
                    prediction = inference.prediction
                    matched, total_fields, mismatches = compare_json(
                        prediction,
                        row["ground_truth"],
                    )
                    schema_valid = not list(validator.iter_errors(prediction))
                    result = {
                        "run_id": run_id,
                        "record_id": row["record_id"],
                        "status": "completed",
                        "latency_ms": inference.latency_ms,
                        "matched_fields": matched,
                        "total_fields": total_fields,
                        "mismatched_paths": mismatches,
                        "schema_valid": schema_valid,
                        "mlflow_trace_id": None,
                        "error_message": None,
                    }
                    actual = _flatten(_canonicalize(prediction))
                    expected = _flatten(_canonicalize(row["ground_truth"]))
                    for path in set(actual) | set(expected):
                        counts = field_counts.setdefault(path, {"matched": 0, "total": 0})
                        counts["total"] += 1
                        if actual.get(path) == expected.get(path):
                            counts["matched"] += 1
                    succeeded_rows += 1
                    records.append(
                        {
                            "inputs": {
                                "record_id": row["record_id"],
                                "source_text": row["input_text"],
                            },
                            "outputs": prediction,
                            "expectations": {
                                "ground_truth": row["ground_truth"],
                                "expected_response": json.dumps(
                                    row["ground_truth"], ensure_ascii=False
                                ),
                            },
                        }
                    )
                    output_rows.append(
                        {
                            "record_id": row["record_id"],
                            "prediction": prediction,
                            "ground_truth": row["ground_truth"],
                            "mismatched_paths": mismatches,
                        }
                    )
                else:
                    try:
                        payload = json.loads(inference.body)
                    except (AttributeError, TypeError, json.JSONDecodeError):
                        payload = {}
                    detail = payload.get("message") or payload.get("detail") or "Inference failed"
                    expected_paths = sorted(
                        _flatten(_canonicalize(row["ground_truth"])).keys()
                    )
                    result = {
                        "run_id": run_id,
                        "record_id": row["record_id"],
                        "status": "failed",
                        "latency_ms": None,
                        "matched_fields": 0,
                        "total_fields": len(expected_paths),
                        "mismatched_paths": expected_paths,
                        "schema_valid": False,
                        "mlflow_trace_id": None,
                        "error_message": str(detail)[:500],
                    }
                    for path in expected_paths:
                        counts = field_counts.setdefault(path, {"matched": 0, "total": 0})
                        counts["total"] += 1
                    failed_rows += 1

                results.append(result)
                database.upsert_evaluation_result(result)
                database.update_evaluation_run(
                    run_id,
                    completed_rows=len(results),
                    succeeded_rows=succeeded_rows,
                    failed_rows=failed_rows,
                )

            metrics = _aggregate_metrics(results, field_counts)
            if records:
                judge_metrics, trace_ids = await asyncio.to_thread(
                    _run_mlflow_evaluation,
                    mlflow_run_id,
                    body,
                    records,
                    metrics,
                    output_rows,
                )
                metrics["judge_metrics"] = judge_metrics
                for result in results:
                    trace_id = trace_ids.get(result["record_id"])
                    if trace_id:
                        result["mlflow_trace_id"] = trace_id
                        database.upsert_evaluation_result(result)
            database.update_evaluation_run(
                run_id,
                status="completed",
                aggregate_metrics=metrics,
                completed_at=datetime.now(timezone.utc),
            )
    except asyncio.CancelledError:
        database.update_evaluation_run(
            run_id,
            status="cancelled",
            completed_at=datetime.now(timezone.utc),
        )
        raise
    except Exception as exc:
        logger.exception("Evaluation run %s failed", run_id)
        database.update_evaluation_run(
            run_id,
            status="failed",
            error_code="EVALUATION_FAILED",
            error_message=str(exc)[:500],
            completed_at=datetime.now(timezone.utc),
        )
        try:
            _terminate_mlflow_run(mlflow_run_id, "FAILED")
        except Exception:
            logger.exception("Failed to mark MLflow run %s as failed", mlflow_run_id)
    finally:
        _evaluation_tasks.pop(run_id, None)


async def _create_run(
    request: Request,
    body: EvaluationCreateRequest,
):
    if not database.is_configured():
        return _api_error(
            503,
            "LAKEBASE_REQUIRED",
            "Lakebase must be configured to run evaluations.",
        )
    owner_id = _owner_id(request)
    if database.has_active_evaluation_run(owner_id):
        return _api_error(
            409,
            "EVALUATION_ALREADY_ACTIVE",
            "Finish or cancel the active evaluation before starting another.",
        )
    if body.model not in set(_load_allowed_endpoints()):
        return _api_error(400, "MODEL_NOT_ALLOWED", "The selected model is not available.")
    try:
        preset = next(
            (preset for preset in database.list_presets() if preset.id == body.preset_id),
            None,
        )
    except Exception as exc:
        logger.exception("Failed to resolve evaluation preset")
        return _api_error(
            503,
            "PRESET_LOOKUP_FAILED",
            "Unable to load the selected preset.",
            str(exc)[:300],
        )
    if not preset:
        return _api_error(400, "PRESET_NOT_FOUND", "The selected preset no longer exists.")
    body = body.model_copy(
        update={
            "json_schema": preset.json_schema,
            "system_prompt": preset.system_prompt,
        }
    )
    try:
        rows, warnings = await _read_evaluation_rows(request, body)
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except TimeoutError as exc:
        return _api_error(504, "WAREHOUSE_TIMEOUT", str(exc))
    except Exception as exc:
        logger.exception("Evaluation dataset read failed")
        return _api_error(422, "INVALID_EVALUATION_DATASET", "Unable to read the evaluation dataset.", str(exc)[:300])
    if warnings:
        return _api_error(
            422,
            "INVALID_GROUND_TRUTH",
            "Every selected row must contain valid ground truth matching the preset schema.",
            " · ".join(warnings[:3]),
        )
    if not rows:
        return _api_error(422, "EMPTY_EVALUATION_DATASET", "No usable evaluation rows were found.")

    run_id = str(uuid.uuid4())
    try:
        created = database.create_evaluation_run(
            {
                "id": run_id,
                "owner_id": owner_id,
                "name": body.name,
                "table_name": body.table_name,
                "input_column": body.input_column,
                "ground_truth_column": body.ground_truth_column,
                "identifier_column": body.identifier_column,
                "model": body.model,
                "preset_id": body.preset_id,
                "schema_json": body.json_schema,
                "system_prompt": body.system_prompt,
                "row_limit": body.row_limit,
                "enable_groundedness": body.enable_groundedness,
                "enable_completeness": body.enable_completeness,
                "total_rows": len(rows),
                "mlflow_experiment_id": None,
                "mlflow_run_id": None,
            }
        )
    except Exception as exc:
        if database.has_active_evaluation_run(owner_id):
            return _api_error(
                409,
                "EVALUATION_ALREADY_ACTIVE",
                "Finish or cancel the active evaluation before starting another.",
            )
        logger.exception("Failed to queue evaluation run")
        return _api_error(
            503,
            "EVALUATION_QUEUE_FAILED",
            "Unable to queue the evaluation run.",
            str(exc)[:300],
        )

    try:
        experiment_id, mlflow_run_id = await asyncio.to_thread(
            _create_mlflow_run,
            run_id,
            owner_id,
            body,
            rows,
        )
        created = database.update_evaluation_run(
            run_id,
            mlflow_experiment_id=experiment_id,
            mlflow_run_id=mlflow_run_id,
        )
    except Exception as exc:
        logger.exception("Failed to create MLflow evaluation run")
        database.update_evaluation_run(
            run_id,
            status="failed",
            error_code="MLFLOW_CREATE_FAILED",
            error_message=str(exc)[:500],
            completed_at=datetime.now(timezone.utc),
        )
        return _api_error(502, "EVALUATION_CREATE_FAILED", "Unable to create the MLflow evaluation run.", str(exc)[:300])

    task = asyncio.create_task(_execute_evaluation(run_id, body, rows, mlflow_run_id))
    _evaluation_tasks[run_id] = task
    return _row_to_run(created)


@router.get("/catalogs", response_model=EvaluationDiscoveryResponse)
async def list_catalogs(request: Request):
    try:
        rows = await _user_query(request, "SHOW CATALOGS", row_limit=1000)
        items = sorted(
            _row_value(row, "catalog", "catalog_name")
            for row in rows
            if _row_value(row, "catalog", "catalog_name")
        )
        return EvaluationDiscoveryResponse(items=items)
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except Exception as exc:
        logger.exception("Catalog discovery failed")
        return _api_error(502, "CATALOG_DISCOVERY_FAILED", "Unable to list Unity Catalog catalogs.", str(exc)[:300])


@router.get("/schemas", response_model=EvaluationDiscoveryResponse)
async def list_schemas(request: Request, catalog: str):
    try:
        rows = await _user_query(
            request,
            f"SHOW SCHEMAS IN {_quote_identifier(catalog)}",
            row_limit=1000,
        )
        items = sorted(
            _row_value(row, "databaseName", "namespace", "schema_name")
            for row in rows
            if _row_value(row, "databaseName", "namespace", "schema_name")
        )
        return EvaluationDiscoveryResponse(items=items)
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except Exception as exc:
        logger.exception("Schema discovery failed")
        return _api_error(502, "SCHEMA_DISCOVERY_FAILED", "Unable to list Unity Catalog schemas.", str(exc)[:300])


@router.get("/tables", response_model=EvaluationDiscoveryResponse)
async def list_tables(request: Request, catalog: str, schema: str):
    try:
        rows = await _user_query(
            request,
            f"SHOW TABLES IN {_quote_identifier(catalog)}.{_quote_identifier(schema)}",
            row_limit=1000,
        )
        items = sorted(
            f"{catalog}.{schema}.{table_name}"
            for row in rows
            if (table_name := _row_value(row, "tableName", "table_name"))
        )
        return EvaluationDiscoveryResponse(items=items)
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except Exception as exc:
        logger.exception("Table discovery failed")
        return _api_error(502, "TABLE_DISCOVERY_FAILED", "Unable to list Unity Catalog tables.", str(exc)[:300])


@router.get("/columns", response_model=EvaluationColumnsResponse)
async def list_columns(request: Request, table_name: str):
    try:
        return EvaluationColumnsResponse(
            columns=await _table_columns(request, table_name)
        )
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except Exception as exc:
        logger.exception("Column discovery failed")
        return _api_error(502, "COLUMN_DISCOVERY_FAILED", "Unable to describe the selected table.", str(exc)[:300])


@router.post("/preflight", response_model=EvaluationPreflightResponse)
async def preflight_evaluation(request: Request, body: EvaluationDatasetRequest):
    try:
        rows, warnings = await _read_evaluation_rows(request, body)
        samples = [
            EvaluationPreflightSample(
                record_id=row["record_id"],
                input_preview=row["input_text"][:240],
                ground_truth=row["ground_truth"],
            )
            for row in rows[:3]
        ]
        return EvaluationPreflightResponse(
            row_count=len(rows) + len(warnings),
            valid_rows=len(rows),
            invalid_rows=len(warnings),
            samples=samples,
            warnings=warnings[:10],
        )
    except PermissionError as exc:
        return _api_error(401, "USER_AUTH_REQUIRED", str(exc))
    except Exception as exc:
        logger.exception("Evaluation preflight failed")
        return _api_error(422, "PREFLIGHT_FAILED", "Unable to validate the evaluation dataset.", str(exc)[:300])


@router.get("/runs", response_model=list[EvaluationRunResponse])
async def list_runs(request: Request):
    if not database.is_configured():
        return []
    return [
        _row_to_run(row)
        for row in database.list_evaluation_runs(_owner_id(request))
    ]


@router.post("/runs", response_model=EvaluationRunResponse, status_code=201)
async def create_run(request: Request, body: EvaluationCreateRequest):
    return await _create_run(request, body)


@router.get("/runs/{run_id}", response_model=EvaluationRunResponse)
async def get_run(request: Request, run_id: str):
    if not database.is_configured():
        return _api_error(503, "LAKEBASE_REQUIRED", "Lakebase is not configured.")
    row = database.get_evaluation_run(run_id, _owner_id(request))
    if not row:
        return _api_error(404, "EVALUATION_NOT_FOUND", "Evaluation run not found.")
    return _row_to_run(row)


@router.get("/runs/{run_id}/report", response_model=EvaluationReportResponse)
async def get_report(request: Request, run_id: str):
    if not database.is_configured():
        return _api_error(503, "LAKEBASE_REQUIRED", "Lakebase is not configured.")
    row = database.get_evaluation_run(run_id, _owner_id(request))
    if not row:
        return _api_error(404, "EVALUATION_NOT_FOUND", "Evaluation run not found.")
    run = _row_to_run(row)
    results = [
        _row_to_result(result)
        for result in database.list_evaluation_results(run_id)
    ]
    return EvaluationReportResponse(
        run=run,
        results=results,
        mlflow_url=_mlflow_url(run),
    )


@router.delete("/runs/{run_id}", response_model=EvaluationRunResponse)
async def cancel_run(request: Request, run_id: str):
    if not database.is_configured():
        return _api_error(503, "LAKEBASE_REQUIRED", "Lakebase is not configured.")
    row = database.get_evaluation_run(run_id, _owner_id(request))
    if not row:
        return _api_error(404, "EVALUATION_NOT_FOUND", "Evaluation run not found.")
    if row["status"] in EVALUATION_TERMINAL_STATUSES:
        return _row_to_run(row)
    updated = database.update_evaluation_run(run_id, cancel_requested=True)
    return _row_to_run(updated)


@router.post("/runs/{run_id}/rerun", response_model=EvaluationRunResponse, status_code=201)
async def rerun(request: Request, run_id: str):
    if not database.is_configured():
        return _api_error(503, "LAKEBASE_REQUIRED", "Lakebase is not configured.")
    row = database.get_evaluation_run(run_id, _owner_id(request))
    if not row:
        return _api_error(404, "EVALUATION_NOT_FOUND", "Evaluation run not found.")
    body = EvaluationCreateRequest(
        name=f"{row['name']} rerun",
        table_name=row["table_name"],
        input_column=row["input_column"],
        ground_truth_column=row["ground_truth_column"],
        identifier_column=row.get("identifier_column"),
        model=row["model"],
        preset_id=row["preset_id"],
        schema_json=row["schema_json"],
        system_prompt=row.get("system_prompt", ""),
        row_limit=row["row_limit"],
        enable_groundedness=row.get("enable_groundedness", False),
        enable_completeness=row.get("enable_completeness", False),
    )
    return await _create_run(request, body)
