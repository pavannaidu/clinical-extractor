import asyncio
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from statistics import median

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from databricks.sdk import WorkspaceClient
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
import requests

from .schemas import (
    InferenceRequest,
    InferenceResponse,
    ExtractionCitation,
    ExtractionCitationBox,
    ExtractionEvidence,
    ExtractionEvidenceField,
    LoadTestConfig,
    LoadTestOutputEstimateRequest,
    LoadTestOutputEstimateResponse,
    LoadTestPlanResponse,
    LoadTestRequest,
    LoadTestRequestResult,
    LoadTestResponse,
    LoadTestRunError,
    LoadTestRunProgress,
    LoadTestRunResponse,
    LoadTestSampleResponse,
    LoadTestSeriesPoint,
    LoadTestSummary,
    ErrorResponse,
    ModelInfo,
    ModelRateLimits,
    ModelSelectionEntry,
    ModelSettings,
    ModelSettingsUpdate,
    ModelWarmupResponse,
    Preset,
    PresetCreate,
    PresetUpdate,
    WorkspaceEndpoint,
)
from .presets import PRESETS

logger = logging.getLogger(__name__)

router = APIRouter()

CUSTOM_ENDPOINT = os.environ.get("ENDPOINT_NAME", "medgemma-xgrammar-endpoint-a10")
CONTRACT_PPT_JSON_SCHEMA = "ppt_json_schema"
CONTRACT_CUSTOM_INPUTS_SCHEMA = "custom_inputs_schema"
CONTRACT_OPENAI_GUIDED_JSON = "openai_guided_json"
DEFAULT_MAX_TOKENS = 4096
MAX_INFERENCE_TIMEOUT_S = 110
DEFAULT_INFERENCE_TIMEOUT_S = MAX_INFERENCE_TIMEOUT_S
DEFAULT_TEMPERATURE = 0.1
MAX_RETRIES = 2
RETRY_BACKOFF_S = 1
MODEL_AVAILABILITY_TTL_S = 60
MODEL_WARMUP_TIMEOUT_S = 45
VALID_SERVING_RUNTIMES = {"vllm_beta", "standard"}
SYSTEM_AI_ENTITY_PREFIX = "system.ai."
MODEL_SERVICE_RESOURCE_PREFIX = "model-services/system.ai."
MODEL_SERVICE_CHAT_API = "mlflow/v1/chat/completions"
MODEL_SERVICE_LIST_PATH = (
    "/api/2.1/unity-catalog/model-services?parent=schemas/system.ai"
)
MODEL_SERVICE_CHAT_PATH = "/ai-gateway/mlflow/v1/chat/completions"
AI_EXTRACT_TARGET_ID = "databricks-ai-extract"
AI_EXTRACT_DISPLAY_NAME = "Agent Bricks · Information Extraction"
AI_EXTRACT_PATH = "/api/2.0/ai-functions/ai-extract"
AI_EXTRACT_VERSION = "2.1"
AI_EXTRACT_MAX_FIELDS = 256
AI_EXTRACT_MAX_DEPTH = 12
AI_EXTRACT_MAX_ENUM_VALUES = 500
AI_EXTRACT_MAX_INSTRUCTIONS_LENGTH = 20_000
AI_EXTRACT_MAX_RETRY_DELAY_S = 10.0
_MODEL_SERVICE_NAMES_CACHE: set[str] = set()
CUSTOM_MODEL_UNAVAILABLE_MESSAGE = (
    "This custom endpoint may be scaled down or unavailable right now. Try again."
)
MODEL_AVAILABILITY_UNVERIFIED_MESSAGE = (
    "Model availability could not be verified. Try again."
)
_MODEL_AVAILABILITY_CACHE: dict[str, tuple[float, dict[str, str | None]]] = {}

# Key under which the default model list is stored in app_settings.
MODEL_SELECTION_SETTING_KEY = "model_selection"
OUTPUT_ESTIMATE_SETTING_PREFIX = "load_test_output_estimate"
# In-memory fallback for each user's selection when Lakebase is unavailable.
# The list/None forms remain tolerated for the shared fallback and older tests.
_MODEL_SELECTION_MEMORY: dict[str, list[dict]] | list[dict] | None = {}
_DEFAULT_MODEL_SETTINGS_OWNER = "shared"
_OUTPUT_ESTIMATE_MEMORY: dict[tuple[str, str], dict] = {}

BENCHMARK_TOTAL_REQUESTS = 150
BENCHMARK_MAX_WORKERS = 10
BENCHMARK_MAX_RETRIES = 10
BENCHMARK_MAX_TOKENS = 4096
LOAD_TEST_MAX_REQUESTS = 1000
LOAD_TEST_MAX_BURST_REQUESTS = 150
LOAD_TEST_MAX_WORKERS = 10
LOAD_TEST_MAX_TOKENS = 4096
LOAD_TEST_MAX_PLANNED_DURATION_S = 90.0
DATABRICKS_APP_PROXY_TIMEOUT_S = 120.0
LOAD_TEST_PROXY_SAFETY_MARGIN_S = 5.0
LOAD_TEST_MAX_WALL_TIME_S = (
    DATABRICKS_APP_PROXY_TIMEOUT_S - LOAD_TEST_PROXY_SAFETY_MARGIN_S
)
LOAD_TEST_BACKGROUND_MAX_RUNTIME_S = 15 * 60.0
LOAD_TEST_MAX_ACTIVE_RUNS = 3
LOAD_TEST_RUN_TTL_S = 60 * 60.0
LOAD_TEST_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class _LoadTestRunRecord:
    response: LoadTestRunResponse
    owner_id: str
    task: asyncio.Task[None] | None = None
    results: list[dict] = field(default_factory=list)


_LOAD_TEST_RUNS: dict[str, _LoadTestRunRecord] = {}
_LOAD_TEST_PENDING_CREATIONS = 0
_LOAD_TEST_PENDING_OWNERS: set[str] = set()
_LOAD_TEST_HTTP_THREAD_LOCAL = threading.local()
BENCHMARK_TEMPERATURE = 0.7
BENCHMARK_CACHE_BUST = True
BENCHMARK_BACKOFF_BASE_DELAY_S = 2.0
BENCHMARK_BACKOFF_MAX_DELAY_S = 60.0
FMAPI_LIMITS_DOCS_URL = (
    "https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/"
    "limits#rate-limits-by-model"
)
FMAPI_LIMITS_DOCS_UPDATED_ON = "2026-03-17"
BENCHMARK_SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. Extract only facts explicitly "
    "supported by the note. Return valid JSON that matches the schema exactly."
)
BENCHMARK_SAMPLE_NAME = "Discharge summary extraction sample"
BENCHMARK_SAMPLE_DESCRIPTION = (
    "A fixed follow-up note with diagnoses, medications, allergies, vitals, and follow-up actions."
)
BENCHMARK_PROMPT = """
Extract a structured discharge summary from the clinical note below.

Clinical note:
Patient: Maria Alvarez, 63-year-old female seen on March 22, 2026 for pneumonia follow-up.
Symptoms have improved. Oxygen saturation is 96% on room air. Blood pressure 128/76, heart
rate 82, temperature 98.7 F. Active diagnoses include community-acquired pneumonia improving
and type 2 diabetes mellitus. Current medications are amoxicillin-clavulanate 875/125 mg
twice daily for 5 days remaining, metformin 1000 mg twice daily, and albuterol inhaler
2 puffs every 6 hours as needed. Allergy: lisinopril causes cough. Plan: finish antibiotics,
continue hydration, repeat chest x-ray in 6 weeks, and follow up with primary care next week.
""".strip()
BENCHMARK_SCHEMA_JSON = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "patient_name": {"type": ["string", "null"]},
            "encounter_date": {"type": ["string", "null"]},
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": ["string", "null"]},
                    },
                    "required": ["name", "status"],
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "dosage": {"type": ["string", "null"]},
                        "frequency": {"type": ["string", "null"]},
                    },
                    "required": ["name", "dosage", "frequency"],
                },
            },
            "allergies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "substance": {"type": "string"},
                        "reaction": {"type": ["string", "null"]},
                    },
                    "required": ["substance", "reaction"],
                },
            },
            "vitals": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "blood_pressure": {"type": ["string", "null"]},
                    "heart_rate": {"type": ["integer", "null"]},
                    "temperature_f": {"type": ["number", "null"]},
                    "oxygen_saturation_percent": {"type": ["integer", "null"]},
                },
                "required": [
                    "blood_pressure",
                    "heart_rate",
                    "temperature_f",
                    "oxygen_saturation_percent",
                ],
            },
            "follow_up_actions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "patient_name",
            "encounter_date",
            "diagnoses",
            "medications",
            "allergies",
            "vitals",
            "follow_up_actions",
        ],
    }
)
BENCHMARK_SCHEMA_VALIDATOR = Draft202012Validator(
    json.loads(BENCHMARK_SCHEMA_JSON)
)
# Load model configuration and the active endpoint allowlist.
_MODELS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.json")

def _load_models_config() -> dict | None:
    """Load models.json configuration. Returns None if file missing."""
    try:
        with open(_MODELS_CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to read models.json: %s", e)
        return None


def _load_endpoint_contracts() -> dict[str, str]:
    config = _load_models_config()
    if config is None:
        return {}
    contracts = config.get("contracts", {})
    return contracts if isinstance(contracts, dict) else {}


def _load_endpoint_settings() -> dict[str, dict]:
    config = _load_models_config()
    if config is None:
        return {}
    settings = config.get("settings", {})
    return settings if isinstance(settings, dict) else {}


def _declared_endpoint_map(config: dict | None = None) -> dict[str, dict]:
    """Return canonical metadata for endpoints declared in models.json."""
    if config is None:
        config = _load_models_config()
    if not isinstance(config, dict):
        return {}

    raw_endpoints = config.get("endpoints", [])
    if not isinstance(raw_endpoints, list):
        return {}

    declared: dict[str, dict] = {}
    for entry in raw_endpoints:
        if isinstance(entry, str):
            name = entry
            model_type = _fallback_model_type(name)
            serving_runtime = None
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            name = entry["name"]
            model_type = entry.get("model_type")
            if model_type not in {"custom", "fmapi"}:
                model_type = _fallback_model_type(name)
            serving_runtime = entry.get("serving_runtime")
            if serving_runtime not in VALID_SERVING_RUNTIMES or model_type != "custom":
                serving_runtime = None
        else:
            continue

        if model_type == "fmapi":
            name = _normalize_model_service_name(name)

        if (
            not name
            or name in declared
        ):
            continue
        declared[name] = {
            "name": name,
            "model_type": model_type,
            "serving_runtime": serving_runtime,
        }
    return declared


def _normalize_model_selection(
    selection: list[dict], config: dict | None = None
) -> list[dict]:
    """Sanitize a saved selection and restore server-owned endpoint metadata."""
    declared = _declared_endpoint_map(config)
    normalized: list[dict] = []
    seen: set[str] = set()
    for entry in selection:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        saved_model_type = entry.get("model_type")
        if isinstance(name, str) and (
            saved_model_type == "fmapi"
            or name.startswith(SYSTEM_AI_ENTITY_PREFIX)
            or name.startswith("databricks-")
        ):
            name = _normalize_model_service_name(name)
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 200
            or name in seen
        ):
            continue
        seen.add(name)
        canonical = declared.get(name)
        if canonical is None:
            model_type = (
                saved_model_type
                if saved_model_type in {"custom", "fmapi"}
                else _fallback_model_type(name)
            )
            canonical = {
                "name": name,
                "model_type": model_type,
                "serving_runtime": None,
            }
        normalized.append(dict(canonical))
    return normalized


def _default_selection_from_config(config: dict | None = None) -> list[dict]:
    """Build the default model selection from canonical models.json entries."""
    effective_config = config if config is not None else _load_models_config()
    declared = _declared_endpoint_map(effective_config)
    configured_defaults = (
        effective_config.get("default_endpoints", [])
        if isinstance(effective_config, dict)
        else []
    )
    if isinstance(configured_defaults, list):
        defaults = [
            dict(declared[name])
            for name in configured_defaults
            if isinstance(name, str) and name in declared
        ]
        if defaults:
            return defaults
    return [dict(entry) for entry in declared.values()]


def _model_settings_owner(request: Request | None) -> str:
    """Return the stable identity supplied by the Databricks Apps proxy."""
    if request is None:
        return _DEFAULT_MODEL_SETTINGS_OWNER
    owner = (
        request.headers.get("x-forwarded-email")
        or request.headers.get("x-forwarded-user")
        or _DEFAULT_MODEL_SETTINGS_OWNER
    )
    return owner.strip().lower() or _DEFAULT_MODEL_SETTINGS_OWNER


def _model_selection_setting_key(owner_id: str) -> str:
    if owner_id == _DEFAULT_MODEL_SETTINGS_OWNER:
        return MODEL_SELECTION_SETTING_KEY
    digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    return f"{MODEL_SELECTION_SETTING_KEY}:{digest}"


def _memory_model_selection(owner_id: str) -> list[dict] | None:
    if isinstance(_MODEL_SELECTION_MEMORY, list):
        return (
            _MODEL_SELECTION_MEMORY
            if owner_id == _DEFAULT_MODEL_SETTINGS_OWNER
            else None
        )
    if isinstance(_MODEL_SELECTION_MEMORY, dict):
        return _MODEL_SELECTION_MEMORY.get(owner_id)
    return None


def _cache_model_selection(owner_id: str, selection: list[dict]) -> None:
    global _MODEL_SELECTION_MEMORY
    if owner_id == _DEFAULT_MODEL_SETTINGS_OWNER and not isinstance(
        _MODEL_SELECTION_MEMORY, dict
    ):
        _MODEL_SELECTION_MEMORY = selection
        return
    if not isinstance(_MODEL_SELECTION_MEMORY, dict):
        previous = (
            _MODEL_SELECTION_MEMORY
            if isinstance(_MODEL_SELECTION_MEMORY, list)
            else None
        )
        _MODEL_SELECTION_MEMORY = {}
        if previous:
            _MODEL_SELECTION_MEMORY[_DEFAULT_MODEL_SETTINGS_OWNER] = previous
    _MODEL_SELECTION_MEMORY[owner_id] = selection


def _get_model_selection(
    owner_id: str = _DEFAULT_MODEL_SETTINGS_OWNER,
) -> tuple[list[dict], str]:
    """Return the canonical active selection and its persistence source.

    Prefers the persisted selection (Lakebase, then in-memory fallback), and
    falls back to the defaults seeded from models.json.
    """
    config = _load_models_config()
    db = _get_db_module()
    if db is not None:
        try:
            saved = db.get_setting(_model_selection_setting_key(owner_id))
            if isinstance(saved, list):
                normalized = _normalize_model_selection(saved, config)
                if normalized:
                    _cache_model_selection(owner_id, normalized)
                    return normalized, "saved"
                logger.warning(
                    "Ignoring a saved model selection with no valid endpoint names"
                )
        except Exception:
            logger.exception("Failed to read model selection from Lakebase")

    memory_selection = _memory_model_selection(owner_id)
    if memory_selection is not None:
        normalized = _normalize_model_selection(memory_selection, config)
        if normalized:
            _cache_model_selection(owner_id, normalized)
            return normalized, "memory"

    return _default_selection_from_config(config), "default"


def _load_allowed_endpoints() -> list[str]:
    """Return chat endpoints the app is allowed to invoke.

    Model selection is a user-interface preference, not a security boundary.
    Keep inference authorization tied to live workspace visibility so each
    browser can maintain its own selected subset without mutating global app
    state.
    """
    try:
        workspace_client = _get_workspace_client()
        serving_endpoints = list(workspace_client.serving_endpoints.list())
        custom_endpoints = [
            endpoint.name
            for endpoint in serving_endpoints
            if endpoint.name
            and not endpoint.name.startswith(SYSTEM_AI_ENTITY_PREFIX)
            and not endpoint.name.startswith("databricks-")
            and _is_chat_endpoint(endpoint)
        ]
        try:
            model_services = _list_model_service_names(workspace_client)
        except Exception:
            logger.exception("Failed to discover system model services")
            model_services = {
                _normalize_model_service_name(endpoint.name)
                for endpoint in serving_endpoints
                if endpoint.name
                and endpoint.name.startswith("databricks-")
                and _is_chat_endpoint(endpoint)
            }
        if not model_services:
            model_services = {
                _normalize_model_service_name(endpoint.name)
                for endpoint in serving_endpoints
                if endpoint.name
                and endpoint.name.startswith("databricks-")
                and _is_chat_endpoint(endpoint)
            }
        return custom_endpoints + sorted(model_services)
    except Exception:
        logger.exception("Failed to discover default inference endpoints")
        selection, _source = _get_model_selection()
        return [
            entry["name"]
            for entry in selection
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and entry["name"]
        ]


def _save_model_selection(
    selection: list[dict], owner_id: str = _DEFAULT_MODEL_SETTINGS_OWNER
) -> str:
    """Persist the selection. Returns 'saved' (Lakebase) or 'memory' (fallback)."""
    db = _get_db_module()
    if db is not None:
        try:
            db.set_setting(_model_selection_setting_key(owner_id), selection)
            _cache_model_selection(owner_id, selection)
            return "saved"
        except Exception:
            logger.exception("Failed to persist model selection to Lakebase")

    _cache_model_selection(owner_id, selection)
    return "memory"


def _error_response(status: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body = ErrorResponse(error_code=code, message=message, detail=detail)
    return JSONResponse(status_code=status, content=body.model_dump())


def _strip_json_markdown_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped

    opening_line_end = stripped.find("\n")
    if opening_line_end == -1:
        return stripped

    opening_fence = stripped[:opening_line_end].strip().lower()
    if opening_fence not in {"```", "```json"}:
        return stripped

    return stripped[opening_line_end + 1 : -3].strip()


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON")


def _format_schema_validation_detail(error: JsonSchemaValidationError) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f"[{json.dumps(str(part))}]"

    if error.validator in {"required", "additionalProperties"}:
        reason = error.message
    elif error.validator == "type":
        reason = f"expected type {error.validator_value!r}"
    else:
        reason = f"{error.validator} constraint was not satisfied"
    return f"At {path}: {reason}"


def _normalize_choice_content(content):
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(text_parts) if text_parts else json.dumps(content)
    return content


def _extract_choice_message_content(resp_body):
    if not isinstance(resp_body, dict):
        return str(resp_body)

    choices = resp_body.get("choices", [])
    if not choices:
        return str(resp_body)

    content = choices[0].get("message", {}).get("content", "")
    return _normalize_choice_content(content)


def _is_rate_limit_error(exc: Exception) -> bool:
    error_text = str(exc)
    error_code = getattr(exc, "error_code", None)
    return (
        getattr(exc, "status_code", None) == 429
        or type(exc).__name__ in {"TooManyRequests", "RequestLimitExceeded"}
        or error_code == "REQUEST_LIMIT_EXCEEDED"
        or "429" in error_text
        or "REQUEST_LIMIT_EXCEEDED" in error_text
    )


def _rate_limit_entry(
    docs_model_name: str,
    *,
    itpm: int | None = None,
    otpm: int | None = None,
    qph: int | None = None,
) -> ModelRateLimits:
    return ModelRateLimits(
        docs_model_name=docs_model_name,
        itpm=itpm,
        otpm=otpm,
        qph=qph,
        docs_url=FMAPI_LIMITS_DOCS_URL,
        docs_updated_on=FMAPI_LIMITS_DOCS_UPDATED_ON,
    )


FMAPI_RATE_LIMITS_BY_MODEL: dict[str, ModelRateLimits] = {
    "databricks-gpt-5-4": _rate_limit_entry("GPT-5.4", itpm=50_000, otpm=5_000, qph=360_000),
    "databricks-gpt-5-4-mini": _rate_limit_entry(
        "GPT-5.4 mini", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-4-nano": _rate_limit_entry(
        "GPT-5.4 nano", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-3-codex": _rate_limit_entry(
        "GPT-5.3 Codex", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-2-codex": _rate_limit_entry(
        "GPT-5.2 Codex", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-2": _rate_limit_entry("GPT-5.2", itpm=50_000, otpm=5_000, qph=360_000),
    "databricks-gpt-5-1": _rate_limit_entry("GPT-5.1", itpm=50_000, otpm=5_000, qph=360_000),
    "databricks-gpt-5-1-codex-max": _rate_limit_entry(
        "GPT-5.1 Codex Max", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-1-codex-mini": _rate_limit_entry(
        "GPT-5.1 Codex Mini", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5": _rate_limit_entry("GPT-5", itpm=50_000, otpm=5_000, qph=360_000),
    "databricks-gpt-5-mini": _rate_limit_entry(
        "GPT-5 mini", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gpt-5-nano": _rate_limit_entry(
        "GPT-5 nano", itpm=50_000, otpm=5_000, qph=360_000
    ),
    "databricks-gemini-3-1-pro": _rate_limit_entry(
        "Gemini 3.1 Pro Preview", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-gemini-3-1-flash-lite": _rate_limit_entry(
        "Gemini 3.1 Flash Lite", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-gemini-3-pro": _rate_limit_entry(
        "Gemini 3 Pro Preview", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-gemini-3-flash": _rate_limit_entry(
        "Gemini 3 Flash", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-gemini-2-5-pro": _rate_limit_entry(
        "Gemini 2.5 Pro", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-gemini-2-5-flash": _rate_limit_entry(
        "Gemini 2.5 Flash", itpm=200_000, otpm=20_000, qph=360_000
    ),
    "databricks-qwen3-next-80b-a3b-instruct": _rate_limit_entry(
        "Qwen3-Next 80B A3B Instruct (Beta)", itpm=200_000, otpm=10_000
    ),
    "databricks-gpt-oss-120b": _rate_limit_entry("GPT OSS 120B", itpm=200_000, otpm=10_000),
    "databricks-gpt-oss-20b": _rate_limit_entry("GPT OSS 20B", itpm=200_000, otpm=10_000),
    "databricks-gemma-3-12b": _rate_limit_entry(
        "Gemma 3 12B", itpm=200_000, otpm=10_000, qph=7_200
    ),
    "databricks-llama-4-maverick": _rate_limit_entry(
        "Llama 4 Maverick", itpm=200_000, otpm=10_000, qph=2_400
    ),
    "databricks-llama-3-3-70b-instruct": _rate_limit_entry(
        "Llama 3.3 70B Instruct", itpm=200_000, otpm=10_000, qph=2_400
    ),
    "databricks-llama-3-1-8b-instruct": _rate_limit_entry(
        "Llama 3.1 8B Instruct", itpm=200_000, otpm=10_000, qph=7_200
    ),
    "databricks-llama-3-1-405b-instruct": _rate_limit_entry(
        "Llama 3.1 405B Instruct", itpm=5_000, otpm=500, qph=1_200
    ),
    "databricks-claude-3-7-sonnet": _rate_limit_entry(
        "Claude 3.7 Sonnet", itpm=50_000, otpm=5_000
    ),
    "databricks-claude-sonnet-4": _rate_limit_entry(
        "Claude Sonnet 4", itpm=50_000, otpm=5_000
    ),
    "databricks-claude-opus-4-1": _rate_limit_entry(
        "Claude Opus 4.1", itpm=50_000, otpm=5_000
    ),
    "databricks-claude-opus-4-6": _rate_limit_entry(
        "Claude Opus 4.6", itpm=200_000, otpm=20_000
    ),
    "databricks-claude-opus-4-5": _rate_limit_entry(
        "Claude Opus 4.5", itpm=200_000, otpm=20_000
    ),
    "databricks-claude-sonnet-4-6": _rate_limit_entry(
        "Claude Sonnet 4.6", itpm=200_000, otpm=20_000
    ),
    "databricks-claude-sonnet-4-5": _rate_limit_entry(
        "Claude Sonnet 4.5", itpm=50_000, otpm=5_000
    ),
    "databricks-claude-haiku-4-5": _rate_limit_entry(
        "Claude Haiku 4.5", itpm=50_000, otpm=5_000
    ),
}


def _lookup_rate_limits(endpoint_name: str) -> ModelRateLimits | None:
    service_name = _normalize_model_service_name(endpoint_name)
    legacy_endpoint_name = f"databricks-{service_name}"
    rate_limits = FMAPI_RATE_LIMITS_BY_MODEL.get(legacy_endpoint_name)
    if rate_limits is not None:
        return rate_limits

    if service_name.startswith("gpt-5"):
        return _rate_limit_entry("GPT-5 family", itpm=50_000, otpm=5_000, qph=360_000)
    if service_name.startswith("gemini-"):
        return _rate_limit_entry("Gemini family", itpm=200_000, otpm=20_000, qph=360_000)
    if service_name.startswith("gpt-oss-"):
        return _rate_limit_entry("GPT OSS family", itpm=200_000, otpm=10_000)
    return None


def _lookup_target_otpm(endpoint_name: str) -> int | None:
    rate_limits = _lookup_rate_limits(endpoint_name)
    return rate_limits.otpm if rate_limits is not None else None


def _get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def _normalize_model_service_name(name: str) -> str:
    """Return the short service id shown under system.ai Services."""
    normalized = name.strip()
    if normalized.startswith(MODEL_SERVICE_RESOURCE_PREFIX):
        normalized = normalized[len(MODEL_SERVICE_RESOURCE_PREFIX) :]
    elif normalized.startswith(SYSTEM_AI_ENTITY_PREFIX):
        normalized = normalized[len(SYSTEM_AI_ENTITY_PREFIX) :]
    if normalized.startswith("databricks-"):
        normalized = normalized[len("databricks-") :]
    return normalized


def _model_service_fqn(name: str) -> str:
    return f"{SYSTEM_AI_ENTITY_PREFIX}{_normalize_model_service_name(name)}"


def _fallback_model_type(endpoint_name: str) -> str:
    if (
        endpoint_name.startswith("databricks-")
        or endpoint_name.startswith(SYSTEM_AI_ENTITY_PREFIX)
        or endpoint_name.startswith(MODEL_SERVICE_RESOURCE_PREFIX)
        or endpoint_name in _MODEL_SERVICE_NAMES_CACHE
    ):
        return "fmapi"
    return "custom"


def _is_model_service_name(name: str) -> bool:
    normalized = _normalize_model_service_name(name)
    if name.startswith(("databricks-", SYSTEM_AI_ENTITY_PREFIX, MODEL_SERVICE_RESOURCE_PREFIX)):
        return True
    if normalized in _MODEL_SERVICE_NAMES_CACHE:
        return True
    declared = _declared_endpoint_map()
    return declared.get(normalized, {}).get("model_type") == "fmapi"


def _list_model_service_names(workspace_client: WorkspaceClient) -> set[str]:
    """List chat-capable system.ai model services using the UC service API."""
    response = workspace_client.api_client.do(
        "GET",
        MODEL_SERVICE_LIST_PATH,
        body={},
    )
    raw_services = response.get("model_services", []) if isinstance(response, dict) else []
    names: set[str] = set()
    for service in raw_services:
        if not isinstance(service, dict):
            continue
        supported = service.get("supported_api_types", [])
        if not isinstance(supported, list) or MODEL_SERVICE_CHAT_API not in supported:
            continue
        raw_name = service.get("name")
        if not isinstance(raw_name, str):
            continue
        name = _normalize_model_service_name(raw_name)
        if name and "." not in name and len(name) <= 200:
            names.add(name)
    _MODEL_SERVICE_NAMES_CACHE.clear()
    _MODEL_SERVICE_NAMES_CACHE.update(names)
    return names


def _enum_value(value):
    return getattr(value, "value", value)


def _endpoint_ready(endpoint) -> bool:
    return _enum_value(getattr(getattr(endpoint, "state", None), "ready", None)) == "READY"


def _field_value(value, field: str):
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _endpoint_deployment_stopped(endpoint) -> bool:
    """Return whether any served entity is explicitly stopped.

    The endpoint-level READY flag describes the control plane. Scale-to-zero
    endpoints can still need a runtime ping, while DEPLOYMENT_STOPPED requires
    an operator action and cannot be fixed by inference traffic.
    """
    config = _field_value(endpoint, "config")
    entities = _field_value(config, "served_entities") or _field_value(
        config, "served_models"
    )
    if not entities:
        return False
    for entity in entities:
        state = _field_value(entity, "state")
        deployment = _enum_value(_field_value(state, "deployment"))
        if deployment == "DEPLOYMENT_STOPPED":
            return True
    return False


def _endpoint_task(endpoint) -> str | None:
    value = _enum_value(getattr(endpoint, "task", None))
    return value if isinstance(value, str) else None


def _is_chat_endpoint(endpoint) -> bool:
    # Older SDK test doubles and some legacy custom endpoints omit task. Keep
    # those visible; explicitly incompatible agent/embedding endpoints are not.
    return _endpoint_task(endpoint) in {None, "llm/v1/chat"}


def _canonical_endpoint_entry(
    endpoint_name: str, declared: dict[str, dict] | None = None
) -> dict:
    declared_map = declared if declared is not None else _declared_endpoint_map()
    return dict(
        declared_map.get(endpoint_name)
        or {
            "name": endpoint_name,
            "model_type": _fallback_model_type(endpoint_name),
            "serving_runtime": None,
        }
    )


def _workspace_cache_key(workspace_client: WorkspaceClient) -> str:
    config = getattr(workspace_client, "config", None)
    host = getattr(config, "host", None)
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")

    if host and profile:
        return f"{host}|{profile}"
    return host or profile or "default"


def _list_endpoint_states(workspace_client: WorkspaceClient) -> dict[str, str | None]:
    endpoints = list(workspace_client.serving_endpoints.list())
    return {
        ep.name: _enum_value(getattr(getattr(ep, "state", None), "ready", None))
        for ep in endpoints
        if ep.name
    }


def _get_endpoint_states(workspace_client: WorkspaceClient) -> dict[str, str | None] | None:
    cache_key = _workspace_cache_key(workspace_client)
    cached = _MODEL_AVAILABILITY_CACHE.get(cache_key)
    now = time.time()

    if cached is not None:
        cached_at, cached_endpoint_states = cached
        if now - cached_at < MODEL_AVAILABILITY_TTL_S:
            return dict(cached_endpoint_states)

    try:
        endpoint_states = _list_endpoint_states(workspace_client)
    except Exception:
        logger.exception("Failed to list serving endpoints for model availability")
        return None

    _MODEL_AVAILABILITY_CACHE[cache_key] = (now, endpoint_states)
    return endpoint_states


def _mark_models_availability_unverified(models: list[ModelInfo]) -> list[ModelInfo]:
    return [
        model.model_copy(
            update={
                "availability_status": "unavailable",
                "availability_message": MODEL_AVAILABILITY_UNVERIFIED_MESSAGE,
            }
        )
        for model in models
    ]


def _list_workspace_endpoints(
    workspace_client: WorkspaceClient,
    selected_names: set[str] | None = None,
) -> list[WorkspaceEndpoint]:
    """List chat endpoints visible in the workspace plus configured overrides."""
    declared = _declared_endpoint_map()
    serving_endpoints = list(workspace_client.serving_endpoints.list())
    discovered_custom = {
        endpoint.name: endpoint
        for endpoint in serving_endpoints
        if endpoint.name
        and not endpoint.name.startswith(SYSTEM_AI_ENTITY_PREFIX)
        and not endpoint.name.startswith("databricks-")
        and _is_chat_endpoint(endpoint)
    }
    try:
        model_services = _list_model_service_names(workspace_client)
    except Exception:
        logger.exception("Failed to list system.ai model services")
        model_services = {
            _normalize_model_service_name(endpoint.name)
            for endpoint in serving_endpoints
            if endpoint.name
            and endpoint.name.startswith("databricks-")
            and _is_chat_endpoint(endpoint)
        }
        if not model_services:
            model_services = {
                name
                for name, entry in declared.items()
                if entry.get("model_type") == "fmapi"
            }
    if not model_services:
        model_services = {
            _normalize_model_service_name(endpoint.name)
            for endpoint in serving_endpoints
            if endpoint.name
            and endpoint.name.startswith("databricks-")
            and _is_chat_endpoint(endpoint)
        }
    if not model_services:
        model_services = {
            name
            for name, entry in declared.items()
            if entry.get("model_type") == "fmapi"
        }
    if selected_names is None:
        selection, _source = _get_model_selection()
        selected_names = {
            entry.get("name")
            for entry in selection
            if isinstance(entry, dict)
        }

    results: list[WorkspaceEndpoint] = []
    custom_names = set(discovered_custom) | {
        name
        for name, entry in declared.items()
        if entry.get("model_type") == "custom"
    }
    for name in custom_names:
        entry = _canonical_endpoint_entry(name, declared)
        endpoint = discovered_custom.get(name)
        results.append(
            WorkspaceEndpoint(
                name=name,
                model_type=entry["model_type"],
                ready=endpoint is not None and _endpoint_ready(endpoint),
                serving_runtime=entry["serving_runtime"],
                selected=name in selected_names,
            )
        )
    for name in model_services:
        results.append(
            WorkspaceEndpoint(
                name=name,
                model_type="fmapi",
                ready=True,
                serving_runtime=None,
                selected=name in selected_names,
            )
        )
    results.sort(key=lambda e: (e.model_type != "custom", e.name))
    return results


def _list_models_from_workspace(workspace_client: WorkspaceClient) -> list[ModelInfo]:
    """Return custom endpoints and chat-capable system.ai model services.

    The frontend applies the current browser's persisted selected subset. This
    keeps workspace discovery independent from a user's dropdown preference.
    """
    settings = _load_endpoint_settings()
    models: list[ModelInfo] = []
    for entry in _list_workspace_endpoints(workspace_client, selected_names=set()):
        name = entry.name
        model_type = entry.model_type
        ready = entry.ready
        models.append(
            ModelInfo(
                name=name,
                model_type=model_type,
                max_tokens=int(_resolve_endpoint_options(name, settings)["max_tokens"]),
                serving_runtime=entry.serving_runtime,
                availability_status="ready" if ready else "unavailable",
                availability_message=(
                    None if ready else CUSTOM_MODEL_UNAVAILABLE_MESSAGE
                ),
                rate_limits=(
                    _lookup_rate_limits(name) if model_type == "fmapi" else None
                ),
            )
        )
    return models


def _load_configured_models() -> list[ModelInfo] | None:
    config = _load_models_config()
    if config is None:
        return None

    settings = config.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}

    selection, _source = _get_model_selection()
    models = []
    for entry in selection:
        if not isinstance(entry, dict):
            logger.warning("Skipping invalid model selection entry: %r", entry)
            continue

        name = entry.get("name")
        if not isinstance(name, str) or not name:
            logger.warning("Skipping model selection entry without a valid name: %r", entry)
            continue

        model_type = entry.get("model_type")
        if model_type not in {"custom", "fmapi"}:
            model_type = _fallback_model_type(name)

        serving_runtime = entry.get("serving_runtime")
        if serving_runtime not in VALID_SERVING_RUNTIMES or model_type != "custom":
            serving_runtime = None

        models.append(
            ModelInfo(
                name=name,
                model_type=model_type,
                max_tokens=int(_resolve_endpoint_options(name, settings)["max_tokens"]),
                serving_runtime=serving_runtime,
                availability_status="ready",
                availability_message=None,
                rate_limits=_lookup_rate_limits(name) if model_type == "fmapi" else None,
            )
        )

    return models


def _build_messages(prompt: str, system_prompt: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})
    return messages


class _AiExtractSchemaError(ValueError):
    pass


@dataclass
class _AiExtractSchema:
    schema: dict[str, dict]
    root_description: str
    warnings: list[str]


_AI_EXTRACT_SHAPE_KEYWORDS = {
    "$ref",
    "$dynamicRef",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "prefixItems",
    "contains",
    "unevaluatedItems",
    "unevaluatedProperties",
}
_AI_EXTRACT_SENT_KEYS = {"type", "description", "properties", "items", "enum"}


def _ai_extract_schema_type(value, path: str, *, root: bool = False) -> tuple[str, bool]:
    nullable = False
    if isinstance(value, str):
        concrete_type = value
    elif isinstance(value, list):
        if root or len(value) != 2 or "null" not in value:
            raise _AiExtractSchemaError(
                f"{path}.type must be one supported type with optional null."
            )
        concrete = [item for item in value if item != "null"]
        if len(concrete) != 1 or not isinstance(concrete[0], str):
            raise _AiExtractSchemaError(
                f"{path}.type must be one supported type with optional null."
            )
        concrete_type = concrete[0]
        nullable = True
    else:
        raise _AiExtractSchemaError(f"{path}.type is required for AI Extract.")

    supported = {"object"} if root else {
        "string",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
    }
    if concrete_type not in supported:
        raise _AiExtractSchemaError(
            f"{path}.type '{concrete_type}' is not supported by AI Extract."
        )
    return concrete_type, nullable


def _convert_json_schema_for_ai_extract(schema_json: str) -> _AiExtractSchema:
    parsed = json.loads(schema_json)
    if not isinstance(parsed, dict):
        raise _AiExtractSchemaError("The schema root must be an object.")
    root_type, _ = _ai_extract_schema_type(parsed.get("type"), "$", root=True)
    if root_type != "object":
        raise _AiExtractSchemaError("The schema root must have type object.")

    field_count = 0
    nullable_seen = False
    required_seen = False
    validation_only_keys: set[str] = set()
    enum_value_count = 0

    def reject_shape_keywords(node: dict, path: str) -> None:
        found = sorted(_AI_EXTRACT_SHAPE_KEYWORDS.intersection(node))
        if found:
            raise _AiExtractSchemaError(
                f"{path} uses unsupported JSON Schema keyword '{found[0]}'."
            )

    def record_omitted_keywords(node: dict) -> None:
        for key in node:
            if key not in _AI_EXTRACT_SENT_KEYS and key not in _AI_EXTRACT_SHAPE_KEYWORDS:
                validation_only_keys.add(key)

    def convert_object(node: dict, path: str, depth: int) -> dict[str, dict]:
        nonlocal field_count, nullable_seen, required_seen
        if depth > AI_EXTRACT_MAX_DEPTH:
            raise _AiExtractSchemaError(
                f"{path} exceeds AI Extract's {AI_EXTRACT_MAX_DEPTH}-level nesting limit."
            )
        reject_shape_keywords(node, path)
        record_omitted_keywords(node)
        if "required" in node:
            required_seen = True
        properties = node.get("properties")
        if not isinstance(properties, dict):
            raise _AiExtractSchemaError(f"{path}.properties must be an object.")

        converted: dict[str, dict] = {}
        for field_name, field_schema in properties.items():
            field_path = f"{path}.properties[{field_name!r}]"
            if not isinstance(field_name, str) or not field_name:
                raise _AiExtractSchemaError(f"{path} contains an empty field name.")
            if len(field_name) > 150:
                raise _AiExtractSchemaError(
                    f"{field_path} exceeds AI Extract's 150-character field-name limit."
                )
            if not isinstance(field_schema, dict):
                raise _AiExtractSchemaError(f"{field_path} must be an object schema.")
            field_count += 1
            if field_count > AI_EXTRACT_MAX_FIELDS:
                raise _AiExtractSchemaError(
                    f"The schema exceeds AI Extract's {AI_EXTRACT_MAX_FIELDS}-field limit."
                )
            converted[field_name] = convert_field(field_schema, field_path, depth)
        return converted

    def convert_field(node: dict, path: str, depth: int) -> dict:
        nonlocal nullable_seen, enum_value_count
        reject_shape_keywords(node, path)
        record_omitted_keywords(node)
        field_type, nullable = _ai_extract_schema_type(node.get("type"), path)
        nullable_seen = nullable_seen or nullable

        description = node.get("description")
        if description is not None and not isinstance(description, str):
            raise _AiExtractSchemaError(f"{path}.description must be a string.")

        result: dict = {"type": field_type}
        if isinstance(description, str) and description.strip():
            result["description"] = description.strip()

        if "enum" in node:
            if field_type != "string" or not isinstance(node["enum"], list):
                raise _AiExtractSchemaError(
                    f"{path}.enum must be a list of string labels."
                )
            labels = [value for value in node["enum"] if value is not None]
            if not labels or any(not isinstance(value, str) for value in labels):
                raise _AiExtractSchemaError(
                    f"{path}.enum must contain at least one string label."
                )
            enum_value_count += len(labels)
            if enum_value_count > AI_EXTRACT_MAX_ENUM_VALUES:
                raise _AiExtractSchemaError(
                    "The schema exceeds AI Extract's "
                    f"{AI_EXTRACT_MAX_ENUM_VALUES}-enum-value limit."
                )
            result["type"] = "enum"
            result["labels"] = labels
            return result

        if field_type == "object":
            result["properties"] = convert_object(node, path, depth + 1)
        elif field_type == "array":
            items = node.get("items")
            if not isinstance(items, dict):
                raise _AiExtractSchemaError(
                    f"{path}.items must be one object schema; tuple arrays are unsupported."
                )
            result["items"] = convert_field(items, f"{path}.items", depth + 1)
        return result

    converted_schema = convert_object(parsed, "$", 1)
    warnings: list[str] = []
    if required_seen or "additionalProperties" in validation_only_keys:
        warnings.append(
            "Required-property and additional-property rules are validated by the app after extraction."
        )
    if nullable_seen:
        warnings.append(
            "Nullable JSON Schema types are validated after extraction; AI Extract may return null when a value is absent."
        )
    other_keys = sorted(
        validation_only_keys
        - {"required", "additionalProperties", "type", "description", "properties", "items", "enum"}
    )
    if other_keys:
        warnings.append(
            "These JSON Schema keywords are validated after extraction only: "
            + ", ".join(other_keys)
            + "."
        )

    root_description = parsed.get("description")
    return _AiExtractSchema(
        schema=converted_schema,
        root_description=(
            root_description.strip() if isinstance(root_description, str) else ""
        ),
        warnings=warnings,
    )


def _build_ai_extract_request(
    *,
    prompt: str,
    schema_json: str,
    system_prompt: str,
    precision: bool,
    enable_citations: bool = False,
    enable_confidence_scores: bool = False,
) -> tuple[str, dict, list[str]]:
    converted = _convert_json_schema_for_ai_extract(schema_json)
    instruction_parts = [
        value
        for value in (system_prompt.strip(), converted.root_description)
        if value
    ]
    options: dict[str, object] = {
        "version": AI_EXTRACT_VERSION,
        "enable_citations": enable_citations,
        "enable_confidence_scores": enable_confidence_scores,
    }
    if instruction_parts:
        instructions = "\n\n".join(instruction_parts)
        if len(instructions) > AI_EXTRACT_MAX_INSTRUCTIONS_LENGTH:
            raise _AiExtractSchemaError(
                "System instructions plus the root schema description exceed "
                f"AI Extract's {AI_EXTRACT_MAX_INSTRUCTIONS_LENGTH:,}-character limit."
            )
        options["instructions"] = instructions
    if precision:
        options["mode"] = "precision"
    return (
        AI_EXTRACT_PATH,
        {"content": prompt, "schema": converted.schema, "options": options},
        converted.warnings,
    )


def _resolve_endpoint_contract(endpoint_name: str, contracts: dict[str, str] | None = None) -> str:
    contract_map = contracts if contracts is not None else _load_endpoint_contracts()
    if endpoint_name in contract_map:
        return contract_map[endpoint_name]
    if _is_model_service_name(endpoint_name):
        return CONTRACT_PPT_JSON_SCHEMA
    return CONTRACT_CUSTOM_INPUTS_SCHEMA


def _resolve_endpoint_options(
    endpoint_name: str,
    settings: dict[str, dict] | None = None,
    *,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> dict[str, int | float]:
    endpoint_settings = settings if settings is not None else _load_endpoint_settings()
    raw_options = endpoint_settings.get(endpoint_name, {})

    max_tokens = raw_options.get("max_tokens", DEFAULT_MAX_TOKENS)
    timeout_s = raw_options.get("timeout_s", DEFAULT_INFERENCE_TIMEOUT_S)
    temperature = raw_options.get("temperature", DEFAULT_TEMPERATURE)

    if max_tokens_override is not None:
        max_tokens = max_tokens_override
    if temperature_override is not None:
        temperature = temperature_override

    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        timeout_s = DEFAULT_INFERENCE_TIMEOUT_S
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE

    if max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS
    if timeout_s <= 0:
        timeout_s = DEFAULT_INFERENCE_TIMEOUT_S
    if temperature < 0:
        temperature = DEFAULT_TEMPERATURE

    # Databricks Apps terminates proxied requests at 120 seconds. Leave a
    # margin so the API can return a structured timeout response first.
    timeout_s = min(timeout_s, MAX_INFERENCE_TIMEOUT_S)

    return {"max_tokens": max_tokens, "timeout_s": timeout_s, "temperature": temperature}


def _build_inference_request(
    endpoint_name: str,
    prompt: str,
    schema_json: str,
    system_prompt: str = "",
    contracts: dict[str, str] | None = None,
    settings: dict[str, dict] | None = None,
    *,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> tuple[str, dict]:
    contract = _resolve_endpoint_contract(endpoint_name, contracts)
    options = _resolve_endpoint_options(
        endpoint_name,
        settings,
        max_tokens_override=max_tokens_override,
        temperature_override=temperature_override,
    )
    messages = _build_messages(prompt, system_prompt)

    if contract == CONTRACT_PPT_JSON_SCHEMA:
        if _is_model_service_name(endpoint_name):
            return (
                MODEL_SERVICE_CHAT_PATH,
                {
                    "model": _model_service_fqn(endpoint_name),
                    "messages": messages,
                    "max_tokens": options["max_tokens"],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "strict": True,
                            "schema": json.loads(schema_json),
                        },
                    },
                },
            )
        return (
            f"/serving-endpoints/{endpoint_name}/invocations",
            {
                "messages": messages,
                "max_tokens": options["max_tokens"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": json.loads(schema_json),
                    },
                },
            },
        )

    if contract == CONTRACT_CUSTOM_INPUTS_SCHEMA:
        return (
            f"/serving-endpoints/{endpoint_name}/invocations",
            {
                "messages": messages,
                "max_tokens": options["max_tokens"],
                "custom_inputs": {"schema": schema_json},
            },
        )

    if contract == CONTRACT_OPENAI_GUIDED_JSON:
        return (
            "/serving-endpoints/chat/completions",
            {
                "model": endpoint_name,
                "messages": messages,
                "max_tokens": options["max_tokens"],
                "temperature": options["temperature"],
                "guided_json": json.loads(schema_json),
                "guided_decoding_backend": "xgrammar:no-fallback",
            },
        )

    raise ValueError(f"Unsupported endpoint contract '{contract}' for endpoint '{endpoint_name}'")


async def _invoke_endpoint(
    workspace_client: WorkspaceClient,
    route_path: str,
    payload: dict,
    timeout_s: int | float,
):
    return await asyncio.wait_for(
        asyncio.to_thread(
            lambda: workspace_client.api_client.do(
                "POST",
                route_path,
                body=payload,
            )
        ),
        timeout=timeout_s,
    )


class _AiExtractResponseError(ValueError):
    pass


_AI_EXTRACT_FIELD_WRAPPER_KEYS = {
    "value",
    "confidence_score",
    "confidence",
    "citation_ids",
}


def _ai_extract_child_path(path: str, key: str) -> str:
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _unwrap_ai_extract_value(
    value,
    *,
    path: str = "$",
    fields: list[ExtractionEvidenceField] | None = None,
):
    evidence_fields = fields if fields is not None else []
    if (
        isinstance(value, dict)
        and "value" in value
        and set(value).issubset(_AI_EXTRACT_FIELD_WRAPPER_KEYS)
    ):
        clean_value = _unwrap_ai_extract_value(
            value.get("value"),
            path=path,
            fields=evidence_fields,
        )
        confidence = value.get("confidence_score", value.get("confidence"))
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = None
        citation_ids = value.get("citation_ids")
        if not isinstance(citation_ids, list):
            citation_ids = []
        citation_ids = [
            citation_id
            for citation_id in citation_ids
            if isinstance(citation_id, int) and not isinstance(citation_id, bool)
        ]
        evidence_fields.append(
            ExtractionEvidenceField(
                path=path,
                value=clean_value,
                confidence_score=(float(confidence) if confidence is not None else None),
                citation_ids=citation_ids,
            )
        )
        return clean_value
    if isinstance(value, dict):
        return {
            key: _unwrap_ai_extract_value(
                child,
                path=_ai_extract_child_path(path, str(key)),
                fields=evidence_fields,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _unwrap_ai_extract_value(
                child,
                path=f"{path}[{index}]",
                fields=evidence_fields,
            )
            for index, child in enumerate(value)
        ]
    return value


def _parse_ai_extract_citations(metadata: dict) -> list[ExtractionCitation]:
    raw_citations = metadata.get("citations")
    if not isinstance(raw_citations, list):
        return []
    citations: list[ExtractionCitation] = []
    for raw in raw_citations:
        if not isinstance(raw, dict):
            continue
        citation_id = raw.get("id")
        if not isinstance(citation_id, int) or isinstance(citation_id, bool):
            continue
        start = raw.get("start")
        stop = raw.get("stop")
        if not isinstance(start, int) or isinstance(start, bool):
            start = None
        if not isinstance(stop, int) or isinstance(stop, bool):
            stop = None
        boxes: list[ExtractionCitationBox] = []
        if isinstance(raw.get("bbox"), list):
            for raw_box in raw["bbox"]:
                if not isinstance(raw_box, dict):
                    continue
                coord = raw_box.get("coord")
                page_id = raw_box.get("page_id")
                if (
                    isinstance(coord, list)
                    and len(coord) == 4
                    and all(isinstance(item, int) and not isinstance(item, bool) for item in coord)
                    and isinstance(page_id, int)
                    and not isinstance(page_id, bool)
                ):
                    boxes.append(ExtractionCitationBox(coord=coord, page_id=page_id))
        citations.append(
            ExtractionCitation(
                id=citation_id,
                start=start,
                stop=stop,
                bbox=boxes or None,
            )
        )
    return citations


def _parse_ai_extract_response(resp_body) -> tuple[object, ExtractionEvidence]:
    if not isinstance(resp_body, dict):
        raise _AiExtractResponseError("AI Extract returned an invalid response.")
    error_message = resp_body.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        raise _AiExtractResponseError(error_message.strip())
    if "response" not in resp_body:
        raise _AiExtractResponseError("AI Extract returned no extraction response.")
    response = resp_body["response"]
    if not isinstance(response, dict):
        raise _AiExtractResponseError("AI Extract returned an invalid extraction response.")

    fields: list[ExtractionEvidenceField] = []
    # The top-level response is always the schema's field map. Unwrap each
    # field independently so a legitimate root field named "value" is not
    # mistaken for AI Extract's per-field wrapper object.
    prediction = {
        key: _unwrap_ai_extract_value(
            value,
            path=_ai_extract_child_path("$", str(key)),
            fields=fields,
        )
        for key, value in response.items()
    }
    metadata = resp_body.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    evidence = ExtractionEvidence(
        version=(metadata.get("version") if isinstance(metadata.get("version"), str) else None),
        mode=(metadata.get("mode") if isinstance(metadata.get("mode"), str) else None),
        chunk_type=(
            metadata.get("chunk_type")
            if isinstance(metadata.get("chunk_type"), str)
            else None
        ),
        fields=fields,
        citations=_parse_ai_extract_citations(metadata),
    )
    return prediction, evidence


class _LoadTestHttpError(Exception):
    """One non-successful load-test HTTP attempt.

    The normal Databricks SDK transport retries 429 responses internally. That
    is useful for application calls but hides the attempts a load test is meant
    to measure, so the load-test path uses a single-attempt transport instead.
    """

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str | None,
        retry_after_secs: float | None,
        message: str,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after_secs = retry_after_secs


def _load_test_http_session() -> requests.Session:
    session = getattr(_LOAD_TEST_HTTP_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        no_retry_adapter = requests.adapters.HTTPAdapter(max_retries=0)
        session.mount("https://", no_retry_adapter)
        session.mount("http://", no_retry_adapter)
        _LOAD_TEST_HTTP_THREAD_LOCAL.session = session
    return session


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    return parsed if parsed >= 0 else None


async def _invoke_load_test_endpoint_once(
    workspace_client: WorkspaceClient,
    route_path: str,
    payload: dict,
    timeout_s: int | float,
):
    """Send exactly one HTTP request so the caller can observe every 429."""
    config = getattr(workspace_client, "config", None)
    authenticate = getattr(config, "authenticate", None)

    # Unit-test clients and compatible lightweight adapters may only implement
    # api_client.do. Production WorkspaceClient instances always expose a
    # callable config.authenticate method and therefore take the no-retry path.
    if not callable(authenticate):
        return await _invoke_endpoint(
            workspace_client,
            route_path,
            payload,
            timeout_s,
        )

    def send_once():
        headers = dict(authenticate())
        headers.setdefault("Content-Type", "application/json")
        user_agent = getattr(config, "user_agent", None)
        if isinstance(user_agent, str) and user_agent:
            headers.setdefault("User-Agent", user_agent)

        response = _load_test_http_session().post(
            f"{str(config.host).rstrip('/')}{route_path}",
            json=payload,
            headers=headers,
            timeout=float(timeout_s),
        )
        if response.status_code >= 400:
            response_body: dict = {}
            try:
                parsed_body = response.json()
                if isinstance(parsed_body, dict):
                    response_body = parsed_body
            except ValueError:
                pass

            error_code = response_body.get("error_code")
            if not isinstance(error_code, str):
                error_code = None
            message = response_body.get("message")
            if not isinstance(message, str) or not message:
                message = f"Endpoint returned HTTP {response.status_code}."
            raise _LoadTestHttpError(
                status_code=response.status_code,
                error_code=error_code,
                retry_after_secs=_parse_retry_after_seconds(
                    response.headers.get("Retry-After")
                ),
                message=message,
            )

        parsed_response = response.json()
        if isinstance(parsed_response, (dict, list)):
            return parsed_response
        return {}

    return await asyncio.wait_for(
        asyncio.to_thread(send_once),
        timeout=float(timeout_s),
    )


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    error_text = str(exc)
    for candidate in (400, 401, 403, 404, 429, 500, 502, 503, 504):
        if str(candidate) in error_text:
            return candidate
    return None


async def _run_ai_extract_inference(
    req: InferenceRequest,
    workspace_client: WorkspaceClient,
):
    try:
        route_path, payload, schema_warnings = _build_ai_extract_request(
            prompt=req.prompt,
            schema_json=req.json_schema,
            system_prompt=req.system_prompt,
            precision=bool(req.ai_extract_options and req.ai_extract_options.precision),
            enable_citations=bool(
                req.ai_extract_options and req.ai_extract_options.enable_citations
            ),
            enable_confidence_scores=bool(
                req.ai_extract_options
                and req.ai_extract_options.enable_confidence_scores
            ),
        )
    except (json.JSONDecodeError, _AiExtractSchemaError) as exc:
        return _error_response(
            422,
            "AI_EXTRACT_SCHEMA_UNSUPPORTED",
            "This JSON Schema cannot be represented by AI Extract.",
            str(exc),
        )

    resp_body = None
    latency_ms = 0.0
    for attempt in range(1 + MAX_RETRIES):
        try:
            start = time.perf_counter()
            resp_body = await _invoke_load_test_endpoint_once(
                workspace_client,
                route_path,
                payload,
                timeout_s=MAX_INFERENCE_TIMEOUT_S,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            break
        except asyncio.TimeoutError:
            return _error_response(
                504,
                "AI_EXTRACT_TIMEOUT",
                "AI Extract did not respond within the application time limit.",
            )
        except Exception as exc:
            status_code = _exception_status_code(exc)
            retry_after = (
                exc.retry_after_secs
                if isinstance(exc, _LoadTestHttpError)
                else None
            )
            retryable = status_code == 429 or (
                status_code is not None and status_code >= 500
            )
            if retryable and attempt < MAX_RETRIES:
                delay = retry_after if retry_after is not None else RETRY_BACKOFF_S * (attempt + 1)
                await asyncio.sleep(min(max(delay, 0.0), AI_EXTRACT_MAX_RETRY_DELAY_S))
                continue
            if status_code in {401, 403}:
                logger.exception("AI Extract permission denied")
                return _error_response(
                    403,
                    "AI_EXTRACT_PERMISSION_DENIED",
                    "The app is not authorized to use Agent Bricks Information Extraction.",
                    "Grant the app service principal access to AI Functions and system.ai.ai_extract.",
                )
            if status_code == 404:
                logger.exception("AI Extract is unavailable in this workspace")
                return _error_response(
                    503,
                    "AI_EXTRACT_UNAVAILABLE",
                    "Agent Bricks Information Extraction is not available in this workspace.",
                    "Check region support and the workspace preview setting.",
                )
            if status_code == 400:
                logger.exception("AI Extract rejected the request")
                return _error_response(
                    422,
                    "AI_EXTRACT_REQUEST_REJECTED",
                    "AI Extract rejected the extraction request.",
                )
            if status_code == 429:
                return _error_response(
                    429,
                    "AI_EXTRACT_RATE_LIMITED",
                    "AI Extract is currently rate limited. Try again shortly.",
                )
            logger.exception("AI Extract request failed after retries")
            return _error_response(
                502,
                "AI_EXTRACT_ERROR",
                "AI Extract could not complete the request.",
            )

    try:
        prediction, evidence = _parse_ai_extract_response(resp_body)
    except _AiExtractResponseError:
        logger.exception("AI Extract returned an error response")
        return _error_response(
            502,
            "AI_EXTRACT_INVALID_RESPONSE",
            "AI Extract returned an invalid or unsuccessful response.",
        )

    try:
        Draft202012Validator(
            json.loads(req.json_schema),
            format_checker=FormatChecker(),
        ).validate(prediction)
    except JsonSchemaValidationError as exc:
        return _error_response(
            502,
            "MODEL_OUTPUT_SCHEMA_MISMATCH",
            "AI Extract returned JSON that did not match the requested schema.",
            _format_schema_validation_detail(exc),
        )

    return InferenceResponse(
        prediction=prediction,
        latency_ms=round(latency_ms, 1),
        raw_output=json.dumps(prediction, ensure_ascii=False, indent=2, allow_nan=False),
        model=AI_EXTRACT_TARGET_ID,
        runtime="ai_extract",
        schema_warnings=schema_warnings,
        evidence=evidence,
    )


def _coerce_token_count(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _resolve_load_test_target(
    *,
    model_info: ModelInfo,
    target_itpm_override: int | None = None,
    target_otpm_override: int | None = None,
    use_documented_targets: bool = True,
) -> tuple[int | None, int | None, str, str, str]:
    documented_itpm = model_info.rate_limits.itpm if model_info.rate_limits is not None else None
    documented_otpm = model_info.rate_limits.otpm if model_info.rate_limits is not None else None

    def resolve_axis(
        override: int | None,
        documented: int | None,
    ) -> tuple[int | None, str]:
        if override is not None:
            return override, "custom"
        if use_documented_targets and documented is not None:
            return documented, "documented"
        return None, "none"

    target_itpm, target_itpm_source = resolve_axis(
        target_itpm_override,
        documented_itpm,
    )
    target_otpm, target_otpm_source = resolve_axis(
        target_otpm_override,
        documented_otpm,
    )

    sources = {target_itpm_source, target_otpm_source}
    if "custom" in sources:
        target_source = "custom"
    elif "documented" in sources:
        target_source = "documented"
    else:
        target_source = "none"

    return (
        target_itpm,
        target_otpm,
        target_source,
        target_itpm_source,
        target_otpm_source,
    )


def _estimate_load_test_input_tokens(
    *,
    endpoint_name: str,
    endpoint_settings: dict[str, dict],
    prompt: str,
    schema_json: str,
    system_prompt: str,
    max_tokens: int,
    cache_bust: bool,
    total_requests: int,
) -> int:
    """Estimate request input tokens from a representative serialized payload.

    Tokenizers differ across endpoints and are not available to the app. Four
    UTF-8 payload bytes per token is therefore an explicit planning heuristic;
    response-reported usage remains authoritative for achieved throughput.
    """
    representative_prompt = prompt
    if cache_bust:
        largest_req_id = max(total_requests - 1, 0)
        representative_prompt = (
            f"{prompt}\n\n<!-- benchmark req={largest_req_id} nonce={'0' * 32} -->"
        )

    _route_path, payload = _build_inference_request(
        endpoint_name=endpoint_name,
        prompt=representative_prompt,
        schema_json=schema_json,
        system_prompt=system_prompt,
        settings=endpoint_settings,
        max_tokens_override=max_tokens,
        temperature_override=BENCHMARK_TEMPERATURE,
    )
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return max(1, math.ceil(len(serialized_payload) / 4))


def _build_load_test_plan(
    *,
    total_requests: int,
    estimated_input_tokens_per_request: int,
    planned_output_tokens_per_request: int,
    target_itpm: int | None,
    target_otpm: int | None,
    scheduling_mode: str = "paced",
    load_multiplier: float = 1.0,
) -> dict[str, int | float | str | None]:
    """Build burst scheduling or an explicitly requested token-paced plan."""
    if scheduling_mode == "burst":
        return {
            "pacing_mode": "burst",
            "request_interval_s": 0.0,
            "planned_requests_per_minute": None,
            "planned_itpm": None,
            "planned_otpm": None,
            "planned_duration_s": 0.0,
            "limiting_target": "none",
        }

    itpm_interval_s = (
        (60.0 * estimated_input_tokens_per_request) / target_itpm
        if target_itpm is not None
        else 0.0
    )
    otpm_interval_s = (
        (60.0 * planned_output_tokens_per_request) / target_otpm
        if target_otpm is not None
        else 0.0
    )
    request_interval_s = max(itpm_interval_s, otpm_interval_s) / load_multiplier

    if request_interval_s <= 0:
        return {
            "pacing_mode": "burst",
            "request_interval_s": 0.0,
            "planned_requests_per_minute": None,
            "planned_itpm": None,
            "planned_otpm": None,
            "planned_duration_s": 0.0,
            "limiting_target": "none",
        }

    if math.isclose(itpm_interval_s, otpm_interval_s, rel_tol=1e-9, abs_tol=1e-12):
        limiting_target = "both"
    elif itpm_interval_s > otpm_interval_s:
        limiting_target = "itpm"
    else:
        limiting_target = "otpm"

    planned_requests_per_minute = 60.0 / request_interval_s
    return {
        "pacing_mode": "token_target",
        "request_interval_s": request_interval_s,
        "planned_requests_per_minute": planned_requests_per_minute,
        "planned_itpm": (
            estimated_input_tokens_per_request * planned_requests_per_minute
        ),
        "planned_otpm": (
            planned_output_tokens_per_request * planned_requests_per_minute
        ),
        "planned_duration_s": request_interval_s * max(total_requests - 1, 0),
        "limiting_target": limiting_target,
    }


class _LoadTestPacer:
    """Reserve token-budgeted start slots shared by initial calls and retries."""

    def __init__(
        self,
        *,
        request_interval_s: float,
        started_at: float,
        initial_slots: int = 0,
    ):
        self.request_interval_s = request_interval_s
        self.started_at = started_at
        self._next_retry_slot = initial_slots
        self._slots_used = 0
        self._last_actual_start_s: float | None = None
        self._lock = asyncio.Lock()

    @property
    def slots_used(self) -> int:
        return self._slots_used

    @property
    def last_actual_start_s(self) -> float:
        return self._last_actual_start_s or 0.0

    async def wait_until(self, scheduled_start_s: float) -> dict[str, float]:
        return await self._wait_until(scheduled_start_s)

    async def wait_for_slot(self) -> dict[str, float]:
        async with self._lock:
            scheduled_start_s = self._next_retry_slot * self.request_interval_s
            self._next_retry_slot += 1
        return await self._wait_until(scheduled_start_s)

    async def _wait_until(self, scheduled_start_s: float) -> dict[str, float]:
        total_pacing_wait_s = 0.0
        while True:
            async with self._lock:
                before_wait_s = max(0.0, time.perf_counter() - self.started_at)
                minimum_next_start_s = (
                    self._last_actual_start_s + self.request_interval_s
                    if self._last_actual_start_s is not None
                    else 0.0
                )
                enforced_start_s = max(scheduled_start_s, minimum_next_start_s)
                pacing_wait_s = max(0.0, enforced_start_s - before_wait_s)
                if pacing_wait_s <= 0:
                    actual_start_s = before_wait_s
                    self._last_actual_start_s = actual_start_s
                    self._slots_used += 1
                    return {
                        "scheduled_start_s": scheduled_start_s,
                        "actual_start_s": actual_start_s,
                        "start_delay_s": max(0.0, actual_start_s - scheduled_start_s),
                        "pacing_wait_s": total_pacing_wait_s,
                    }

            # Never hold the lock while sleeping. A retry can reserve a slot after
            # every initial request; sleeping under the lock for that far-future
            # slot would otherwise block initial launches scheduled sooner.
            await asyncio.sleep(pacing_wait_s)
            total_pacing_wait_s += pacing_wait_s


async def _run_single_load_test_request(
    *,
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    endpoint_settings: dict[str, dict],
    req_id: int,
    max_tokens: int,
    max_retries: int,
    cache_bust: bool,
    prompt: str = BENCHMARK_PROMPT,
    schema_json: str = BENCHMARK_SCHEMA_JSON,
    system_prompt: str = BENCHMARK_SYSTEM_PROMPT,
    schema_validator: Draft202012Validator | None = None,
    attempt_pacer: _LoadTestPacer | None = None,
    initial_start_timing: dict[str, float] | None = None,
    attempt_semaphore: asyncio.Semaphore | None = None,
    test_started: float | None = None,
    scheduled_start_s: float = 0.0,
):
    effective_prompt = prompt
    if cache_bust:
        effective_prompt = (
            f"{prompt}\n\n<!-- benchmark req={req_id} nonce={uuid.uuid4().hex} -->"
        )

    validator = schema_validator or BENCHMARK_SCHEMA_VALIDATOR

    options = _resolve_endpoint_options(
        endpoint_name,
        endpoint_settings,
        max_tokens_override=max_tokens,
        temperature_override=BENCHMARK_TEMPERATURE,
    )

    logical_request_started: float | None = None
    first_start_timing = {
        "scheduled_start_s": 0.0,
        "actual_start_s": 0.0,
        "start_delay_s": 0.0,
    }
    total_pacing_wait_s = 0.0
    retry_events: list[dict] = []
    observed_prompt_tokens = 0
    observed_completion_tokens = 0
    usage_observed = False
    last_attempt_start_s = 0.0

    def result_timing() -> dict[str, float]:
        return {
            **first_start_timing,
            "pacing_wait_s": round(total_pacing_wait_s, 4),
            "last_attempt_start_s": round(last_attempt_start_s, 4),
        }

    for attempt in range(1, max_retries + 2):
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        if attempt == 1 and initial_start_timing is not None:
            start_timing = initial_start_timing
        elif attempt_pacer is not None:
            start_timing = await attempt_pacer.wait_for_slot()
        elif attempt == 1 and test_started is not None:
            before_wait_s = max(0.0, time.perf_counter() - test_started)
            pacing_wait_s = max(0.0, scheduled_start_s - before_wait_s)
            if pacing_wait_s > 0:
                await asyncio.sleep(pacing_wait_s)
                total_pacing_wait_s += pacing_wait_s
            arrival_s = max(0.0, time.perf_counter() - test_started)
            start_timing = {
                "scheduled_start_s": scheduled_start_s,
                "actual_start_s": arrival_s,
                "start_delay_s": max(0.0, arrival_s - scheduled_start_s),
                "pacing_wait_s": 0.0,
            }
        else:
            start_timing = None

        if start_timing is not None:
            if attempt == 1:
                first_start_timing = {
                    key: round(start_timing[key], 4)
                    for key in (
                        "scheduled_start_s",
                        "actual_start_s",
                        "start_delay_s",
                    )
                }
            total_pacing_wait_s += start_timing["pacing_wait_s"]

        if logical_request_started is None:
            logical_request_started = time.perf_counter()
        try:
            route_path, payload = _build_inference_request(
                endpoint_name=endpoint_name,
                prompt=effective_prompt,
                schema_json=schema_json,
                system_prompt=system_prompt,
                settings=endpoint_settings,
                max_tokens_override=max_tokens,
                temperature_override=BENCHMARK_TEMPERATURE,
            )
            async def invoke_once():
                nonlocal last_attempt_start_s, first_start_timing
                attempt_started_s = (
                    max(0.0, time.perf_counter() - test_started)
                    if test_started is not None
                    else float(start_timing["actual_start_s"])
                    if start_timing is not None
                    else 0.0
                )
                last_attempt_start_s = attempt_started_s
                if attempt == 1:
                    scheduled_s = float(first_start_timing["scheduled_start_s"])
                    first_start_timing["actual_start_s"] = round(attempt_started_s, 4)
                    first_start_timing["start_delay_s"] = round(
                        max(0.0, attempt_started_s - scheduled_s), 4
                    )
                return await _invoke_load_test_endpoint_once(
                    workspace_client,
                    route_path,
                    payload,
                    timeout_s=options["timeout_s"],
                )

            if attempt_semaphore is None:
                resp_body = await invoke_once()
            else:
                async with attempt_semaphore:
                    resp_body = await invoke_once()

            usage = resp_body.get("usage", {}) if isinstance(resp_body, dict) else {}
            prompt_tokens = _coerce_token_count(usage.get("prompt_tokens"))
            completion_tokens = _coerce_token_count(usage.get("completion_tokens"))
            if prompt_tokens is not None and completion_tokens is not None:
                usage_observed = True
                observed_prompt_tokens += prompt_tokens
                observed_completion_tokens += completion_tokens
            content = _extract_choice_message_content(resp_body)
            if isinstance(content, str):
                if not content.strip():
                    raise json.JSONDecodeError("Empty response body", content, 0)
                parsed_content = json.loads(content)
            else:
                parsed_content = content
            validator.validate(parsed_content)

            return {
                "req_id": req_id,
                "status": "ok",
                "attempts": attempt,
                "prompt_tokens": observed_prompt_tokens if usage_observed else None,
                "completion_tokens": observed_completion_tokens if usage_observed else None,
                "response_completion_tokens": completion_tokens,
                "elapsed_s": round(time.perf_counter() - logical_request_started, 2),
                "abs_end_s": 0.0,
                "error": None,
                "retry_events": retry_events,
                **result_timing(),
            }
        except asyncio.TimeoutError:
            return {
                "req_id": req_id,
                "status": "error",
                "attempts": attempt,
                "prompt_tokens": observed_prompt_tokens if usage_observed else None,
                "completion_tokens": observed_completion_tokens if usage_observed else None,
                "elapsed_s": round(time.perf_counter() - logical_request_started, 2),
                "abs_end_s": 0.0,
                "error": "The model endpoint did not respond before the timeout.",
                "retry_events": retry_events,
                **result_timing(),
            }
        except Exception as exc:
            if (
                (
                    _is_rate_limit_error(exc)
                    or isinstance(
                        exc,
                        (json.JSONDecodeError, JsonSchemaValidationError),
                    )
                )
                and attempt <= max_retries
            ):
                delay_s = random.uniform(
                    0,
                    min(
                        BENCHMARK_BACKOFF_BASE_DELAY_S * (2 ** (attempt - 1)),
                        BENCHMARK_BACKOFF_MAX_DELAY_S,
                    ),
                )
                retry_events.append(
                    {
                        "attempt": attempt,
                        "elapsed_s": round(
                            time.perf_counter() - logical_request_started,
                            2,
                        ),
                        "status_code": getattr(exc, "status_code", None),
                        "error_code": getattr(exc, "error_code", None),
                        "retry_after_s": getattr(exc, "retry_after_secs", None),
                        "backoff_s": round(delay_s, 2),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )
                await asyncio.sleep(delay_s)
                continue

            logger.warning(
                "Load-test request %s failed after %s attempt(s)",
                req_id,
                attempt,
                exc_info=True,
            )
            if _is_rate_limit_error(exc):
                safe_error = "Rate-limit retries were exhausted."
            elif isinstance(exc, JsonSchemaValidationError):
                safe_error = "The model output did not match the load-test schema."
            elif isinstance(exc, json.JSONDecodeError):
                safe_error = "The model returned output that was not valid JSON."
            else:
                safe_error = "The model endpoint request failed."

            return {
                "req_id": req_id,
                "status": (
                    "429_exhausted"
                    if _is_rate_limit_error(exc)
                    else "schema_validation_error"
                    if isinstance(exc, JsonSchemaValidationError)
                    else "invalid_json"
                    if isinstance(exc, json.JSONDecodeError)
                    else "error"
                ),
                "attempts": attempt,
                "prompt_tokens": observed_prompt_tokens if usage_observed else None,
                "completion_tokens": observed_completion_tokens if usage_observed else None,
                "elapsed_s": round(time.perf_counter() - logical_request_started, 2),
                "abs_end_s": 0.0,
                "error": safe_error,
                "retry_events": retry_events,
                **result_timing(),
            }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (
        (ordered[upper_index] - ordered[lower_index]) * weight
    )


def _make_load_test_config(
    *,
    total_requests: int,
    max_workers: int,
    max_retries: int,
    max_tokens: int,
    cache_bust: bool,
    use_documented_targets: bool,
    target_itpm: int | None,
    target_otpm: int | None,
    target_source: str,
    target_itpm_source: str,
    target_otpm_source: str,
    estimated_input_tokens_per_request: int,
    planned_output_tokens_per_request: int,
    pacing_plan: dict[str, int | float | str | None],
    scheduling_mode: str = "burst",
    duration_s: int | None = None,
    load_multiplier: float = 1.0,
) -> LoadTestConfig:
    return LoadTestConfig(
        total_requests=total_requests,
        max_workers=max_workers,
        max_retries=max_retries,
        max_tokens=max_tokens,
        cache_bust=cache_bust,
        target_source=target_source,
        use_documented_targets=use_documented_targets,
        target_itpm=target_itpm,
        target_otpm=target_otpm,
        target_itpm_source=target_itpm_source,
        target_otpm_source=target_otpm_source,
        scheduling_mode=("burst" if scheduling_mode == "burst" else "target_rate"),
        duration_s=duration_s if scheduling_mode != "burst" else None,
        load_multiplier=load_multiplier,
        pacing_mode=str(pacing_plan["pacing_mode"]),
        estimated_input_tokens_per_request=estimated_input_tokens_per_request,
        planned_output_tokens_per_request=planned_output_tokens_per_request,
        request_interval_s=round(float(pacing_plan["request_interval_s"]), 4),
        planned_requests_per_minute=(
            round(float(pacing_plan["planned_requests_per_minute"]), 2)
            if pacing_plan["planned_requests_per_minute"] is not None
            else None
        ),
        planned_itpm=(
            round(float(pacing_plan["planned_itpm"]), 1)
            if pacing_plan["planned_itpm"] is not None
            else None
        ),
        planned_otpm=(
            round(float(pacing_plan["planned_otpm"]), 1)
            if pacing_plan["planned_otpm"] is not None
            else None
        ),
        planned_duration_s=round(float(pacing_plan["planned_duration_s"]), 2),
        limiting_target=str(pacing_plan["limiting_target"]),
    )


def _build_load_test_response(
    *,
    model: str,
    results: list[dict],
    wall_time_s: float,
    target_itpm: int | None,
    target_otpm: int | None,
    total_requests: int,
    max_workers: int,
    max_retries: int,
    max_tokens: int,
    cache_bust: bool,
    target_source: str,
    use_documented_targets: bool = True,
    target_itpm_source: str = "none",
    target_otpm_source: str = "none",
    estimated_input_tokens_per_request: int = 0,
    planned_output_tokens_per_request: int = 0,
    pacing_plan: dict[str, int | float | str | None] | None = None,
    pacing_slots_used: int | None = None,
    actual_last_start_s: float | None = None,
    scheduling_mode: str = "burst",
    duration_s: int | None = None,
    load_multiplier: float = 1.0,
) -> LoadTestResponse:
    if pacing_plan is None:
        pacing_plan = _build_load_test_plan(
            total_requests=total_requests,
            estimated_input_tokens_per_request=estimated_input_tokens_per_request,
            planned_output_tokens_per_request=planned_output_tokens_per_request,
            target_itpm=target_itpm,
            target_otpm=target_otpm,
        )

    ordered_results = sorted(results, key=lambda row: row["abs_end_s"])
    succeeded = sum(1 for row in ordered_results if row["status"] == "ok")
    failed = len(ordered_results) - succeeded
    retried = sum(1 for row in ordered_results if row["attempts"] > 1)
    retried_succeeded = sum(
        1
        for row in ordered_results
        if row["attempts"] > 1 and row["status"] == "ok"
    )
    retry_exhausted = sum(
        1
        for row in ordered_results
        if row["status"] != "ok" and row["attempts"] >= max_retries + 1
    )
    total_attempts = sum(row["attempts"] for row in ordered_results)
    total_prompt_tokens = sum(row.get("prompt_tokens") or 0 for row in ordered_results)
    total_completion_tokens = sum(row.get("completion_tokens") or 0 for row in ordered_results)
    usage_reported = sum(
        1
        for row in ordered_results
        if row.get("prompt_tokens") is not None
        and row.get("completion_tokens") is not None
    )
    usage_missing = len(ordered_results) - usage_reported
    usage_coverage = (
        round(usage_reported / len(ordered_results), 4)
        if ordered_results
        else 0.0
    )
    has_complete_usage = bool(ordered_results) and usage_missing == 0
    achieved_itpm = (
        round((total_prompt_tokens / wall_time_s) * 60, 1)
        if has_complete_usage and wall_time_s > 0
        else None
    )
    achieved_otpm = (
        round((total_completion_tokens / wall_time_s) * 60, 1)
        if has_complete_usage and wall_time_s > 0
        else None
    )
    achieved_requests_per_minute = (
        round((len(ordered_results) / wall_time_s) * 60, 2)
        if ordered_results and wall_time_s > 0
        else None
    )
    successful_requests_per_minute = (
        round((succeeded / wall_time_s) * 60, 2)
        if ordered_results and wall_time_s > 0
        else None
    )
    itpm_target_achievement_ratio = (
        round(achieved_itpm / target_itpm, 4)
        if achieved_itpm is not None and target_itpm is not None
        else None
    )
    otpm_target_achievement_ratio = (
        round(achieved_otpm / target_otpm, 4)
        if achieved_otpm is not None and target_otpm is not None
        else None
    )
    start_delays = [float(row.get("start_delay_s") or 0.0) for row in ordered_results]
    actual_starts = [
        float(row.get("last_attempt_start_s") or row.get("actual_start_s") or 0.0)
        for row in ordered_results
    ]
    successful_latencies = [
        float(row["elapsed_s"])
        for row in ordered_results
        if row["status"] == "ok"
    ]
    p50_latency_s = _percentile(successful_latencies, 0.50)
    p95_latency_s = _percentile(successful_latencies, 0.95)
    p99_latency_s = _percentile(successful_latencies, 0.99)
    config = _make_load_test_config(
        total_requests=total_requests,
        max_workers=max_workers,
        max_retries=max_retries,
        max_tokens=max_tokens,
        cache_bust=cache_bust,
        use_documented_targets=use_documented_targets,
        target_itpm=target_itpm,
        target_otpm=target_otpm,
        target_source=target_source,
        target_itpm_source=target_itpm_source,
        target_otpm_source=target_otpm_source,
        estimated_input_tokens_per_request=estimated_input_tokens_per_request,
        planned_output_tokens_per_request=planned_output_tokens_per_request,
        pacing_plan=pacing_plan,
        scheduling_mode=scheduling_mode,
        duration_s=duration_s,
        load_multiplier=load_multiplier,
    )

    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0
    series: list[LoadTestSeriesPoint] = []
    for row in ordered_results:
        cumulative_prompt_tokens += row.get("prompt_tokens") or 0
        cumulative_completion_tokens += row.get("completion_tokens") or 0
        target_cumulative_prompt_tokens = (
            round((target_itpm * row["abs_end_s"]) / 60, 1) if target_itpm is not None else None
        )
        target_cumulative_completion_tokens = (
            round((target_otpm * row["abs_end_s"]) / 60, 1) if target_otpm is not None else None
        )
        window_start_s = max(0.0, float(row["abs_end_s"]) - 60.0)
        rolling_rows = [
            candidate
            for candidate in ordered_results
            if window_start_s < float(candidate["abs_end_s"]) <= float(row["abs_end_s"])
        ]
        rolling_itpm = float(sum(candidate.get("prompt_tokens") or 0 for candidate in rolling_rows))
        rolling_otpm = float(sum(candidate.get("completion_tokens") or 0 for candidate in rolling_rows))
        series.append(
            LoadTestSeriesPoint(
                elapsed_s=row["abs_end_s"],
                cumulative_prompt_tokens=cumulative_prompt_tokens,
                cumulative_completion_tokens=cumulative_completion_tokens,
                target_cumulative_prompt_tokens=target_cumulative_prompt_tokens,
                target_cumulative_completion_tokens=target_cumulative_completion_tokens,
                target_cumulative_tokens=target_cumulative_completion_tokens,
                rolling_itpm=rolling_itpm,
                rolling_otpm=rolling_otpm,
            )
        )

    peak_rolling_itpm = max((point.rolling_itpm or 0.0 for point in series), default=0.0)
    peak_rolling_otpm = max((point.rolling_otpm or 0.0 for point in series), default=0.0)

    return LoadTestResponse(
        model=model,
        config=config,
        summary=LoadTestSummary(
            succeeded=succeeded,
            failed=failed,
            retried=retried,
            retried_succeeded=retried_succeeded,
            retry_exhausted=retry_exhausted,
            total_attempts=total_attempts,
            wall_time_s=round(wall_time_s, 2),
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            usage_reported=usage_reported,
            usage_missing=usage_missing,
            usage_coverage=usage_coverage,
            achieved_itpm=achieved_itpm,
            achieved_otpm=achieved_otpm,
            target_itpm=target_itpm,
            target_otpm=target_otpm,
            planned_itpm=(
                round(float(pacing_plan["planned_itpm"]), 1)
                if pacing_plan["planned_itpm"] is not None
                else None
            ),
            planned_otpm=(
                round(float(pacing_plan["planned_otpm"]), 1)
                if pacing_plan["planned_otpm"] is not None
                else None
            ),
            achieved_requests_per_minute=achieved_requests_per_minute,
            successful_requests_per_minute=successful_requests_per_minute,
            itpm_target_achievement_ratio=itpm_target_achievement_ratio,
            otpm_target_achievement_ratio=otpm_target_achievement_ratio,
            pacing_slots_used=(
                total_attempts if pacing_slots_used is None else pacing_slots_used
            ),
            average_start_delay_s=(
                round(sum(start_delays) / len(start_delays), 4)
                if start_delays
                else 0.0
            ),
            max_start_delay_s=round(max(start_delays), 4) if start_delays else 0.0,
            actual_last_start_s=(
                round(actual_last_start_s, 4)
                if actual_last_start_s is not None
                else round(max(actual_starts), 4)
                if actual_starts
                else 0.0
            ),
            p50_latency_s=(round(p50_latency_s, 3) if p50_latency_s is not None else None),
            p95_latency_s=(round(p95_latency_s, 3) if p95_latency_s is not None else None),
            p99_latency_s=(round(p99_latency_s, 3) if p99_latency_s is not None else None),
            peak_rolling_itpm=(round(peak_rolling_itpm, 1) if series else None),
            peak_rolling_otpm=(round(peak_rolling_otpm, 1) if series else None),
        ),
        series=series,
        requests=[LoadTestRequestResult(**row) for row in ordered_results],
    )


async def _run_load_test(
    *,
    workspace_client: WorkspaceClient,
    endpoint_name: str,
    endpoint_settings: dict[str, dict],
    total_requests: int,
    max_workers: int,
    max_retries: int,
    max_tokens: int,
    cache_bust: bool,
    use_documented_targets: bool,
    target_itpm: int | None,
    target_otpm: int | None,
    target_source: str,
    target_itpm_source: str,
    target_otpm_source: str,
    estimated_input_tokens_per_request: int,
    pacing_plan: dict[str, int | float | str | None],
    prompt: str,
    schema_json: str,
    system_prompt: str,
    schema_validator: Draft202012Validator,
    scheduling_mode: str = "burst",
    duration_s: int | None = None,
    load_multiplier: float = 1.0,
    progress_callback: Callable[[dict], None] | None = None,
) -> LoadTestResponse:
    semaphore = asyncio.Semaphore(max_workers)
    test_started = time.perf_counter()

    async def runner(req_id: int):
        return await _run_single_load_test_request(
            workspace_client=workspace_client,
            endpoint_name=endpoint_name,
            endpoint_settings=endpoint_settings,
            req_id=req_id,
            max_tokens=max_tokens,
            max_retries=max_retries,
            cache_bust=cache_bust,
            prompt=prompt,
            schema_json=schema_json,
            system_prompt=system_prompt,
            schema_validator=schema_validator,
            attempt_semaphore=semaphore,
            test_started=test_started,
            scheduled_start_s=(
                req_id * float(pacing_plan["request_interval_s"])
            ),
        )

    tasks = [asyncio.create_task(runner(req_id)) for req_id in range(total_requests)]
    results: list[dict] = []
    try:
        for task in asyncio.as_completed(tasks):
            row = await task
            row["abs_end_s"] = round(time.perf_counter() - test_started, 2)
            results.append(row)
            if progress_callback is not None:
                progress_callback(row)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    wall_time_s = time.perf_counter() - test_started
    return _build_load_test_response(
        model=endpoint_name,
        results=results,
        wall_time_s=wall_time_s,
        target_itpm=target_itpm,
        target_otpm=target_otpm,
        total_requests=total_requests,
        max_workers=max_workers,
        max_retries=max_retries,
        max_tokens=max_tokens,
        cache_bust=cache_bust,
        target_source=target_source,
        use_documented_targets=use_documented_targets,
        target_itpm_source=target_itpm_source,
        target_otpm_source=target_otpm_source,
        estimated_input_tokens_per_request=estimated_input_tokens_per_request,
        planned_output_tokens_per_request=max_tokens,
        pacing_plan=pacing_plan,
        scheduling_mode=scheduling_mode,
        duration_s=duration_s,
        load_multiplier=load_multiplier,
    )


async def _prepare_load_test(
    req: LoadTestRequest,
    *,
    max_planned_duration_s: float = LOAD_TEST_MAX_PLANNED_DURATION_S,
    max_wall_time_s: float = LOAD_TEST_MAX_WALL_TIME_S,
) -> dict | JSONResponse:
    """Resolve one validated request into the exact payload and pacing plan."""
    available_models = await list_models()
    selected_model = next(
        (model for model in available_models if model.name == req.model),
        None,
    )

    if selected_model is None:
        return _error_response(
            404,
            "MODEL_NOT_FOUND",
            f"Model '{req.model}' is not available for load testing.",
        )

    if selected_model.availability_status != "ready":
        return _error_response(
            503,
            "MODEL_UNAVAILABLE",
            selected_model.availability_message or "Model is not ready for load testing.",
        )

    effective_max_tokens_cap = min(selected_model.max_tokens, LOAD_TEST_MAX_TOKENS)
    if req.max_tokens > effective_max_tokens_cap:
        return _error_response(
            422,
            "INVALID_MAX_TOKENS",
            f"Max tokens must be between 1 and {effective_max_tokens_cap} for '{req.model}'.",
        )
    if req.expected_output_tokens is not None and req.expected_output_tokens > req.max_tokens:
        return _error_response(
            422,
            "INVALID_EXPECTED_OUTPUT_TOKENS",
            "Expected output tokens cannot exceed the per-request max output tokens.",
        )

    prompt = req.prompt if req.prompt is not None else BENCHMARK_PROMPT
    schema_json = (
        req.json_schema if req.json_schema is not None else BENCHMARK_SCHEMA_JSON
    )
    system_prompt = (
        req.system_prompt
        if req.system_prompt is not None
        else BENCHMARK_SYSTEM_PROMPT
    )
    schema_validator = Draft202012Validator(json.loads(schema_json))
    endpoint_settings = _load_endpoint_settings()
    endpoint_timeout_s = float(
        _resolve_endpoint_options(
            req.model,
            endpoint_settings,
            max_tokens_override=req.max_tokens,
            temperature_override=BENCHMARK_TEMPERATURE,
        )["timeout_s"]
    )

    (
        target_itpm,
        target_otpm,
        target_source,
        target_itpm_source,
        target_otpm_source,
    ) = _resolve_load_test_target(
        model_info=selected_model,
        target_itpm_override=req.target_itpm,
        target_otpm_override=req.target_otpm,
        use_documented_targets=req.use_documented_targets,
    )
    estimated_input_tokens_per_request = _estimate_load_test_input_tokens(
        endpoint_name=req.model,
        endpoint_settings=endpoint_settings,
        prompt=prompt,
        schema_json=schema_json,
        system_prompt=system_prompt,
        max_tokens=req.max_tokens,
        cache_bust=BENCHMARK_CACHE_BUST,
        total_requests=req.total_requests,
    )
    if req.scheduling_mode == "burst" and req.total_requests > LOAD_TEST_MAX_BURST_REQUESTS:
        return _error_response(
            422,
            "INVALID_BURST_REQUESTS",
            f"Burst tests support at most {LOAD_TEST_MAX_BURST_REQUESTS} requests.",
        )
    if req.scheduling_mode == "target_rate" and target_itpm is None and target_otpm is None:
        return _error_response(
            422,
            "TARGET_RATE_REQUIRED",
            "Enter an ITPM target, an OTPM target, or both for a target-rate test.",
        )

    planned_output_tokens = req.expected_output_tokens or req.max_tokens
    effective_total_requests = req.total_requests
    if req.scheduling_mode == "target_rate":
        rate_plan = _build_load_test_plan(
            total_requests=1,
            estimated_input_tokens_per_request=estimated_input_tokens_per_request,
            planned_output_tokens_per_request=planned_output_tokens,
            target_itpm=target_itpm,
            target_otpm=target_otpm,
            scheduling_mode=req.scheduling_mode,
            load_multiplier=req.load_multiplier,
        )
        interval_s = float(rate_plan["request_interval_s"])
        effective_total_requests = min(
            LOAD_TEST_MAX_REQUESTS,
            max(1, math.floor(req.duration_s / interval_s) + 1),
        )

    pacing_plan = _build_load_test_plan(
        total_requests=effective_total_requests,
        estimated_input_tokens_per_request=estimated_input_tokens_per_request,
        planned_output_tokens_per_request=planned_output_tokens,
        target_itpm=target_itpm,
        target_otpm=target_otpm,
        scheduling_mode=req.scheduling_mode,
        load_multiplier=req.load_multiplier,
    )
    request_interval_s = float(pacing_plan["request_interval_s"])
    effective_max_planned_duration_s = min(
        max_planned_duration_s,
        max(0.0, max_wall_time_s - endpoint_timeout_s),
    )
    suggested_max_requests = (
        min(
            LOAD_TEST_MAX_REQUESTS,
            max(
                1,
                math.floor(effective_max_planned_duration_s / request_interval_s)
                + 1,
            ),
        )
        if request_interval_s > 0
        else LOAD_TEST_MAX_REQUESTS
    )
    runnable = (
        float(pacing_plan["planned_duration_s"])
        <= effective_max_planned_duration_s
    )
    validation_message = None
    if not runnable:
        if max_wall_time_s == LOAD_TEST_MAX_WALL_TIME_S:
            reserved_budget_message = (
                f"after reserving {endpoint_timeout_s:.0f}s for the endpoint response "
                f"and {LOAD_TEST_PROXY_SAFETY_MARGIN_S:.0f}s for the Apps proxy"
            )
        else:
            reserved_budget_message = (
                f"after reserving {endpoint_timeout_s:.0f}s for the endpoint response "
                "within the background-run budget"
            )
        validation_message = (
            f"The planned {float(pacing_plan['planned_duration_s']):.2f}s request-launch "
            f"window exceeds the {effective_max_planned_duration_s:.0f}s safe launch window "
            f"{reserved_budget_message}. "
            f"Use at most {suggested_max_requests} requests, lower max tokens, "
            "or increase the token targets."
        )

    config = _make_load_test_config(
        total_requests=effective_total_requests,
        max_workers=req.max_workers,
        max_retries=BENCHMARK_MAX_RETRIES,
        max_tokens=req.max_tokens,
        cache_bust=BENCHMARK_CACHE_BUST,
        use_documented_targets=req.use_documented_targets,
        target_itpm=target_itpm,
        target_otpm=target_otpm,
        target_source=target_source,
        target_itpm_source=target_itpm_source,
        target_otpm_source=target_otpm_source,
        estimated_input_tokens_per_request=estimated_input_tokens_per_request,
        planned_output_tokens_per_request=planned_output_tokens,
        pacing_plan=pacing_plan,
        scheduling_mode=req.scheduling_mode,
        duration_s=req.duration_s,
        load_multiplier=req.load_multiplier,
    )
    return {
        "model_info": selected_model,
        "endpoint_settings": endpoint_settings,
        "prompt": prompt,
        "schema_json": schema_json,
        "system_prompt": system_prompt,
        "schema_validator": schema_validator,
        "target_itpm": target_itpm,
        "target_otpm": target_otpm,
        "target_source": target_source,
        "target_itpm_source": target_itpm_source,
        "target_otpm_source": target_otpm_source,
        "estimated_input_tokens_per_request": estimated_input_tokens_per_request,
        "total_requests": effective_total_requests,
        "scheduling_mode": req.scheduling_mode,
        "duration_s": req.duration_s,
        "load_multiplier": req.load_multiplier,
        "pacing_plan": pacing_plan,
        "config": config,
        "runnable": runnable,
        "max_planned_duration_s": effective_max_planned_duration_s,
        "suggested_max_requests": suggested_max_requests,
        "validation_message": validation_message,
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_test_owner_id(request: Request) -> str:
    """Resolve the Databricks-authenticated caller without exposing it in responses."""
    for header_name in (
        "x-forwarded-email",
        "x-databricks-user-id",
        "x-forwarded-user",
    ):
        value = request.headers.get(header_name)
        if value and value.strip():
            return value.strip().casefold()
    return "anonymous"


def _load_test_workload_key(
    *,
    model: str,
    max_tokens: int,
    prompt: str,
    schema_json: str,
    system_prompt: str,
) -> str:
    """Return a stable, non-reversible key for one endpoint workload."""
    try:
        canonical_schema = json.dumps(
            json.loads(schema_json),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        canonical_schema = schema_json.strip()
    canonical = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "schema": canonical_schema,
            "system_prompt": system_prompt,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _output_estimate_setting_key(owner_id: str, workload_key: str) -> str:
    owner_digest = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()
    return f"{OUTPUT_ESTIMATE_SETTING_PREFIX}:{owner_digest}:{workload_key}"


def _normalize_output_estimate(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    tokens = value.get("tokens")
    sample_count = value.get("sample_count")
    source = value.get("source")
    measured_at = value.get("measured_at")
    if (
        not isinstance(tokens, int)
        or tokens < 1
        or not isinstance(sample_count, int)
        or sample_count < 1
        or source not in {"calibration", "load_test"}
        or not isinstance(measured_at, str)
    ):
        return None
    try:
        datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return {
        "tokens": tokens,
        "sample_count": sample_count,
        "source": source,
        "measured_at": measured_at,
    }


def _get_output_estimate(owner_id: str, workload_key: str) -> tuple[dict | None, str]:
    """Load a user's measured output size, preferring Lakebase."""
    setting_key = _output_estimate_setting_key(owner_id, workload_key)
    db = _get_db_module()
    if db is not None:
        try:
            saved = _normalize_output_estimate(db.get_setting(setting_key))
            if saved is not None:
                _OUTPUT_ESTIMATE_MEMORY[(owner_id, workload_key)] = saved
                return saved, "lakebase"
        except Exception:
            logger.exception("Failed to read output estimate from Lakebase")
    memory = _normalize_output_estimate(
        _OUTPUT_ESTIMATE_MEMORY.get((owner_id, workload_key))
    )
    return (memory, "memory") if memory is not None else (None, "none")


def _save_output_estimate(
    *,
    owner_id: str,
    workload_key: str,
    completion_tokens: list[int],
    source: str,
) -> tuple[dict, str]:
    """Persist the latest representative output measurement for a workload."""
    valid_tokens = [value for value in completion_tokens if value > 0]
    if not valid_tokens:
        raise ValueError("At least one positive completion-token count is required")
    record = {
        "tokens": max(1, int(round(median(valid_tokens)))),
        "sample_count": len(valid_tokens),
        "source": source,
        "measured_at": _utc_now().isoformat(),
    }
    _OUTPUT_ESTIMATE_MEMORY[(owner_id, workload_key)] = record
    db = _get_db_module()
    if db is not None:
        try:
            db.set_setting(
                _output_estimate_setting_key(owner_id, workload_key), record
            )
            return record, "lakebase"
        except Exception:
            logger.exception("Failed to persist output estimate to Lakebase")
    return record, "memory"


def _cleanup_load_test_runs(now: datetime | None = None) -> None:
    """Discard terminal run metadata after its bounded in-memory retention window."""
    current_time = now or _utc_now()
    expired_ids = [
        run_id
        for run_id, record in _LOAD_TEST_RUNS.items()
        if record.response.status in LOAD_TEST_TERMINAL_STATUSES
        and record.response.expires_at is not None
        and record.response.expires_at <= current_time
        and (record.task is None or record.task.done())
    ]
    for run_id in expired_ids:
        _LOAD_TEST_RUNS.pop(run_id, None)


def _load_test_run_snapshot(record: _LoadTestRunRecord) -> LoadTestRunResponse:
    snapshot = record.response.model_copy(deep=True)
    run = record.response
    if run.status == "running" and run.started_at is not None:
        config = run.config
        elapsed_s = max((_utc_now() - run.started_at).total_seconds(), 0.01)
        pacing_plan: dict[str, int | float | str | None] = {
            "pacing_mode": config.pacing_mode,
            "request_interval_s": config.request_interval_s,
            "planned_requests_per_minute": config.planned_requests_per_minute,
            "planned_itpm": config.planned_itpm,
            "planned_otpm": config.planned_otpm,
            "planned_duration_s": config.planned_duration_s,
            "limiting_target": config.limiting_target,
        }
        snapshot.partial_result = _build_load_test_response(
            model=run.model,
            results=record.results,
            wall_time_s=elapsed_s,
            target_itpm=config.target_itpm,
            target_otpm=config.target_otpm,
            total_requests=config.total_requests,
            max_workers=config.max_workers,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            cache_bust=config.cache_bust,
            target_source=config.target_source,
            use_documented_targets=config.use_documented_targets,
            target_itpm_source=config.target_itpm_source,
            target_otpm_source=config.target_otpm_source,
            estimated_input_tokens_per_request=(
                config.estimated_input_tokens_per_request
            ),
            planned_output_tokens_per_request=(
                config.planned_output_tokens_per_request
            ),
            pacing_plan=pacing_plan,
            scheduling_mode=config.scheduling_mode,
            duration_s=config.duration_s,
            load_multiplier=config.load_multiplier,
        )
    return snapshot


def _mark_load_test_run_terminal(
    record: _LoadTestRunRecord,
    *,
    status: str,
    result: LoadTestResponse | None = None,
    error: LoadTestRunError | None = None,
) -> None:
    run = record.response
    if run.status in LOAD_TEST_TERMINAL_STATUSES:
        return

    completed_at = _utc_now()
    run.status = status
    run.completed_at = completed_at
    run.expires_at = completed_at + timedelta(seconds=LOAD_TEST_RUN_TTL_S)
    run.result = result
    run.error = error
    if result is not None:
        run.progress.completed_requests = len(result.requests)
        run.progress.succeeded = result.summary.succeeded
        run.progress.failed = result.summary.failed
        run.progress.total_attempts = result.summary.total_attempts


def _record_load_test_progress(record: _LoadTestRunRecord, row: dict) -> None:
    if record.response.status != "running":
        return
    record.results.append(dict(row))
    progress = record.response.progress
    progress.completed_requests += 1
    if row.get("status") == "ok":
        progress.succeeded += 1
    else:
        progress.failed += 1
    attempts = row.get("attempts")
    if isinstance(attempts, int) and attempts > 0:
        progress.total_attempts += attempts


def _active_load_test_run_count() -> int:
    return sum(
        1
        for record in _LOAD_TEST_RUNS.values()
        if record.response.status in {"queued", "running"}
    )


def _clear_load_test_task(
    run_id: str,
    completed_task: asyncio.Task[None],
) -> None:
    record = _LOAD_TEST_RUNS.get(run_id)
    if record is not None and record.task is completed_task:
        record.task = None


async def _execute_load_test_run(
    run_id: str,
    req: LoadTestRequest,
    prepared: dict,
) -> None:
    record = _LOAD_TEST_RUNS.get(run_id)
    if record is None or record.response.status == "cancelled":
        return

    record.response.status = "running"
    record.response.started_at = _utc_now()

    try:
        result = await asyncio.wait_for(
            _run_load_test(
                workspace_client=_get_workspace_client(),
                endpoint_name=req.model,
                endpoint_settings=prepared["endpoint_settings"],
                total_requests=prepared["total_requests"],
                max_workers=req.max_workers,
                max_retries=BENCHMARK_MAX_RETRIES,
                max_tokens=req.max_tokens,
                cache_bust=BENCHMARK_CACHE_BUST,
                use_documented_targets=req.use_documented_targets,
                target_itpm=prepared["target_itpm"],
                target_otpm=prepared["target_otpm"],
                target_source=prepared["target_source"],
                target_itpm_source=prepared["target_itpm_source"],
                target_otpm_source=prepared["target_otpm_source"],
                estimated_input_tokens_per_request=prepared[
                    "estimated_input_tokens_per_request"
                ],
                pacing_plan=prepared["pacing_plan"],
                prompt=prepared["prompt"],
                schema_json=prepared["schema_json"],
                system_prompt=prepared["system_prompt"],
                schema_validator=prepared["schema_validator"],
                scheduling_mode=prepared["scheduling_mode"],
                duration_s=prepared["duration_s"],
                load_multiplier=prepared["load_multiplier"],
                progress_callback=lambda row: _record_load_test_progress(
                    record, row
                ),
            ),
            timeout=LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
        )
    except asyncio.TimeoutError:
        logger.warning("Background load test timed out for %s", req.model)
        _mark_load_test_run_terminal(
            record,
            status="failed",
            error=LoadTestRunError(
                error_code="LOAD_TEST_TIMEOUT",
                message="The load test exceeded the 15-minute runtime limit.",
                detail="Reduce requests or max tokens, then start a new run.",
            ),
        )
    except asyncio.CancelledError:
        _mark_load_test_run_terminal(record, status="cancelled")
        raise
    except Exception:
        logger.exception("Background load test failed for %s", req.model)
        _mark_load_test_run_terminal(
            record,
            status="failed",
            error=LoadTestRunError(
                error_code="LOAD_TEST_ERROR",
                message="The load test failed before it could complete.",
            ),
        )
    else:
        completion_tokens = [
            row.response_completion_tokens
            for row in result.requests
            if row.status == "ok"
            and isinstance(row.response_completion_tokens, int)
            and row.response_completion_tokens > 0
        ]
        if completion_tokens:
            workload_key = _load_test_workload_key(
                model=req.model,
                max_tokens=req.max_tokens,
                prompt=prepared["prompt"],
                schema_json=prepared["schema_json"],
                system_prompt=prepared["system_prompt"],
            )
            await asyncio.to_thread(
                _save_output_estimate,
                owner_id=record.owner_id,
                workload_key=workload_key,
                completion_tokens=completion_tokens,
                source="load_test",
            )
        _mark_load_test_run_terminal(
            record,
            status="completed",
            result=result,
        )


# ---------------------------------------------------------------------------
# Database helpers (imported lazily to allow app to start without Lakebase)
# ---------------------------------------------------------------------------

def _get_db_module():
    """Lazily import the configured database module, or return None."""
    try:
        from . import database
        return database if database.is_configured() else None
    except Exception:
        logger.exception("Failed to initialize the Lakebase database module")
        return None


# ---------------------------------------------------------------------------
# Model discovery endpoint
# ---------------------------------------------------------------------------

def _ai_extract_model_info() -> ModelInfo:
    return ModelInfo(
        name=AI_EXTRACT_TARGET_ID,
        display_name=AI_EXTRACT_DISPLAY_NAME,
        model_type="ai_extract",
        max_tokens=None,
        serving_runtime=None,
        availability_status="ready",
        availability_message=None,
        rate_limits=None,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    try:
        workspace_client = _get_workspace_client()
        models = await asyncio.to_thread(_list_models_from_workspace, workspace_client)
        return [*models, _ai_extract_model_info()]
    except Exception as e:
        logger.warning("Failed to discover workspace chat endpoints: %s", e)
        configured_models = _load_configured_models()
        return [
            *_mark_models_availability_unverified(configured_models or []),
            _ai_extract_model_info(),
        ]
# ---------------------------------------------------------------------------
# Workspace model-settings endpoints
# ---------------------------------------------------------------------------

@router.get("/workspace/endpoints", response_model=list[WorkspaceEndpoint])
async def list_workspace_endpoints(request: Request = None):
    """List workspace chat endpoints with readiness and selection state."""
    try:
        workspace_client = _get_workspace_client()
        selection, _source = _get_model_selection(_model_settings_owner(request))
        selected_names = {
            entry["name"]
            for entry in selection
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
        return await asyncio.to_thread(
            _list_workspace_endpoints, workspace_client, selected_names
        )
    except Exception:
        logger.exception("Failed to list configured workspace endpoints")
        return _error_response(
            502,
            "WORKSPACE_LIST_ERROR",
            "Unable to list chat serving endpoints in this workspace.",
        )


@router.post(
    "/workspace/endpoints/{endpoint_name}/warmup",
    response_model=ModelWarmupResponse,
)
async def warmup_workspace_endpoint(endpoint_name: str):
    """Wake a custom vLLM endpoint with a tiny synthetic structured request."""
    if not endpoint_name or len(endpoint_name) > 200:
        return _error_response(422, "INVALID_MODEL", "Invalid model endpoint name.")

    try:
        workspace_client = _get_workspace_client()
        workspace_endpoints = list(workspace_client.serving_endpoints.list())
    except Exception:
        logger.exception("Failed to inspect endpoint before warm-up")
        return _error_response(
            502,
            "WORKSPACE_LIST_ERROR",
            "Unable to inspect this model endpoint.",
        )

    endpoint = next(
        (
            item
            for item in workspace_endpoints
            if item.name == endpoint_name and _is_chat_endpoint(item)
        ),
        None,
    )
    if endpoint is None:
        return _error_response(
            404,
            "MODEL_NOT_AVAILABLE",
            "This model endpoint is not visible to the app.",
        )

    canonical = _canonical_endpoint_entry(endpoint_name)
    if (
        canonical["model_type"] != "custom"
        or canonical["serving_runtime"] != "vllm_beta"
    ):
        return _error_response(
            422,
            "MODEL_NOT_WARMABLE",
            "Only custom vLLM endpoints can be woken from this settings pane.",
        )

    try:
        endpoint_detail = await asyncio.to_thread(
            workspace_client.serving_endpoints.get, endpoint_name
        )
    except Exception:
        # Query permission can be available even when detailed configuration
        # is not. The synthetic request remains the authoritative runtime probe.
        endpoint_detail = None
    if endpoint_detail is not None and _endpoint_deployment_stopped(endpoint_detail):
        return _error_response(
            409,
            "MODEL_DEPLOYMENT_STOPPED",
            "This endpoint deployment is stopped. Start it in Databricks Model Serving before using it here.",
        )

    warmup_schema = json.dumps(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    try:
        route_path, payload = _build_inference_request(
            endpoint_name=endpoint_name,
            prompt='Return {"ok": true}.',
            schema_json=warmup_schema,
            system_prompt="Return only the requested JSON object.",
            max_tokens_override=16,
            temperature_override=0.0,
        )
        await _invoke_endpoint(
            workspace_client,
            route_path,
            payload,
            timeout_s=MODEL_WARMUP_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        _MODEL_AVAILABILITY_CACHE.pop(_workspace_cache_key(workspace_client), None)
        return ModelWarmupResponse(
            name=endpoint_name,
            status="warming",
            message="Wake request sent. The endpoint is still starting.",
        )
    except Exception as exc:
        error_text = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        error_name = type(exc).__name__
        if status_code == 429 or "429" in error_text:
            _MODEL_AVAILABILITY_CACHE.pop(_workspace_cache_key(workspace_client), None)
            return ModelWarmupResponse(
                name=endpoint_name,
                status="ready",
                message="Endpoint responded and is ready (currently rate limited).",
            )
        if status_code in {503, 504} or "not ready" in error_text or "starting" in error_text:
            _MODEL_AVAILABILITY_CACHE.pop(_workspace_cache_key(workspace_client), None)
            return ModelWarmupResponse(
                name=endpoint_name,
                status="warming",
                message="Wake request sent. The endpoint is still starting.",
            )
        if status_code == 403 or "PermissionDenied" in error_name or "403" in error_text:
            return _error_response(
                403,
                "PERMISSION_DENIED",
                "The app is not authorized to wake this endpoint.",
            )
        logger.exception("Warm-up request failed for %s", endpoint_name)
        return _error_response(
            502,
            "MODEL_WARMUP_ERROR",
            "The endpoint could not be woken. Try again in a moment.",
        )

    _MODEL_AVAILABILITY_CACHE.pop(_workspace_cache_key(workspace_client), None)
    return ModelWarmupResponse(
        name=endpoint_name,
        status="ready",
        message="Endpoint responded successfully and is ready.",
    )


@router.get("/settings/models", response_model=ModelSettings)
async def get_model_settings(request: Request):
    selection, source = _get_model_selection(_model_settings_owner(request))
    entries = [
        ModelSelectionEntry(
            name=entry["name"],
            model_type=entry.get("model_type") or _fallback_model_type(entry["name"]),
            serving_runtime=entry.get("serving_runtime"),
        )
        for entry in selection
        if isinstance(entry, dict) and entry.get("name")
    ]
    return ModelSettings(endpoints=entries, source=source)


@router.put("/settings/models", response_model=ModelSettings)
async def update_model_settings(
    req: ModelSettingsUpdate, request: Request = None
):
    declared = _declared_endpoint_map()
    try:
        workspace_client = _get_workspace_client()
        workspace_endpoints = await asyncio.to_thread(
            _list_workspace_endpoints, workspace_client
        )
    except Exception:
        logger.exception("Failed to validate workspace endpoints for model settings")
        return _error_response(
            502,
            "WORKSPACE_LIST_ERROR",
            "Unable to validate chat serving endpoints in this workspace.",
        )

    available_names = {entry.name for entry in workspace_endpoints}
    unknown_names = {
        entry.name for entry in req.endpoints if entry.name not in available_names
    }
    if unknown_names:
        return _error_response(
            422,
            "MODEL_NOT_AVAILABLE",
            "Every selected endpoint must be a visible chat endpoint in this workspace.",
        )

    # Deduplicate by name, preserving request order, while restoring all
    # security-sensitive metadata from server-owned overrides or the endpoint
    # naming convention.
    seen: set[str] = set()
    selection: list[dict] = []
    for entry in req.endpoints:
        if entry.name in seen:
            continue
        seen.add(entry.name)
        selection.append(_canonical_endpoint_entry(entry.name, declared))

    source = _save_model_selection(selection, _model_settings_owner(request))
    entries = [
        ModelSelectionEntry(
            name=item["name"],
            model_type=item["model_type"],
            serving_runtime=item.get("serving_runtime"),
        )
        for item in selection
    ]
    return ModelSettings(endpoints=entries, source=source)


# ---------------------------------------------------------------------------
# Preset endpoints
# ---------------------------------------------------------------------------

@router.get("/presets", response_model=list[Preset])
async def get_presets():
    db = _get_db_module()
    if db:
        try:
            rows = db.list_presets()
            if rows:
                return rows
        except Exception:
            logger.exception("DB presets unavailable; falling back to built-in presets")
    return [Preset(**p, is_builtin=True) for p in PRESETS]


@router.post("/presets", response_model=Preset, status_code=201)
async def create_preset(req: PresetCreate):
    db = _get_db_module()
    if not db:
        return _error_response(503, "DB_UNAVAILABLE", "Preset storage is not available")
    try:
        preset = db.create_preset(req)
        return preset
    except Exception:
        logger.exception("Failed to create preset")
        return _error_response(500, "DB_ERROR", "Failed to create preset")


@router.put("/presets/{preset_id}", response_model=Preset)
async def update_preset(preset_id: str, req: PresetUpdate):
    db = _get_db_module()
    if not db:
        return _error_response(503, "DB_UNAVAILABLE", "Preset storage is not available")
    try:
        preset = db.update_preset(preset_id, req)
        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")
        return preset
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update preset")
        return _error_response(500, "DB_ERROR", "Failed to update preset")


@router.delete("/presets/{preset_id}")
async def delete_preset(preset_id: str):
    db = _get_db_module()
    if not db:
        return _error_response(503, "DB_UNAVAILABLE", "Preset storage is not available")
    try:
        result = db.delete_preset(preset_id)
        if result == "builtin":
            return _error_response(403, "BUILTIN_PRESET", "Cannot delete built-in presets")
        if result == "not_found":
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete preset")
        return _error_response(500, "DB_ERROR", "Failed to delete preset")


# ---------------------------------------------------------------------------
# Inference endpoint
# ---------------------------------------------------------------------------

@router.get("/load-test/sample", response_model=LoadTestSampleResponse)
async def get_load_test_sample():
    return LoadTestSampleResponse(
        name=BENCHMARK_SAMPLE_NAME,
        description=BENCHMARK_SAMPLE_DESCRIPTION,
        prompt=BENCHMARK_PROMPT,
        schema_json=BENCHMARK_SCHEMA_JSON,
        system_prompt=BENCHMARK_SYSTEM_PROMPT,
    )


def _output_estimate_response(
    req: LoadTestOutputEstimateRequest,
    owner_id: str,
) -> LoadTestOutputEstimateResponse:
    workload_key = _load_test_workload_key(
        model=req.model,
        max_tokens=req.max_tokens,
        prompt=req.prompt,
        schema_json=req.json_schema,
        system_prompt=req.system_prompt,
    )
    record, persistence = _get_output_estimate(owner_id, workload_key)
    if record is None:
        return LoadTestOutputEstimateResponse(
            model=req.model,
            workload_key=workload_key,
        )
    return LoadTestOutputEstimateResponse(
        model=req.model,
        workload_key=workload_key,
        tokens=record["tokens"],
        sample_count=record["sample_count"],
        source=record["source"],
        measured_at=datetime.fromisoformat(
            record["measured_at"].replace("Z", "+00:00")
        ),
        persistence=persistence,
    )


@router.post(
    "/load-test/output-estimate",
    response_model=LoadTestOutputEstimateResponse,
)
async def get_load_test_output_estimate(
    req: LoadTestOutputEstimateRequest,
    request: Request,
):
    return await asyncio.to_thread(
        _output_estimate_response,
        req,
        _load_test_owner_id(request),
    )


@router.post(
    "/load-test/output-estimate/calibrate",
    response_model=LoadTestOutputEstimateResponse,
)
async def calibrate_load_test_output_estimate(
    req: LoadTestOutputEstimateRequest,
    request: Request,
):
    if req.model not in set(_load_allowed_endpoints()):
        return _error_response(
            400,
            "MODEL_NOT_ALLOWED",
            f"Model endpoint '{req.model}' is not available to this app.",
        )

    endpoint_settings = _load_endpoint_settings()
    try:
        result = await _run_single_load_test_request(
            workspace_client=_get_workspace_client(),
            endpoint_name=req.model,
            endpoint_settings=endpoint_settings,
            req_id=0,
            max_tokens=req.max_tokens,
            max_retries=MAX_RETRIES,
            cache_bust=False,
            prompt=req.prompt,
            schema_json=req.json_schema,
            system_prompt=req.system_prompt,
            schema_validator=Draft202012Validator(json.loads(req.json_schema)),
        )
    except Exception:
        logger.exception("Output calibration failed for %s", req.model)
        return _error_response(
            502,
            "OUTPUT_CALIBRATION_FAILED",
            "The calibration request failed.",
        )

    if result.get("status") != "ok":
        return _error_response(
            502,
            "OUTPUT_CALIBRATION_FAILED",
            result.get("error") or "The calibration request failed.",
        )
    completion_tokens = result.get("response_completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 1:
        return _error_response(
            422,
            "OUTPUT_USAGE_UNAVAILABLE",
            "This endpoint did not report output-token usage.",
            detail="Use a manual output estimate in Advanced for this endpoint.",
        )

    owner_id = _load_test_owner_id(request)
    workload_key = _load_test_workload_key(
        model=req.model,
        max_tokens=req.max_tokens,
        prompt=req.prompt,
        schema_json=req.json_schema,
        system_prompt=req.system_prompt,
    )
    record, persistence = await asyncio.to_thread(
        _save_output_estimate,
        owner_id=owner_id,
        workload_key=workload_key,
        completion_tokens=[completion_tokens],
        source="calibration",
    )
    return LoadTestOutputEstimateResponse(
        model=req.model,
        workload_key=workload_key,
        tokens=record["tokens"],
        sample_count=record["sample_count"],
        source="calibration",
        measured_at=datetime.fromisoformat(record["measured_at"]),
        persistence=persistence,
    )


@router.post("/load-test/plan", response_model=LoadTestPlanResponse)
async def plan_load_test(req: LoadTestRequest):
    try:
        prepared = await _prepare_load_test(
            req,
            max_planned_duration_s=LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
            max_wall_time_s=LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
        )
        if isinstance(prepared, JSONResponse):
            return prepared
        return LoadTestPlanResponse(
            model=req.model,
            config=prepared["config"],
            rate_limits=prepared["model_info"].rate_limits,
            runnable=prepared["runnable"],
            max_planned_duration_s=prepared["max_planned_duration_s"],
            suggested_max_requests=prepared["suggested_max_requests"],
            validation_message=prepared["validation_message"],
        )
    except Exception:
        logger.exception("Failed to plan load test for %s", req.model)
        return _error_response(
            500,
            "LOAD_TEST_PLAN_ERROR",
            "The load-test plan could not be prepared.",
        )


@router.post(
    "/load-test/runs",
    response_model=LoadTestRunResponse,
    status_code=202,
)
async def create_load_test_run(request: Request, req: LoadTestRequest):
    global _LOAD_TEST_PENDING_CREATIONS

    _cleanup_load_test_runs()
    owner_id = _load_test_owner_id(request)
    existing_record = next(
        (
            record
            for record in _LOAD_TEST_RUNS.values()
            if record.owner_id == owner_id
            and record.response.status in {"queued", "running"}
        ),
        None,
    )
    if existing_record is not None:
        return _error_response(
            409,
            "LOAD_TEST_RUN_ALREADY_ACTIVE",
            "A load test is already active for this user.",
            detail=f"Active run id: {existing_record.response.run_id}.",
        )
    if owner_id in _LOAD_TEST_PENDING_OWNERS:
        return _error_response(
            409,
            "LOAD_TEST_RUN_ALREADY_ACTIVE",
            "A load test is already being prepared for this user.",
        )
    if (
        _active_load_test_run_count() + _LOAD_TEST_PENDING_CREATIONS
        >= LOAD_TEST_MAX_ACTIVE_RUNS
    ):
        return _error_response(
            429,
            "LOAD_TEST_RUN_LIMIT",
            "Too many load tests are already active.",
            detail="Wait for an active run to finish or cancel one before retrying.",
        )

    _LOAD_TEST_PENDING_CREATIONS += 1
    _LOAD_TEST_PENDING_OWNERS.add(owner_id)
    try:
        prepared = await _prepare_load_test(
            req,
            max_planned_duration_s=LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
            max_wall_time_s=LOAD_TEST_BACKGROUND_MAX_RUNTIME_S,
        )
    except Exception:
        logger.exception("Failed to prepare background load test for %s", req.model)
        return _error_response(
            500,
            "LOAD_TEST_RUN_CREATE_ERROR",
            "The load test could not be prepared.",
        )
    finally:
        _LOAD_TEST_PENDING_CREATIONS -= 1
        _LOAD_TEST_PENDING_OWNERS.discard(owner_id)

    if isinstance(prepared, JSONResponse):
        return prepared
    if not prepared["runnable"]:
        return _error_response(
            422,
            "LOAD_TEST_DURATION_EXCEEDED",
            prepared["validation_message"],
            detail=(
                f"Suggested maximum requests: "
                f"{prepared['suggested_max_requests']}."
            ),
        )

    run_id = str(uuid.uuid4())
    created_at = _utc_now()
    record = _LoadTestRunRecord(
        response=LoadTestRunResponse(
            run_id=run_id,
            model=req.model,
            status="queued",
            created_at=created_at,
            progress=LoadTestRunProgress(total_requests=prepared["total_requests"]),
            config=prepared["config"].model_copy(deep=True),
        ),
        owner_id=owner_id,
    )
    _LOAD_TEST_RUNS[run_id] = record
    task = asyncio.create_task(_execute_load_test_run(run_id, req, prepared))
    record.task = task
    task.add_done_callback(
        lambda completed_task, current_run_id=run_id: _clear_load_test_task(
            current_run_id, completed_task
        )
    )
    return _load_test_run_snapshot(record)


@router.get(
    "/load-test/runs/active",
    response_model=list[LoadTestRunResponse],
)
async def list_active_load_test_runs(request: Request):
    _cleanup_load_test_runs()
    owner_id = _load_test_owner_id(request)
    records = [
        record
        for record in _LOAD_TEST_RUNS.values()
        if record.owner_id == owner_id
        and record.response.status in {"queued", "running"}
    ]
    records.sort(key=lambda record: record.response.created_at, reverse=True)
    return [_load_test_run_snapshot(record) for record in records]


@router.get(
    "/load-test/runs/{run_id}",
    response_model=LoadTestRunResponse,
)
async def get_load_test_run(run_id: str, request: Request):
    _cleanup_load_test_runs()
    record = _LOAD_TEST_RUNS.get(run_id)
    if record is None or record.owner_id != _load_test_owner_id(request):
        return _error_response(
            404,
            "LOAD_TEST_RUN_NOT_FOUND",
            "The load-test run was not found or has expired.",
        )
    return _load_test_run_snapshot(record)


@router.delete(
    "/load-test/runs/{run_id}",
    response_model=LoadTestRunResponse,
)
async def cancel_load_test_run(run_id: str, request: Request):
    _cleanup_load_test_runs()
    record = _LOAD_TEST_RUNS.get(run_id)
    if record is None or record.owner_id != _load_test_owner_id(request):
        return _error_response(
            404,
            "LOAD_TEST_RUN_NOT_FOUND",
            "The load-test run was not found or has expired.",
        )

    if record.response.status not in LOAD_TEST_TERMINAL_STATUSES:
        _mark_load_test_run_terminal(record, status="cancelled")
        if record.task is not None and not record.task.done():
            record.task.cancel()
    return _load_test_run_snapshot(record)


@router.post("/load-test", response_model=LoadTestResponse)
async def run_load_test(req: LoadTestRequest):
    request_started = time.perf_counter()
    try:
        prepared = await _prepare_load_test(req)
        if isinstance(prepared, JSONResponse):
            return prepared
        if not prepared["runnable"]:
            return _error_response(
                422,
                "LOAD_TEST_DURATION_EXCEEDED",
                prepared["validation_message"],
                detail=(
                    f"Suggested maximum requests: "
                    f"{prepared['suggested_max_requests']}."
                ),
            )

        remaining_wall_time_s = LOAD_TEST_MAX_WALL_TIME_S - (
            time.perf_counter() - request_started
        )
        if remaining_wall_time_s <= 0:
            raise asyncio.TimeoutError

        return await asyncio.wait_for(
            _run_load_test(
                workspace_client=_get_workspace_client(),
                endpoint_name=req.model,
                endpoint_settings=prepared["endpoint_settings"],
                total_requests=prepared["total_requests"],
                max_workers=req.max_workers,
                max_retries=BENCHMARK_MAX_RETRIES,
                max_tokens=req.max_tokens,
                cache_bust=BENCHMARK_CACHE_BUST,
                use_documented_targets=req.use_documented_targets,
                target_itpm=prepared["target_itpm"],
                target_otpm=prepared["target_otpm"],
                target_source=prepared["target_source"],
                target_itpm_source=prepared["target_itpm_source"],
                target_otpm_source=prepared["target_otpm_source"],
                estimated_input_tokens_per_request=prepared[
                    "estimated_input_tokens_per_request"
                ],
                pacing_plan=prepared["pacing_plan"],
                prompt=prepared["prompt"],
                schema_json=prepared["schema_json"],
                system_prompt=prepared["system_prompt"],
                schema_validator=prepared["schema_validator"],
                scheduling_mode=prepared["scheduling_mode"],
                duration_s=prepared["duration_s"],
                load_multiplier=prepared["load_multiplier"],
            ),
            timeout=remaining_wall_time_s,
        )
    except asyncio.TimeoutError:
        logger.warning("Load test exceeded the synchronous Apps request budget for %s", req.model)
        return _error_response(
            504,
            "LOAD_TEST_TIMEOUT",
            "The load test exceeded the synchronous Databricks Apps request time limit.",
            detail="Reduce requests or max tokens, or increase the token targets.",
        )
    except Exception:
        logger.exception("Load test failed for %s", req.model)
        return _error_response(
            502,
            "LOAD_TEST_ERROR",
            "The benchmark request failed.",
        )


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(req: InferenceRequest):
    endpoint_name = req.model if req.model else CUSTOM_ENDPOINT
    use_ai_extract = req.runtime == "ai_extract" or endpoint_name == AI_EXTRACT_TARGET_ID
    if use_ai_extract and endpoint_name != AI_EXTRACT_TARGET_ID:
        return _error_response(
            422,
            "INVALID_INFERENCE_TARGET",
            "AI Extract requests must use the Agent Bricks Information Extraction target.",
        )

    if use_ai_extract:
        try:
            workspace_client = _get_workspace_client()
        except Exception:
            logger.exception("Failed to initialize the Databricks workspace client")
            return _error_response(
                503,
                "WORKSPACE_CLIENT_ERROR",
                "Unable to access the Databricks workspace for AI Extract.",
            )
        return await _run_ai_extract_inference(req, workspace_client)

    allowed_endpoints = set(_load_allowed_endpoints())
    if endpoint_name not in allowed_endpoints:
        return _error_response(
            400,
            "MODEL_NOT_ALLOWED",
            f"Model endpoint '{endpoint_name}' is not available to this app.",
        )

    try:
        w = _get_workspace_client()
    except Exception:
        logger.exception("Failed to initialize the Databricks workspace client")
        return _error_response(
            503,
            "WORKSPACE_CLIENT_ERROR",
            "Unable to access the Databricks workspace for model inference.",
        )

    endpoint_settings = _load_endpoint_settings()
    options = _resolve_endpoint_options(
        endpoint_name, endpoint_settings, max_tokens_override=req.max_tokens
    )
    try:
        route_path, payload = _build_inference_request(
            endpoint_name=endpoint_name,
            prompt=req.prompt,
            schema_json=req.json_schema,
            system_prompt=req.system_prompt,
            settings=endpoint_settings,
            max_tokens_override=req.max_tokens,
        )
    except ValueError:
        logger.exception("Invalid endpoint configuration for %s", endpoint_name)
        return _error_response(
            500,
            "CONFIG_ERROR",
            "Model endpoint configuration is invalid.",
        )

    for attempt in range(1 + MAX_RETRIES):
        try:
            start = time.perf_counter()
            resp_body = await _invoke_endpoint(
                w,
                route_path,
                payload,
                timeout_s=options["timeout_s"],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            break
        except asyncio.TimeoutError:
            return _error_response(
                504,
                "TIMEOUT",
                "The model endpoint did not respond within the time limit.",
                f"Timeout after {options['timeout_s']}s",
            )
        except Exception as e:
            err_str = str(e)

            if "ResourceDoesNotExist" in type(e).__name__ or "404" in err_str:
                logger.exception("Model endpoint %s was not found or not ready", endpoint_name)
                return _error_response(
                    502,
                    "ENDPOINT_NOT_FOUND",
                    f"Model endpoint '{endpoint_name}' not found or not ready.",
                )
            if "PermissionDenied" in type(e).__name__ or "403" in err_str:
                logger.exception("Permission denied querying model endpoint %s", endpoint_name)
                return _error_response(
                    403,
                    "PERMISSION_DENIED",
                    "Not authorized to query the model endpoint.",
                )

            if attempt < MAX_RETRIES:
                logger.warning("Inference attempt %d failed (%s), retrying...", attempt + 1, e)
                await asyncio.sleep(RETRY_BACKOFF_S * (attempt + 1))
                continue
            logger.exception("Model endpoint %s failed after retries", endpoint_name)
            return _error_response(
                502,
                "ENDPOINT_ERROR",
                "The model endpoint returned an error.",
            )

    usage = resp_body.get("usage", {}) if isinstance(resp_body, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    finish_reason = None
    truncated = False
    output_tokens_per_second = None
    if isinstance(completion_tokens, (int, float)) and latency_ms > 0:
        output_tokens_per_second = round(float(completion_tokens) / (latency_ms / 1000), 1)

    choices = resp_body.get("choices", []) if isinstance(resp_body, dict) else []
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        return _error_response(
            502,
            "INVALID_MODEL_OUTPUT",
            "The model endpoint returned a response without JSON output.",
        )

    finish_reason = choices[0].get("finish_reason")
    content = _extract_choice_message_content(resp_body)
    try:
        if isinstance(content, str):
            raw_output = content
            prediction = json.loads(
                _strip_json_markdown_fence(content),
                parse_constant=_reject_non_json_constant,
            )
        else:
            raw_output = json.dumps(content, allow_nan=False)
            prediction = content
    except (json.JSONDecodeError, TypeError, ValueError):
        return _error_response(
            502,
            "INVALID_MODEL_OUTPUT",
            "The model endpoint returned output that was not valid JSON.",
        )

    try:
        response_validator = Draft202012Validator(json.loads(req.json_schema))
        response_validator.validate(prediction)
    except JsonSchemaValidationError as exc:
        return _error_response(
            502,
            "MODEL_OUTPUT_SCHEMA_MISMATCH",
            "The model endpoint returned JSON that did not match the requested schema.",
            _format_schema_validation_detail(exc),
        )

    if finish_reason == "length":
        truncated = True
    elif (
        isinstance(completion_tokens, int)
        and isinstance(options["max_tokens"], int)
        and completion_tokens >= options["max_tokens"]
    ):
        truncated = True

    return InferenceResponse(
        prediction=prediction,
        latency_ms=round(latency_ms, 1),
        raw_output=raw_output,
        model=endpoint_name,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        truncated=truncated,
        repaired_output=False,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        output_tokens_per_second=output_tokens_per_second,
    )
