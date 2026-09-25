import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from api import routes  # noqa: E402
from api.schemas import (  # noqa: E402
    InferenceRequest,
    LoadTestRequest,
    ModelInfo,
    PresetCreate,
    PresetUpdate,
)


SCHEMA_JSON = """
{
  "type": "object",
  "properties": {
    "medications": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "dosage": {"type": "string"}
        },
        "required": ["name", "dosage"]
      }
    }
  },
  "required": ["medications"]
}
""".strip()

SYSTEM_PROMPT = (
    "You are a clinical information extraction assistant. Extract only information "
    "explicitly supported by the provided medical text."
)

VALID_BENCHMARK_OUTPUT = {
    "patient_name": "Maria Alvarez",
    "encounter_date": "2026-03-22",
    "diagnoses": [
        {"name": "community-acquired pneumonia", "status": "improving"},
        {"name": "type 2 diabetes mellitus", "status": None},
    ],
    "medications": [
        {
            "name": "amoxicillin-clavulanate",
            "dosage": "875/125 mg",
            "frequency": "twice daily",
        }
    ],
    "allergies": [{"substance": "lisinopril", "reaction": "cough"}],
    "vitals": {
        "blood_pressure": "128/76",
        "heart_rate": 82,
        "temperature_f": 98.7,
        "oxygen_saturation_percent": 96,
    },
    "follow_up_actions": ["Repeat chest x-ray in 6 weeks"],
}
VALID_BENCHMARK_OUTPUT_JSON = json.dumps(VALID_BENCHMARK_OUTPUT)


class FakeAPIClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def do(self, method, path, body):
        self.calls.append({"method": method, "path": path, "body": body})
        return self.response


class FakeWorkspaceClient:
    def __init__(
        self,
        response=None,
        serving_endpoints=None,
        host="https://workspace.example.com",
        api_client=None,
    ):
        self.api_client = api_client or FakeAPIClient(response or {})
        self.serving_endpoints = serving_endpoints or FakeServingEndpoints()
        self.config = SimpleNamespace(host=host)


class SequencedAPIClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def do(self, method, path, body):
        self.calls.append({"method": method, "path": path, "body": body})
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class FakeRateLimitError(Exception):
    status_code = 429


class ResourceDoesNotExistLeak(Exception):
    pass


class PermissionDeniedLeak(Exception):
    pass


class RaisingAPIClient:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def do(self, method, path, body):
        self.calls.append({"method": method, "path": path, "body": body})
        raise self.error


class FakeServingEndpoints:
    def __init__(self, endpoints=None, details=None, errors=None, list_error=None):
        self._endpoints = endpoints or []
        self._details = details or {}
        self._errors = errors or {}
        self.list_error = list_error
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        if self.list_error:
            raise self.list_error
        return self._endpoints

    def get(self, name):
        if name in self._errors:
            raise self._errors[name]
        return self._details[name]


def _endpoint_summary(name: str, ready: str = "READY", task: str = "llm/v1/chat"):
    return SimpleNamespace(
        name=name,
        state=SimpleNamespace(ready=SimpleNamespace(value=ready)),
        task=task,
    )


def _endpoint_detail(*, foundation_model=None, served_entities=None):
    if served_entities is None:
        served_entities = [SimpleNamespace(foundation_model=foundation_model)]
    return SimpleNamespace(config=SimpleNamespace(served_entities=served_entities))


def _run_inference(monkeypatch, model_name: str, response: dict, *, max_tokens=None):
    fake_workspace = FakeWorkspaceClient(response)
    monkeypatch.setattr(routes, "_load_allowed_endpoints", lambda: [model_name])
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: fake_workspace)
    req = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model=model_name,
        max_tokens=max_tokens,
    )
    result = asyncio.run(routes.run_inference(req))
    return result, fake_workspace.api_client.calls[0]


def test_resolve_endpoint_contract_defaults():
    assert (
        routes._resolve_endpoint_contract(
            "medgemma-1-5-4b-vllm-endpoint",
            {"medgemma-1-5-4b-vllm-endpoint": routes.CONTRACT_OPENAI_GUIDED_JSON},
        )
        == routes.CONTRACT_OPENAI_GUIDED_JSON
    )
    assert (
        routes._resolve_endpoint_contract("gpt-5-4-mini", {})
        == routes.CONTRACT_PPT_JSON_SCHEMA
    )
    assert (
        routes._resolve_endpoint_contract("unknown-custom-endpoint", {})
        == routes.CONTRACT_CUSTOM_INPUTS_SCHEMA
    )


def test_resolve_endpoint_options_defaults_and_overrides():
    default_options = routes._resolve_endpoint_options("qwen3-4b-vllm-endpoint", {})
    assert default_options == {
        "max_tokens": routes.DEFAULT_MAX_TOKENS,
        "timeout_s": routes.DEFAULT_INFERENCE_TIMEOUT_S,
        "temperature": routes.DEFAULT_TEMPERATURE,
    }
    assert default_options["timeout_s"] <= 110
    assert routes._resolve_endpoint_options(
        "medgemma-1-5-4b-vllm-endpoint",
        {"medgemma-1-5-4b-vllm-endpoint": {"max_tokens": 2048, "timeout_s": 600, "temperature": 0.0}},
    ) == {
        "max_tokens": 2048,
        "timeout_s": routes.MAX_INFERENCE_TIMEOUT_S,
        "temperature": 0.0,
    }
    assert routes.MAX_INFERENCE_TIMEOUT_S <= 110


def test_load_configured_models_reads_explicit_types(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {
                    "name": "qwen3-4b-vllm-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "vllm_beta",
                },
                {"name": "gpt-5-4-mini", "model_type": "fmapi"},
            ],
            "settings": {
                "qwen3-4b-vllm-endpoint": {"max_tokens": 2048},
            },
        },
    )

    models = routes._load_configured_models()

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        ("qwen3-4b-vllm-endpoint", "custom", 2048, "vllm_beta", "ready", None),
        ("gpt-5-4-mini", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
    ]


def test_load_configured_models_supports_legacy_string_entries(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                "qwen3-4b-vllm-endpoint",
                "databricks-gpt-5-4-mini",
            ]
        },
    )

    models = routes._load_configured_models()

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        ("qwen3-4b-vllm-endpoint", "custom", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        ("gpt-5-4-mini", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
    ]


def test_get_endpoint_states_uses_cache_within_ttl(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_AVAILABILITY_CACHE", {})
    serving_endpoints = FakeServingEndpoints(
        endpoints=[
            _endpoint_summary("medgemma-1_5-4b-xgrammar-a10"),
            _endpoint_summary("qwen3-4b-vllm-endpoint", ready="NOT_READY"),
        ]
    )
    workspace = FakeWorkspaceClient(serving_endpoints=serving_endpoints, host="https://fevm.example.com")

    first = routes._get_endpoint_states(workspace)
    second = routes._get_endpoint_states(workspace)

    assert first == {
        "medgemma-1_5-4b-xgrammar-a10": "READY",
        "qwen3-4b-vllm-endpoint": "NOT_READY",
    }
    assert second == {
        "medgemma-1_5-4b-xgrammar-a10": "READY",
        "qwen3-4b-vllm-endpoint": "NOT_READY",
    }
    assert serving_endpoints.list_calls == 1


def test_list_models_returns_visible_custom_models_and_ready_fmapi(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_AVAILABILITY_CACHE", {})
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {
                    "name": "medgemma-1_5-4b-xgrammar-a10",
                    "model_type": "custom",
                    "serving_runtime": "standard",
                },
                {"name": "gpt-5-4-mini", "model_type": "fmapi"},
            ],
            "settings": {
                "medgemma-1_5-4b-xgrammar-a10": {"max_tokens": 2048},
            },
        },
    )
    serving_endpoints = FakeServingEndpoints(
        endpoints=[
            _endpoint_summary("medgemma-1_5-4b-xgrammar-a10", ready="NOT_READY"),
            _endpoint_summary("databricks-gpt-5-4-mini"),
        ]
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(serving_endpoints=serving_endpoints, host="https://fevm.example.com"),
    )

    models = asyncio.run(routes.list_models())

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        (
            "medgemma-1_5-4b-xgrammar-a10",
            "custom",
            2048,
            "standard",
            "unavailable",
            routes.CUSTOM_MODEL_UNAVAILABLE_MESSAGE,
        ),
        ("gpt-5-4-mini", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        (
            routes.AI_EXTRACT_TARGET_ID,
            "ai_extract",
            None,
            None,
            "ready",
            None,
        ),
    ]
    assert models[1].rate_limits is not None
    assert models[1].rate_limits.docs_model_name == "GPT-5.4 mini"
    assert models[1].rate_limits.otpm == 5_000


def test_list_models_keeps_absent_custom_visible_and_uses_model_services(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_AVAILABILITY_CACHE", {})
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {
                    "name": "qwen3-4b-vllm-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "vllm_beta",
                },
                {
                    "name": "medgemma-1_5-4b-xgrammar-a10",
                    "model_type": "custom",
                    "serving_runtime": "standard",
                },
                {"name": "gpt-5-4-mini", "model_type": "fmapi"},
                {"name": "gpt-oss-20b", "model_type": "fmapi"},
            ]
        },
    )
    serving_endpoints = FakeServingEndpoints(
        endpoints=[
            _endpoint_summary("medgemma-1_5-4b-xgrammar-a10"),
            _endpoint_summary("databricks-gpt-5-4-mini"),
            _endpoint_summary("databricks-gpt-oss-20b", ready="NOT_READY"),
        ]
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(serving_endpoints=serving_endpoints, host="https://fevm.example.com"),
    )

    models = asyncio.run(routes.list_models())

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        (
            "medgemma-1_5-4b-xgrammar-a10",
            "custom",
            routes.DEFAULT_MAX_TOKENS,
            "standard",
            "ready",
            None,
        ),
        (
            "qwen3-4b-vllm-endpoint",
            "custom",
            routes.DEFAULT_MAX_TOKENS,
            "vllm_beta",
            "unavailable",
            routes.CUSTOM_MODEL_UNAVAILABLE_MESSAGE,
        ),
        ("gpt-5-4-mini", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        ("gpt-oss-20b", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        (
            routes.AI_EXTRACT_TARGET_ID,
            "ai_extract",
            None,
            None,
            "ready",
            None,
        ),
    ]


def test_list_models_marks_configured_models_unavailable_when_lookup_fails(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_AVAILABILITY_CACHE", {})
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {
                    "name": "qwen3-4b-vllm-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "vllm_beta",
                },
                {
                    "name": "medgemma-1_5-4b-xgrammar-a10",
                    "model_type": "custom",
                    "serving_runtime": "standard",
                },
                {"name": "gpt-5-4-mini", "model_type": "fmapi"},
            ]
        },
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            serving_endpoints=FakeServingEndpoints(list_error=RuntimeError("boom")),
            host="https://fevm.example.com",
        ),
    )

    models = asyncio.run(routes.list_models())

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        (
            "qwen3-4b-vllm-endpoint",
            "custom",
            routes.DEFAULT_MAX_TOKENS,
            "vllm_beta",
            "unavailable",
            routes.MODEL_AVAILABILITY_UNVERIFIED_MESSAGE,
        ),
        (
            "medgemma-1_5-4b-xgrammar-a10",
            "custom",
            routes.DEFAULT_MAX_TOKENS,
            "standard",
            "unavailable",
            routes.MODEL_AVAILABILITY_UNVERIFIED_MESSAGE,
        ),
        (
            "gpt-5-4-mini",
            "fmapi",
            routes.DEFAULT_MAX_TOKENS,
            None,
            "unavailable",
            routes.MODEL_AVAILABILITY_UNVERIFIED_MESSAGE,
        ),
        (
            routes.AI_EXTRACT_TARGET_ID,
            "ai_extract",
            None,
            None,
            "ready",
            None,
        ),
    ]


def test_list_models_marks_configured_models_unavailable_when_client_init_fails(
    monkeypatch,
):
    def _raise_workspace_error():
        raise RuntimeError("workspace auth failed")

    configured_model = ModelInfo(
        name="gpt-5-4-mini",
        model_type="fmapi",
        max_tokens=routes.DEFAULT_MAX_TOKENS,
        serving_runtime=None,
        availability_status="ready",
        availability_message=None,
        rate_limits=routes._lookup_rate_limits("gpt-5-4-mini"),
    )

    monkeypatch.setattr(routes, "_load_configured_models", lambda: [configured_model])
    monkeypatch.setattr(routes, "_get_workspace_client", _raise_workspace_error)

    models = asyncio.run(routes.list_models())

    assert len(models) == 2
    assert models[0].availability_status == "unavailable"
    assert (
        models[0].availability_message
        == routes.MODEL_AVAILABILITY_UNVERIFIED_MESSAGE
    )
    assert models[1] == routes._ai_extract_model_info()


def test_list_models_falls_back_to_workspace_listing_when_config_missing(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_AVAILABILITY_CACHE", {})
    serving_endpoints = FakeServingEndpoints(
        endpoints=[
            _endpoint_summary("databricks-gpt-5-4-mini"),
            _endpoint_summary(routes.CUSTOM_ENDPOINT),
            _endpoint_summary("offline-custom-endpoint", ready="NOT_READY"),
            _endpoint_summary("another-custom-endpoint"),
            _endpoint_summary("agent-endpoint", task="agent/v1/responses"),
        ],
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(serving_endpoints=serving_endpoints),
    )
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: None,
    )

    models = asyncio.run(routes.list_models())

    assert [
        (
            model.name,
            model.model_type,
            model.max_tokens,
            model.serving_runtime,
            model.availability_status,
            model.availability_message,
        )
        for model in models
    ] == [
        ("another-custom-endpoint", "custom", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        (routes.CUSTOM_ENDPOINT, "custom", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        (
            "offline-custom-endpoint",
            "custom",
            routes.DEFAULT_MAX_TOKENS,
            None,
            "unavailable",
            routes.CUSTOM_MODEL_UNAVAILABLE_MESSAGE,
        ),
        ("gpt-5-4-mini", "fmapi", routes.DEFAULT_MAX_TOKENS, None, "ready", None),
        (
            routes.AI_EXTRACT_TARGET_ID,
            "ai_extract",
            None,
            None,
            "ready",
            None,
        ),
    ]


def test_build_ai_extract_request_converts_schema_and_precision_options():
    schema = {
        "type": "object",
        "description": "Extract the clinical facts.",
        "properties": {
            "patient_name": {
                "type": "string",
                "description": "The patient's full name.",
                "minLength": 1,
            },
            "age": {"type": ["integer", "null"], "minimum": 0},
            "active": {"type": "boolean"},
            "risk_score": {"type": "number"},
            "status": {
                "type": "string",
                "enum": ["current", "resolved"],
                "description": "Clinical status.",
            },
            "visits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "format": "date",
                            "description": "Visit date.",
                        }
                    },
                    "required": ["date"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["patient_name"],
        "additionalProperties": False,
    }

    path, payload, warnings = routes._build_ai_extract_request(
        prompt="Patient Ada Lovelace was seen on 2026-09-11.",
        schema_json=json.dumps(schema),
        system_prompt="Use only facts present in the note.",
        precision=True,
    )

    assert path == routes.AI_EXTRACT_PATH
    assert payload["content"].startswith("Patient Ada")
    assert payload["options"] == {
        "version": "2.1",
        "enable_citations": True,
        "enable_confidence_scores": True,
        "instructions": (
            "Use only facts present in the note.\n\nExtract the clinical facts."
        ),
        "mode": "precision",
    }
    assert payload["schema"] == {
        "patient_name": {
            "type": "string",
            "description": "The patient's full name.",
        },
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
        "risk_score": {"type": "number"},
        "status": {
            "type": "enum",
            "labels": ["current", "resolved"],
            "description": "Clinical status.",
        },
        "visits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Visit date."}
                },
            },
        },
    }
    assert any("Required-property" in warning for warning in warnings)
    assert any("Nullable" in warning for warning in warnings)
    assert any("format" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("field_schema", "expected_detail"),
    [
        ({"$ref": "#/$defs/person"}, r"\$ref"),
        ({"oneOf": [{"type": "string"}, {"type": "number"}]}, "oneOf"),
        ({"type": ["string", "number"]}, "optional null"),
        ({"type": "array", "items": [{"type": "string"}]}, "tuple arrays"),
    ],
)
def test_ai_extract_schema_rejects_shape_changing_constructs(
    field_schema,
    expected_detail,
):
    schema = {
        "type": "object",
        "properties": {"value": field_schema},
    }

    with pytest.raises(routes._AiExtractSchemaError, match=expected_detail):
        routes._convert_json_schema_for_ai_extract(json.dumps(schema))


def test_ai_extract_schema_enforces_documented_limits():
    too_many_fields = {
        "type": "object",
        "properties": {
            f"field_{index}": {"type": "string"}
            for index in range(routes.AI_EXTRACT_MAX_FIELDS + 1)
        },
    }
    with pytest.raises(routes._AiExtractSchemaError, match="field limit"):
        routes._convert_json_schema_for_ai_extract(json.dumps(too_many_fields))

    enum_overflow = {
        "type": "object",
        "properties": {
            "first": {
                "type": "string",
                "enum": [f"a{index}" for index in range(300)],
            },
            "second": {
                "type": "string",
                "enum": [f"b{index}" for index in range(201)],
            },
        },
    }
    with pytest.raises(routes._AiExtractSchemaError, match="enum-value limit"):
        routes._convert_json_schema_for_ai_extract(json.dumps(enum_overflow))

    nested = {"type": "string"}
    for level in range(routes.AI_EXTRACT_MAX_DEPTH):
        nested = {
            "type": "object",
            "properties": {f"level_{level}": nested},
        }
    too_deep = {"type": "object", "properties": {"root": nested}}
    with pytest.raises(routes._AiExtractSchemaError, match="nesting limit"):
        routes._convert_json_schema_for_ai_extract(json.dumps(too_deep))


def test_ai_extract_instructions_limit_is_checked_before_api_call():
    schema = {
        "type": "object",
        "description": "x" * routes.AI_EXTRACT_MAX_INSTRUCTIONS_LENGTH,
        "properties": {"value": {"type": "string"}},
    }
    with pytest.raises(routes._AiExtractSchemaError, match="character limit"):
        routes._build_ai_extract_request(
            prompt="source",
            schema_json=json.dumps(schema),
            system_prompt="extra",
            precision=False,
        )


def test_parse_ai_extract_response_unwraps_nested_values_and_evidence():
    prediction, evidence = routes._parse_ai_extract_response(
        {
            "response": {
                "patient": {
                    "value": {
                        "name": {
                            "value": "Ada Lovelace",
                            "confidence": 0.97,
                            "citation_ids": [4],
                        },
                        "visits": [
                            {
                                "date": {
                                    "value": "2026-09-11",
                                    "confidence_score": 0.91,
                                    "citation_ids": [5],
                                }
                            }
                        ],
                    },
                    "confidence": 0.94,
                }
            },
            "metadata": {
                "version": "2.1",
                "mode": "precision",
                "chunk_type": "text",
                "citations": [
                    {"id": 4, "start": 0, "stop": 12},
                    {"id": 5, "start": 25, "stop": 35},
                ],
            },
        }
    )

    assert prediction == {
        "patient": {
            "name": "Ada Lovelace",
            "visits": [{"date": "2026-09-11"}],
        }
    }
    fields = {field.path: field for field in evidence.fields}
    assert fields['$["patient"]["name"]'].citation_ids == [4]
    assert fields['$["patient"]["visits"][0]["date"]'].confidence_score == 0.91
    assert fields['$["patient"]'].value == prediction["patient"]
    assert evidence.citations[0].start == 0
    assert evidence.citations[0].stop == 12


def test_run_inference_ai_extract_posts_payload_and_returns_clean_prediction(
    monkeypatch,
):
    source = "Ada Lovelace was seen on 2026-09-11."
    response = {
        "response": {
            "patient_name": {
                "value": "Ada Lovelace",
                "confidence": 0.98,
                "citation_ids": [1],
            }
        },
        "metadata": {
            "version": "2.1",
            "mode": "precision",
            "chunk_type": "text",
            "citations": [{"id": 1, "start": 0, "stop": 12}],
        },
    }
    workspace = FakeWorkspaceClient(response=response)
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)
    schema = {
        "type": "object",
        "properties": {"patient_name": {"type": "string"}},
        "required": ["patient_name"],
        "additionalProperties": False,
    }

    result = asyncio.run(
        routes.run_inference(
            InferenceRequest(
                prompt=source,
                schema_json=json.dumps(schema),
                system_prompt="Extract the patient name.",
                model=routes.AI_EXTRACT_TARGET_ID,
                runtime="ai_extract",
                ai_extract_options={"precision": True},
            )
        )
    )

    assert result.runtime == "ai_extract"
    assert result.model == routes.AI_EXTRACT_TARGET_ID
    assert result.prediction == {"patient_name": "Ada Lovelace"}
    assert result.evidence.fields[0].citation_ids == [1]
    call = workspace.api_client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == routes.AI_EXTRACT_PATH
    assert call["body"]["options"]["mode"] == "precision"
    assert "max_tokens" not in call["body"]


@pytest.mark.parametrize(
    ("status_code", "expected_http_status", "expected_error_code"),
    [
        (403, 403, "AI_EXTRACT_PERMISSION_DENIED"),
        (404, 503, "AI_EXTRACT_UNAVAILABLE"),
        (400, 422, "AI_EXTRACT_REQUEST_REJECTED"),
        (429, 429, "AI_EXTRACT_RATE_LIMITED"),
    ],
)
def test_run_inference_ai_extract_maps_api_errors(
    monkeypatch,
    status_code,
    expected_http_status,
    expected_error_code,
):
    error = routes._LoadTestHttpError(
        status_code=status_code,
        error_code="TEST_ERROR",
        retry_after_secs=0,
        message="sentinel-secret",
    )
    workspace = FakeWorkspaceClient(api_client=RaisingAPIClient(error))
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)
    monkeypatch.setattr(routes, "MAX_RETRIES", 0)
    req = InferenceRequest(
        prompt="source",
        schema_json=json.dumps(
            {"type": "object", "properties": {"value": {"type": "string"}}}
        ),
        model=routes.AI_EXTRACT_TARGET_ID,
        runtime="ai_extract",
    )

    response = asyncio.run(routes.run_inference(req))
    body = json.loads(response.body)

    assert response.status_code == expected_http_status
    assert body["error_code"] == expected_error_code
    assert "sentinel-secret" not in json.dumps(body)


def test_run_inference_ai_extract_retries_429_and_5xx(monkeypatch):
    delays = []

    async def _record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(routes.asyncio, "sleep", _record_sleep)
    api_client = SequencedAPIClient(
        [
            routes._LoadTestHttpError(
                status_code=429,
                error_code="RATE_LIMITED",
                retry_after_secs=0.25,
                message="rate limited",
            ),
            routes._LoadTestHttpError(
                status_code=503,
                error_code="UNAVAILABLE",
                retry_after_secs=None,
                message="unavailable",
            ),
            {
                "response": {"value": {"value": "ok", "confidence": 0.8}},
                "metadata": {"version": "2.1"},
            },
        ]
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(api_client=api_client),
    )
    req = InferenceRequest(
        prompt="source",
        schema_json=json.dumps(
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        ),
        model=routes.AI_EXTRACT_TARGET_ID,
        runtime="ai_extract",
    )

    result = asyncio.run(routes.run_inference(req))

    assert result.prediction == {"value": "ok"}
    assert len(api_client.calls) == 3
    assert delays == [0.25, 2]


@pytest.mark.parametrize(
    "api_response",
    [
        {"error_message": "successful HTTP response with an embedded failure"},
        {"metadata": {"version": "2.1"}},
        {"response": "not-an-object"},
    ],
)
def test_run_inference_ai_extract_rejects_unsuccessful_or_malformed_responses(
    monkeypatch,
    api_response,
):
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(response=api_response),
    )
    req = InferenceRequest(
        prompt="source",
        schema_json=json.dumps(
            {"type": "object", "properties": {"value": {"type": "string"}}}
        ),
        model=routes.AI_EXTRACT_TARGET_ID,
        runtime="ai_extract",
    )

    response = asyncio.run(routes.run_inference(req))
    body = json.loads(response.body)

    assert response.status_code == 502
    assert body["error_code"] == "AI_EXTRACT_INVALID_RESPONSE"


def test_run_inference_ai_extract_validates_original_schema_formats(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            response={
                "response": {"date": {"value": "September 11"}},
                "metadata": {"version": "2.1"},
            }
        ),
    )
    req = InferenceRequest(
        prompt="The encounter occurred September 11.",
        schema_json=json.dumps(
            {
                "type": "object",
                "properties": {"date": {"type": "string", "format": "date"}},
                "required": ["date"],
            }
        ),
        model=routes.AI_EXTRACT_TARGET_ID,
        runtime="ai_extract",
    )

    response = asyncio.run(routes.run_inference(req))
    body = json.loads(response.body)

    assert response.status_code == 502
    assert body["error_code"] == "MODEL_OUTPUT_SCHEMA_MISMATCH"


def test_parse_retry_after_supports_seconds_and_http_date():
    assert routes._parse_retry_after_seconds("2.5") == 2.5
    assert routes._parse_retry_after_seconds("not-a-date") is None
    assert routes._parse_retry_after_seconds("Wed, 21 Oct 2099 07:28:00 GMT") > 0


def test_run_inference_medgemma_uses_custom_inputs(monkeypatch):
    result, call = _run_inference(
        monkeypatch,
        "medgemma-xgrammar-endpoint-a10",
        {"choices": [{"message": {"content": '{"medications": []}'}}]},
    )

    assert call["path"] == "/serving-endpoints/medgemma-xgrammar-endpoint-a10/invocations"
    assert call["body"]["max_tokens"] == 4096
    assert call["body"]["custom_inputs"]["schema"] == SCHEMA_JSON
    assert call["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["body"]["messages"][1] == {"role": "user", "content": "Extract medication details"}
    assert result.prediction == {"medications": []}


def test_run_inference_qwen_uses_openai_guided_json(monkeypatch):
    result, call = _run_inference(
        monkeypatch,
        "qwen3-4b-vllm-endpoint",
        {
            "choices": [
                {
                    "message": {
                        "content": '{"medications":[{"name":"Amoxicillin","dosage":"500mg"}]}'
                    }
                }
            ]
        },
    )

    assert call["path"] == "/serving-endpoints/chat/completions"
    assert call["body"]["model"] == "qwen3-4b-vllm-endpoint"
    assert call["body"]["max_tokens"] == 4096
    assert call["body"]["temperature"] == 0.0
    assert call["body"]["guided_decoding_backend"] == "xgrammar:no-fallback"
    assert call["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["body"]["messages"][1] == {"role": "user", "content": "Extract medication details"}
    assert call["body"]["guided_json"]["type"] == "object"
    assert result.prediction == {"medications": [{"name": "Amoxicillin", "dosage": "500mg"}]}


def test_run_inference_max_tokens_override(monkeypatch):
    # Endpoint default for medgemma is 2048; a per-request override wins.
    _, call = _run_inference(
        monkeypatch,
        "medgemma-1-5-4b-vllm-endpoint",
        {
            "choices": [
                {"message": {"content": '{"medications":[]}'}}
            ]
        },
        max_tokens=8192,
    )
    assert call["body"]["max_tokens"] == 8192


def test_inference_request_rejects_max_tokens_above_ceiling():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        InferenceRequest(
            prompt="x",
            schema_json=SCHEMA_JSON,
            model="qwen3-4b-vllm-endpoint",
            max_tokens=9000,
        )


def test_inference_request_rejects_invalid_json_schema_keyword_value():
    with pytest.raises(ValidationError, match="Invalid JSON Schema"):
        InferenceRequest(
            prompt="x",
            schema_json='{"type":"not-a-real-json-schema-type"}',
            model="gpt-5-4-mini",
        )


def test_run_inference_medgemma_vllm_uses_endpoint_settings(monkeypatch):
    result, call = _run_inference(
        monkeypatch,
        "medgemma-1-5-4b-vllm-endpoint",
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"medications":[{"name":"Doxycycline","dosage":"100 mg twice daily"}]}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 42, "total_tokens": 162},
        },
    )

    assert call["path"] == "/serving-endpoints/chat/completions"
    assert call["body"]["model"] == "medgemma-1-5-4b-vllm-endpoint"
    assert call["body"]["max_tokens"] == 2048
    assert call["body"]["temperature"] == 0.0
    assert call["body"]["guided_decoding_backend"] == "xgrammar:no-fallback"
    assert call["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["body"]["messages"][1] == {"role": "user", "content": "Extract medication details"}
    assert result.finish_reason == "stop"
    assert result.truncated is False
    assert result.repaired_output is False
    assert result.prediction == {
        "medications": [{"name": "Doxycycline", "dosage": "100 mg twice daily"}]
    }


def test_run_inference_ppt_uses_response_format(monkeypatch):
    _, call = _run_inference(
        monkeypatch,
        "gpt-5-4-mini",
        {"choices": [{"message": {"content": '{"medications": []}'}}]},
    )

    assert call["path"] == routes.MODEL_SERVICE_CHAT_PATH
    assert call["body"]["model"] == "system.ai.gpt-5-4-mini"
    assert call["body"]["max_tokens"] == 4096
    assert "temperature" not in call["body"]
    assert call["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["body"]["messages"][1] == {"role": "user", "content": "Extract medication details"}
    response_format = call["body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["type"] == "object"
    assert response_format["json_schema"]["schema"]["required"] == ["medications"]


def test_workspace_endpoint_listing_does_not_expose_backend_exception(monkeypatch):
    secret_error = "workspace-list credential=sentinel-secret"
    workspace = FakeWorkspaceClient(
        serving_endpoints=FakeServingEndpoints(
            list_error=RuntimeError(secret_error),
        )
    )
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)

    response = asyncio.run(routes.list_workspace_endpoints())
    body = json.loads(response.body)

    assert response.status_code == 502
    assert body["error_code"] == "WORKSPACE_LIST_ERROR"
    assert secret_error not in json.dumps(body)


def test_preset_crud_does_not_expose_backend_exceptions(monkeypatch):
    secret_error = "postgres password=sentinel-secret"

    class FailingPresetDB:
        def create_preset(self, _request):
            raise RuntimeError(secret_error)

        def update_preset(self, _preset_id, _request):
            raise RuntimeError(secret_error)

        def delete_preset(self, _preset_id):
            raise RuntimeError(secret_error)

    monkeypatch.setattr(routes, "_get_db_module", lambda: FailingPresetDB())

    responses = [
        asyncio.run(
            routes.create_preset(
                PresetCreate(
                    name="Sentinel preset",
                    schema_json=SCHEMA_JSON,
                    example_prompt="Example clinical note",
                )
            )
        ),
        asyncio.run(
            routes.update_preset(
                "preset-id",
                PresetUpdate(description="Updated description"),
            )
        ),
        asyncio.run(routes.delete_preset("preset-id")),
    ]

    for response in responses:
        body = json.loads(response.body)
        assert response.status_code == 500
        assert body["error_code"] == "DB_ERROR"
        assert secret_error not in json.dumps(body)


def test_run_inference_returns_safe_error_when_workspace_client_fails(monkeypatch):
    secret_error = "credential secret-token-123 was rejected"

    def _raise_workspace_error():
        raise RuntimeError(secret_error)

    monkeypatch.setattr(
        routes,
        "_load_allowed_endpoints",
        lambda: ["gpt-5-4-mini"],
    )
    monkeypatch.setattr(routes, "_get_workspace_client", _raise_workspace_error)
    request = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-5-4-mini",
    )

    response = asyncio.run(routes.run_inference(request))
    body = json.loads(response.body)

    assert response.status_code == 503
    assert body["error_code"] == "WORKSPACE_CLIENT_ERROR"
    assert "workspace" in body["message"].lower()
    assert secret_error not in json.dumps(body)


def test_run_inference_does_not_expose_invalid_config_exception(monkeypatch):
    secret_error = "contract config contained sentinel-secret"

    def _raise_config_error(*_args, **_kwargs):
        raise ValueError(secret_error)

    monkeypatch.setattr(
        routes,
        "_load_allowed_endpoints",
        lambda: ["gpt-5-4-mini"],
    )
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(routes, "_build_inference_request", _raise_config_error)
    request = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-5-4-mini",
    )

    response = asyncio.run(routes.run_inference(request))
    body = json.loads(response.body)

    assert response.status_code == 500
    assert body["error_code"] == "CONFIG_ERROR"
    assert secret_error not in json.dumps(body)


@pytest.mark.parametrize(
    ("endpoint_error", "expected_status", "expected_code"),
    [
        (
            ResourceDoesNotExistLeak("404 upstream included sentinel-secret"),
            502,
            "ENDPOINT_NOT_FOUND",
        ),
        (
            PermissionDeniedLeak("403 upstream included sentinel-secret"),
            403,
            "PERMISSION_DENIED",
        ),
        (
            RuntimeError("endpoint transport included sentinel-secret"),
            502,
            "ENDPOINT_ERROR",
        ),
    ],
)
def test_run_inference_does_not_expose_endpoint_exceptions(
    monkeypatch,
    endpoint_error,
    expected_status,
    expected_code,
):
    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(
        routes,
        "_load_allowed_endpoints",
        lambda: ["gpt-5-4-mini"],
    )
    workspace = FakeWorkspaceClient(api_client=RaisingAPIClient(endpoint_error))
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)
    request = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-5-4-mini",
    )

    response = asyncio.run(routes.run_inference(request))
    body = json.loads(response.body)

    assert response.status_code == expected_status
    assert body["error_code"] == expected_code
    assert "sentinel-secret" not in json.dumps(body)


def test_run_inference_rejects_model_not_visible_to_app(monkeypatch):
    workspace_accessed = False

    def _get_workspace_client():
        nonlocal workspace_accessed
        workspace_accessed = True
        return FakeWorkspaceClient()

    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {"name": "qwen3-4b-vllm-endpoint", "model_type": "custom"},
                {"name": "gpt-5-4-mini", "model_type": "fmapi"},
            ]
        },
    )
    monkeypatch.setattr(
        routes,
        "_get_model_selection",
        lambda: (
            [
                {
                    "name": "qwen3-4b-vllm-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "vllm_beta",
                }
            ],
            "saved",
        ),
    )
    monkeypatch.setattr(routes, "_get_workspace_client", _get_workspace_client)
    request = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-5-4-mini",
    )

    response = asyncio.run(routes.run_inference(request))
    body = json.loads(response.body)

    assert response.status_code == 400
    assert body["error_code"] == "MODEL_NOT_ALLOWED"
    assert "not available to this app" in body["message"]
    assert workspace_accessed is True


def test_run_inference_rejects_unknown_model(monkeypatch):
    monkeypatch.setattr(
        routes,
        "_load_allowed_endpoints",
        lambda: ["qwen3-4b-vllm-endpoint"],
    )

    def _unexpected_workspace_access():
        pytest.fail("unknown model should be rejected before workspace access")

    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        _unexpected_workspace_access,
    )
    request = InferenceRequest(
        prompt="Extract medication details",
        schema_json=SCHEMA_JSON,
        system_prompt=SYSTEM_PROMPT,
        model="unknown-endpoint",
    )

    response = asyncio.run(routes.run_inference(request))
    body = json.loads(response.body)

    assert response.status_code == 400
    assert body["error_code"] == "MODEL_NOT_ALLOWED"


def test_run_inference_rejects_non_json_model_output(monkeypatch):
    response, call = _run_inference(
        monkeypatch,
        "medgemma-1-5-4b-vllm-endpoint",
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{\n  "study_metadata": {\n    "modality": "echo"\n  \n  \n  \n'
                    },
                }
            ],
            "usage": {"prompt_tokens": 144, "completion_tokens": 1024, "total_tokens": 1168},
        },
    )

    assert call["body"]["max_tokens"] == 2048
    assert response.status_code == 502
    body = json.loads(response.body)
    assert body["error_code"] == "INVALID_MODEL_OUTPUT"


def test_run_inference_accepts_json_wrapped_in_markdown_fence(monkeypatch):
    result, _call = _run_inference(
        monkeypatch,
        "gpt-5-4-mini",
        {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"medications": []}\n```'
                    }
                }
            ]
        },
    )

    assert result.prediction == {"medications": []}
    assert result.raw_output.startswith("```json")
    assert result.repaired_output is False


def test_run_inference_rejects_json_that_does_not_match_schema(monkeypatch):
    response, _call = _run_inference(
        monkeypatch,
        "gpt-5-4-mini",
        {
            "choices": [
                {
                    "message": {
                        "content": '{"medications":[{"name":"Amoxicillin"}]}'
                    }
                }
            ]
        },
    )

    assert response.status_code == 502
    body = json.loads(response.body)
    assert body["error_code"] == "MODEL_OUTPUT_SCHEMA_MISMATCH"
    assert "did not match the requested schema" in body["message"]
    assert body["detail"] == (
        'At $["medications"][0]: \'dosage\' is a required property'
    )


def test_build_inference_request_omits_temperature_for_gpt_benchmark():
    route_path, payload = routes._build_inference_request(
        endpoint_name="gpt-5-4-mini",
        prompt="Benchmark prompt",
        schema_json=SCHEMA_JSON,
        system_prompt="System prompt",
        settings={},
        max_tokens_override=1024,
        temperature_override=0.7,
    )

    assert route_path == routes.MODEL_SERVICE_CHAT_PATH
    assert payload["model"] == "system.ai.gpt-5-4-mini"
    assert payload["max_tokens"] == 1024
    assert "temperature" not in payload
    assert payload["messages"][0] == {"role": "system", "content": "System prompt"}
    assert payload["messages"][1] == {"role": "user", "content": "Benchmark prompt"}


def test_build_inference_request_omits_temperature_for_gpt_5_6_sol(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_SERVICE_NAMES_CACHE", {"gpt-5-6-sol"})

    route_path, payload = routes._build_inference_request(
        endpoint_name="gpt-5-6-sol",
        prompt="Extract structured data",
        schema_json=SCHEMA_JSON,
        system_prompt="Return JSON only",
        settings={},
    )

    assert route_path == routes.MODEL_SERVICE_CHAT_PATH
    assert payload["model"] == "system.ai.gpt-5-6-sol"
    assert "temperature" not in payload


def test_load_test_request_applies_request_worker_and_token_caps():
    with pytest.raises(ValidationError):
        LoadTestRequest(model="gpt-5-4-mini", total_requests=151)

    with pytest.raises(ValidationError):
        LoadTestRequest(model="gpt-5-4-mini", max_workers=11)

    with pytest.raises(ValidationError):
        LoadTestRequest(model="gpt-5-4-mini", max_tokens=4097)

    with pytest.raises(ValidationError):
        LoadTestRequest(model="gpt-5-4-mini", target_itpm=10_000_001)

    with pytest.raises(ValidationError):
        LoadTestRequest(model="gpt-5-4-mini", target_otpm=10_000_001)


def test_lookup_rate_limits_matches_known_fmapi_models():
    gemini_limits = routes._lookup_rate_limits("gemini-3-1-flash-lite")
    assert gemini_limits is not None
    assert gemini_limits.docs_model_name == "Gemini 3.1 Flash Lite"
    assert gemini_limits.itpm == 200_000
    assert gemini_limits.otpm == 20_000
    assert gemini_limits.qph == 360_000
    assert gemini_limits.docs_updated_on == routes.FMAPI_LIMITS_DOCS_UPDATED_ON

    gpt_limits = routes._lookup_rate_limits("gpt-5-4-mini")
    assert gpt_limits is not None
    assert gpt_limits.docs_model_name == "GPT-5.4 mini"
    assert gpt_limits.otpm == 5_000

    claude_limits = routes._lookup_rate_limits("claude-opus-4-6")
    assert claude_limits is not None
    assert claude_limits.docs_model_name == "Claude Opus 4.6"
    assert claude_limits.otpm == 20_000

    assert routes._lookup_rate_limits("databricks-unknown-model") is None
    assert routes._lookup_target_otpm("gpt-oss-20b") == 10_000


def test_resolve_load_test_target_treats_explicit_override_as_user_reference():
    model_info = ModelInfo(
        name="gpt-5-4-mini",
        model_type="fmapi",
        max_tokens=4096,
        serving_runtime=None,
        availability_status="ready",
        availability_message=None,
        rate_limits=routes._lookup_rate_limits("gpt-5-4-mini"),
    )

    (
        target_itpm,
        target_otpm,
        target_source,
        target_itpm_source,
        target_otpm_source,
    ) = routes._resolve_load_test_target(
        model_info=model_info,
        target_itpm_override=50_000,
        target_otpm_override=5_000,
    )

    assert target_itpm == 50_000
    assert target_otpm == 5_000
    assert target_source == "custom"
    assert target_itpm_source == "custom"
    assert target_otpm_source == "custom"


def test_build_load_test_response_summarizes_results():
    response = routes._build_load_test_response(
        model="gemini-3-1-flash-lite",
        results=[
            {
                "req_id": 0,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 150,
                "completion_tokens": 100,
                "elapsed_s": 1.2,
                "abs_end_s": 2.0,
                "error": None,
            },
            {
                "req_id": 1,
                "status": "ok",
                "attempts": 3,
                "prompt_tokens": 180,
                "completion_tokens": 200,
                "elapsed_s": 2.6,
                "abs_end_s": 5.0,
                "error": None,
            },
            {
                "req_id": 2,
                "status": "429_exhausted",
                "attempts": 6,
                "prompt_tokens": None,
                "completion_tokens": None,
                "elapsed_s": 4.5,
                "abs_end_s": 6.0,
                "error": "429",
            },
        ],
        wall_time_s=6.0,
        target_itpm=60_000,
        target_otpm=20_000,
        total_requests=30,
        max_workers=5,
        max_retries=5,
        max_tokens=1024,
        cache_bust=True,
        target_source="documented",
    )

    assert response.summary.succeeded == 2
    assert response.summary.failed == 1
    assert response.summary.retried == 2
    assert response.summary.retried_succeeded == 1
    assert response.summary.retry_exhausted == 1
    assert response.summary.total_attempts == 10
    assert response.summary.total_prompt_tokens == 330
    assert response.summary.total_completion_tokens == 300
    assert response.summary.usage_reported == 2
    assert response.summary.usage_missing == 1
    assert response.summary.usage_coverage == pytest.approx(2 / 3, abs=0.0001)
    assert response.summary.achieved_itpm is None
    assert response.summary.achieved_otpm is None
    assert response.summary.target_itpm == 60_000
    assert response.config.target_source == "documented"
    assert response.series[-1].cumulative_prompt_tokens == 330
    assert response.series[-1].cumulative_completion_tokens == 300
    assert response.series[-1].target_cumulative_prompt_tokens == 6000.0
    assert response.series[-1].target_cumulative_completion_tokens == 2000.0
    assert response.series[-1].target_cumulative_tokens == 2000.0


def test_build_load_test_response_handles_partial_targets():
    response = routes._build_load_test_response(
        model="databricks-custom-limits",
        results=[
            {
                "req_id": 0,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 90,
                "completion_tokens": None,
                "elapsed_s": 1.1,
                "abs_end_s": 3.0,
                "error": None,
            }
        ],
        wall_time_s=3.0,
        target_itpm=12_000,
        target_otpm=None,
        total_requests=1,
        max_workers=1,
        max_retries=5,
        max_tokens=512,
        cache_bust=True,
        target_source="documented",
    )

    assert response.summary.total_prompt_tokens == 90
    assert response.summary.total_completion_tokens == 0
    assert response.summary.usage_reported == 0
    assert response.summary.usage_missing == 1
    assert response.summary.usage_coverage == 0.0
    assert response.summary.achieved_itpm is None
    assert response.summary.achieved_otpm is None
    assert response.summary.target_itpm == 12_000
    assert response.summary.target_otpm is None
    assert response.series[0].target_cumulative_prompt_tokens == 600.0
    assert response.series[0].target_cumulative_completion_tokens is None
    assert response.series[0].target_cumulative_tokens is None


def test_build_load_test_response_tolerates_missing_token_fields():
    response = routes._build_load_test_response(
        model="gemini-3-1-flash-lite",
        results=[
            {
                "req_id": 0,
                "status": "ok",
                "attempts": 1,
                "elapsed_s": 0.8,
                "abs_end_s": 1.5,
                "error": None,
            }
        ],
        wall_time_s=1.5,
        target_itpm=None,
        target_otpm=None,
        total_requests=1,
        max_workers=1,
        max_retries=5,
        max_tokens=256,
        cache_bust=True,
        target_source="none",
    )

    assert response.summary.total_prompt_tokens == 0
    assert response.summary.total_completion_tokens == 0
    assert response.summary.usage_reported == 0
    assert response.summary.usage_missing == 1
    assert response.summary.usage_coverage == 0.0
    assert response.summary.achieved_itpm is None
    assert response.summary.achieved_otpm is None
    assert response.requests[0].prompt_tokens is None
    assert response.requests[0].completion_tokens is None


def test_build_load_test_response_calculates_rates_with_complete_usage():
    response = routes._build_load_test_response(
        model="gpt-5-4-mini",
        results=[
            {
                "req_id": 0,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "elapsed_s": 0.5,
                "abs_end_s": 1.0,
                "error": None,
            },
            {
                "req_id": 1,
                "status": "ok",
                "attempts": 1,
                "prompt_tokens": 75,
                "completion_tokens": 20,
                "elapsed_s": 0.8,
                "abs_end_s": 2.0,
                "error": None,
            },
        ],
        wall_time_s=2.0,
        target_itpm=50_000,
        target_otpm=5_000,
        total_requests=2,
        max_workers=2,
        max_retries=5,
        max_tokens=512,
        cache_bust=True,
        target_source="documented",
    )

    assert response.summary.usage_reported == 2
    assert response.summary.usage_missing == 0
    assert response.summary.usage_coverage == 1.0
    assert response.summary.achieved_itpm == 3000.0
    assert response.summary.achieved_otpm == 900.0


def test_run_single_load_test_request_retries_rate_limit(monkeypatch):
    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 0.0)
    api_client = SequencedAPIClient(
        [
            FakeRateLimitError("REQUEST_LIMIT_EXCEEDED"),
            {
                "choices": [{"message": {"content": VALID_BENCHMARK_OUTPUT_JSON}}],
                "usage": {"prompt_tokens": 222, "completion_tokens": 111},
            },
        ]
    )
    workspace = FakeWorkspaceClient(api_client=api_client)

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gemini-3-1-flash-lite",
            endpoint_settings={},
            req_id=7,
            max_tokens=768,
            max_retries=5,
            cache_bust=True,
        )
    )

    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["prompt_tokens"] == 222
    assert result["completion_tokens"] == 111
    assert len(api_client.calls) == 2


def test_run_single_load_test_request_retries_malformed_json(monkeypatch):
    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 0.0)
    api_client = SequencedAPIClient(
        [
            {"choices": [{"message": {"content": '{"broken": '}}]},
            {
                "choices": [{"message": {"content": VALID_BENCHMARK_OUTPUT_JSON}}],
                "usage": {"prompt_tokens": 99, "completion_tokens": 87},
            },
        ]
    )
    workspace = FakeWorkspaceClient(api_client=api_client)

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=3,
            max_tokens=1024,
            max_retries=5,
            cache_bust=True,
        )
    )

    assert result["status"] == "ok"
    assert result["attempts"] == 2
    assert result["prompt_tokens"] == 99
    assert result["completion_tokens"] == 87


def test_run_single_load_test_request_retries_then_reports_schema_validation_error(
    monkeypatch,
):
    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(routes.random, "uniform", lambda _a, _b: 0.0)
    invalid_response = {
        "choices": [{"message": {"content": '{"patient_name": "Maria Alvarez"}'}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }
    api_client = SequencedAPIClient([invalid_response, invalid_response])
    workspace = FakeWorkspaceClient(api_client=api_client)

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=4,
            max_tokens=512,
            max_retries=1,
            cache_bust=True,
        )
    )

    assert result["status"] == "schema_validation_error"
    assert result["attempts"] == 2
    assert result["prompt_tokens"] == 40
    assert result["completion_tokens"] == 10
    assert len(api_client.calls) == 2


def test_run_single_load_test_request_uses_configured_max_tokens(monkeypatch):
    api_client = SequencedAPIClient(
        [
            {
                "choices": [{"message": {"content": VALID_BENCHMARK_OUTPUT}}],
                "usage": {"prompt_tokens": 55, "completion_tokens": 64},
            }
        ]
    )
    workspace = FakeWorkspaceClient(api_client=api_client)

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=workspace,
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=1,
            max_tokens=256,
            max_retries=5,
            cache_bust=True,
        )
    )

    assert result["status"] == "ok"
    assert result["prompt_tokens"] == 55
    assert api_client.calls[0]["body"]["max_tokens"] == 256


@pytest.mark.parametrize(
    ("api_result", "expected_status"),
    [
        (RuntimeError("transport included sentinel-secret"), "error"),
        (FakeRateLimitError("429 included sentinel-secret"), "429_exhausted"),
        (
            {"choices": [{"message": {"content": "sentinel-secret is not JSON"}}]},
            "invalid_json",
        ),
    ],
)
def test_run_single_load_test_request_uses_safe_error_messages(
    monkeypatch,
    api_result,
    expected_status,
):
    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(routes.asyncio, "sleep", _no_sleep)
    api_client = SequencedAPIClient([api_result])

    result = asyncio.run(
        routes._run_single_load_test_request(
            workspace_client=FakeWorkspaceClient(api_client=api_client),
            endpoint_name="gpt-5-4-mini",
            endpoint_settings={},
            req_id=9,
            max_tokens=256,
            max_retries=0,
            cache_bust=False,
        )
    )

    assert result["status"] == expected_status
    assert result["error"]
    assert "sentinel-secret" not in result["error"]


def test_run_load_test_route_allows_ready_custom_models(monkeypatch):
    captured = {}

    async def _list_models():
        return [
            ModelInfo(
                name="qwen3-4b-vllm-endpoint",
                model_type="custom",
                max_tokens=routes.DEFAULT_MAX_TOKENS,
                serving_runtime="vllm_beta",
                availability_status="ready",
                availability_message=None,
            )
        ]

    async def _fake_run_load_test(**kwargs):
        captured.update(kwargs)
        return routes._build_load_test_response(
            model=kwargs["endpoint_name"],
            results=[],
            wall_time_s=1.0,
            target_itpm=kwargs["target_itpm"],
            target_otpm=kwargs["target_otpm"],
            total_requests=kwargs["total_requests"],
            max_workers=kwargs["max_workers"],
            max_retries=kwargs["max_retries"],
            max_tokens=kwargs["max_tokens"],
            cache_bust=kwargs["cache_bust"],
            target_source=kwargs["target_source"],
        )

    monkeypatch.setattr(routes, "list_models", _list_models)
    monkeypatch.setattr(routes, "_run_load_test", _fake_run_load_test)
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="qwen3-4b-vllm-endpoint",
                total_requests=2,
                max_workers=1,
                max_tokens=128,
            )
        )
    )

    assert response.model == "qwen3-4b-vllm-endpoint"
    assert captured["target_itpm"] is None
    assert captured["target_otpm"] is None
    assert captured["pacing_plan"]["pacing_mode"] == "burst"


def test_run_load_test_route_rejects_max_tokens_above_model_limit(monkeypatch):
    async def _list_models():
        return [
            ModelInfo(
                name="gpt-5-4-mini",
                model_type="fmapi",
                max_tokens=512,
                serving_runtime=None,
                availability_status="ready",
                availability_message=None,
                rate_limits=routes._lookup_rate_limits("gpt-5-4-mini"),
            )
        ]

    monkeypatch.setattr(routes, "list_models", _list_models)

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                total_requests=30,
                max_workers=5,
                max_tokens=1024,
            )
        )
    )

    assert response.status_code == 422
    assert b"Max tokens must be between 1 and 512" in response.body


def test_run_load_test_route_does_not_expose_top_level_exception(monkeypatch):
    secret_error = "load-test backend included sentinel-secret"

    async def _list_models():
        return [
            ModelInfo(
                name="gpt-5-4-mini",
                model_type="fmapi",
                max_tokens=4096,
                serving_runtime=None,
                availability_status="ready",
                availability_message=None,
                rate_limits=routes._lookup_rate_limits("gpt-5-4-mini"),
            )
        ]

    async def _raise_load_test_error(**_kwargs):
        raise RuntimeError(secret_error)

    monkeypatch.setattr(routes, "list_models", _list_models)
    monkeypatch.setattr(routes, "_run_load_test", _raise_load_test_error)
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                total_requests=1,
                max_workers=1,
                max_tokens=256,
            )
        )
    )
    body = json.loads(response.body)

    assert response.status_code == 502
    assert body["error_code"] == "LOAD_TEST_ERROR"
    assert secret_error not in json.dumps(body)


def test_run_load_test_route_uses_custom_target_override(monkeypatch):
    captured = {}

    async def _list_models():
        return [
            ModelInfo(
                name="gpt-5-4-mini",
                model_type="fmapi",
                max_tokens=4096,
                serving_runtime=None,
                availability_status="ready",
                availability_message=None,
                rate_limits=routes._lookup_rate_limits("gpt-5-4-mini"),
            )
        ]

    async def _fake_run_load_test(**kwargs):
        captured.update(kwargs)
        return routes._build_load_test_response(
            model=kwargs["endpoint_name"],
            results=[],
            wall_time_s=1.0,
            target_itpm=kwargs["target_itpm"],
            target_otpm=kwargs["target_otpm"],
            total_requests=kwargs["total_requests"],
            max_workers=kwargs["max_workers"],
            max_retries=kwargs["max_retries"],
            max_tokens=kwargs["max_tokens"],
            cache_bust=kwargs["cache_bust"],
            target_source=kwargs["target_source"],
        )

    monkeypatch.setattr(routes, "list_models", _list_models)
    monkeypatch.setattr(routes, "_run_load_test", _fake_run_load_test)
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(
        routes,
        "_load_endpoint_settings",
        lambda: {"gpt-5-4-mini": {"timeout_s": 5}},
    )

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="gpt-5-4-mini",
                total_requests=10,
                max_workers=8,
                max_tokens=512,
                target_itpm=75_000,
                target_otpm=20_000,
            )
        )
    )

    assert response.config.total_requests == 10
    assert response.config.max_workers == 8
    assert response.config.max_tokens == 512
    assert response.config.target_source == "custom"
    assert response.summary.target_itpm == 75_000
    assert response.summary.target_otpm == 20_000
    assert captured["target_itpm"] == 75_000
    assert captured["target_otpm"] == 20_000
    assert captured["target_source"] == "custom"


def test_run_load_test_route_uses_none_target_source_for_unknown_limit(monkeypatch):
    async def _list_models():
        return [
            ModelInfo(
                name="databricks-unknown-model",
                model_type="fmapi",
                max_tokens=4096,
                serving_runtime=None,
                availability_status="ready",
                availability_message=None,
                rate_limits=None,
            )
        ]

    async def _fake_run_load_test(**kwargs):
        return routes._build_load_test_response(
            model=kwargs["endpoint_name"],
            results=[],
            wall_time_s=1.0,
            target_itpm=kwargs["target_itpm"],
            target_otpm=kwargs["target_otpm"],
            total_requests=kwargs["total_requests"],
            max_workers=kwargs["max_workers"],
            max_retries=kwargs["max_retries"],
            max_tokens=kwargs["max_tokens"],
            cache_bust=kwargs["cache_bust"],
            target_source=kwargs["target_source"],
        )

    monkeypatch.setattr(routes, "list_models", _list_models)
    monkeypatch.setattr(routes, "_run_load_test", _fake_run_load_test)
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})

    response = asyncio.run(
        routes.run_load_test(
            LoadTestRequest(
                model="databricks-unknown-model",
                total_requests=30,
                max_workers=5,
                max_tokens=512,
            )
        )
    )

    assert response.config.target_source == "none"
    assert response.summary.target_itpm is None
    assert response.summary.target_otpm is None


def test_get_load_test_sample_returns_fixed_payload():
    sample = asyncio.run(routes.get_load_test_sample())

    assert sample.name == routes.BENCHMARK_SAMPLE_NAME
    assert sample.description == routes.BENCHMARK_SAMPLE_DESCRIPTION
    assert sample.prompt == routes.BENCHMARK_PROMPT
    assert sample.system_prompt == routes.BENCHMARK_SYSTEM_PROMPT
    assert sample.json_schema == routes.BENCHMARK_SCHEMA_JSON
