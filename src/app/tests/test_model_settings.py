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

from api import database, routes  # noqa: E402
from api.schemas import ModelSelectionEntry, ModelSettingsUpdate  # noqa: E402


DECLARED_CONFIG = {
    "endpoints": [
        {
            "name": "qwen3-4b-vllm-endpoint",
            "model_type": "custom",
            "serving_runtime": "vllm_beta",
        },
        {"name": "gpt-5-4-mini", "model_type": "fmapi"},
    ],
    "contracts": {
        "qwen3-4b-vllm-endpoint": routes.CONTRACT_OPENAI_GUIDED_JSON,
    },
    "settings": {
        "qwen3-4b-vllm-endpoint": {
            "max_tokens": 777,
            "timeout_s": 90,
            "temperature": 0.0,
        }
    },
}


def _endpoint_summary(name: str, ready: str = "READY", task: str = "llm/v1/chat"):
    return SimpleNamespace(
        name=name,
        state=SimpleNamespace(ready=SimpleNamespace(value=ready)),
        task=task,
    )


class FakeServingEndpoints:
    def __init__(self, endpoints, list_error=None, details=None):
        self._endpoints = endpoints
        self._list_error = list_error
        self._details = details or {}

    def list(self):
        if self._list_error is not None:
            raise self._list_error
        return self._endpoints

    def get(self, name):
        return self._details[name]


class FakeAPIClient:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def do(self, method, path, body):
        assert method == "GET"
        assert path == routes.MODEL_SERVICE_LIST_PATH
        service_names = [
            endpoint.name
            for endpoint in self._endpoints
            if endpoint.name.startswith(
                ("gpt-", "claude-", "gemini-", "glm-", "deepseek-")
            )
        ]
        return {
            "model_services": [
                {
                    "name": f"model-services/system.ai.{name}",
                    "supported_api_types": [routes.MODEL_SERVICE_CHAT_API],
                }
                for name in service_names
            ]
        }


class FakeWorkspaceClient:
    def __init__(self, endpoints, details=None):
        self.serving_endpoints = FakeServingEndpoints(endpoints, details=details)
        self.api_client = FakeAPIClient(endpoints)
        self.config = SimpleNamespace(host="https://workspace.example.com")


class FakeDB:
    """Minimal stand-in for the Lakebase database module."""

    def __init__(self, initial=None, raise_on=None):
        self.store = dict(initial or {})
        self.raise_on = raise_on or set()

    def get_setting(self, key):
        if "get" in self.raise_on:
            raise RuntimeError("db down")
        return self.store.get(key)

    def set_setting(self, key, value):
        if "set" in self.raise_on:
            raise RuntimeError("db down")
        self.store[key] = value


def _reset_memory(monkeypatch):
    monkeypatch.setattr(routes, "_MODEL_SELECTION_MEMORY", None)
    monkeypatch.setattr(routes, "_MODEL_SERVICE_NAMES_CACHE", set())


def _use_declared_config(monkeypatch):
    monkeypatch.setattr(routes, "_load_models_config", lambda: DECLARED_CONFIG)


def test_get_model_selection_prefers_saved_over_default(monkeypatch):
    _reset_memory(monkeypatch)
    db = FakeDB(
        {
            routes.MODEL_SELECTION_SETTING_KEY: [
                {"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}
            ]
        }
    )
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    selection, source = routes._get_model_selection()

    assert source == "saved"
    assert selection == [
        {"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}
    ]


def test_get_model_selection_falls_back_to_config_defaults(monkeypatch):
    _reset_memory(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    monkeypatch.setattr(
        routes,
        "_load_models_config",
        lambda: {
            "endpoints": [
                {"name": "qwen3-4b-vllm-endpoint", "model_type": "custom", "serving_runtime": "vllm_beta"},
                "databricks-gpt-5-4-mini",
            ]
        },
    )

    selection, source = routes._get_model_selection()

    assert source == "default"
    assert selection == [
        {"name": "qwen3-4b-vllm-endpoint", "model_type": "custom", "serving_runtime": "vllm_beta"},
        {"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None},
    ]


def test_model_selection_rejects_an_empty_active_set():
    with pytest.raises(ValidationError):
        ModelSettingsUpdate(endpoints=[])


def test_save_model_selection_uses_memory_when_db_unavailable(monkeypatch):
    _reset_memory(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)

    outcome = routes._save_model_selection(
        [{"name": "gpt-oss-20b", "model_type": "fmapi", "serving_runtime": None}]
    )

    assert outcome == "memory"
    selection, source = routes._get_model_selection()
    assert source == "memory"
    assert selection[0]["name"] == "gpt-oss-20b"


def test_save_model_selection_persists_to_db(monkeypatch):
    _reset_memory(monkeypatch)
    db = FakeDB()
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    outcome = routes._save_model_selection(
        [{"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}]
    )

    assert outcome == "saved"
    assert db.store[routes.MODEL_SELECTION_SETTING_KEY][0]["name"] == "gpt-5-4-mini"
    assert routes._MODEL_SELECTION_MEMORY == [
        {"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}
    ]


def test_get_model_selection_uses_cached_saved_selection_on_db_failure(monkeypatch):
    cached = [
        {"name": "gpt-oss-20b", "model_type": "fmapi", "serving_runtime": None}
    ]
    monkeypatch.setattr(routes, "_MODEL_SELECTION_MEMORY", cached)
    monkeypatch.setattr(routes, "_get_db_module", lambda: FakeDB(raise_on={"get"}))

    selection, source = routes._get_model_selection()

    assert selection == cached
    assert source == "memory"


def test_save_model_selection_falls_back_to_memory_on_db_error(monkeypatch):
    _reset_memory(monkeypatch)
    db = FakeDB(raise_on={"set"})
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    outcome = routes._save_model_selection(
        [{"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}]
    )

    assert outcome == "memory"


def test_update_model_settings_deduplicates_and_uses_canonical_metadata(monkeypatch):
    _reset_memory(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [
                _endpoint_summary("qwen3-4b-vllm-endpoint"),
                _endpoint_summary("gpt-5-4-mini"),
            ]
        ),
    )

    req = ModelSettingsUpdate(
        endpoints=[
            ModelSelectionEntry(
                name="qwen3-4b-vllm-endpoint",
                model_type="fmapi",
                serving_runtime=None,
            ),
            ModelSelectionEntry(
                name="gpt-5-4-mini",
                model_type="custom",
                serving_runtime="standard",
            ),
            ModelSelectionEntry(
                name="qwen3-4b-vllm-endpoint",
                model_type="custom",
                serving_runtime="standard",
            ),
        ]
    )

    result = asyncio.run(routes.update_model_settings(req))

    assert [e.name for e in result.endpoints] == [
        "qwen3-4b-vllm-endpoint",
        "gpt-5-4-mini",
    ]
    custom_entry = next(e for e in result.endpoints if e.name == "qwen3-4b-vllm-endpoint")
    assert custom_entry.model_type == "custom"
    assert custom_entry.serving_runtime == "vllm_beta"
    fmapi_entry = next(e for e in result.endpoints if e.name == "gpt-5-4-mini")
    assert fmapi_entry.model_type == "fmapi"
    assert fmapi_entry.serving_runtime is None
    assert result.source == "memory"
    assert routes._MODEL_SELECTION_MEMORY == [
        {
            "name": "qwen3-4b-vllm-endpoint",
            "model_type": "custom",
            "serving_runtime": "vllm_beta",
        },
        {
            "name": "gpt-5-4-mini",
            "model_type": "fmapi",
            "serving_runtime": None,
        },
    ]


def test_update_model_settings_rejects_non_workspace_endpoint(monkeypatch):
    _reset_memory(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [
                _endpoint_summary("qwen3-4b-vllm-endpoint"),
                _endpoint_summary("gpt-5-4-mini"),
            ]
        ),
    )

    result = asyncio.run(
        routes.update_model_settings(
            ModelSettingsUpdate(
                endpoints=[
                    ModelSelectionEntry(
                        name="undeclared-secret-endpoint",
                        model_type="custom",
                        serving_runtime="standard",
                    )
                ]
            )
        )
    )

    assert result.status_code == 422
    assert json.loads(result.body)["error_code"] == "MODEL_NOT_AVAILABLE"
    assert routes._MODEL_SELECTION_MEMORY is None


def test_update_model_settings_accepts_discovered_chat_endpoint(monkeypatch):
    _reset_memory(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [_endpoint_summary("gpt-6-astra")]
        ),
    )

    result = asyncio.run(
        routes.update_model_settings(
            ModelSettingsUpdate(
                endpoints=[
                    ModelSelectionEntry(
                        name="gpt-6-astra",
                        model_type="custom",
                        serving_runtime="standard",
                    )
                ]
            )
        )
    )

    assert result.source == "memory"
    assert [entry.model_dump() for entry in result.endpoints] == [
        {
            "name": "gpt-6-astra",
            "model_type": "fmapi",
            "serving_runtime": None,
        }
    ]


def test_list_workspace_endpoints_flags_current_selection(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    db = FakeDB(
        {
            routes.MODEL_SELECTION_SETTING_KEY: [
                {"name": "gpt-5-4-mini", "model_type": "fmapi", "serving_runtime": None}
            ]
        }
    )
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)
    workspace = FakeWorkspaceClient(
        [
            _endpoint_summary("gpt-5-4-mini"),
            _endpoint_summary("qwen3-4b-vllm-endpoint", ready="NOT_READY"),
            _endpoint_summary("undeclared-workspace-endpoint"),
        ]
    )

    endpoints = routes._list_workspace_endpoints(workspace)

    by_name = {e.name: e for e in endpoints}
    assert by_name["gpt-5-4-mini"].selected is True
    assert by_name["gpt-5-4-mini"].model_type == "fmapi"
    assert by_name["qwen3-4b-vllm-endpoint"].selected is False
    assert by_name["qwen3-4b-vllm-endpoint"].ready is False
    assert by_name["undeclared-workspace-endpoint"].model_type == "custom"
    assert by_name["undeclared-workspace-endpoint"].ready is True
    # Custom endpoints sort before fmapi ones.
    assert endpoints[0].model_type == "custom"


def test_list_workspace_endpoints_includes_declared_endpoint_absent_from_workspace(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)

    endpoints = routes._list_workspace_endpoints(
        FakeWorkspaceClient([_endpoint_summary("qwen3-4b-vllm-endpoint")])
    )

    by_name = {entry.name: entry for entry in endpoints}
    assert set(by_name) == {
        "qwen3-4b-vllm-endpoint",
        "gpt-5-4-mini",
    }
    assert by_name["gpt-5-4-mini"].ready is True


def test_workspace_endpoint_list_never_exposes_system_ai_entity_names(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)

    endpoints = routes._list_workspace_endpoints(
        FakeWorkspaceClient(
            [
                _endpoint_summary("system.ai.gpt-5-4-mini"),
                _endpoint_summary("gpt-5-4-mini"),
            ]
        )
    )

    names = {entry.name for entry in endpoints}
    assert "system.ai.gpt-5-4-mini" not in names
    assert "gpt-5-4-mini" in names


def test_saved_system_ai_entity_names_migrate_to_short_service_ids(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    db = FakeDB(
        {
            routes.MODEL_SELECTION_SETTING_KEY: [
                {
                    "name": "system.ai.gpt-5-4-mini",
                    "model_type": "fmapi",
                    "serving_runtime": None,
                }
            ]
        }
    )
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    selection, source = routes._get_model_selection()

    assert source == "saved"
    assert selection == [
        {
            "name": "gpt-5-4-mini",
            "model_type": "fmapi",
            "serving_runtime": None,
        }
    ]


def test_get_model_selection_filters_stale_entries_and_canonicalizes_saved_metadata(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    db = FakeDB(
        {
            routes.MODEL_SELECTION_SETTING_KEY: [
                {
                    "name": "qwen3-4b-vllm-endpoint",
                    "model_type": "fmapi",
                    "serving_runtime": "standard",
                },
                {
                    "name": "removed-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "standard",
                },
            ]
        }
    )
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    selection, source = routes._get_model_selection()

    assert source == "saved"
    assert selection == [
        {
            "name": "qwen3-4b-vllm-endpoint",
            "model_type": "custom",
            "serving_runtime": "vllm_beta",
        },
        {
            "name": "removed-endpoint",
            "model_type": "custom",
            "serving_runtime": None,
        },
    ]


def test_get_model_selection_falls_back_to_defaults_when_saved_set_is_all_stale(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    db = FakeDB(
        {
            routes.MODEL_SELECTION_SETTING_KEY: [
                {
                    "name": "removed-endpoint",
                    "model_type": "custom",
                    "serving_runtime": "standard",
                }
            ]
        }
    )
    monkeypatch.setattr(routes, "_get_db_module", lambda: db)

    selection, source = routes._get_model_selection()

    assert source == "saved"
    assert selection == [
        {
            "name": "removed-endpoint",
            "model_type": "custom",
            "serving_runtime": None,
        }
    ]


def test_config_contracts_and_settings_survive_model_selection_update(monkeypatch):
    _reset_memory(monkeypatch)
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(routes, "_get_db_module", lambda: None)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient([_endpoint_summary("qwen3-4b-vllm-endpoint")]),
    )

    result = asyncio.run(
        routes.update_model_settings(
            ModelSettingsUpdate(
                endpoints=[
                    ModelSelectionEntry(
                        name="qwen3-4b-vllm-endpoint",
                        model_type="fmapi",
                        serving_runtime=None,
                    )
                ]
            )
        )
    )
    models = routes._load_configured_models()
    route_path, payload = routes._build_inference_request(
        endpoint_name="qwen3-4b-vllm-endpoint",
        prompt="Extract fields",
        schema_json='{"type":"object","properties":{}}',
        settings=routes._load_endpoint_settings(),
        contracts=routes._load_endpoint_contracts(),
    )

    assert result.source == "memory"
    assert [(model.name, model.model_type, model.serving_runtime, model.max_tokens) for model in models] == [
        ("qwen3-4b-vllm-endpoint", "custom", "vllm_beta", 777)
    ]
    assert routes._load_endpoint_contracts() == DECLARED_CONFIG["contracts"]
    assert routes._load_endpoint_settings() == DECLARED_CONFIG["settings"]
    assert route_path == "/serving-endpoints/chat/completions"
    assert payload["guided_decoding_backend"] == "xgrammar:no-fallback"
    assert payload["max_tokens"] == 777


def test_lakebase_is_unconfigured_without_required_environment(monkeypatch):
    monkeypatch.delenv("LAKEBASE_DB_HOST", raising=False)
    monkeypatch.delenv("LAKEBASE_DB_NAME", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.delenv("LAKEBASE_ENDPOINT", raising=False)

    assert database.is_configured() is False
    assert routes._get_db_module() is None


def test_lakebase_is_configured_with_platform_binding(monkeypatch):
    monkeypatch.setenv("PGHOST", "lakebase.example.com")
    monkeypatch.setenv("PGDATABASE", "clinical_extractor")
    monkeypatch.setenv("LAKEBASE_ENDPOINT", "projects/p/branches/b/endpoints/e")

    assert database.is_configured() is True
    assert routes._get_db_module() is database


def test_model_selection_keys_are_stable_and_user_scoped():
    alice = routes._model_selection_setting_key("alice@example.com")
    alice_again = routes._model_selection_setting_key("alice@example.com")
    bob = routes._model_selection_setting_key("bob@example.com")

    assert alice == alice_again
    assert alice != bob
    assert "alice@example.com" not in alice


def test_warmup_ready_vllm_still_pings_runtime(monkeypatch):
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [_endpoint_summary("qwen3-4b-vllm-endpoint", ready="READY")]
        ),
    )

    monkeypatch.setattr(
        routes,
        "_build_inference_request",
        lambda **_kwargs: ("/warm", {"synthetic": True}),
    )
    invoked = []

    async def fake_invoke(*args, **kwargs):
        invoked.append((args, kwargs))
        return {"choices": []}

    monkeypatch.setattr(routes, "_invoke_endpoint", fake_invoke)

    result = asyncio.run(
        routes.warmup_workspace_endpoint("qwen3-4b-vllm-endpoint")
    )

    assert result.status == "ready"
    assert invoked


def test_warmup_reports_stopped_deployment_without_invocation(monkeypatch):
    _use_declared_config(monkeypatch)
    endpoint_name = "qwen3-4b-vllm-endpoint"
    detail = SimpleNamespace(
        config=SimpleNamespace(
            served_entities=[
                SimpleNamespace(
                    state=SimpleNamespace(deployment="DEPLOYMENT_STOPPED")
                )
            ]
        )
    )
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [_endpoint_summary(endpoint_name, ready="NOT_READY")],
            details={endpoint_name: detail},
        ),
    )

    async def unexpected_invoke(*_args, **_kwargs):
        raise AssertionError("a stopped deployment cannot be warmed by inference")

    monkeypatch.setattr(routes, "_invoke_endpoint", unexpected_invoke)

    result = asyncio.run(routes.warmup_workspace_endpoint(endpoint_name))
    body = json.loads(result.body)

    assert result.status_code == 409
    assert body["error_code"] == "MODEL_DEPLOYMENT_STOPPED"
    assert "Start it in Databricks Model Serving" in body["message"]


def test_warmup_cold_vllm_sends_tiny_synthetic_request(monkeypatch):
    _use_declared_config(monkeypatch)
    workspace = FakeWorkspaceClient(
        [_endpoint_summary("qwen3-4b-vllm-endpoint", ready="NOT_READY")]
    )
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)
    build_calls = []
    invoke_calls = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return "/serving-endpoints/chat/completions", {"synthetic": True}

    async def fake_invoke(client, route_path, payload, timeout_s):
        invoke_calls.append((client, route_path, payload, timeout_s))
        return {"choices": []}

    monkeypatch.setattr(routes, "_build_inference_request", fake_build)
    monkeypatch.setattr(routes, "_invoke_endpoint", fake_invoke)

    result = asyncio.run(
        routes.warmup_workspace_endpoint("qwen3-4b-vllm-endpoint")
    )

    assert result.status == "ready"
    assert build_calls == [
        {
            "endpoint_name": "qwen3-4b-vllm-endpoint",
            "prompt": 'Return {"ok": true}.',
            "schema_json": json.dumps(
                {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                }
            ),
            "system_prompt": "Return only the requested JSON object.",
            "max_tokens_override": 16,
            "temperature_override": 0.0,
        }
    ]
    assert invoke_calls == [
        (
            workspace,
            "/serving-endpoints/chat/completions",
            {"synthetic": True},
            routes.MODEL_WARMUP_TIMEOUT_S,
        )
    ]


def test_warmup_rejects_fmapi_endpoint(monkeypatch):
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [_endpoint_summary("gpt-5-4-mini", ready="NOT_READY")]
        ),
    )

    result = asyncio.run(
        routes.warmup_workspace_endpoint("gpt-5-4-mini")
    )

    assert result.status_code == 422
    assert json.loads(result.body)["error_code"] == "MODEL_NOT_WARMABLE"


def test_warmup_timeout_reports_warming(monkeypatch):
    _use_declared_config(monkeypatch)
    workspace = FakeWorkspaceClient(
        [_endpoint_summary("qwen3-4b-vllm-endpoint", ready="NOT_READY")]
    )
    monkeypatch.setattr(routes, "_get_workspace_client", lambda: workspace)
    monkeypatch.setattr(
        routes,
        "_build_inference_request",
        lambda **_kwargs: ("/warm", {"synthetic": True}),
    )

    async def timeout(*_args, **_kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(routes, "_invoke_endpoint", timeout)

    result = asyncio.run(
        routes.warmup_workspace_endpoint("qwen3-4b-vllm-endpoint")
    )

    assert result.status == "warming"
    assert "still starting" in result.message


def test_warmup_permission_error_does_not_leak_detail(monkeypatch):
    _use_declared_config(monkeypatch)
    monkeypatch.setattr(
        routes,
        "_get_workspace_client",
        lambda: FakeWorkspaceClient(
            [_endpoint_summary("qwen3-4b-vllm-endpoint", ready="NOT_READY")]
        ),
    )
    monkeypatch.setattr(
        routes,
        "_build_inference_request",
        lambda **_kwargs: ("/warm", {"synthetic": True}),
    )

    class PermissionDeniedSecret(Exception):
        status_code = 403

    async def denied(*_args, **_kwargs):
        raise PermissionDeniedSecret("secret upstream authorization detail")

    monkeypatch.setattr(routes, "_invoke_endpoint", denied)

    result = asyncio.run(
        routes.warmup_workspace_endpoint("qwen3-4b-vllm-endpoint")
    )
    body = json.loads(result.body)

    assert result.status_code == 403
    assert body["error_code"] == "PERMISSION_DENIED"
    assert "secret" not in json.dumps(body).lower()
