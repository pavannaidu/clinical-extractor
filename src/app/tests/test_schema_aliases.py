import importlib.util
import sys
import warnings
from pathlib import Path

from pydantic import warnings as pydantic_warnings


APP_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_PATH = APP_ROOT / "api" / "schemas.py"
VALID_SCHEMA = '{"type":"object","properties":{}}'


def _load_schemas_with_fresh_model_classes():
    module_name = "_clinical_extractor_schema_warning_probe"
    spec = importlib.util.spec_from_file_location(module_name, SCHEMAS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_schema_json_aliases_are_warning_free_and_preserve_the_wire_contract():
    unsupported_field_warning = getattr(
        pydantic_warnings, "UnsupportedFieldAttributeWarning", None
    )

    with warnings.catch_warnings():
        if unsupported_field_warning is not None:
            warnings.simplefilter("error", unsupported_field_warning)

        schemas = _load_schemas_with_fresh_model_classes()
        cases = [
            (
                schemas.InferenceRequest,
                {"prompt": "Extract", "schema_json": VALID_SCHEMA},
            ),
            (
                schemas.LoadTestSampleResponse,
                {
                    "name": "Sample",
                    "description": "Description",
                    "prompt": "Extract",
                    "schema_json": VALID_SCHEMA,
                    "system_prompt": "Return JSON",
                },
            ),
            (
                schemas.LoadTestRequest,
                {
                    "model": "test-endpoint",
                    "schema_json": VALID_SCHEMA,
                },
            ),
            (
                schemas.Preset,
                {
                    "id": "preset-1",
                    "name": "Sample",
                    "description": "Description",
                    "schema_json": VALID_SCHEMA,
                    "example_prompt": "Extract",
                },
            ),
            (
                schemas.PresetCreate,
                {"name": "Sample", "schema_json": VALID_SCHEMA},
            ),
            (schemas.PresetUpdate, {"schema_json": VALID_SCHEMA}),
        ]

        for model_class, payload in cases:
            alias_field = model_class.model_fields["json_schema"]
            assert alias_field.alias == "schema_json"
            assert alias_field.alias_priority == 2

            instance = model_class.model_validate(payload)
            assert instance.json_schema == VALID_SCHEMA
            assert instance.model_dump(by_alias=True)["schema_json"] == VALID_SCHEMA

            field_name_payload = {**payload, "json_schema": payload["schema_json"]}
            field_name_payload.pop("schema_json")
            by_field_name = model_class.model_validate(field_name_payload)
            assert by_field_name.json_schema == VALID_SCHEMA

            generated_schema = model_class.model_json_schema(by_alias=True)
            assert "schema_json" in generated_schema["properties"]
            assert "json_schema" not in generated_schema["properties"]


def test_fastapi_openapi_uses_schema_json_for_aliased_route_fields():
    unsupported_field_warning = getattr(
        pydantic_warnings, "UnsupportedFieldAttributeWarning", None
    )
    with warnings.catch_warnings():
        if unsupported_field_warning is not None:
            warnings.simplefilter("error", unsupported_field_warning)

        from main import app

        components = app.openapi()["components"]["schemas"]

    for model_name in (
        "InferenceRequest",
        "LoadTestRequest",
        "LoadTestSampleResponse",
        "Preset",
        "PresetCreate",
        "PresetUpdate",
    ):
        properties = components[model_name]["properties"]
        assert "schema_json" in properties
        assert "json_schema" not in properties


def test_inference_route_accepts_schema_json_without_field_alias_warning(monkeypatch):
    from api import routes
    from fastapi.testclient import TestClient
    from main import app

    async def _fake_invoke_endpoint(*_args, **_kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(routes, "_load_allowed_endpoints", lambda: ["test-endpoint"])
    monkeypatch.setattr(routes, "_get_workspace_client", object)
    monkeypatch.setattr(routes, "_load_endpoint_settings", lambda: {})
    monkeypatch.setattr(routes, "_invoke_endpoint", _fake_invoke_endpoint)

    unsupported_field_warning = getattr(
        pydantic_warnings, "UnsupportedFieldAttributeWarning", None
    )
    with warnings.catch_warnings():
        if unsupported_field_warning is not None:
            warnings.simplefilter("error", unsupported_field_warning)

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/inference",
                json={
                    "prompt": "Extract",
                    "schema_json": VALID_SCHEMA,
                    "model": "test-endpoint",
                },
            )

    assert response.status_code == 200
    assert response.json()["prediction"] == {}
