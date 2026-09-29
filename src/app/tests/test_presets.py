import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from api import database, routes  # noqa: E402
from api.presets import PRESETS  # noqa: E402
from api.schemas import Preset, PresetCreate  # noqa: E402


EXPECTED_PRESET_IDS = {
    "medication_extraction",
    "diagnosis_extraction",
    "vitals_extraction",
    "clinical_summary",
    "lab_results",
    "allergy_extraction",
    "procedure_extraction",
    "imaging_findings",
    "medication_reconciliation",
    "discharge_follow_up_plan",
    "veterinary_lab_panel",
    "imaging_device_report",
    "adverse_event_reporting",
    "clinical_trial_screening",
    "drug_interaction_screening",
    "ehr_note_extraction",
    "prior_authorization",
    "disease_mention_extraction",
    "snomed_omop_mapping",
}


def _has_type(schema: dict[str, Any], expected_type: str) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return expected_type in schema_type
    return schema_type == expected_type


def _assert_strict_objects(schema: dict[str, Any]) -> None:
    if _has_type(schema, "object"):
        properties = schema.get("properties")
        assert isinstance(properties, dict)
        assert schema.get("additionalProperties") is False
        assert schema.get("required") == list(properties.keys())
        for child in properties.values():
            _assert_strict_objects(child)

    if _has_type(schema, "array"):
        items = schema.get("items")
        assert isinstance(items, dict)
        _assert_strict_objects(items)


def _assert_schema_descriptions(schema: dict[str, Any]) -> None:
    description = schema.get("description")
    assert isinstance(description, str)
    assert description.strip()
    assert len(description) <= 120

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for child in properties.values():
            assert isinstance(child, dict)
            _assert_schema_descriptions(child)

    items = schema.get("items")
    if isinstance(items, dict):
        _assert_schema_descriptions(items)


def _contains_source_text(schema: dict[str, Any]) -> bool:
    if _has_type(schema, "object"):
        properties = schema.get("properties", {})
        if "source_text" in properties:
            return True
        return any(_contains_source_text(child) for child in properties.values())

    if _has_type(schema, "array"):
        items = schema.get("items")
        return isinstance(items, dict) and _contains_source_text(items)

    return False


def _assert_visual_editor_nesting(schema: dict[str, Any], field_depth: int = 0) -> None:
    """Mirror the Visual editor's supported object/array nesting for built-ins."""
    assert _has_type(schema, "object")
    properties = schema.get("properties", {})
    assert isinstance(properties, dict)

    for field_name, field_schema in properties.items():
        assert isinstance(field_schema, dict), field_name
        if _has_type(field_schema, "object"):
            assert field_depth < 2, field_name
            _assert_visual_editor_nesting(field_schema, field_depth + 1)
        elif _has_type(field_schema, "array"):
            assert field_depth < 2, field_name
            items = field_schema.get("items")
            assert isinstance(items, dict), field_name
            assert not _has_type(items, "array"), field_name
            if _has_type(items, "object"):
                assert field_depth < 1, field_name
                _assert_visual_editor_nesting(items, field_depth + 1)


def test_builtin_presets_cover_expected_catalog():
    preset_ids = {preset["id"] for preset in PRESETS}
    assert preset_ids == EXPECTED_PRESET_IDS
    assert len(PRESETS) == len(EXPECTED_PRESET_IDS)
    assert len({preset["name"] for preset in PRESETS}) == len(PRESETS)


def test_builtin_presets_have_descriptions_and_examples():
    for preset in PRESETS:
        assert preset["description"].strip()
        assert len(preset["example_prompt"].strip()) > 40
        assert len(preset["system_prompt"].strip()) > 120
        assert "source text" in preset["system_prompt"].lower()
        assert "schema" in preset["system_prompt"].lower()

    assert len({preset["system_prompt"] for preset in PRESETS}) == len(PRESETS)


def test_clinical_concept_extraction_is_evidence_grounded_and_uncoded():
    preset = next(preset for preset in PRESETS if preset["id"] == "snomed_omop_mapping")
    schema = json.loads(preset["schema_json"])
    entity_properties = schema["properties"]["entities"]["items"]["properties"]

    assert set(entity_properties) == {
        "source_text",
        "entity_type",
        "assertion",
        "temporality",
        "experiencer",
        "body_site",
        "laterality",
    }
    assert not {
        "concept_id",
        "concept_code",
        "concept_name",
        "vocabulary_id",
        "omop_concept_id",
    } & entity_properties.keys()

    system_prompt = preset["system_prompt"].lower()
    assert "mention extraction, not terminology normalization or coding" in system_prompt
    assert "do not generate, recall, infer, or output snomed ct, omop" in system_prompt
    assert "separate, versioned vocabulary lookup" in system_prompt
    assert "copy source_text verbatim" in system_prompt

    example = preset["example_prompt"].lower()
    assert "denies chest pain" in example
    assert "prior echocardiogram" in example
    assert "mother had breast cancer" in example
    assert "repeat echocardiogram tomorrow" in example
    assert "only if the pleural effusions worsen" in example


def test_builtin_presets_satisfy_api_models_and_field_limits():
    for preset in PRESETS:
        response_model = Preset(**preset, is_builtin=True)
        create_model = PresetCreate(
            name=response_model.name,
            description=response_model.description,
            schema_json=response_model.json_schema,
            example_prompt=response_model.example_prompt,
            system_prompt=response_model.system_prompt,
        )
        assert len(create_model.json_schema) <= 20_000
        assert len(create_model.example_prompt) <= 10_000
        assert len(create_model.system_prompt) <= 4_000


def test_builtin_preset_schemas_are_strict_compatible():
    for preset in PRESETS:
        schema = json.loads(preset["schema_json"])
        assert schema["type"] == "object"
        _assert_strict_objects(schema)


def test_builtin_preset_schema_fields_have_short_descriptions():
    for preset in PRESETS:
        _assert_schema_descriptions(json.loads(preset["schema_json"]))


def test_builtin_preset_schemas_fit_the_visual_editor():
    for preset in PRESETS:
        _assert_visual_editor_nesting(json.loads(preset["schema_json"]))


def test_only_evidence_heavy_presets_include_source_text():
    expected_source_text_presets = {
        "snomed_omop_mapping",
        "diagnosis_extraction",
        "procedure_extraction",
        "imaging_findings",
    }

    actual_source_text_presets = {
        preset["id"]
        for preset in PRESETS
        if _contains_source_text(json.loads(preset["schema_json"]))
    }

    assert actual_source_text_presets == expected_source_text_presets


def test_builtin_presets_are_compatible_with_ai_extract():
    for preset in PRESETS:
        converted = routes._convert_json_schema_for_ai_extract(preset["schema_json"])
        assert converted.schema
        assert isinstance(converted.warnings, list)


def test_builtin_presets_build_requests_for_every_default_serving_model():
    config = routes._load_models_config()
    assert config is not None
    defaults = routes._default_selection_from_config(config)
    assert {entry["name"] for entry in defaults} == {
        "medgemma-1-5-4b-vllm-endpoint",
        "qwen3-4b-vllm-endpoint",
        "qwen3-4b-finetuned-vllm-endpoint",
        "gpt-5-4-mini",
    }

    for preset in PRESETS:
        schema = json.loads(preset["schema_json"])
        for model in defaults:
            path, payload = routes._build_inference_request(
                model["name"],
                preset["example_prompt"],
                preset["schema_json"],
                preset["system_prompt"],
                contracts=config.get("contracts"),
                settings=config.get("settings"),
            )
            assert path
            assert payload["messages"][0] == {
                "role": "system",
                "content": preset["system_prompt"],
            }
            assert payload["messages"][1] == {
                "role": "user",
                "content": preset["example_prompt"],
            }
            if model["model_type"] == "fmapi":
                sent_schema = payload["response_format"]["json_schema"]["schema"]
            else:
                sent_schema = payload["guided_json"]
                assert payload["guided_decoding_backend"] == "xgrammar:no-fallback"
            assert sent_schema == schema


class _RecordingCursor:
    def __init__(self, returned_row: dict[str, Any] | None = None):
        self.calls: list[tuple[str, tuple[Any, ...] | None]] = []
        self.returned_row = returned_row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement: str, params: tuple[Any, ...] | None = None):
        self.calls.append((statement, params))

    def fetchone(self):
        return self.returned_row


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor):
        self._cursor = cursor

    def cursor(self, **_kwargs):
        return self._cursor


def _recording_db(cursor: _RecordingCursor):
    @contextmanager
    def get_db():
        yield _RecordingConnection(cursor)

    return get_db


def test_lakebase_migration_and_seed_persist_system_prompts(monkeypatch):
    migration_cursor = _RecordingCursor()
    monkeypatch.setattr(database, "get_db", _recording_db(migration_cursor))
    database.init_db()
    statements = "\n".join(statement for statement, _ in migration_cursor.calls)
    assert "ADD COLUMN IF NOT EXISTS system_prompt" in statements

    seed_cursor = _RecordingCursor()
    monkeypatch.setattr(database, "get_db", _recording_db(seed_cursor))
    database.seed_defaults()
    assert len(seed_cursor.calls) == len(PRESETS)
    for preset, (_statement, params) in zip(PRESETS, seed_cursor.calls, strict=True):
        assert params is not None
        assert params[5] == preset["system_prompt"]


def test_custom_preset_create_round_trips_system_prompt(monkeypatch):
    returned_row = {
        "id": "custom-1",
        "name": "Custom trial preset",
        "description": "A custom screening workflow",
        "schema_json": PRESETS[0]["schema_json"],
        "example_prompt": "Patient note",
        "system_prompt": "Use the custom screening instructions.",
        "is_builtin": False,
    }
    cursor = _RecordingCursor(returned_row)
    monkeypatch.setattr(database, "get_db", _recording_db(cursor))
    request = PresetCreate(
        name=returned_row["name"],
        description=returned_row["description"],
        schema_json=returned_row["schema_json"],
        example_prompt=returned_row["example_prompt"],
        system_prompt=returned_row["system_prompt"],
    )

    created = database.create_preset(request)

    assert created.system_prompt == returned_row["system_prompt"]
    assert cursor.calls[0][1] is not None
    assert cursor.calls[0][1][5] == returned_row["system_prompt"]
