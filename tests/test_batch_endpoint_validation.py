"""Surface endpoint compatibility errors before loading the GPU model."""

import ast
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


@pytest.fixture
def validate_row():
    notebook = Path(__file__).resolve().parents[1] / "src/models/medgemma/batch/02_batch_inference_udf.py"
    cell = next(cell for cell in notebook.read_text().split("# COMMAND ----------")
                if "def validate_endpoint_row(" in cell)
    definition = next(node for node in ast.parse(cell).body
                      if isinstance(node, ast.FunctionDef) and node.name == "validate_endpoint_row")
    namespace = {"json": json, "validate": validate}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["validate_endpoint_row"]


def row(content='{"heart_rate":72}', error=None):
    return {
        "schema": json.dumps({"type": "object", "properties": {"heart_rate": {"type": "integer"}},
                              "required": ["heart_rate"]}),
        "_ai_response": {
            "errorMessage": error,
            "result": None if error else {"choices": [{"message": {"content": content}}]},
        },
    }


def test_valid_chat_completion(validate_row):
    validate_row(row())


def test_remote_parse_error_is_preserved(validate_row):
    with pytest.raises(RuntimeError, match="REMOTE_FUNCTION_HTTP_RESULT_PARSE_ERROR"):
        validate_row(row(error="REMOTE_FUNCTION_HTTP_RESULT_PARSE_ERROR: expected predictions"))


def test_missing_chat_completion_is_reported(validate_row):
    value = row()
    value["_ai_response"]["result"] = None
    with pytest.raises(RuntimeError, match="no chat completion"):
        validate_row(value)


def test_schema_invalid_completion_is_rejected(validate_row):
    with pytest.raises(ValidationError):
        validate_row(row('{"heart_rate":"72"}'))
