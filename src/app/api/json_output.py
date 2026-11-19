"""JSON output normalization and adaptation to strict endpoint schemas."""

import json

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError


def strip_json_markdown_fence(content: str) -> str:
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


def reject_non_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON")


def format_schema_validation_detail(error: JsonSchemaValidationError) -> str:
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


def _schema_allows_null(schema: dict) -> bool:
    schema_type = schema.get("type")
    return (
        schema_type == "null"
        or isinstance(schema_type, list)
        and "null" in schema_type
        or isinstance(schema.get("enum"), list)
        and None in schema["enum"]
    )


def _make_schema_nullable(schema: dict) -> dict:
    result = dict(schema)
    schema_type = result.get("type")
    if isinstance(schema_type, str) and schema_type != "null":
        result["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        result["type"] = [*schema_type, "null"]
    if isinstance(result.get("enum"), list) and None not in result["enum"]:
        result["enum"] = [*result["enum"], None]
    return result


def strict_response_schema(schema: dict) -> dict:
    result = dict(schema)
    schema_type = schema.get("type")
    concrete_types = schema_type if isinstance(schema_type, list) else [schema_type]

    if "object" in concrete_types and isinstance(schema.get("properties"), dict):
        original_required = set(schema.get("required", []))
        properties = {}
        for name, child in schema["properties"].items():
            transformed = strict_response_schema(child)
            if name not in original_required and not _schema_allows_null(transformed):
                transformed = _make_schema_nullable(transformed)
            properties[name] = transformed
        result["properties"] = properties
        result["required"] = list(properties)
        result["additionalProperties"] = False

    if "array" in concrete_types and isinstance(schema.get("items"), dict):
        result["items"] = strict_response_schema(schema["items"])

    return result


def normalize_prediction_to_schema(value, schema: dict):
    if value is None:
        return None

    schema_type = schema.get("type")
    concrete_types = schema_type if isinstance(schema_type, list) else [schema_type]

    if "object" in concrete_types and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        normalized = {}
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if not isinstance(child_schema, dict):
                normalized[key] = child_value
                continue
            child_value = normalize_prediction_to_schema(child_value, child_schema)
            if child_value is None and key not in required and not _schema_allows_null(child_schema):
                continue
            normalized[key] = child_value
        return normalized

    if "array" in concrete_types and isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        return [
            normalized
            for item in value
            if (
                (normalized := normalize_prediction_to_schema(item, item_schema)) is not None
                or _schema_allows_null(item_schema)
            )
        ]

    return value
