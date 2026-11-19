"""Translate JSON Schema requests and parse AI Extract values and evidence."""

import json
from dataclasses import dataclass

from .schemas import (
    ExtractionCitation,
    ExtractionCitationBox,
    ExtractionEvidence,
    ExtractionEvidenceField,
)

AI_EXTRACT_TARGET_ID = "databricks-ai-extract"
AI_EXTRACT_DISPLAY_NAME = "Agent Bricks · Information Extraction"
AI_EXTRACT_PATH = "/api/2.0/ai-functions/ai-extract"
AI_EXTRACT_VERSION = "2.1"
AI_EXTRACT_MAX_FIELDS = 256
AI_EXTRACT_MAX_DEPTH = 12
AI_EXTRACT_MAX_ENUM_VALUES = 500
AI_EXTRACT_MAX_INSTRUCTIONS_LENGTH = 20_000


class AiExtractSchemaError(ValueError):
    """The requested schema cannot be represented by AI Extract."""


@dataclass
class AiExtractSchema:
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
            raise AiExtractSchemaError(
                f"{path}.type must be one supported type with optional null."
            )
        concrete = [item for item in value if item != "null"]
        if len(concrete) != 1 or not isinstance(concrete[0], str):
            raise AiExtractSchemaError(
                f"{path}.type must be one supported type with optional null."
            )
        concrete_type = concrete[0]
        nullable = True
    else:
        raise AiExtractSchemaError(f"{path}.type is required for AI Extract.")

    supported = {"object"} if root else {
        "string",
        "integer",
        "number",
        "boolean",
        "object",
        "array",
    }
    if concrete_type not in supported:
        raise AiExtractSchemaError(
            f"{path}.type '{concrete_type}' is not supported by AI Extract."
        )
    return concrete_type, nullable


def convert_json_schema_for_ai_extract(schema_json: str) -> AiExtractSchema:
    parsed = json.loads(schema_json)
    if not isinstance(parsed, dict):
        raise AiExtractSchemaError("The schema root must be an object.")
    root_type, _ = _ai_extract_schema_type(parsed.get("type"), "$", root=True)
    if root_type != "object":
        raise AiExtractSchemaError("The schema root must have type object.")

    field_count = 0
    nullable_seen = False
    required_seen = False
    validation_only_keys: set[str] = set()
    enum_value_count = 0

    def reject_shape_keywords(node: dict, path: str) -> None:
        found = sorted(_AI_EXTRACT_SHAPE_KEYWORDS.intersection(node))
        if found:
            raise AiExtractSchemaError(
                f"{path} uses unsupported JSON Schema keyword '{found[0]}'."
            )

    def record_omitted_keywords(node: dict) -> None:
        for key in node:
            if key not in _AI_EXTRACT_SENT_KEYS and key not in _AI_EXTRACT_SHAPE_KEYWORDS:
                validation_only_keys.add(key)

    def convert_object(node: dict, path: str, depth: int) -> dict[str, dict]:
        nonlocal field_count, nullable_seen, required_seen
        if depth > AI_EXTRACT_MAX_DEPTH:
            raise AiExtractSchemaError(
                f"{path} exceeds AI Extract's {AI_EXTRACT_MAX_DEPTH}-level nesting limit."
            )
        reject_shape_keywords(node, path)
        record_omitted_keywords(node)
        if "required" in node:
            required_seen = True
        properties = node.get("properties")
        if not isinstance(properties, dict):
            raise AiExtractSchemaError(f"{path}.properties must be an object.")

        converted: dict[str, dict] = {}
        for field_name, field_schema in properties.items():
            field_path = f"{path}.properties[{field_name!r}]"
            if not isinstance(field_name, str) or not field_name:
                raise AiExtractSchemaError(f"{path} contains an empty field name.")
            if len(field_name) > 150:
                raise AiExtractSchemaError(
                    f"{field_path} exceeds AI Extract's 150-character field-name limit."
                )
            if not isinstance(field_schema, dict):
                raise AiExtractSchemaError(f"{field_path} must be an object schema.")
            field_count += 1
            if field_count > AI_EXTRACT_MAX_FIELDS:
                raise AiExtractSchemaError(
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
            raise AiExtractSchemaError(f"{path}.description must be a string.")

        result: dict = {"type": field_type}
        if isinstance(description, str) and description.strip():
            result["description"] = description.strip()

        if "enum" in node:
            if field_type != "string" or not isinstance(node["enum"], list):
                raise AiExtractSchemaError(
                    f"{path}.enum must be a list of string labels."
                )
            labels = [value for value in node["enum"] if value is not None]
            if not labels or any(not isinstance(value, str) for value in labels):
                raise AiExtractSchemaError(
                    f"{path}.enum must contain at least one string label."
                )
            enum_value_count += len(labels)
            if enum_value_count > AI_EXTRACT_MAX_ENUM_VALUES:
                raise AiExtractSchemaError(
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
                raise AiExtractSchemaError(
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
    return AiExtractSchema(
        schema=converted_schema,
        root_description=(
            root_description.strip() if isinstance(root_description, str) else ""
        ),
        warnings=warnings,
    )


def build_ai_extract_request(
    *,
    prompt: str,
    schema_json: str,
    system_prompt: str,
    precision: bool,
    enable_citations: bool = False,
    enable_confidence_scores: bool = False,
) -> tuple[str, dict, list[str]]:
    converted = convert_json_schema_for_ai_extract(schema_json)
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
            raise AiExtractSchemaError(
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


class AiExtractResponseError(ValueError):
    """AI Extract returned a missing, malformed, or unsuccessful response."""


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


def parse_ai_extract_response(resp_body) -> tuple[object, ExtractionEvidence]:
    if not isinstance(resp_body, dict):
        raise AiExtractResponseError("AI Extract returned an invalid response.")
    error_message = resp_body.get("error_message")
    if isinstance(error_message, str) and error_message.strip():
        raise AiExtractResponseError(error_message.strip())
    if "response" not in resp_body:
        raise AiExtractResponseError("AI Extract returned no extraction response.")
    response = resp_body["response"]
    if not isinstance(response, dict):
        raise AiExtractResponseError("AI Extract returned an invalid extraction response.")

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
