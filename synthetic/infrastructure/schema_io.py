"""Schema and JSON parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping


JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
VISIBLE_REASONING_RES = [
    re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel>thought\s*.*?<channel\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>thought\s*.*?<\|channel\|>", re.DOTALL | re.IGNORECASE),
]
LEADING_GEMMA4_THINK_TOKEN_RE = re.compile(r"^\s*<\|think\|>\s*", re.IGNORECASE)
LEADING_QWEN_REASONING_WITH_CLOSER_RE = re.compile(
    r"^\s*(?:(?!<think(?:ing)?>).)*?</think(?:ing)?>\s*"
    r"(?=(?:```(?:json)?\s*)?\{)",
    re.DOTALL | re.IGNORECASE,
)


class SchemaValidationError(ValueError):
    """Raised when generated JSON does not match the expected schema."""


def load_json_schema(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    match = JSON_FENCE_RE.match(stripped)
    if match:
        stripped = match.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Generated text is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SchemaValidationError("Generated JSON must be an object.")
    return parsed


def strip_visible_reasoning(text: str) -> tuple[str, bool]:
    """Remove visible thinking blocks while leaving the final JSON text intact."""

    cleaned = text
    removal_count = 0
    for pattern in VISIBLE_REASONING_RES:
        cleaned, removed = pattern.subn("", cleaned)
        removal_count += removed

    cleaned, removed_trigger = LEADING_GEMMA4_THINK_TOKEN_RE.subn("", cleaned)
    removal_count += removed_trigger
    # Qwen's chat template may prefill the opening <think> token, leaving only
    # the generated reasoning text and closing </think> in completion content.
    cleaned, removed_qwen_prefix = LEADING_QWEN_REASONING_WITH_CLOSER_RE.subn(
        "", cleaned
    )
    removal_count += removed_qwen_prefix
    return cleaned.strip(), removal_count > 0


def validate_json(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate JSON with jsonschema when available, fallback to basic checks."""

    try:
        import jsonschema  # type: ignore
    except ImportError:
        _basic_validate(payload, schema)
        return

    try:
        jsonschema.validate(instance=dict(payload), schema=dict(schema))
    except Exception as exc:  # jsonschema has several validation exception classes.
        raise SchemaValidationError(str(exc)) from exc


def _basic_validate(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [field for field in required if field not in payload]
        if missing:
            raise SchemaValidationError(f"Missing required fields: {missing}")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return

    for field, field_schema in properties.items():
        if field not in payload or not isinstance(field_schema, dict):
            continue
        expected_type = field_schema.get("type")
        if expected_type and not _matches_json_type(payload[field], expected_type):
            raise SchemaValidationError(
                f"Field {field!r} expected JSON type {expected_type!r}, "
                f"got {type(payload[field]).__name__}."
            )


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True
