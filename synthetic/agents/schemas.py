"""Strict JSON schemas for synthetic generation agents."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping


EXAMPLE_SCHEMA_VERSION = "synthetic_example_v4"
CLAIM_PROMPT_VERSION = "claim_agent_v10"
QA_PROMPT_VERSION = "qa_agent_v7"
VERIFIER_PROMPT_VERSION = "verifier_agent_v8"
VLLM_UNSUPPORTED_SCHEMA_KEYS = frozenset({"uniqueItems"})

CLAIM_STATUSES = ["SUPPORTED", "CONTRADICTED"]

EVIDENCE_SOURCE_SCHEMA = {
    "type": "string",
    "pattern": "^AUDIO_[1-9][0-9]*$",
}

CLAIM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_text",
        "claim_status",
        "evidence_sources",
        "supporting_caption_phrases",
        "contradiction_basis",
        "forbidden_inferences",
        "confidence",
    ],
    "properties": {
        "claim_text": {"type": "string", "minLength": 1},
        "claim_status": {"type": "string", "enum": CLAIM_STATUSES},
        "evidence_sources": {
            "type": "array",
            "items": EVIDENCE_SOURCE_SCHEMA,
            "uniqueItems": True,
        },
        "supporting_caption_phrases": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contradiction_basis": {"type": "string"},
        "forbidden_inferences": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

QA_GENERATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "question",
        "claim_evaluation_explanation",
    ],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "claim_evaluation_explanation": {"type": "string", "minLength": 1},
    },
}

QA_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "question",
        "answer",
        "answer_source",
        "claim_evaluation_explanation",
        "required_evidence_sources",
    ],
    "properties": {
        "question": {"type": "string", "minLength": 1},
        "answer": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "string", "minLength": 1},
        },
        "answer_source": {
            "type": "array",
            "items": EVIDENCE_SOURCE_SCHEMA,
            "uniqueItems": True,
        },
        "claim_evaluation_explanation": {"type": "string", "minLength": 1},
        "required_evidence_sources": {
            "type": "array",
            "items": EVIDENCE_SOURCE_SCHEMA,
            "uniqueItems": True,
        },
    },
}

VERIFIER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verifier_status",
        "validation_errors",
        "validation_notes",
        "corrected_claim_status",
        "corrected_evidence_sources",
    ],
    "properties": {
        "verifier_status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "validation_errors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "validation_notes": {"type": "string"},
        "corrected_claim_status": {
            "type": ["string", "null"],
            "enum": ["SUPPORTED", "CONTRADICTED", None],
        },
        "corrected_evidence_sources": {
            "type": "array",
            "items": EVIDENCE_SOURCE_SCHEMA,
            "uniqueItems": True,
        },
    },
}


def schema_json(schema: Mapping[str, Any]) -> str:
    return json.dumps(dict(schema), ensure_ascii=False, indent=2, sort_keys=True)


def response_format_json_schema(
    name: str,
    schema: Mapping[str, Any],
    strict: bool = True,
) -> Dict[str, Any]:
    """Build an OpenAI/vLLM JSON-schema response_format for one agent call."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": vllm_compatible_schema(schema),
        },
    }


def vllm_compatible_schema(value: Any) -> Any:
    """Copy a JSON schema without keywords unsupported by vLLM grammars."""

    if isinstance(value, Mapping):
        return {
            key: vllm_compatible_schema(item)
            for key, item in value.items()
            if key not in VLLM_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [vllm_compatible_schema(item) for item in value]
    return value
