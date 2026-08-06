"""Validation and final-row assembly for generated synthetic examples."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from synthetic.infrastructure.schema_io import SchemaValidationError, validate_json

from .conditions import TargetCondition
from .schemas import (
    CLAIM_OUTPUT_SCHEMA,
    EXAMPLE_SCHEMA_VERSION,
    QA_OUTPUT_SCHEMA,
    VERIFIER_OUTPUT_SCHEMA,
)
from .state import SyntheticGenerationState, audio_source_labels


class AgentValidationError(ValueError):
    """Raised when an agent output is valid JSON but inconsistent with the task."""


def validate_audio_unit_record(unit_record: Mapping[str, Any]) -> None:
    audio_count = int(unit_record.get("audio_count", 0))
    if audio_count < 1:
        raise AgentValidationError("audio_count must be at least 1.")

    for field_name in ["audio_ids", "audio_file_names", "audio_captions"]:
        value = unit_record.get(field_name)
        if not isinstance(value, list):
            raise AgentValidationError(f"{field_name} must be a list.")
        if len(value) != audio_count:
            raise AgentValidationError(
                f"{field_name} length {len(value)} does not match audio_count={audio_count}."
            )

    local_paths = unit_record.get("local_audio_paths")
    if local_paths is not None and len(local_paths) != audio_count:
        raise AgentValidationError(
            "local_audio_paths length does not match audio_count."
        )


def validate_claim_record(
    claim_record: Mapping[str, Any],
    target_condition: TargetCondition,
    audio_count: int,
) -> None:
    validate_json(claim_record, CLAIM_OUTPUT_SCHEMA)
    _require_equal("claim_type", claim_record, target_condition.claim_type)
    _require_equal("claim_status", claim_record, target_condition.claim_status)
    _require_equal(
        "counterfactual_edit_type",
        claim_record,
        target_condition.counterfactual_edit_type,
    )

    evidence_sources = _string_list(claim_record.get("evidence_sources"))
    _validate_evidence_sources(evidence_sources, audio_count)
    if sorted(evidence_sources) != sorted(target_condition.evidence_sources):
        raise AgentValidationError(
            "evidence_sources do not match target condition: "
            f"{evidence_sources} != {target_condition.evidence_sources}"
        )

    claim_text = str(claim_record.get("claim_text", "")).strip()
    if len(claim_text.split()) < 3:
        raise AgentValidationError("claim_text is too short.")

    supporting_phrases = _string_list(claim_record.get("supporting_caption_phrases"))
    if claim_record.get("claim_status") == "SUPPORTED" and not supporting_phrases:
        raise AgentValidationError("SUPPORTED claims need supporting_caption_phrases.")

    if claim_record.get("claim_type") == "faithful":
        if claim_record.get("counterfactual_edit_type") != "none":
            raise AgentValidationError("Faithful claims must use edit type 'none'.")
        return

    contradiction_basis = str(claim_record.get("contradiction_basis", "")).strip()
    if not contradiction_basis:
        raise AgentValidationError("Counterfactual claims need contradiction_basis.")
    if _uses_caption_absence_as_negative_evidence(contradiction_basis):
        raise AgentValidationError(
            "Counterfactual contradiction_basis relies on caption absence."
        )


def validate_qa_record(
    qa_record: Mapping[str, Any],
    claim_record: Mapping[str, Any],
    audio_count: int,
) -> None:
    validate_json(qa_record, QA_OUTPUT_SCHEMA)
    required_sources = _string_list(qa_record.get("required_evidence_sources"))
    _validate_evidence_sources(required_sources, audio_count)

    claim_sources = _string_list(claim_record.get("evidence_sources"))
    if sorted(required_sources) != sorted(claim_sources):
        raise AgentValidationError(
            "QA required_evidence_sources do not match claim evidence_sources."
        )

    answer = str(qa_record.get("answer", "")).strip().lower()
    claim_status = str(claim_record.get("claim_status", "")).strip().lower()
    expected_word = "supported" if claim_status == "supported" else "contradicted"
    if expected_word and expected_word not in answer:
        raise AgentValidationError(
            f"answer should explicitly say the claim is {expected_word}."
        )


def validate_verifier_record(verifier_record: Mapping[str, Any], audio_count: int) -> None:
    validate_json(verifier_record, VERIFIER_OUTPUT_SCHEMA)
    corrected_sources = _string_list(verifier_record.get("corrected_evidence_sources"))
    _validate_evidence_sources(corrected_sources, audio_count)


def build_final_example(
    state: SyntheticGenerationState,
    generation_model: str,
) -> Dict[str, Any]:
    if state.target_condition is None:
        raise AgentValidationError("target_condition is missing.")
    if state.claim_record is None:
        raise AgentValidationError("claim_record is missing.")
    if state.qa_record is None:
        raise AgentValidationError("qa_record is missing.")

    unit = state.unit_record
    claim = state.claim_record
    qa = state.qa_record
    unit_id = str(unit.get("unit_id", ""))
    state.example_id = _example_id(unit_id, claim["claim_text"], qa["question"])

    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "audio_count": int(unit["audio_count"]),
        "audio_ids": list(unit.get("audio_ids", [])),
        "local_audio_paths": list(unit.get("local_audio_paths", [])),
        "audio_file_names": list(unit.get("audio_file_names", [])),
        "audio_captions": list(unit.get("audio_captions", [])),
        "claim_text": claim["claim_text"],
        "claim_type": claim["claim_type"],
        "claim_status": claim["claim_status"],
        "evidence_sources": list(claim["evidence_sources"]),
        "counterfactual_edit_type": claim["counterfactual_edit_type"],
        "question": qa["question"],
        "answer": qa["answer"],
        "generation_model": generation_model,
    }


def _require_equal(field_name: str, record: Mapping[str, Any], expected: str) -> None:
    actual = record.get(field_name)
    if actual != expected:
        raise AgentValidationError(
            f"{field_name} must be {expected!r}, got {actual!r}."
        )


def _validate_evidence_sources(sources: Sequence[str], audio_count: int) -> None:
    valid_sources = set(audio_source_labels(audio_count))
    unknown = sorted(set(sources) - valid_sources)
    if unknown:
        raise AgentValidationError(f"Unknown evidence sources: {unknown}.")


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        raise SchemaValidationError("Expected a list of strings.")
    return [str(item) for item in value]


def _uses_caption_absence_as_negative_evidence(text: str) -> bool:
    lowered = text.lower()
    red_flags = [
        "not mentioned",
        "does not mention",
        "no caption",
        "absent from the caption",
        "absence from the caption",
        "not in the captions",
    ]
    return any(flag in lowered for flag in red_flags)


def _example_id(unit_id: str, claim_text: str, question: str) -> str:
    key = f"{unit_id}\n{claim_text}\n{question}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"synthetic_{digest}"
