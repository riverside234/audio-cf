"""Deterministic selection of distinct verified candidates per audio unit."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Sequence, Tuple

from .state import SyntheticGenerationState


def select_distinct_verified_states(
    states: Sequence[SyntheticGenerationState],
    similarity_threshold: float,
) -> Tuple[List[SyntheticGenerationState], int]:
    """Keep the first candidate from each within-unit near-duplicate group."""

    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be > 0 and <= 1.")

    accepted: List[SyntheticGenerationState] = []
    accepted_by_unit: Dict[Tuple[int, str], List[SyntheticGenerationState]] = {}
    duplicates_removed = 0

    for state in states:
        if state.final_example is None:
            raise ValueError("Candidate selection requires finalized verified states.")
        unit_key = (
            state.unit_index,
            str(state.unit_record.get("unit_id", "")),
        )
        previous = accepted_by_unit.setdefault(unit_key, [])
        if any(
            candidate_examples_are_similar(state, other, similarity_threshold)
            for other in previous
        ):
            duplicates_removed += 1
            continue
        previous.append(state)
        accepted.append(state)

    return accepted, duplicates_removed


def candidate_examples_are_similar(
    left: SyntheticGenerationState,
    right: SyntheticGenerationState,
    similarity_threshold: float,
) -> bool:
    """Compare finalized claim/QA pairs while preserving useful label contrasts."""

    left_example = _final_example(left)
    right_example = _final_example(right)
    left_claim = _normalize_text(left_example.get("claim_text", ""))
    right_claim = _normalize_text(right_example.get("claim_text", ""))
    left_question = _normalize_text(left_example.get("question", ""))
    right_question = _normalize_text(right_example.get("question", ""))

    if left_claim == right_claim and left_question == right_question:
        return True
    if _label_source_signature(left_example) != _label_source_signature(right_example):
        return False

    claim_similarity = _similarity(left_claim, right_claim)
    pair_similarity = _similarity(
        f"{left_claim} {left_question}",
        f"{right_claim} {right_question}",
    )
    return max(claim_similarity, pair_similarity) >= similarity_threshold


def _final_example(state: SyntheticGenerationState) -> Dict[str, Any]:
    if state.final_example is None:
        raise ValueError("Candidate comparison requires finalized examples.")
    return state.final_example


def _label_source_signature(example: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return (
        str(example.get("claim_status", "")),
        tuple(str(item) for item in example.get("evidence_sources", [])),
    )


def _normalize_text(value: Any) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in str(value)
        ).split()
    )


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()
