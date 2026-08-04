"""Shared state and formatting helpers for the synthetic generation graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


JsonObject = Dict[str, Any]
AudioUnitRecord = Dict[str, Any]


@dataclass
class SyntheticGraphState:
    """Mutable state passed through the generation graph."""

    unit_record: AudioUnitRecord
    unit_index: int = 0
    target_condition: Optional[JsonObject] = None
    claim_record: Optional[JsonObject] = None
    qa_record: Optional[JsonObject] = None
    verifier_record: Optional[JsonObject] = None
    final_example: Optional[JsonObject] = None
    raw_claim_text: Optional[str] = None
    raw_qa_text: Optional[str] = None
    raw_verifier_text: Optional[str] = None
    visible_reasoning_stripped: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> JsonObject:
        return {
            "unit_record": self.unit_record,
            "unit_index": self.unit_index,
            "target_condition": self.target_condition,
            "claim_record": self.claim_record,
            "qa_record": self.qa_record,
            "verifier_record": self.verifier_record,
            "final_example": self.final_example,
            "raw_claim_text": self.raw_claim_text,
            "raw_qa_text": self.raw_qa_text,
            "raw_verifier_text": self.raw_verifier_text,
            "visible_reasoning_stripped": list(self.visible_reasoning_stripped),
            "validation_errors": list(self.validation_errors),
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticGraphState":
        return cls(
            unit_record=dict(payload["unit_record"]),
            unit_index=int(payload.get("unit_index", 0)),
            target_condition=_optional_dict(payload.get("target_condition")),
            claim_record=_optional_dict(payload.get("claim_record")),
            qa_record=_optional_dict(payload.get("qa_record")),
            verifier_record=_optional_dict(payload.get("verifier_record")),
            final_example=_optional_dict(payload.get("final_example")),
            raw_claim_text=_optional_str(payload.get("raw_claim_text")),
            raw_qa_text=_optional_str(payload.get("raw_qa_text")),
            raw_verifier_text=_optional_str(payload.get("raw_verifier_text")),
            visible_reasoning_stripped=[
                str(item) for item in payload.get("visible_reasoning_stripped", [])
            ],
            validation_errors=[str(item) for item in payload.get("validation_errors", [])],
            retry_count=int(payload.get("retry_count", 0)),
        )


def _optional_dict(value: Any) -> Optional[JsonObject]:
    if value is None:
        return None
    return dict(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def audio_source_labels(audio_count: int) -> List[str]:
    if audio_count < 1:
        raise ValueError("audio_count must be at least 1.")
    return [f"AUDIO_{index}" for index in range(1, audio_count + 1)]


def normalize_caption_groups(value: Any) -> List[List[str]]:
    if not isinstance(value, list):
        return []

    caption_groups: List[List[str]] = []
    for group in value:
        if group is None:
            caption_groups.append([])
        elif isinstance(group, list):
            caption_groups.append([str(item).strip() for item in group if str(item).strip()])
        else:
            text = str(group).strip()
            caption_groups.append([text] if text else [])
    return caption_groups


def list_value(value: Any, expected_len: int) -> List[str]:
    if isinstance(value, list):
        items = ["" if item is None else str(item) for item in value]
    elif value is None:
        items = []
    else:
        items = [str(value)]

    if len(items) < expected_len:
        items.extend([""] * (expected_len - len(items)))
    return items[:expected_len]


def format_audio_context(unit_record: Mapping[str, Any]) -> str:
    """Render AUDIO_1 ... AUDIO_N captions for prompt input."""

    audio_count = int(unit_record.get("audio_count", 0))
    labels = audio_source_labels(audio_count)
    audio_ids = list_value(unit_record.get("audio_ids"), audio_count)
    audio_file_names = list_value(unit_record.get("audio_file_names"), audio_count)
    caption_groups = normalize_caption_groups(unit_record.get("audio_captions"))

    if len(caption_groups) < audio_count:
        caption_groups.extend([[] for _ in range(audio_count - len(caption_groups))])

    blocks: List[str] = []
    for index, label in enumerate(labels):
        lines = [label]
        if audio_ids[index]:
            lines.append(f"audio_id: {audio_ids[index]}")
        if audio_file_names[index]:
            lines.append(f"audio_file_name: {audio_file_names[index]}")
        captions = caption_groups[index]
        if captions:
            for caption_index, caption in enumerate(captions, start=1):
                lines.append(f"caption_{caption_index}: {caption}")
        else:
            lines.append("captions: none")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def validation_feedback(errors: Sequence[str]) -> str:
    if not errors:
        return "None."
    return "\n".join(f"- {error}" for error in errors[-5:])
