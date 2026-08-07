"""Shared state and formatting helpers for synthetic generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


JsonObject = Dict[str, Any]
AudioUnitRecord = Dict[str, Any]
DEFAULT_PROMPT_CAPTIONS_PER_AUDIO = 3


@dataclass
class SyntheticGenerationState:
    """Mutable state passed through the generation sequence."""

    unit_record: AudioUnitRecord
    unit_index: int = 0
    target_condition: Optional[JsonObject] = None
    claim_record: Optional[JsonObject] = None
    qa_record: Optional[JsonObject] = None
    verifier_record: Optional[JsonObject] = None
    final_example: Optional[JsonObject] = None
    example_id: Optional[str] = None
    raw_claim_text: Optional[str] = None
    raw_qa_text: Optional[str] = None
    raw_verifier_text: Optional[str] = None
    visible_reasoning_stripped: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    retry_count: int = 0

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


def prompt_audio_source_labels(
    unit_record: Mapping[str, Any],
    target_condition: Mapping[str, Any],
) -> List[str]:
    """Return ordered sources explicitly used or referenced by the target."""

    labels = audio_source_labels(int(unit_record.get("audio_count", 0)))
    requested = {
        str(item)
        for item in target_condition.get("evidence_sources", [])
        if str(item) in labels
    }
    target_text = " ".join(
        str(target_condition.get(key, ""))
        for key in ("condition_name", "instruction")
    )
    requested.update(re.findall(r"AUDIO_[1-9][0-9]*", target_text))
    selected = [label for label in labels if label in requested]
    return selected or labels


def prompt_target_condition(target_condition: Mapping[str, Any]) -> JsonObject:
    """Drop the redundant condition name from prompt-facing target data."""

    keys = (
        "claim_type",
        "claim_status",
        "evidence_sources",
        "counterfactual_edit_type",
        "instruction",
    )
    return {key: target_condition[key] for key in keys if key in target_condition}


def compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def select_prompt_captions(
    captions: Sequence[str],
    limit: int = DEFAULT_PROMPT_CAPTIONS_PER_AUDIO,
    offset: int = 0,
) -> List[str]:
    """Select a bounded rotating window so repeated units use all captions."""

    cleaned = [str(caption).strip() for caption in captions if str(caption).strip()]
    if limit < 1 or len(cleaned) <= limit:
        return cleaned
    start = offset % len(cleaned)
    return [cleaned[(start + index) % len(cleaned)] for index in range(limit)]


def format_audio_context(
    unit_record: Mapping[str, Any],
    source_labels: Optional[Sequence[str]] = None,
    caption_limit: int = DEFAULT_PROMPT_CAPTIONS_PER_AUDIO,
    caption_offset: int = 0,
) -> str:
    """Render only prompt-relevant captions; omit IDs, paths, and filenames."""

    audio_count = int(unit_record.get("audio_count", 0))
    labels = audio_source_labels(audio_count)
    caption_groups = normalize_caption_groups(unit_record.get("audio_captions"))

    if len(caption_groups) < audio_count:
        caption_groups.extend([[] for _ in range(audio_count - len(caption_groups))])

    blocks: List[str] = []
    selected_labels = set(source_labels or labels)
    for index, label in enumerate(labels):
        if label not in selected_labels:
            continue
        lines = [f"{label} captions:"]
        captions = select_prompt_captions(
            caption_groups[index],
            limit=caption_limit,
            offset=caption_offset,
        )
        if captions:
            lines.extend(f"- {caption}" for caption in captions)
        else:
            lines.append("- none")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def validation_feedback(
    errors: Sequence[str],
    agent_name: Optional[str] = None,
    max_items: int = 2,
    max_chars_per_item: int = 360,
) -> str:
    relevant = list(errors)
    if agent_name:
        prefix = f"{agent_name} attempt "
        relevant = [error for error in relevant if str(error).startswith(prefix)]
    if not relevant:
        return "None."

    compact_errors: List[str] = []
    for error in relevant[-max_items:]:
        text = " ".join(str(error).split())
        if len(text) > max_chars_per_item:
            text = f"{text[: max_chars_per_item - 3].rstrip()}..."
        compact_errors.append(f"- {text}")
    return "\n".join(compact_errors)
