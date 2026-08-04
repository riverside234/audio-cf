"""Controlled reasoning policy for synthetic-data agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from synthetic.infrastructure.schema_io import SchemaValidationError, strip_visible_reasoning


DEFAULT_PRIVATE_REASONING_INSTRUCTION = """Use Gemma's reasoning ability before writing the JSON object.
Privately check: which captions explicitly support the claim, whether the target condition is satisfied, whether a counterfactual contradiction is based on explicit caption evidence, and whether any forbidden inference was introduced.
Keep this reasoning private. Do not output chain-of-thought, analysis text, markdown, or <think> tags. Return only the final JSON object that matches the schema."""

DEFAULT_VISIBLE_THINKING_INSTRUCTION = """Use Gemma's default thinking behavior if it is enabled by the served checkpoint or chat template.
You may place temporary reasoning inside a <think>...</think> block. After the closing </think> tag, return exactly one final JSON object that matches the schema.
Do not put the final JSON inside the thinking block. Do not output markdown fences or extra explanation outside the final JSON object. The pipeline will strip thinking blocks before parsing and will not store them in the final dataset."""

DEFAULT_GEMMA4_VLLM_INSTRUCTION = """Use Gemma 4 reasoning as configured by vLLM.
The runtime may place temporary reasoning in Gemma 4's thought channel and may suppress it from the API response. The final message content must be exactly one JSON object that matches the schema.
Do not put final JSON inside a reasoning/thought block. Do not output markdown fences or extra explanation around the final JSON. The pipeline suppresses or strips reasoning before parsing and never stores reasoning in the final dataset."""


@dataclass(frozen=True)
class ReasoningPolicy:
    """Prompt and sanitation policy for controlled reasoning."""

    enabled: bool = True
    mode: str = "private_json"
    instruction: str = ""
    effort: str = "medium"
    strip_visible_reasoning: bool = True
    reject_visible_reasoning: bool = False

    def prompt_text(self) -> str:
        if not self.enabled:
            return "Return the final JSON object directly. Do not output reasoning text."
        if self.instruction.strip():
            return self.instruction.strip()
        if self.mode == "gemma4_vllm":
            return DEFAULT_GEMMA4_VLLM_INSTRUCTION
        if self.mode == "visible_thinking_strip":
            return DEFAULT_VISIBLE_THINKING_INSTRUCTION
        return DEFAULT_PRIVATE_REASONING_INSTRUCTION


def prepare_agent_response(
    raw_text: str,
    agent_name: str,
    reasoning_policy: ReasoningPolicy,
) -> Tuple[str, bool]:
    """Return JSON-facing text and whether visible reasoning was removed."""

    if not (
        reasoning_policy.strip_visible_reasoning
        or reasoning_policy.reject_visible_reasoning
    ):
        return raw_text, False

    cleaned_text, removed = strip_visible_reasoning(raw_text)
    if removed and reasoning_policy.reject_visible_reasoning:
        raise SchemaValidationError(
            f"{agent_name} emitted visible reasoning despite reasoning policy."
        )
    return cleaned_text, removed
