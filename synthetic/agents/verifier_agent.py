"""Mandatory verifier agent for generated examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from synthetic.infrastructure.llm_client import VLLMClient
from synthetic.infrastructure.prompt_loader import PromptTemplate, load_prompt
from synthetic.infrastructure.retry import RetryConfig
from synthetic.infrastructure.schema_io import parse_json_object, validate_json

from .reasoning import ReasoningPolicy, prepare_agent_response
from .schemas import (
    VERIFIER_OUTPUT_SCHEMA,
    VERIFIER_PROMPT_VERSION,
    response_format_json_schema,
)
from .state import (
    SyntheticGenerationState,
    compact_json,
    format_audio_context,
    prompt_audio_source_labels,
    prompt_target_condition,
    validation_feedback,
)


class VerifierAgent:
    """Check whether generated claim/QA records obey caption-grounded rules."""

    def __init__(
        self,
        llm_client: VLLMClient,
        prompt_path: Path,
        retry_config: Optional[RetryConfig] = None,
        prompt_version: str = VERIFIER_PROMPT_VERSION,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_policy: Optional[ReasoningPolicy] = None,
    ):
        self.llm_client = llm_client
        self.prompt_template: PromptTemplate = load_prompt(prompt_path)
        self.retry_config = retry_config
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_policy = reasoning_policy or ReasoningPolicy()

    async def verify(self, state: SyntheticGenerationState) -> Dict[str, Any]:
        if state.target_condition is None:
            raise ValueError("state.target_condition must be set before VerifierAgent runs.")
        if state.claim_record is None:
            raise ValueError("state.claim_record must be set before VerifierAgent runs.")
        if state.qa_record is None:
            raise ValueError("state.qa_record must be set before VerifierAgent runs.")

        source_labels = prompt_audio_source_labels(
            state.unit_record,
            state.target_condition,
        )
        prompt = self.prompt_template.render(
            {
                "audio_context": format_audio_context(
                    state.unit_record,
                    source_labels=source_labels,
                    caption_offset=state.unit_index,
                ),
                "target_condition_json": compact_json(
                    prompt_target_condition(state.target_condition)
                ),
                "claim_record_json": compact_json(state.claim_record),
                "qa_record_json": compact_json(state.qa_record),
                "validation_feedback": validation_feedback(
                    state.validation_errors,
                    "VerifierAgent",
                ),
                "reasoning_instruction": self.reasoning_policy.prompt_text(),
            }
        )
        raw_text = await self.llm_client.chat_text(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            response_format=response_format_json_schema(
                "verifier_agent_output",
                VERIFIER_OUTPUT_SCHEMA,
            ),
            retry_config=self.retry_config,
        )
        clean_text, stripped_reasoning = prepare_agent_response(
            raw_text,
            "VerifierAgent",
            self.reasoning_policy,
        )
        if stripped_reasoning:
            state.visible_reasoning_stripped.append("verifier_agent")
        state.raw_verifier_text = clean_text
        payload = parse_json_object(clean_text)
        validate_json(payload, VERIFIER_OUTPUT_SCHEMA)
        return payload
