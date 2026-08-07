"""Question-answer generation agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from synthetic.infrastructure.llm_client import VLLMClient
from synthetic.infrastructure.prompt_loader import PromptTemplate, load_prompt
from synthetic.infrastructure.retry import RetryConfig
from synthetic.infrastructure.schema_io import parse_json_object, validate_json

from .reasoning import ReasoningPolicy, prepare_agent_response
from .schemas import (
    QA_OUTPUT_SCHEMA,
    QA_PROMPT_VERSION,
    response_format_json_schema,
)
from .state import SyntheticGenerationState, compact_json, validation_feedback


class QAAgent:
    """Generate a question and final answer from captions plus a claim."""

    def __init__(
        self,
        llm_client: VLLMClient,
        prompt_path: Path,
        retry_config: Optional[RetryConfig] = None,
        prompt_version: str = QA_PROMPT_VERSION,
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

    async def generate(self, state: SyntheticGenerationState) -> Dict[str, Any]:
        if state.target_condition is None:
            raise ValueError("state.target_condition must be set before QAAgent runs.")
        if state.claim_record is None:
            raise ValueError("state.claim_record must be set before QAAgent runs.")

        prompt = self.prompt_template.render(
            {
                "claim_record_json": compact_json(state.claim_record),
                "validation_feedback": validation_feedback(
                    state.validation_errors,
                    "QAAgent",
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
                "qa_agent_output",
                QA_OUTPUT_SCHEMA,
            ),
            retry_config=self.retry_config,
        )
        clean_text, stripped_reasoning = prepare_agent_response(
            raw_text,
            "QAAgent",
            self.reasoning_policy,
        )
        if stripped_reasoning:
            state.visible_reasoning_stripped.append("qa_agent")
        state.raw_qa_text = clean_text
        payload = parse_json_object(clean_text)
        validate_json(payload, QA_OUTPUT_SCHEMA)
        return payload
