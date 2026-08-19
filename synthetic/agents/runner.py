"""Deterministic orchestration for synthetic example generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .claim_agent import ClaimAgent
from .conditions import TargetCondition, TargetConditionSampler
from .qa_agent import QAAgent
from .state import SyntheticGenerationState
from .validators import (
    build_final_example,
    validate_audio_unit_record,
    validate_claim_record,
    validate_qa_record,
    validate_verifier_record,
)
from .verifier_agent import VerifierAgent


@dataclass
class SyntheticGenerationRunner:
    """Run the fixed claim, QA, verifier, and finalization sequence."""

    claim_agent: ClaimAgent
    qa_agent: QAAgent
    verifier_agent: VerifierAgent
    condition_sampler: Optional[TargetConditionSampler] = None
    max_validation_attempts: int = 2
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if self.condition_sampler is None:
            self.condition_sampler = TargetConditionSampler()
        if self.max_validation_attempts < 1:
            raise ValueError("max_validation_attempts must be at least 1.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")

    async def run_unit(
        self,
        unit_record: Mapping[str, Any],
        unit_index: int = 0,
    ) -> SyntheticGenerationState:
        state = SyntheticGenerationState(
            unit_record=dict(unit_record),
            unit_index=unit_index,
        )
        validate_audio_unit_record(state.unit_record)
        state.target_condition = self._choose_condition(state).to_dict()
        await self._run_claim_with_retries(state)
        await self._run_qa_with_retries(state)
        await self._run_verifier(state)

        state.final_example = build_final_example(state=state)
        return state

    def _choose_condition(self, state: SyntheticGenerationState) -> TargetCondition:
        assert self.condition_sampler is not None
        return self.condition_sampler.choose(state.unit_record, state.unit_index)

    async def _run_claim_with_retries(self, state: SyntheticGenerationState) -> None:
        assert state.target_condition is not None
        target = TargetCondition(**state.target_condition)

        for attempt in range(1, self.max_validation_attempts + 1):
            state.retry_count = attempt - 1
            try:
                state.claim_record = await self.claim_agent.generate(state)
                validate_claim_record(state.claim_record, target, state.unit_record)
                return
            except Exception as exc:
                state.validation_errors.append(f"ClaimAgent attempt {attempt}: {exc}")
                if attempt >= self.max_validation_attempts:
                    raise

    async def _run_qa_with_retries(self, state: SyntheticGenerationState) -> None:
        audio_count = int(state.unit_record["audio_count"])
        for attempt in range(1, self.max_validation_attempts + 1):
            state.retry_count = attempt - 1
            try:
                state.qa_record = await self.qa_agent.generate(state)
                assert state.claim_record is not None
                validate_qa_record(state.qa_record, state.claim_record, audio_count)
                return
            except Exception as exc:
                state.validation_errors.append(f"QAAgent attempt {attempt}: {exc}")
                if attempt >= self.max_validation_attempts:
                    raise

    async def _run_verifier(self, state: SyntheticGenerationState) -> None:
        audio_count = int(state.unit_record["audio_count"])
        state.verifier_record = await self.verifier_agent.verify(state)
        validate_verifier_record(state.verifier_record, audio_count)


def build_runner(
    claim_agent: ClaimAgent,
    qa_agent: QAAgent,
    verifier_agent: VerifierAgent,
    condition_sampler: Optional[TargetConditionSampler] = None,
    max_validation_attempts: int = 2,
    max_concurrency: int = 8,
) -> SyntheticGenerationRunner:
    return SyntheticGenerationRunner(
        claim_agent=claim_agent,
        qa_agent=qa_agent,
        verifier_agent=verifier_agent,
        condition_sampler=condition_sampler,
        max_validation_attempts=max_validation_attempts,
        max_concurrency=max_concurrency,
    )
