"""Graph orchestration for synthetic example generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .claim_agent import ClaimAgent
from .conditions import TargetCondition, TargetConditionSampler
from .qa_agent import QAAgent
from .state import SyntheticGraphState
from .validators import (
    build_final_example,
    validate_audio_unit_record,
    validate_claim_record,
    validate_qa_record,
    validate_verifier_record,
)
from .verifier_agent import VerifierAgent


@dataclass
class SyntheticGenerationGraph:
    """CPU-safe graph runner that can be wrapped by LangGraph later."""

    claim_agent: ClaimAgent
    qa_agent: QAAgent
    verifier_agent: Optional[VerifierAgent] = None
    condition_sampler: Optional[TargetConditionSampler] = None
    run_verifier: bool = False
    max_validation_attempts: int = 2
    max_concurrency: int = 8
    generation_model: str = ""
    prompt_version: str = "claim_agent_v0+qa_agent_v0"

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
    ) -> SyntheticGraphState:
        state = SyntheticGraphState(unit_record=dict(unit_record), unit_index=unit_index)
        validate_audio_unit_record(state.unit_record)
        state.target_condition = self._choose_condition(state).to_dict()
        await self._run_claim_with_retries(state)
        await self._run_qa_with_retries(state)

        if self.run_verifier and self.verifier_agent is not None:
            await self._run_verifier(state)

        state.final_example = build_final_example(
            state=state,
            generation_model=self.generation_model,
            prompt_version=self.prompt_version,
        )
        return state

    async def run_many(
        self,
        unit_records: Sequence[Mapping[str, Any]],
        start_index: int = 0,
    ) -> List[SyntheticGraphState]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_one(offset: int, row: Mapping[str, Any]) -> SyntheticGraphState:
            async with semaphore:
                return await self.run_unit(row, unit_index=start_index + offset)

        tasks = [run_one(index, row) for index, row in enumerate(unit_records)]
        return list(await asyncio.gather(*tasks))

    def _choose_condition(self, state: SyntheticGraphState) -> TargetCondition:
        assert self.condition_sampler is not None
        return self.condition_sampler.choose(state.unit_record, state.unit_index)

    async def _run_claim_with_retries(self, state: SyntheticGraphState) -> None:
        audio_count = int(state.unit_record["audio_count"])
        assert state.target_condition is not None
        target = TargetCondition(**state.target_condition)

        for attempt in range(1, self.max_validation_attempts + 1):
            state.retry_count = attempt - 1
            try:
                state.claim_record = await self.claim_agent.generate(state)
                validate_claim_record(state.claim_record, target, audio_count)
                return
            except Exception as exc:
                state.validation_errors.append(f"ClaimAgent attempt {attempt}: {exc}")
                if attempt >= self.max_validation_attempts:
                    raise

    async def _run_qa_with_retries(self, state: SyntheticGraphState) -> None:
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

    async def _run_verifier(self, state: SyntheticGraphState) -> None:
        assert self.verifier_agent is not None
        audio_count = int(state.unit_record["audio_count"])
        state.verifier_record = await self.verifier_agent.verify(state)
        validate_verifier_record(state.verifier_record, audio_count)


def build_graph(
    claim_agent: ClaimAgent,
    qa_agent: QAAgent,
    verifier_agent: Optional[VerifierAgent] = None,
    condition_sampler: Optional[TargetConditionSampler] = None,
    run_verifier: bool = False,
    max_validation_attempts: int = 2,
    max_concurrency: int = 8,
    generation_model: str = "",
    prompt_version: str = "claim_agent_v0+qa_agent_v0",
) -> SyntheticGenerationGraph:
    return SyntheticGenerationGraph(
        claim_agent=claim_agent,
        qa_agent=qa_agent,
        verifier_agent=verifier_agent,
        condition_sampler=condition_sampler,
        run_verifier=run_verifier,
        max_validation_attempts=max_validation_attempts,
        max_concurrency=max_concurrency,
        generation_model=generation_model,
        prompt_version=prompt_version,
    )


def build_langgraph_app(runner: SyntheticGenerationGraph) -> Any:
    """Build a LangGraph app when the optional langgraph package is installed."""

    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph to use build_langgraph_app, or use "
            "SyntheticGenerationGraph.run_unit/run_many directly."
        ) from exc

    async def choose_target_condition(payload: Dict[str, Any]) -> Dict[str, Any]:
        state = SyntheticGraphState.from_dict(payload)
        validate_audio_unit_record(state.unit_record)
        state.target_condition = runner._choose_condition(state).to_dict()
        return state.to_dict()

    async def generate_claim(payload: Dict[str, Any]) -> Dict[str, Any]:
        state = SyntheticGraphState.from_dict(payload)
        await runner._run_claim_with_retries(state)
        return state.to_dict()

    async def generate_qa(payload: Dict[str, Any]) -> Dict[str, Any]:
        state = SyntheticGraphState.from_dict(payload)
        await runner._run_qa_with_retries(state)
        return state.to_dict()

    async def verify(payload: Dict[str, Any]) -> Dict[str, Any]:
        state = SyntheticGraphState.from_dict(payload)
        if runner.run_verifier and runner.verifier_agent is not None:
            await runner._run_verifier(state)
        return state.to_dict()

    async def finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
        state = SyntheticGraphState.from_dict(payload)
        state.final_example = build_final_example(
            state=state,
            generation_model=runner.generation_model,
            prompt_version=runner.prompt_version,
        )
        return state.to_dict()

    workflow = StateGraph(dict)
    workflow.add_node("choose_target_condition", choose_target_condition)
    workflow.add_node("claim_agent", generate_claim)
    workflow.add_node("qa_agent", generate_qa)
    workflow.add_node("verifier_agent", verify)
    workflow.add_node("finalize", finalize)
    workflow.set_entry_point("choose_target_condition")
    workflow.add_edge("choose_target_condition", "claim_agent")
    workflow.add_edge("claim_agent", "qa_agent")
    workflow.add_edge("qa_agent", "verifier_agent")
    workflow.add_edge("verifier_agent", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()
