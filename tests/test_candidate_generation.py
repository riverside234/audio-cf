from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, List, Mapping, Optional

from data_synthetic import build_stats, load_yaml, run_batch
from synthetic.agents.candidate_selection import select_distinct_verified_states
from synthetic.agents.state import SyntheticGenerationState


class CandidateSelectionTests(unittest.TestCase):
    def test_exact_and_near_duplicates_keep_only_first_verified_candidate(self) -> None:
        states = [
            candidate_state(0, 0, "A dog barks loudly near a gate."),
            candidate_state(0, 1, "A dog barks loudly near the gate."),
            candidate_state(0, 2, "A dog barks loudly near a gate."),
        ]

        accepted, duplicates_removed = select_distinct_verified_states(
            states,
            similarity_threshold=0.85,
        )

        self.assertEqual([state.candidate_index for state in accepted], [0])
        self.assertEqual(duplicates_removed, 2)

    def test_label_contrasts_and_different_units_are_preserved(self) -> None:
        supported = candidate_state(0, 0, "Cars pass quietly.")
        contradicted = candidate_state(
            0,
            1,
            "Cars pass loudly.",
            status="CONTRADICTED",
        )
        other_unit = candidate_state(1, 0, "Cars pass quietly.")

        accepted, duplicates_removed = select_distinct_verified_states(
            [supported, contradicted, other_unit],
            similarity_threshold=0.80,
        )

        self.assertEqual(len(accepted), 3)
        self.assertEqual(duplicates_removed, 0)

    def test_default_config_requests_two_candidates(self) -> None:
        config = load_yaml(ROOT_CONFIG)

        self.assertEqual(config["generation"]["examples_per_unit"], 2)
        self.assertEqual(config["generation"]["similarity_threshold"], 0.90)

    def test_stats_account_for_all_candidate_outcomes(self) -> None:
        config = load_yaml(ROOT_CONFIG)

        stats = build_stats(
            config=config,
            vllm_config={"client": {"model": "fake"}},
            input_rows_loaded=3,
            selected_units=2,
            examples_written=2,
            errors_written=1,
            dry_run=False,
            start_index=0,
            max_units=2,
            status="completed",
            completed_batches=1,
            near_duplicates_removed=1,
        )

        self.assertEqual(stats["candidate_generations_planned"], 4)
        self.assertEqual(stats["candidate_generations_completed"], 4)
        self.assertEqual(stats["near_duplicates_removed"], 1)


class CandidateBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_batch_generates_every_candidate_with_stable_indices(self) -> None:
        runner = RecordingRunner()

        states, errors, duplicates_removed = await run_batch(
            runner=runner,
            rows=[{"unit_id": "unit-5"}, {"unit_id": "unit-6"}],
            start_index=5,
            fail_fast=False,
            continue_on_error=True,
            examples_per_unit=2,
            similarity_threshold=0.90,
        )

        self.assertEqual(len(states), 4)
        self.assertEqual(errors, [])
        self.assertEqual(duplicates_removed, 0)
        self.assertEqual(
            sorted(runner.calls),
            [(5, 0, 10), (5, 1, 11), (6, 0, 12), (6, 1, 13)],
        )

    async def test_candidate_failure_does_not_discard_sibling(self) -> None:
        runner = RecordingRunner(failing_candidate=0)

        states, errors, duplicates_removed = await run_batch(
            runner=runner,
            rows=[{"unit_id": "unit-2"}],
            start_index=2,
            fail_fast=False,
            continue_on_error=True,
            examples_per_unit=2,
            similarity_threshold=0.90,
        )

        self.assertEqual([state.candidate_index for state in states], [1])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["candidate_index"], 0)
        self.assertEqual(errors[0]["unit_index"], 2)
        self.assertEqual(duplicates_removed, 0)


class RecordingRunner:
    max_concurrency = 4

    def __init__(self, failing_candidate: Optional[int] = None) -> None:
        self.failing_candidate = failing_candidate
        self.calls: List[tuple[int, int, int]] = []

    async def run_unit(
        self,
        unit_record: Mapping[str, Any],
        unit_index: int,
        candidate_index: int,
        condition_index: int,
    ) -> SyntheticGenerationState:
        self.calls.append((unit_index, candidate_index, condition_index))
        if candidate_index == self.failing_candidate:
            raise ValueError("candidate failed verification")
        status = "SUPPORTED" if candidate_index % 2 == 0 else "CONTRADICTED"
        return candidate_state(
            unit_index,
            candidate_index,
            f"Distinct claim {unit_index} candidate {candidate_index}.",
            status=status,
            unit_id=str(unit_record.get("unit_id", "")),
            condition_index=condition_index,
        )


def candidate_state(
    unit_index: int,
    candidate_index: int,
    claim_text: str,
    *,
    status: str = "SUPPORTED",
    unit_id: Optional[str] = None,
    condition_index: Optional[int] = None,
) -> SyntheticGenerationState:
    source = "AUDIO_1"
    return SyntheticGenerationState(
        unit_record={"unit_id": unit_id or f"unit-{unit_index}"},
        unit_index=unit_index,
        candidate_index=candidate_index,
        condition_index=condition_index,
        final_example={
            "claim_text": claim_text,
            "claim_status": status,
            "evidence_sources": [source],
            "question": f"Which audio determines whether {claim_text} is supported?",
        },
    )


ROOT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "data_synthetic.yaml"
)


if __name__ == "__main__":
    unittest.main()
