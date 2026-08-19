"""Target-condition sampling for balanced synthetic examples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from .state import audio_source_labels


@dataclass(frozen=True)
class TargetCondition:
    """A pre-sampled generation target that the LLM must satisfy."""

    condition_name: str
    claim_status: str
    evidence_sources: List[str]
    instruction: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "claim_status": self.claim_status,
            "evidence_sources": list(self.evidence_sources),
            "instruction": self.instruction,
        }


def build_target_conditions(audio_count: int) -> List[TargetCondition]:
    labels = audio_source_labels(audio_count)
    conditions: List[TargetCondition] = []

    for label in labels:
        conditions.append(
            TargetCondition(
                condition_name=f"supported_by_{label}",
                claim_status="SUPPORTED",
                evidence_sources=[label],
                instruction=(
                    f"Create one short claim explicitly supported by one or more "
                    f"captions from {label}. Describe one coherent event; one or "
                    "several related propositions are allowed."
                ),
            )
        )
        conditions.append(
            TargetCondition(
                condition_name=f"contradicted_by_{label}",
                claim_status="CONTRADICTED",
                evidence_sources=[label],
                instruction=(
                    f"Create one short claim about one coherent event in {label} that "
                    "changes at least one central caption-established fact into a "
                    "conflicting alternative about the same scene. Other details may "
                    "remain supported or neutral. Explicitly captioned subjective or "
                    "relative attributes are allowed. Use positive caption evidence "
                    f"from {label}; do not rely only on omission, an unrelated event, "
                    "or another audio source."
                ),
            )
        )
    return conditions


class TargetConditionSampler:
    """Deterministic target-condition sampler."""

    def __init__(self, seed: int = 42, strategy: str = "cycle"):
        if strategy not in {"cycle", "random"}:
            raise ValueError("strategy must be 'cycle' or 'random'.")
        self.seed = seed
        self.strategy = strategy
        self._rng = random.Random(seed)

    def choose(
        self,
        unit_record: Mapping[str, Any],
        unit_index: int = 0,
        condition_index: Optional[int] = None,
    ) -> TargetCondition:
        audio_count = int(unit_record.get("audio_count", 0))
        conditions = build_target_conditions(audio_count)
        if not conditions:
            raise ValueError(f"No target conditions available for audio_count={audio_count}.")

        if condition_index is not None:
            if self.strategy == "random":
                return random.Random(self.seed + condition_index).choice(conditions)
            return conditions[condition_index % len(conditions)]
        if self.strategy == "random":
            return self._rng.choice(conditions)
        return conditions[unit_index % len(conditions)]
