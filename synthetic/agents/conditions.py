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
    claim_type: str
    claim_status: str
    evidence_sources: List[str]
    counterfactual_edit_type: str
    instruction: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "claim_type": self.claim_type,
            "claim_status": self.claim_status,
            "evidence_sources": list(self.evidence_sources),
            "counterfactual_edit_type": self.counterfactual_edit_type,
            "instruction": self.instruction,
        }


def build_target_conditions(audio_count: int) -> List[TargetCondition]:
    labels = audio_source_labels(audio_count)
    conditions: List[TargetCondition] = []

    for label in labels:
        conditions.append(
            TargetCondition(
                condition_name=f"faithful_supported_by_{label}",
                claim_type="faithful",
                claim_status="SUPPORTED",
                evidence_sources=[label],
                counterfactual_edit_type="none",
                instruction=(
                    f"Create one short faithful claim supported by one or more "
                    f"explicit captions from {label}. Describe one coherent event; "
                    "one or several related propositions are allowed."
                ),
            )
        )
        conditions.append(
            TargetCondition(
                condition_name=f"counterfactual_explicit_contradiction_{label}",
                claim_type="counterfactual",
                claim_status="CONTRADICTED",
                evidence_sources=[label],
                counterfactual_edit_type="explicit_fact_modification",
                instruction=(
                    f"Create one short claim about one coherent event in {label} that "
                    "changes one or more related caption-established propositions into "
                    "objectively incompatible alternatives. Avoid subjective contrasts. "
                    f"The positive caption evidence from {label} must prove the "
                    "contradiction for every change; do not rely on omission or another "
                    "audio source."
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
            return conditions[condition_index % len(conditions)]
        if self.strategy == "random":
            return self._rng.choice(conditions)
        return conditions[unit_index % len(conditions)]
