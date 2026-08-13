from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from data_synthetic import build_audit_row
from synthetic.agents import (
    ClaimAgent,
    QAAgent,
    ReasoningPolicy,
    SyntheticGenerationRunner,
)
from synthetic.agents.conditions import build_target_conditions
from synthetic.agents.schemas import (
    CLAIM_OUTPUT_SCHEMA,
    EXAMPLE_SCHEMA_VERSION,
    QA_OUTPUT_SCHEMA,
    response_format_json_schema,
)
from synthetic.agents.state import (
    format_audio_context,
    prompt_audio_source_labels,
    validation_feedback,
)
from synthetic.agents.validators import validate_claim_record, validate_qa_record
from synthetic.infrastructure.schema_io import SchemaValidationError, validate_json


ROOT = Path(__file__).resolve().parents[1]


class FakeLLMClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def chat_text(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("FakeLLMClient has no response left.")
        return self.responses.pop(0)


class SyntheticGenerationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        prompt_dir = Path(self.temp_dir.name)
        self.claim_prompt = prompt_dir / "claim.md"
        self.qa_prompt = prompt_dir / "qa.md"
        self.claim_prompt.write_text(
            "{audio_context}\n{target_condition_json}\n"
            "{validation_feedback}\n{reasoning_instruction}\n",
            encoding="utf-8",
        )
        self.qa_prompt.write_text(
            "{claim_record_json}\n{validation_feedback}\n{reasoning_instruction}\n",
            encoding="utf-8",
        )

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_fake_client_generates_valid_example_and_strips_reasoning(self) -> None:
        claim_response = f"<think>private check</think>\n{json.dumps(valid_claim())}"
        fake_client = FakeLLMClient([claim_response, json.dumps(valid_qa())])
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=0)

        self.assertIsNotNone(state.final_example)
        assert state.final_example is not None
        self.assertEqual(state.final_example["schema_version"], EXAMPLE_SCHEMA_VERSION)
        self.assertEqual(state.final_example["claim_status"], "SUPPORTED")
        self.assertEqual(state.final_example["answer"], ["supported", "AUDIO_1"])
        provenance_fields = {
            "example_id",
            "unit_id",
            "prompt_version",
            "grounding_standard",
            "audio_captions",
            "generation_model",
        }
        self.assertTrue(provenance_fields.isdisjoint(state.final_example))
        self.assertTrue((state.example_id or "").startswith("synthetic_"))
        audit_row = build_audit_row(state)
        self.assertEqual(audit_row["example_id"], state.example_id)
        self.assertEqual(audit_row["unit_id"], "unit-1")
        self.assertEqual(state.visible_reasoning_stripped, ["claim_agent"])
        self.assertNotIn("private check", state.raw_claim_text or "")
        self.assertEqual(len(fake_client.calls), 2)
        for call in fake_client.calls:
            self.assertEqual(call["response_format"]["type"], "json_schema")

    async def test_semantic_validation_failure_retries_claim(self) -> None:
        invalid_claim = valid_claim()
        invalid_claim["claim_status"] = "CONTRADICTED"
        fake_client = FakeLLMClient(
            [
                json.dumps(invalid_claim),
                json.dumps(valid_claim()),
                json.dumps(valid_qa()),
            ]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=0)

        self.assertEqual(len(fake_client.calls), 3)
        self.assertEqual(len(state.validation_errors), 1)
        self.assertIn("claim_status must be 'SUPPORTED'", state.validation_errors[0])
        self.assertIn("ClaimAgent attempt 1", fake_client.calls[1]["messages"][0]["content"])
        self.assertNotIn(
            "ClaimAgent attempt 1",
            fake_client.calls[2]["messages"][0]["content"],
        )

    async def test_qa_answer_judgment_and_source_are_retried(self) -> None:
        invalid_qa = valid_qa()
        invalid_qa["answer"] = ["contradicted", "AUDIO_2"]
        fake_client = FakeLLMClient(
            [
                json.dumps(valid_claim()),
                json.dumps(invalid_qa),
                json.dumps(valid_qa()),
            ]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=0)

        self.assertEqual(len(fake_client.calls), 3)
        self.assertEqual(state.final_example["answer"], ["supported", "AUDIO_1"])
        self.assertIn(
            "answer must be ['supported', 'AUDIO_1']",
            state.validation_errors[0],
        )

    async def test_runner_generates_unsupported_none_example(self) -> None:
        fake_client = FakeLLMClient(
            [json.dumps(unsupported_claim()), json.dumps(unsupported_qa())]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=2)

        self.assertEqual(state.target_condition["claim_status"], "UNSUPPORTED")
        self.assertEqual(state.final_example["evidence_sources"], [])
        self.assertEqual(state.final_example["answer"], ["unsupported", "NONE"])

    def _runner(self, fake_client: FakeLLMClient) -> SyntheticGenerationRunner:
        policy = ReasoningPolicy(strip_visible_reasoning=True)
        claim_agent = ClaimAgent(
            llm_client=fake_client,  # type: ignore[arg-type]
            prompt_path=self.claim_prompt,
            reasoning_policy=policy,
        )
        qa_agent = QAAgent(
            llm_client=fake_client,  # type: ignore[arg-type]
            prompt_path=self.qa_prompt,
            reasoning_policy=policy,
        )
        return SyntheticGenerationRunner(
            claim_agent=claim_agent,
            qa_agent=qa_agent,
            max_validation_attempts=2,
        )


class PromptContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_production_prompts_use_only_stage_relevant_context(self) -> None:
        unit = audio_unit()
        unit["audio_file_names"] = ["private/one.wav", "private/two.wav"]
        unit["audio_captions"] = [
            [
                "Steady rain falls outside.",
                "Rain taps against a roof.",
                "Water patters during a shower.",
                "A long storm continues overnight.",
                "Distant rainfall can be heard.",
            ],
            [
                "A dog barks near a gate.",
                "Several vehicles pass on a road.",
                "A horn sounds in traffic.",
                "Footsteps cross the pavement.",
                "People talk beside the street.",
            ],
        ]
        fake_client = FakeLLMClient(
            [json.dumps(valid_claim()), json.dumps(valid_qa())]
        )
        policy = ReasoningPolicy(mode="gemma4_vllm", strip_visible_reasoning=True)
        runner = SyntheticGenerationRunner(
            claim_agent=ClaimAgent(
                llm_client=fake_client,  # type: ignore[arg-type]
                prompt_path=ROOT / "prompts" / "synthetic" / "claim_agent_v3.md",
                reasoning_policy=policy,
            ),
            qa_agent=QAAgent(
                llm_client=fake_client,  # type: ignore[arg-type]
                prompt_path=ROOT / "prompts" / "synthetic" / "qa_agent_v4.md",
                reasoning_policy=policy,
            ),
        )

        await runner.run_unit(unit, unit_index=0)

        claim_prompt = fake_client.calls[0]["messages"][0]["content"]
        qa_prompt = fake_client.calls[1]["messages"][0]["content"]
        self.assertIn("Steady rain falls outside.", claim_prompt)
        self.assertIn("Water patters during a shower.", claim_prompt)
        self.assertNotIn("A long storm continues overnight.", claim_prompt)
        self.assertNotIn("A dog barks near a gate.", claim_prompt)
        self.assertNotIn("private/one.wav", claim_prompt)
        self.assertNotIn("additionalProperties", claim_prompt)
        self.assertNotIn("condition_name", claim_prompt)
        self.assertNotIn("Relevant captions:", qa_prompt)
        self.assertNotIn("A dog barks near a gate.", qa_prompt)
        self.assertLess(len(claim_prompt), 3600)
        self.assertLess(len(qa_prompt), 2200)

    def test_qa_prompt_allows_varied_questions_and_canonical_answers(self) -> None:
        prompt = (
            ROOT / "prompts" / "synthetic" / "qa_agent_v4.md"
        ).read_text(encoding="utf-8")

        self.assertIn("naturally varied question", prompt)
        self.assertIn("use only the distinguishing part", prompt)
        self.assertIn("Do not repeatedly use the template", prompt)
        self.assertIn('["contradicted", "AUDIO_2"]', prompt)
        self.assertIn('["unsupported", "NONE"]', prompt)
        self.assertIn("Never put claim_type values", prompt)

        verifier_prompt = (
            ROOT / "prompts" / "synthetic" / "verifier_agent_v4.md"
        ).read_text(encoding="utf-8")
        self.assertIn('UNSUPPORTED to ["unsupported", "NONE"]', verifier_prompt)

    def test_claim_prompt_requires_atomic_and_careful_source_grounding(self) -> None:
        prompt = (
            ROOT / "prompts" / "synthetic" / "claim_agent_v3.md"
        ).read_text(encoding="utf-8")

        self.assertIn("one atomic event or attribute", prompt)
        self.assertIn("contradiction_basis to \"none\"", prompt)
        self.assertIn("caption silence as proof", prompt)
        self.assertIn("change only the requested dimension", prompt)
        self.assertIn("For unsupported_detail", prompt)

    def test_source_references_and_feedback_are_bounded(self) -> None:
        labels = prompt_audio_source_labels(
            audio_unit(),
            {
                "evidence_sources": ["AUDIO_1"],
                "instruction": "Move one fact from AUDIO_1 to AUDIO_2.",
            },
        )
        self.assertEqual(labels, ["AUDIO_1", "AUDIO_2"])

        context = format_audio_context(
            audio_unit(),
            source_labels=["AUDIO_1"],
        )
        self.assertIn("AUDIO_1 captions:", context)
        self.assertNotIn("audio-1", context)
        self.assertNotIn("one.wav", context)
        self.assertNotIn("AUDIO_2", context)

        feedback = validation_feedback(
            [
                f"ClaimAgent attempt 1: {'x' * 1000}",
                "QAAgent attempt 1: unrelated",
            ],
            "ClaimAgent",
            max_chars_per_item=100,
        )
        self.assertNotIn("QAAgent", feedback)
        self.assertLessEqual(len(feedback), 102)


class VLLMResponseSchemaTests(unittest.TestCase):
    def test_vllm_schema_omits_unique_items_without_mutating_local_schema(self) -> None:
        response_format = response_format_json_schema(
            "claim_agent_output",
            CLAIM_OUTPUT_SCHEMA,
        )
        wire_schema = response_format["json_schema"]["schema"]

        self.assertFalse(contains_key(wire_schema, "uniqueItems"))
        self.assertTrue(contains_key(CLAIM_OUTPUT_SCHEMA, "uniqueItems"))

    def test_local_schema_still_rejects_duplicate_evidence_sources(self) -> None:
        claim = valid_claim()
        claim["evidence_sources"] = ["AUDIO_1", "AUDIO_1"]

        with self.assertRaises(SchemaValidationError):
            validate_json(claim, CLAIM_OUTPUT_SCHEMA)

    def test_qa_schema_requires_a_list_answer(self) -> None:
        qa = valid_qa()
        validate_json(qa, QA_OUTPUT_SCHEMA)

        qa["answer"] = "faithful, AUDIO_1"
        with self.assertRaises(SchemaValidationError):
            validate_json(qa, QA_OUTPUT_SCHEMA)

        qa["answer"] = ["supported", "AUDIO_1", "AUDIO_2"]
        with self.assertRaises(SchemaValidationError):
            validate_json(qa, QA_OUTPUT_SCHEMA)

    def test_multi_source_answer_is_rejected_by_fixed_benchmark_contract(self) -> None:
        claim = valid_claim()
        claim["evidence_sources"] = ["AUDIO_1", "AUDIO_2"]
        qa = valid_qa()
        qa["answer"] = ["supported", "AUDIO_1"]
        qa["answer_source"] = ["AUDIO_1", "AUDIO_2"]
        qa["required_evidence_sources"] = ["AUDIO_1", "AUDIO_2"]

        with self.assertRaisesRegex(
            ValueError,
            "must have exactly one evidence source",
        ):
            validate_qa_record(qa, claim, audio_count=2)

    def test_unsupported_answer_uses_none_and_empty_source_lists(self) -> None:
        claim = unsupported_claim()
        qa = unsupported_qa()

        validate_qa_record(qa, claim, audio_count=2)

        qa["answer"] = ["unsupported", "AUDIO_1"]
        with self.assertRaisesRegex(ValueError, r"\['unsupported', 'NONE'\]"):
            validate_qa_record(qa, claim, audio_count=2)

    def test_contradicted_answer_uses_its_determining_source(self) -> None:
        claim = valid_claim()
        claim["claim_type"] = "counterfactual"
        claim["claim_status"] = "CONTRADICTED"
        claim["evidence_sources"] = ["AUDIO_2"]
        qa = valid_qa()
        qa["answer"] = ["contradicted", "AUDIO_2"]
        qa["answer_source"] = ["AUDIO_2"]
        qa["required_evidence_sources"] = ["AUDIO_2"]

        validate_qa_record(qa, claim, audio_count=2)

    def test_generation_targets_have_one_determining_source(self) -> None:
        for audio_count in (1, 2, 3, 5):
            with self.subTest(audio_count=audio_count):
                conditions = build_target_conditions(audio_count)
                self.assertTrue(conditions)
                unsupported = [
                    condition
                    for condition in conditions
                    if condition.claim_status == "UNSUPPORTED"
                ]
                self.assertEqual(len(unsupported), 1)
                self.assertEqual(unsupported[0].evidence_sources, [])
                self.assertTrue(
                    all(
                        len(condition.evidence_sources) == 1
                        for condition in conditions
                        if condition.claim_status != "UNSUPPORTED"
                    )
                )

    def test_unsupported_claim_has_no_evidence_or_contradiction(self) -> None:
        target = next(
            condition
            for condition in build_target_conditions(2)
            if condition.claim_status == "UNSUPPORTED"
        )

        validate_claim_record(unsupported_claim(), target, audio_count=2)


def audio_unit() -> Dict[str, Any]:
    return {
        "unit_id": "unit-1",
        "schema_version": "audio_unit_v0",
        "grounding_standard": "caption_grounded",
        "audio_count": 2,
        "audio_ids": ["audio-1", "audio-2"],
        "local_audio_paths": ["one.wav", "two.wav"],
        "audio_file_names": ["audio/one.wav", "audio/two.wav"],
        "audio_captions": [
            ["Steady rain falls outside."],
            ["Cars pass on a busy street."],
        ],
    }


def valid_claim() -> Dict[str, Any]:
    return {
        "claim_text": "Steady rain falls outside.",
        "claim_type": "faithful",
        "claim_status": "SUPPORTED",
        "evidence_sources": ["AUDIO_1"],
        "counterfactual_edit_type": "none",
        "supporting_caption_phrases": ["Steady rain falls outside."],
        "contradiction_basis": "",
        "forbidden_inferences": [],
        "confidence": 0.98,
    }


def valid_qa() -> Dict[str, Any]:
    return {
        "question": "How should the rainfall description be classified, and which audio supports it?",
        "answer": ["supported", "AUDIO_1"],
        "answer_source": ["AUDIO_1"],
        "claim_evaluation_explanation": "AUDIO_1 explicitly describes steady rain.",
        "required_evidence_sources": ["AUDIO_1"],
    }


def unsupported_claim() -> Dict[str, Any]:
    claim = valid_claim()
    claim.update(
        {
            "claim_text": "Steady rain falls outside during the morning.",
            "claim_type": "counterfactual",
            "claim_status": "UNSUPPORTED",
            "evidence_sources": [],
            "counterfactual_edit_type": "unsupported_detail",
            "supporting_caption_phrases": [],
            "contradiction_basis": "none",
            "confidence": 0.9,
        }
    )
    return claim


def unsupported_qa() -> Dict[str, Any]:
    qa = valid_qa()
    qa.update(
        {
            "answer": ["unsupported", "NONE"],
            "answer_source": [],
            "claim_evaluation_explanation": (
                "No caption establishes or contradicts the morning detail."
            ),
            "required_evidence_sources": [],
        }
    )
    return qa


def contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(
            contains_key(item, target) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(item, target) for item in value)
    return False


if __name__ == "__main__":
    unittest.main()
