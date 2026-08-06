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
from synthetic.agents.schemas import CLAIM_OUTPUT_SCHEMA, response_format_json_schema
from synthetic.infrastructure.schema_io import SchemaValidationError, validate_json


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
            "{audio_context}\n{target_condition_json}\n{claim_schema_json}\n"
            "{validation_feedback}\n{reasoning_instruction}\n",
            encoding="utf-8",
        )
        self.qa_prompt.write_text(
            "{audio_context}\n{target_condition_json}\n{claim_record_json}\n"
            "{qa_schema_json}\n{validation_feedback}\n{reasoning_instruction}\n",
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
        self.assertEqual(state.final_example["claim_status"], "SUPPORTED")
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
        "question": "Is the rainfall claim supported by the captions?",
        "answer": "The claim is supported by AUDIO_1.",
        "answer_source": "AUDIO_1",
        "claim_evaluation_explanation": "AUDIO_1 explicitly describes steady rain.",
        "required_evidence_sources": ["AUDIO_1"],
    }


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
