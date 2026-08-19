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
    VerifierAgent,
)
from synthetic.agents.conditions import build_target_conditions
from synthetic.agents.schemas import (
    CLAIM_OUTPUT_SCHEMA,
    EXAMPLE_SCHEMA_VERSION,
    QA_GENERATION_SCHEMA,
    QA_OUTPUT_SCHEMA,
    response_format_json_schema,
)
from synthetic.agents.state import (
    format_audio_context,
    prompt_audio_source_labels,
    validation_feedback,
)
from synthetic.agents.validators import (
    validate_claim_record,
    validate_qa_record,
    validate_verifier_record,
)
from synthetic.infrastructure.schema_io import (
    SchemaValidationError,
    parse_json_object,
    validate_json,
)


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
        self.verifier_prompt = prompt_dir / "verifier.md"
        self.verifier_prompt.write_text(
            "{audio_context}\n{target_condition_json}\n{claim_record_json}\n"
            "{qa_record_json}\n{validation_feedback}\n{reasoning_instruction}\n",
            encoding="utf-8",
        )

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    async def test_fake_client_generates_valid_example_and_strips_reasoning(self) -> None:
        claim_response = f"<think>private check</think>\n{json.dumps(valid_claim())}"
        fake_client = FakeLLMClient(
            [
                claim_response,
                json.dumps(valid_qa_generation()),
                json.dumps(valid_verifier()),
            ]
        )
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
            "claim_type",
            "counterfactual_edit_type",
        }
        self.assertTrue(provenance_fields.isdisjoint(state.final_example))
        self.assertTrue((state.example_id or "").startswith("synthetic_"))
        audit_row = build_audit_row(state)
        self.assertEqual(audit_row["example_id"], state.example_id)
        self.assertEqual(audit_row["unit_id"], "unit-1")
        self.assertEqual(state.visible_reasoning_stripped, ["claim_agent"])
        self.assertNotIn("private check", state.raw_claim_text or "")
        self.assertEqual(state.verifier_record, valid_verifier())
        self.assertEqual(len(fake_client.calls), 3)
        for call in fake_client.calls:
            self.assertEqual(call["response_format"]["type"], "json_schema")
        qa_wire_schema = fake_client.calls[1]["response_format"]["json_schema"][
            "schema"
        ]
        claim_wire_schema = fake_client.calls[0]["response_format"]["json_schema"][
            "schema"
        ]
        self.assertNotIn("claim_type", claim_wire_schema["properties"])
        self.assertNotIn(
            "counterfactual_edit_type",
            claim_wire_schema["properties"],
        )
        self.assertEqual(
            set(qa_wire_schema["properties"]),
            {"question", "claim_evaluation_explanation"},
        )
        self.assertEqual(
            json.loads(state.raw_qa_text or "{}"),
            valid_qa_generation(),
        )

    async def test_semantic_validation_failure_retries_claim(self) -> None:
        invalid_claim = valid_claim()
        invalid_claim["claim_status"] = "CONTRADICTED"
        fake_client = FakeLLMClient(
            [
                json.dumps(invalid_claim),
                json.dumps(valid_claim()),
                json.dumps(valid_qa_generation()),
                json.dumps(valid_verifier()),
            ]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=0)

        self.assertEqual(len(fake_client.calls), 4)
        self.assertEqual(len(state.validation_errors), 1)
        self.assertIn("claim_status must be 'SUPPORTED'", state.validation_errors[0])
        self.assertIn("ClaimAgent attempt 1", fake_client.calls[1]["messages"][0]["content"])
        self.assertNotIn(
            "ClaimAgent attempt 1",
            fake_client.calls[2]["messages"][0]["content"],
        )

    async def test_qa_question_retries_while_answer_is_constructed(self) -> None:
        invalid_qa = valid_qa_generation()
        invalid_qa["question"] = "Rain?"
        fake_client = FakeLLMClient(
            [
                json.dumps(valid_claim()),
                json.dumps(invalid_qa),
                json.dumps(valid_qa_generation()),
                json.dumps(valid_verifier()),
            ]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=0)

        self.assertEqual(len(fake_client.calls), 4)
        self.assertEqual(state.final_example["answer"], ["supported", "AUDIO_1"])
        self.assertIn(
            "question must contain at least six words",
            state.validation_errors[0],
        )

    async def test_runner_generates_same_audio_contradiction(self) -> None:
        fake_client = FakeLLMClient(
            [
                json.dumps(contradicted_claim()),
                json.dumps(contradicted_qa_generation()),
                json.dumps(valid_verifier()),
            ]
        )
        runner = self._runner(fake_client)

        state = await runner.run_unit(audio_unit(), unit_index=1)

        self.assertEqual(state.target_condition["claim_status"], "CONTRADICTED")
        self.assertEqual(state.target_condition["evidence_sources"], ["AUDIO_1"])
        self.assertEqual(state.final_example["answer"], ["contradicted", "AUDIO_1"])

    async def test_verifier_fail_rejects_example_before_finalization(self) -> None:
        fake_client = FakeLLMClient(
            [
                json.dumps(valid_claim()),
                json.dumps(valid_qa_generation()),
                json.dumps(failed_verifier("The question is unrelated to the claim.")),
            ]
        )
        runner = self._runner(fake_client)

        with self.assertRaisesRegex(
            ValueError,
            "Verifier rejected example: The question is unrelated",
        ):
            await runner.run_unit(audio_unit(), unit_index=0)

        self.assertEqual(len(fake_client.calls), 3)

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
        verifier_agent = VerifierAgent(
            llm_client=fake_client,  # type: ignore[arg-type]
            prompt_path=self.verifier_prompt,
            reasoning_policy=policy,
        )
        return SyntheticGenerationRunner(
            claim_agent=claim_agent,
            qa_agent=qa_agent,
            verifier_agent=verifier_agent,
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
            [
                json.dumps(valid_claim()),
                json.dumps(valid_qa_generation()),
                json.dumps(valid_verifier()),
            ]
        )
        policy = ReasoningPolicy(mode="gemma4_vllm", strip_visible_reasoning=True)
        runner = SyntheticGenerationRunner(
            claim_agent=ClaimAgent(
                llm_client=fake_client,  # type: ignore[arg-type]
                prompt_path=ROOT / "prompts" / "synthetic" / "claim_agent_v12.md",
                reasoning_policy=policy,
            ),
            qa_agent=QAAgent(
                llm_client=fake_client,  # type: ignore[arg-type]
                prompt_path=ROOT / "prompts" / "synthetic" / "qa_agent_v9.md",
                reasoning_policy=policy,
            ),
            verifier_agent=VerifierAgent(
                llm_client=fake_client,  # type: ignore[arg-type]
                prompt_path=ROOT
                / "prompts"
                / "synthetic"
                / "verifier_agent_v11.md",
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

    def test_qa_prompt_delegates_canonical_fields_to_code(self) -> None:
        prompt = (
            ROOT / "prompts" / "synthetic" / "qa_agent_v9.md"
        ).read_text(encoding="utf-8")

        previous_prompt = (
            ROOT / "prompts" / "synthetic" / "qa_agent_v8.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Evaluate the literal claim_text", prompt)
        self.assertIn("Vary framing and clause order", prompt)
        self.assertIn("do not imply every modifier conflicts", prompt)
        self.assertIn("Never name an AUDIO_N label", prompt)
        self.assertIn("caption's corrective alternative", prompt)
        self.assertIn("application constructs them deterministically", prompt)
        self.assertNotIn('["contradicted", "AUDIO_N"]', prompt)
        self.assertLess(len(prompt), 1800)

        verifier_prompt = (
            ROOT / "prompts" / "synthetic" / "verifier_agent_v11.md"
        ).read_text(encoding="utf-8")
        previous_verifier_prompt = (
            ROOT / "prompts" / "synthetic" / "verifier_agent_v10.md"
        ).read_text(encoding="utf-8")
        self.assertIn("at least one central claim detail", verifier_prompt)
        self.assertIn("not proof of physical impossibility", verifier_prompt)
        self.assertIn("change missing from the literal claim_text", verifier_prompt)
        self.assertIn("cross-audio source swaps", verifier_prompt)
        self.assertIn("mandatory final acceptance gate", verifier_prompt)
        self.assertIn("empty validation_errors list", verifier_prompt)
        self.assertIn("request for which audio/recording", verifier_prompt)
        self.assertIn("names an AUDIO_N label", verifier_prompt)
        self.assertGreater(len(verifier_prompt), len(previous_verifier_prompt))

    def test_claim_prompt_allows_related_and_explicit_subjective_contrasts(self) -> None:
        prompt = (
            ROOT / "prompts" / "synthetic" / "claim_agent_v12.md"
        ).read_text(encoding="utf-8")

        previous_prompt = (
            ROOT / "prompts" / "synthetic" / "claim_agent_v11.md"
        ).read_text(encoding="utf-8")
        self.assertIn("one coherent event", prompt)
        self.assertIn("combine related propositions", prompt)
        self.assertIn("multiple phrases must describe the same scene", prompt)
        self.assertIn("at least one central caption-established fact", prompt)
        self.assertIn("not by proving that the alternative was physically absent", prompt)
        self.assertIn("do not each need separate contradictory evidence", prompt)
        self.assertIn("gentle versus aggressive", prompt)
        self.assertIn("loud versus quiet", prompt)
        self.assertIn("another audio", prompt)
        self.assertIn("directly stated by a selected caption", prompt)
        self.assertIn("conflicting alternative must appear in claim_text", prompt)
        self.assertIn("compare the literal claim_text", prompt)
        self.assertIn("must not merely restate the caption", prompt)
        self.assertIn("may describe only conflicts present in claim_text", prompt)
        self.assertNotIn("disprove every changed proposition", prompt)
        self.assertNotIn("counterfactual", prompt.lower())
        self.assertNotIn("claim_type", prompt)
        self.assertNotIn("counterfactual_edit_type", prompt)
        self.assertGreater(len(prompt), len(previous_prompt))
        self.assertLess(len(prompt), 3200)

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
    def test_doubled_json_opening_is_rejected_with_actionable_error(self) -> None:
        with self.assertRaisesRegex(
            SchemaValidationError,
            "doubled opening brace.*MTP disabled",
        ):
            parse_json_object('{{"claim_text":"A sound occurs."}')

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

    def test_claim_schema_rejects_removed_taxonomy_fields(self) -> None:
        for field, value in (
            ("claim_type", "faithful"),
            ("counterfactual_edit_type", "none"),
        ):
            with self.subTest(field=field):
                claim = valid_claim()
                claim[field] = value
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

    def test_qa_generation_schema_rejects_model_supplied_answer(self) -> None:
        generated = valid_qa_generation()
        validate_json(generated, QA_GENERATION_SCHEMA)

        generated["answer"] = ["supported", "AUDIO_1"]
        with self.assertRaises(SchemaValidationError):
            validate_json(generated, QA_GENERATION_SCHEMA)

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

    def test_unsupported_status_and_answer_are_rejected(self) -> None:
        claim = valid_claim()
        claim["claim_status"] = "UNSUPPORTED"
        with self.assertRaises(SchemaValidationError):
            validate_json(claim, CLAIM_OUTPUT_SCHEMA)

        qa = valid_qa()
        qa["answer"] = ["unsupported", "NONE"]
        with self.assertRaisesRegex(ValueError, r"\['supported', 'AUDIO_1'\]"):
            validate_qa_record(qa, valid_claim(), audio_count=2)

    def test_contradicted_answer_uses_its_determining_source(self) -> None:
        claim = valid_claim()
        claim["claim_status"] = "CONTRADICTED"
        claim["evidence_sources"] = ["AUDIO_2"]
        qa = valid_qa()
        qa["answer"] = ["contradicted", "AUDIO_2"]
        qa["answer_source"] = ["AUDIO_2"]
        qa["required_evidence_sources"] = ["AUDIO_2"]

        validate_qa_record(qa, claim, audio_count=2)

    def test_qa_question_must_reference_a_meaningful_claim_detail(self) -> None:
        qa = valid_qa()
        qa["question"] = "Which audio contains a barking dog near the gate?"

        with self.assertRaisesRegex(ValueError, "meaningful detail from claim_text"):
            validate_qa_record(qa, valid_claim(), audio_count=2)

        qa["question"] = (
            "Which recording establishes the rainfall judgment for this benchmark?"
        )
        validate_qa_record(qa, valid_claim(), audio_count=2)

    def test_verifier_requires_a_clean_pass(self) -> None:
        validate_verifier_record(valid_verifier(), audio_count=2)

        with self.assertRaisesRegex(ValueError, "Verifier rejected example"):
            validate_verifier_record(
                failed_verifier("The evidence source is incorrect."),
                audio_count=2,
            )

        pass_with_errors = valid_verifier()
        pass_with_errors["validation_errors"] = ["Unexpected residual error."]
        with self.assertRaisesRegex(ValueError, "PASS must have an empty"):
            validate_verifier_record(pass_with_errors, audio_count=2)

        pass_with_correction = valid_verifier()
        pass_with_correction["corrected_claim_status"] = "CONTRADICTED"
        with self.assertRaisesRegex(ValueError, "must not propose"):
            validate_verifier_record(pass_with_correction, audio_count=2)

    def test_contradiction_basis_may_paraphrase_positive_caption_evidence(self) -> None:
        claim = contradicted_claim()
        claim["contradiction_basis"] = (
            "Ongoing rainfall conflicts with the claim of completely dry weather."
        )
        target = build_target_conditions(2)[1]

        validate_claim_record(claim, target, unit_record=audio_unit())

        claim["contradiction_basis"] = (
            "Dry weather is not mentioned in the captions."
        )
        with self.assertRaisesRegex(ValueError, "relies on caption absence"):
            validate_claim_record(claim, target, unit_record=audio_unit())

    def test_supporting_phrases_must_match_declared_source_captions(self) -> None:
        claim = valid_claim()
        target = build_target_conditions(2)[0]

        claim["supporting_caption_phrases"] = ["An elephant trumpets nearby."]
        with self.assertRaisesRegex(ValueError, "unmatched phrases"):
            validate_claim_record(claim, target, unit_record=audio_unit())

        claim["supporting_caption_phrases"] = ["Cars pass on a busy street."]
        with self.assertRaisesRegex(ValueError, "AUDIO_1 captions"):
            validate_claim_record(claim, target, unit_record=audio_unit())

    def test_supporting_phrase_allows_normalized_caption_fragment(self) -> None:
        claim = valid_claim()
        claim["supporting_caption_phrases"] = ["the steady RAIN falls outside"]

        validate_claim_record(
            claim,
            build_target_conditions(2)[0],
            unit_record=audio_unit(),
        )

    def test_claim_can_use_multiple_captions_from_one_audio(self) -> None:
        supported = valid_claim()
        supported["supporting_caption_phrases"] = [
            "Steady rain falls outside.",
            "Rain taps against a roof.",
        ]
        unit = audio_unit()
        unit["audio_captions"][0].append("Rain taps against a roof.")
        validate_claim_record(
            supported,
            build_target_conditions(2)[0],
            unit_record=unit,
        )

        contradicted = contradicted_claim()
        contradicted["supporting_caption_phrases"] = [
            "Steady rain falls outside.",
            "Rain taps against a roof.",
        ]
        contradicted["contradiction_basis"] = (
            "The evidence says 'Steady rain falls outside' and 'Rain taps against "
            "a roof', both of which are incompatible with completely dry weather."
        )
        validate_claim_record(
            contradicted,
            build_target_conditions(2)[1],
            unit_record=unit,
        )

        contradicted["contradiction_basis"] = (
            "Ongoing rain conflicts with the claim of completely dry weather."
        )
        validate_claim_record(
            contradicted,
            build_target_conditions(2)[1],
            unit_record=unit,
        )

    def test_generation_targets_have_one_determining_source(self) -> None:
        for audio_count in (1, 2, 3, 5):
            with self.subTest(audio_count=audio_count):
                conditions = build_target_conditions(audio_count)
                self.assertEqual(len(conditions), 2 * audio_count)
                statuses = [condition.claim_status for condition in conditions]
                self.assertEqual(statuses.count("SUPPORTED"), audio_count)
                self.assertEqual(statuses.count("CONTRADICTED"), audio_count)
                self.assertTrue(
                    all(len(condition.evidence_sources) == 1 for condition in conditions)
                )
                self.assertTrue(
                    all(
                        "source_swap" not in condition.condition_name
                        for condition in conditions
                    )
                )
                self.assertTrue(
                    all(
                        "counterfactual" not in condition.condition_name
                        for condition in conditions
                    )
                )
                self.assertTrue(
                    all(
                        "claim_type" not in condition.to_dict()
                        and "counterfactual_edit_type" not in condition.to_dict()
                        for condition in conditions
                    )
                )


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
        "claim_status": "SUPPORTED",
        "evidence_sources": ["AUDIO_1"],
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


def valid_qa_generation() -> Dict[str, Any]:
    return {
        "question": "How should the rainfall description be classified, and which audio supports it?",
        "claim_evaluation_explanation": "AUDIO_1 explicitly describes steady rain.",
    }


def contradicted_qa_generation() -> Dict[str, Any]:
    return {
        "question": (
            "Which audio determines whether the completely dry weather claim is "
            "supported or contradicted?"
        ),
        "claim_evaluation_explanation": (
            "AUDIO_1 describes steady rain, which conflicts with completely dry weather."
        ),
    }


def valid_verifier() -> Dict[str, Any]:
    return {
        "verifier_status": "PASS",
        "validation_errors": [],
        "validation_notes": "All checks passed.",
        "corrected_claim_status": None,
        "corrected_evidence_sources": [],
    }


def failed_verifier(message: str) -> Dict[str, Any]:
    return {
        "verifier_status": "FAIL",
        "validation_errors": [message],
        "validation_notes": "The example must be rejected.",
        "corrected_claim_status": None,
        "corrected_evidence_sources": [],
    }


def contradicted_claim() -> Dict[str, Any]:
    claim = valid_claim()
    claim.update(
        {
            "claim_text": "The weather outside is completely dry.",
            "claim_status": "CONTRADICTED",
            "contradiction_basis": (
                "The caption explicitly says steady rain falls outside, which is "
                "incompatible with completely dry weather."
            ),
        }
    )
    return claim


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
