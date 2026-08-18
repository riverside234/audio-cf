from __future__ import annotations

import unittest
from pathlib import Path

from data_synthetic import (
    DEFAULT_VLLM_CONFIG_PATH,
    build_request_extra_body,
    load_yaml,
)
from synthetic.infrastructure.schema_io import strip_visible_reasoning


ROOT = Path(__file__).resolve().parents[1]


class ModelProfileTests(unittest.TestCase):
    def test_configured_and_fallback_profiles_exist(self) -> None:
        config = load_yaml(ROOT / "configs" / "data_synthetic.yaml")
        configured_path = str(config["vllm"]["config_path"]).replace(
            "${cwd}/", ""
        )

        self.assertTrue((ROOT / configured_path).is_file())
        self.assertTrue((ROOT / DEFAULT_VLLM_CONFIG_PATH).is_file())

    def test_client_and_server_profiles_use_matching_names_and_parsers(self) -> None:
        cases = [
            ("gemma4", "gemma4_vllm", "gemma4"),
            ("qwen38", "qwen3_vllm", "qwen3"),
        ]

        for profile, reasoning_mode, parser in cases:
            with self.subTest(profile=profile):
                client = load_yaml(ROOT / "configs" / f"vllm_client_{profile}.yaml")
                server = load_yaml(ROOT / "configs" / f"vllm_server_{profile}.yaml")

                self.assertEqual(
                    client["client"]["model"],
                    server["served-model-name"],
                )
                self.assertEqual(
                    client["generation"]["reasoning"]["mode"],
                    reasoning_mode,
                )
                self.assertEqual(server["reasoning-parser"], parser)
                self.assertEqual(server["dtype"], "auto")
                self.assertEqual(server["max-model-len"], 8192)
                for prompt_path in client["agents"]["prompts"].values():
                    self.assertTrue((ROOT / prompt_path).is_file())
                for agent_name in (
                    "default",
                    "claim_agent",
                    "qa_agent",
                    "verifier_agent",
                ):
                    self.assertEqual(
                        client["generation"][agent_name]["max_tokens"],
                        2048,
                    )

    def test_gemma_uses_reasoning_effort_request_fields(self) -> None:
        config = load_yaml(ROOT / "configs" / "vllm_client_gemma4.yaml")
        extra_body = build_request_extra_body(config, config["client"])

        self.assertEqual(extra_body["reasoning_effort"], "low")
        self.assertFalse(extra_body["include_reasoning"])
        self.assertEqual(extra_body["thinking_token_budget"], 1024)
        self.assertEqual(extra_body["top_k"], 64)
        self.assertNotIn("chat_template_kwargs", extra_body)

        server = load_yaml(ROOT / "configs" / "vllm_server_gemma4.yaml")
        self.assertEqual(
            server["structured-outputs-config"],
            {
                "backend": "xgrammar",
                "disable_any_whitespace": True,
            },
        )

    def test_qwen38_uses_native_reasoning_and_chat_template_fields(self) -> None:
        config = load_yaml(ROOT / "configs" / "vllm_client_qwen38.yaml")
        extra_body = build_request_extra_body(config, config["client"])

        self.assertEqual(extra_body["reasoning_effort"], "low")
        self.assertTrue(extra_body["include_reasoning"])
        self.assertEqual(extra_body["thinking_token_budget"], 1024)
        self.assertEqual(
            extra_body["chat_template_kwargs"],
            {"enable_thinking": True, "preserve_thinking": False},
        )
        self.assertEqual(extra_body["top_k"], 20)

        server = load_yaml(ROOT / "configs" / "vllm_server_qwen38.yaml")
        self.assertTrue(server["language-model-only"])
        self.assertEqual(
            server["speculative-config"],
            {"method": "mtp", "num_speculative_tokens": 3},
        )

    def test_model_profiles_use_stage_specific_synthetic_sampling(self) -> None:
        expected = {
            "default": (0.5, 0.95),
            "claim_agent": (0.7, 0.95),
            "qa_agent": (0.5, 0.95),
            "verifier_agent": (0.0, 1.0),
        }
        for profile in ("gemma4", "qwen38"):
            with self.subTest(profile=profile):
                config = load_yaml(
                    ROOT / "configs" / f"vllm_client_{profile}.yaml"
                )
                for agent_name, (temperature, top_p) in expected.items():
                    sampling = config["generation"][agent_name]
                    self.assertEqual(sampling["temperature"], temperature)
                    self.assertEqual(sampling["top_p"], top_p)
                    self.assertEqual(sampling["max_tokens"], 2048)

    def test_qwen_prefilled_opening_think_token_is_sanitized(self) -> None:
        clean, removed = strip_visible_reasoning(
            'private reasoning from prefilled token\n</think>\n{"answer":"ok"}'
        )

        self.assertTrue(removed)
        self.assertEqual(clean, '{"answer":"ok"}')

    def test_qwen_fallback_does_not_strip_non_json_text(self) -> None:
        text = "ordinary text </think> without a JSON object"

        clean, removed = strip_visible_reasoning(text)

        self.assertFalse(removed)
        self.assertEqual(clean, text)


if __name__ == "__main__":
    unittest.main()
