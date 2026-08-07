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
    def test_qwen_is_the_active_default_profile(self) -> None:
        config = load_yaml(ROOT / "configs" / "data_synthetic.yaml")

        self.assertEqual(
            DEFAULT_VLLM_CONFIG_PATH,
            Path("configs/vllm_client_qwen36.yaml"),
        )
        self.assertTrue(
            config["vllm"]["config_path"].endswith(
                "configs/vllm_client_qwen36.yaml"
            )
        )

    def test_client_and_server_profiles_use_matching_names_and_parsers(self) -> None:
        cases = [
            ("gemma4", "gemma4_vllm", "gemma4"),
            ("qwen36", "qwen3_vllm", "qwen3"),
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

    def test_gemma_uses_reasoning_effort_request_fields(self) -> None:
        config = load_yaml(ROOT / "configs" / "vllm_client_gemma4.yaml")
        extra_body = build_request_extra_body(config, config["client"])

        self.assertEqual(extra_body["reasoning_effort"], "medium")
        self.assertFalse(extra_body["include_reasoning"])
        self.assertNotIn("chat_template_kwargs", extra_body)

    def test_qwen_uses_chat_template_thinking_fields(self) -> None:
        config = load_yaml(ROOT / "configs" / "vllm_client_qwen36.yaml")
        extra_body = build_request_extra_body(config, config["client"])

        self.assertNotIn("reasoning_effort", extra_body)
        self.assertFalse(extra_body["include_reasoning"])
        self.assertEqual(
            extra_body["chat_template_kwargs"],
            {"enable_thinking": True, "preserve_thinking": False},
        )
        self.assertEqual(extra_body["top_k"], 20)

        server = load_yaml(ROOT / "configs" / "vllm_server_qwen36.yaml")
        self.assertTrue(server["language-model-only"])
        self.assertEqual(
            server["speculative-config"],
            {
                "method": "qwen3_next_mtp",
                "num_speculative_tokens": 2,
            },
        )

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
