"""Check real dataset export and safe token alignment without loading models."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from on_policy_distillation.prepare import conversation, export
from on_policy_distillation.train import guard_teacher_requests, shared_response_ids
from on_policy_distillation import train


class DistillationDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.row = {
            "audio_count": 3,
            "local_audio_paths": ["third.wav", "first.wav", "second.wav"],
            "claim_text": "A cat meows.", "question": "Is the claim supported?",
            "claim_status": "SUPPORTED", "evidence_sources": ["AUDIO_2"],
            "answer": ["supported", "AUDIO_2"],
            "audio_captions": [["Private reference caption"]] * 3,
        }
        for path in self.row["local_audio_paths"]:
            (self.root / path).touch()

    def test_all_recordings_keep_source_order_and_captions_stay_private(self):
        result = conversation(self.row, self.root)
        self.assertEqual([Path(path).name for path in result["audios"]], self.row["local_audio_paths"])
        self.assertTrue(all(Path(path).is_absolute() for path in result["audios"]))
        prompt = result["messages"][1]["content"]
        self.assertIn("AUDIO_1: <audio>\nAUDIO_2: <audio>\nAUDIO_3: <audio>", prompt)
        self.assertNotIn("Private reference caption", prompt)
        self.assertNotIn(json.dumps(self.row["answer"]), prompt)
        self.assertEqual(json.loads(result["messages"][-1]["content"]), self.row["answer"])

    def test_relocated_release_audio_is_used_for_the_same_source(self):
        self.row["audio_file_names"] = ["replacement.wav", "first.wav", "second.wav"]
        (self.root / "third.wav").unlink()
        (self.root / "replacement.wav").touch()
        self.assertEqual(Path(conversation(self.row, self.root)["audios"][0]).name, "replacement.wav")

    def test_missing_audio_and_inconsistent_labels_are_rejected(self):
        (self.root / "third.wav").unlink()
        with self.assertRaises(FileNotFoundError):
            conversation(self.row, self.root)
        self.row["answer"] = ["supported", "AUDIO_4"]
        with self.assertRaises(ValueError):
            conversation(self.row, self.root)
        self.row["answer"] = ["contradicted", "AUDIO_2"]
        with self.assertRaises(ValueError):
            conversation(self.row, self.root)

    def test_parquet_export_honors_limit_and_preserves_previous_output_on_failure(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        source, target = self.root / "examples.parquet", self.root / "train.jsonl"
        pq.write_table(pa.Table.from_pylist([self.row, self.row]), source)
        self.assertEqual(export(source, target, self.root, limit=1), 1)
        previous = target.read_text(encoding="utf-8")
        self.assertEqual(len(previous.splitlines()), 1)
        self.assertEqual(json.loads(previous)["audios"][1], str((self.root / "first.wav").resolve()))
        (self.root / "third.wav").unlink()
        with self.assertRaises(FileNotFoundError):
            export(source, target, self.root)
        self.assertEqual(target.read_text(encoding="utf-8"), previous)
        self.assertFalse(list(self.root.glob("*.tmp")))


class TokenAlignmentTests(unittest.TestCase):
    def tokenizer(self, audio_token, changed_text=False):
        vocab = {"a": 0, "b": 1 if not changed_text else 8, "c": 2, "<|im_end|>": 4, audio_token: 5}
        return SimpleNamespace(vocab_size=3, eos_token_id=4, get_vocab=lambda: vocab)

    def test_shared_text_and_eos_are_allowed_but_different_audio_tokens_are_not(self):
        student, teacher = self.tokenizer("<|AUDIO|>"), self.tokenizer("<|object_ref_start|>")
        shared = shared_response_ids(student, teacher)
        self.assertEqual(shared, {0, 1, 2, 4})
        calls = []
        guarded = guard_teacher_requests(lambda samples, template: calls.append((samples, template)), shared)
        samples = [SimpleNamespace(response_token_ids=[0, 1, 4])]
        guarded(samples, "native-template")
        self.assertIs(calls[0][0], samples)
        self.assertEqual(calls[0][1], "native-template")
        with self.assertRaises(ValueError):
            guarded([SimpleNamespace(response_token_ids=[0, 5, 4])])
        with self.assertRaises(ValueError):
            guarded([SimpleNamespace(response_token_ids=[])])
        self.assertEqual(len(calls), 1)

    def test_mismatched_base_vocabulary_fails_before_training(self):
        with self.assertRaises(ValueError):
            shared_response_ids(self.tokenizer("audio"), self.tokenizer("audio", changed_text=True))

    def test_launcher_forwards_overrides_and_restores_native_builder_after_failure(self):
        native = lambda samples, template=None: samples
        adapter = SimpleNamespace(build_teacher_requests=native)
        seen = []

        def run_native(args):
            seen.append(args)
            adapter.build_teacher_requests([SimpleNamespace(response_token_ids=[5])])

        # The real Swift API is inspected separately; the stub exercises our adapter's
        # lifecycle without loading CUDA models or connecting to an inference service.
        def parse_arguments(cls, flags):
            values = {key.removeprefix("--"): value for key, value in zip(flags[::2], flags[1::2])}
            return SimpleNamespace(**values), []

        swift = ModuleType("swift")
        swift.__path__ = []
        modules = {
            "swift": swift,
            "swift.arguments": SimpleNamespace(RLHFArguments=object),
            "swift.utils": SimpleNamespace(parse_args=parse_arguments),
            "swift.pipelines": SimpleNamespace(rlhf_main=run_native),
            "swift.rlhf_trainers": SimpleNamespace(grpo_trainer=adapter),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda model: self.tokenizer("teacher-audio" if "Qwen3" in model else "student-audio")
            )),
        }
        environment = {"TEACHER_MODEL": train.TEACHER, "TEACHER_URL": "http://localhost:8001"}
        with patch.dict(sys.modules, modules), patch.dict(train.os.environ, environment):
            with self.assertRaisesRegex(ValueError, "different teacher meanings"):
                train.main(["--dataset", "custom.jsonl", "--max_steps", "2"])
        self.assertEqual(seen[0].dataset, "custom.jsonl")
        self.assertEqual(seen[0].max_steps, "2")
        self.assertEqual(seen[0].teacher_model_server, "http://localhost:8001")
        self.assertIs(adapter.build_teacher_requests, native)


if __name__ == "__main__":
    unittest.main()
