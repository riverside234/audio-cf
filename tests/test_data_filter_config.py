from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_filter import parse_args


class DataFilterConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "data_filter.yaml"
        self.config_path.write_text(
            """
input:
  manifest_path: "${cwd}/custom_manifest.parquet"
  splits: [development, validation]
output:
  output_dir: "${cwd}/custom_output"
  write_parquet: true
  write_jsonl: false
sampling:
  subset_size: 0
  unit_count: 100000
  audio_count: 2
  random_seed: 7
  grouping_strategy: random
  max_audio_reuse: null
  max_enumerated_units: 100000
similarity:
  similarity_min: 0.2
  similarity_max: 0.9
  dissimilarity_max: 0.04
runtime:
  overwrite: false
  strict: true
""".lstrip(),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_config_supplies_large_run_defaults(self) -> None:
        args = parse_args(["--config", str(self.config_path)])

        self.assertEqual(args.unit_count, 100000)
        self.assertEqual(args.subset_size, 0)
        self.assertEqual(args.splits, ["development", "validation"])
        self.assertTrue(args.write_parquet)
        self.assertFalse(args.write_jsonl)
        self.assertTrue(args.strict)
        self.assertEqual(
            Path(args.input_manifest_path),
            Path.cwd() / "custom_manifest.parquet",
        )

    def test_cli_values_override_config_defaults(self) -> None:
        args = parse_args(
            [
                "--config",
                str(self.config_path),
                "--unit-count",
                "25",
                "--write-jsonl",
                "--no-strict",
            ]
        )

        self.assertEqual(args.unit_count, 25)
        self.assertTrue(args.write_jsonl)
        self.assertFalse(args.strict)


if __name__ == "__main__":
    unittest.main()
