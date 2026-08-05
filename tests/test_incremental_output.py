from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from synthetic.infrastructure.output_writer import IncrementalOutputWriter


class IncrementalOutputWriterTests(unittest.TestCase):
    def test_completed_batch_is_durable_before_run_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            writer = make_writer(output_dir)

            writer.write_batch(
                start_index=0,
                input_count=2,
                examples=[{"example_id": "example-1"}],
                audit_rows=[],
                error_rows=[{"unit_id": "unit-2", "message": "failed"}],
            )

            batch_dir = output_dir / "generation_batches" / "batch-000000000-000000001"
            self.assertTrue((batch_dir / "batch.json").is_file())
            self.assertEqual(
                read_jsonl(batch_dir / "examples.jsonl"),
                [{"example_id": "example-1"}],
            )
            self.assertEqual(
                read_jsonl(batch_dir / "errors.jsonl"),
                [{"message": "failed", "unit_id": "unit-2"}],
            )

    def test_finalize_combines_batches_without_an_in_memory_run_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            writer = make_writer(output_dir)
            writer.write_batch(
                start_index=0,
                input_count=1,
                examples=[{"example_id": "example-1"}],
                audit_rows=[],
                error_rows=[],
            )
            writer.write_batch(
                start_index=1,
                input_count=1,
                examples=[{"example_id": "example-2"}],
                audit_rows=[],
                error_rows=[{"unit_id": "unit-2", "message": "failed"}],
            )

            writer.finalize()

            self.assertEqual(
                read_jsonl(output_dir / "examples.jsonl"),
                [{"example_id": "example-1"}, {"example_id": "example-2"}],
            )
            self.assertEqual(
                read_jsonl(output_dir / "errors.jsonl"),
                [{"message": "failed", "unit_id": "unit-2"}],
            )
            self.assertFalse((output_dir / "generation_batches").exists())

    def test_finalize_merges_parquet_parts_with_compatible_inferred_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            writer = make_writer(
                output_dir,
                write_parquet_enabled=True,
                write_audit_enabled=True,
            )
            writer.write_batch(
                start_index=0,
                input_count=1,
                examples=[{"example_id": "example-1", "labels": []}],
                audit_rows=[{"example_id": "example-1", "validation_errors": []}],
                error_rows=[],
            )
            writer.write_batch(
                start_index=1,
                input_count=1,
                examples=[{"example_id": "example-2", "labels": ["AUDIO_1"]}],
                audit_rows=[
                    {
                        "example_id": "example-2",
                        "validation_errors": ["retried once"],
                    }
                ],
                error_rows=[],
            )

            writer.finalize()

            self.assertEqual(
                pq.read_table(output_dir / "examples.parquet").to_pylist(),
                [
                    {"example_id": "example-1", "labels": []},
                    {"example_id": "example-2", "labels": ["AUDIO_1"]},
                ],
            )
            self.assertEqual(
                pq.read_table(output_dir / "examples_audit.parquet").to_pylist(),
                [
                    {"example_id": "example-1", "validation_errors": []},
                    {
                        "example_id": "example-2",
                        "validation_errors": ["retried once"],
                    },
                ],
            )


def make_writer(
    output_dir: Path,
    *,
    write_parquet_enabled: bool = False,
    write_audit_enabled: bool = False,
) -> IncrementalOutputWriter:
    return IncrementalOutputWriter(
        checkpoint_dir=output_dir / "generation_batches",
        examples_parquet=output_dir / "examples.parquet",
        examples_jsonl=output_dir / "examples.jsonl",
        examples_audit_parquet=output_dir / "examples_audit.parquet",
        sample_for_human_review_jsonl=output_dir / "review.jsonl",
        errors_jsonl=output_dir / "errors.jsonl",
        write_parquet_enabled=write_parquet_enabled,
        write_jsonl_enabled=True,
        write_audit_enabled=write_audit_enabled,
        review_sample_size=10,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    unittest.main()
