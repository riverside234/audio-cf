"""Durable batch output for synthetic generation runs."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .dataset_io import write_json, write_jsonl, write_parquet


class IncrementalOutputWriter:
    """Commit completed batches to disk and assemble final run artifacts."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        examples_parquet: Path,
        examples_jsonl: Path,
        examples_audit_parquet: Path,
        sample_for_human_review_jsonl: Path,
        errors_jsonl: Path,
        write_parquet_enabled: bool,
        write_jsonl_enabled: bool,
        write_audit_enabled: bool,
        review_sample_size: int,
    ) -> None:
        if review_sample_size < 0:
            raise ValueError("review_sample_size must be >= 0.")
        self.checkpoint_dir = checkpoint_dir
        self.examples_parquet = examples_parquet
        self.examples_jsonl = examples_jsonl
        self.examples_audit_parquet = examples_audit_parquet
        self.sample_for_human_review_jsonl = sample_for_human_review_jsonl
        self.errors_jsonl = errors_jsonl
        self.write_parquet_enabled = write_parquet_enabled
        self.write_jsonl_enabled = write_jsonl_enabled
        self.write_audit_enabled = write_audit_enabled
        self.review_sample_size = review_sample_size

    def write_batch(
        self,
        *,
        start_index: int,
        input_count: int,
        examples: Sequence[Dict[str, Any]],
        audit_rows: Sequence[Dict[str, Any]],
        error_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Atomically persist one completed input batch."""

        if start_index < 0:
            raise ValueError("start_index must be >= 0.")
        if input_count < 1:
            raise ValueError("input_count must be >= 1.")

        end_index = start_index + input_count - 1
        batch_name = f"batch-{start_index:09d}-{end_index:09d}"
        final_dir = self.checkpoint_dir / batch_name
        if final_dir.exists():
            raise FileExistsError(f"Batch output already exists: {final_dir}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = self.checkpoint_dir / f".{batch_name}.{uuid.uuid4().hex}.tmp"
        temp_dir.mkdir(parents=False, exist_ok=False)
        metadata = {
            "batch_name": batch_name,
            "start_index": start_index,
            "end_index": end_index,
            "input_count": input_count,
            "examples_count": len(examples),
            "audit_count": len(audit_rows),
            "errors_count": len(error_rows),
        }

        try:
            write_jsonl(temp_dir / "examples.jsonl", examples)
            write_jsonl(temp_dir / "errors.jsonl", error_rows)
            if self.write_audit_enabled:
                write_jsonl(temp_dir / "audit.jsonl", audit_rows)
            if self.write_parquet_enabled and examples:
                write_parquet(temp_dir / "examples.parquet", examples)
            if self.write_audit_enabled and audit_rows:
                write_parquet(temp_dir / "audit.parquet", audit_rows)
            write_json(temp_dir / "batch.json", metadata)
            os.replace(temp_dir, final_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return metadata

    def finalize(self) -> None:
        """Merge committed batch parts into the configured final artifacts."""

        batch_dirs = self._batch_dirs()
        if not batch_dirs:
            raise RuntimeError("No completed batch outputs are available to finalize.")

        example_jsonl_parts = [path / "examples.jsonl" for path in batch_dirs]
        error_jsonl_parts = [path / "errors.jsonl" for path in batch_dirs]
        if self.write_jsonl_enabled:
            _combine_files_atomically(example_jsonl_parts, self.examples_jsonl)
        _combine_files_atomically(error_jsonl_parts, self.errors_jsonl)

        if self.write_parquet_enabled:
            example_parquet_parts = [
                path / "examples.parquet"
                for path in batch_dirs
                if (path / "examples.parquet").exists()
            ]
            if example_parquet_parts:
                _merge_parquet_files(example_parquet_parts, self.examples_parquet)

        if self.write_audit_enabled:
            audit_parquet_parts = [
                path / "audit.parquet"
                for path in batch_dirs
                if (path / "audit.parquet").exists()
            ]
            if audit_parquet_parts:
                _merge_parquet_files(
                    audit_parquet_parts,
                    self.examples_audit_parquet,
                )
            audit_jsonl_parts = [
                path / "audit.jsonl"
                for path in batch_dirs
                if (path / "audit.jsonl").exists()
            ]
            review_rows = _take_jsonl_rows(
                audit_jsonl_parts,
                self.review_sample_size,
            )
            write_jsonl(self.sample_for_human_review_jsonl, review_rows)

        shutil.rmtree(self.checkpoint_dir)

    def _batch_dirs(self) -> List[Path]:
        if not self.checkpoint_dir.exists():
            return []
        return sorted(
            path
            for path in self.checkpoint_dir.glob("batch-*")
            if path.is_dir() and (path / "batch.json").is_file()
        )


def _combine_files_atomically(part_paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling(output_path)
    try:
        with temp_path.open("wb") as output_handle:
            for part_path in part_paths:
                with part_path.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output_handle)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _merge_parquet_files(part_paths: Sequence[Path], output_path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to merge Parquet batch outputs."
        ) from exc

    schemas = [pq.read_schema(path) for path in part_paths]
    unified_schema = pa.unify_schemas(schemas, promote_options="permissive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling(output_path)
    try:
        with pq.ParquetWriter(temp_path, unified_schema) as writer:
            for part_path in part_paths:
                table = pq.read_table(part_path)
                if table.schema != unified_schema:
                    table = table.cast(unified_schema, safe=False)
                writer.write_table(table)
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _take_jsonl_rows(part_paths: Iterable[Path], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if limit <= 0:
        return rows
    for part_path in part_paths:
        with part_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
                if len(rows) >= limit:
                    return rows
    return rows


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
