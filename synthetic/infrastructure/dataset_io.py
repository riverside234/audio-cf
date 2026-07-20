"""Dataset reading/writing helpers for synthetic generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


REQUIRED_AUDIO_UNIT_COLUMNS = [
    "unit_id",
    "schema_version",
    "grounding_standard",
    "audio_count",
    "audio_ids",
    "audio_file_names",
    "audio_captions",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read Parquet. Install it with `pip install pyarrow`."
        ) from exc
    return pq.read_table(path).to_pylist()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")


def write_parquet(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write Parquet. Install it with `pip install pyarrow`."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows)), path)


def batched(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def validate_audio_unit_rows(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No audio-unit rows were loaded.")

    missing = [column for column in REQUIRED_AUDIO_UNIT_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f"Audio-unit rows are missing required columns: {missing}")

    for index, row in enumerate(rows):
        audio_count = int(row.get("audio_count", 0))
        if audio_count < 1:
            raise ValueError(f"Row {index} has invalid audio_count={audio_count}.")

        for column in ["audio_ids", "audio_file_names", "audio_captions"]:
            value = row.get(column)
            if not isinstance(value, list):
                raise ValueError(f"Row {index} column {column} must be a list.")
            if len(value) != audio_count:
                raise ValueError(
                    f"Row {index} column {column} length {len(value)} "
                    f"does not match audio_count={audio_count}."
                )

        local_paths = row.get("local_audio_paths")
        if local_paths is not None and len(local_paths) != audio_count:
            raise ValueError(
                f"Row {index} local_audio_paths length {len(local_paths)} "
                f"does not match audio_count={audio_count}."
            )

