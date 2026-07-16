#!/usr/bin/env python3
"""Create reproducible Clotho subsets and unique audio units.

This script is the second deterministic stage of the dataset pipeline:

    full_manifest.parquet -> subset_manifest.parquet + audio_units.parquet

An audio unit is one or more audio records grouped for a single synthetic
example. The original two-audio pair setup is now represented by
--audio-count 2. Use --audio-count 1 for single-audio examples, or larger
values for multi-audio examples.

It intentionally does not parse raw Zenodo CSV files. Run data_process.py first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


GROUNDING_STANDARD = "caption_grounded"
UNIT_SCHEMA_VERSION = "audio_unit_manifest_v0"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a Clotho manifest and create unique audio units."
    )
    parser.add_argument(
        "--input-manifest-path",
        default=os.path.join(os.getcwd(), "data", "log", "full_manifest.parquet"),
        help="Path to full_manifest.parquet or full_manifest.jsonl from data_process.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "data", "final"),
        help="Directory where subset and audio-unit files are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["development"],
        help="Only records from these split names are eligible.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=40,
        help="Number of audio records to sample. Use 0 to keep all eligible records.",
    )
    parser.add_argument(
        "--unit-count",
        "--pair-count",
        dest="unit_count",
        type=int,
        default=20,
        help=(
            "Number of unique audio units to create. --pair-count is kept as a "
            "backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--audio-count",
        type=int,
        default=2,
        help="Number of audio clips per unit. Use 2 for the original paired-audio setup.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible subsetting and grouping.",
    )
    parser.add_argument(
        "--grouping-strategy",
        "--pairing-strategy",
        dest="grouping_strategy",
        choices=["random", "caption_similar", "caption_dissimilar"],
        default="random",
        help=(
            "Audio-unit grouping strategy. Similar/dissimilar use simple lexical "
            "caption overlap. --pairing-strategy is a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--max-audio-reuse",
        type=int,
        default=None,
        help="Maximum times an audio_id may appear across units. Default is unlimited.",
    )
    parser.add_argument(
        "--max-enumerated-units",
        "--max-enumerated-pairs",
        dest="max_enumerated_units",
        type=int,
        default=2_000_000,
        help=(
            "Maximum candidate units to enumerate before falling back to random "
            "attempts. --max-enumerated-pairs is a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--similarity-min",
        type=float,
        default=0.10,
        help="Minimum lexical similarity for caption_similar strategy.",
    )
    parser.add_argument(
        "--similarity-max",
        type=float,
        default=1.00,
        help="Maximum lexical similarity for caption_similar strategy.",
    )
    parser.add_argument(
        "--dissimilarity-max",
        type=float,
        default=0.05,
        help="Maximum lexical similarity for caption_dissimilar strategy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if requested unit_count cannot be satisfied.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Re-run with --overwrite to replace outputs."
        )


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
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return rows


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read Parquet. Install it with `pip install pyarrow`."
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet(path)
    if suffix == ".jsonl":
        return read_jsonl(path)
    raise ValueError(f"Unsupported manifest extension: {path.suffix}")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_parquet(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to write Parquet. Install it with `pip install pyarrow`."
        ) from exc

    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, path)


def ensure_pyarrow_available() -> None:
    try:
        import pyarrow  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required because Parquet files are the canonical handoff "
            "artifacts. Install it with `pip install pyarrow`."
        ) from exc


def validate_manifest_rows(rows: Sequence[Dict[str, Any]]) -> None:
    required = [
        "audio_id",
        "split",
        "local_audio_path",
        "captions",
    ]
    if not rows:
        raise ValueError("Input manifest is empty.")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"Manifest is missing required fields: {missing}")


def filter_by_split(rows: Sequence[Dict[str, Any]], splits: Sequence[str]) -> List[Dict[str, Any]]:
    split_set = set(splits)
    return [row for row in rows if row.get("split") in split_set]


def sample_subset(
    rows: Sequence[Dict[str, Any]],
    subset_size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    if subset_size <= 0 or subset_size >= len(shuffled):
        return shuffled
    return shuffled[:subset_size]


def get_list(record: Dict[str, Any], key: str) -> List[str]:
    value = record.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def lean_manifest_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "audio_id": record.get("audio_id", ""),
        "split": record.get("split", ""),
        "original_file_name": record.get("original_file_name", ""),
        "local_audio_path": record.get("local_audio_path", ""),
        "duration_seconds": record.get("duration_seconds"),
        "captions": get_list(record, "captions"),
        "caption_summary": record.get("caption_summary", ""),
        "keywords": get_list(record, "keywords"),
        "sound_id": record.get("sound_id", ""),
    }


def canonical_unit_key(audio_ids: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(audio_id) for audio_id in audio_ids))


def unit_id_for(audio_ids: Sequence[str]) -> str:
    import hashlib

    key = "::".join(canonical_unit_key(audio_ids))
    return f"unit_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


def caption_tokens(record: Dict[str, Any]) -> Set[str]:
    text = record.get("caption_summary") or " ".join(get_list(record, "captions"))
    return set(TOKEN_RE.findall(str(text).lower()))


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(union))


def group_similarity(indices: Sequence[int], token_sets: Sequence[Set[str]]) -> float:
    if len(indices) < 2:
        return 0.0

    scores = [
        jaccard_similarity(token_sets[i], token_sets[j])
        for i, j in itertools.combinations(indices, 2)
    ]
    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def candidate_allowed_by_strategy(
    indices: Sequence[int],
    strategy: str,
    token_sets: Sequence[Set[str]],
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> bool:
    if strategy == "random" or len(indices) < 2:
        return True

    score = group_similarity(indices, token_sets)
    if strategy == "caption_similar":
        return similarity_min <= score <= similarity_max
    if strategy == "caption_dissimilar":
        return score <= dissimilarity_max
    raise ValueError(f"Unknown grouping strategy: {strategy}")


def enumerate_candidates(
    records: Sequence[Dict[str, Any]],
    audio_count: int,
    strategy: str,
    token_sets: Sequence[Set[str]],
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> List[Tuple[int, ...]]:
    candidates: List[Tuple[int, ...]] = []
    for indices in itertools.combinations(range(len(records)), audio_count):
        if candidate_allowed_by_strategy(
            indices,
            strategy,
            token_sets,
            similarity_min,
            similarity_max,
            dissimilarity_max,
        ):
            candidates.append(indices)
    return candidates


def select_units_from_candidates(
    candidates: List[Tuple[int, ...]],
    records: Sequence[Dict[str, Any]],
    unit_count: int,
    rng: random.Random,
    max_audio_reuse: Optional[int],
) -> Tuple[List[Tuple[int, ...]], Dict[str, int]]:
    rng.shuffle(candidates)
    selected: List[Tuple[int, ...]] = []
    seen: Set[Tuple[str, ...]] = set()
    reuse_counts: Counter[str] = Counter()
    stats = {
        "duplicate_unit_attempts_skipped": 0,
        "repeated_audio_in_unit_rejections": 0,
        "audio_reuse_rejections": 0,
    }

    for indices in candidates:
        if len(selected) >= unit_count:
            break

        audio_ids = [str(records[index]["audio_id"]) for index in indices]
        if len(set(audio_ids)) != len(audio_ids):
            stats["repeated_audio_in_unit_rejections"] += 1
            continue

        key = canonical_unit_key(audio_ids)
        if key in seen:
            stats["duplicate_unit_attempts_skipped"] += 1
            continue

        if max_audio_reuse is not None and any(
            reuse_counts[audio_id] >= max_audio_reuse for audio_id in audio_ids
        ):
            stats["audio_reuse_rejections"] += 1
            continue

        seen.add(key)
        for audio_id in audio_ids:
            reuse_counts[audio_id] += 1
        selected.append(indices)

    stats["unique_audio_ids_in_units"] = len(reuse_counts)
    return selected, stats


def select_units_by_random_attempts(
    records: Sequence[Dict[str, Any]],
    audio_count: int,
    unit_count: int,
    rng: random.Random,
    max_audio_reuse: Optional[int],
    max_attempts: int,
) -> Tuple[List[Tuple[int, ...]], Dict[str, int]]:
    selected: List[Tuple[int, ...]] = []
    seen: Set[Tuple[str, ...]] = set()
    reuse_counts: Counter[str] = Counter()
    stats = {
        "duplicate_unit_attempts_skipped": 0,
        "repeated_audio_in_unit_rejections": 0,
        "audio_reuse_rejections": 0,
        "random_attempts": 0,
    }

    n = len(records)
    while len(selected) < unit_count and stats["random_attempts"] < max_attempts:
        stats["random_attempts"] += 1
        indices = tuple(rng.sample(range(n), audio_count))
        audio_ids = [str(records[index]["audio_id"]) for index in indices]
        if len(set(audio_ids)) != len(audio_ids):
            stats["repeated_audio_in_unit_rejections"] += 1
            continue

        key = canonical_unit_key(audio_ids)
        if key in seen:
            stats["duplicate_unit_attempts_skipped"] += 1
            continue

        if max_audio_reuse is not None and any(
            reuse_counts[audio_id] >= max_audio_reuse for audio_id in audio_ids
        ):
            stats["audio_reuse_rejections"] += 1
            continue

        seen.add(key)
        for audio_id in audio_ids:
            reuse_counts[audio_id] += 1
        selected.append(indices)

    stats["unique_audio_ids_in_units"] = len(reuse_counts)
    return selected, stats


def create_units(
    records: Sequence[Dict[str, Any]],
    unit_count: int,
    audio_count: int,
    grouping_strategy: str,
    rng: random.Random,
    max_audio_reuse: Optional[int],
    max_enumerated_units: int,
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> Tuple[List[Tuple[int, ...]], Dict[str, int], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    n = len(records)
    if audio_count < 1:
        raise ValueError("--audio-count must be at least 1.")
    if n < audio_count:
        raise ValueError(
            f"Subset contains {n} records, fewer than --audio-count={audio_count}."
        )
    if unit_count <= 0:
        return [], {"possible_unordered_units": math.comb(n, audio_count)}, errors

    possible_units = math.comb(n, audio_count)
    requested = min(unit_count, possible_units)
    token_sets = [caption_tokens(record) for record in records]

    if possible_units <= max_enumerated_units:
        candidates = enumerate_candidates(
            records,
            audio_count,
            grouping_strategy,
            token_sets,
            similarity_min,
            similarity_max,
            dissimilarity_max,
        )
        selected, stats = select_units_from_candidates(
            candidates,
            records,
            requested,
            rng,
            max_audio_reuse,
        )
        stats["candidate_units_considered"] = len(candidates)
        stats["possible_unordered_units"] = possible_units
    elif grouping_strategy == "random":
        max_attempts = max(10_000, requested * 100)
        selected, stats = select_units_by_random_attempts(
            records,
            audio_count,
            requested,
            rng,
            max_audio_reuse,
            max_attempts,
        )
        stats["candidate_units_considered"] = stats.get("random_attempts", 0)
        stats["possible_unordered_units"] = possible_units
    else:
        raise ValueError(
            f"{grouping_strategy} needs candidate enumeration, but {possible_units} "
            f"possible units exceeds --max-enumerated-units={max_enumerated_units}. "
            "Use a smaller subset or raise the limit."
        )

    if unit_count > possible_units:
        errors.append(
            {
                "stage": "unit",
                "error_type": "unit_count_exceeds_possible",
                "requested_unit_count": unit_count,
                "possible_unordered_units": possible_units,
            }
        )

    if len(selected) < unit_count:
        errors.append(
            {
                "stage": "unit",
                "error_type": "unit_count_not_satisfied",
                "requested_unit_count": unit_count,
                "units_generated": len(selected),
                "message": "Constraints or candidate availability prevented more units.",
            }
        )

    return selected, stats, errors


def audio_path_for(record: Dict[str, Any]) -> str:
    return str(record.get("local_audio_path") or record.get("local_audio_path_abs") or "")


def make_unit_record(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    audio_ids = [str(record["audio_id"]) for record in records]
    return {
        "unit_id": unit_id_for(audio_ids),
        "schema_version": UNIT_SCHEMA_VERSION,
        "grounding_standard": GROUNDING_STANDARD,
        "audio_count": len(records),
        "audio_ids": audio_ids,
        "audio_paths": [audio_path_for(record) for record in records],
        "audio_captions": [get_list(record, "captions") for record in records],
    }


def check_duplicate_audio_ids(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(str(row.get("audio_id", "")) for row in rows)
    return [
        {
            "stage": "manifest",
            "error_type": "duplicate_audio_id",
            "audio_id": audio_id,
            "count": count,
        }
        for audio_id, count in counts.items()
        if audio_id and count > 1
    ]


def main() -> int:
    args = parse_args()
    input_manifest = Path(args.input_manifest_path)
    output_dir = Path(args.output_dir)
    ensure_pyarrow_available()
    ensure_output_dir(output_dir)

    outputs = {
        "subset_jsonl": output_dir / "subset_manifest.jsonl",
        "subset_parquet": output_dir / "subset_manifest.parquet",
        "audio_units_jsonl": output_dir / "audio_units.jsonl",
        "audio_units_parquet": output_dir / "audio_units.parquet",
        "config": output_dir / "filter_config_used.yaml",
        "stats": output_dir / "filter_stats.json",
        "errors": output_dir / "filter_errors.jsonl",
    }
    for path in outputs.values():
        assert_can_write(path, args.overwrite)

    if not input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_manifest}")

    rng = random.Random(args.random_seed)
    rows = load_manifest(input_manifest)
    validate_manifest_rows(rows)

    errors = check_duplicate_audio_ids(rows)
    eligible = filter_by_split(rows, args.splits)
    if not eligible:
        raise ValueError(f"No records found for requested splits: {args.splits}")

    subset = sample_subset(eligible, args.subset_size, rng)
    if len(subset) < args.audio_count:
        raise ValueError(
            f"Subset contains {len(subset)} records, fewer than --audio-count={args.audio_count}."
        )

    selected_unit_indices, unit_stats, unit_errors = create_units(
        records=subset,
        unit_count=args.unit_count,
        audio_count=args.audio_count,
        grouping_strategy=args.grouping_strategy,
        rng=rng,
        max_audio_reuse=args.max_audio_reuse,
        max_enumerated_units=args.max_enumerated_units,
        similarity_min=args.similarity_min,
        similarity_max=args.similarity_max,
        dissimilarity_max=args.dissimilarity_max,
    )
    errors.extend(unit_errors)

    unit_rows = [
        make_unit_record([subset[index] for index in indices])
        for indices in selected_unit_indices
    ]
    subset_rows = [lean_manifest_record(record) for record in subset]

    if args.strict and errors:
        write_jsonl(outputs["errors"], errors)
        raise RuntimeError(f"Strict mode failed with {len(errors)} recorded errors.")

    config = {
        "input_manifest_path": str(input_manifest),
        "output_dir": str(output_dir),
        "splits": args.splits,
        "subset_size": args.subset_size,
        "unit_count": args.unit_count,
        "audio_count": args.audio_count,
        "random_seed": args.random_seed,
        "grouping_strategy": args.grouping_strategy,
        "max_audio_reuse": args.max_audio_reuse,
        "max_enumerated_units": args.max_enumerated_units,
        "similarity_min": args.similarity_min,
        "similarity_max": args.similarity_max,
        "dissimilarity_max": args.dissimilarity_max,
    }
    stats = {
        "filter_timestamp": utc_now(),
        "original_records_loaded": len(rows),
        "records_after_split_filter": len(eligible),
        "subset_records_selected": len(subset_rows),
        "requested_audio_count": args.audio_count,
        "units_requested": args.unit_count,
        "units_generated": len(unit_rows),
        "random_seed": args.random_seed,
        "split_names": args.splits,
        "grouping_strategy": args.grouping_strategy,
        "max_audio_reuse": args.max_audio_reuse,
        "total_errors": len(errors),
        **unit_stats,
    }

    write_jsonl(outputs["subset_jsonl"], subset_rows)
    write_parquet(outputs["subset_parquet"], subset_rows)
    write_jsonl(outputs["audio_units_jsonl"], unit_rows)
    write_parquet(outputs["audio_units_parquet"], unit_rows)
    write_json(outputs["config"], config)
    write_json(outputs["stats"], stats)
    write_jsonl(outputs["errors"], errors)

    print(f"Wrote {len(subset_rows)} subset records to {outputs['subset_parquet']}")
    print(f"Wrote {len(unit_rows)} audio units to {outputs['audio_units_parquet']}")
    if errors:
        print(f"Recorded {len(errors)} filter warnings/errors in {outputs['errors']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
