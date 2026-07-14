#!/usr/bin/env python3
"""Create reproducible Clotho subsets and unique two-audio pairs.

This script is the second deterministic stage of the dataset pipeline:

    full_manifest.parquet -> subset_manifest.parquet + pairs.parquet

It intentionally does not parse raw Zenodo CSV files. Run data_process.py first.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


GROUNDING_STANDARD = "caption_grounded"
PAIR_SCHEMA_VERSION = "pair_manifest_v0"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a Clotho manifest and create unique two-audio pairs."
    )
    parser.add_argument(
        "--input-manifest-path",
        default=os.path.join(os.getcwd(), "data", "log", "full_manifest.parquet"),
        help="Path to full_manifest.parquet or full_manifest.jsonl from data_process.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "data", "log"),
        help="Directory where subset and pair files are written.",
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
        default=500,
        help="Number of audio records to sample. Use 0 to keep all eligible records.",
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=2000,
        help="Number of unique two-audio pairs to create.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible subsetting and pairing.",
    )
    parser.add_argument(
        "--pairing-strategy",
        choices=["random", "caption_similar", "caption_dissimilar"],
        default="random",
        help="Pairing strategy. Similar/dissimilar use simple lexical caption overlap.",
    )
    parser.add_argument(
        "--max-audio-reuse",
        type=int,
        default=None,
        help="Maximum times an audio_id may appear across pairs. Default is unlimited.",
    )
    parser.add_argument(
        "--max-enumerated-pairs",
        type=int,
        default=2_000_000,
        help="Maximum candidate pairs to enumerate before falling back to random attempts.",
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
        help="Fail if requested pair_count cannot be satisfied.",
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
        "original_file_name",
        "local_audio_path",
        "captions",
        "caption_summary",
        "license_url",
        "sound_link",
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


def pair_key(audio_id_1: str, audio_id_2: str) -> Tuple[str, str]:
    return tuple(sorted((audio_id_1, audio_id_2)))


def pair_id_for(audio_id_1: str, audio_id_2: str) -> str:
    key = "::".join(pair_key(audio_id_1, audio_id_2))
    import hashlib

    return f"pair_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


def caption_tokens(record: Dict[str, Any]) -> Set[str]:
    text = record.get("caption_summary") or " ".join(record.get("captions") or [])
    return set(TOKEN_RE.findall(str(text).lower()))


def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(union))


def candidate_allowed_by_strategy(
    i: int,
    j: int,
    strategy: str,
    token_sets: Sequence[Set[str]],
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> bool:
    if strategy == "random":
        return True

    score = jaccard_similarity(token_sets[i], token_sets[j])
    if strategy == "caption_similar":
        return similarity_min <= score <= similarity_max
    if strategy == "caption_dissimilar":
        return score <= dissimilarity_max
    raise ValueError(f"Unknown pairing strategy: {strategy}")


def enumerate_candidates(
    records: Sequence[Dict[str, Any]],
    strategy: str,
    token_sets: Sequence[Set[str]],
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    for i, j in itertools.combinations(range(len(records)), 2):
        if candidate_allowed_by_strategy(
            i,
            j,
            strategy,
            token_sets,
            similarity_min,
            similarity_max,
            dissimilarity_max,
        ):
            candidates.append((i, j))
    return candidates


def select_pairs_from_candidates(
    candidates: List[Tuple[int, int]],
    records: Sequence[Dict[str, Any]],
    pair_count: int,
    rng: random.Random,
    max_audio_reuse: Optional[int],
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    rng.shuffle(candidates)
    selected: List[Tuple[int, int]] = []
    seen: Set[Tuple[str, str]] = set()
    reuse_counts: Counter[str] = Counter()
    stats = {
        "duplicate_pair_attempts_skipped": 0,
        "self_pair_rejections": 0,
        "audio_reuse_rejections": 0,
    }

    for i, j in candidates:
        if len(selected) >= pair_count:
            break

        id_i = str(records[i]["audio_id"])
        id_j = str(records[j]["audio_id"])
        if id_i == id_j:
            stats["self_pair_rejections"] += 1
            continue

        key = pair_key(id_i, id_j)
        if key in seen:
            stats["duplicate_pair_attempts_skipped"] += 1
            continue

        if max_audio_reuse is not None:
            if reuse_counts[id_i] >= max_audio_reuse or reuse_counts[id_j] >= max_audio_reuse:
                stats["audio_reuse_rejections"] += 1
                continue

        seen.add(key)
        reuse_counts[id_i] += 1
        reuse_counts[id_j] += 1
        selected.append((i, j))

    stats["unique_audio_ids_in_pairs"] = len(reuse_counts)
    return selected, stats


def select_pairs_by_random_attempts(
    records: Sequence[Dict[str, Any]],
    pair_count: int,
    rng: random.Random,
    max_audio_reuse: Optional[int],
    max_attempts: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    selected: List[Tuple[int, int]] = []
    seen: Set[Tuple[str, str]] = set()
    reuse_counts: Counter[str] = Counter()
    stats = {
        "duplicate_pair_attempts_skipped": 0,
        "self_pair_rejections": 0,
        "audio_reuse_rejections": 0,
        "random_attempts": 0,
    }

    n = len(records)
    while len(selected) < pair_count and stats["random_attempts"] < max_attempts:
        stats["random_attempts"] += 1
        i, j = rng.sample(range(n), 2)
        id_i = str(records[i]["audio_id"])
        id_j = str(records[j]["audio_id"])
        if id_i == id_j:
            stats["self_pair_rejections"] += 1
            continue

        key = pair_key(id_i, id_j)
        if key in seen:
            stats["duplicate_pair_attempts_skipped"] += 1
            continue

        if max_audio_reuse is not None:
            if reuse_counts[id_i] >= max_audio_reuse or reuse_counts[id_j] >= max_audio_reuse:
                stats["audio_reuse_rejections"] += 1
                continue

        seen.add(key)
        reuse_counts[id_i] += 1
        reuse_counts[id_j] += 1
        selected.append((i, j))

    stats["unique_audio_ids_in_pairs"] = len(reuse_counts)
    return selected, stats


def get_list(record: Dict[str, Any], key: str) -> List[str]:
    value = record.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def audio_prefix(record: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_id": record.get("audio_id", ""),
        f"{prefix}_split": record.get("split", ""),
        f"{prefix}_file_name": record.get("original_file_name", ""),
        f"{prefix}_path": record.get("local_audio_path", ""),
        f"{prefix}_path_abs": record.get("local_audio_path_abs", ""),
        f"{prefix}_duration_seconds": record.get("duration_seconds"),
        f"{prefix}_captions": get_list(record, "captions"),
        f"{prefix}_caption_summary": record.get("caption_summary", ""),
        f"{prefix}_keywords": get_list(record, "keywords"),
        f"{prefix}_sound_id": record.get("sound_id", ""),
        f"{prefix}_sound_link": record.get("sound_link", ""),
        f"{prefix}_start_end_samples": record.get("start_end_samples", ""),
        f"{prefix}_manufacturer": record.get("manufacturer", ""),
        f"{prefix}_license_url": record.get("license_url", ""),
    }


def make_pair_record(
    pair_index: int,
    record_1: Dict[str, Any],
    record_2: Dict[str, Any],
    pairing_strategy: str,
    random_seed: int,
) -> Dict[str, Any]:
    id_1 = str(record_1["audio_id"])
    id_2 = str(record_2["audio_id"])
    record = {
        "pair_id": pair_id_for(id_1, id_2),
        "pair_index": pair_index,
        "schema_version": PAIR_SCHEMA_VERSION,
        "source_dataset": record_1.get("source_dataset", "clotho"),
        "source_record_id": record_1.get("source_record_id", ""),
        "source_record_url": record_1.get("source_record_url", ""),
        "source_version": record_1.get("source_version", ""),
        "grounding_standard": GROUNDING_STANDARD,
        "pairing_strategy": pairing_strategy,
        "random_seed": random_seed,
        "canonical_pair_key": "::".join(pair_key(id_1, id_2)),
    }
    record.update(audio_prefix(record_1, "audio_1"))
    record.update(audio_prefix(record_2, "audio_2"))
    return record


def create_pairs(
    records: Sequence[Dict[str, Any]],
    pair_count: int,
    pairing_strategy: str,
    rng: random.Random,
    max_audio_reuse: Optional[int],
    max_enumerated_pairs: int,
    similarity_min: float,
    similarity_max: float,
    dissimilarity_max: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, int], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    n = len(records)
    if n < 2:
        raise ValueError("At least two records are required to create pairs.")
    if pair_count <= 0:
        return [], {}, errors

    possible_pairs = n * (n - 1) // 2
    requested = min(pair_count, possible_pairs)
    token_sets = [caption_tokens(record) for record in records]

    if possible_pairs <= max_enumerated_pairs:
        candidates = enumerate_candidates(
            records,
            pairing_strategy,
            token_sets,
            similarity_min,
            similarity_max,
            dissimilarity_max,
        )
        selected, stats = select_pairs_from_candidates(
            candidates,
            records,
            requested,
            rng,
            max_audio_reuse,
        )
        stats["candidate_pairs_considered"] = len(candidates)
        stats["possible_unordered_pairs"] = possible_pairs
    elif pairing_strategy == "random":
        max_attempts = max(10_000, requested * 100)
        selected, stats = select_pairs_by_random_attempts(
            records,
            requested,
            rng,
            max_audio_reuse,
            max_attempts,
        )
        stats["candidate_pairs_considered"] = stats.get("random_attempts", 0)
        stats["possible_unordered_pairs"] = possible_pairs
    else:
        raise ValueError(
            f"{pairing_strategy} needs candidate enumeration, but {possible_pairs} "
            f"possible pairs exceeds --max-enumerated-pairs={max_enumerated_pairs}. "
            "Use a smaller subset or raise the limit."
        )

    if pair_count > possible_pairs:
        errors.append(
            {
                "stage": "pair",
                "error_type": "pair_count_exceeds_possible",
                "requested_pair_count": pair_count,
                "possible_unordered_pairs": possible_pairs,
            }
        )

    if len(selected) < pair_count:
        errors.append(
            {
                "stage": "pair",
                "error_type": "pair_count_not_satisfied",
                "requested_pair_count": pair_count,
                "pairs_generated": len(selected),
                "message": "Constraints or candidate availability prevented more pairs.",
            }
        )

    return selected, stats, errors


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
        "pairs_jsonl": output_dir / "pairs.jsonl",
        "pairs_parquet": output_dir / "pairs.parquet",
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
    if len(subset) < 2:
        raise ValueError("Subset contains fewer than two records.")

    selected_pair_indices, pair_stats, pair_errors = create_pairs(
        records=subset,
        pair_count=args.pair_count,
        pairing_strategy=args.pairing_strategy,
        rng=rng,
        max_audio_reuse=args.max_audio_reuse,
        max_enumerated_pairs=args.max_enumerated_pairs,
        similarity_min=args.similarity_min,
        similarity_max=args.similarity_max,
        dissimilarity_max=args.dissimilarity_max,
    )
    errors.extend(pair_errors)

    pair_rows = [
        make_pair_record(
            pair_index=index,
            record_1=subset[i],
            record_2=subset[j],
            pairing_strategy=args.pairing_strategy,
            random_seed=args.random_seed,
        )
        for index, (i, j) in enumerate(selected_pair_indices)
    ]

    if args.strict and errors:
        write_jsonl(outputs["errors"], errors)
        raise RuntimeError(f"Strict mode failed with {len(errors)} recorded errors.")

    config = {
        "input_manifest_path": str(input_manifest),
        "output_dir": str(output_dir),
        "splits": args.splits,
        "subset_size": args.subset_size,
        "pair_count": args.pair_count,
        "random_seed": args.random_seed,
        "pairing_strategy": args.pairing_strategy,
        "max_audio_reuse": args.max_audio_reuse,
        "max_enumerated_pairs": args.max_enumerated_pairs,
        "similarity_min": args.similarity_min,
        "similarity_max": args.similarity_max,
        "dissimilarity_max": args.dissimilarity_max,
    }
    stats = {
        "filter_timestamp": utc_now(),
        "original_records_loaded": len(rows),
        "records_after_split_filter": len(eligible),
        "subset_records_selected": len(subset),
        "pairs_requested": args.pair_count,
        "pairs_generated": len(pair_rows),
        "random_seed": args.random_seed,
        "split_names": args.splits,
        "pairing_strategy": args.pairing_strategy,
        "max_audio_reuse": args.max_audio_reuse,
        "total_errors": len(errors),
        **pair_stats,
    }

    write_jsonl(outputs["subset_jsonl"], subset)
    write_parquet(outputs["subset_parquet"], subset)
    write_jsonl(outputs["pairs_jsonl"], pair_rows)
    write_parquet(outputs["pairs_parquet"], pair_rows)
    write_json(outputs["config"], config)
    write_json(outputs["stats"], stats)
    write_jsonl(outputs["errors"], errors)

    print(f"Wrote {len(subset)} subset records to {outputs['subset_parquet']}")
    print(f"Wrote {len(pair_rows)} pairs to {outputs['pairs_parquet']}")
    if errors:
        print(f"Recorded {len(errors)} filter warnings/errors in {outputs['errors']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
