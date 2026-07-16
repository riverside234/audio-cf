#!/usr/bin/env python3
"""Process local Zenodo Clotho files into a normalized manifest.

This script is the first deterministic stage of the dataset pipeline:

    Zenodo raw files -> full_manifest.parquet/jsonl + full_manifest_provenance.parquet

It expects the Clotho files from Zenodo record 3490684, including the
development, validation, and evaluation splits. Paths are configurable so
the same script can be used on another machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os

SOURCE_DATASET = "clotho"
SOURCE_RECORD_ID = "3490684"
SOURCE_RECORD_URL = "https://zenodo.org/records/3490684"
SOURCE_VERSION = "v1.0"

SPLIT_FILES = {
    "development": {
        "archives": ["clotho_audio_development.7z"],
        "captions": "clotho_captions_development.csv",
        "metadata": "clotho_metadata_development.csv",
        "audio_dir": "development",
    },
    "validation": {
        "archives": ["clotho_audio_validation.7z"],
        "captions": "clotho_captions_validation.csv",
        "metadata": "clotho_metadata_validation.csv",
        "audio_dir": "validation",
    },
    "evaluation": {
        "archives": [
            "clotho_audio_evaluation.7z",
            "clotho_audio_evalution.7z",
        ],
        "captions": "clotho_captions_evaluation.csv",
        "metadata": "clotho_metadata_evaluation.csv",
        "audio_dir": "evaluation",
    },
}

CAPTION_COLUMNS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]
METADATA_COLUMNS = [
    "file_name",
    "keywords",
    "sound_id",
    "sound_link",
    "start_end_samples",
    "manufacturer",
    "license",
]

CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized Clotho manifest from local Zenodo files."
    )
    parser.add_argument(
        "--clotho-root",
        default=os.path.join(os.getcwd(), "data"),
        help="Root directory containing Zenodo Clotho files or subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "data", "log"),
        help="Directory where manifest and process logs are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["development", "validation", "evaluation"],
        choices=sorted(SPLIT_FILES),
        help="Clotho splits to process.",
    )
    parser.add_argument(
        "--extract-archives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract 7z audio archives when audio files are not already present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Store audio paths relative to cwd or as absolute paths.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any caption row lacks metadata or audio.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def as_posix(path: Path) -> str:
    return path.as_posix()


def display_path(path: Path, mode: str, base: Optional[Path] = None) -> str:
    resolved = path.resolve()
    if mode == "absolute":
        return as_posix(resolved)
    base_path = (base or Path.cwd()).resolve()
    try:
        return as_posix(resolved.relative_to(base_path))
    except ValueError:
        return as_posix(resolved)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def assert_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Re-run with --overwrite to replace outputs."
        )


def find_file(root: Path, preferred: Path, filename: str) -> Optional[Path]:
    if preferred.exists():
        return preferred
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def find_first_file(root: Path, preferred_dir: Path, filenames: Sequence[str]) -> Optional[Path]:
    for filename in filenames:
        found = find_file(root, preferred_dir / filename, filename)
        if found is not None:
            return found
    return None


def split_dirs(root: Path) -> Dict[str, Path]:
    return {
        "archives": root / "archives",
        "captions": root / "captions",
        "metadata": root / "metadata",
        "audio": root / "audio",
    }


def audio_files_present(audio_dir: Path) -> bool:
    return audio_dir.exists() and any(audio_dir.rglob("*.wav"))


def extract_with_py7zr(archive_path: Path, output_dir: Path) -> bool:
    try:
        import py7zr  # type: ignore
    except ImportError:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=output_dir)
    return True


def find_7z_executable() -> Optional[str]:
    for name in ["7z", "7zz", "7za", "7zr"]:
        found = shutil.which(name)
        if found:
            return found
    return None


def extract_with_executable(archive_path: Path, output_dir: Path) -> bool:
    executable = find_7z_executable()
    if not executable:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [executable, "x", str(archive_path), f"-o{output_dir}", "-y"]
    subprocess.run(command, check=True)
    return True


def maybe_extract_archive(
    archive_path: Optional[Path],
    audio_dir: Path,
    split: str,
    extract_archives: bool,
    errors: List[Dict[str, Any]],
    expected_archives: Optional[Sequence[str]] = None,
) -> None:
    if audio_files_present(audio_dir):
        return

    if not extract_archives:
        errors.append(
            {
                "stage": "extract",
                "split": split,
                "error_type": "audio_missing",
                "message": f"No wav files found in {audio_dir}; extraction disabled.",
            }
        )
        return

    if archive_path is None or not archive_path.exists():
        errors.append(
            {
                "stage": "extract",
                "split": split,
                "error_type": "archive_missing",
                "expected_archives": list(expected_archives or []),
                "message": f"Archive not found for split {split}.",
            }
        )
        return

    try:
        if extract_with_py7zr(archive_path, audio_dir):
            return
        if extract_with_executable(archive_path, audio_dir):
            return
        errors.append(
            {
                "stage": "extract",
                "split": split,
                "error_type": "extractor_missing",
                "archive_path": str(archive_path),
                "message": "Install py7zr or make a 7z executable available on PATH.",
            }
        )
    except Exception as exc:  # noqa: BLE001 - record extraction diagnostics.
        errors.append(
            {
                "stage": "extract",
                "split": split,
                "error_type": "extract_failed",
                "archive_path": str(archive_path),
                "message": str(exc),
            }
        )


def clean_csv_row(row: Dict[Optional[str], Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, list):
            cleaned[key] = " ".join(str(item) for item in value).strip()
        else:
            cleaned[key] = str(value).strip()
    return cleaned


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    last_decode_error: Optional[UnicodeDecodeError] = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return [clean_csv_row(row) for row in reader]
        except UnicodeDecodeError as exc:
            last_decode_error = exc

    if last_decode_error is not None:
        raise UnicodeDecodeError(
            last_decode_error.encoding,
            last_decode_error.object,
            last_decode_error.start,
            last_decode_error.end,
            f"Could not decode {path} using {CSV_ENCODINGS}: {last_decode_error.reason}",
        ) from last_decode_error
    raise RuntimeError(f"Could not read CSV file: {path}")


def require_columns(
    rows: Sequence[Dict[str, str]],
    required: Sequence[str],
    path: Path,
) -> None:
    if not rows:
        raise ValueError(f"{path} is empty.")
    columns = set(rows[0].keys())
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def captions_from_row(row: Dict[str, str]) -> List[str]:
    captions = []
    for column in CAPTION_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            captions.append(value)
    return captions


def split_keywords(value: str) -> List[str]:
    if not value:
        return []
    separator = ";" if ";" in value else ","
    return [part.strip() for part in value.split(separator) if part.strip()]


def caption_summary(captions: Sequence[str]) -> str:
    return " ".join(captions)


def build_audio_index(audio_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not audio_dir.exists():
        return index
    for path in sorted(audio_dir.rglob("*.wav")):
        index.setdefault(path.name, path)
    return index


def wav_duration_seconds(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return round(frames / float(rate), 6)
    except Exception:
        return None


def normalize_record(
    split: str,
    caption_row: Dict[str, str],
    metadata_row: Dict[str, str],
    audio_path: Path,
    path_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    file_name = caption_row["file_name"].strip()
    captions = captions_from_row(caption_row)
    metadata = {
        "keywords": metadata_row.get("keywords", ""),
        "sound_id": metadata_row.get("sound_id", ""),
        "sound_link": metadata_row.get("sound_link", ""),
        "start_end_samples": metadata_row.get("start_end_samples", ""),
        "manufacturer": metadata_row.get("manufacturer", ""),
        "license": metadata_row.get("license", ""),
    }
    audio_id = f"clotho_{split}_{stable_hash(file_name)}"
    manifest_record = {
        "audio_id": audio_id,
        "split": split,
        "original_file_name": file_name,
        "local_audio_path": display_path(audio_path, path_mode),
        "duration_seconds": wav_duration_seconds(audio_path),
        "captions": captions,
        "caption_summary": caption_summary(captions),
        "keywords": split_keywords(metadata["keywords"]),
        "sound_id": metadata["sound_id"],
    }
    provenance_record = {
        "audio_id": audio_id,
        "source_dataset": SOURCE_DATASET,
        "source_record_id": SOURCE_RECORD_ID,
        "source_record_url": SOURCE_RECORD_URL,
        "source_version": SOURCE_VERSION,
        "split": split,
        "original_file_name": file_name,
        "local_audio_path_abs": display_path(audio_path, "absolute"),
        "sound_id": metadata["sound_id"],
        "sound_link": metadata["sound_link"],
        "freesound_url": metadata["sound_link"],
        "start_end_samples": metadata["start_end_samples"],
        "manufacturer": metadata["manufacturer"],
        "freesound_uploader": metadata["manufacturer"],
        "license_url": metadata["license"],
        "metadata": metadata,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }
    return manifest_record, provenance_record


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
            "pyarrow is required because full_manifest.parquet is the canonical "
            "output. Install it with `pip install pyarrow`."
        ) from exc


def process_split(
    root: Path,
    split: str,
    extract_archives: bool,
    path_mode: str,
    errors: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    dirs = split_dirs(root)
    spec = SPLIT_FILES[split]

    archive_path = find_first_file(root, dirs["archives"], spec["archives"])
    captions_path = find_file(root, dirs["captions"] / spec["captions"], spec["captions"])
    metadata_path = find_file(root, dirs["metadata"] / spec["metadata"], spec["metadata"])
    audio_dir = dirs["audio"] / spec["audio_dir"]

    if captions_path is None:
        raise FileNotFoundError(f"Caption CSV not found for split {split}.")
    if metadata_path is None:
        raise FileNotFoundError(f"Metadata CSV not found for split {split}.")

    maybe_extract_archive(
        archive_path,
        audio_dir,
        split,
        extract_archives,
        errors,
        expected_archives=spec["archives"],
    )

    caption_rows = read_csv_dicts(captions_path)
    metadata_rows = read_csv_dicts(metadata_path)
    require_columns(caption_rows, ["file_name", *CAPTION_COLUMNS], captions_path)
    require_columns(metadata_rows, METADATA_COLUMNS, metadata_path)

    metadata_by_file = {row["file_name"].strip(): row for row in metadata_rows}
    audio_by_file = build_audio_index(audio_dir)
    if not audio_by_file:
        audio_by_file = build_audio_index(root)

    records: List[Dict[str, Any]] = []
    provenance_records: List[Dict[str, Any]] = []
    missing_audio = 0
    missing_metadata = 0

    for row in caption_rows:
        file_name = row["file_name"].strip()
        if not file_name:
            errors.append(
                {
                    "stage": "normalize",
                    "split": split,
                    "error_type": "missing_file_name",
                    "message": "Caption row has an empty file_name.",
                }
            )
            continue

        metadata = metadata_by_file.get(file_name)
        if metadata is None:
            missing_metadata += 1
            errors.append(
                {
                    "stage": "join",
                    "split": split,
                    "error_type": "metadata_missing",
                    "file_name": file_name,
                }
            )
            continue

        audio_path = audio_by_file.get(file_name)
        if audio_path is None:
            missing_audio += 1
            errors.append(
                {
                    "stage": "join",
                    "split": split,
                    "error_type": "audio_missing",
                    "file_name": file_name,
                }
            )
            continue

        record, provenance = normalize_record(split, row, metadata, audio_path, path_mode)
        records.append(record)
        provenance_records.append(provenance)

    stats = {
        "caption_rows_loaded": len(caption_rows),
        "metadata_rows_loaded": len(metadata_rows),
        "audio_files_found": len(audio_by_file),
        "normalized_records_written": len(records),
        "provenance_records_written": len(provenance_records),
        "missing_audio_files": missing_audio,
        "missing_metadata_rows": missing_metadata,
        "missing_caption_rows": max(0, len(metadata_rows) - len(caption_rows)),
    }
    return records, provenance_records, stats


def main() -> int:
    args = parse_args()
    root = Path(args.clotho_root)
    output_dir = Path(args.output_dir)
    ensure_pyarrow_available()
    ensure_output_dir(output_dir)

    outputs = {
        "jsonl": output_dir / "full_manifest.jsonl",
        "parquet": output_dir / "full_manifest.parquet",
        "provenance_parquet": output_dir / "full_manifest_provenance.parquet",
        "config": output_dir / "process_config_used.yaml",
        "stats": output_dir / "process_stats.json",
        "errors": output_dir / "process_errors.jsonl",
    }
    for path in outputs.values():
        assert_can_write(path, args.overwrite)

    all_records: List[Dict[str, Any]] = []
    all_provenance_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    split_stats: Dict[str, Dict[str, int]] = {}

    for split in args.splits:
        records, provenance_records, stats = process_split(
            root=root,
            split=split,
            extract_archives=args.extract_archives,
            path_mode=args.path_mode,
            errors=errors,
        )
        all_records.extend(records)
        all_provenance_records.extend(provenance_records)
        split_stats[split] = stats

    if args.strict and errors:
        write_jsonl(outputs["errors"], errors)
        raise RuntimeError(f"Strict mode failed with {len(errors)} recorded errors.")

    config = {
        "source_type": "zenodo_local",
        "zenodo_record_id": SOURCE_RECORD_ID,
        "source_record_url": SOURCE_RECORD_URL,
        "clotho_root": str(root),
        "splits": args.splits,
        "extract_archives": args.extract_archives,
        "output_dir": str(output_dir),
        "path_mode": args.path_mode,
        "strict": args.strict,
    }
    stats = {
        "source_dataset": SOURCE_DATASET,
        "source_record_id": SOURCE_RECORD_ID,
        "source_record_url": SOURCE_RECORD_URL,
        "source_version": SOURCE_VERSION,
        "split_names": args.splits,
        "processing_timestamp": utc_now(),
        "total_records_written": len(all_records),
        "total_provenance_records_written": len(all_provenance_records),
        "total_errors": len(errors),
        "split_stats": split_stats,
    }

    write_jsonl(outputs["jsonl"], all_records)
    write_parquet(outputs["parquet"], all_records)
    write_parquet(outputs["provenance_parquet"], all_provenance_records)
    write_json(outputs["config"], config)
    write_json(outputs["stats"], stats)
    write_jsonl(outputs["errors"], errors)

    print(f"Wrote {len(all_records)} records to {outputs['parquet']}")
    print(
        f"Wrote {len(all_provenance_records)} provenance records "
        f"to {outputs['provenance_parquet']}"
    )
    if errors:
        print(f"Recorded {len(errors)} process errors in {outputs['errors']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
