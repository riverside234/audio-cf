"""Export canonical Audio-CF Parquet rows as MS-Swift audio conversations."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

SYSTEM = (
    "Judge the claim using the recordings and identify the single determining source. "
    'Return only ["supported", "AUDIO_k"] or ["contradicted", "AUDIO_k"], '
    "replacing k with the source number."
)


def conversation(row, audio_root):
    count = int(row["audio_count"])
    answer = row["answer"]
    sources = {f"AUDIO_{i}" for i in range(1, count + 1)}
    if (count < 1 or len(answer) != 2
            or answer[0] not in {"supported", "contradicted"} or answer[1] not in sources):
        raise ValueError("Expected one judgment and one valid AUDIO_k source.")
    if (row.get("claim_status", answer[0]).lower() != answer[0]
            or row.get("evidence_sources", [answer[1]]) != [answer[1]]):
        raise ValueError("The answer disagrees with claim_status or evidence_sources.")

    local = row.get("local_audio_paths") or [None] * count
    release = row.get("audio_file_names") or [None] * count
    if len(local) != count or len(release) != count:
        raise ValueError("Audio paths must match audio_count and preserve source order.")
    audios = []
    for i, choices in enumerate(zip(local, release), 1):
        candidates = [audio_root / path for path in choices if path]
        path = next((path.resolve() for path in candidates if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"AUDIO_{i}: no recording found among {candidates}")
        audios.append(str(path))

    recordings = "\n".join(f"AUDIO_{i}: <audio>" for i in range(1, count + 1))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"{recordings}\n\nClaim: {row['claim_text']}\nQuestion: {row['question']}"
            )},
            {"role": "assistant", "content": json.dumps(answer)},
        ],
        "audios": audios,
    }


def export(input_path, output_path, audio_root, limit=None):
    import pyarrow.parquet as pq

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for batch in pq.ParquetFile(input_path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    handle.write(json.dumps(conversation(row, audio_root), ensure_ascii=False) + "\n")
                    count += 1
                    if limit is not None and count >= limit:
                        break
                if limit is not None and count >= limit:
                    break
        if count == 0:
            raise ValueError("No training examples were found.")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/distillation/train.jsonl"))
    parser.add_argument("--audio-root", type=Path, default=Path.cwd(), help="Base directory for relative audio paths.")
    parser.add_argument("--limit", type=int, help="Export only the first N examples for a smoke test.")
    args = parser.parse_args()
    count = export(args.input, args.output, args.audio_root.resolve(), args.limit)
    print(f"Exported {count} examples to {args.output}")


if __name__ == "__main__":
    main()
