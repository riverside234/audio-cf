"""Run logging utilities for synthetic generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLogger:
    """Append-only JSONL logger plus small stats/error helpers."""

    def __init__(self, output_dir: Path, run_name: Optional[str] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or utc_now().replace(":", "")
        self.events_path = self.output_dir / "events.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.stats_path = self.output_dir / "stats.json"

    def log_event(self, event_type: str, payload: Optional[Mapping[str, Any]] = None) -> None:
        row = {
            "timestamp": utc_now(),
            "run_name": self.run_name,
            "event_type": event_type,
            "payload": dict(payload or {}),
        }
        self._append_jsonl(self.events_path, row)

    def log_error(
        self,
        stage: str,
        error: BaseException,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        row = {
            "timestamp": utc_now(),
            "run_name": self.run_name,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "payload": dict(payload or {}),
        }
        self._append_jsonl(self.errors_path, row)

    def write_stats(self, stats: Mapping[str, Any]) -> None:
        payload: Dict[str, Any] = {
            "timestamp": utc_now(),
            "run_name": self.run_name,
            **dict(stats),
        }
        with self.stats_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")

    @staticmethod
    def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")

