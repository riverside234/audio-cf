"""Prompt-template loading and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PromptTemplate:
    path: Path
    text: str

    def render(self, values: Mapping[str, Any]) -> str:
        return self.text.format_map(_StrictFormatMap(values))


class _StrictFormatMap(dict):
    def __init__(self, values: Mapping[str, Any]):
        super().__init__((key, str(value)) for key, value in values.items())

    def __missing__(self, key: str) -> str:
        raise KeyError(f"Missing prompt template value: {key}")


def load_prompt(path: Path) -> PromptTemplate:
    with path.open("r", encoding="utf-8") as handle:
        return PromptTemplate(path=path, text=handle.read())


def render_prompt_file(path: Path, values: Mapping[str, Any]) -> str:
    return load_prompt(path).render(values)

