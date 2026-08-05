"""Infrastructure adapters for synthetic-data generation."""

from .dataset_io import (
    REQUIRED_AUDIO_UNIT_COLUMNS,
    batched,
    load_rows,
    validate_audio_unit_rows,
    write_json,
    write_jsonl,
    write_parquet,
)
from .llm_client import LLMClientConfig, VLLMClient
from .output_writer import IncrementalOutputWriter
from .prompt_loader import PromptTemplate, load_prompt, render_prompt_file
from .retry import RetryConfig, retry_async, retry_sync
from .run_logger import RunLogger
from .schema_io import (
    SchemaValidationError,
    load_json_schema,
    parse_json_object,
    strip_visible_reasoning,
    validate_json,
)

__all__ = [
    "LLMClientConfig",
    "IncrementalOutputWriter",
    "PromptTemplate",
    "REQUIRED_AUDIO_UNIT_COLUMNS",
    "RetryConfig",
    "RunLogger",
    "SchemaValidationError",
    "VLLMClient",
    "batched",
    "load_json_schema",
    "load_prompt",
    "load_rows",
    "parse_json_object",
    "render_prompt_file",
    "retry_async",
    "retry_sync",
    "strip_visible_reasoning",
    "validate_audio_unit_rows",
    "validate_json",
    "write_json",
    "write_jsonl",
    "write_parquet",
]
