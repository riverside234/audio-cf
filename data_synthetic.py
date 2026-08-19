#!/usr/bin/env python3
"""Run caption-grounded synthetic data generation.

This file is intentionally a thin CLI entrypoint:

    audio_units.parquet/jsonl -> deterministic agents -> examples.parquet/jsonl

Agent behavior, schemas, prompt rendering, model calls, retry, and dataset IO live
under synthetic/. This script only loads config, wires dependencies, and writes
run artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from synthetic.agents import (
    ClaimAgent,
    QAAgent,
    ReasoningPolicy,
    TargetConditionSampler,
    VerifierAgent,
    build_runner,
)
from synthetic.infrastructure.dataset_io import (
    batched,
    load_rows,
    validate_audio_unit_rows,
    write_json,
    write_jsonl,
)
from synthetic.infrastructure.llm_client import (
    LLMClientConfig,
    LLMResponseError,
    VLLMClient,
    VLLMHTTPError,
)
from synthetic.infrastructure.output_writer import IncrementalOutputWriter
from synthetic.infrastructure.retry import RetryConfig
from synthetic.infrastructure.run_logger import RunLogger, utc_now


DEFAULT_CONFIG_PATH = Path("configs/data_synthetic.yaml")
DEFAULT_VLLM_CONFIG_PATH = Path("configs/vllm_client_qwen38.yaml")


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    checkpoint_dir: Path
    examples_parquet: Path
    examples_jsonl: Path
    examples_audit_parquet: Path
    sample_for_human_review_jsonl: Path
    generation_config_used: Path
    stats: Path
    errors_jsonl: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic claim/question/answer examples from audio units."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to configs/data_synthetic.yaml.",
    )
    parser.add_argument(
        "--vllm-config",
        default=None,
        help="Optional override for the Python vLLM client config.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional override for input audio_units parquet/jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for synthetic output directory.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Optional override for the first input row index to process.",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Optional limit on number of audio units to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generation outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load configs/prompts/input rows and exit before contacting vLLM.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = resolve_path(args.config, Path.cwd())
        config = load_yaml(config_path)
        config = expand_placeholders(config, Path.cwd())
        apply_cli_overrides(config, args)

        vllm_config_path = resolve_path(
            args.vllm_config or get_nested(
                config,
                ["vllm", "config_path"],
                str(DEFAULT_VLLM_CONFIG_PATH),
            ),
            Path.cwd(),
        )
        vllm_config = expand_placeholders(load_yaml(vllm_config_path), Path.cwd())

        return asyncio.run(
            run_generation(
                config=config,
                vllm_config=vllm_config,
                config_path=config_path,
                vllm_config_path=vllm_config_path,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI should report a clear failure.
        print(f"data_synthetic.py failed: {exc}", file=sys.stderr)
        return 1


async def run_generation(
    config: Mapping[str, Any],
    vllm_config: Mapping[str, Any],
    config_path: Path,
    vllm_config_path: Path,
) -> int:
    input_path = resolve_path(get_nested(config, ["input", "audio_units_path"]), Path.cwd())
    start_index = int(get_nested(config, ["input", "start_index"], 0) or 0)
    max_units = optional_int(get_nested(config, ["input", "max_units"], None))
    dry_run = bool(get_nested(config, ["run", "dry_run"], False))
    fail_fast = bool(get_nested(config, ["run", "fail_fast"], False))
    continue_on_error = bool(get_nested(config, ["run", "continue_on_error"], True))
    overwrite = bool(get_nested(config, ["output", "overwrite"], False))
    run_name = optional_str(get_nested(config, ["run", "run_name"], None))

    if start_index < 0:
        raise ValueError("input.start_index must be >= 0.")
    if max_units is not None and max_units < 1:
        raise ValueError("input.max_units must be null or >= 1.")

    output_paths = build_output_paths(config)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = expected_output_files(output_paths, config)
    assert_outputs_writable(output_files, overwrite)
    clear_existing_outputs(output_files, overwrite)
    logger = RunLogger(output_paths.output_dir, run_name=run_name)
    logger.log_event(
        "generation_started",
        {
            "config_path": str(config_path),
            "vllm_config_path": str(vllm_config_path),
            "input_path": str(input_path),
            "dry_run": dry_run,
        },
    )

    rows = load_rows(input_path)
    validate_audio_unit_rows(rows)
    selected_rows = select_rows(rows, start_index=start_index, max_units=max_units)
    if not selected_rows:
        raise ValueError("No audio-unit rows selected for generation.")

    prompt_paths = resolve_prompt_paths(vllm_config)
    check_prompt_paths(prompt_paths)
    write_config_used(
        output_paths.generation_config_used,
        config=config,
        vllm_config=vllm_config,
        config_path=config_path,
        vllm_config_path=vllm_config_path,
    )

    if dry_run:
        stats = build_stats(
            config=config,
            vllm_config=vllm_config,
            input_rows_loaded=len(rows),
            selected_units=len(selected_rows),
            examples_written=0,
            errors_written=0,
            dry_run=True,
            start_index=start_index,
            max_units=max_units,
            status="dry_run",
            completed_batches=0,
        )
        write_json(output_paths.stats, stats)
        write_jsonl(output_paths.errors_jsonl, [])
        logger.write_stats(stats)
        logger.log_event("dry_run_completed", stats)
        print(
            f"Dry run OK: loaded {len(rows)} rows, selected {len(selected_rows)} rows."
        )
        return 0

    retry_config = build_retry_config(vllm_config)
    llm_client_config = build_llm_client_config(vllm_config)
    batch_size = int(get_nested(vllm_config, ["batching", "unit_batch_size"], 32) or 32)
    if batch_size < 1:
        raise ValueError("batching.unit_batch_size must be >= 1.")

    output_writer = build_incremental_output_writer(output_paths, config)
    examples_written = 0
    errors_written = 0
    completed_batches = 0

    async with VLLMClient(llm_client_config) as llm_client:
        runner = build_configured_runner(
            llm_client=llm_client,
            retry_config=retry_config,
            vllm_config=vllm_config,
            prompt_paths=prompt_paths,
        )
        absolute_index = start_index
        for batch in batched(selected_rows, batch_size):
            logger.log_event(
                "batch_started",
                {"start_index": absolute_index, "batch_size": len(batch)},
            )
            states, batch_errors = await run_batch(
                runner=runner,
                rows=batch,
                start_index=absolute_index,
                fail_fast=fail_fast,
                continue_on_error=continue_on_error,
            )
            batch_examples: List[Dict[str, Any]] = []
            batch_audit_rows: List[Dict[str, Any]] = []
            for state in states:
                if state.final_example is not None:
                    batch_examples.append(dict(state.final_example))
                if output_writer.write_audit_enabled:
                    batch_audit_rows.append(build_audit_row(state))

            output_writer.write_batch(
                start_index=absolute_index,
                input_count=len(batch),
                examples=batch_examples,
                audit_rows=batch_audit_rows,
                error_rows=batch_errors,
            )
            examples_written += len(batch_examples)
            errors_written += len(batch_errors)
            completed_batches += 1
            progress_stats = build_stats(
                config=config,
                vllm_config=vllm_config,
                input_rows_loaded=len(rows),
                selected_units=len(selected_rows),
                examples_written=examples_written,
                errors_written=errors_written,
                dry_run=False,
                start_index=start_index,
                max_units=max_units,
                status="in_progress",
                completed_batches=completed_batches,
            )
            write_json(output_paths.stats, progress_stats)
            logger.write_stats(progress_stats)
            logger.log_event(
                "batch_completed",
                {
                    "start_index": absolute_index,
                    "batch_size": len(batch),
                    "examples_total": examples_written,
                    "errors_total": errors_written,
                    "checkpoint_dir": str(output_paths.checkpoint_dir),
                },
            )
            absolute_index += len(batch)

    output_writer.finalize()
    stats = build_stats(
        config=config,
        vllm_config=vllm_config,
        input_rows_loaded=len(rows),
        selected_units=len(selected_rows),
        examples_written=examples_written,
        errors_written=errors_written,
        dry_run=False,
        start_index=start_index,
        max_units=max_units,
        status="completed",
        completed_batches=completed_batches,
    )
    write_json(output_paths.stats, stats)
    logger.write_stats(stats)
    logger.log_event("generation_completed", stats)

    print(f"Wrote {examples_written} examples to {output_paths.examples_parquet}")
    if errors_written:
        print(f"Recorded {errors_written} generation errors in {output_paths.errors_jsonl}")
    return 0 if examples_written or not errors_written else 1


def build_configured_runner(
    llm_client: VLLMClient,
    retry_config: RetryConfig,
    vllm_config: Mapping[str, Any],
    prompt_paths: Mapping[str, Path],
) -> Any:
    agents_config = dict(get_nested(vllm_config, ["agents"], {}) or {})
    run_verifier = bool(agents_config.get("run_verifier", False))
    max_validation_attempts = int(agents_config.get("max_validation_attempts", 2) or 2)

    condition_config = dict(agents_config.get("condition_sampling") or {})
    condition_sampler = TargetConditionSampler(
        seed=int(condition_config.get("seed", 42) or 42),
        strategy=str(condition_config.get("strategy", "cycle")),
    )

    claim_settings = generation_settings(vllm_config, "claim_agent")
    qa_settings = generation_settings(vllm_config, "qa_agent")
    verifier_settings = generation_settings(vllm_config, "verifier_agent")
    reasoning_policy = build_reasoning_policy(vllm_config)

    claim_agent = ClaimAgent(
        llm_client=llm_client,
        prompt_path=prompt_paths["claim_agent"],
        retry_config=retry_config,
        temperature=claim_settings.get("temperature"),
        top_p=claim_settings.get("top_p"),
        max_tokens=claim_settings.get("max_tokens"),
        reasoning_policy=reasoning_policy,
    )
    qa_agent = QAAgent(
        llm_client=llm_client,
        prompt_path=prompt_paths["qa_agent"],
        retry_config=retry_config,
        temperature=qa_settings.get("temperature"),
        top_p=qa_settings.get("top_p"),
        max_tokens=qa_settings.get("max_tokens"),
        reasoning_policy=reasoning_policy,
    )

    verifier_agent: Optional[VerifierAgent] = None
    if run_verifier:
        verifier_agent = VerifierAgent(
            llm_client=llm_client,
            prompt_path=prompt_paths["verifier_agent"],
            retry_config=retry_config,
            temperature=verifier_settings.get("temperature"),
            top_p=verifier_settings.get("top_p"),
            max_tokens=verifier_settings.get("max_tokens"),
            reasoning_policy=reasoning_policy,
        )

    return build_runner(
        claim_agent=claim_agent,
        qa_agent=qa_agent,
        verifier_agent=verifier_agent,
        condition_sampler=condition_sampler,
        run_verifier=run_verifier,
        max_validation_attempts=max_validation_attempts,
        max_concurrency=int(
            get_nested(vllm_config, ["batching", "runner_max_concurrency"], 8) or 8
        ),
    )


async def run_batch(
    runner: Any,
    rows: Sequence[Mapping[str, Any]],
    start_index: int,
    fail_fast: bool,
    continue_on_error: bool,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    semaphore = asyncio.Semaphore(int(runner.max_concurrency))

    async def run_one(offset: int, row: Mapping[str, Any]) -> Any:
        async with semaphore:
            return await runner.run_unit(row, unit_index=start_index + offset)

    tasks = [run_one(offset, row) for offset, row in enumerate(rows)]
    if fail_fast or not continue_on_error:
        states = await asyncio.gather(*tasks)
        return list(states), []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    states: List[Any] = []
    errors: List[Dict[str, Any]] = []
    for offset, result in enumerate(results):
        unit = rows[offset]
        if isinstance(result, Exception):
            errors.append(
                build_generation_error(
                    result,
                    unit_index=start_index + offset,
                    unit_id=str(unit.get("unit_id", "")),
                )
            )
        else:
            states.append(result)
    return states, errors


def build_generation_error(
    error: Exception,
    *,
    unit_index: int,
    unit_id: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": utc_now(),
        "stage": "generation",
        "unit_index": unit_index,
        "unit_id": unit_id,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, VLLMHTTPError):
        row.update(
            {
                "http_status": error.status_code,
                "vllm_error": error.response_detail,
                "retryable": error.retryable,
                "request_summary": error.request_summary,
            }
        )
    elif isinstance(error, LLMResponseError):
        row.update(
            {
                "finish_reason": error.finish_reason,
                "completion_tokens": error.completion_tokens,
                "message_fields": error.message_fields,
                "message_field_states": error.message_field_states,
                "requested_model": error.requested_model,
                "requested_max_tokens": error.requested_max_tokens,
                "prompt_chars": error.prompt_chars,
            }
        )
    return row


def build_audit_row(state: Any) -> Dict[str, Any]:
    return {
        "example_id": state.example_id or "",
        "unit_id": state.unit_record.get("unit_id", ""),
        "audio_count": state.unit_record.get("audio_count", 0),
        "target_condition": state.target_condition,
        "claim_record": state.claim_record,
        "qa_record": state.qa_record,
        "verifier_record": state.verifier_record,
        "validation_errors": list(state.validation_errors),
        "visible_reasoning_stripped": list(state.visible_reasoning_stripped),
        "raw_claim_text": state.raw_claim_text,
        "raw_qa_text": state.raw_qa_text,
        "raw_verifier_text": state.raw_verifier_text,
    }


def build_incremental_output_writer(
    output_paths: OutputPaths,
    config: Mapping[str, Any],
) -> IncrementalOutputWriter:
    return IncrementalOutputWriter(
        checkpoint_dir=output_paths.checkpoint_dir,
        examples_parquet=output_paths.examples_parquet,
        examples_jsonl=output_paths.examples_jsonl,
        examples_audit_parquet=output_paths.examples_audit_parquet,
        sample_for_human_review_jsonl=output_paths.sample_for_human_review_jsonl,
        errors_jsonl=output_paths.errors_jsonl,
        write_parquet_enabled=bool(
            get_nested(config, ["output", "write_parquet"], True)
        ),
        write_jsonl_enabled=bool(
            get_nested(config, ["output", "write_jsonl"], True)
        ),
        write_audit_enabled=bool(
            get_nested(config, ["output", "write_audit"], True)
        ),
        review_sample_size=int(
            get_nested(config, ["output", "review_sample_size"], 100)
        ),
    )


def build_stats(
    config: Mapping[str, Any],
    vllm_config: Mapping[str, Any],
    input_rows_loaded: int,
    selected_units: int,
    examples_written: int,
    errors_written: int,
    dry_run: bool,
    start_index: int,
    max_units: Optional[int],
    status: str,
    completed_batches: int,
) -> Dict[str, Any]:
    return {
        "timestamp": utc_now(),
        "status": status,
        "dry_run": dry_run,
        "input_rows_loaded": input_rows_loaded,
        "selected_units": selected_units,
        "start_index": start_index,
        "max_units": max_units,
        "examples_written": examples_written,
        "errors_written": errors_written,
        "completed_batches": completed_batches,
        "checkpoint_dir": str(build_output_paths(config).checkpoint_dir),
        "generation_model": get_nested(vllm_config, ["client", "model"], ""),
        "run_verifier": bool(get_nested(vllm_config, ["agents", "run_verifier"], False)),
        "reasoning_enabled": bool(
            get_nested(vllm_config, ["generation", "reasoning", "enabled"], True)
        ),
        "reasoning_mode": get_nested(
            vllm_config,
            ["generation", "reasoning", "mode"],
            "private_json",
        ),
        "reasoning_effort": get_nested(
            vllm_config,
            ["generation", "reasoning", "effort"],
            "medium",
        ),
        "include_reasoning": bool(
            get_nested(
                vllm_config,
                ["generation", "reasoning", "include_reasoning"],
                False,
            )
        ),
        "strip_visible_reasoning": bool(
            get_nested(
                vllm_config,
                ["generation", "reasoning", "strip_visible_reasoning"],
                True,
            )
        ),
        "reject_visible_reasoning": bool(
            get_nested(
                vllm_config,
                ["generation", "reasoning", "reject_visible_reasoning"],
                False,
            )
        ),
        "prompt_version": get_nested(
            vllm_config,
            ["agents", "prompt_version"],
            "claim_agent_v10+qa_agent_v7+verifier_agent_v8",
        ),
        "runner_max_concurrency": get_nested(
            vllm_config,
            ["batching", "runner_max_concurrency"],
            8,
        ),
        "unit_batch_size": get_nested(vllm_config, ["batching", "unit_batch_size"], 32),
        "output_dir": get_nested(config, ["output", "output_dir"], ""),
    }


def build_llm_client_config(vllm_config: Mapping[str, Any]) -> LLMClientConfig:
    client_config = dict(get_nested(vllm_config, ["client"], {}) or {})
    default_generation = generation_settings(vllm_config, "default")
    response_format = get_nested(vllm_config, ["generation", "response_format"], None)
    extra_body = build_request_extra_body(vllm_config, client_config)

    return LLMClientConfig(
        base_url=str(client_config.get("base_url", "http://localhost:8000/v1")),
        model=str(client_config.get("model", "")),
        api_key=str(client_config.get("api_key", "EMPTY")),
        temperature=float(default_generation.get("temperature", 0.2)),
        top_p=float(default_generation.get("top_p", 0.95)),
        max_tokens=int(default_generation.get("max_tokens", 512)),
        timeout_s=float(client_config.get("timeout_s", 120)),
        client_max_inflight_requests=int(
            client_config.get("client_max_inflight_requests", 16)
        ),
        response_format=dict(response_format) if isinstance(response_format, dict) else None,
        extra_body=extra_body,
        extra_headers=dict(client_config.get("extra_headers") or {}),
        verify_ssl=bool(client_config.get("verify_ssl", True)),
    )


def build_retry_config(vllm_config: Mapping[str, Any]) -> RetryConfig:
    retry_config = dict(get_nested(vllm_config, ["retry"], {}) or {})
    return RetryConfig(
        max_attempts=int(retry_config.get("max_attempts", 3)),
        initial_delay_s=float(retry_config.get("initial_delay_s", 1.0)),
        max_delay_s=float(retry_config.get("max_delay_s", 30.0)),
        backoff_multiplier=float(retry_config.get("backoff_multiplier", 2.0)),
        jitter_s=float(retry_config.get("jitter_s", 0.25)),
    )


def generation_settings(
    vllm_config: Mapping[str, Any],
    agent_name: str,
) -> Dict[str, Any]:
    generation = dict(get_nested(vllm_config, ["generation"], {}) or {})
    defaults = dict(generation.get("default") or {})
    overrides = dict(generation.get(agent_name) or {})
    return {**defaults, **overrides}


def build_reasoning_policy(vllm_config: Mapping[str, Any]) -> ReasoningPolicy:
    generation = dict(get_nested(vllm_config, ["generation"], {}) or {})
    reasoning = dict(generation.get("reasoning") or {})
    instruction = str(reasoning.get("instruction") or "").strip()

    return ReasoningPolicy(
        enabled=bool(reasoning.get("enabled", True)),
        mode=str(reasoning.get("mode") or "private_json"),
        instruction=instruction,
        effort=str(reasoning.get("effort") or "medium"),
        strip_visible_reasoning=bool(reasoning.get("strip_visible_reasoning", True)),
        reject_visible_reasoning=bool(reasoning.get("reject_visible_reasoning", False)),
    )


def build_request_extra_body(
    vllm_config: Mapping[str, Any],
    client_config: Mapping[str, Any],
) -> Dict[str, Any]:
    extra_body = dict(client_config.get("extra_body") or {})
    generation = dict(get_nested(vllm_config, ["generation"], {}) or {})
    reasoning = dict(generation.get("reasoning") or {})

    if not reasoning:
        return extra_body

    enabled = bool(reasoning.get("enabled", True))
    mode = str(reasoning.get("mode") or "private_json").strip().lower()
    effort = str(reasoning.get("effort") or "medium").strip().lower()
    include_reasoning = bool(reasoning.get("include_reasoning", False))

    allowed_efforts = {"none", "low", "medium", "high", "xhigh"}
    if effort not in allowed_efforts:
        raise ValueError(
            "generation.reasoning.effort must be one of "
            f"{sorted(allowed_efforts)}, got {effort!r}."
        )

    if mode == "qwen3_vllm":
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", enabled)
        chat_template_kwargs.setdefault(
            "preserve_thinking",
            bool(reasoning.get("preserve_thinking", False)),
        )
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        if enabled:
            # Qwen3.8 supports low/medium/xhigh reasoning effort natively.
            extra_body.setdefault("reasoning_effort", effort)
    else:
        # Gemma 4 and generic vLLM reasoning profiles use the OpenAI-compatible
        # effort field without Qwen's chat-template controls.
        extra_body.setdefault("reasoning_effort", effort if enabled else "none")

    # vLLM enforces this sampling budget from parser-derived reasoning
    # boundaries. It is independent of how each model enables thinking.
    thinking_token_budget = reasoning.get("thinking_token_budget")
    if enabled and thinking_token_budget is not None:
        thinking_token_budget = int(thinking_token_budget)
        if thinking_token_budget < 1:
            raise ValueError(
                "generation.reasoning.thinking_token_budget must be positive."
            )
        extra_body.setdefault("thinking_token_budget", thinking_token_budget)
    extra_body.setdefault("include_reasoning", include_reasoning)
    return extra_body


def build_output_paths(config: Mapping[str, Any]) -> OutputPaths:
    output_dir = resolve_path(get_nested(config, ["output", "output_dir"]), Path.cwd())
    return OutputPaths(
        output_dir=output_dir,
        checkpoint_dir=output_dir
        / str(
            get_nested(
                config,
                ["output", "batch_checkpoint_dir"],
                "generation_batches",
            )
        ),
        examples_parquet=output_dir
        / str(get_nested(config, ["output", "examples_parquet"], "examples.parquet")),
        examples_jsonl=output_dir
        / str(get_nested(config, ["output", "examples_jsonl"], "examples.jsonl")),
        examples_audit_parquet=output_dir
        / str(
            get_nested(
                config,
                ["output", "examples_audit_parquet"],
                "examples_audit.parquet",
            )
        ),
        sample_for_human_review_jsonl=output_dir
        / str(
            get_nested(
                config,
                ["output", "sample_for_human_review_jsonl"],
                "sample_for_human_review.jsonl",
            )
        ),
        generation_config_used=output_dir
        / str(
            get_nested(
                config,
                ["output", "generation_config_used"],
                "generation_config_used.yaml",
            )
        ),
        stats=output_dir / str(get_nested(config, ["output", "stats"], "stats.json")),
        errors_jsonl=output_dir
        / str(get_nested(config, ["output", "errors_jsonl"], "errors.jsonl")),
    )


def expected_output_files(
    output_paths: OutputPaths,
    config: Mapping[str, Any],
) -> List[Path]:
    paths: List[Path] = [
        output_paths.generation_config_used,
        output_paths.stats,
        output_paths.errors_jsonl,
        output_paths.output_dir / "events.jsonl",
        output_paths.checkpoint_dir,
    ]
    if bool(get_nested(config, ["output", "write_parquet"], True)):
        paths.append(output_paths.examples_parquet)
    if bool(get_nested(config, ["output", "write_jsonl"], True)):
        paths.append(output_paths.examples_jsonl)
    if bool(get_nested(config, ["output", "write_audit"], True)):
        paths.extend(
            [
                output_paths.examples_audit_parquet,
                output_paths.sample_for_human_review_jsonl,
            ]
        )
    return paths


def assert_outputs_writable(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Generation outputs already exist. Re-run with --overwrite or set "
            f"output.overwrite: true.\n{formatted}"
        )


def clear_existing_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    if not overwrite:
        return
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def resolve_prompt_paths(vllm_config: Mapping[str, Any]) -> Dict[str, Path]:
    prompts = dict(get_nested(vllm_config, ["agents", "prompts"], {}) or {})
    return {
        "claim_agent": resolve_path(
            prompts.get("claim_agent", "prompts/synthetic/claim_agent_v10.md"),
            Path.cwd(),
        ),
        "qa_agent": resolve_path(
            prompts.get("qa_agent", "prompts/synthetic/qa_agent_v7.md"),
            Path.cwd(),
        ),
        "verifier_agent": resolve_path(
            prompts.get("verifier_agent", "prompts/synthetic/verifier_agent_v8.md"),
            Path.cwd(),
        ),
    }


def check_prompt_paths(prompt_paths: Mapping[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in prompt_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prompt files:\n" + "\n".join(missing))


def select_rows(
    rows: Sequence[Dict[str, Any]],
    start_index: int,
    max_units: Optional[int],
) -> List[Dict[str, Any]]:
    if start_index >= len(rows):
        return []
    end_index = None if max_units is None else start_index + max_units
    return list(rows[start_index:end_index])


def write_config_used(
    path: Path,
    config: Mapping[str, Any],
    vllm_config: Mapping[str, Any],
    config_path: Path,
    vllm_config_path: Path,
) -> None:
    payload = {
        "config_path": str(config_path),
        "vllm_config_path": str(vllm_config_path),
        "data_synthetic_config": config,
        "vllm_config": redact_secrets(vllm_config),
    }
    dump_yaml(path, payload)


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.input_path is not None:
        set_nested(config, ["input", "audio_units_path"], args.input_path)
    if args.output_dir is not None:
        set_nested(config, ["output", "output_dir"], args.output_dir)
    if args.start_index is not None:
        set_nested(config, ["input", "start_index"], args.start_index)
    if args.max_units is not None:
        set_nested(config, ["input", "max_units"], args.max_units)
    if args.overwrite:
        set_nested(config, ["output", "overwrite"], True)
    if args.dry_run:
        set_nested(config, ["run", "dry_run"], True)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config files. Install it with `pip install PyYAML`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return dict(payload)


def dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to write generation_config_used.yaml."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=False)


def expand_placeholders(value: Any, cwd: Path) -> Any:
    if isinstance(value, str):
        expanded = value.replace("${cwd}", str(cwd))
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [expand_placeholders(item, cwd) for item in value]
    if isinstance(value, dict):
        return {key: expand_placeholders(item, cwd) for key, item in value.items()}
    return value


def resolve_path(value: Any, cwd: Path) -> Path:
    if value is None:
        raise ValueError("Path value cannot be null.")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return cwd / path


def get_nested(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(payload: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    current = payload
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def redact_secrets(payload: Mapping[str, Any]) -> Dict[str, Any]:
    redacted = dict(payload)
    client = redacted.get("client")
    if isinstance(client, dict) and client.get("api_key") not in {None, "", "EMPTY"}:
        client = dict(client)
        client["api_key"] = "<redacted>"
        redacted["client"] = client
    return redacted


if __name__ == "__main__":
    raise SystemExit(main())
