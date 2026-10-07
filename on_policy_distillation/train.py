"""Run MS-Swift OPD-RL with a frozen remote Qwen3-Omni teacher."""

import os
import sys

TEACHER = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
STUDENT = "Qwen/Qwen2.5-Omni-3B"


def shared_response_ids(student, teacher):
    teacher_vocab = teacher.get_vocab()
    shared = {index for token, index in student.get_vocab().items() if teacher_vocab.get(token) == index}
    if not set(range(student.vocab_size)).issubset(shared) or student.eos_token_id not in shared:
        raise ValueError("Teacher and student must share all base text token IDs and the response EOS token.")
    return shared


def guard_teacher_requests(build_requests, shared):
    def guarded(samples, template=None):
        for sample in samples:
            if not sample.response_token_ids:
                raise ValueError("The student rollout must retain response token IDs for on-policy scoring.")
            incompatible = set(sample.response_token_ids) - shared
            if incompatible:
                raise ValueError(f"The student emitted tokens with different teacher meanings: {sorted(incompatible)}")
        return build_requests(samples, template)
    return guarded


def main(argv=None):
    os.environ.setdefault("USE_HF", "1")
    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    from transformers import AutoTokenizer
    from swift.arguments import RLHFArguments
    from swift.pipelines import rlhf_main
    from swift.rlhf_trainers import grpo_trainer
    from swift.utils import parse_args

    defaults = {
        "rlhf_type": "grpo", "model": STUDENT,
        "teacher_model_server": os.environ.get("TEACHER_URL", "http://localhost:8001"),
        "dataset": "data/distillation/train.jsonl", "output_dir": "data/distillation/checkpoints",
        "tuner_type": "lora", "target_modules": "all-linear", "lora_rank": 8, "lora_alpha": 16,
        "freeze_vit": True, "freeze_aligner": True, "torch_dtype": "bfloat16", "attn_impl": "sdpa",
        "use_vllm": False, "num_generations": 1, "num_iterations": 1, "beta": 0, "teacher_kl_coef": 1,
        "num_train_epochs": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5, "max_length": 8192, "max_completion_length": 128,
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "gradient_checkpointing": True,
        "split_dataset_ratio": 0, "eval_strategy": "no", "logging_steps": 1,
        "save_steps": 50, "save_total_limit": 2, "log_completions": True, "report_to": "none",
    }
    flags = [
        part for key, value in defaults.items()
        for part in (f"--{key}", str(value).lower() if isinstance(value, bool) else str(value))
    ]
    args, remaining = parse_args(RLHFArguments, flags + (sys.argv[1:] if argv is None else argv))
    if remaining:
        raise ValueError(f"Unknown MS-Swift arguments: {remaining}")
    if args.rlhf_type != "grpo" or not args.teacher_model_server:
        raise ValueError("This launcher requires OPD-RL with an external teacher_model_server.")

    student = AutoTokenizer.from_pretrained(args.model)
    teacher = AutoTokenizer.from_pretrained(os.environ.get("TEACHER_MODEL", TEACHER))
    shared = shared_response_ids(student, teacher)
    # The models have different audio tokens. Score only identically mapped response
    # tokens; Swift independently encodes each model's audio prompt at the server.
    original = grpo_trainer.build_teacher_requests
    grpo_trainer.build_teacher_requests = guard_teacher_requests(original, shared)
    try:
        rlhf_main(args)
    finally:
        grpo_trainer.build_teacher_requests = original


if __name__ == "__main__":
    main()
