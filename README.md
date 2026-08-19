# audio-cf

## Environment

Use Python 3.12 on Linux for the GPU serving environment. Python itself cannot
be installed from `requirements.txt`; create and manage the environment with
Conda.

Create a fresh environment and install all project dependencies with:

```bash
conda create --name audio-cf python=3.12 -y
conda activate audio-cf
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the least host-specific setup, use the official `vllm/vllm-openai` Docker
image. See the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for current wheel, driver, and container options.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Large Generation Run

The checked-in configs use a high-throughput profile: 100,000 random two-audio
units, Parquet-only data outputs, no generation audit table, and 256-unit durable
checkpoints. Gemma is the checked-in default in `data_synthetic.yaml`; select
Qwen3.8 with the explicit `--vllm-config` command below.

Agent prompts are stage-specific: ClaimAgent receives at most three rotating
captions from target-relevant sources, QAAgent receives the validated claim
instead of the raw captions, and VerifierAgent independently receives relevant
captions when enabled. File names, IDs, full schemas, and unrelated retry errors
are excluded from prompts.
Final benchmark answers contain exactly two labels: an evidence judgment and its
single determining source. Valid forms are `["supported", "AUDIO_1"]` and
`["contradicted", "AUDIO_2"]`. Supported claims require explicit caption
support; contradicted claims require positive, mutually incompatible caption
evidence from the same audio. Evidence may use one or several captions from that
audio to establish one or several related propositions about one coherent event.
Contradictions use objectively checkable changes rather than subjective manner,
intensity, emotion, or quality contrasts. Other cases and source swaps are excluded.
The default output directory is `data/synthetic/clotho_audio_units_v3`; do not
append these rows to an older checkpoint with a different answer contract.

```bash
python data_filter.py --config configs/data_filter.yaml --overwrite

# Keep vLLM compile and runtime caches off the home filesystem.
export VLLM_CACHE_ROOT="/data/scratch/yxu209/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
```

### Qwen3.8 BF16

The Qwen profile uses the local `/data/not_backed_up/yxu209/models/qwen`
Qwen3.8-27B checkpoint, the `qwen3` reasoning parser, and text-only loading.
Both server profiles keep `dtype: auto`; their BF16 checkpoint configs make
vLLM resolve this to BF16 while preserving model-metadata compatibility.
The client sends Qwen3.8's native `reasoning_effort: low`. Both model profiles
use stage-specific synthetic-data sampling: ClaimAgent uses `0.7/0.95`, QAAgent
uses `0.5/0.95`, and VerifierAgent uses greedy decoding. A 1,024-token thinking
budget leaves the remainder of each 2,048-token completion budget for the
required final JSON; the server context is 8,192.
Current vLLM Model Runner V2 does not support `thinking_token_budget`, so keep
`VLLM_USE_V2_MODEL_RUNNER=0` while this profile sends its 1,024-token cap. The
Triton GDN prefill backend avoids FlashInfer's first-run GDN JIT on H100.
The server uses xgrammar with structured output enabled during reasoning. MTP
is intentionally disabled: released vLLM builds can miss the transition from
Qwen reasoning to the schema-constrained final answer when speculative decoding
is active, producing malformed JSON prefixes. Restart the server after pulling
this configuration; changing the YAML does not affect an already running vLLM.
The Qwen client requests the reasoning field solely to recover a complete JSON
object misplaced there by this parser bug; reasoning prose is never accepted.

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_USE_V2_MODEL_RUNNER=0 \
  vllm serve --config configs/vllm_server_qwen38.yaml \
  --gdn-prefill-backend triton
python data_synthetic.py \
  --config configs/data_synthetic.yaml \
  --vllm-config configs/vllm_client_qwen38.yaml \
  --output-dir data/synthetic/clotho_audio_units_qwen38_v0 \
  --overwrite
```

### Gemma 4

Stop the Qwen server before switching because both profiles use port 8000. The
Gemma server also uses an 8,192-token context and 2,048-token agent completion
budgets. For Gemma, `reasoning_effort: low` enables thinking but does not lower
its token use relative to `medium` or `high`; `thinking_token_budget: 1024` is
the actual private-reasoning cap. Set the effort to `none` only to disable
thinking entirely.
The server pins xgrammar and disables arbitrary JSON whitespace to mitigate the
Gemma 4 constrained-decoding loop reported in vLLM issue 40080. Restart the
Gemma server after changing or pulling its server YAML.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 CUDA_VISIBLE_DEVICES=0 \
  vllm serve --config configs/vllm_server_gemma4.yaml
python data_synthetic.py \
  --config configs/data_synthetic.yaml \
  --vllm-config configs/vllm_client_gemma4.yaml \
  --output-dir data/synthetic/clotho_audio_units_gemma4_v0 \
  --overwrite
```

The client `model` must equal the server `served-model-name`. Monitor vLLM for
KV-cache preemption or OOM and lower the matching client, runner, and server
concurrency values together if needed. Set `VLLM_CACHE_ROOT` in every shell or
batch job that launches vLLM; its default is `~/.cache/vllm`.
