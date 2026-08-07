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
checkpoints. Qwen3.6 is the active default generator in `data_synthetic.yaml`.

```bash
python data_filter.py --config configs/data_filter.yaml --overwrite

# Keep vLLM compile and runtime caches off the home filesystem.
export VLLM_CACHE_ROOT="/data/scratch/yxu209/.cache/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
```

### Qwen3.6 FP8

The Qwen profile uses the local `/data/not_backed_up/yxu209/models/qwen`
checkpoint, the `qwen3` reasoning parser, text-only loading, and Qwen's MTP head.
Its 1,024-token thinking budget leaves the remainder of each 2,048-token
completion budget for the required final JSON.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve --config configs/vllm_server_qwen36.yaml
python data_synthetic.py \
  --config configs/data_synthetic.yaml \
  --vllm-config configs/vllm_client_qwen36.yaml \
  --output-dir data/synthetic/clotho_audio_units_qwen36_v0 \
  --overwrite
```

### Gemma 4

Stop the Qwen server before switching because both profiles use port 8000.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve --config configs/vllm_server_gemma4.yaml
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
