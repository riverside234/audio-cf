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
units, Parquet-only data outputs, no generation audit table, 32 concurrent agent
jobs, medium Gemma reasoning, and 256-unit durable checkpoints. This profile
targets one full H100 80GB; a half-H100/MIG slice cannot hold the 62.6GB BF16
Gemma 4 31B checkpoint plus runtime memory.

```bash
python data_filter.py --config configs/data_filter.yaml --overwrite
CUDA_VISIBLE_DEVICES=0 vllm serve --config configs/vllm_server.yaml
python data_synthetic.py --config configs/data_synthetic.yaml --overwrite
```

If the lab CUDA/CCCL mismatch triggers the FlashInfer sampler compile error,
start vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`. Monitor vLLM for KV-cache
preemption or OOM; reduce `runner_max_concurrency` and `max-num-seqs` together
from `32` to `16` if either occurs. For two full H100s, expose both GPUs and
change `tensor-parallel-size` to `2` before benchmarking.
