# audio-cf

## Environment

Use Python 3.12 on Linux for the GPU serving environment. Python itself cannot
be installed from `requirements.txt`; `.python-version` lets `uv` or `pyenv`
select the project interpreter.

Create a fresh environment and install all project dependencies with:

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=auto -r requirements.txt
```

Install the NVIDIA display/compute driver on the host and verify it with
`nvidia-smi`. Do not manually install the individual `nvidia-*` wheels from an
old environment: they are CUDA runtime libraries rather than the host driver,
and vLLM/PyTorch will resolve one compatible CUDA family. A full CUDA Toolkit is
only needed when building CUDA code or vLLM from source.

For the least host-specific setup, use the official `vllm/vllm-openai` Docker
image. See the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
for current wheel, driver, and container options.

## Tests

```bash
python -m unittest discover -s tests -v
```
