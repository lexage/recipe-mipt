# vLLM Experiments

A repository for experimenting with and benchmarking [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference and serving engine for LLMs.

## Installation

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Starting vLLM Server

Use the wrapper script for convenient server startup:

```bash
./vllm_run.sh meta-llama/Llama-3.2-1B-Instruct
```

The script automatically:
- Sets CUDA device to GPU 0 (override with `CUDA_VISIBLE_DEVICES`)
- Binds to your machine's IP address for network access
- Passes all arguments to `vllm serve`

**Examples:**

```bash
# Basic usage
./vllm_run.sh meta-llama/Llama-3.2-1B-Instruct

# Custom CUDA device
CUDA_VISIBLE_DEVICES=1 ./vllm_run.sh meta-llama/Llama-3.2-1B-Instruct

# With additional vLLM arguments
./vllm_run.sh meta-llama/Llama-3.2-1B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.9

# Custom host
./vllm_run.sh meta-llama/Llama-3.2-1B-Instruct --host 127.0.0.1
```
