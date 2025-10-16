#!/bin/bash

ENV_FILE="./.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "Environment variables loaded from $ENV_FILE"
else
    echo "Error: $ENV_FILE not found."
    exit 1
fi

python3 -m vllm.entrypoints.openai.api_server \
    --model $LLM \
    --download-dir $LLM_PATH \
    --max-model-len 40000 \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.5 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
