python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-FP8 \
    --download-dir /path/to/model \
    --max-model-len 40000 \
    --host 0.0.0.0 \
    --port 11455 \
    --gpu-memory-utilization 0.5 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
