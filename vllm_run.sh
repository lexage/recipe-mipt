#!/usr/bin/bash
set -euo pipefail

# Default CUDA_VISIBLE_DEVICES to 0 iff it's not set at all
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ] || [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Collect args and detect if --host is already present
args=("$@")
has_host=false
for a in "${args[@]}"; do
  if [[ "$a" == "--host" ]]; then
    has_host=true; break
  fi
  if [[ "$a" == --host=* ]]; then
    has_host=true; break
  fi
done

# Determine first non-loopback IP from `hostname -I` (falls back to 0.0.0.0)
first_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "${first_ip}" ]]; then
  first_ip="0.0.0.0"
fi

# If no --host provided, append it
if ! $has_host; then
  args+=("--host" "${first_ip}")
fi

# Echo the command for visibility
echo "Running: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} vllm serve ${args[*]}"

# Run vLLM
exec vllm serve "${args[@]}"

