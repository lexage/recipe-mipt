#!/bin/bash
#SBATCH --job-name=svc-embed-7216
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Квота ассоциации — 120 ГБ RAM на все задания пользователя сразу, а --mem
# здесь считается от памяти узла. Эмбеддеру 4B столько не нужно, и запас
# оставляем солверу, иначе его задание висит с AssocGrpMemLimit.
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/vizilter/logs/%x-%j.log
set -eu

# Эмбеддер финального эксперимента. Раньше на 7216 висел постоянный сервис
# лаборатории; его сняли, поэтому поднимаем те же веса сами и под тем же именем
# в API — конфиги эксперимента из-за этого менять не нужно.
#
#   sbatch -p "$LAB_LONG" scripts/svc-embed-7216.sbatch

MODEL_DIR="${EMBED_MODEL_DIR:-/home/trofimov/models/Qwen3-Embedding-4B}"
API_NAME="${EMBED_API_NAME:-Qwen/Qwen3-Embedding-4B}"
PORT=7216
IMG="${IMG_ASLLM:-10.0.117.197:5000/asllm:1.10.3-pytorch2.10.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.12-vllm0.20.1-py312}"

# Доп. флаги движку, без правки файла: EMBED_EXTRA_ARGS="--enforce-eager".
# На этих картах захват графов ускорителя при старте может падать с
# "CUDA error: invalid argument" (profile_cudagraph_memory); --enforce-eager
# его отключает, для эмбеддингов цена этого невелика.
EXTRA_ARGS="${EMBED_EXTRA_ARGS:-}"

CARDS_LIST="${CUDA_VISIBLE_DEVICES:-0}"
NCARDS=$(set -- ${CARDS_LIST//,/ }; echo $#)

docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"

NAME="svc-embed-${PORT}-${SLURM_JOB_ID:-manual}"
docker rm -f "$NAME" >/dev/null 2>&1 || true

DEVS=""
for d in /dev/alixpu*; do DEVS="$DEVS --device $d"; done

echo "=== $(date '+%F %T') $MODEL_DIR как $API_NAME на $NCARDS карт(ах), порт $PORT ==="

exec docker run --rm --name "$NAME" --network host --ipc host $DEVS \
  -v "$MODEL_DIR:/models/Qwen3-Embedding-4B:ro" \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model "/models/Qwen3-Embedding-4B" --served-model-name "$API_NAME" \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$NCARDS" \
    --trust-remote-code --gpu-memory-utilization 0.8 \
    $EXTRA_ARGS
