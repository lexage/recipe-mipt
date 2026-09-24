#!/bin/bash
#SBATCH --job-name=svc-qwen36-7217
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# 88 ГБ + 24 ГБ эмбеддера укладываются в квоту 120 ГБ на пользователя.
#SBATCH --mem=88G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/vizilter/logs/%x-%j.log
set -eu

# Солвер и генератор правил финального эксперимента: одна модель, один порт.
# Подаётся в длинную очередь, имя на svc- — сторож простоя такие не трогает:
#   sbatch -p "$LAB_LONG" scripts/svc-qwen36-7217.sbatch
# Эмбеддер здесь не поднимается: берём постоянный сервис лаборатории на 7216.

MODEL_DIR="${MODELS_ROOT:-/bmcp_lvm_fs/cusa/models}/Qwen3.6-35B-A3B"
PORT=7217
# Доп. флаги движку, без правки файла:
#   EXTRA_ARGS="--enforce-eager" sbatch -p "$LAB_LONG" ...
# --enforce-eager отключает захват графов ускорителя. На этих картах он
# умеет залипать (в скрипте тулкита для sglang графы отключены штатно),
# ценой примерно 10-20% скорости генерации.
EXTRA_ARGS="${EXTRA_ARGS:-}"
IMG="${IMG_ASLLM:-10.0.117.197:5000/asllm:1.10.3-pytorch2.10.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.12-vllm0.20.1-py312}"

# Имя модели в API — её же путь, и внутри контейнера он смонтирован по тому же
# адресу. Солвер берёт идентификатор модели с эндпоинта и этим же
# идентификатором грузит токенайзер, поэтому путь должен быть настоящим.
API_NAME="$MODEL_DIR"

# Число карт — из того, что выдал планировщик, а не руками: при смене --gres
# иначе забывается --tensor-parallel-size.
CARDS_LIST="${CUDA_VISIBLE_DEVICES:-0}"
NCARDS=$(set -- ${CARDS_LIST//,/ }; echo $#)

podman image exists "$IMG" || podman pull "$IMG"

NAME="svc-qwen36-${PORT}-${SLURM_JOB_ID:-manual}"
podman rm -f "$NAME" >/dev/null 2>&1 || true

# Все узлы alixpu*: лишние закрыты правами задания, внутрь попадут только
# выданные карты. Без управляющих узлов драйвер в контейнере не встанет.
DEVS=""
for d in /dev/alixpu*; do DEVS="$DEVS --device $d"; done

echo "=== $(date '+%F %T') $MODEL_DIR на $NCARDS карт(ах), порт $PORT ==="
echo "=== первый запуск может занять до часа из-за JIT-компиляции ==="

exec podman run --rm --name "$NAME" --network host --ipc host $DEVS \
  -v "$MODEL_DIR:$MODEL_DIR:ro" \
  "$IMG" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$API_NAME" \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$NCARDS" \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    $EXTRA_ARGS
