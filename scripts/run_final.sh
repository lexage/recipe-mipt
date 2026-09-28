#!/bin/bash
# Пять прогонов финального эксперимента, последовательно, внутри контейнера.
#
#   bash scripts/run_final.sh
#
# Карты не нужны: прогон — клиент к двум эндпоинтам (7216 эмбеддер лаборатории,
# 7217 наша модель) плюс исполнение кода задач на CPU. Поэтому запускается прямо
# на узле, без задания в очередь.
#
# Перед запуском должны быть готовы:
#   - venv (scripts/setup_runner_env.sh);
#   - модель на 7217 (sbatch -p "$LAB_LONG" scripts/svc-qwen36-7217.sbatch);
#   - корпус data/docs_database_examples.db и датасет data/ds1000/ds1000.jsonl.gz.
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/venv}"
IMG="${IMG_ASLLM:-10.0.117.197:5000/asllm:1.10.3-pytorch2.10.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.12-vllm0.20.1-py312}"
MODELS="${MODELS_ROOT:-/bmcp_lvm_fs/cusa/models}"
CONFIGS="${CONFIGS:-final_test_configs}"
WORKERS="${WORKERS:-4}"

for path in "$REPO/data/docs_database_examples.db" "$REPO/data/ds1000/ds1000.jsonl.gz"; do
    test -f "$path" || { echo "нет файла: $path" >&2; exit 1; }
done
# Проверяем файл, а не bin/python: там символическая ссылка на интерпретатор
# контейнера, и с самого узла она выглядит битой.
test -f "$VENV/bin/activate" || { echo "нет venv: $VENV" >&2; exit 1; }

for port in 7216 7217; do
    curl -sf -m 5 "http://0.0.0.0:$port/v1/models" >/dev/null \
        || { echo "порт $port не отвечает" >&2; exit 1; }
done

mkdir -p "$HOME/.cache/fastembed"

# Токенайзер солвер грузит по пути, который вернул эндпоинт, поэтому веса
# монтируются по тому же адресу и внутрь контейнера прогона.
#
# HF_ENDPOINT: образ подсовывает зеркало hf-mirror.com, на котором нет модели
# BM25 для разреженных векторов qdrant, а запасной путь через Google Storage с
# узла закрыт (400). Настоящий хаб из контейнера доступен, поэтому указываем его
# явно. Кеш fastembed уводим в дом: по умолчанию он в /tmp, а /tmp здесь в
# оперативной памяти и очищается.
exec docker run --rm --network host --ipc host \
  -v "$HOME:$HOME" -v "$MODELS:$MODELS:ro" -w "$REPO" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -e FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed" \
  "$IMG" "$VENV/bin/python" run_ds1000.py \
      -c "$CONFIGS" --log-chunks -n "$WORKERS"
