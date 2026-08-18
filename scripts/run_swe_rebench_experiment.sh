#!/usr/bin/env bash

set -Eeuo pipefail

RECIPE_DIR="/workspace/proj/grant_swe_rebench/recipe-mipt"
SNAPSHOTS_DIR="$RECIPE_DIR/datasets/snapshots"
FORK="/external/SWE-bench-fork"

# Конфигурация агента, которую нужно использовать в экспериментах.
CONFIG_SOURCE="$RECIPE_DIR/pipeline_configs/react_sgr_swe_rebench.yaml"

DATASET_ALIAS="${1:-first10}"
EXPERIMENT_NAME="${2:-react-sgr-qwen}"

case "$DATASET_ALIAS" in
    one)
        DATASET_SOURCE="$SNAPSHOTS_DIR/task.jsonl"
        ;;
    first10)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2024-10-first-10.jsonl"
        ;;
    october)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2024-10-01.jsonl"
        ;;
    *)
        echo "Неизвестная выборка: $DATASET_ALIAS" >&2
        echo "Допустимые значения: one, first10, october" >&2
        exit 2
        ;;
esac

export RECIPE_DIR
export FORK

export EVALUATOR_COMMIT="e4907b7a90eafaa1f0a6428fd04fe31cdd8b4284"
export DATASET="nebius/SWE-rebench"
export DATASET_REVISION="89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
export SPLIT="test"

export MODEL_URL="http://localhost:11455/v1"
export LLM_MODEL="Qwen/Qwen2.5-32B-Instruct"
export MODEL_NAME_OR_PATH="react-sgr/${LLM_MODEL}"

export RUN_ID="${EXPERIMENT_NAME}-${DATASET_ALIAS}-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"

export TASK_FILE="$RUN_DIR/task.jsonl"
export INSTANCE_IDS_FILE="$RUN_DIR/instance_ids.txt"
export CONFIG="$RUN_DIR/pipeline.yaml"

if [[ ! -f "$DATASET_SOURCE" ]]; then
    echo "Датасет не найден: $DATASET_SOURCE" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_SOURCE" ]]; then
    echo "Конфигурация не найдена: $CONFIG_SOURCE" >&2
    exit 1
fi

if [[ ! -d "$FORK" ]]; then
    echo "SWE-bench fork не найден: $FORK" >&2
    exit 1
fi

cd "$RECIPE_DIR"

mkdir -p "$RUN_DIR/logs"

# Сохраняем копии входных файлов конкретного эксперимента.
cp "$DATASET_SOURCE" "$TASK_FILE"
cp "$CONFIG_SOURCE" "$CONFIG"

METADATA_SOURCE="${DATASET_SOURCE}.metadata.json"

if [[ -f "$METADATA_SOURCE" ]]; then
    cp "$METADATA_SOURCE" "$RUN_DIR/task.jsonl.metadata.json"
fi

# Формируем instance_ids.txt из уже подготовленного JSONL.
python - "$TASK_FILE" "$INSTANCE_IDS_FILE" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])

instance_ids = []

with dataset_path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue

        task = json.loads(line)
        instance_id = task.get("instance_id")

        if not instance_id:
            raise ValueError(
                f"В строке {line_number} отсутствует instance_id"
            )

        instance_ids.append(instance_id)

if not instance_ids:
    raise ValueError("Выбранный датасет не содержит задач")

output_path.write_text(
    "\n".join(instance_ids) + "\n",
    encoding="utf-8",
)

print(f"Количество задач: {len(instance_ids)}")
PY

echo
echo "Параметры эксперимента:"
printf '%-24s %s\n' \
    "DATASET_ALIAS" "$DATASET_ALIAS" \
    "DATASET_SOURCE" "$DATASET_SOURCE" \
    "RUN_ID" "$RUN_ID" \
    "RUN_DIR" "$RUN_DIR" \
    "CONFIG" "$CONFIG" \
    "MODEL_URL" "$MODEL_URL" \
    "LLM_MODEL" "$LLM_MODEL"

echo
echo "Проверка API модели..."

curl --fail --silent --show-error \
    "$MODEL_URL/models" >/dev/null

echo "API модели доступен."
echo "Запуск инференса..."

python run_swe_rebench.py \
    --config "$CONFIG" \
    --dataset "$TASK_FILE" \
    --split "$SPLIT" \
    --instance-ids-file "$INSTANCE_IDS_FILE" \
    --output "$RUN_DIR/predictions.jsonl" \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --run-id "$RUN_ID" \
    --num-workers 1 \
    --pull-policy missing \
    --task-timeout 3600 \
    --memory-limit 16g \
    --nano-cpus 4000000000 \
    --logs-path "$RUN_DIR/logs" \
    --evaluator-fork-path "$FORK"

echo
echo "Инференс завершён:"
echo "  predictions: $RUN_DIR/predictions.jsonl"
echo "  errors:      $RUN_DIR/predictions.errors.jsonl"
echo "  logs:        $RUN_DIR/logs"