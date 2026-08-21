#!/usr/bin/env bash

set -Eeuo pipefail

EXPERIMENT_STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EXPERIMENT_STARTED_EPOCH="$(date +%s)"

RECIPE_DIR="/workspace/proj/grant_swe_rebench/recipe-mipt"
SNAPSHOTS_DIR="$RECIPE_DIR/datasets/snapshots"
FORK="/external/SWE-bench-fork"
CONFIG_SOURCE="$RECIPE_DIR/pipeline_configs/react_sgr_swe_rebench.yaml"

DATASET_ALIAS="${1:-first10}"
EXPERIMENT_NAME="${2:-react-sgr-qwen}"

case "$DATASET_ALIAS" in
    one)
        DATASET_SOURCE="$SNAPSHOTS_DIR/task.jsonl"
        EVAL_MAX_WORKERS=1
        ;;
    first10)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2024-10-first-10.jsonl"
        EVAL_MAX_WORKERS=2
        ;;
    first100)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2024-10-first-100.jsonl"
        EVAL_MAX_WORKERS=8
        ;;
    october)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2024-10-01.jsonl"
        EVAL_MAX_WORKERS=8
        ;;
    two)
        DATASET_SOURCE="$SNAPSHOTS_DIR/tasks-from-2025-04-01-first-2.jsonl"
        EVAL_MAX_WORKERS=8
        ;;
    *)
        echo "Неизвестная выборка: $DATASET_ALIAS" >&2
        echo "Допустимые значения: one, first10, first100, october, two" >&2
        exit 2
        ;;
esac

export RECIPE_DIR
export FORK
export EVALUATOR_COMMIT="e4907b7a90eafaa1f0a6428fd04fe31cdd8b4284"
export DATASET="nebius/SWE-rebench"
export DATASET_REVISION="89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
export SPLIT="test"
export EVAL_NAMESPACE="swerebench"

export MODEL_URL="http://127.0.0.1:11455/v1"
export LLM_MODEL="Qwen/Qwen3.8-27B-FP8"
export MODEL_NAME_OR_PATH="react-sgr/${LLM_MODEL}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

export RUN_ID="${EXPERIMENT_NAME}-${DATASET_ALIAS}-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"
export TASK_FILE="$RUN_DIR/task.jsonl"
export INSTANCE_IDS_FILE="$RUN_DIR/instance_ids.txt"
export CONFIG="$RUN_DIR/pipeline.yaml"

mkdir -p "$RUN_DIR/logs"

TIMINGS_FILE="$RUN_DIR/timings.json"
TIMINGS_LOG="$RUN_DIR/timings.log"
RUN_STATUS="preparing"
ACTIVE_STAGE=""
ACTIVE_STAGE_STARTED_EPOCH=""
EXPERIMENT_FINISHED_AT_UTC=""
TOTAL_DURATION_SECONDS=""
INFERENCE_STARTED_AT_UTC=""
INFERENCE_FINISHED_AT_UTC=""
INFERENCE_DURATION_SECONDS=""
INFERENCE_EXIT_CODE=""
EVALUATION_STARTED_AT_UTC=""
EVALUATION_FINISHED_AT_UTC=""
EVALUATION_DURATION_SECONDS=""
EVALUATION_CHECK_EXIT_CODE=""
EVALUATION_EXIT_CODE=""

format_duration() {
    local total_seconds="$1"
    printf '%02d:%02d:%02d' \
        "$((total_seconds / 3600))" \
        "$(((total_seconds % 3600) / 60))" \
        "$((total_seconds % 60))"
}

log_timing() {
    printf '%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$*" | tee -a "$TIMINGS_LOG"
}

write_timings() {
    python - \
        "$TIMINGS_FILE" \
        "$RUN_ID" \
        "$DATASET_ALIAS" \
        "$RUN_STATUS" \
        "$EXPERIMENT_STARTED_AT_UTC" \
        "$EXPERIMENT_FINISHED_AT_UTC" \
        "$TOTAL_DURATION_SECONDS" \
        "$INFERENCE_STARTED_AT_UTC" \
        "$INFERENCE_FINISHED_AT_UTC" \
        "$INFERENCE_DURATION_SECONDS" \
        "$INFERENCE_EXIT_CODE" \
        "$EVALUATION_STARTED_AT_UTC" \
        "$EVALUATION_FINISHED_AT_UTC" \
        "$EVALUATION_DURATION_SECONDS" \
        "$EVALUATION_CHECK_EXIT_CODE" \
        "$EVALUATION_EXIT_CODE" <<'PY'
import json
import sys
from pathlib import Path

(
    timings_file,
    run_id,
    dataset_alias,
    status,
    started_at,
    finished_at,
    total_duration,
    inference_started_at,
    inference_finished_at,
    inference_duration,
    inference_exit_code,
    evaluation_started_at,
    evaluation_finished_at,
    evaluation_duration,
    evaluation_check_exit_code,
    evaluation_exit_code,
) = sys.argv[1:]


def optional_string(value):
    return value or None


def optional_int(value):
    return int(value) if value else None


data = {
    "schema_version": 1,
    "run_id": run_id,
    "dataset_alias": dataset_alias,
    "status": status,
    "started_at_utc": started_at,
    "finished_at_utc": optional_string(finished_at),
    "total_duration_seconds": optional_int(total_duration),
    "inference": {
        "started_at_utc": optional_string(inference_started_at),
        "finished_at_utc": optional_string(inference_finished_at),
        "duration_seconds": optional_int(inference_duration),
        "exit_code": optional_int(inference_exit_code),
    },
    "evaluation": {
        "started_at_utc": optional_string(evaluation_started_at),
        "finished_at_utc": optional_string(evaluation_finished_at),
        "duration_seconds": optional_int(evaluation_duration),
        "check_exit_code": optional_int(evaluation_check_exit_code),
        "exit_code": optional_int(evaluation_exit_code),
    },
}

path = Path(timings_file)
temporary_path = path.with_suffix(".json.tmp")
temporary_path.write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
temporary_path.replace(path)
PY
}

finalize_timings() {
    local script_exit_code=$?
    local finished_epoch

    trap - EXIT
    set +e
    finished_epoch="$(date +%s)"

    if [[ "$ACTIVE_STAGE" == "inference" ]]; then
        INFERENCE_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        INFERENCE_DURATION_SECONDS=$((finished_epoch - ACTIVE_STAGE_STARTED_EPOCH))
        INFERENCE_EXIT_CODE="${INFERENCE_EXIT_CODE:-$script_exit_code}"
    elif [[ "$ACTIVE_STAGE" == "evaluation" ]]; then
        EVALUATION_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        EVALUATION_DURATION_SECONDS=$((finished_epoch - ACTIVE_STAGE_STARTED_EPOCH))
        if [[ "${EVALUATION_CHECK_EXIT_CODE:-0}" == "0" ]]; then
            EVALUATION_EXIT_CODE="${EVALUATION_EXIT_CODE:-$script_exit_code}"
        fi
    fi

    EXPERIMENT_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    TOTAL_DURATION_SECONDS=$((finished_epoch - EXPERIMENT_STARTED_EPOCH))

    if (( script_exit_code == 0 )); then
        if [[ -n "$INFERENCE_EXIT_CODE" ]] && (( INFERENCE_EXIT_CODE != 0 )); then
            RUN_STATUS="completed_with_inference_errors"
        else
            RUN_STATUS="completed"
        fi
    else
        RUN_STATUS="failed"
    fi

    write_timings
    log_timing \
        "run_finished status=$RUN_STATUS exit_code=$script_exit_code total_seconds=$TOTAL_DURATION_SECONDS"

    echo
    echo "Общее время эксперимента: $(format_duration "$TOTAL_DURATION_SECONDS")"
    echo "Время сохранено:"
    echo "  $TIMINGS_FILE"
    echo "  $TIMINGS_LOG"

    exit "$script_exit_code"
}

trap finalize_timings EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_timings
log_timing "run_started run_id=$RUN_ID"

if [[ ! -f "$DATASET_SOURCE" ]]; then
    echo "Датасет не найден: $DATASET_SOURCE" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_SOURCE" ]]; then
    echo "Конфигурация не найдена: $CONFIG_SOURCE" >&2
    exit 1
fi

if [[ ! -d "$FORK/.git" ]]; then
    echo "SWE-bench fork не найден: $FORK" >&2
    exit 1
fi

ACTUAL_EVALUATOR_COMMIT="$(git -C "$FORK" rev-parse HEAD)"
if [[ "$ACTUAL_EVALUATOR_COMMIT" != "$EVALUATOR_COMMIT" ]]; then
    echo "Неверный commit evaluator." >&2
    echo "Ожидается: $EVALUATOR_COMMIT" >&2
    echo "Получен:   $ACTUAL_EVALUATOR_COMMIT" >&2
    exit 1
fi

cd "$RECIPE_DIR"

cp "$DATASET_SOURCE" "$TASK_FILE"
cp "$CONFIG_SOURCE" "$CONFIG"

METADATA_SOURCE="${DATASET_SOURCE}.metadata.json"
if [[ -f "$METADATA_SOURCE" ]]; then
    cp "$METADATA_SOURCE" "$RUN_DIR/task.jsonl.metadata.json"
fi

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
            raise ValueError(f"В строке {line_number} отсутствует instance_id")
        instance_ids.append(instance_id)

if not instance_ids:
    raise ValueError("Выбранный датасет не содержит задач")

output_path.write_text("\n".join(instance_ids) + "\n", encoding="utf-8")
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
    "LLM_MODEL" "$LLM_MODEL" \
    "EVALUATOR_COMMIT" "$EVALUATOR_COMMIT"

echo
echo "Проверка API модели..."

python - <<PY
from openai import OpenAI

expected_model = "${LLM_MODEL}"
client = OpenAI(base_url="${MODEL_URL}", api_key="vllm")
models = {model.id for model in client.models.list()}
if expected_model not in models:
    raise RuntimeError(
        f"Модель {expected_model!r} отсутствует. Доступные модели: {sorted(models)}"
    )
print(f"API доступен, модель найдена: {expected_model}")
PY

echo
echo "Запуск инференса..."

RUN_STATUS="inference_running"
ACTIVE_STAGE="inference"
ACTIVE_STAGE_STARTED_EPOCH="$(date +%s)"
INFERENCE_STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
INFERENCE_EXIT_CODE=""
write_timings
log_timing "inference_started"

if python run_swe_rebench.py \
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
    --evaluator-fork-path "$FORK"; then
    INFERENCE_EXIT_CODE=0
else
    INFERENCE_EXIT_CODE=$?
fi

INFERENCE_FINISHED_EPOCH="$(date +%s)"
INFERENCE_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
INFERENCE_DURATION_SECONDS=$((INFERENCE_FINISHED_EPOCH - ACTIVE_STAGE_STARTED_EPOCH))
ACTIVE_STAGE=""
RUN_STATUS="inference_finished"
write_timings
log_timing \
    "inference_finished exit_code=$INFERENCE_EXIT_CODE duration_seconds=$INFERENCE_DURATION_SECONDS"
echo "Время инференса: $(format_duration "$INFERENCE_DURATION_SECONDS")"

if (( INFERENCE_EXIT_CODE != 0 )); then
    echo >&2
    echo "Предупреждение: инференс завершился с кодом $INFERENCE_EXIT_CODE." >&2
    echo "Оценка будет продолжена, если predictions.jsonl и run_metadata.json доступны." >&2
fi

if [[ ! -s "$RUN_DIR/predictions.jsonl" ]]; then
    echo "Инференс не создал predictions.jsonl" >&2
    exit 1
fi

if [[ ! -s "$RUN_DIR/run_metadata.json" ]]; then
    echo "Инференс не создал run_metadata.json" >&2
    exit 1
fi

mapfile -t INSTANCE_IDS < "$INSTANCE_IDS_FILE"

RUN_STATUS="evaluation_running"
ACTIVE_STAGE="evaluation"
ACTIVE_STAGE_STARTED_EPOCH="$(date +%s)"
EVALUATION_STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EVALUATION_CHECK_EXIT_CODE=""
EVALUATION_EXIT_CODE=""
write_timings
log_timing "evaluation_started"

echo
echo "Проверка совместимости predictions и evaluator..."

if python evaluate_swe_rebench.py \
    --fork-path "$FORK" \
    --predictions-path "$RUN_DIR/predictions.jsonl" \
    --dataset-name "$TASK_FILE" \
    --split "$SPLIT" \
    --run-id "$RUN_ID" \
    --max-workers "$EVAL_MAX_WORKERS" \
    --timeout 1800 \
    --namespace "$EVAL_NAMESPACE" \
    --inference-metadata "$RUN_DIR/run_metadata.json" \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --report-dir "$RUN_DIR/evaluation" \
    --check-only; then
    EVALUATION_CHECK_EXIT_CODE=0
else
    EVALUATION_CHECK_EXIT_CODE=$?
fi

if (( EVALUATION_CHECK_EXIT_CODE != 0 )); then
    exit "$EVALUATION_CHECK_EXIT_CODE"
fi

echo
echo "Запуск оценки..."

if python evaluate_swe_rebench.py \
    --fork-path "$FORK" \
    --predictions-path "$RUN_DIR/predictions.jsonl" \
    --dataset-name "$TASK_FILE" \
    --split "$SPLIT" \
    --run-id "$RUN_ID" \
    --max-workers "$EVAL_MAX_WORKERS" \
    --timeout 1800 \
    --namespace "$EVAL_NAMESPACE" \
    --inference-metadata "$RUN_DIR/run_metadata.json" \
    --instance-ids "${INSTANCE_IDS[@]}" \
    --report-dir "$RUN_DIR/evaluation"; then
    EVALUATION_EXIT_CODE=0
else
    EVALUATION_EXIT_CODE=$?
fi

EVALUATION_FINISHED_EPOCH="$(date +%s)"
EVALUATION_FINISHED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EVALUATION_DURATION_SECONDS=$((EVALUATION_FINISHED_EPOCH - ACTIVE_STAGE_STARTED_EPOCH))
ACTIVE_STAGE=""
RUN_STATUS="evaluation_finished"
write_timings
log_timing \
    "evaluation_finished exit_code=$EVALUATION_EXIT_CODE duration_seconds=$EVALUATION_DURATION_SECONDS"
echo "Время оценки: $(format_duration "$EVALUATION_DURATION_SECONDS")"

if (( EVALUATION_EXIT_CODE != 0 )); then
    exit "$EVALUATION_EXIT_CODE"
fi

echo
echo "Эксперимент завершён:"
echo "  predictions: $RUN_DIR/predictions.jsonl"
echo "  errors:      $RUN_DIR/errors.jsonl"
echo "  logs:        $RUN_DIR/logs"
echo "  evaluation:  $RUN_DIR/evaluation"
echo "  timings:     $TIMINGS_FILE"
echo "  timing log:  $TIMINGS_LOG"