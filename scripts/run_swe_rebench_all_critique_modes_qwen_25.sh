#!/usr/bin/env bash

set -Eeuo pipefail

RECIPE_DIR="${RECIPE_DIR:-/workspace/proj/grant_swe_rebench/recipe-mipt}"
RUNNER="$RECIPE_DIR/scripts/run_swe_rebench_experiment_qwen_25_critic.sh"

if (( $# > 9 )); then
    echo "Использование: $0 [dataset_alias] [8 конфигов в заданном порядке]" >&2
    exit 2
fi

DATASET_ALIAS="${1:-first10}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-react-sgr-qwen2-5}"

resolve_config_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$RECIPE_DIR" "$path"
    fi
}

SELF_REFINE_TOOL_CONFIG="$(resolve_config_path "${2:-pipeline_configs/react_sgr_swe_self_refine_tool.yaml}")"
DECRIM_TOOL_CONFIG="$(resolve_config_path "${3:-pipeline_configs/react_sgr_swe_decrim_tool.yaml}")"
REFLEXION_TOOL_CONFIG="$(resolve_config_path "${4:-pipeline_configs/react_sgr_swe_reflexion_tool.yaml}")"
CRITIC_TOOL_CONFIG="$(resolve_config_path "${5:-pipeline_configs/react_sgr_swe_critic_tool.yaml}")"

SELF_REFINE_PIPELINE_CONFIG="$(resolve_config_path "${6:-pipeline_configs/react_sgr_swe_self_refine_pipeline.yaml}")"
DECRIM_PIPELINE_CONFIG="$(resolve_config_path "${7:-pipeline_configs/react_sgr_swe_decrim_pipeline.yaml}")"
REFLEXION_PIPELINE_CONFIG="$(resolve_config_path "${8:-pipeline_configs/react_sgr_swe_reflexion_pipeline.yaml}")"
CRITIC_PIPELINE_CONFIG="$(resolve_config_path "${9:-pipeline_configs/react_sgr_swe_critic_pipeline.yaml}")"

if [[ ! -f "$RUNNER" ]]; then
    echo "Сценарий запуска эксперимента не найден: $RUNNER" >&2
    exit 1
fi

CONFIGS=(
    "$SELF_REFINE_TOOL_CONFIG"
    "$DECRIM_TOOL_CONFIG"
    "$REFLEXION_TOOL_CONFIG"
    "$CRITIC_TOOL_CONFIG"
    "$SELF_REFINE_PIPELINE_CONFIG"
    "$DECRIM_PIPELINE_CONFIG"
    "$REFLEXION_PIPELINE_CONFIG"
    "$CRITIC_PIPELINE_CONFIG"
)

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo "Конфигурация не найдена: $config" >&2
        exit 1
    fi
done

run_experiment() {
    local mode="$1"
    local method="$2"
    local config="$3"
    local experiment_name="${EXPERIMENT_PREFIX}-${method}-${mode}"

    echo
    echo "============================================================"
    echo "Запуск: mode=$mode method=$method dataset=$DATASET_ALIAS"
    echo "Конфигурация: $config"
    echo "Имя эксперимента: $experiment_name"
    echo "============================================================"

    bash "$RUNNER" \
        "$DATASET_ALIAS" \
        "$experiment_name" \
        "$config"
}

echo "Сначала запускаются все методы в режиме инструмента, затем в pipeline."
echo "Порядок методов: self-refine, decrim, reflexion, code (critic)."

run_experiment "tool" "self-refine" "$SELF_REFINE_TOOL_CONFIG"
run_experiment "tool" "decrim" "$DECRIM_TOOL_CONFIG"
run_experiment "tool" "reflexion" "$REFLEXION_TOOL_CONFIG"
run_experiment "tool" "critic" "$CRITIC_TOOL_CONFIG"

run_experiment "pipeline" "self-refine" "$SELF_REFINE_PIPELINE_CONFIG"
run_experiment "pipeline" "decrim" "$DECRIM_PIPELINE_CONFIG"
run_experiment "pipeline" "reflexion" "$REFLEXION_PIPELINE_CONFIG"
run_experiment "pipeline" "critic" "$CRITIC_PIPELINE_CONFIG"

echo
echo "Все восемь экспериментов завершены."
