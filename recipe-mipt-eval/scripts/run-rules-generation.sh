#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-rules-generation.sh [--rules-config PATH]

Отдельный запуск метода генерации правил для системного промпта решателя
(без пайплайна). Нужна модель на порту 7217 (./scripts/launch-test-models.sh).

По умолчанию метод и параметры берутся из pmi_configs/generation/g1_rules.yaml
(компонент rule_writer). Правила пишутся заново при каждом запуске.

Результат — каталог results/<время UTC>-rules-generation/:
  rules.json         итог: текст блока правил, список правил, число вызовов по
                     исходам, Pусп, модель, seed;
  rules_dump.jsonl   журнал метода: каждый вызов (промпт, ответ, вердикт) и итог;
  run_manifest.json  что и из какого кода запускалось.
Итог печатается и в консоль.
EOF
}

ARGS=()
while (( $# )); do
  case "$1" in
    --rules-config) ARGS+=("$1" "${2:?не указано значение для $1}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

LOG_TAG=filter-gen-eval:generation
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0 ${ARGS[*]:-}"
require_under_home "$RECIPE_ROOT" "$EVAL_ROOT"
require_venv
require_endpoint "$LLM_PORT" "$LLM_MODEL_DIR"

RUN_DIR="$(new_run_dir rules-generation)"
log "каталог прогона: $RUN_DIR"
code=0
eval_python generation --run-dir "$RUN_DIR" generation "${ARGS[@]}" || code=$?
log "результаты: $RUN_DIR (код $code)"
exit "$code"
