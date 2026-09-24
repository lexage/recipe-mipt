#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-filtration.sh [--filter-config PATH] [--input PATH]

Отдельный запуск метода фильтрации (без пайплайна и без моделей):
JSON-корпус -> отфильтрованный JSON-корпус того же формата.

По умолчанию (config.toml, [experiments.filtration]):
  метод и параметры — pmi_configs/filtration/f1_api_genre_inline.yaml (document_filter);
  вход              — data/docs_database_examples.json.gz.
Пути — от корня recipe-mipt.

Результат — каталог results/<время UTC>-filtration/:
  filtered.json.gz         отфильтрованный корпус (подаётся в пайплайн как path_to_db);
  filtration_report.json   отчёт: число документов до и после, решения фильтра,
                           Pусп, сокращение объёма, проверки, время;
  run_manifest.json        что и из какого кода запускалось.
Отчёт печатается и в консоль.
EOF
}

ARGS=()
while (( $# )); do
  case "$1" in
    --filter-config|--input) ARGS+=("$1" "${2:?не указано значение для $1}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

LOG_TAG=filter-gen-eval:filtration
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0 ${ARGS[*]:-}"
require_under_home "$RECIPE_ROOT" "$EVAL_ROOT"
require_venv

RUN_DIR="$(new_run_dir filtration)"
log "каталог прогона: $RUN_DIR"
code=0
eval_python filtration --run-dir "$RUN_DIR" filtration "${ARGS[@]}" || code=$?
log "результаты: $RUN_DIR (код $code)"
exit "$code"
