#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-input-contract-test.sh

Проверка встроенных средств проверки вводимой информации (п. 3.5.5.1 ТЗ).
Отдельным запускам фильтрации и генерации подаются корректный вход и набор
некорректных: обрыв JSON, пустой файл, не UTF-8, повреждённый gzip, не тот
верхний уровень, нет обязательного поля или таблицы, чужой формат, короткая
строка таблицы, значение не того типа, id вне диапазона, повтор id,
недопустимый символ, ссылка на несуществующую секцию, пустой корпус,
параметры фильтра и генератора не того типа или вне допустимых значений.

Корректный вход должен обрабатываться штатно (код 0, status ok), каждый
некорректный — отклоняться контролируемо: код 2, JSON-ответ со status error и
error.type invalid_input, без аварийного завершения (трассировки стека).
Модели не нужны.

Результат — каталог results/<время UTC>-input-contract/ с
input_contract_report.json; итоговый статус (passed / failed) печатается
последней строкой.
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

LOG_TAG=filter-gen-eval:input-contract
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0"
require_in_workdir "$COMPONENT_ROOT" "$EVAL_ROOT" "$VENV"
require_venv

RUN_DIR="$(new_run_dir input-contract)"
log "каталог прогона: $RUN_DIR"
eval_python input-contract --run-dir "$RUN_DIR" contract
