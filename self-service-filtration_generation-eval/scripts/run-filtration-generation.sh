#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-filtration-generation.sh

Запуск компонента фильтрации и генерации в составе программного решения
«Платформа А2.Pro»: компонент берётся из состава кода платформы — каталога
services/components/filtration_generation рабочего каталога. По очереди:
  1) отдельный запуск фильтрации (./scripts/run-filtration.sh):
     JSON-корпус -> отфильтрованный JSON-корпус;
  2) отдельный запуск генерации правил для системного промпта решателя
     (./scripts/run-rules-generation.sh).
Если фильтрация завершилась с ошибкой, генерация не запускается.

Нужна модель на порту 7217 (./scripts/launch-test-models.sh).

Результат — два каталога, как у отдельных запусков:
  results/<время UTC>-filtration/        filtered.json.gz, filtration_report.json;
  results/<время UTC>-rules-generation/  rules.json, rules_dump.jsonl.
Отчёты печатаются и в консоль.
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

LOG_TAG=filter-gen-eval:a2pro
SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPTS/common.sh"

COMPONENT_ROOT="$WORK_ROOT/services/components/filtration_generation"
[[ -f "$COMPONENT_ROOT/run_ds1000.py" ]] \
  || die "нет компонента в составе кода платформы: $COMPONENT_ROOT"
export COMPONENT_ROOT
log "компонент: $COMPONENT_ROOT"

log "1/2: отдельный запуск фильтрации"
code=0
"$SCRIPTS/run-filtration.sh" || code=$?
if (( code != 0 )); then
  log "ОШИБКА: фильтрация завершилась с кодом $code — генерация не запускается"
  exit "$code"
fi

log "2/2: отдельный запуск генерации правил"
code=0
"$SCRIPTS/run-rules-generation.sh" || code=$?
if (( code != 0 )); then
  log "ОШИБКА: генерация правил завершилась с кодом $code"
  exit "$code"
fi

log "Фильтрация и генерация правил завершились штатно (код 0)."
