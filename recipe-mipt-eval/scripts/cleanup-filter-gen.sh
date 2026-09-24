#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/cleanup-filter-gen.sh [--yes] [--models]

Остановка контейнеров испытаний pk3 (имена filter-gen-*), запущенных текущим
пользователем. Без --yes только показывает, что будет остановлено.

Options:
  --yes      остановить найденные контейнеры
  --models   также снять задания моделей (./scripts/launch-test-models.sh --stop)

Файлы результатов и индексы не удаляются.
EOF
}

YES=0
MODELS=0
while (( $# )); do
  case "$1" in
    --yes) YES=1; shift ;;
    --models) MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

LOG_TAG=filter-gen-eval:cleanup
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings

mapfile -t names < <(docker ps --format '{{.Names}}' 2>/dev/null | grep '^filter-gen-' || true)
if (( ${#names[@]} == 0 )); then
  log "контейнеров испытаний нет"
elif (( YES )); then
  docker stop "${names[@]}" >/dev/null
  log "остановлены: ${names[*]}"
else
  log "будут остановлены (добавьте --yes): ${names[*]}"
fi

if (( MODELS )); then
  if (( YES )); then
    "$EVAL_ROOT/scripts/launch-test-models.sh" --stop
  else
    log "задания моделей будут сняты с --yes --models"
  fi
fi
