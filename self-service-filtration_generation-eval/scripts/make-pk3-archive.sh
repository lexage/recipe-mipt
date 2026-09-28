#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/make-pk3-archive.sh [путь к pk3.zip]

Сборка архива поставки pk3.zip (по умолчанию dist/pk3.zip) из закоммиченного
состояния репозитория: services/components/filtration_generation/ (компонент)
и self-service-filtration_generation-eval/ (испытания), только файлы из
белого списка tools/make_archive.py.
Инструмент разработчика, в архив не входит. Нужны git и python3.
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(git -C "$EVAL_ROOT" rev-parse --show-toplevel)"
OUT="${1:-$EVAL_ROOT/dist/pk3.zip}"
python3 "$EVAL_ROOT/tools/make_archive.py" "$REPO" "$OUT"
