#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/clean-workdir.sh [--yes] [--models]

Возврат рабочего каталога испытаний к состоянию до распаковки архива: в
self-service-filtration_generation остаётся только pk3.zip, всё остальное
удаляется — распакованный компонент и каталог испытаний (вместе с этим
скриптом), venv, кеш, результаты прогонов (results/) и логи.

Без --yes только показывает, что будет удалено. Перед удалением
останавливаются контейнеры испытаний (filter-gen-*); с --models снимаются и
задания моделей (./scripts/launch-test-models.sh --stop).

Работает только в раскладке после распаковки pk3.zip:
  self-service-filtration_generation/{pk3.zip, services/components/filtration_generation/,
  self-service-filtration_generation-eval/}; иначе ничего не удаляет.

Options:
  --yes      удалить
  --models   также снять задания моделей
EOF
}

log() {
  printf '[filter-gen-eval:clean-workdir] %s\n' "$*" >&2
}

# Вся работа — в функции: скрипт удаляет и свой каталог, поэтому bash должен
# прочитать его целиком до начала удаления.
main() {
  local yes=0 models=0
  while (( $# )); do
    case "$1" in
      --yes) yes=1 ;;
      --models) models=1 ;;
      -h|--help) usage; return 0 ;;
      *) usage >&2; return 2 ;;
    esac
    shift
  done

  local eval_root work_root home
  eval_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
  work_root="$(cd -- "$eval_root/.." && pwd -P)"
  home="$(cd -- "$HOME" && pwd -P)"

  local problems=()
  [[ "$(basename -- "$work_root")" == self-service-filtration_generation ]] \
    || problems+=("рабочий каталог называется не self-service-filtration_generation")
  [[ "$(basename -- "$eval_root")" == self-service-filtration_generation-eval ]] \
    || problems+=("каталог испытаний называется не self-service-filtration_generation-eval")
  [[ -f "$work_root/pk3.zip" ]] || problems+=("нет $work_root/pk3.zip")
  [[ -f "$work_root/services/components/filtration_generation/run_ds1000.py" ]] \
    || problems+=("нет компонента $work_root/services/components/filtration_generation")
  [[ "$work_root" != "$home" && "$work_root" != / ]] \
    || problems+=("рабочий каталог — домашний или корневой: $work_root")
  if (( ${#problems[@]} )); then
    log "ОШИБКА: раскладка не та, что после распаковки pk3.zip — ничего не удаляю:"
    printf '  %s\n' "${problems[@]}" >&2
    return 1
  fi

  local items=()
  mapfile -t items < <(find "$work_root" -mindepth 1 -maxdepth 1 ! -name pk3.zip | sort)
  if (( ${#items[@]} == 0 )); then
    log "в $work_root уже только pk3.zip"
    return 0
  fi

  log "рабочий каталог: $work_root"
  log "удаляется всё, кроме pk3.zip:"
  du -sh -- "${items[@]}" >&2 2>/dev/null || printf '  %s\n' "${items[@]}" >&2
  local runs=()
  if [[ -d "$eval_root/results" ]]; then
    mapfile -t runs < <(find "$eval_root/results" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
  if (( ${#runs[@]} )); then
    log "ВНИМАНИЕ: удаляются и результаты прогонов (${#runs[@]}) в $eval_root/results:"
    printf '  %s\n' "${runs[@]##*/}" >&2
  fi

  local cleanup_args=()
  (( yes )) && cleanup_args+=(--yes)
  (( models )) && cleanup_args+=(--models)
  "$eval_root/scripts/cleanup-filter-gen.sh" "${cleanup_args[@]}"
  (( models )) || log "задания моделей не снимаются (для этого --models)"

  if (( ! yes )); then
    log "ничего не удалено; чтобы удалить: ./scripts/clean-workdir.sh --yes"
    return 0
  fi

  cd -- "$work_root"
  rm -rf -- "${items[@]}"
  log "готово: в $work_root остался только pk3.zip"
  log "текущий каталог удалён — перейдите в $work_root: cd $work_root"
}

main "$@"; exit $?
