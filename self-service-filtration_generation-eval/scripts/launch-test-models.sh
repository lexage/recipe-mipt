#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/launch-test-models.sh [--no-wait | --status | --stop]

Подъём моделей для испытаний pk3 заданиями slurm на узле alibaba:
  Qwen3.6-35B-A3B     порт 7217 — решатель DS-1000, выбор API и генератор правил
                      (scripts/svc-qwen36-7217.sbatch компонента, 1 карта, 88 ГБ ОЗУ);
  Qwen3-Embedding-4B  порт 7216 — эмбеддер
                      (scripts/svc-embed-7216.sbatch компонента, 1 карта, 24 ГБ ОЗУ).
Веса моделей загружены на узел заранее (LLM_MODEL_DIR, EMBED_MODEL_DIR в .env).

Если модель уже отвечает на своём порту, она не перезапускается; если задание
с тем же именем уже стоит в очереди, новое не ставится. Скрипт ждёт, пока оба
эндпоинта ответят именно этими моделями, и печатает "All model servers are ready.".
Модели продолжают работать после выхода скрипта.

Options:
  --no-wait   поставить задания и выйти, не дожидаясь готовности
  --status    показать задания и состояние эндпоинтов
  --stop      снять свои задания моделей (scancel)
  -h, --help  эта справка

Настройки (.env): LLM_MODEL_DIR, EMBED_MODEL_DIR, EMBED_API_NAME,
MODELS_PARTITION (очередь, по умолчанию prims), MODELS_TIME (лимит задания,
по умолчанию 04:00:00), MODELS_READY_TIMEOUT (секунд ожидания, первый старт
модели — до часа из-за JIT-компиляции). Логи заданий — logs/models/.
EOF
}

MODE=start
case "${1:-}" in
  "") ;;
  --no-wait) MODE=no-wait ;;
  --status) MODE=status ;;
  --stop) MODE=stop ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

LOG_TAG=filter-gen-eval:models
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings

ME="$(id -un)"
LOG_DIR="$EVAL_ROOT/logs/models"
MODELS_PARTITION="${MODELS_PARTITION:-prims}"
MODELS_TIME="${MODELS_TIME:-04:00:00}"
MODELS_READY_TIMEOUT="${MODELS_READY_TIMEOUT:-7200}"

# имя задания | sbatch-файл | порт | ожидаемый идентификатор модели | память (пусто — из sbatch-файла)
SERVICES=(
  "svc-qwen36-7217|svc-qwen36-7217.sbatch|$LLM_PORT|$LLM_MODEL_DIR|${LLM_MEM:-}"
  "svc-embed-7216|svc-embed-7216.sbatch|$EMBED_PORT|$EMBED_API_NAME|${EMBED_MEM:-}"
)

job_of() {
  squeue -h -u "$ME" -n "$1" -o '%i' 2>/dev/null | head -n 1
}

job_state() {
  squeue -h -j "$1" -o '%T %R' 2>/dev/null | head -n 1
}

endpoint_status() {
  local code=0
  check_endpoint "$1" "$2" || code=$?
  case "$code" in
    0) echo "готов" ;;
    1) echo "не отвечает" ;;
    *) echo "другая модель: $(endpoint_models "$1" | tr '\n' ' ')" ;;
  esac
}

if [[ "$MODE" == stop ]]; then
  for svc in "${SERVICES[@]}"; do
    IFS='|' read -r name _ _ _ _ <<<"$svc"
    ids="$(squeue -h -u "$ME" -n "$name" -o '%i' 2>/dev/null | tr '\n' ' ')"
    if [[ -n "${ids// /}" ]]; then
      # shellcheck disable=SC2086
      scancel $ids
      log "$name: сняты задания $ids"
    else
      log "$name: заданий нет"
    fi
  done
  exit 0
fi

if [[ "$MODE" == status ]]; then
  for svc in "${SERVICES[@]}"; do
    IFS='|' read -r name _ port expected _ <<<"$svc"
    job="$(job_of "$name")"
    log "$name: задание ${job:-нет}${job:+ ($(job_state "$job"))}; порт $port: $(endpoint_status "$port" "$expected")"
  done
  exit 0
fi

mkdir -p "$LOG_DIR"
declare -A JOB=() PENDING=()
for svc in "${SERVICES[@]}"; do
  IFS='|' read -r name file port expected mem <<<"$svc"
  code=0
  check_endpoint "$port" "$expected" || code=$?
  if [[ "$code" == 0 ]]; then
    log "$name: уже работает на порту $port"
    continue
  fi
  if [[ "$code" == 2 ]]; then
    die "на порту $port другая модель ($(endpoint_models "$port" | tr '\n' ' ')), ожидалась $expected"
  fi
  job="$(job_of "$name")"
  if [[ -n "$job" ]]; then
    log "$name: задание $job уже в очереди ($(job_state "$job"))"
  else
    mem_args=()
    [[ -n "$mem" ]] && mem_args=(--mem "$mem")
    job="$(sbatch --parsable -p "$MODELS_PARTITION" --time "$MODELS_TIME" "${mem_args[@]}" \
      --output "$LOG_DIR/%x-%j.log" "$COMPONENT_ROOT/scripts/$file")"
    job="${job%%;*}"
    log "$name: поставлено задание $job в очередь $MODELS_PARTITION на $MODELS_TIME${mem:+, память $mem} (лог: $LOG_DIR/$name-$job.log)"
  fi
  JOB[$name]="$job"
  PENDING[$name]="$port|$expected"
done

if [[ "$MODE" == no-wait ]]; then
  log "готовность не проверяется (--no-wait); состояние: ./scripts/launch-test-models.sh --status"
  exit 0
fi

deadline=$(( $(date +%s) + MODELS_READY_TIMEOUT ))
last_report=0
while (( ${#PENDING[@]} > 0 )); do
  for name in "${!PENDING[@]}"; do
    IFS='|' read -r port expected <<<"${PENDING[$name]}"
    code=0
    check_endpoint "$port" "$expected" || code=$?
    if [[ "$code" == 0 ]]; then
      log "$name: готов (порт $port)"
      unset "PENDING[$name]"
      continue
    fi
    if [[ "$code" == 2 ]]; then
      die "на порту $port отвечает другая модель ($(endpoint_models "$port" | tr '\n' ' '))"
    fi
    if [[ -z "$(job_state "${JOB[$name]}")" ]]; then
      log "$name: задание ${JOB[$name]} завершилось, а модель не поднялась; конец лога:"
      tail -n 40 "$LOG_DIR/$name-${JOB[$name]}.log" >&2 2>/dev/null || true
      die "$name не запустился"
    fi
  done
  (( ${#PENDING[@]} == 0 )) && break
  now=$(date +%s)
  if (( now > deadline )); then
    die "модели не поднялись за $MODELS_READY_TIMEOUT с: ${!PENDING[*]}"
  fi
  if (( now - last_report >= 60 )); then
    for name in "${!PENDING[@]}"; do
      log "$name: жду (задание ${JOB[$name]}: $(job_state "${JOB[$name]}"))"
    done
    last_report=$now
  fi
  sleep 10
done

echo "All model servers are ready."
