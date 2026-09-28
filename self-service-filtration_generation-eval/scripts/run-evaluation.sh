#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-evaluation.sh --experiment filtration|generation|all
                                 [--limit N] [--max-docs N] [--workers N] [--no-log-chunks]
                                 [--skip-preflight]

Прогон пайплайна (run_ds1000.py компонента, как есть) на наборе DS-1000:
бейзлайн и метод эксперимента подряд, в одном пайплайне. У каждого
эксперимента свой бейзлайн:

  filtration  сначала отдельный запуск фильтрации (filter_corpus.py) полного
              JSON-корпуса; затем f0_full (полный корпус) и f1_api_genre
              (отфильтрованный корпус как path_to_db, шага фильтрации нет);
  generation  g0_base (системный промпт без правил) и g1_rules (генератор
              правил в пайплайне, правила пишутся заново);
  all         оба эксперимента одним запуском: отдельная фильтрация, затем
              четыре конфига одной папкой — f0_full, f1_api_genre, g0_base,
              g1_rules; сводка по обоим экспериментам.

Каждый прогон строит свой индекс с нуля в каталоге прогона. Нужны обе модели
(./scripts/launch-test-models.sh). Полный прогон занимает несколько часов:
запускайте в tmux/screen или через nohup.

Options:
  --limit N          только N задач DS-1000, равномерно по набору (проверка стенда)
  --max-docs N       индекс только по первым N документам корпуса (проверка стенда;
                     метрики такого прогона не являются результатом испытаний)
  --workers N        параллельные воркеры (по умолчанию WORKERS из .env)
  --no-log-chunks    не писать retrieved_chunks.jsonl
  --skip-preflight   не проверять модели перед стартом

Результат — каталог results/<время UTC>-<эксперимент>/:
  summary.md, summary.json   Pусп, метрики с порогами config.toml, таблица прогонов;
  runs/<конфиг>/<время>/     results.csv, runtime_stats.json, summary.txt, ...;
  configs/                   конфиги прогона (копии pmi_configs с путями этого прогона);
  filtration/                отдельный запуск фильтрации (для filtration);
  rules/                     журнал генератора правил (для generation);
                             для all — и filtration/, и rules/;
  run_manifest.json          что и из какого кода запускалось.
summary.md печатается и в консоль.
EOF
}

EXPERIMENT=""
LIMIT=0
MAX_DOCS=0
WORKERS_ARG=""
LOG_CHUNKS=1
PREFLIGHT=1
while (( $# )); do
  case "$1" in
    --experiment) EXPERIMENT="${2:?не указан эксперимент}"; shift 2 ;;
    --limit) LIMIT="${2:?не указано N}"; shift 2 ;;
    --max-docs) MAX_DOCS="${2:?не указано N}"; shift 2 ;;
    --workers) WORKERS_ARG="${2:?не указано N}"; shift 2 ;;
    --no-log-chunks) LOG_CHUNKS=0; shift ;;
    --skip-preflight) PREFLIGHT=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$EXPERIMENT" in
  filtration|generation|all) ;;
  *) usage >&2; exit 2 ;;
esac

LOG_TAG="filter-gen-eval:evaluation-$EXPERIMENT"
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0 --experiment $EXPERIMENT --limit $LIMIT --max-docs $MAX_DOCS"
require_in_workdir "$COMPONENT_ROOT" "$EVAL_ROOT" "$VENV" "$FASTEMBED_CACHE_PATH"
require_venv
if (( PREFLIGHT )); then
  preflight_models
fi

EXTRA=(--workers "${WORKERS_ARG:-$WORKERS}" --limit "$LIMIT" --max-docs "$MAX_DOCS")
(( LOG_CHUNKS )) || EXTRA+=(--no-log-chunks)

RUN_DIR="$(new_run_dir "$EXPERIMENT")"
log "каталог прогона: $RUN_DIR"
code=0
eval_python "eval-$EXPERIMENT" --run-dir "$RUN_DIR" evaluate --experiment "$EXPERIMENT" "${EXTRA[@]}" || code=$?
log "сводка: $RUN_DIR/summary.md (код $code)"
exit "$code"
