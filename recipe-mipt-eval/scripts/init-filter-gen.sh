#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/init-filter-gen.sh

Подготовка окружения испытаний компонента фильтрации и генерации (pk3):
  1) образ asllm на узле (если его нет — docker pull);
  2) venv поверх образа с недостающими пакетами и проверка импорта всей
     цепочки пайплайна (recipe-mipt/scripts/setup_runner_env.sh, check_env.py);
  3) модули испытаний: filter_corpus.py, generate_rules.py, filter_gen_eval;
  4) модель BM25 для разреженных векторов qdrant — в кеш fastembed
     (загружается один раз из хаба HF_ENDPOINT);
  5) проверка JSON-корпуса data/docs_database_examples.json.gz.

Модели для этого шага не нужны. Повторный запуск безопасен.
Настройки — в .env (создаётся из .env.example).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

LOG_TAG=filter-gen-eval:init
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0 $*"

mkdir -p "$(dirname -- "$VENV")" "$FASTEMBED_CACHE_PATH"
require_under_home "$RECIPE_ROOT" "$EVAL_ROOT" "$(dirname -- "$VENV")" "$FASTEMBED_CACHE_PATH"

log "recipe-mipt: $RECIPE_ROOT"
log "образ: $IMG_ASLLM"
if ! docker image inspect "$IMG_ASLLM" >/dev/null 2>&1; then
  log "образа нет на узле — загружаю"
  docker pull "$IMG_ASLLM"
fi

log "venv: $VENV"
VENV="$VENV" IMG_ASLLM="$IMG_ASLLM" bash "$RECIPE_ROOT/scripts/setup_runner_env.sh"

log "модули испытаний"
container_python init-modules -c \
  "import filter_corpus, generate_rules, src.db.json_corpus, filter_gen_eval.cli; print('модули испытаний: ok')"

log "модель BM25 в кеше fastembed ($FASTEMBED_CACHE_PATH)"
container_python init-bm25 -c \
  "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25'); print('BM25: ok')"

log "JSON-корпус"
container_python init-corpus db_scripts/corpus_json.py validate data/docs_database_examples.json.gz

log "Окружение готово."
