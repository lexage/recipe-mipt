#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/init-filter-gen.sh

Подготовка окружения испытаний компонента фильтрации и генерации (pk3):
  1) образ asllm на узле (если его нет — docker pull);
  2) venv поверх образа с недостающими пакетами и проверка импорта всей
     цепочки пайплайна (scripts/setup_runner_env.sh и scripts/check_env.py
     компонента); venv — в рабочем каталоге (VENV в .env);
  3) модули испытаний: filter_corpus.py, generate_rules.py, filter_gen_eval;
  4) модель BM25 для разреженных векторов qdrant — в кеш fastembed в рабочем
     каталоге (загружается один раз из хаба HF_ENDPOINT) и проверка её
     офлайн-загрузки так, как её грузит пайплайн;
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

require_in_workdir "$COMPONENT_ROOT" "$EVAL_ROOT" "$(dirname -- "$VENV")" "$FASTEMBED_CACHE_PATH"
mkdir -p "$(dirname -- "$VENV")" "$FASTEMBED_CACHE_PATH"

log "рабочий каталог: $WORK_ROOT"
log "компонент: $COMPONENT_ROOT"
log "образ: $IMG_ASLLM"
if ! docker image inspect "$IMG_ASLLM" >/dev/null 2>&1; then
  log "образа нет на узле — загружаю"
  docker pull "$IMG_ASLLM"
fi

log "venv: $VENV"
VENV="$VENV" IMG_ASLLM="$IMG_ASLLM" MOUNT_ROOT="$WORK_ROOT" bash "$COMPONENT_ROOT/scripts/setup_runner_env.sh"

log "модули испытаний"
container_python init-modules -c \
  "import filter_corpus, generate_rules, src.db.json_corpus, filter_gen_eval.cli; print('модули испытаний: ok')"

log "модель BM25 в кеше fastembed ($FASTEMBED_CACHE_PATH)"
BM25_PY="$(cat <<'PY'
import os, shutil
from pathlib import Path
from fastembed import SparseTextEmbedding

cache = Path(os.environ["FASTEMBED_CACHE_PATH"])
# 1) Загрузка из хаба HF_ENDPOINT (раскладка Hugging Face).
snapshot = Path(SparseTextEmbedding(model_name="Qdrant/bm25").model._model_dir)
# 2) Пайплайн (run_ds1000.py) грузит модель офлайн (HF_HUB_OFFLINE=1). fastembed
#    0.8 в этом режиме отвергает снимок Hugging Face — требует mock.file и
#    tamil.txt, которых в репозитории модели нет, — и ищет каталог <кеш>/bm25.
#    Кладём туда те же файлы (копии, без ссылок).
target = cache / "bm25"
if not (target.is_dir() and any(target.iterdir())):
    tmp = cache / "bm25.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(snapshot, tmp, symlinks=False)
    tmp.rename(target)
# 3) Проверка так же, как в пайплайне: офлайн.
os.environ["HF_HUB_OFFLINE"] = "1"
SparseTextEmbedding(model_name="Qdrant/bm25")
print("BM25: ok (офлайн-загрузка, как в пайплайне)")
PY
)"
container_python init-bm25 -c "$BM25_PY"

log "JSON-корпус"
container_python init-corpus db_scripts/corpus_json.py validate data/docs_database_examples.json.gz

log "Окружение готово."
