#!/usr/bin/env bash
# Общие функции скриптов испытаний pk3. Подключается через source, сам не
# запускается.
#
# Раскладка каталогов: recipe-mipt-eval лежит либо рядом с recipe-mipt (как в
# архиве pk3.zip), либо внутри него (как в репозитории). Путь к recipe-mipt
# можно задать явно переменной RECIPE_ROOT.
#
# Весь питон исполняется в контейнере с образом asllm: на узле питон 3.6.
# В контейнер монтируется домашний каталог, поэтому recipe-mipt,
# recipe-mipt-eval и venv должны лежать в $HOME.

EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_TAG="${LOG_TAG:-filter-gen-eval}"

LLM_PORT=7217
EMBED_PORT=7216

log() {
  printf '[%s] %s\n' "$LOG_TAG" "$*" >&2
}

die() {
  log "ОШИБКА: $*"
  exit 1
}

resolve_recipe_root() {
  if [[ -n "${RECIPE_ROOT:-}" ]]; then
    [[ -f "$RECIPE_ROOT/run_ds1000.py" ]] || die "RECIPE_ROOT=$RECIPE_ROOT: нет run_ds1000.py"
  elif [[ -f "$EVAL_ROOT/../recipe-mipt/run_ds1000.py" ]]; then
    RECIPE_ROOT="$EVAL_ROOT/../recipe-mipt"
  elif [[ -f "$EVAL_ROOT/../run_ds1000.py" ]]; then
    RECIPE_ROOT="$EVAL_ROOT/.."
  else
    die "не найден каталог recipe-mipt (рядом с recipe-mipt-eval или над ним); задайте RECIPE_ROOT"
  fi
  RECIPE_ROOT="$(cd -- "$RECIPE_ROOT" && pwd)"
  export RECIPE_ROOT
}

# Читает .env и config.toml (создаёт их из примеров, если их нет).
load_settings() {
  if [[ ! -f "$EVAL_ROOT/.env" ]]; then
    cp "$EVAL_ROOT/.env.example" "$EVAL_ROOT/.env"
    log "создан .env из .env.example"
  fi
  if [[ ! -f "$EVAL_ROOT/config.toml" ]]; then
    cp "$EVAL_ROOT/config.example.toml" "$EVAL_ROOT/config.toml"
    log "создан config.toml из config.example.toml"
  fi
  set -a
  # shellcheck disable=SC1091
  source "$EVAL_ROOT/.env"
  set +a
  : "${IMG_ASLLM:?не задан IMG_ASLLM в .env}"
  : "${MODELS_ROOT:?не задан MODELS_ROOT в .env}"
  : "${LLM_MODEL_DIR:?не задан LLM_MODEL_DIR в .env}"
  : "${EMBED_API_NAME:?не задан EMBED_API_NAME в .env}"
  : "${VENV:?не задан VENV в .env}"
  HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-$HOME/.cache/fastembed}"
  WORKERS="${WORKERS:-4}"
  resolve_recipe_root
  EVAL_CONFIG="$EVAL_ROOT/config.toml"
  git_state
}

# Все пути, которые монтируются в контейнер, должны быть внутри $HOME.
require_under_home() {
  local path real
  for path in "$@"; do
    real="$(readlink -f -- "$path")"
    case "$real/" in
      "$HOME"/*) ;;
      *) die "путь $path вне домашнего каталога: в контейнер монтируется только $HOME" ;;
    esac
  done
}

require_venv() {
  # bin/python в venv — ссылка на интерпретатор контейнера, с узла она выглядит
  # битой; поэтому проверяем bin/activate.
  [[ -f "$VENV/bin/activate" ]] || die "нет venv $VENV — сначала ./scripts/init-filter-gen.sh"
}

utc_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}

# Создаёт каталог прогона results/<время UTC>-<вид> и печатает его путь.
new_run_dir() {
  local results="${RESULTS_DIR:-$EVAL_ROOT/results}"
  local dir="$results/$(utc_stamp)-$1"
  mkdir -p "$dir"
  printf '%s\n' "$dir"
}

# Запуск питона в контейнере: container_python <метка> <аргументы python...>.
# Имя контейнера — filter-gen-<метка>-<pid> или $CONTAINER_NAME, если задано.
container_python() {
  local tag="$1"
  shift
  local name="${CONTAINER_NAME:-filter-gen-${tag}-$$}"
  docker run --rm --name "$name" --network host --ipc host \
    -v "$HOME:$HOME" -v "$MODELS_ROOT:$MODELS_ROOT:ro" \
    -w "$RECIPE_ROOT" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="$RECIPE_ROOT:$EVAL_ROOT/src" \
    -e RECIPE_ROOT="$RECIPE_ROOT" -e EVAL_ROOT="$EVAL_ROOT" \
    -e HF_ENDPOINT="$HF_ENDPOINT" \
    -e FASTEMBED_CACHE_PATH="$FASTEMBED_CACHE_PATH" \
    -e IMG_ASLLM="$IMG_ASLLM" -e LLM_MODEL_DIR="$LLM_MODEL_DIR" -e EMBED_API_NAME="$EMBED_API_NAME" \
    -e GIT_COMMIT="$GIT_COMMIT" -e GIT_BRANCH="$GIT_BRANCH" -e GIT_DIRTY="$GIT_DIRTY" \
    -e RUN_HOST="$(hostname)" -e RUN_COMMAND="${RUN_COMMAND:-}" \
    "$IMG_ASLLM" "$VENV/bin/python" "$@"
}

# Команда обвязки в контейнере: eval_python <метка> <подкоманда и её аргументы>.
eval_python() {
  local tag="$1"
  shift
  container_python "$tag" -m filter_gen_eval --config-toml "$EVAL_CONFIG" "$@"
}

# Идентификаторы моделей, которые отдаёт эндпоинт (по строке на модель).
endpoint_models() {
  local port="$1" body
  body="$(curl -sf -m 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null)" || return 1
  python3 -c 'import json,sys; print("\n".join(m.get("id","") for m in json.load(sys.stdin).get("data",[])))' <<<"$body"
}

# 0 — на порту нужная модель; 1 — порт не отвечает; 2 — на порту другая модель.
check_endpoint() {
  local port="$1" expected="$2" ids
  ids="$(endpoint_models "$port")" || return 1
  grep -qxF -- "$expected" <<<"$ids" && return 0
  return 2
}

# Модель на порту поднята и это именно она — иначе выход с ошибкой.
require_endpoint() {
  local port="$1" expected="$2" code=0
  check_endpoint "$port" "$expected" || code=$?
  case "$code" in
    0) log "порт $port: $expected — готов" ;;
    1) die "порт $port не отвечает — сначала ./scripts/launch-test-models.sh" ;;
    *) die "на порту $port другая модель ($(endpoint_models "$port" | tr '\n' ' ')), ожидалась $expected" ;;
  esac
}

# Проверка перед прогоном пайплайна: обе модели подняты.
preflight_models() {
  require_endpoint "$LLM_PORT" "$LLM_MODEL_DIR"
  require_endpoint "$EMBED_PORT" "$EMBED_API_NAME"
}

# Состояние git каталога recipe-mipt для run_manifest.json.
git_state() {
  if git -C "$RECIPE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$RECIPE_ROOT" rev-parse HEAD)"
    GIT_BRANCH="$(git -C "$RECIPE_ROOT" rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git -C "$RECIPE_ROOT" status --porcelain --untracked-files=no)" ]]; then
      GIT_DIRTY=true
    else
      GIT_DIRTY=false
    fi
  elif [[ -f "$RECIPE_ROOT/DEPLOYED_COMMIT" ]]; then
    # Поставка без .git (архив pk3.zip или доставка rsync): коммит записан в файл.
    GIT_COMMIT="$(sed -n 1p "$RECIPE_ROOT/DEPLOYED_COMMIT")"
    GIT_BRANCH="$(sed -n 2p "$RECIPE_ROOT/DEPLOYED_COMMIT")"
    GIT_DIRTY="$(sed -n 3p "$RECIPE_ROOT/DEPLOYED_COMMIT")"
    [[ "$GIT_DIRTY" == true ]] || GIT_DIRTY=false
  else
    GIT_COMMIT="неизвестен (нет .git и DEPLOYED_COMMIT)"
    GIT_BRANCH=""
    GIT_DIRTY=false
  fi
  export GIT_COMMIT GIT_BRANCH GIT_DIRTY
}
