#!/usr/bin/env bash
# Общие функции скриптов испытаний pk3. Подключается через source, сам не
# запускается.
#
# Рабочий каталог (WORK_ROOT) — каталог, в котором лежит этот каталог
# испытаний. Раскладка после распаковки pk3.zip:
#
#   self-service-filtration_generation/            рабочий каталог
#   ├── pk3.zip
#   ├── services/components/filtration_generation/  компонент
#   └── self-service-filtration_generation-eval/    испытания (этот каталог)
#
# В репозитории каталог испытаний лежит в корне компонента; тогда рабочий
# каталог — корень репозитория. Путь к компоненту можно задать явно
# переменной COMPONENT_ROOT (внутри рабочего каталога).
#
# Весь питон исполняется в контейнере с образом asllm: на узле питон 3.6.
# В контейнер монтируется только рабочий каталог (и, только для чтения, веса
# моделей), поэтому компонент, испытания, venv и кеш должны лежать в нём.

EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="$(cd -- "$EVAL_ROOT/.." && pwd)"
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

resolve_component_root() {
  if [[ -n "${COMPONENT_ROOT:-}" ]]; then
    [[ -f "$COMPONENT_ROOT/run_ds1000.py" ]] || die "COMPONENT_ROOT=$COMPONENT_ROOT: нет run_ds1000.py"
  elif [[ -f "$WORK_ROOT/services/components/filtration_generation/run_ds1000.py" ]]; then
    COMPONENT_ROOT="$WORK_ROOT/services/components/filtration_generation"
  elif [[ -f "$WORK_ROOT/run_ds1000.py" ]]; then
    COMPONENT_ROOT="$WORK_ROOT"
  else
    die "не найден компонент: ни $WORK_ROOT/services/components/filtration_generation, ни $WORK_ROOT; задайте COMPONENT_ROOT"
  fi
  COMPONENT_ROOT="$(cd -- "$COMPONENT_ROOT" && pwd)"
  export COMPONENT_ROOT
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
  FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-$WORK_ROOT/.cache/fastembed}"
  WORKERS="${WORKERS:-4}"
  resolve_component_root
  EVAL_CONFIG="$EVAL_ROOT/config.toml"
  git_state
}

# Все пути, с которыми работают скрипты, должны быть внутри рабочего каталога:
# в контейнер монтируется только он.
require_in_workdir() {
  local path real root
  root="$(readlink -f -- "$WORK_ROOT")"
  for path in "$@"; do
    # -m: путь может ещё не существовать (venv и кеш создаются после проверки).
    real="$(readlink -m -- "$path")"
    case "$real/" in
      "$root"/*) ;;
      *) die "путь $path вне рабочего каталога $WORK_ROOT: в контейнер монтируется только он" ;;
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
  local dir="$EVAL_ROOT/results/$(utc_stamp)-$1"
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
    -v "$WORK_ROOT:$WORK_ROOT" -v "$MODELS_ROOT:$MODELS_ROOT:ro" \
    -w "$COMPONENT_ROOT" \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="$COMPONENT_ROOT:$EVAL_ROOT/src" \
    -e COMPONENT_ROOT="$COMPONENT_ROOT" -e EVAL_ROOT="$EVAL_ROOT" \
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

# Состояние git компонента для run_manifest.json.
git_state() {
  if git -C "$COMPONENT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$COMPONENT_ROOT" rev-parse HEAD)"
    GIT_BRANCH="$(git -C "$COMPONENT_ROOT" rev-parse --abbrev-ref HEAD)"
    if [[ -n "$(git -C "$COMPONENT_ROOT" status --porcelain --untracked-files=no)" ]]; then
      GIT_DIRTY=true
    else
      GIT_DIRTY=false
    fi
  elif [[ -f "$COMPONENT_ROOT/DEPLOYED_COMMIT" ]]; then
    # Поставка без .git (архив pk3.zip или доставка rsync): коммит записан в файл.
    GIT_COMMIT="$(sed -n 1p "$COMPONENT_ROOT/DEPLOYED_COMMIT")"
    GIT_BRANCH="$(sed -n 2p "$COMPONENT_ROOT/DEPLOYED_COMMIT")"
    GIT_DIRTY="$(sed -n 3p "$COMPONENT_ROOT/DEPLOYED_COMMIT")"
    [[ "$GIT_DIRTY" == true ]] || GIT_DIRTY=false
  else
    GIT_COMMIT="неизвестен (нет .git и DEPLOYED_COMMIT)"
    GIT_BRANCH=""
    GIT_DIRTY=false
  fi
  export GIT_COMMIT GIT_BRANCH GIT_DIRTY
}
