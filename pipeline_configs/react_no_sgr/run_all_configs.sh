#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-data/ds1000/ds1000.jsonl.gz}"
SAVE_ROOT="${SAVE_ROOT:-results/react_no_sgr}"
NUM_WORKERS="${NUM_WORKERS:-1}"

DOCS_DB="/home/shtraukh/work/recipe-mipt/data/docs_database_examples.db"
VECTOR_DB="/home/shtraukh/work/recipe-mipt/data/docs_vector_database_qwen_4b"

if [[ ! -f "$DOCS_DB" ]]; then
  echo "ERROR: SQLite documentation DB not found: $DOCS_DB" >&2
  exit 1
fi

if [[ ! -d "$VECTOR_DB" ]]; then
  echo "ERROR: vector DB not found: $VECTOR_DB" >&2
  echo "Build it first with: python scripts/build_docs_vector_db.py" >&2
  exit 1
fi

CONFIG_DIRS=(
  "pipeline_configs/react_no_sgr/baseline"
  "pipeline_configs/react_no_sgr/table9/baseline_llm"
  "pipeline_configs/react_no_sgr/table9/react_code_tool"
  "pipeline_configs/react_no_sgr/table9/react_llm_tool"
  "pipeline_configs/react_no_sgr/table9/react_critic_tool"
  "pipeline_configs/react_no_sgr/table9/react_simple_rag"
  "pipeline_configs/react_no_sgr/table9/react_instruct_rag"
  "pipeline_configs/react_no_sgr/table9/react_corag"
)

mkdir -p "$SAVE_ROOT"

for config_dir in "${CONFIG_DIRS[@]}"; do
  yaml_count="$(find "$config_dir" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | wc -l)"
  if [[ "$yaml_count" -ne 1 ]]; then
    echo "ERROR: expected exactly one YAML config in $config_dir, found $yaml_count" >&2
    exit 1
  fi

  mkdir -p "$config_dir/logs"
  config_file="$(find "$config_dir" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print -quit)"
  config_name="$(basename "$config_file")"

  echo
  echo "================================================================"
  echo "Running: $config_name"
  echo "Config dir: $config_dir"
  echo "Logs: $config_dir/logs"
  echo "================================================================"

  # Each config is launched in a fresh Python process so local Qdrant
  # resources/file locks are released before the next configuration.
  python run_ds1000.py \
    -c "$config_dir" \
    -d "$DATASET" \
    -s "$SAVE_ROOT" \
    -n "$NUM_WORKERS"

  echo "Finished: $config_name"
done

echo
echo "All ReAct no-SGR configs finished successfully."
