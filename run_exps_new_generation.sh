#!/usr/bin/env bash
#
# Run the exp16 battery in order, resumably.
#
#   ./run_exp16.sh                 # run everything still missing
#   ./run_exp16.sh --dry-run       # print the plan and what would be skipped
#   ./run_exp16.sh --limit 150     # smoke pass over every config
#   ./run_exp16.sh --from e2_minus_t1   # resume at a specific step
#   ./run_exp16.sh --force         # re-run even completed configs
#
# A config counts as done when results/<stem>/<timestamp>/runtime_stats.json
# exists — the runner writes that file last, after bench.eval returns, so a
# half-finished run is never mistaken for a complete one.
#
# Sequential on purpose. The configs share one vLLM endpoint and the local
# qdrant takes a filesystem lock on its index directory, so running these in
# parallel gets you contention at best and a corrupt index at worst.

set -uo pipefail

CONFIG_DIR="test_configs_experimental_16"
RESULTS_DIR="results"
LOG_DIR="logs/exp16"
SCRIPTS_DIR="db_scripts"
LIMIT=""
DRY_RUN=0
FORCE=0
START_AT=""

# Ordered plan. @warm-start is a hook, not a config: it seeds each variant's
# vector index from the base one so only the ~200 added documents get embedded.
# It has to land after e0_baseline (which builds that base index) and before
# everything else.
PLAN=(
  e0_baseline
  e0_aug_full
  @warm-start
  e3_guide_in_prompt
  e6_base_inlined
  e6_aug_inlined
  e2_minus_t1
  e2_minus_t1_t2
  e1_only_guide
  e1_only_migr
  e1_only_howto
  e1_minus_guide
  e1_minus_migr
  e1_minus_howto
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)      DRY_RUN=1; shift ;;
    --force)        FORCE=1; shift ;;
    --limit)        LIMIT="$2"; shift 2 ;;
    --from)         START_AT="$2"; shift 2 ;;
    --config-dir)   CONFIG_DIR="$2"; shift 2 ;;
    --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

is_done() {  # $1 = config stem
  compgen -G "$RESULTS_DIR/$1/*/runtime_stats.json" > /dev/null 2>&1
}

resolve_config() {  # $1 = stem -> path, honouring both flat and nested layouts
  local flat="$CONFIG_DIR/$1.yaml"
  [[ -f "$flat" ]] && { echo "$flat"; return 0; }
  local nested
  nested=$(find "$CONFIG_DIR" -mindepth 2 -maxdepth 2 -name "$1.yaml" 2>/dev/null | head -1)
  [[ -n "$nested" ]] && { echo "$nested"; return 0; }
  return 1
}

hms() { printf '%02d:%02d:%02d' $(($1/3600)) $((($1%3600)/60)) $(($1%60)); }

run_config() {  # $1 = stem, $2 = config path
  local stem="$1" cfg="$2" tmp status
  # `-c` is staged through a throwaway directory rather than passed as a file:
  # unpatched run_ds1000.py does Path(arg).glob("*.yaml"), which silently yields
  # nothing for a file path — the process then exits 0 in about a second having
  # run no configs at all. A directory works on every version. The copy keeps
  # the stem, so results/<stem>/ and the config snapshot are unaffected.
  tmp=$(mktemp -d)
  cp "$cfg" "$tmp/$stem.yaml"
  # shellcheck disable=SC2086  # $LIMIT is intentionally word-split (may be empty)
  python run_ds1000.py -c "$tmp" --log-chunks ${LIMIT:+-l $LIMIT} \
    2>&1 | tee "$LOG_DIR/$stem.log"
  status=${PIPESTATUS[0]}
  rm -rf "$tmp"
  return $status
}

explain_failure() {  # $1 = stem — turn a bare exit code into something actionable
  local log="$LOG_DIR/$1.log"
  if grep -q "CONFIG BUILD FAILED" "$log" 2>/dev/null; then
    echo "    the pipeline failed to build — the traceback is in the log:"
    grep -A3 "CONFIG BUILD FAILED" "$log" | tail -3 | sed 's/^/      /'
  elif ! grep -q "processing file" "$log" 2>/dev/null; then
    echo "    run_ds1000.py started but ran no configs at all."
    echo "    Usually a path problem — check that $CONFIG_DIR is right."
  else
    echo "    see $LOG_DIR/$1.log"
  fi
}

mkdir -p "$LOG_DIR"
BATTERY_START=$(date +%s)
declare -a SUMMARY=()
skipping=0
[[ -n "$START_AT" ]] && skipping=1

echo "=== exp16 battery ==="
echo "configs: $CONFIG_DIR   results: $RESULTS_DIR   logs: $LOG_DIR"
[[ -n "$LIMIT" ]] && echo "LIMIT: $LIMIT tasks per config (smoke pass — CI on PASS@1"\
                          "is roughly +/-4pp at 150 tasks, so treat it as a"\
                          "health check, not a result)"
echo

for step in "${PLAN[@]}"; do
  if [[ $skipping -eq 1 ]]; then
    if [[ "$step" == "$START_AT" ]]; then skipping=0; else
      echo "  ..  $step (before --from)"; continue
    fi
  fi

  # --- hook -------------------------------------------------------------
  if [[ "$step" == "@warm-start" ]]; then
    echo ">>  warm-starting variant indexes from the base one"
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    (dry run) python $SCRIPTS_DIR/warm_start_exp16_vdbs.py"
    else
      python "$SCRIPTS_DIR/warm_start_exp16_vdbs.py" \
        2>&1 | tee "$LOG_DIR/warm_start.log"
    fi
    echo
    continue
  fi

  # --- config -----------------------------------------------------------
  if ! cfg=$(resolve_config "$step"); then
    echo "!!  $step — no such config under $CONFIG_DIR, skipping"
    SUMMARY+=("MISSING   $step")
    continue
  fi

  if [[ $FORCE -eq 0 ]] && is_done "$step"; then
    echo "==  $step — already has results, skipping (use --force to redo)"
    SUMMARY+=("SKIPPED   $step")
    continue
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo ">>  $step — would run ($cfg)"
    SUMMARY+=("PLANNED   $step")
    continue
  fi

  echo ">>  $step — starting $(date '+%H:%M:%S')"
  step_start=$(date +%s)
  run_config "$step" "$cfg"
  status=$?
  elapsed=$(( $(date +%s) - step_start ))

  if [[ $status -eq 0 ]] && is_done "$step"; then
    echo "    done in $(hms $elapsed)"
    SUMMARY+=("OK        $step   $(hms $elapsed)")
  else
    # Keep going: one broken config should not cost the whole overnight batch.
    # Note run_ds1000.py exits 0 even when a config fails to BUILD, so a zero
    # status with no runtime_stats.json still counts as a failure here.
    echo "    FAILED (exit $status) after $(hms $elapsed)"
    explain_failure "$step"
    SUMMARY+=("FAILED    $step   $(hms $elapsed)")
  fi
  echo
done

echo "=== summary (total $(hms $(( $(date +%s) - BATTERY_START )))) ==="
printf '  %s\n' "${SUMMARY[@]}"

if printf '%s\n' "${SUMMARY[@]}" | grep -q '^FAILED'; then
  echo
  echo "Some configs failed. Fix them, then re-run this script — completed"
  echo "configs are skipped automatically."
  exit 1
fi
