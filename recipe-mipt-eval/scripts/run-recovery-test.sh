#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-recovery-test.sh

Проверка восстановления после отказа (пп. 3.5.4.2, 3.5.4.3 ТЗ) на отдельном
запуске фильтрации:
  1) эталон — штатный запуск фильтрации, замеряется его время;
  2) отказ — такой же запуск в контейнере с известным именем принудительно
     уничтожается (docker kill) через KILL_AFTER секунд, во время обработки;
  3) восстановление — повторный запуск той же команды, без ручных действий по
     очистке; время восстановления — от повторного запуска до его штатного
     завершения;
  4) проверка — результат повторного запуска совпадает с эталоном побайтно
     (sha256 выходного корпуса), время восстановления не больше RECOVERY_MAX_S.

Модели не нужны. Переменные окружения: KILL_AFTER (по умолчанию 5 с),
RECOVERY_MAX_S (по умолчанию 600 с).

Результат — каталог results/<время UTC>-recovery/ с recovery_report.json;
итоговый статус (passed / failed) печатается последней строкой.
EOF
}

case "${1:-}" in
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

LOG_TAG=filter-gen-eval:recovery
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_settings
RUN_COMMAND="$0"
require_under_home "$RECIPE_ROOT" "$EVAL_ROOT"
require_venv

KILL_AFTER="${KILL_AFTER:-5}"
RECOVERY_MAX_S="${RECOVERY_MAX_S:-600}"
RUN_DIR="$(new_run_dir recovery)"
REF_DIR="$RUN_DIR/reference"
REC_DIR="$RUN_DIR/recovered"
NAME="filter-gen-recovery-$$"
log "каталог прогона: $RUN_DIR"

running() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qxF -- "$NAME"
}

# 1) эталон
log "эталонный запуск фильтрации"
t0=$(date +%s.%N)
eval_python recovery-reference --run-dir "$REF_DIR" filtration >"$RUN_DIR/reference.log" 2>&1 \
  || die "эталонный запуск не удался, см. $RUN_DIR/reference.log"
t1=$(date +%s.%N)

# 2) отказ
log "запуск, который будет прерван через $KILL_AFTER с"
( CONTAINER_NAME="$NAME" eval_python recovery-interrupted --run-dir "$REC_DIR" filtration \
    >"$RUN_DIR/interrupted.log" 2>&1 ) &
BG=$!
for _ in $(seq 1 120); do
  running && break
  kill -0 "$BG" 2>/dev/null || break
  sleep 0.5
done
running || die "контейнер $NAME не запустился, см. $RUN_DIR/interrupted.log"
sleep "$KILL_AFTER"
running || die "запуск завершился раньше имитации отказа — уменьшите KILL_AFTER (сейчас $KILL_AFTER)"
docker kill "$NAME" >/dev/null
interrupted_code=0
wait "$BG" || interrupted_code=$?
log "контейнер $NAME уничтожен (код завершения запуска: $interrupted_code)"
running && die "контейнер $NAME всё ещё работает после docker kill"

# 3) восстановление
log "повторный запуск той же команды"
t2=$(date +%s.%N)
recovered_code=0
eval_python recovery-restart --run-dir "$REC_DIR" filtration >"$RUN_DIR/recovered.log" 2>&1 \
  || recovered_code=$?
t3=$(date +%s.%N)

# 4) проверка
ref_sha="$(sha256sum "$REF_DIR/filtered.json.gz" | cut -d' ' -f1)"
rec_sha="missing"
[[ -f "$REC_DIR/filtered.json.gz" ]] && rec_sha="$(sha256sum "$REC_DIR/filtered.json.gz" | cut -d' ' -f1)"
leftovers=$(find "$REC_DIR" -maxdepth 1 -name '.filter-work-*' | wc -l)

# Питон узла (3.6): кодировку вывода задаём явно — при локали C кириллица иначе не печатается.
PYTHONIOENCODING=utf-8 python3 - "$RUN_DIR/recovery_report.json" <<PY
import json, sys
reference_s = round($t1 - $t0, 1)
recovery_s = round($t3 - $t2, 1)
identical = "$ref_sha" == "$rec_sha"
passed = ($recovered_code == 0) and identical and recovery_s <= $RECOVERY_MAX_S
report = {
    "status": "passed" if passed else "failed",
    "kill_after_s": $KILL_AFTER,
    "interrupted_exit_code": $interrupted_code,
    "reference_run_s": reference_s,
    "recovery_s": recovery_s,
    "recovery_max_s": $RECOVERY_MAX_S,
    "recovered_exit_code": $recovered_code,
    "reference_sha256": "$ref_sha",
    "recovered_sha256": "$rec_sha",
    "identical": identical,
    "leftover_work_dirs_after_failure": $leftovers,
}
with open(sys.argv[1], "w") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print("Штатный запуск: {} с; восстановление (повторный запуск): {} с; порог {} с".format(
    reference_s, recovery_s, $RECOVERY_MAX_S))
print("Результат после восстановления совпадает с эталоном: {}".format("да" if identical else "нет"))
print(report["status"])
sys.exit(0 if passed else 1)
PY
