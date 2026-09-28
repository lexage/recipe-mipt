#!/bin/bash
# Окружение для прогона бенчмарка на кластере.
#
# Питон на самом узле 3.6, поэтому всё живёт в контейнере с образом asllm.
# В образе уже есть numpy, pandas, scipy, sklearn, torch, transformers, openai,
# networkx, tqdm, psutil, datasets. Не хватает четырёх пакетов:
#   qdrant-client, chromadb, fastembed — без них не импортируется LocalDB
#     (qdrant-адаптер держит разреженные векторы через fastembed);
#   matplotlib, tensorflow-cpu — без них падают задачи этих семейств DS-1000.
#
# venv создаётся с --system-site-packages поверх образа: тяжёлое (torch и
# прочее) берётся из образа, доставленное лежит в каталоге на узле и
# переживает перезапуск контейнера.
#
#   bash scripts/setup_runner_env.sh
#
# venv по умолчанию — venv/ в корне компонента. В контейнер монтируется только
# MOUNT_ROOT (по умолчанию корень компонента): и компонент, и venv должны лежать
# внутри него. Скрипты испытаний передают venv и рабочий каталог испытаний:
#   VENV=<рабочий каталог>/venv MOUNT_ROOT=<рабочий каталог> bash scripts/setup_runner_env.sh
#
# Индекс pip. В образе зашит внутренний индекс, требующий пароля, и задан он в
# профиле, поэтому переменной окружения его не перебить: логин-шелл
# перечитывает профиль, и pip уходит спрашивать логин. Здесь индекс задаётся
# флагом командной строки — этот приоритет выше, — а логин-шелл в контейнере не
# используется. Зеркала перебираются по порядку, берётся первое живое.
# Свой список: PIP_INDEXES="https://.../simple" bash scripts/setup_runner_env.sh
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/venv}"
MOUNT_ROOT="${MOUNT_ROOT:-$REPO}"
IMG="${IMG_ASLLM:-10.0.117.197:5000/asllm:1.10.3-pytorch2.10.0-ubuntu24.04-sail2.1.0-cuda13.0-sglang0.5.12-vllm0.20.1-py312}"
PIP_INDEXES="${PIP_INDEXES:-https://mirrors.aliyun.com/pypi/simple/ https://pypi.org/simple https://pypi.tuna.tsinghua.edu.cn/simple}"
PACKAGES="${PACKAGES:-qdrant-client chromadb fastembed matplotlib tensorflow-cpu}"
# numpy закреплён на образной версии: под неё собраны sklearn, scipy и
# tensorflow, а pip иначе тянет numpy 2.x как зависимость и ломает их импорт.
CONSTRAINTS="${CONSTRAINTS:-scripts/constraints.txt}"
# Пересобрать venv с нуля: RECREATE=1 bash scripts/setup_runner_env.sh
RECREATE="${RECREATE:-0}"

MOUNT_ROOT="$(cd "$MOUNT_ROOT" && pwd)"
for path in "$REPO" "$VENV"; do
    case "$(readlink -m "$path")/" in
        "$(readlink -f "$MOUNT_ROOT")"/*) ;;
        *) echo "путь $path вне $MOUNT_ROOT: в контейнер монтируется только MOUNT_ROOT" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$VENV")"

docker run --rm --network host \
  -v "$MOUNT_ROOT:$MOUNT_ROOT" -w "$REPO" \
  -e VENV="$VENV" -e PIP_INDEXES="$PIP_INDEXES" -e PACKAGES="$PACKAGES" -e CONSTRAINTS="$CONSTRAINTS" -e RECREATE="$RECREATE" \
  "$IMG" bash -c '
set -eu
if [ "$RECREATE" = 1 ]; then rm -rf "$VENV"; fi
test -d "$VENV" || python3 -m venv --system-site-packages "$VENV"
PIP="$VENV/bin/pip"

echo "=== доступность зеркал ==="
for index in $PIP_INDEXES; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 15 "$index" || echo "нет-ответа")
    echo "  $code  $index"
done

chosen=""
for index in $PIP_INDEXES; do
    host=$(printf "%s" "$index" | sed -E "s#^https?://([^/]+)/.*#\1#")
    echo "=== ставлю с $index ==="
    if "$PIP" install --no-input --no-cache-dir --disable-pip-version-check \
            --retries 1 --timeout 30 --index-url "$index" --trusted-host "$host" \
            --constraint "$CONSTRAINTS" $PACKAGES; then
        chosen="$index"
        break
    fi
    echo "--- не вышло с $index, пробую следующее ---"
done

test -n "$chosen" || { echo "НИ ОДНО ЗЕРКАЛО НЕ ОТРАБОТАЛО" >&2; exit 1; }
echo "=== поставлено с $chosen ==="
"$VENV/bin/python" scripts/check_env.py
'

echo
echo "venv готов: $VENV"
