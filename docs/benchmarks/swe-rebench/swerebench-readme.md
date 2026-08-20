# Запуск и оценка SWE-rebench

Этот документ описывает полный цикл работы с SWE-rebench в данном репозитории: подготовку окружения, фиксацию датасета, запуск инференса, формирование `predictions.jsonl`, официальную оценку и интерпретацию результатов.

Основные точки входа:

- [`snapshot_swe_rebench.py`](../../../snapshot_swe_rebench.py) — создаёт локальный зафиксированный snapshot датасета;
- [`run_swe_rebench.py`](../../../run_swe_rebench.py) — запускает агента в отдельном контейнере для каждой задачи и формирует predictions;
- [`evaluate_swe_rebench.py`](../../../evaluate_swe_rebench.py) — проверяет predictions и передаёт их во внешний evaluator;
- [`SWE-rebench/SWE-bench-fork`](https://github.com/SWE-rebench/SWE-bench-fork) — официальный внешний harness для оценки.

## Содержание

1. [Как устроен запуск](#как-устроен-запуск)
2. [Требования](#требования)
3. [Переменные и их источники](#переменные-и-их-источники)
4. [Установка](#установка)
5. [Настройка модели и pipeline](#настройка-модели-и-pipeline)
6. [Запуск на одной задаче](#запуск-на-одной-задаче)
7. [Запуск на части датасета](#запуск-на-части-датасета)
8. [Запуск на всём датасете](#запуск-на-всём-датасете)
9. [Параметры команд](#параметры-команд)
10. [Форматы данных и артефакты](#форматы-данных-и-артефакты)
11. [Логи](#логи)
12. [Интерпретация оценки](#интерпретация-оценки)
13. [Воспроизводимость](#воспроизводимость)
14. [Типовые ошибки](#типовые-ошибки)

## Как устроен запуск

Полный pipeline состоит из двух независимых стадий:

~~~text
Зафиксированный snapshot датасета
               |
               v
       Выбор набора задач
               |
               v
 Определение Docker-образа каждой задачи
               |
               v
 Отдельный inference-контейнер на задачу
               |
               v
 Агент изменяет репозиторий в /testbed
               |
               v
 Получение git diff в model_patch
               |
               v
         predictions.jsonl
               |
               v
 Отдельный чистый evaluation-контейнер
               |
               v
 Применение model_patch и запуск тестов
               |
               v
       Отчёт официального evaluator
~~~

Для каждой задачи инференса создаётся собственный контейнер из образа этой задачи. Репозиторий внутри контейнера должен находиться на `base_commit`, рабочая директория — `/testbed`. После завершения работы агента runner получает итоговый Git patch, останавливает и удаляет контейнер, если не передан `--keep-containers`.

Evaluation не переиспользует inference-контейнер. Внешний fork создаёт новый чистый контейнер, применяет только `model_patch` и запускает официальные тесты. Поэтому установленные агентом пакеты, временные файлы и другие изменения, не попавшие в patch, не могут повлиять на итоговую оценку.

### Граница ответственности

Этот репозиторий отвечает за:

1. загрузку и фильтрацию задач;
2. создание изолированной среды инференса;
3. запуск pipeline агента;
4. получение patch относительно `base_commit`;
5. запись строгого `predictions.jsonl`;
6. запись metadata, логов и ошибок.

Внешний `SWE-rebench/SWE-bench-fork` отвечает за:

1. создание чистой среды оценки;
2. применение `model_patch`;
3. запуск тестов benchmark;
4. определение `resolved` или `unresolved`;
5. создание отчётов оценки.

## Требования

Перед запуском необходимы:

- Linux;
- Python 3.12;
- Git;
- curl;
- Docker Engine;
- доступ текущего пользователя к Docker daemon;
- свободное место для Docker-образов задач;
- OpenAI-compatible endpoint с запущенной моделью;
- доступ к Hugging Face, если snapshot создаётся из Hub;
- отдельный checkout `SWE-rebench/SWE-bench-fork`.

Проверьте окружение:

~~~bash
python3.12 --version
git --version
docker version
docker info
~~~

Команда `docker info` должна завершаться без ошибки доступа к Docker socket.


## Переменные и их источники

В командах ниже используются три типа переменных:

- **получить и зафиксировать** — значение приходит из внешнего источника и должно оставаться неизменным в рамках эксперимента;
- **выбрать** — значение задаёт пользователь для конкретного запуска;
- **вычислить** — значение однозначно получается из уже заданных переменных.

Фрагмент вида `<значение>` означает шаблон, который нельзя копировать без замены. `$NAME` означает ранее определённую shell-переменную. Имена вида `<run-id>` в деревьях каталогов являются только обозначениями структуры, а не командами.

| Переменная или поле | Тип | Откуда взять | Зачем используется |
| --- | --- | --- | --- |
| `RECIPE_DIR` | Вычислить | `pwd` после перехода в корень клонированного репозитория | Абсолютная база для конфигов, snapshot и результатов |
| `FORK_PARENT` | Выбрать | Любой создаваемый локальный каталог; в примерах — `$RECIPE_DIR/external` | Родительский каталог checkout evaluator |
| `FORK` | Вычислить | `$FORK_PARENT/SWE-bench-fork` | Путь к evaluator |
| `EVALUATOR_COMMIT` | Получить и зафиксировать | Полный SHA из [истории SWE-bench-fork](https://github.com/SWE-rebench/SWE-bench-fork/commits/main) или `git -C "$FORK" rev-parse origin/main` после `git fetch` | Фиксирует реализацию harness, CLI и правила оценки |
| `DATASET` | Выбрать | ID страницы Hub dataset; основной вариант — `nebius/SWE-rebench` | Источник задач при создании snapshot |
| `DATASET_REVISION` | Получить и зафиксировать | Полный SHA из [истории датасета](https://huggingface.co/datasets/nebius/SWE-rebench/commits/main) или через `HfApi.dataset_info(...).sha` | Фиксирует точное содержимое датасета |
| `SPLIT` | Выбрать | Из списка splits выбранной revision; команда приведена ниже | Определяет исходное множество задач |
| `CONFIG` | Выбрать | Один из YAML в `pipeline_configs/`: ReAct или ReWOO | Определяет pipeline, инструменты и параметры модели |
| `MODEL_URL` | Получить из конфига | Поле `url` агента или planner в выбранном YAML | Адрес OpenAI-compatible endpoint |
| `LLM_MODEL` | Получить из конфига и сверить с сервером | Поле `model_name`; проверить через `GET /v1/models` | Модель, которую реально вызывает pipeline |
| `MODEL_NAME_OR_PATH` | Выбрать как метку | Обычно `<pipeline-id>/$LLM_MODEL` | Записывается в predictions; сама по себе модель не переключает |
| `EVAL_NAMESPACE` | Выбрать по evaluator и образам | Правило выбора приведено ниже и в README зафиксированного fork | Управляет поиском или локальной сборкой evaluation images |
| `RUN_ID` | Выбрать | Уникальное понятное имя эксперимента | Связывает predictions, metadata, логи и evaluation reports |
| `RUN_DIR` | Вычислить | `$RECIPE_DIR/runs/$RUN_ID` | Изолирует артефакты одного запуска |
| `TASK_FILE`, `SUBSET_FILE`, `FULL_DATASET` | Вычислить | Файлы внутри `RUN_DIR` | Локальные snapshot для одного, части или полного набора |
| `INSTANCE_IDS_FILE` | Вычислить или создать | Путь внутри `RUN_DIR`; содержимое извлекается из snapshot либо задаётся списком | Ограничивает запуск конкретными задачами |

### Как получить revision датасета

Следующая команда получает полный SHA текущего состояния Hub dataset и сохраняет его в shell. После этого значение не изменяется само по себе в рамках текущего запуска:

~~~bash
export DATASET=nebius/SWE-rebench
export DATASET_REVISION="$(
  python -c 'import os; from huggingface_hub import HfApi; print(HfApi().dataset_info(os.environ["DATASET"]).sha)'
)"

printf 'DATASET_REVISION=%s\n' "$DATASET_REVISION"
~~~

Текущий HEAD подходит для первичной проверки, но не означает автоматически проверенную комбинацию версий. Для повторения ранее проведённого эксперимента задайте сохранённый полный SHA явно. Полный SHA возьмите из metadata предыдущего запуска или истории датасета; не сокращайте его до семи символов.

Посмотрите допустимые splits именно для выбранной revision:

~~~bash
python -c 'import os; from datasets import get_dataset_split_names; print(get_dataset_split_names(os.environ["DATASET"], revision=os.environ["DATASET_REVISION"]))'
~~~

После этого задайте одно из выведенных значений, например:

~~~bash
export SPLIT=filtered
~~~

### Как выбрать namespace

`EVAL_NAMESPACE` — не имя эксперимента. Он определяет, где evaluator ищет instance images.

| Сценарий | Значение |
| --- | --- |
| Датасет и зафиксированный evaluator рассчитаны на локальную сборку образов | `export EVAL_NAMESPACE=""` |
| Используются опубликованные образы leaderboard в namespace `swerebench` | `export EVAL_NAMESPACE=swerebench` |
| Используется собственный registry | Namespace этого registry согласно документации выбранного evaluator |

В [официальном SWE-bench-fork](https://github.com/SWE-rebench/SWE-bench-fork) пример для leaderboard использует `swerebench`, а пример для `nebius/SWE-rebench` — пустой namespace. Поэтому не копируйте `swerebench` автоматически: сначала сопоставьте dataset, commit evaluator и способ получения образов.

### Что именно означает «зафиксировать»

Фиксация не означает, что один SHA нужно использовать всегда. Она означает: выбрать точную версию для конкретного эксперимента, записать её в metadata и повторно использовать при сравнении или продолжении запуска.

- Dataset revision фиксирует состав задач, поля, тестовые данные и ссылки на образы.
- Evaluator commit фиксирует применение patch, команды тестов, обработку timeout и формирование отчёта.
- Commit этого репозитория фиксирует prompt, инструменты, runner и сбор `model_patch`.
- SHA-256 pipeline YAML фиксирует параметры агента.
- Split, список `instance_id` и фильтры `created_at` фиксируют фактический знаменатель метрики.
- Ресурсные лимиты и timeout не являются версиями, но тоже должны сохраняться: они влияют на долю ошибок и незавершённых задач.
- `RUN_ID` не является версией. Новый эксперимент получает новый ID; прежний ID используется только для `--resume`.

## Установка

### 1. Клонирование репозитория

~~~bash
git clone --branch swe-rebench https://github.com/ansiane/swe-rebench-recipe.git
cd swe-rebench-recipe
export RECIPE_DIR="$(pwd)"
~~~

Все последующие команды предполагают запуск из `$RECIPE_DIR`.

### 2. Виртуальное окружение и зависимости

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

### 3. Установка evaluator fork

Для первого запуска можно взять текущий commit `origin/main`, немедленно сохранить его полный SHA и перейти в detached HEAD. Для повторения эксперимента заранее экспортируйте SHA из его metadata: команда ниже сохранит уже заданное значение.

~~~bash
export FORK_PARENT="${FORK_PARENT:-$RECIPE_DIR/external}"
export FORK="$FORK_PARENT/SWE-bench-fork"

mkdir -p "$FORK_PARENT"
git clone https://github.com/SWE-rebench/SWE-bench-fork.git "$FORK"
git -C "$FORK" fetch origin main

export EVALUATOR_COMMIT="${EVALUATOR_COMMIT:-$(git -C "$FORK" rev-parse origin/main)}"
git -C "$FORK" checkout --detach "$EVALUATOR_COMMIT"
python -m pip install -e "$FORK"

printf 'EVALUATOR_COMMIT=%s\n' "$EVALUATOR_COMMIT"
git -C "$FORK" rev-parse HEAD
~~~

Две последние команды должны вывести один и тот же полный SHA. Значение можно также выбрать в [истории коммитов evaluator](https://github.com/SWE-rebench/SWE-bench-fork/commits/main). Последний commit удобен для smoke test, но не гарантирует совместимость с конкретной revision датасета. Для сравнительных запусков используйте заранее проверенную пару `DATASET_REVISION + EVALUATOR_COMMIT`.

Если `$FORK` уже существует, не клонируйте его повторно:

~~~bash
git -C "$FORK" fetch origin main
export EVALUATOR_COMMIT="${EVALUATOR_COMMIT:-$(git -C "$FORK" rev-parse origin/main)}"
git -C "$FORK" checkout --detach "$EVALUATOR_COMMIT"
python -m pip install -e "$FORK"
~~~

## Настройка модели и pipeline

В репозитории есть две базовые конфигурации:

- [`pipeline_configs/react_sgr_swe_rebench.yaml`](../../../pipeline_configs/react_sgr_swe_rebench.yaml);
- [`pipeline_configs/rewoo_sgr_swe_rebench.yaml`](../../../pipeline_configs/rewoo_sgr_swe_rebench.yaml).

Перед запуском проверьте в выбранном YAML:

| Поле | Назначение |
| --- | --- |
| `url` | Базовый URL OpenAI-compatible API, обычно `http://<host>:<port>/v1`; не endpoint `/v1/models` |
| `model_name` | Имя модели, которое принимает endpoint |
| `temperature` | Температура генерации |
| `max_iterations` | Максимальное число итераций ReAct |
| `few_shot_type` | Режим примеров ReAct: `zero_shot`, `cot`, `contrastive_cot` или `auto_cot` |
| `maximum_steps` | Максимальное число шагов плана ReWOO |
| `default_timeout` | Стандартный timeout shell-команды |
| `max_timeout` | Максимально разрешённый timeout shell-команды |
| `logs_path` | Базовая директория логов; может быть переопределена через CLI |

Для ReAct выбирайте `react_sgr_swe_rebench.yaml`, для ReWOO — `rewoo_sgr_swe_rebench.yaml`. Перед изменением рекомендуется скопировать базовый YAML в каталог запуска: тогда файл и его hash останутся рядом с результатами.

Для сопоставимого zero-shot запуска явно фиксируйте `few_shot_type: zero_shot`. Режимы `cot`, `contrastive_cot` и `auto_cot` добавляют разные наборы примеров к исходной задаче и должны рассматриваться как отдельные экспериментальные условия.

Значения `url` и `model_name` в репозитории относятся к конкретному окружению и не должны без проверки переноситься на другой кластер. Получите их из выбранного YAML:

~~~bash
export MODEL_URL="$(
  python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); x=c["components"]; p=(x.get("agent") or x.get("planner") or {}).get("params", {}); print(p["url"])' "$CONFIG"
)"
export LLM_MODEL="$(
  python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); x=c["components"]; p=(x.get("agent") or x.get("planner") or {}).get("params", {}); print(p["model_name"])' "$CONFIG"
)"

printf 'MODEL_URL=%s\nLLM_MODEL=%s\n' "$MODEL_URL" "$LLM_MODEL"
curl -fsS "${MODEL_URL%/}/models" | python -m json.tool
~~~

`MODEL_URL` должен указывать на корень OpenAI-compatible API и обычно оканчивается на `/v1`. Значение вида `http://localhost:11455/v1/models` неверно: `/models` — отдельный ресурс для проверки списка моделей, а OpenAI-клиент сам добавляет к base URL путь `/chat/completions`. Для запущенного через `vllm serve --port 11455` сервера укажите `url: http://localhost:11455/v1`, а список моделей проверяйте отдельной командой `curl http://localhost:11455/v1/models`. `LLM_MODEL` должен совпадать с одним из ID в этом ответе. В ReWOO отдельно проверьте согласованность `url` и `model_name` у `planner`, `worker`, `solver` и `LLM_TOOL`. Реальную модель выбирают поля YAML; аргумент `--model-name-or-path` только маркирует predictions и должен правдиво описывать эту модель.

### Инструменты агента

Конфигурации SWE-rebench предоставляют агенту repository-bound инструменты:

| Инструмент | Назначение |
| --- | --- |
| `list_files` | Просмотр файлов и каталогов репозитория |
| `read_file` | Чтение файла или диапазона строк |
| `search_code` | Поиск текста или регулярного выражения в коде |
| `apply_patch` | Одна структурированная файловая операция: точная замена блока, создание, удаление или перемещение файла |
| `run_command` | Запуск shell-команд и тестов в контейнере задачи |
| `git_diff` | Просмотр изменений относительно `base_commit` |

`apply_patch` сохраняет стабильное имя инструмента, но не требует от модели
генерировать unified diff. Для изменения существующего файла агент передаёт
`operation=replace`, относительный `path`, точный уникальный `old_text` из
текущего файла и `new_text`. Также доступны `create`, `delete` и `move`;
за один вызов выполняется ровно одна операция. Среда проверяет пути, запрещает
неоднозначную замену и возвращает короткую диагностику без неявного fuzzy-edit.
Итоговый `model_patch` по-прежнему собирается через `git diff --binary`, поэтому
формат predictions и официальный evaluator не меняются. Описание и JSON-схема
инструмента формируются из фактически подключённого компонента YAML.


## Запуск на одной задаче

Ниже приведён полный сценарий: от загрузки одной записи до официальной оценки.

### 1. Переменные запуска

Пример ниже получает текущую revision датасета, извлекает endpoint и ID модели из ReAct-конфига и создаёт уникальный каталог smoke test. Для повторного запуска замените автоматически полученную revision значением из metadata исходного эксперимента.

~~~bash
cd "$RECIPE_DIR"
source .venv/bin/activate

export DATASET=nebius/SWE-rebench
export DATASET_REVISION="${DATASET_REVISION:-$(
  python -c 'import os; from huggingface_hub import HfApi; print(HfApi().dataset_info(os.environ["DATASET"]).sha)'
)}"
export SPLIT=filtered

export CONFIG="$RECIPE_DIR/pipeline_configs/react_sgr_swe_rebench.yaml"
export PIPELINE_ID=react-sgr
export MODEL_URL="$(
  python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); x=c["components"]; p=(x.get("agent") or x.get("planner") or {}).get("params", {}); print(p["url"])' "$CONFIG"
)"
export LLM_MODEL="$(
  python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding="utf-8")); x=c["components"]; p=(x.get("agent") or x.get("planner") or {}).get("params", {}); print(p["model_name"])' "$CONFIG"
)"
export MODEL_NAME_OR_PATH="$PIPELINE_ID/$LLM_MODEL"

export EVAL_NAMESPACE="${EVAL_NAMESPACE:-}"
export RUN_ID="${PIPELINE_ID}-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"
export TASK_FILE="$RUN_DIR/task.jsonl"
export INSTANCE_IDS_FILE="$RUN_DIR/instance_ids.txt"

mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/pipeline.yaml"
export CONFIG="$RUN_DIR/pipeline.yaml"
~~~

Проверьте, что `SPLIT` существует в выбранной revision и что `EVAL_NAMESPACE` соответствует образам зафиксированного evaluator. Один и тот же локальный `$TASK_FILE` далее передаётся инференсу и evaluation. Новый `RUN_ID` создавайте для каждого независимого эксперимента; при `--resume` используйте прежний.

### 2. Проверка параметров перед запуском

Перед долгим запуском распечатайте все значения, пришедшие извне, и проверьте, что пути существуют:

~~~bash
printf '%-24s %s\n' \
  RECIPE_DIR "$RECIPE_DIR" \
  FORK "$FORK" \
  EVALUATOR_COMMIT "$EVALUATOR_COMMIT" \
  DATASET "$DATASET" \
  DATASET_REVISION "$DATASET_REVISION" \
  SPLIT "$SPLIT" \
  CONFIG "$CONFIG" \
  MODEL_URL "$MODEL_URL" \
  LLM_MODEL "$LLM_MODEL" \
  MODEL_NAME_OR_PATH "$MODEL_NAME_OR_PATH" \
  EVAL_NAMESPACE "${EVAL_NAMESPACE:-[пусто]}" \
  RUN_ID "$RUN_ID" \
  RUN_DIR "$RUN_DIR"

test -d "$RECIPE_DIR/.git"
test -d "$FORK/.git"
test -f "$CONFIG"
test "$(git -C "$FORK" rev-parse HEAD)" = "$EVALUATOR_COMMIT"
git -C "$RECIPE_DIR" status --short
~~~

После получения predictions выполните безопасную проверку wrapper без запуска evaluation-контейнеров:

~~~bash
python evaluate_swe_rebench.py \
  --fork-path "$FORK" \
  --predictions-path "$RUN_DIR/predictions.jsonl" \
  --dataset-name "$TASK_FILE" \
  --split "$SPLIT" \
  --run-id "$RUN_ID" \
  --namespace "$EVAL_NAMESPACE" \
  --inference-metadata "$RUN_DIR/run_metadata.json" \
  --check-only
~~~

### 3. Формирование одного примера

Чтобы взять первую задачу выбранного split:

~~~bash
python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --limit 1   --output "$TASK_FILE"
~~~

Команда создаст:

- `$TASK_FILE` — одну полную запись датасета в JSONL;
- `$TASK_FILE.metadata.json` — источник, revision, SHA-256 snapshot и выбранный `instance_id`.

Получите ID выбранной задачи:

~~~bash
python -c 'import json,sys; print(json.loads(open(sys.argv[1], encoding="utf-8").readline())["instance_id"])'   "$TASK_FILE" > "$INSTANCE_IDS_FILE"

cat "$INSTANCE_IDS_FILE"
~~~

Чтобы выбрать не первую, а конкретную задачу, заранее запишите её ID в файл:

~~~bash
printf '%s\n' 'PennyLaneAI__pennylane-7671' > "$INSTANCE_IDS_FILE"

python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --instance-ids-file "$INSTANCE_IDS_FILE"   --output "$TASK_FILE"
~~~

### 4. Инференс на одной задаче

~~~bash
python run_swe_rebench.py   --config "$CONFIG"   --dataset "$TASK_FILE"   --split "$SPLIT"   --instance-ids-file "$INSTANCE_IDS_FILE"   --output "$RUN_DIR/predictions.jsonl"   --model-name-or-path "$MODEL_NAME_OR_PATH"   --run-id "$RUN_ID"   --num-workers 1   --pull-policy missing   --task-timeout 3600   --memory-limit 16g   --nano-cpus 4000000000   --logs-path "$RUN_DIR/logs"   --evaluator-fork-path "$FORK"
~~~

`--nano-cpus 4000000000` соответствует четырём CPU. Значения CPU, памяти и timeout нужно адаптировать под кластер и задачи.

При успешном инференсе в `$RUN_DIR` появятся:

~~~text
predictions.jsonl
run_metadata.json
patches.jsonl
logs/
└── react-sgr-one/
    ├── log_main.log
    └── log_process_swe-rebench-*.log
~~~

`errors.jsonl` создаётся, если хотя бы одна задача завершилась ошибкой.

Проверьте prediction:

~~~bash
python -m json.tool "$RUN_DIR/run_metadata.json"
sed -n '1p' "$RUN_DIR/predictions.jsonl"
~~~

### 5. Оценка одной задачи

Прочитайте `instance_id` в Bash-массив и запустите evaluator:

~~~bash
mapfile -t INSTANCE_IDS < "$INSTANCE_IDS_FILE"

python evaluate_swe_rebench.py   --fork-path "$FORK"   --predictions-path "$RUN_DIR/predictions.jsonl"   --dataset-name "$TASK_FILE"   --split "$SPLIT"   --run-id "$RUN_ID"   --max-workers 1   --timeout 1800   --namespace "$EVAL_NAMESPACE"   --inference-metadata "$RUN_DIR/run_metadata.json"   --instance-ids "${INSTANCE_IDS[@]}"   --report-dir "$RUN_DIR/evaluation"
~~~

`--namespace` обязателен для wrapper, но его значение может быть пустой строкой. Используйте namespace образов из зафиксированной версии evaluator: для `nebius/SWE-rebench` официальный пример использует `--namespace ""`, а `swerebench` относится к опубликованным leaderboard images. Не меняйте namespace между проверкой одной задачи и полным запуском.

## Запуск на части датасета

Сначала выполните установку и задайте базовые переменные из раздела запуска на одной задаче, затем создайте новый `RUN_ID` для подвыборки.

Для воспроизводимости рекомендуется сначала создать отдельный snapshot выбранной части, а затем использовать этот же файл на стадиях инференса и оценки.

### Фильтрация по дате создания задачи

В [Hub dataset](https://huggingface.co/datasets/nebius/SWE-rebench/viewer/default/filtered) поле `created_at` хранит дату создания задачи. В указанном URL `filtered` — название split, поэтому для него передавайте `--split filtered`.

Snapshot поддерживает полуоткрытый UTC-интервал:

```text
created_at_from <= created_at < created_at_before
```

- `--created-at-from` задаёт включённую нижнюю границу;
- `--created-at-before` задаёт исключённую верхнюю границу;
- обе границы необязательны;
- дата без времени означает полночь UTC;
- дата без timezone считается UTC;
- фильтр по дате применяется до `--start`, `--stop` и `--limit`.

Пример выбора задач, созданных в течение 2024 года:

```bash
export RUN_ID=react-sgr-created-in-2024
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"
export SUBSET_FILE="$RUN_DIR/tasks.jsonl"
mkdir -p "$RUN_DIR"

python snapshot_swe_rebench.py \
  --dataset nebius/SWE-rebench \
  --split filtered \
  --dataset-revision "$DATASET_REVISION" \
  --created-at-from 2024-01-01 \
  --created-at-before 2025-01-01 \
  --output "$SUBSET_FILE"
```

Можно задать только одну границу:

```bash
# Все задачи начиная с 1 июля 2024 года включительно
--created-at-from 2024-07-01

# Все задачи строго раньше 1 января 2023 года
--created-at-before 2023-01-01
```

Поддерживаются значения `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS` и ISO 8601 с `Z` или timezone offset. Нормализованные границы и количества записей до и после фильтра сохраняются в `<snapshot>.metadata.json`.

Полученный snapshot далее передаётся в неизменные команды `run_swe_rebench.py --dataset "$SUBSET_FILE"` и `evaluate_swe_rebench.py --dataset-name "$SUBSET_FILE"`.

### Вариант 1: явный список instance ID

Создайте список:

~~~bash
export RUN_ID=react-sgr-subset
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"
export SUBSET_FILE="$RUN_DIR/subset.jsonl"
export INSTANCE_IDS_FILE="$RUN_DIR/instance_ids.txt"
mkdir -p "$RUN_DIR"

printf '%s
'   'owner__repo-123'   'owner__repo-456'   'owner__repo-789' > "$INSTANCE_IDS_FILE"
~~~

Создайте snapshot:

~~~bash
python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --instance-ids-file "$INSTANCE_IDS_FILE"   --output "$SUBSET_FILE"
~~~

Запустите инференс:

~~~bash
python run_swe_rebench.py   --config "$CONFIG"   --dataset "$SUBSET_FILE"   --split "$SPLIT"   --instance-ids-file "$INSTANCE_IDS_FILE"   --output "$RUN_DIR/predictions.jsonl"   --model-name-or-path "$MODEL_NAME_OR_PATH"   --run-id "$RUN_ID"   --num-workers 2   --pull-policy missing   --task-timeout 3600   --logs-path "$RUN_DIR/logs"   --evaluator-fork-path "$FORK"
~~~

Запустите оценку:

~~~bash
mapfile -t INSTANCE_IDS < "$INSTANCE_IDS_FILE"

python evaluate_swe_rebench.py   --fork-path "$FORK"   --predictions-path "$RUN_DIR/predictions.jsonl"   --dataset-name "$SUBSET_FILE"   --split "$SPLIT"   --run-id "$RUN_ID"   --max-workers 2   --timeout 1800   --namespace "$EVAL_NAMESPACE"   --inference-metadata "$RUN_DIR/run_metadata.json"   --instance-ids "${INSTANCE_IDS[@]}"   --report-dir "$RUN_DIR/evaluation"
~~~

### Вариант 2: диапазон задач

`--start` включается в диапазон, `--stop` не включается:

~~~bash
python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --start 100   --stop 110   --output "$RUN_DIR/subset.jsonl"
~~~

Этот пример выбирает позиции с 100 по 109.

### Вариант 3: первые N задач после фильтрации

~~~bash
python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --limit 10   --output "$RUN_DIR/subset.jsonl"
~~~

Если snapshot уже содержит только нужные записи, дополнительные `--start`, `--stop` и `--limit` при инференсе обычно не требуются. Не применяйте разные фильтры на стадиях snapshot, inference и evaluation: это может привести к несовпадению ID.

## Запуск на всём датасете

Сначала выполните установку и задайте базовые переменные из раздела запуска на одной задаче, затем создайте новый `RUN_ID` для полного эксперимента.

Полный запуск выполняйте только после успешного запуска на одной задаче и небольшой подвыборке.

### 1. Полный snapshot

~~~bash
export RUN_ID=react-sgr-full
export RUN_DIR="$RECIPE_DIR/runs/$RUN_ID"
export FULL_DATASET="$RUN_DIR/swe-rebench.jsonl"
mkdir -p "$RUN_DIR"

python snapshot_swe_rebench.py   --dataset "$DATASET"   --split "$SPLIT"   --dataset-revision "$DATASET_REVISION"   --output "$FULL_DATASET"
~~~

### 2. Полный инференс

~~~bash
python run_swe_rebench.py   --config "$CONFIG"   --dataset "$FULL_DATASET"   --split "$SPLIT"   --output "$RUN_DIR/predictions.jsonl"   --model-name-or-path "$MODEL_NAME_OR_PATH"   --run-id "$RUN_ID"   --num-workers 8   --pull-policy missing   --task-timeout 3600   --logs-path "$RUN_DIR/logs"   --evaluator-fork-path "$FORK"
~~~

Выбирайте `--num-workers` с учётом пропускной способности model endpoint, CPU, RAM, Docker и доступного диска. Каждый worker может одновременно владеть отдельным контейнером задачи.

Если запуск был прерван, повторите ту же команду с `--resume`. Runner пропустит `instance_id`, уже записанные в `predictions.jsonl`. Для продолжения логического эксперимента используйте прежний `run_id` и тот же output.

### 3. Полная оценка

Для полного snapshot `--instance-ids` не передаётся:

~~~bash
python evaluate_swe_rebench.py   --fork-path "$FORK"   --predictions-path "$RUN_DIR/predictions.jsonl"   --dataset-name "$FULL_DATASET"   --split "$SPLIT"   --run-id "$RUN_ID"   --max-workers 8   --timeout 1800   --namespace "$EVAL_NAMESPACE"   --inference-metadata "$RUN_DIR/run_metadata.json"   --report-dir "$RUN_DIR/evaluation"
~~~

## Параметры команд

Актуальный источник параметров — вывод `--help` соответствующего скрипта.

### `snapshot_swe_rebench.py`

| Параметр | Обязательный | По умолчанию | Назначение |
| --- | ---: | --- | --- |
| `--dataset` | Да | — | Имя Hub dataset либо путь к локальному JSON/JSONL |
| `--split` | Да | — | Split датасета |
| `--dataset-revision` | Для Hub | — | Immutable revision или commit SHA Hub dataset |
| `--created-at-from` | Нет | — | Включает задачи с `created_at` не раньше указанной UTC-даты |
| `--created-at-before` | Нет | — | Включает задачи с `created_at` строго раньше указанной UTC-даты |
| `--instance-ids-file` | Нет | — | Файл с выбранными `instance_id`, по одному на строку |
| `--start` | Нет | `0` | Начальная позиция выборки, включительно |
| `--stop` | Нет | до конца | Конечная позиция выборки, не включительно |
| `--limit` | Нет | без ограничения | Максимальное число записей после фильтрации |
| `--output` | Да | — | Путь к итоговому зафиксированному JSONL |

Для локального JSON/JSONL `--dataset-revision` не требуется. Для Hub dataset отсутствие immutable revision является ошибкой.

### `run_swe_rebench.py`

| Параметр | Обязательный | По умолчанию | Назначение |
| --- | ---: | --- | --- |
| `--config` | Да | — | YAML-конфигурация pipeline |
| `--dataset` | Да | — | Hub dataset или локальный snapshot JSON/JSONL |
| `--split` | Да | — | Split |
| `--dataset-revision` | Нет | — | Revision Hub dataset; для локального snapshot обычно не задаётся |
| `--instance-ids-file` | Нет | — | Явный список задач |
| `--start` | Нет | `0` | Начальная позиция |
| `--stop` | Нет | до конца | Конечная позиция, не включительно |
| `--limit` | Нет | без ограничения | Максимальное число задач |
| `--output` | Да | — | Путь к `predictions.jsonl` |
| `--model-name-or-path` | Да | — | Идентификатор комбинации pipeline и модели в prediction |
| `--run-id` | Да | — | Идентификатор логического запуска |
| `--num-workers` | Нет | `1` | Число параллельных задач инференса |
| `--namespace` | Нет | — | Namespace для convention fallback образов |
| `--architecture` | Нет | `x86_64` | Архитектура образов |
| `--image-tag` | Нет | `latest` | Тег образа для convention fallback |
| `--image-manifest` | Нет | — | JSON manifest с точными образами задач |
| `--allow-image-convention` | Нет | выключен | Разрешает вычисление имени образа по convention; только для локальной отладки |
| `--pull-policy` | Нет | `missing` | `always`, `missing` или `never` |
| `--memory-limit` | Нет | без ограничения | Docker memory limit, например `16g` |
| `--nano-cpus` | Нет | без ограничения | CPU limit в nano-CPU; `1000000000` = один CPU |
| `--allow-network` | Нет | выключен | Разрешает сеть внутри inference-контейнера |
| `--keep-containers` | Нет | выключен | Не удаляет контейнеры после задачи; только для диагностики |
| `--include-hints` | Нет | выключен | Добавляет `hints_text` в prompt |
| `--resume` | Нет | выключен | Пропускает уже записанные predictions |
| `--fail-fast` | Нет | выключен | Прекращает отправку новых задач после первой ошибки |
| `--task-timeout` | Нет | `3600` | Максимальное время одной задачи, секунд |
| `--logs-path` | Нет | YAML или `<output-dir>/logs` | Базовая директория или имя файла логов |
| `--evaluator-fork-path` | Нет | — | Путь к fork для записи его commit SHA в metadata |

Production-запуск должен использовать точные образы из dataset или manifest. `--allow-image-convention` не предназначен для публикуемых результатов.

### `evaluate_swe_rebench.py`

| Параметр | Обязательный | По умолчанию | Назначение |
| --- | ---: | --- | --- |
| `--fork-path` | Да | — | Путь к checkout `SWE-bench-fork` |
| `--predictions-path` | Да | — | JSONL с predictions |
| `--dataset-name` | Да | — | Тот же dataset или локальный snapshot, что использовался в inference |
| `--dataset-revision` | Нет | — | Revision dataset; должна совпадать с inference metadata |
| `--split` | Да | — | Split; должен совпадать с inference |
| `--run-id` | Да | — | Идентификатор оценки |
| `--max-workers` | Нет | `4` | Число параллельных evaluation-контейнеров |
| `--timeout` | Нет | `1800` | Timeout одной evaluation-задачи, секунд |
| `--namespace` | Да | — | Namespace образов; допускается пустая строка |
| `--inference-metadata` | Да | — | `run_metadata.json` соответствующего inference |
| `--instance-image-tag` | Нет | `latest` | Тег instance image |
| `--report-dir` | Нет | `swe-rebench-evaluation` | Директория нормализованных отчётов |
| `--instance-ids` | Нет | все | Список выбранных ID после одного флага |
| `--check-only` | Нет | выключен | Проверяет JSONL и совместимость CLI без запуска evaluation-контейнеров |

## Форматы данных и артефакты

### Поля задачи

Inference loader использует:

| Поле | Обязательное | Назначение |
| --- | ---: | --- |
| `instance_id` | Да | Стабильный идентификатор задачи |
| `repo` | Да | Репозиторий в формате `owner/name` |
| `base_commit` | Да | Исходный commit рабочего дерева |
| `created_at` | Для фильтра по дате | Дата создания задачи; используется snapshot и не добавляется в prompt |
| `problem_statement` | Да | Описание issue для агента |
| `hints_text` | Нет | Необязательные подсказки |
| `version` | Нет | Версия проекта |
| `image_name` или `docker_image` | Для готового образа | Точный образ конкретной задачи |
| `install_config` | Нет | Допустимые настройки запуска контейнера, например `cap_add` |

Snapshot сохраняет evaluator-only поля `patch`, `test_patch`, `FAIL_TO_PASS` и `PASS_TO_PASS`, потому что они нужны официальной оценке. Inference loader намеренно удаляет их перед построением task и prompt. Эти поля нельзя показывать агенту: это утечка эталонного решения и тестовой информации.

### Формат prediction

`predictions.jsonl` содержит ровно один JSON-объект на строку:

~~~json
{
  "instance_id": "owner__repo-123",
  "model_name_or_path": "react-sgr/model-name",
  "model_patch": "diff --git a/src/file.py b/src/file.py\n..."
}
~~~

Требования:

- каждый выбранный `instance_id` встречается ровно один раз;
- `model_name_or_path` — непустая строка;
- `model_patch` — Git patch относительно `base_commit`;
- при timeout, пустом решении или infrastructure failure runner записывает пустой patch, чтобы задача не исчезла из знаменателя оценки.

Текстовый ответ агента, tool trace, timings и ошибки не включаются в prediction.

### Структура запуска

При output `runs/<run-id>/predictions.jsonl` директория может содержать:

~~~text
runs/<run-id>/
├── predictions.jsonl
├── run_metadata.json
├── errors.jsonl
├── patches.jsonl
├── task.jsonl
├── task.jsonl.metadata.json
├── instance_ids.txt
├── logs/
│   └── <run-id>/
│       ├── log_main.log
│       └── log_process_swe-rebench-*.log
└── evaluation/
    ├── evaluation_metadata.json
    ├── <aggregate-report>.json
    └── instances/
        └── ...
~~~

Некоторые файлы создаются только при наличии соответствующих событий. Например, `errors.jsonl` отсутствует, если ошибок не было.

| Артефакт | Назначение |
| --- | --- |
| `predictions.jsonl` | Единственный основной вход evaluator |
| `run_metadata.json` | Dataset, pipeline, модель, commits, config hash и выбранные ID |
| `patches.jsonl` | Диагностика собранных patch |
| `errors.jsonl` | Ошибки inference, traceback, duration и bounded diagnostics |
| `evaluation_metadata.json` | Точная команда evaluator и версии компонентов |
| aggregate report | Общая статистика resolved/unresolved/error |
| `evaluation/instances` | Отчёты и логи отдельных задач evaluator |

## Логи

`--logs-path` переопределяет `logs_path` из pipeline YAML. Для каждого `run_id` создаётся отдельная поддиректория:

~~~text
<logs-path>/
└── <run-id>/
    ├── log_main.log
    └── log_process_swe-rebench-*.log
~~~

`log_main.log` содержит события всего запуска: загрузку датасета, выбор задач, разрешение образов, отправку workers, timeout и итоговый summary.

`log_process_*.log` содержит lifecycle отдельного worker, сообщения pipeline и агента, создание контейнера и сбор patch. Для ReAct SGR в нём также записываются `LLM_REQUEST`, сырой `LLM_RESPONSE` (не более 8000 символов), `LLM_REQUEST_FAILED`, `LLM_RESPONSE_INVALID` и `LLM_RETRY`. Поэтому HTTP-ошибка endpoint или невалидный JSON видны до появления общего `EmptyModelPatchError`.

Повторный запуск с `--resume` и тем же `run_id` продолжает тот же логический эксперимент и дописывает его логи. Для независимого эксперимента используйте новый `run_id`.

## Интерпретация оценки

Основные поля итогового отчёта:

| Поле | Значение |
| --- | --- |
| `total_instances` | Общее число задач в отчёте |
| `submitted_instances` | Число predictions, переданных evaluator |
| `completed_instances` | Число задач с завершившейся оценкой |
| `resolved_instances` | Число задач, прошедших официальные тесты |
| `unresolved_instances` | Оценка завершилась, но patch не решил задачу |
| `empty_patch_instances` | Задачи с пустым `model_patch` |
| `error_instances` | Задачи, для которых evaluator завершился ошибкой |
| `resolved_ids` | ID решённых задач |
| `unresolved_ids` | ID нерешённых задач |
| `empty_patch_ids` | ID задач без patch |
| `error_ids` | ID задач с ошибкой оценки |

`resolved` означает, что patch был применён в чистой среде и прошёл критерии официального evaluator. `unresolved` не является infrastructure error: evaluation завершилась, но тесты не подтвердили решение.

## Воспроизводимость

Версии фиксируются не ради формальности. Без `DATASET_REVISION` повторная команда может получить другой состав или содержимое задач. Без `EVALUATOR_COMMIT` те же predictions могут пройти через изменившиеся правила применения patch, тестирования, timeout или формирования отчёта. Фиксация означает неизменность внутри одного эксперимента, а не запрет обновлять компоненты в следующих экспериментах.

Для каждого эксперимента сохраните:

- immutable revision исходного датасета;
- локальный snapshot и его `.metadata.json`;
- SHA-256 snapshot;
- commit этого репозитория;
- commit `SWE-bench-fork`;
- pipeline YAML и его SHA-256;
- точное имя модели и endpoint deployment;
- `run_id`;
- список `instance_id`;
- `predictions.jsonl`;
- inference и evaluation metadata;
- логи и отчёты.

Перед evaluation убедитесь, что `dataset`, `split`, `dataset_revision` и выбранные ID совпадают с `run_metadata.json`. Wrapper выполняет эту проверку автоматически.

## Типовые ошибки

### Docker daemon недоступен

Признаки: `Permission denied`, `Cannot connect to the Docker daemon`.

Проверьте:

~~~bash
docker info
~~~

Исправьте права доступа к Docker в соответствии с политикой сервера.

### Не найден образ задачи

Проверьте `image_name` или `docker_image` в snapshot, registry login, `--pull-policy` и доступность образа для нужной архитектуры. Для воспроизводимого запуска не подменяйте отсутствующий образ convention fallback без фиксации причины.

### `EmptyModelPatchError`

Агент завершился, но рабочее дерево не изменилось. Проверьте:

- `errors.jsonl`;
- `details.agent_result`;
- `details.git_status`;
- `details.diff_summary`;
- process log задачи;
- число итераций и историю агента;
- действительно ли агент вызвал `apply_patch` с одной из операций
  `replace/create/delete/move`.

### ReAct завершился по `retry_limit`

Если между `PIPELINE_STARTED` и ошибкой проходит доля секунды и ни один инструмент не вызван, откройте process log и найдите `LLM_REQUEST_FAILED`. В первую очередь проверьте `base_url`:

~~~yaml
url: http://localhost:11455/v1  # правильно
# url: http://localhost:11455/v1/models  # неправильно
~~~

`max_iterations` ограничивает все шаги ReAct. Последовательные ошибки запроса, разбора или валидации ответа ограничены исходной константой `AgentConfig.MAX_RETRY_COUNT = 5`, поэтому агент может остановиться раньше, чем исчерпает `max_iterations`. Конкретная техническая причина сохраняется в process log, но модели при повторе передаётся исходное универсальное сообщение об ошибке, чтобы не изменять экспериментальную логику.

### Timeout задачи

Увеличьте `--task-timeout`, timeout модели или `RUN_COMMAND_TOOL.default_timeout`. Сначала выясните, зависла модель, тесты или Docker operation.

### `predictions.jsonl` уже существует

Без `--resume` runner не перезаписывает непустой файл. Используйте новый `RUN_DIR` для нового эксперимента либо `--resume` для продолжения того же запуска.

### Не совпадают instance ID

Используйте один snapshot и один `instance_ids.txt` для inference и evaluation. Не создавайте prediction из одного snapshot, а evaluation — из другого.

### Не совпадают dataset, split или revision

Значения в команде evaluation должны совпадать с `run_metadata.json`. Для локального snapshot обычно не передавайте `--dataset-revision` ни inference, ни evaluation.

### Evaluator CLI несовместим

Зафиксированная версия fork должна поддерживать параметры `--dataset_name`, `--split`, `--predictions_path`, `--max_workers` и `--run_id`. Зафиксируйте правильный commit fork или синхронно обновите wrapper и документацию.

### Evaluation завершилась без отчёта

Проверьте return code evaluator, директорию `$FORK/logs/run_evaluation/$RUN_ID`, aggregate JSON в checkout fork и `--report-dir`. Wrapper считает отсутствие отчёта после успешного процесса ошибкой.

### Неверный namespace

Используйте namespace, соответствующий выбранному registry и commit evaluator. Пустой namespace допустим, но означает использование режима локально собираемых образов в поддерживающем его fork.

## Рекомендации перед полным запуском

1. Зафиксируйте revision датасета и evaluator fork.
2. Запустите одну задачу от snapshot до официального отчёта.
3. Проверьте, что patch непустой и применим.
4. Запустите подвыборку из 5–10 задач.
5. Оцените среднее время, использование диска, CPU, RAM и нагрузку на model endpoint.
6. Только после этого увеличивайте `--num-workers` и запускайте весь dataset.
7. Архивируйте snapshot, configs, metadata, predictions, логи и evaluation reports вместе.

Проверка документации и параметров перед публикацией запуска:

- все shell-переменные либо заданы в копируемом блоке, либо отмечены как вычисляемые;
- каждый шаблон в угловых скобках объяснён как обозначение, а не готовое значение;
- dataset и evaluator представлены полными SHA, а не названиями плавающих веток;
- выбранный split существует в зафиксированной revision;
- namespace соответствует источнику evaluation images;
- `MODEL_URL` доступен, а `LLM_MODEL` присутствует в ответе `/v1/models`;
- `run_metadata.json` содержит dataset, split, commits, config hash, выбранные ID и каталог логов.
