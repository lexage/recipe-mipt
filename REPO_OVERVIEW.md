# recipe-mipt — обзор репозитория для быстрого погружения

Файл-шпаргалка для Claude Code и нового разработчика. Описывает структуру репозитория, как запускаются эксперименты на бенчмарке DS1000, как включать/выключать модули фильтрации и генерации, а также перечисляет реализованные методы.

> Стек: Python 3.12+, OpenAI-совместимый клиент к vLLM, pydantic, chromadb/qdrant, sentence-transformers, scikit-learn, networkx (для DAG-сборки пайплайна). Зависимости — `pyproject.toml` / `requirements.txt`.

---

## 1. Структура репозитория

Корневые файлы и каталоги:

```
recipe-mipt/
├── run.py                  # Запуск пайплайна на одном запросе (-c config.yaml -q "query")
├── run_ds1000.py           # Прогон директории YAML-конфигов на бенчмарке DS1000
├── rag_eval.py             # Программный (без YAML) пример прогона на DS1000
├── pyproject.toml          # Зависимости проекта (uv/pip)
├── requirements.txt        # Альтернативный список зависимостей
├── .env                    # Переменные окружения (ключи и т.п.)
│
├── pipeline_configs/       # YAML-конфиги пайплайнов (simple, corag, maps, mars, panel, react, rewoo, iccl, instruct, api_selector)
├── data/                   # Данные: ds1000, vector DB, эмбеддинги, аннотации
├── results/                # Сохранённые результаты прогонов (создаются автоматически)
├── docs/                   # Документация по модулям (filtration, rag, icl, agents, services, benchmarks)
├── doc_parser/             # Парсер документации в БД
├── dtools/                 # Dockerfile, docker_compose, build/run скрипты для агентов и сервисов
├── services/               # vLLM, БД и сторонние сервисы (Docker-окружение)
├── scripts/                # Вспомогательные скрипты (torchmetrics_benchmark, и т.д.)
├── tests/                  # Тесты
│
└── src/                    # Весь код проекта
    ├── client.py
    ├── pipelines/                     # ✦ Сборщик пайплайнов из YAML
    │   ├── pipeline_builder.py        #   PipelineBuilder — строит DAG и собирает пайплайн
    │   ├── configs.py                 #   ConfigLoader, PipelineConfig, ComponentConfig
    │   ├── registry.py                #   ComponentRegistry — ленивый импорт компонент
    │   ├── factory.py                 #   ComponentFactory — создаёт инстансы
    │   ├── constants.py               #   ⭐ ComponentNames / PipelinesNames — каталог всех модулей
    │   └── templates/                 #   Шаблоны пайплайнов (Pipeline-наследники)
    │       ├── simple_pipeline.py     #     SimplePipeline (универсальный, см. ниже)
    │       ├── maps_pipeline.py       #     MAPS, MARS, REWOO, REACT, PANEL
    │       ├── mars_pipeline.py
    │       ├── rewoo_pipeline.py
    │       └── …
    │
    ├── agent_constructor/             # ✦ Базовые абстракции «конструктора агента»
    │   ├── core.py                    #   Block, Document, Chunk, Text, Metadata
    │   ├── pipeline.py                #   Pipeline (ABC), AgentPipeline, Planner/Critic/Student
    │   ├── filters.py                 #   Filter (ABC) — apply(chunks: List[Chunk])
    │   ├── generator.py               #   Generator (ABC) — generate(documents)
    │   ├── chunkers.py                #   Chunker (Dummy/Simple/Recursive)
    │   ├── augmenters.py
    │   ├── context_engine.py          #   Retriever (ABC), ContextAssembler, SimpleContextAssembler
    │   ├── agent.py                   #   Agent (ABC)
    │   ├── icl.py                     #   ICLBlock (ABC)
    │   ├── db.py                      #   IDB (ABC)
    │   ├── examples.py
    │   └── ⚠ filters.py и generator.py — здесь только абстракции; реализации в src/filtering, src/generation
    │
    ├── filtering/                     # ✦ Реализации фильтров (см. §4)
    │   ├── simple_filters/filters.py          # LengthFilter
    │   ├── lexical_filtration/simple_lexic_filtrator.py  # SimpleLexicalFiltrator
    │   ├── exactsubstr/main.py                # ExactSubstrFiltrator (+ experiments_ds1000/)
    │   ├── textbooks_are_all_you_need/main.py # EducationValueClassifierFilter (+ experiments_ds1000/)
    │   └── orthorules/main.py                 # orthorules_pipeline (не зарегистрирован в ComponentNames)
    │
    ├── generation/                    # ✦ Реализации синтетических генераторов (см. §5)
    │   ├── zero_shot.py               # ZeroShotGenerator
    │   ├── one_shot.py                # OneShotGenerator
    │   ├── random_topic.py            # RandomTopicGenerator
    │   ├── random_word.py             # RandomWordGenerator
    │   ├── new_insruct.py             # InstructGenerator
    │   ├── code_eval.py               # CodeEvalGenerator
    │   └── incorrect_examples.py      # IncorrectExampleGenerator
    │
    ├── agents/                        # LLM-агенты (роли)
    │   ├── critique/                  #   Critic, ComplexCritic, Decrim, Reflexion, SelfRefine
    │   ├── planning/                  #   LeastToMost, PlanAndExecute, MPC (predictive_decoding)
    │   ├── reasoning/                 #   CoT, AutoCoT, ContrastiveCoT, SelectionInference
    │   ├── pipelines/                 #   Роли для MAPS/MARS/REWOO/REACT/PANEL
    │   ├── rag/                       #   InstructRationality, CoRAGFinalSolver, Raptor*
    │   ├── generation/                #   APISelector, QueryGenerator (in-inference, не Generator!)
    │   └── general/                   #   DS1000Solver, EmbeddingAgent, SimpleAgent, TFIDFEmbedding
    │
    ├── rag/                           # RAG-ретриверы: simple, corag, raptor, instructrag, api
    ├── icl/                           # In-context learning: fewshot, iccl, lens, icv
    ├── context_assemblers/            # Сборка контекста: corag_assembler, instruct_assembler, examples_assembler
    ├── db/                            # Backend для хранения документов: docs_db (LocalDB), github_db, raptor_db
    ├── tools/                         # Инструменты для REWOO/REACT: DBSearchTool, LLMTool
    ├── mcp/                           # MCP-сервер
    ├── benchmarks/                    # Бенчмарки
    │   ├── __init__.py                #   реэкспорт: DS1000, DataItemDS1000, ResultsDS1000
    │   ├── ds1000/                    #   DS1000 — основной бенчмарк
    │   └── SWE-bench/                 #   SWE-bench (отдельная подсистема)
    └── utils/                         # Хелперы, логгеры, парсер GitHub, адаптеры
```

Главные сущности:

| Имя | Где | Назначение |
|---|---|---|
| `Block` | `src/agent_constructor/core.py` | Базовый класс всех «компонентов» (фильтры, генераторы, агенты, ретриверы...). Аргументы конструктора, аннотированные `Block`-подтипом, автоматически считаются зависимостями при сборке. |
| `Document` / `Chunk` | `src/agent_constructor/core.py` | Документы из БД и их чанки. Документы помечены `source` (`'documents'` или `'examples'`, см. `src/utils`). |
| `Pipeline` | `src/agent_constructor/pipeline.py` | Базовый класс пайплайна. Все шаблоны в `src/pipelines/templates/` его наследуют. |
| `ComponentNames` / `PipelinesNames` | `src/pipelines/constants.py` | ⭐ Единый каталог всех компонент и пайплайнов: enum → импорт-путь. Используется в YAML-конфигах. |
| `PipelineBuilder` | `src/pipelines/pipeline_builder.py` | Берёт `PipelineConfig`, строит DAG зависимостей через `networkx`, собирает пайплайн. |
| `DS1000` | `src/benchmarks/ds1000.py` | Бенчмарк. `bench.eval(run_method, save_path, num_workers)` — параллельный прогон. |

---

## 2. Где что хранится

Поставляется с репозиторием (физически лежит в `data/`):

| Артефакт | Путь |
|---|---|
| Датасет DS1000 | `data/ds1000/ds1000.jsonl.gz` |
| Датасет torchmetrics | `data/ds1000/torchmetrics.jsonl.gz` |
| База документов (sqlite) | `data/docs_database.db`, `data/docs_database_examples.db` |

Создаётся при запуске (путь задаётся в YAML):

| Артефакт | Где задаётся |
|---|---|
| Векторная БД (Chroma/Qdrant) | `data_base.params.path_to_vector_db` — в `simple_example.yaml` это `data/docs_vdb_rc_1000_meta_ml20` |
| Кэши education-фильтра (эмбеддинги, аннотации, модель, датасет, метаданные) | `filter.params.{embeddings_path, annotations_path, models_path, datasets_path, metadata_path}` (все обязательные) |
| Результаты прогонов | CLI-флаг `-s` у `run_ds1000.py` (по умолчанию `results/`); внутри `<config_stem>/<timestamp>/` |
| Логи прогонов | `logs_path` в YAML (создаётся через `create_logging` в `src/utils/loggers.py`) |

Прочее:

| Артефакт | Путь |
|---|---|
| YAML-конфиги экспериментов | `pipeline_configs/*.yaml` (эталон — `simple_example.yaml`) |
| Документация по модулям | `docs/{filtration,rag,icl,agents,services,benchmarks}/*.md` |

---

## 3. Как проводятся эксперименты

В репозитории два уровня запуска: декларативный (YAML) и программный.

### 3.1 Декларативный — `run.py` и `run_ds1000.py`

**Один запрос:**

```bash
python run.py -c pipeline_configs/simple_example.yaml -q "What is pandas?"
```

**Прогон всей директории конфигов на DS1000:**

```bash
python run_ds1000.py \
    -c pipeline_configs \                # директория с YAML
    -d data/ds1000/ds1000.jsonl.gz \     # датасет
    -s results \                         # куда писать
    -n 4                                 # воркеров
```

`run_ds1000.py` итерируется по всем `*.yaml/*.yml` в `-c`, для каждого: загружает конфиг → строит пайплайн → прогоняет `bench.eval(...)`. Результаты сохраняются в `results/<имя_конфига>/<timestamp>/`. Поддерживается параметр `continue_exp` для дооценки уже посчитанных прогонов (см. `docs/benchmarks/ds1000/ds1000-readme.md`).

> ⚠ Бенчмарк использует `ProcessPoolExecutor` — точку входа обязательно оборачивать в `if __name__ == "__main__":`.

### 3.2 Программный — `rag_eval.py` как пример

Компоненты создаются и связываются вручную в Python (см. [rag_eval.py](rag_eval.py)). Удобно для экспериментов, которых нет в шаблонах, или для отладки.

### 3.3 Как устроен YAML-конфиг

```yaml
type: SIMPLE                  # значение PipelinesNames (см. src/pipelines/constants.py)
params:                       # параметры конструктора пайплайна
  top_k: 5
logs_path: "simple/logs"      # опционально
components:                   # словарь {имя_параметра_пайплайна: ComponentConfig}
  embedder:
    type: EMBEDDING_AGENT     # значение ComponentNames
    params: { url: "...", model_name: "..." }
  data_base:
    type: LOCAL_DB
    params: { path_to_db: "...", path_to_vector_db: "...", collection_name: "docs" }
  chunker:
    type: RECURSIVE_CHUNKER
    params: { max_chunk_size: 1000 }
  filter:                     # ← фильтрация чанков
    type: LENGTH_FILTER
    params: { min_len: 20 }
  context_assembler: { type: CORAG_CONTEXT_ASSEMBLER }
  retriever:        { type: SIMPLE_RETRIEVER }
  agent:            { type: DS1000_SOLVER_AGENT, params: { url: "...", api: "chat", ... } }
```

Ключевые правила:
- **Ключ компонента в `components` должен совпадать с именем аргумента `__init__`** того блока, который его потребляет (см. сигнатуру `SimplePipeline.__init__` в `src/pipelines/templates/simple_pipeline.py`). Например, `SimplePipeline` ожидает аргументы `data_base, agent, chunker, retriever, filter, icl_block, generator, context_assembler, enhancer, top_k` — поэтому ключи в YAML такие.
- **Зависимости выводятся автоматически** через `inspect.signature` + проверку наследования от `Block` (`ComponentRegistry._is_dependency_annotation`).
- **Порядок сборки** определяется топологической сортировкой DAG зависимостей (`PipelineBuilder._get_build_order`).
- **Переопределить имя зависимости** можно через `deps_mapping: {ожидаемое_имя: ключ_в_components}` в `ComponentConfig`.
- Один компонент может быть **списком** конфигов (например `tools:` в REWOO/REACT — см. `pipeline_configs/rewoo_example.yaml`).

### 3.4 Как включается/выключается фильтрация и генерация ⭐

Базовый шаблон `SimplePipeline` ([src/pipelines/templates/simple_pipeline.py](src/pipelines/templates/simple_pipeline.py)) делает следующее на этапе сборки:

```python
documents = data_base.get_documents()
if generator:
    synth_docs = generator.generate(documents=documents)   # генерация ВКЛ
    documents.extend(synth_docs)

chunks = []
[chunks.extend(chunker.chunk(doc)) for doc in documents]
if filter:
    chunks = filter.apply(chunks)                          # фильтрация ВКЛ

data_base.add_chunks(chunks)
```

Включение/выключение управляется **наличием ключа в YAML**:

| Действие | Что делаем в YAML |
|---|---|
| **Выключить фильтрацию** | Полностью удалить ключ `filter:` из `components`. Аргумент `filter` у `SimplePipeline` имеет `= None` по умолчанию. |
| **Включить фильтрацию** | Добавить блок `filter: { type: <ИМЯ_ИЗ_ComponentNames>, params: {...} }`. Список доступных типов — см. §4. |
| **Сменить фильтр** | Поменять `type:` (например `LENGTH_FILTER` → `EDUCATION_VALUE_FILTER`) и параметры. |
| **Несколько фильтров подряд** | YAML позволяет передать **список** конфигов на одну позицию (см. `_parse_component_config` в `configs.py`), однако `SimplePipeline.filter` принимает одиночный объект — для пайплайна потребуется либо обёртка-композит, либо использование шаблона, который принимает `List[Filter]`. |
| **Выключить генерацию** | Удалить ключ `generator:` (или не добавлять). По умолчанию `generator=None`. |
| **Включить генерацию** | Добавить блок `generator: { type: <ИМЯ_ИЗ_ComponentNames>, params: { url, model_name, prob } }`. `prob` ∈ [0, 1] — доля документов БД, используемых как затравка. |
| **Выключить ICL** | Удалить ключ `icl_block:`. |
| **Выключить query-расширение** | Удалить ключ `enchancer:` (в коде используется именно эта орфография). |

Примеры в `pipeline_configs/` (большинство — `type: SIMPLE` с разной комбинацией компонент: с/без ICL, с/без enhancer, с разными retriever-ами). В существующих конфигах в качестве фильтра задействован только `LENGTH_FILTER`, а блок `generator:` отсутствует — для подключения других фильтров/генераторов нужно добавить соответствующие секции вручную, опираясь на список из §4–§5.

### 3.5 Как добавить новый компонент

1. Реализовать класс, унаследованный от соответствующего ABC (`Filter`, `Generator`, `Agent`, `Retriever`, `ContextAssembler`, `Chunker`, `ICLBlock`, ...). Все они в итоге наследуют `Block`.
2. Зарегистрировать его в `src/pipelines/constants.py`:

   ```python
   class ComponentNames(Enum):
       MY_FILTER = "src.filtering.my_filter.MyFilter"
   ```

3. Использовать в YAML: `type: MY_FILTER`. Сборщик подхватит автоматически (ленивый импорт).

---

## 4. Реализованные методы фильтрации

Базовый интерфейс — `src/agent_constructor/filters.py`:

```python
class Filter(Block):
    required: bool = False
    @abstractmethod
    def apply(self, chunks: List[Chunk]) -> List[Chunk]: ...
```

Все фильтры применяются **на чанках** (не на документах) сразу после `Chunker`.

| ComponentName | Класс / файл | Идея | Ключевые параметры |
|---|---|---|---|
| `LENGTH_FILTER` | [`LengthFilter`](src/filtering/simple_filters/filters.py) | Отбрасывает чанки короче порога после `strip()`. Самый базовый фильтр шума. | `min_len: int = 20` |
| `SIMPLE_LEXICAL_FILTER` | [`SimpleLexicalFiltrator`](src/filtering/lexical_filtration/simple_lexic_filtrator.py) | Чистит код-документацию от мусора регулярками + дедуп. Конфигурируется через `FilteringConfig` (pydantic). Этапы: удаление терминальных секций (`References#...`), мусорных строк (даты, DOI, ISBN, email...), навигационных строк (`see also`, `View on TensorFlow.org`, `Download notebook`...), нормализация пустых строк, дедуп блоков скользящим окном, MD5-дедуп строк. | `config: FilteringConfig` со флагами `remove_junk_blocks`, `remove_navigation_lines`, `remove_terminal_sections`, `normalize_empty_lines`, `min_len_for_dedup`, `sliding_window_max_length` |
| `EXACT_SUBSTR_FILTER` | [`ExactSubstrFiltrator`](src/filtering/exactsubstr/main.py) | Дедупликация точных повторов подстрок между чанками через suffix array + LCP (Kasai). Реализация по статье *Deduplicating Training Data Makes Language Models Better*. При нахождении общей подстроки ≥ `threshold` сохраняется чанк с меньшим индексом, в чанке с бо́льшим индексом интервал вырезается; перекрывающиеся интервалы сливаются. Можно работать на байтах UTF-8 или через токенайзер. | `threshold: int`, `enable_bytes: bool = True`, `enable_tokenizer: bool = False`, `tokenizer` |
| `EDUCATION_VALUE_FILTER` | [`EducationValueClassifierFilter`](src/filtering/textbooks_are_all_you_need/main.py) | Реализация идеи статьи *Textbooks Are All You Need*. Три стадии: (1) построить эмбеддинги всех чанков (кэш на диске); (2) LLM-аннотатор размечает подвыборку как «учебный / не учебный» с балансом классов; (3) Random Forest учится на эмбеддингах и предсказывает класс для всех чанков. Остаются только `label == 1`. Все артефакты (эмбеддинги, аннотации, модель, метаданные) кэшируются — при повторном запуске не пересчитываются. | **Обязательные** (без дефолтов, задаются в YAML): `llm_url`, `llm_model`, `embedding_url`, `embedding_model`, `embeddings_path`, `annotations_path`, `models_path`, `datasets_path`, `metadata_path`. Опциональные: `subsample_size=1000`, `limit_labels=20`, `save_metadata=False`, `test_size=0.2`, `random_state=42` |
| (не зарегистрирован в Enum) | [`orthorules_pipeline`](src/filtering/orthorules/main.py) | Отбор данных по ортогональным правилам: rating подвыборки по 50 правилам → отбор `r` правил через DPP (`select_rules`) → rating всего корпуса по выбранным правилам → семплирование `k` лучших (`gumbel`/детерминированно) с температурой `tau`. Не интегрирован в `ComponentNames`, вызывается напрямую как функция над `List[Document]`. | `batch_size`, `chunk_size`, `r`, `tau`, `k`, `sampling_method` |

> Подробные конспекты по фильтрам: `docs/filtration/exactsubstr.md`, `docs/filtration/textbooks_are_all_you_need.md`.

---

## 5. Реализованные методы генерации

Базовый интерфейс — `src/agent_constructor/generator.py`:

```python
class Generator(Block):
    @abstractmethod
    def generate(self, documents: List[Document]) -> List[Document]: ...
```

Все генераторы **синтетических документов** (тип «пре-инференс», см. `docs/agents/generation/generation_agents.md`) работают по одной схеме: берут случайные `prob` документов из БД нужного `source` (`'documents'` или `'examples'`, см. `src/utils`), для каждого через LLM по системному+пользовательскому промпту генерируют новый текст и возвращают `List[Document]` с `metadata={"generated_from": doc.id}`. Результат подмешивается в общий список перед `chunker.chunk(...)` (см. `SimplePipeline`).

Общие параметры конструктора всех генераторов: `url`, `model_name`, `prob: float = 0.01`.

| ComponentName | Класс / файл | Идея | Источник | Температура |
|---|---|---|---|---|
| `ZERO_SHOT_GENERATOR` | [`ZeroShotGenerator`](src/generation/zero_shot.py) | Zero-shot: просит модель «создать документацию о любой Python-библиотеке» без примера. | `documents` | 0.6 |
| `ONE_SHOT_GENERATOR` | [`OneShotGenerator`](src/generation/one_shot.py) | One-shot: тот же запрос, но даёт текст исходного документа как образец стиля. | `documents` | 0.3 |
| `RANDOM_TOPIC_GENERATOR` | [`RandomTopicGenerator`](src/generation/random_topic.py) | Выбирает случайную тему из фиксированного списка (15 numpy-тем: «Creating arrays», «Loading data from files», …) и просит написать пример документации именно по ней, ориентируясь на стиль исходного документа. | `documents` | 0.1 |
| `RANDOM_WORD_GENERATOR` | [`RandomWordGenerator`](src/generation/random_word.py) | Two-step: (1) формулирует задачу по программированию с 3–5 случайными словами из списка из ~100 ключевых слов; (2) решает её как Python-код, опираясь на пример. На выходе синтетический пример кода. | `examples` | 0.6 / 0.1 |
| `INSTRUCT_GENERATOR` | [`InstructGenerator`](src/generation/new_insruct.py) | Two-step: (1) сгенерировать инструкцию-промпт, который привёл бы к данному примеру кода; (2) выполнить эту инструкцию и получить новый код. | `examples` | 0.6 / 0.1 |
| `CODE_EVAL_GENERATOR` | [`CodeEvalGenerator`](src/generation/code_eval.py) | «Усложнитель» существующего примера: добавляет ограничения, шаги, повышает сложность по подсказанным эвристикам, остаётся в рамках того же синтаксиса. | `examples` | 0.1 |
| `INCORRECT_EXAMPLES_GENERATOR` | [`IncorrectExampleGenerator`](src/generation/incorrect_examples.py) | Генерирует **некорректную** версию примера: с семантическими багами, но всё ещё компилируемую/запускаемую тестами. Полезно как hard-негативы. | `examples` | 0.1 |

> ⚠ Не путать с `src/agents/generation/` — там лежат **in-inference** агенты `APISelector` и `QueryGenerator` (перефразирование запроса для RAG, выбор API). Они **не** являются `Generator` — это обычные `Agent`-ы, подключаются как `enchancer:` / `api_selector:` в YAML, а не как `generator:`.

> Подробный конспект: `docs/agents/generation/generation_agents.md`.

---

## 6. Полезные точки входа при работе с репозиторием

- **Каталог всех компонентов:** [`src/pipelines/constants.py`](src/pipelines/constants.py) — здесь enum'ы для всех агентов, ретриверов, фильтров, чанкеров, генераторов, пайплайнов.
- **Как собирается пайплайн:** [`src/pipelines/pipeline_builder.py`](src/pipelines/pipeline_builder.py) и [`registry.py`](src/pipelines/registry.py).
- **Базовый шаблон с фильтрацией+генерацией:** [`src/pipelines/templates/simple_pipeline.py`](src/pipelines/templates/simple_pipeline.py) — смотреть `__init__` и `run` для понимания, какие компоненты подключаются и в каком порядке.
- **Базовые абстракции:** [`src/agent_constructor/core.py`](src/agent_constructor/core.py), [`filters.py`](src/agent_constructor/filters.py), [`generator.py`](src/agent_constructor/generator.py), [`pipeline.py`](src/agent_constructor/pipeline.py).
- **Документация по экспериментам:** [`docs/exp_setup.md`](docs/exp_setup.md).
- **Документация по DS1000:** [`docs/benchmarks/ds1000/ds1000-readme.md`](docs/benchmarks/ds1000/ds1000-readme.md).
- **Описание генераторов и фильтров:** [`docs/agents/generation/generation_agents.md`](docs/agents/generation/generation_agents.md), [`docs/filtration/`](docs/filtration/).
