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
├── optimize_prompts.py     # ⭐ Оптимизация optimizable-промптов пайплайна (см. §8)
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
    │   ├── prompts.py                 #   ⭐ Prompt + PromptDiscoverable (см. §7)
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
    │   ├── incorrect_examples.py      # IncorrectExampleGenerator
    │   ├── paraphrase.py              # ParaphraseGenerator (exp6, oracle)
    │   ├── code2doc.py                # CodeToDocGenerator (exp8, oracle)
    │   ├── code2task.py               # CodeToTaskGenerator (exp8, корпусный)
    │   └── rag_guides.py              # ⭐ RagGuideGenerator (exp13, наш метод; промпты — §7)
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
    ├── optimization/                  # ✦ Поиск по промптам (см. §8)
    │   ├── candidates.py              #   Кандидат = {имя_промпта: текст}
    │   ├── split.py                   #   Сид-сплит DS1000 train/eval
    │   ├── evaluator.py               #   Роллаут: pipeline.run(task) → тест DS1000
    │   └── optimizers/                #   Бэкенды: gepa (upstream), random (контроль)
    ├── mcp/                           # MCP-сервер
    ├── benchmarks/                    # Бенчмарки
    │   ├── __init__.py                #   реэкспорт: DS1000, DataItemDS1000, ResultsDS1000
    │   ├── ds1000/                    #   DS1000 — основной бенчмарк
    │   └── SWE-bench/                 #   SWE-bench (отдельная подсистема)
    └── utils/                         # Хелперы, логгеры, парсер GitHub, адаптеры
        ├── token_tracker.py           #   Подсчёт токенов (thread-local активный трекер)
        ├── retrieval_log.py           #   ⭐ Что ретривер отдал каждой задаче (см. §6)
        └── prompt_registry.py         #   ⭐ Сбор промптов со всего пайплайна (см. §7)
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
| `Prompt` / `PromptDiscoverable` | `src/agent_constructor/prompts.py` | ⭐ Промпт как объект: изменяемый на месте, с откатом и отпечатком. Миксин даёт компоненту `self.prompt(...)` — промпт остаётся в теле метода, но становится обнаружимым. См. §7. |

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

## 6. Что сохраняется по итогам прогона

Каталог `results/<имя_конфига>/<timestamp>/`:

| Файл | Кто пишет | Что внутри |
|---|---|---|
| `answers.jsonl` | `DS1000._run_method` | Ответ модели на каждую задачу (после `_postprocess`), пишется потоково. Задачи, на которых `pipeline.run` упал, молча пропускаются — строк может быть меньше 1000. |
| `results.csv` | `ResultsDS1000.save` | Построчно `score` (0/1) + `result` (текст ошибки/`passed`) + метаданные задачи. |
| `summary.txt` | `ResultsDS1000.summary` | Итог + разрезы по `library` и `perturbation_type`. |
| `runtime_stats.json` | `run_ds1000.py` | Тайминги (`init_time_s`, `filter_apply_time_s`, …), пик RSS, токены, **блок `prompts`** (§7). |
| `retrieved_chunks.jsonl` | `run_ds1000.py --log-chunks` | Чанки, которые ретривер отдал каждой задаче. |
| `config.yaml` | `run_ds1000.py` | Копия конфига, породившего прогон. |
| `prompts.json` | `run_ds1000.py` | Полный текст всех промптов прогона (§7). |

Последние три появились, чтобы папка результатов была **самоописывающей**: раньше по ней нельзя было восстановить, какой конфиг и какой текст промпта дали это число.

### 6.1 Лог извлечённых чанков (`--log-chunks`)

Зачем: посмотреть, что реально доехало до солвера — особенно на упавших задачах.

Как устроено ([src/utils/retrieval_log.py](src/utils/retrieval_log.py)):

- `bench.eval` гоняет **один** объект пайплайна из нескольких потоков, поэтому запись хранится в `threading.local()`. Обычный атрибут на пайплайне перетирался бы соседней задачей, и чанки приписались бы чужому `problem_id` — молча. Тот же приём, что у `token_tracker`.
- `SimplePipeline.run` / `SimplePipelineWithDocFilter.run` вызывают `record_chunks(context)` сразу после ретрива (то есть уже с учётом примеров, которые `LocalDB.query` дотягивает при `return_examples: True`) и `record_context(...)` после сборки контекста.
- `run_ds1000.py` вызывает `take()` в том же потоке сразу после `pipeline.run`, где известен `problem_id`, копит записи под `Lock` и пишет их одним файлом после прогона.

Флаги: `--log-chunks` включает, `--log-chunk-chars N` ограничивает сохраняемый текст чанка (по умолчанию 800, `0` — целиком). В записи есть `chars` — истинная длина чанка, по ней видно, был ли текст обрезан.

Просмотр — [results/analyze_chunks.py](results/analyze_chunks.py): печатает по задаче «промпт → извлечённые чанки → решение», подтягивая `answers.jsonl`, `results.csv` и датасет из той же папки. Зависимостей нет (только stdlib).

```bash
python results/analyze_chunks.py results/<конфиг>/<ts>/retrieved_chunks.jsonl --failed --library Pandas -n 10
python results/analyze_chunks.py results/<конфиг>/<ts>/retrieved_chunks.jsonl -n 0 --out context_report.txt
```

---

## 7. Промпты как объекты (`Prompt` / `PromptDiscoverable`)

**Проблема.** Промпт каждого LLM-модуля лежал инлайн-литералом в теле метода. Следствия: правка промпта — это правка кода, которая не оставляет следа в результатах; и ничто не могло перечислить или подменить промпты (нужно и для логирования, и для любого оптимизатора промптов).

**Решение.** Текст промпта **остаётся там, где он был**. Литерал заворачивается в `self.prompt(key, default)`: первый вызов регистрирует его в пер-инстансном хранилище, последующие возвращают живой (возможно, изменённый) текст.

### 7.1 API

| Сущность | Файл | Назначение |
|---|---|---|
| `Prompt` | [src/agent_constructor/prompts.py](src/agent_constructor/prompts.py) | `text`, `required`, `revision`, `set()`, `rollback()`, `temporarily()`, `render(**kw)`, `fingerprint()` |
| `PromptDiscoverable` | там же | Миксин: `prompt(key, default, optimizable=True)`, `render_prompt(key, default, **kw)`, `named_prompts()` |
| `collect_prompts` / `snapshot_prompts` / `apply_candidate` / `describe_prompts` | [src/utils/prompt_registry.py](src/utils/prompt_registry.py) | Сбор промптов со всего пайплайна, ключи вида `"<имя_компонента>.<ключ>"` |

Ключевые решения и **почему** именно так:

- **Мутация — на месте.** Компонент держит ссылку на хранилище, поэтому `set()` доезжает до компонента без пересборки пайплайна. Возврат нового объекта оставил бы компонент на старом.
- **`set()` перепроверяет плейсхолдеры.** Кандидат, потерявший `$lib`, падает в момент присваивания, а не внутри рабочего потока по ходу прогона.
- **Плейсхолдеры — `string.Template` (`$lib`), не `str.format`.** Эти промпты учат писать Python и регулярно содержат литеральные фигурные скобки (`{}`, `f'{x:.2f}'`) — `str.format` на них падает. Литеральный `$` пишется как `$$`; голый `$` отвергается на конструировании.
- **Формат-контракты заморожены** (`optimizable=False`). Хвосты `### DOC` / `TITLE:` — это протокол, который разбирает `_parse_docs`. Оптимизатор, переписавший хвост, дал бы **ноль** распарсенных документов, молча, и поиск учился бы на шуме.
- **Регистрация ленивая** → правило: **регистрировать все варианты безусловно, ветвиться после**. Иначе прогон под `policy=v2` никогда не покажет v1-промпт, и инвентарь окажется неполным.
- **Никакого `from __future__ import annotations`** в этих модулях — сломает интроспекцию зависимостей (см. общий грабли-лист).

### 7.2 Как перевести модуль на промпт-объекты

1. Добавить миксин в базы: `class MyAgent(PromptDiscoverable, Agent):` — менять `__init__` не нужно.
2. Обернуть литерал: `self.prompt("system", "You are …")`, а с подстановками — `self.render_prompt("user", "Solve $task.", task=task)`.
3. Оба варианта policy-развилки регистрировать **до** `if`, выбирать уже между возвращёнными строками.

### 7.3 Что уже переведено

Только [src/generation/rag_guides.py](src/generation/rag_guides.py) (`RagGuideGenerator`) — цель будущей оптимизации:

| Ключ | Где | Изменяемый |
|---|---|---|
| `chat_system` | `_chat` | да |
| `format_common`, `format_v2_extra` | `_format_jobs` | да |
| `format_tail` | `_format_jobs` | **нет** (контракт) |
| `migration_style_v1`, `migration_style_v2` | `_migration_jobs` | да |
| `migration_tail` | `_migration_jobs` | **нет** (контракт) |
| `recipe_main`, `recipe_v2_rule` | `_recipe_jobs` | да |
| `recipe_tail` | `_recipe_jobs` | **нет** (контракт) |

`DEPRECATIONS` промптом **не** является: это фактические данные (какие API удалены и чем заменены), их переписывание породило бы выдуманные API, которых не поймает ни один тест.

Остальные ~40 LLM-модулей не тронуты. `collect_prompts` их просто не видит — переводить можно по мере надобности.

### 7.4 Обнаружение промптов в пайплайне

`PipelineBuilder.build` теперь **сохраняет** карту `{имя_в_YAML: инстанс}` в `pipeline._components` (раньше выбрасывалась). Отсюда `collect_prompts` берёт точный инвентарь компонентов, без рефлексии по `__dict__`.

Границы, о которых стоит помнить:

- Промпты можно менять **между** прогонами задач, но не во время: `bench.eval` держит 4 потока на одном пайплайне.
- Живая мутация доезжает до тех, кто рендерит на каждый вызов (солвер, LLM-фильтры). До `RagGuideGenerator` она **не** доезжает: его `generate()` уже отработал внутри `SimplePipeline.__init__` — такой генератор надо дёргать напрямую.

### 7.5 Тест

[test_rag_guide_prompts.py](test_rag_guide_prompts.py) + эталон `test_rag_guide_prompts_golden.json`. Эталон снят с **дорефакторного** кода и сверен с ним по AST, поэтому тест доказывает побайтовую идентичность шести промптов (3 слота × v1/v2) плюс `chat_system`. Дополнительно проверяются контракт мутации, полнота инвентаря, литеральные скобки и round-trip реестра. Сеть не нужна: `openai`/`tqdm` подменяются заглушками, если не установлены.

```bash
python test_rag_guide_prompts.py
```

---

## 8. Оптимизация промптов

Промпты стали объектами (§7) и обнаружимы через пайплайн — значит по ним можно
искать. [optimize_prompts.py](optimize_prompts.py) берёт обычный YAML-конфиг, строит
пайплайн тем же `PipelineBuilder`, что и `run_ds1000.py`, находит в нём
`optimizable`-промпты и запускает по ним поиск.

### 8.1 Что считается роллаутом

**Роллаут гоняет сам пайплайн**: `pipeline.run(task.prompt)` → реальный тест DS1000.
Не переписанная копия его логики. Это принципиально: пайплайн с несколькими солверами
и агрегатором будет оценён как этот пайплайн, а не как что-то похожее на него. Оценщик
ничего не знает ни про число солверов, ни про то, какому компоненту принадлежит промпт.

Функция оценки ответа (`score_answer`) внедряется снаружи — по умолчанию это реальный
исполнитель DS1000, но её можно подменить (так делает тест, чтобы не поднимать
подпроцесс).

| Слой | Файл | Роль |
|---|---|---|
| Кандидат | [candidates.py](src/optimization/candidates.py) | `{имя_промпта: текст}` по **optimizable**-промптам; замороженные контракты не входят |
| Сплит | [split.py](src/optimization/split.py) | Сид-сплит DS1000: оптимизатор видит только `train` |
| Роллаут | [evaluator.py](src/optimization/evaluator.py) | Собрать пайплайн → прогнать задачи → оценить |
| Бэкенды | [optimizers/](src/optimization/optimizers/) | `gepa` (upstream-пакет), `random` (контроль) |

Ключи кандидата — это `Prompt.name` (`"<имя_компонента>.<ключ>"`), ровно те же, что
ищет `prompts.set_overrides`. Поэтому `best_prompts.json` скармливается
`run_ds1000.py --prompts` без всякой трансляции.

### 8.2 Как кандидат доезжает до компонентов

Развилка та же, что в §7.4:

- **Промпты build-time** (генератор: его `generate()` работает внутри
  `SimplePipeline.__init__`) — правка после сборки уже ничего не изменит. Поэтому
  оптимизатор ставит process-wide оверрайды и **пересобирает пайплайн на каждый
  роллаут**.
- **Промпты query-time** (системный промпт солвера, фильтры на запросе) — достаточно
  мутации на месте, флаг `--reuse-pipeline`.

⚠ Каждая пересборка обязана получать **свежую директорию векторной БД**: Qdrant
пропускает уже лежащие в нём id чанков, поэтому переиспользование каталога означало бы,
что новые сгенерированные документы не проиндексируются вовсе и **все кандидаты
получат одинаковый скор**. Стоимость пересборки давится `--max-docs` (урезание корпуса)
и `--gen-limit` (лимит LLM-вызовов генератора).

### 8.3 Бэкенды

`gepa` — upstream-пакет (Agrawal и др., 2025): рефлексивная мутация по текстовым
трассам исполнения + Парето-фронт по обучающим примерам (набор **кандидатов**, каждый
лучший на своём подмножестве задач — защита от локального оптимума по среднему).
Наш адаптер отдаёт ему `scores` (вектор по задачам) и `trajectories` (что произошло,
словами) — именно текст ошибки исполнения делает мутацию осмысленной.

`random` — контроль: те же роллауты и артефакты, но модель переписывает промпт **не
видя** обратной связи. Если gepa не бьёт random, рефлексивный сигнал не работает и
результат не стоит показывать.

Свой бэкенд = реализовать протокол `PromptOptimizer` ([base.py](src/optimization/optimizers/base.py))
и добавить фабрику в `optimizers/__init__.py`. Выше по стеку не меняется ничего.

⚠ `gepa` не в `pyproject.toml` и не проверялся вживую (пакета нет локально, у сервера
нет сети). Адаптер написан по документированному протоколу, с защитными lookup-ами.
Перед длинным прогоном:

```bash
python optimize_prompts.py --check
```

### 8.4 Запуск и артефакты

```bash
python optimize_prompts.py -c test_configs_experimental_13/simple_example_gen_guides_v2.yaml \
    --optimizer random --budget 6 --n-train 15 --max-docs 150
```

`results/optimization/<ts>/`: `split.json` (что видел оптимизатор), `seed_prompts.json`,
`best_prompts.json`, `result.json` (скоры, история, потраченный бюджет).

Замер настоящего эффекта — на задачах, которых оптимизатор не видел:

```bash
python run_ds1000.py -c <configs_dir> \
    --prompts results/optimization/<ts>/best_prompts.json \
    --exclude-split results/optimization/<ts>/split.json
```

Скор внутри оптимизации снят на урезанном корпусе — это **эвристика поиска**, а не
цифра для отчёта. Отчётная цифра берётся только из прогона выше.

### 8.5 Ограничения (знать до запуска)

- Для любой сборки пайплайна нужен кэш `Qdrant/bm25` (fastembed) — тот самый, что уже
  ломал прогон.
- Путь исполнения DS1000 локально не проверялся: `check_correctness` порождает
  подпроцесс, на Windows он переимпортирует модули и заглушки не переживают. Это тот же
  код, что уже гоняет `run_ds1000.py`.
- Локальный тест — [test_optimization.py](test_optimization.py): фейковый пайплайн,
  чей ответ зависит от его же build-time-промпта, поэтому проверяется вся цепочка
  (оверрайды → сборка → роллаут → отбор кандидата) без сервера.

---

## 9. Полезные точки входа при работе с репозиторием

- **Каталог всех компонентов:** [`src/pipelines/constants.py`](src/pipelines/constants.py) — здесь enum'ы для всех агентов, ретриверов, фильтров, чанкеров, генераторов, пайплайнов.
- **Как собирается пайплайн:** [`src/pipelines/pipeline_builder.py`](src/pipelines/pipeline_builder.py) и [`registry.py`](src/pipelines/registry.py).
- **Базовый шаблон с фильтрацией+генерацией:** [`src/pipelines/templates/simple_pipeline.py`](src/pipelines/templates/simple_pipeline.py) — смотреть `__init__` и `run` для понимания, какие компоненты подключаются и в каком порядке.
- **Базовые абстракции:** [`src/agent_constructor/core.py`](src/agent_constructor/core.py), [`filters.py`](src/agent_constructor/filters.py), [`generator.py`](src/agent_constructor/generator.py), [`pipeline.py`](src/agent_constructor/pipeline.py).
- **Промпты:** [`src/agent_constructor/prompts.py`](src/agent_constructor/prompts.py) и [`src/utils/prompt_registry.py`](src/utils/prompt_registry.py) — §7; пример перевода модуля — [`src/generation/rag_guides.py`](src/generation/rag_guides.py).
- **Что доехало до солвера:** [`src/utils/retrieval_log.py`](src/utils/retrieval_log.py) + просмотрщик [`results/analyze_chunks.py`](results/analyze_chunks.py) — §6.1.
- **Оптимизация промптов:** [`optimize_prompts.py`](optimize_prompts.py) и [`src/optimization/`](src/optimization/) — §8; контракт бэкенда — [`optimizers/base.py`](src/optimization/optimizers/base.py).
- **Документация по экспериментам:** [`docs/exp_setup.md`](docs/exp_setup.md).
- **Документация по DS1000:** [`docs/benchmarks/ds1000/ds1000-readme.md`](docs/benchmarks/ds1000/ds1000-readme.md).
- **Описание генераторов и фильтров:** [`docs/agents/generation/generation_agents.md`](docs/agents/generation/generation_agents.md), [`docs/filtration/`](docs/filtration/).
