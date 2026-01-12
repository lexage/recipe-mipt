# Проведение эксперимента

## Простой вариант

В самом простом варианте для прогона на бенчмарке DS1000 нужно будет подготовить `run` метод, который будет принимать `DataItemDS1000` и возвращать ответ (см. [DS100](./benchmarks/ds1000/ds1000-readme.md)).

```python
from benchmarks.ds1000 import DS1000, DataItemDS1000

def main():
    benchmark = DS1000("data/ds1000.jsonl.gz")
    
    agent = ...

    def run(task: DataItemDS1000):
        return agent.run(task.prompt)

    results = benchmark.eval(
        run_method=run,
        save_path="results"
    )

if __name__ == "__main__":
    main()
```

## Вариант с использованием сборщика

`PipelineBuilder` получает `PipelineConfig`, автоматически вычисляет порядок сборки на основе зависимостей и возвращает готовый `Pipeline`.

```python
from src.pipelines.configs import PipelineConfig, ComponentConfig
from src.pipelines.pipeline_builder import PipelineBuilder

def main():
    cfg = PipelineConfig(
        pipeline_type="simple",
        params={"max_context_tokens": 8_000},
        components={
            "retriever": ComponentConfig(
                type="corag_retriever",
                params={"max_sub_queries": 3}
            ),
            "agent": ComponentConfig(
                type="corag_final_solver",
                params={"url": "http://localhost:7215/v1"}
            ),
        }
    )

    builder = PipelineBuilder(cfg)
    pipeline = builder.build()
    result = pipeline.run("What is numpy?")
    print(result)

if __name__ == "__main__":
    main()
```

### Регистрация компонент

Компоненты регистрируются через ленивую загрузку:

- В `src/pipelines/constants.py` добавляется значение енума `ComponentNames`, где в качестве значения указан полный путь до класса (модуль + имя класса).
- При сборке по конфигу `PipelineBuilder` берет только нужные компоненты, импортирует их по указанному пути и проверяет зависимости/параметры (логика валидации не менялась).

> **_NOTE:_**  Декоратор `ComponentRegistry.register_component` больше не используется: чтобы компонент собирался из конфига, достаточно добавить элемент в `ComponentNames`.

Пример добавления новой компоненты:

```python
# src/pipelines/constants.py
class ComponentNames(Enum):
    SIMPLE_CHUNKER = "src.agent_constructor.chunkers.SimpleChunker"
    # добавьте свою компоненту
    MY_COMPONENT = "src.my_package.MyComponent"
```

Список всех доступных `ComponentNames` находится в `src/pipelines/constants.py`.

### Конфигурация пайплайна

`PipelineConfig` теперь содержит только три поля:

- #### `pipeline_type`

  Имя шаблона из `src/pipelines/templates` (`simple`, `rewoo`, `maps`). Шаблон должен поддерживаться внутри `PipelineBuilder`.

- #### `components`

  Словарь `{"component_name": ComponentConfig(...)}`. Название компонента **совпадает** с именем аргумента конструктора (`__init__`) тех блоков, которые на него зависят. Например, если `LocalDB.__init__` принимает аргументы `chunker`, `filter`, `embedding_agent`, то в конфиге должны присутствовать компоненты с такими ключами.

- #### `params`

  Дополнительные параметры шаблона, например `max_iterations` для `MAPSPipeline`.

### Конфигурация компонент

`ComponentConfig` класс конфигурации компонент:

```python
ComponentConfig(
    type=ComponentNames.LOCAL_DB,
    params={
        "path_to_db": "data/docs_database.db",
        "path_to_vector_db": "data/docs_vector_database",
        "collection_name": "docs"
    }
)
```

- `type` — одно из значений `ComponentNames`, зарегистрированное в `ComponentRegistry`.
- `params` — дополнительные аргументы конструктора (кроме зависимостей).
- зависимости указывать не требуется: они выводятся автоматически.

### Автоматическое определение зависимостей

1. При регистрации класс анализируется через `inspect.signature`.
2. Любой параметр, аннотированный типом, унаследованным от `agent_constructor.core.Block`, считается зависимостью.
3. `PipelineBuilder` строит граф зависимостей и делает `topological_sort`, и автоматически формирует порядок сборки.
4. Если зависимость отсутствует в конфиге или компонент не зарегистрирован, сборщик упадёт с понятной ошибкой.

### Пример сборки простого пайплайна с CoRAG ретривером

```python
from src.pipelines.configs import PipelineConfig, ComponentConfig
from src.pipelines.constants import ComponentNames
from src.pipelines.pipeline_builder import PipelineBuilder
from src.benchmarks import DataItemDS1000, DS1000

corag_config = PipelineConfig(
    pipeline_type="simple",
    components={
        "embedding_agent": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={
                "url": "http://localhost:7216/v1",
                "model_name": "Qwen/Qwen3-Embedding-0.6B"
            }
        ),
        "context_assembler": ComponentConfig(
            type=ComponentNames.CORAG_CONTEXT_ASSEMBLER
        ),
        "chunker": ComponentConfig(
            type=ComponentNames.SIMPLE_CHUNKER,
            params={"max_chars": 1000}
        ),
        "filter": ComponentConfig(
            type=ComponentNames.LENGTH_FILTER,
            params={"min_len": 200}
        ),
        "db": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "data/docs_database.db",
                "path_to_vector_db": "data/docs_vector_database",
                "collection_name": "docs"
            }
        ),
        "generator": ComponentConfig(
            type=ComponentNames.CORAG_SUB_GENERATOR,
            params={
                "url": "http://localhost:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat",
                "max_num_queries": 3
            }
        ),
        "sub_solver": ComponentConfig(
            type=ComponentNames.CORAG_SUB_SOLVER,
            params={
                "url": "http://localhost:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat",
                "max_num_queries": 3
            }
        ),
        "retriever": ComponentConfig(
            type=ComponentNames.CORAG_RETRIEVER,
            params={"max_sub_queries": 5}
        ),
        "agent": ComponentConfig(
            type=ComponentNames.CORAG_FINAL_SOLVER,
            params={
                "url": "http://localhost:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat"
            }
        )
    }
)

def bench():
    builder = PipelineBuilder(corag_config)
    pipeline = builder.build()

    def run_pipeline(task: DataItemDS1000):
        return pipeline.run(task.prompt)
    
    benchmark = DS1000(
        dataset_path="data/ds1000/ds1000.jsonl.gz"
    )

    benchmark.eval(
        run_method=run_pipeline,
        save_path="results/first_exp",
        num_workers=16,
    )

if __name__ == "__main__":
    bench()
```