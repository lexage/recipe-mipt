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

Любая сборочная единица регистрируется в `ComponentRegistry` с помощью декоратора:

```python
from src.pipelines.registry import ComponentRegistry
from src.pipelines.constants import ComponentNames

@ComponentRegistry.register_component(ComponentNames.SIMPLE_CHUNKER)
class SimpleChunker(Chunker):
    ...
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