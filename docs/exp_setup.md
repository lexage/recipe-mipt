# Проведение эксперимента

## Простой вариант

В самом простом варианте для прогона на бенчмарке DS1000 нужно будет подготовить `run` метод, который будет принимать `DataItemDS1000` и возвращать ответ. (см. [DS100](./benchmarks/ds1000/ds1000-readme.md))

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

Сборщик пайплайна собирает экземпляр Pipeline по указанному конфигу.

```python
from src.pipelines.configs import (
    PipelineConfig, 
    AgentConfig, 
    DBConfig, 
    RetrieverConfig, 
    ChunkerConfig, 
    FilterConfig, 
    ContextAssemblerConfig
)
from src.pipelines.pipeline_builder import PipelineBuilder

def main():
    cfg = PipelineConfig(
        pipeline_type="simple",
        params = {...}
        execution_order=["context_assembler", "retriver", "agent"],
        components={
            "context_assembler": ContextAssemblerConfig(
                type="...",
                params = {...},
                dependencies = [...]
            ),
            "retriever": RetrieverConfig(
                type="...",
                params = {...},
                dependencies = [...]
            ),
            "agent": AgentConfig(
                type="...",
                params = {...},
                dependencies = [...]
            )
        }
    )

    builder = PipelineBuilder(cfg)
    pipeline = builder.build()
    result = pipeline.run("What is numpy?")
    print(result)

if __name__ == "__main__":
    main()
```
### Конфигурация пайплайна
В `PipelineConfig` нужно указать три основных параметра:

- #### `pipeline_type` - Тип пайплайна из [шаблонов](../src/pipelines/templates/). 

    Важно, что бы указанный шаблон пайплайна был зарегистрирован в [сборщике](../src/pipelines/pipeline_builder.py#L32)

- #### `components` - Основные компоненты из которых состоит пайплайн. 

    Компоненты - это сущности, которые также необходимо собирать, и у которых могут быть зависимости от других компонент (ретривер, агент, сборщик контекста и т.д.). Для каждого типа пайплайна нужен свой набор компонент. К примеру для [`SimplePipline`](../src/pipelines/templates/simple_pipeline.py#L7) нужен следующий набор компонент: `retriever: Retriever`, `agent: Agent`, `context_assembler: ContextAssembler`. 

- #### `execution_order` - Порядок, в котором должны собираться компоненты. 

    Нужен в тех случаях, когда у компонент есть свои зависимости, которые нужно собрать заранее. 

- #### `params` - Другие параметры пайлпайна. 

    Например для [`MAPSPipline`](../src/pipelines/templates/maps_pipeline.py#L9) помимо комонентнов агентов нужно указать параметр `max_iterations`.

### Конфигурация компонент пайплайна

Конфигурация компоненты указывается в `PiplineConfig` в поле `components`:

```python
cfg = PipelineConfig(
        ...
        components={
            ...
            "component_name": AgentConfig(
                type="...",
                params = {...},
                dependencies = [...]
            ),
            ...
        }
    )
```

Для сбора разного типа компонент реализованно несколько типов конфига:

- `AgentConfig` - конфигурация агента
- `DBConfig` - конфигурация БД
- `RetrieverConfig` - конфигурация ретривера
- `ChunkerConfig` - конфигурация чанкера
- `FilterConfig` - конфигурация блока фильтрации
- `ContextAssemblerConfig` - конфигурация сборщика контекста

Во всех типах конфигурации компонент нужно указать три параметра (_некторые из них могут быть необязательными - зависит от компоненты_):

- #### `type` - Тип компоненты. 

    По сути это указание на конкретую реализацию компоненты.
    Например для `RetrieverConfig` доступные типы компонент: `corag`, `raptor`, `instruct`. 

    Для того, что бы добавить новый тип компоненты, нужно добавить соответствующий блок в [сборщик компонент](../src/pipelines/factory.py), в соответствующую функцию. 

    Например если мы хотим добавить новую реализацию ретривера, то надо добавить блок сборки в [`_create_retriever`](../src/pipelines/factory.py#L146). 

- #### `dependencies` - Зависимости компоненты от других компонент. 

    К примеру у компоненты БД [`LocalDB`](../src/agent_constructor/db.py#L56) есть зависимости от компонент `chunker: Chunker`, `embedding_model: Agent` `filter: Filter`. 

    Сборка этой компоненты будет выглядеть следующим образом:

    ```python
    cfg = PipelineConfig(
            ...
            components={
                ...
                "chunker": ChunkerConfig(
                    type="...",
                    params = {...},
                    dependencies = [...]
                ),
                "embedding_model": AgentConfig(
                    type="...",
                    params = {...},
                    dependencies = [...]
                ),
                "filter": FilterConfig(
                    type="...",
                    params = {...},
                    dependencies = [...]
                ),
                "db": DBConfig(
                    type="local",
                    params = {...},
                    dependencies = ["chunker", "embedding_model", "filter"]
                ),
                ...
            }
        )
    ```

    Вот в таких случаях в `execution_order` нужно указать корректный сбор компонент. Сначала `chunker`, `embedding_model` и `filter`, а потом `db`. 

- #### `params` - Другие параметры компоненты. 

    К примеру для все той же компоненты БД [`LocalDB`](../src/agent_constructor/db.py#L56) нужно указать параметры: `path_to_db: str`, `path_to_vector_db: str`, `collection_name: str`.

### Пример сборки простого пайплайна с CoRAG ретривером. 

Как уже говорилось ранее, для `SimplePipeline` нужны только следующие компоненты: `retriever: Retriever`, `agent: Agent`, `context_assembler: ContextAssembler`. 

В качестве ретривера мы хотим использовать [`CoRAGRetriver`](../src/rag/corag/retriver.py). Для сборки [`CoRAGRetriver`] нужны компоненты:
`data_base: IDB`, `generator: Agent`, `sub_solver: Agent`. 

В качестве `generator: Agent` будем использовать [`CoRAGSubQueryGeneratorAgent`](../src/agents/corag_agents.py#L7). 

В качестве `sub_solver: Agent` будем использовать [`CoRAGSubSolver`](../src/agents/corag_agents.py#L55). 

В качестве `data_base: IDB` будем использовать [`LocalDB`](../src/agent_constructor/db.py#L56). Как описывалось выше, у этого компонента также есть свои зависимости и параметры. 

В качестве сборщика контекста `context_assembler: ContextAssembler` будем использовать [`CoRAGContextAssembler`](../src/agent_constructor/context_engine.py#L46)

В качестве основного агента пайплайна `agent: Agent` будем использовать [`CoRAGFinalSolver`](../src/agents/corag_agents.py#L100)

В итоге для такого пайплана код сборки с конфигурацией будут выглядеть следующим образом:


```python
from src.pipelines.configs import (
    PipelineConfig, 
    AgentConfig, 
    DBConfig, 
    RetrieverConfig, 
    ChunkerConfig, 
    FilterConfig, 
    ContextAssemblerConfig
)
from src.pipelines.pipeline_builder import PipelineBuilder
from src.benchmarks import DataItemDS1000, DS1000

corag_config = PipelineConfig(
    pipeline_type="simple",
    execution_order=[
        "embedding_agent", 
        "contex_assembler", 
        "chunker", 
        "filter", 
        "db", 
        "generator", 
        "sub_solver", 
        "retriever", 
        "agent"
    ],
    components={
        "embedding_agent": AgentConfig(
            type="embedding",
            params={"url": "http://localhost:7216/v1", "model_name": "Qwen/Qwen3-Embedding-0.6B"}
        ),
        "context_assembler": ContextAssemblerConfig(
            type="corag"
        ),
        "chunker": ChunkerConfig(
            type="simple",
            params={"max_chars": 1000}
        ),
        "filter": FilterConfig(
            type="length",
            params={"min_len": 200}
        ),
        "db": DBConfig(
            type="local",
            params={
                "path_to_db": "data/docs_database.db",
                "path_to_vector_db": "data/docs_vector_database",
                "collection_name": "docs"
                },
            dependencies=["chunker", "filter", "embedding_agent"]
        ),
        "generator": AgentConfig(
            type="corag_sub_generator",
            params={"url": "http://localhost:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat", "max_num_queries": 3}
        ),
        "sub_solver": AgentConfig(
            type="corag_sub_solver",
            params={"url": "http://localhost:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat", "max_num_queries": 3}
        ),
        "retriever": RetrieverConfig(
            type="corag",
            params={"max_sub_queries": 5},
            dependencies=["db", "generator", "sub_solver"]
        ),
        "agent": AgentConfig(
            type="corag_final_solver",
            params={"url": "http://localhost:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat"}
        )
    }
)

def bench():
    builder = PipelineBuilder(corag_config)
    pipeline = builder.build()

    def run_pipline(task: DataItemDS1000):
        return pipeline.run(task.prompt)
    
    benchmark = DS1000(
        dataset_path="data/ds1000/ds1000.jsonl.gz"
    )

    benchmark.eval(
        run_method=run_pipline,
        save_path="results/first_exp",
        num_workers=16,
    )

if __name__ == "__main__":
    bench()

```