from src.pipelines.configs import (
    PipelineConfig,
    ComponentConfig,
)

from src.pipelines.constants import ComponentNames, PipelinesNames

pipeline_config = PipelineConfig(
    type=PipelinesNames.REACT,
    components={
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={
                "name": "embedder",
                "url": "http://localhost:7216/v1",
                "model_name": "Qwen/Qwen3-Embedding-4B",
            },
        ),
        "db": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "data/docs_database_examples.db",
                "path_to_vector_db": "data/docs_vector_database",
                "collection_name": "docs",
            },
        ),
        "context_assembler": ComponentConfig(
            type=ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "SimpleContext"},
        ),
        "tools": [
            ComponentConfig(type=ComponentNames.DB_SEARCH_TOOL, params={"top_k": 6})
        ],
        "agent": ComponentConfig(
            type=ComponentNames.REACT_AGENT,
            params={
                "name": "ReActAgent",
                "url": "http://localhost:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ",
                "temperature": 0.5,
            },
        ),
    },
)
