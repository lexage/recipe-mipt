from src.pipelines.configs import (
    PipelineConfig,
    ComponentConfig,
)

from src.pipelines.constants import ComponentNames, PipelinesNames

pipeline_config = PipelineConfig(
    type=PipelinesNames.SIMPLE,
    params={"top_k": 3},
    components={
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={
                "name": "embedder",
                "url": "http://localhost:7216/v1",
                "model_name": "Qwen/Qwen3-Embedding-4B",
            },
        ),
        "data_base": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "data/docs_database_examples.db",
                "path_to_vector_db": "data/docs_vector_database",
                "collection_name": "docs",
                "search_filter": {"source": "documents"}
            },
        ),
        "chunker": ComponentConfig(
            type=ComponentNames.RECURSIVE_CHUNKER, 
            params={
                "name": "chunker",
                "max_chunk_size": 1000
            },
        ),
        "filter": ComponentConfig(
            type=ComponentNames.LENGTH_FILTER, 
            params={
                "name": "filter", 
                "min_len": 100
            }
        ),
        "context_assembler": ComponentConfig(
            type=ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "simple_assembler"},
        ),
        "retriever": ComponentConfig(
            type=ComponentNames.SIMPLE_RETRIEVER,
            params={"name": "simple_retriever"},
        ),
        "agent": ComponentConfig(
            type=ComponentNames.SIMPLE_AGENT,
            params={
                "name": "simple_agent",
                "url": "http://localhost:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ",
            },
        ),
    },
)
