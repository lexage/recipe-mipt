from src.pipelines.configs import (
    PipelineConfig,
    ComponentConfig,
)

from src.pipelines.constants import ComponentNames, PipelinesNames

from src.pipelines.pipeline_builder import PipelineBuilder
import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

pipeline_config = PipelineConfig(
    type=PipelinesNames.SIMPLE,
    components={
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={"name": "embedder", "url": "http://localhost:7216/v1", "model_name": "Qwen/Qwen3-Embedding-4B"}
        ),
        "data_base": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "data/docs_database_examples.db",
                "path_to_vector_db": "data/docs_vector_database",
                "collection_name": "docs"
                },
        ),
        "chunker": ComponentConfig(
            type=ComponentNames.DUMMY_CHUNKER,
            params={"name": "chunker"}
        ),
        "filter": ComponentConfig(
            type=ComponentNames.LENGTH_FILTER,
            params={"name": "filter", "min_len": 100}
        ),
        "context_assembler": ComponentConfig(
            type=ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "simple_assembler"},
        ),
        "retriever": ComponentConfig(
            type=ComponentNames.SIMPLE_RETRIEVER,
            params={"name": "simple_retriever"},
        ),
        "generator": ComponentConfig(
            type=ComponentNames.CODE_EVAL_GENERATOR,
            params={"url": "http://localhost:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"}
        ),
        "agent": ComponentConfig(
            type=ComponentNames.SIMPLE_AGENT,
            params={"name": "simple_agent", "url": "http://localhost:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"}
        )
    }
)

def main():
    pipeline = PipelineBuilder().build(pipeline_config)
    result = pipeline.run("What is numpy")
    print(result)

if __name__ == "__main__":
    main()
