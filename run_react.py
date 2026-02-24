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
    type=PipelinesNames.REACT,
    components={
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={"name": "embedder", "url": "http://team_recipe-dev-ssh:7216/v1", "model_name": "Qwen/Qwen3-Embedding-4B"}
        ),
        "db": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "/workspace/data/docs_database_examples.db",
                "path_to_vector_db": "/workspace/data/docs_vector_database_qwen_4b",
                "collection_name": "docs"
                },
        ),
        "context_assembler": ComponentConfig(
            type = ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "SimpleContext"}
        ),
        "tools": [
            ComponentConfig(
                type=ComponentNames.DB_SEARCH_TOOL,
                params={}
            )
        ],
        "agent": ComponentConfig(
            type=ComponentNames.REACT_AGENT,
            params={"name": "ReActAgent", "url": "http://team_recipe-dev-ssh:7215/v1", "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"}
        )
    }
)


def main():
    pipeline = PipelineBuilder().build(pipeline_config)
    result = pipeline.run("What is numpy?")
    print("RESULT: ", result)


if __name__ == "__main__":
    main()
