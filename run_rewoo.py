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
    type=PipelinesNames.REWOO, 
    components={
        # "embedder": ComponentConfig(
        #     type=ComponentNames.TFIDF_EMBEDDING,
        #     params={
        #         "name": "embedder",
        #         "vectorizer_path": "grant/db/tfidf_vectorizer.pkl"
        #     }
        # ),
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={
                "name": "embedder",
                "url": "http://team_recipe-dev-ssh:7216/v1",
                "model_name": "Qwen/Qwen3-Embedding-4B"
            }
        ),
        # "db": ComponentConfig(
        #     type=ComponentNames.LOCAL_DB,
        #     params={
        #         "path_to_db": "grant/db/docs_database_examples.db",
        #         "path_to_vector_db": "grant/db/docs_vector_database",
        #         "collection_name": "docs"
        #         },
        # ),
        "db": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "grant/db/docs_database_examples.db",
                "path_to_vector_db": "grant/db/docs_vector_database_qwen_4b",
                "collection_name": "docs"
                },
        ),
        "context_assembler": ComponentConfig(
            type = ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "SimpleContext"}
        ),
        "planner": ComponentConfig(
            type=ComponentNames.REWOO_PLANNER,
            params={
                "name": "PlannerREWOO",
                "url": "http://team_recipe-dev-ssh:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"
            }
        ),
        "worker": ComponentConfig(
            type=ComponentNames.REWOO_WORKER,
            params={
                "name": "WorkerREWOO",
                "url": "http://team_recipe-dev-ssh:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"
            }
        ),
        "solver": ComponentConfig(
            type=ComponentNames.REWOO_SOLVER,
            params={
                "name": "SolverREWOO",
                "url": "http://team_recipe-dev-ssh:7215/v1",
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"
            }
        )
    }
)


def main():
    pipeline = PipelineBuilder().build(pipeline_config)
    result = pipeline.run("What is pandas?")
    print(result)


if __name__ == "__main__":
    main()
