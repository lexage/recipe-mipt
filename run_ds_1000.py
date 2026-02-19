from src.pipelines.configs import (
    PipelineConfig,
    ComponentConfig,
)

from src.pipelines.constants import ComponentNames, PipelinesNames
from src.pipelines.pipeline_builder import PipelineBuilder

from src.benchmarks import DS1000, DataItemDS1000

import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

pipeline_config = PipelineConfig(
    type=PipelinesNames.SIMPLE,
    params={"top_k" : 10},
    components={
        "embedder": ComponentConfig(
            type=ComponentNames.EMBEDDING_AGENT,
            params={
                "name": "embedder", 
                "url": "http://localhost:7216/v1", 
                "model_name": "Qwen/Qwen3-Embedding-4B"
                }
        ),
        "data_base": ComponentConfig(
            type=ComponentNames.LOCAL_DB,
            params={
                "path_to_db": "data/docs_database_examples.db",
                "path_to_vector_db": "data/docs_vdb",
                "collection_name": "docs",
                "search_filter": {"source": "documents"}
                },
        ),
        "chunker": ComponentConfig(
            type=ComponentNames.RECURSIVE_CHUNKER,
            params={"name": "chunker", "max_chunk_size": 1000}
        ),
        "filter": ComponentConfig(
            type=ComponentNames.LENGTH_FILTER,
            params={"name": "filter", "min_len": 20}
        ),
        "context_assembler": ComponentConfig(
            type=ComponentNames.SIMPLE_CONTEXT_ASSEMBLER,
            params={"name": "simple_assembler"},
        ),
        "retriever": ComponentConfig(
            type=ComponentNames.SIMPLE_RETRIEVER,
            params={"name": "simple_retriever"},
        ),
        "enhancer": ComponentConfig(
            type=ComponentNames.QUERY_GENERATOR,
            params={
                "name": "query_generator", 
                "url": "http://localhost:7215/v1", 
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ", 
                "options": 1
                }
        ),
        "agent": ComponentConfig(
            type=ComponentNames.SIMPLE_AGENT,
            params={
                "name": "simple_agent", 
                "url": "http://localhost:7215/v1", 
                "model_name": "Qwen/Qwen1.5-32B-Chat-AWQ"
                }
        )
    }
)


def bench():

    pipeline = PipelineBuilder().build(pipeline_config)
    bench = DS1000("data/ds1000/ds1000.jsonl.gz")

    def run_pipline(item: DataItemDS1000):
        return pipeline.run(item.prompt)

    bench.eval(run_pipline, save_path="results/experiment_name", num_workers=4)

if __name__ == "__main__":
    bench()
