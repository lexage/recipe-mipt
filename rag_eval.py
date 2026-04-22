import logging
import os

from pathlib import Path
from typing import List

from src.benchmarks import DS1000, DataItemDS1000
from src.agent_constructor.core import Chunk
from src.utils.api_parser import parse_api_calls, ParseResult

from src.agents.general.embedding_agents import EmbeddingAgent
from src.db.docs_db import LocalDB
from src.context_assemblers.corag_assembler import CoRAGContextAssembler
from src.agents.general.ds1000_solver import DS1000Solver


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


def main():

    bench = DS1000(dataset_path="data/ds1000/ds1000.jsonl.gz")
    
    embedder = EmbeddingAgent(
        url="http://localhost:7216/v1",
        model_name="Qwen/Qwen3-Embedding-4B",
    )

    db = LocalDB(
        embedder=embedder,

        return_full_docs=False,
        return_examples=True,
        merge_examples=True,

        path_to_db="data/docs_database_examples.db",
        path_to_vector_db="data/sparse_qdrant_test",
        collection_name="docs",
    )

    assembler = CoRAGContextAssembler(name="assembler")

    agent = DS1000Solver(
        url="http://localhost:7215/v1",
        context_after_task=False,
        api="chat",
    )


    def run_pipeline(task: DataItemDS1000):
        api_results = parse_api_calls(task).apis

        chunks : List[Chunk] = []

        for api in api_results:
            query = api.qualified

            chunks.extend(
                db.query(
                    query_text=query,
                    top_k=1
                )
            )
        
        context = assembler.assemble(
            chunks
        )

        return agent.run(
            task=task.prompt,
            context=context,
        )
    
    bench.eval(
        run_method=run_pipeline,
        save_path="results/bench_eval_base/",
        num_workers=1
        )


if __name__ == "__main__":
    main()
