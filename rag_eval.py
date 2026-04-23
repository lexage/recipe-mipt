import logging

from typing import List
import pandas as pd

from src.benchmarks import DS1000, DataItemDS1000
from src.agent_constructor.core import Chunk
from src.utils.api_parser import parse_api_calls
from src.agent_constructor.chunkers import RecursiveChunker
from src.filtering.simple_filters import LengthFilter
from src.agents.generation.generation_agents import DocRewriter

from src.agents.general.embedding_agents import EmbeddingAgent
from src.db.docs_db import LocalDB
from src.context_assemblers.corag_assembler import CoRAGContextAssembler
from src.agents.general.ds1000_solver import DS1000Solver


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

SAVE_PATH = "results/bench_eval_base/"
BENCH_PATH = "data/ds1000/ds1000_test.jsonl.gz"

def main():

    bench = DS1000(dataset_path=BENCH_PATH)
    
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

    rewriter = DocRewriter(
        url="http://localhost:7215/v1"
    )

    chunker = RecursiveChunker()
    filter = LengthFilter()

    docs = db.get_documents()

    chunks = []

    for doc in docs:
        chunks.extend(chunker.chunk(doc))

    chunks = filter.apply(chunks)

    db.add_chunks(chunks)


    assembler = CoRAGContextAssembler(name="assembler")

    agent = DS1000Solver(
        url="http://localhost:7215/v1",
        api="chat",
    )

    retrieve_results = []

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
        
        for chunk in chunks:
            chunk.text = rewriter.run(chunk.text)

        context = assembler.assemble(
            chunks
        )

        result = agent.run(
            task=task.prompt,
            context=context,
        )

        for chunk, api in zip(chunks, api_results):
            retrieve_results.append(
                {
                    "problem_id": task.p_id,
                    "problem_lib": task.metadata.get("library"),
                    "problem": task.prompt,
                    "api": api.qualified,
                    "chunk": chunk.text,
                    "result": result,
                }
            )

        return result
    
    bench.eval(
        run_method=run_pipeline,
        save_path=SAVE_PATH,
        num_workers=4
        )

    pd.DataFrame().from_records(retrieve_results).to_excel(f"{SAVE_PATH}/retrieve_results.xlsx")

if __name__ == "__main__":
    main()
