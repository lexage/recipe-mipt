import logging

import pandas as pd

from src.benchmarks import DS1000, DataItemDS1000
from src.agent_constructor.chunkers import RecursiveChunker
from src.filtering.simple_filters import LengthFilter
from src.agents.generation.generation_agents import APISelector

from src.agents.general.embedding_agents import EmbeddingAgent
from src.db.docs_db import LocalDB
from src.context_assemblers.corag_assembler import CoRAGContextAssembler
from src.agents.general.ds1000_solver import DS1000Solver
from src.rag.api import APIRetriever


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

SAVE_PATH = "results/bench_beter_assembly_chunks_chat_selector_lim6/"
BENCH_PATH = "data/ds1000/ds1000.jsonl.gz"

MAX_APIS = 8
API_TOP_K = 4
MAX_CONTEXT_CHUNKS = 6

IGNORE_LIBS = [
    #"Matplotlib",
]


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

    selector = APISelector(
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

    retriever = APIRetriever(
        name="api_retriever",
        data_base=db,
        api_selector=selector,
        max_apis=MAX_APIS,
        api_top_k=API_TOP_K,
        ignore_libs=IGNORE_LIBS,
    )

    assembler = CoRAGContextAssembler(name="assembler")

    agent = DS1000Solver(
        url="http://localhost:7215/v1",
        api="completions",
    )

    retrieve_results = []

    def run_pipeline(task: DataItemDS1000):
        chunks = retriever.retrieve(task, k=MAX_CONTEXT_CHUNKS)
        context = assembler.assemble(chunks) if chunks else ""

        result = agent.run(
            task=task.prompt,
            context=context,
        )

        for chunk in chunks:
            retrieve_results.append(
                {
                    "problem_id": task.p_id,
                    "problem_lib": task.metadata.get("library"),
                    "problem": task.prompt,
                    "api": (chunk.metadata or {}).get("retrieval_api"),
                    "chunk_id": chunk.id,
                    "chunk_doc_name": (chunk.metadata or {}).get("doc_name"),
                    "chunk_library": (chunk.metadata or {}).get("library"),
                    "chunk_score": (chunk.metadata or {}).get("score"),
                    "chunk": chunk.text,
                    "result": result,
                }
            )

        return result
    
    bench.eval(
        run_method=run_pipeline,
        save_path=SAVE_PATH,
        num_workers=1
        )

    pd.DataFrame().from_records(retrieve_results).to_excel(f"{SAVE_PATH}/retrieve_results.xlsx")

if __name__ == "__main__":
    main()
