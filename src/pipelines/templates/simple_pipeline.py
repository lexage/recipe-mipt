import time

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.filters import Filter, DocumentFilter
from src.agent_constructor.agent import Agent
from src.agent_constructor.generator import Generator
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.icl import ICLBlock
from src.benchmarks.ds1000 import DataItemDS1000
from src.pipelines.corpus import build_corpus
from src.utils.retrieval_log import record_chunks, record_context


class SimplePipeline(Pipeline):
    """Load -> [document filter] -> [generate] -> [materialize] -> chunk -> index.

    Corpus construction lives in `src.pipelines.corpus.build_corpus`, shared with
    `SimplePipelineWithDocFilter`. Every stage is optional: with no generator and
    no filters this is the historical behaviour. `materialize_db_path` writes the
    indexed corpus to its own .db (see that module for what the file guarantees).
    """

    def __init__(self,
                 data_base: IDB,
                 agent: Agent,
                 chunker: Chunker,
                 retriever: Retriever = None,
                 filter: Filter = None,
                 document_filter: DocumentFilter = None,
                 icl_block: ICLBlock = None,
                 generator: Generator = None,
                 context_assembler: ContextAssembler = None,
                 enhancer: Agent = None,
                 keep_source_docs: bool = True,
                 document_filter_stage: str = "pre_generation",
                 materialize_db_path: str = "",
                 on_exists: str = "rebuild",
                 generation_seed: int = None,
                 top_k: int = 1):


        super().__init__("simple_pipeline")

        build = build_corpus(
            data_base=data_base,
            generator=generator,
            document_filter=document_filter,
            document_filter_stage=document_filter_stage,
            keep_source_docs=keep_source_docs,
            materialize_db_path=materialize_db_path,
            on_exists=on_exists,
            generation_seed=generation_seed,
        )
        documents = build.documents
        self._document_filter_apply_time = (
            build.document_filter_time if document_filter else None)
        self._generation_time = build.generation_time
        self._materialize_time = build.materialize_time
        self._build_stats = build.stats

        chunks = []
        [chunks.extend(chunker.chunk(doc)) for doc in documents]

        # Time the filter step separately — exposed via self._filter_apply_time
        # for the runner to record in runtime_stats.json.
        filter_time = 0.0
        if filter:
            t0 = time.time()
            chunks = filter.apply(chunks)
            filter_time = time.time() - t0
        self._filter_apply_time = filter_time

        data_base.add_chunks(chunks)

        self.data_base = data_base
        self.retriever = retriever
        self.agent = agent
        self.context_assembler = context_assembler
        self.icl_block = icl_block
        self.enhancer = enhancer
        self.top_k = top_k

    def close(self):
        self.data_base.close()

    def run(self, task: str) -> str:

        if self.enhancer:
            tasks = self.enhancer.run(task)
        else:
            tasks = [task]

        context = []
        if self.retriever:
            for query in tasks:
                context.extend(self.retriever.retrieve(query=query, k=self.top_k))

        # Diagnostic only — read by run_ds1000.py --log-chunks, no-op otherwise.
        record_chunks(context)

        if self.icl_block:
            context = self.icl_block.apply(context)

        if self.context_assembler:
            context = self.context_assembler.assemble(context)

        record_context(context if isinstance(context, str) else "")

        if context:
            return self.agent.run(task, context)
        
        return self.agent.run(task, "")
