"""SimplePipelineWithDocFilter — variant of SimplePipeline with an extra
document-level filtration step BEFORE chunking.

Order of operations in __init__ (corpus stage shared with SimplePipeline,
see src/pipelines/corpus.py):
    documents = data_base.get_documents()
    if document_filter and stage == pre_generation:  documents = filter(documents)
    if generator:        documents += generator.generate(documents)
    if document_filter and stage == post_generation: documents = filter(documents)
    if materialize_db_path: the indexed corpus is written to its own .db
    chunks = [c for d in documents for c in chunker.chunk(d)]
    if filter:           chunks = filter.apply(chunks)
    data_base.add_chunks(chunks)

Apart from the added `document_filter` parameter, behaviour and `run()`
are identical to `SimplePipeline`.  Self-instrumentation: stores
`_filter_apply_time` and `_document_filter_apply_time` for the runner
to read into runtime_stats.json.
"""

import time
from typing import Optional

from src.agent_constructor.agent import Agent
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.context_engine import ContextAssembler, Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.filters import Filter, DocumentFilter
from src.agent_constructor.generator import Generator
from src.agent_constructor.icl import ICLBlock
from src.agent_constructor.pipeline import Pipeline
from src.pipelines.corpus import build_corpus
from src.utils.retrieval_log import record_chunks, record_context


class SimplePipelineWithDocFilter(Pipeline):
    """`SimplePipeline` + optional document-level filter step."""

    def __init__(
        self,
        data_base: IDB,
        agent: Agent,
        chunker: Chunker,
        document_filter: DocumentFilter = None,
        retriever: Retriever = None,
        filter: Filter = None,
        icl_block: ICLBlock = None,
        generator: Generator = None,
        context_assembler: ContextAssembler = None,
        enhancer: Agent = None,
        document_filter_stage: str = "pre_generation",
        materialize_db_path: str = "",
        on_exists: str = "rebuild",
        generation_seed: int = None,
        top_k: int = 1,
    ):
        super().__init__("simple_pipeline_with_doc_filter")

        # Same corpus stage as SimplePipeline (see src/pipelines/corpus.py); this
        # template keeps its historical default of always filtering before
        # generation and of always keeping the source documents.
        build = build_corpus(
            data_base=data_base,
            generator=generator,
            document_filter=document_filter,
            document_filter_stage=document_filter_stage,
            keep_source_docs=True,
            materialize_db_path=materialize_db_path,
            on_exists=on_exists,
            generation_seed=generation_seed,
        )
        documents = build.documents
        self._document_filter_apply_time = build.document_filter_time
        self._generation_time = build.generation_time
        self._materialize_time = build.materialize_time
        self._build_stats = build.stats

        chunks = []
        [chunks.extend(chunker.chunk(doc)) for doc in documents]

        # ----- chunk-level filter (same as SimplePipeline) ----------------
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
