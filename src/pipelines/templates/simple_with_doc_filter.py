"""SimplePipelineWithDocFilter — variant of SimplePipeline with an extra
document-level filtration step BEFORE chunking.

Order of operations in __init__:
    documents = data_base.get_documents()
    if document_filter:  documents = document_filter.apply(documents)
    if generator:        documents += generator.generate(documents)
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
        top_k: int = 1,
    ):
        super().__init__("simple_pipeline_with_doc_filter")

        documents = data_base.get_documents()

        # ----- document-level filter (NEW vs SimplePipeline) ---------------
        doc_filter_time = 0.0
        if document_filter:
            t0 = time.time()
            documents = document_filter.apply(documents)
            doc_filter_time = time.time() - t0
        self._document_filter_apply_time = doc_filter_time

        if generator:
            synth_docs = generator.generate(documents=documents)
            documents.extend(synth_docs)

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

        if self.icl_block:
            context = self.icl_block.apply(context)

        if self.context_assembler:
            context = self.context_assembler.assemble(context)

        if context:
            return self.agent.run(task, context)

        return self.agent.run(task, "")
