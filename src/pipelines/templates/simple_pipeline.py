from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.filters import Filter
from src.agent_constructor.agent import Agent
from src.agent_constructor.generator import Generator
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.icl import ICLBlock
from src.benchmarks.ds1000 import DataItemDS1000
from src.agents.general.codemmlu import AnswerFormat, RagFormat

prompt_rag = """You are a retrieval-query generator for a code-RAG system.

CONTEXT ABOUT THE INDEX:
- The vector index contains short reference overviews of Python standard-library
  modules: collections, math, heapq, functools, bisect, string, queue, enum, typing
  (plus numpy, pandas, scipy, sklearn, torch, tensorflow, matplotlib).
- Each document lists the module's main classes/functions by name
  (e.g. "deque", "Counter", "defaultdict", "lru_cache", "reduce", "bisect_right",
   "heappush", "PriorityQueue", "Enum", "List", "Dict", "Optional").
- Retrieval is cosine similarity over text embeddings (Qwen3-Embedding-4B).
  Lexical overlap with API names BOOSTS recall. Long prose problem statements
  DILUTE the signal.

YOUR JOB:
Given an algorithmic problem statement, output 3 short search queries that
will pull the most useful stdlib documentation/examples for solving it.

RULES:
1. Each query MUST be one line, 6-15 words, no punctuation except dots in dotted
   API names (e.g. "collections.deque", "functools.lru_cache").
2. Each query MUST mention at least one concrete Python identifier: either a
   module name (collections, heapq, ...) or an API name (deque, Counter,
   lru_cache, bisect_right, heappush, PriorityQueue, Enum, lru_cache, ...).
3. Each query MUST mention the algorithmic pattern in plain words:
   BFS, DFS, topological sort, dynamic programming, memoization, two pointers,
   binary search, sliding window, priority queue, hash map, monotonic stack,
   linked list pointer manipulation, etc.
4. NEVER paste the full problem statement, constraints, examples or I/O
   description. Strip everything except the underlying technique.
5. The all queries must cover DIFFERENT axes:
   - query 1: algorithmic pattern + data structure
   - query 2: concrete stdlib module/API to call
   - query 3: typing/return-type shape (if relevant) or a secondary helper module
6. If no stdlib library is plausibly useful (pure arithmetic / string indexing),
   still emit queries naming the closest concept (e.g. "string.digits ascii table").
7. Output ONLY the bulleted list. No preface, no trailing commentary.

OUTPUT FORMAT (exactly, including the leading dash and space):
- <query 1>
- <query 2>
- <query 3>

A Problem: """

class SimplePipeline(Pipeline):

    def __init__(self, 
                 data_base: IDB, 
                 agent: Agent, 
                 chunker: Chunker,
                 retriever: Retriever = None, 
                 filter: Filter = None,
                 icl_block: ICLBlock = None,
                 generator: Generator = None,
                 context_assembler: ContextAssembler = None,
                 enhancer: Agent = None,
                 top_k: int = 1):
        

        super().__init__("simple_pipeline")

        documents = data_base.get_documents()
        if generator:
            synth_docs = generator.generate(documents=documents)
            documents.extend(synth_docs)

        chunks = []
        [chunks.extend(chunker.chunk(doc)) for doc in documents]
        if filter:
            chunks = filter.apply(chunks)

        data_base.add_chunks(chunks)

        self.data_base = data_base
        self.retriever = retriever
        self.agent = agent
        self.context_assembler = context_assembler
        self.icl_block = icl_block
        self.enhancer = enhancer
        self.top_k = top_k

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
            return self.agent.run(task, context, AnswerFormat)
        
        return self.agent.run(task, "", AnswerFormat)

    def close(self) -> None:
        self.data_base.close()
