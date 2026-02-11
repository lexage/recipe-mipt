from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.chunkers import Chunker
from src.agent_constructor.filters import Filter
from src.agent_constructor.agent import Agent
from src.agent_constructor.generator import Generator
from src.agent_constructor.pipeline import Pipeline
from src.agent_constructor.context_engine import ContextAssembler
from src.agent_constructor.icl import ICLBlock


class SimplePipeline(Pipeline):

    def __init__(self, 
                 data_base: IDB, 
                 retriever: Retriever, 
                 agent: Agent, 
                 chunker: Chunker,
                 filter: Filter = None, 
                 icl_block: ICLBlock = None,
                 generator: Generator = None,
                 context_assembler: ContextAssembler = None,
                 enhancer: Agent = None,
                 top_k: int = 1):
        

        super().__init__("simple_pipeline")

        documents = data_base.get_documents()
        if generator:
            synth_docs=generator.generate(documents=documents)
            documents.extend(synth_docs)
            
        chunks = []
        [chunks.extend(chunker.chunk(doc)) for doc in documents]
        if filter:
            chunks = filter.apply(chunks)
            
        data_base.add_chunks(chunks)



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
        for query in tasks:
            context.extend(self.retriever.retrieve(query=query, k=self.top_k))

        if self.icl_block:
            context = self.icl_block.apply(context)

        if self.context_assembler:
            context = self.context_assembler.assemble(context)
        
        return self.agent.run(task, context)
