from src.rag.raptor.core import RetrievalAugmentation, RetrievalAugmentationConfig
from src.rag.raptor.wrappers import QAModelWrapper, SummarizationWrapper, EmbeddingWrapper

from src.agent_constructor.context_engine import Retriever, Text
from src.agent_constructor.db import IDB
from src.agent_constructor.agent import Agent
from src.pipelines.registry import ComponentRegistry
from src.pipelines.constants import ComponentNames


@ComponentRegistry.register_component(ComponentNames.RAPTOR_RETRIEVER)
class RaptorRetriever(Retriever):
    def __init__(self, data_base: IDB, path_to_raptor_db: str, embedding_agent: Agent, qa_agent: Agent, summarization_agent: Agent):
        super().__init__("raptor_retirever")

        rac = RetrievalAugmentationConfig(
            embedding_model=EmbeddingWrapper(embedding_agent),
            qa_model=QAModelWrapper(qa_agent),
            summarization_model=SummarizationWrapper(summarization_agent),
        )

        try:

            raptor = RetrievalAugmentation(
                tree=path_to_raptor_db,
                config=rac
            )

        except ValueError as e:
            
            raptor = RetrievalAugmentation(
                config=rac
            )

            chunks = data_base.all_chunks()

            chunks_text = [chunk.text for chunk in chunks]
            giant_ass_text = '\n'.join(chunks_text)
            
            raptor.add_documents(giant_ass_text)

            raptor.save(path_to_raptor_db)

        self.raptor = raptor

    def retrieve(self, query:str, top_k:int = 5, collapsed:bool = True) -> Text:
        result, _ = self.raptor.retrieve(question=query, top_k=top_k, collapse_tree=collapsed)
        return result
