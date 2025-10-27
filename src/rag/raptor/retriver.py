from src.agents.agent_constructor.context_engine import Retriever, Text
from src.agents.agent_constructor.db import IDB
from src.rag.raptor.core import RetrievalAugmentation, RetrievalAugmentationConfig
from src.agents.agent_constructor.pipeline import Agent
from src.rag.raptor.wrappers import QAModelWrapper, SummarizationWrapper, EmbeddingWrapper


class RaptorRetriver(Retriever):
    def __init__(self, name: str, data_base: IDB, path_to_raptor_db: str, embeddig_model: Agent, qa_model: Agent, summarization_model: Agent):
        super().__init__(name)

        rac = RetrievalAugmentationConfig(
            embedding_model=EmbeddingWrapper(embeddig_model),
            qa_model=QAModelWrapper(qa_model),
            summarization_model=SummarizationWrapper(summarization_model),
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
