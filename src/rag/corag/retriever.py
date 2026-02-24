from typing import List, Tuple

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.db import IDB
from src.agent_constructor.core import Chunk

from .src.vllm_client import VllmClient, get_vllm_model_id
from .src.agent import CoRagAgent
from .constants import CoRAGSearchTypes


class CoRAGRetriever(Retriever):
    def __init__(
            self, name: str, url: str,
            data_base: IDB, 
            search_type: CoRAGSearchTypes,

            max_path_length: int = 3,
            max_message_length: int = 4096,
            temperature: float = 0.7,
            task_description: str = "answer multihop question",

            expand_size: int = 4, 
            num_rollouts: int = 2, 
            beam_size: int = 1,

            n: int = 4,
            
            ):
        
        super().__init__(name)

        vllm_client = VllmClient(
            model = get_vllm_model_id(url=url),
            url = url
        )

        self.corag_agent = CoRagAgent(
            vllm_client=vllm_client, 
            data_base=data_base
            )
        
        self.data_base = data_base

        self.search_type = search_type

        self.max_path_length = max_path_length
        self.max_message_length = max_message_length
        self.temperature = temperature
        self.task_description= task_description

        self.expand_size = expand_size
        self.num_rollouts = num_rollouts
        self.beam_size = beam_size

        self.n = n

    def retrieve(self, query: str, k: int = 1) -> Tuple[List, List[Chunk]]:
        
        match self.search_type:
            case CoRAGSearchTypes.SAMPLE_SEARCH:
                results = self.corag_agent.sample_path(
                    query=query,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.temperature,
                    top_k=k,
                )
            case CoRAGSearchTypes.TREE_SEARCH:
                results = self.corag_agent.tree_search(
                    query=query,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.temperature,
                    expand_size=self.expand_size,
                    num_rollouts=self.num_rollouts,
                    beam_size=self.beam_size,
                    top_k=k,
                )
            case CoRAGSearchTypes.BEST_OF_N_SEARCH:
                results = self.corag_agent.best_of_n(
                    query=query,
                    task_desc=self.task_description,
                    max_path_length=self.max_path_length,
                    max_message_length=self.max_message_length,
                    temperature=self.temperature,
                    n=self.n,
                    top_k=k,
                )

        chunks = [self._create_path_chunk(
            past_subqueries=results.past_subqueries,
            past_subanswers=results.past_subanswers,
            task_desc=self.task_description,
        )]

        doc_ids = list(set([item for sublist in results.past_doc_ids for item in sublist]))

        chunks.extend([
            Chunk(
                id=doc.id,
                doc_id=doc.id,
                text=doc.text,
                metadata=doc.metadata
            ) for doc in self.data_base.get_documents(doc_ids)
        ])

        return chunks

    def _create_path_chunk(self,
        past_subqueries: List[str], past_subanswers: List[str], task_desc: str) -> Chunk:

        assert len(past_subqueries) == len(past_subanswers)
        past = ''
        for idx in range(len(past_subqueries)):
            past += f"""Intermediate query {idx+1}: {past_subqueries[idx]}
    Intermediate answer {idx+1}: {past_subanswers[idx]}\n"""
        past = past.strip()

        text = f"""## Intermediate queries and answers
    {past or 'Nothing yet'}

## Task description
    {task_desc}"""
        
        return Chunk(
            id="corag_intermediate_steps",
            doc_id=None,
            text=text,
            metadata={
                "source": "corag"
            }
        )
