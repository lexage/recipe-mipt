from typing import List

from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.core import Chunk, Document, Text
from src.agent_constructor.db import IDB
from src.benchmarks import DataItemDS1000

from .constants import CoRAGSearchTypes
from .retriever import CoRAGRetriever
from .src.agent.api_corag_agent import APICoRagAgent
from .src.vllm_client import VllmClient, get_vllm_model_id

class APICoRAGRetriever(CoRAGRetriever):

    def __init__(
        self,
        name: str,
        url: str,
        data_base: IDB,
        search_type: CoRAGSearchTypes,
        include_original_docs: bool = True,
        use_full_documents: bool = False,
        max_path_length: int = 3,
        max_message_length: int = 4096,
        temperature: float = 0.7,
        task_description: str = "solve Python programming problem using library APIs",
        expand_size: int = 4,
        num_rollouts: int = 2,
        beam_size: int = 1,
        n: int = 2,
        ignore_libs: List[str] | None = None,
    ):
        Retriever.__init__(self, name)

        vllm_client = VllmClient(
            model=get_vllm_model_id(url=url),
            url=url,
        )

        self.corag_agent = APICoRagAgent(
            vllm_client=vllm_client,
            data_base=data_base,
        )
        self.data_base = data_base
        self.ignore_libs = set(ignore_libs or [])

        available_types = (
            CoRAGSearchTypes.SAMPLE_SEARCH.value,
            CoRAGSearchTypes.TREE_SEARCH.value,
            CoRAGSearchTypes.BEST_OF_N_SEARCH.value,
        )
        if search_type not in available_types:
            raise ValueError(
                f"Invalid search type: {search_type}. "
                f"Available search types are {available_types}"
            )

        self.search_type = search_type
        self.max_path_length = max_path_length
        self.max_message_length = max_message_length
        self.temperature = temperature
        self.task_description = task_description
        self.expand_size = expand_size
        self.num_rollouts = num_rollouts
        self.beam_size = beam_size
        self.include_original_docs = include_original_docs
        self.use_full_documents = use_full_documents
        self.return_only_path = not self.include_original_docs
        self.return_docs_as_chunks = self.use_full_documents
        self.n = n

    def retrieve(self, query: DataItemDS1000 | Text, k: int = 1) -> List[Chunk]:
        prompt, metadata = self._extract_prompt_and_metadata(query)

        if metadata.get("library") in self.ignore_libs:
            return []

        self.corag_agent.set_task_metadata(metadata)
        return super().retrieve(query=prompt, k=k)

    @staticmethod
    def _extract_prompt_and_metadata(task: DataItemDS1000 | Text) -> tuple[str, dict]:
        if isinstance(task, str):
            return task, {}
        return task.prompt, task.metadata

    def _create_path_chunk(
        self,
        past_subqueries: List[str],
        past_subanswers: List[str],
        task_desc: str,
    ) -> Chunk:
        assert len(past_subqueries) == len(past_subanswers)
        past = ""
        for idx in range(len(past_subqueries)):
            past += f"""Intermediate API {idx + 1}: {past_subqueries[idx]}
    Intermediate summary {idx + 1}: {past_subanswers[idx]}\n"""
        past = past.strip()

        text = f"""## Explored APIs and summaries
    {past or "Nothing yet"}"""

        return Chunk(
            id="corag_intermediate_steps",
            doc_id=None,
            text=text,
            metadata={"source": "api_corag"},
        )
