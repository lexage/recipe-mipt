from dataclasses import replace
from typing import List, Dict, Tuple

from src.agent_constructor.core import Chunk, Document
from src.rag.api.retriever import APIRetriever, LIBRARY_TO_DOC_NAME

from ..api_prompts import (
    get_generate_api_subquery_prompt,
    get_generate_api_intermediate_answer_prompt,
    get_generate_api_final_answer_prompt,
)
from .corag_agent import CoRagAgent, _normalize_subquery as _normalize_web_subquery
from .agent_utils import RagPath


class APICoRagAgent(CoRagAgent):

    def __init__(self, *args, task_metadata: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_metadata = task_metadata or {}

    def set_task_metadata(self, task_metadata: dict | None):
        self.task_metadata = task_metadata or {}

    def _subquery_messages(
        self,
        query: str,
        past_subqueries: List[str],
        past_subanswers: List[str],
        task_desc: str,
    ) -> List[Dict]:
        return get_generate_api_subquery_prompt(
            query=query,
            past_subqueries=past_subqueries,
            past_subanswers=past_subanswers,
            task_desc=task_desc,
        )

    def _intermediate_messages(
        self,
        subquery: str,
        documents: List[str],
        main_task: str = "",
    ) -> List[Dict]:
        return get_generate_api_intermediate_answer_prompt(
            api=subquery,
            documents=documents,
            main_task=main_task or subquery,
        )

    def _normalize_subquery(self, subquery: str) -> str | None:
        subquery = _normalize_web_subquery(subquery)
        if not subquery:
            return None
        return APIRetriever.normalize_api(subquery)

    def _library_filtered(self, chunks: List[Chunk]) -> List[Chunk]:
        doc_library = LIBRARY_TO_DOC_NAME.get(self.task_metadata.get("library"))
        if not doc_library:
            return chunks

        return [
            chunk
            for chunk in chunks
            if (chunk.metadata or {}).get("library") == doc_library
        ]

    def _get_subanswer_and_doc_ids(
        self,
        subquery: str,
        main_task: str = "",
        max_message_length: int = 4096,
        top_k: int = 1,
    ) -> Tuple[str, List[Document], List[Chunk]]:
        retrieve_results = self._library_filtered(
            self.data_base.query(query_text=subquery, top_k=top_k)
        )

        if not retrieve_results:
            return "No relevant information found", [], []

        doc_ids = list({int(chunk.doc_id) for chunk in retrieve_results})
        documents = self.data_base.get_documents(doc_ids)
        doc_texts = [doc.text for doc in documents]

        messages = self._intermediate_messages(
            subquery=subquery,
            documents=doc_texts,
            main_task=main_task,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)

        subanswer = self.vllm_client.call_chat(
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )

        annotated_chunks = [
            self._with_retrieval_metadata(chunk, subquery)
            for chunk in retrieve_results
        ]
        return subanswer, documents, annotated_chunks

    @staticmethod
    def _with_retrieval_metadata(chunk: Chunk, api: str) -> Chunk:
        metadata = dict(chunk.metadata or {})
        metadata["retrieval_api"] = api
        return replace(chunk, metadata=metadata)

    def generate_final_answer(
        self,
        corag_sample: RagPath,
        task_desc: str,
        max_message_length: int = 4096,
        documents=None,
        **kwargs,
    ) -> str:
        messages = get_generate_api_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents,
        )
        self._truncate_long_messages(messages, max_length=max_message_length)
        return self.vllm_client.call_chat(messages=messages, **kwargs)
