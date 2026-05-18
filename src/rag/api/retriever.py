from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional

from src.agent_constructor.agent import Agent
from src.agent_constructor.context_engine import Retriever
from src.agent_constructor.core import Chunk, Text
from src.agent_constructor.db import IDB

if TYPE_CHECKING:
    from src.benchmarks import DataItemDS1000


LIBRARY_ROOTS = {
    "Pandas": {"pandas"},
    "Numpy": {"numpy"},
    "Matplotlib": {"matplotlib"},
    "Sklearn": {"sklearn"},
    "Scipy": {"scipy"},
    "Pytorch": {"torch"},
    "Tensorflow": {"tensorflow"},
}

LIBRARY_TO_DOC_NAME = {
    "Pandas": "pandas",
    "Numpy": "numpy",
    "Matplotlib": "matplotlib",
    "Sklearn": "sklearn",
    "Scipy": "scipy",
    "Pytorch": "torch",
    "Tensorflow": "tensorflow",
}

API_ALIASES = {
    "pd": "pandas",
    "np": "numpy",
    "plt": "matplotlib.pyplot",
    "tf": "tensorflow",
}


class APIRetriever(Retriever):
    def __init__(
        self,
        name: str,
        data_base: IDB,
        api_selector: Agent,
        max_apis: int = 8,
        api_top_k: int = 4,
        chunks_per_api: int = 2,
        ignore_libs: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self.data_base = data_base
        self.api_selector = api_selector
        self.max_apis = max_apis
        self.api_top_k = api_top_k
        self.chunks_per_api = chunks_per_api
        self.ignore_libs = set(ignore_libs or [])

    def retrieve(self, task: DataItemDS1000 | Text, k: int = 6) -> List[Chunk]:
        prompt, metadata = self._extract_prompt_and_metadata(task)

        if metadata.get("library") in self.ignore_libs:
            return []

        selected_apis = self.select_task_apis(
            raw_apis=self.api_selector.run(prompt),
            task_metadata=metadata,
        )

        chunks: List[Chunk] = []
        seen_chunk_ids: set[str] = set()

        for api in selected_apis:
            api_chunks = self._library_filtered(
                chunks=self.data_base.query(
                    query_text=api,
                    top_k=self.api_top_k,
                ),
                task_metadata=metadata,
            )
            api_chunks = sorted(
                api_chunks,
                key=lambda chunk: self._chunk_sort_key(chunk, api),
                reverse=True,
            )

            for chunk in api_chunks[:self.chunks_per_api]:
                self._add_unique_chunk(
                    chunks=chunks,
                    seen=seen_chunk_ids,
                    chunk=self._with_retrieval_metadata(chunk, api),
                )

        return sorted(
            chunks,
            key=self._chunk_sort_key,
            reverse=True,
        )[:k]

    def select_task_apis(
        self,
        raw_apis: List[str],
        task_metadata: dict | None = None,
    ) -> List[str]:
        task_metadata = task_metadata or {}
        allowed_roots = LIBRARY_ROOTS.get(task_metadata.get("library"), set())
        selected = []
        seen = set()

        for raw_api in raw_apis:
            api = self.normalize_api(raw_api)
            if not api:
                continue

            root = api.split(".", 1)[0]
            if allowed_roots and root not in allowed_roots:
                continue

            if api not in seen:
                selected.append(api)
                seen.add(api)

        return selected[:self.max_apis]

    @staticmethod
    def normalize_api(api: str) -> str | None:
        api = api.strip().strip("`")
        api = re.sub(r"^\s*[-*\d.)]+\s*", "", api)
        api = api.split("#", 1)[0].strip()
        if not api or not re.match(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$", api):
            return None

        root, tail = api.split(".", 1)
        if root in API_ALIASES:
            api = f"{API_ALIASES[root]}.{tail}"

        return api

    @staticmethod
    def _chunk_sort_key(chunk: Chunk, api: str = ""):
        score = chunk.metadata.get("score", 0) if chunk.metadata else 0
        doc_name = (chunk.metadata or {}).get("doc_name", "").lower()
        api_tail = api.rsplit(".", 1)[-1].lower()
        exact_name_bonus = 1 if api_tail and api_tail in doc_name else 0
        return exact_name_bonus, score

    @staticmethod
    def _add_unique_chunk(chunks: List[Chunk], seen: set[str], chunk: Chunk):
        chunk_id = str(chunk.id)
        if chunk_id in seen:
            return

        seen.add(chunk_id)
        chunks.append(chunk)

    @staticmethod
    def _extract_prompt_and_metadata(task: DataItemDS1000 | Text) -> tuple[Text, dict]:
        if isinstance(task, str):
            return task, {}

        return task.prompt, task.metadata

    @staticmethod
    def _library_filtered(chunks: List[Chunk], task_metadata: dict | None = None) -> List[Chunk]:
        task_metadata = task_metadata or {}
        doc_library = LIBRARY_TO_DOC_NAME.get(task_metadata.get("library"))
        if not doc_library:
            return chunks

        return [
            chunk
            for chunk in chunks
            if (chunk.metadata or {}).get("library") == doc_library
        ]

    @staticmethod
    def _with_retrieval_metadata(chunk: Chunk, api: str) -> Chunk:
        metadata = dict(chunk.metadata or {})
        metadata["retrieval_api"] = api
        return replace(chunk, metadata=metadata)
