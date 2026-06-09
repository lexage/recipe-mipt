"""Topic-aware cleaners (F3.1, F3.2) — F2.x + an off-topic DOCUMENT gate.

STEP 0 of filtering: drop whole documents whose topic does not match the
benchmark domain (e.g. gossip / physics injected into a data-science corpus).
Cheap gate + LLM confirm:
  * embed the document, compare to the `benchmark_topic` embedding;
  * if clearly on-topic (cosine >= topic_gate) -> keep, NO LLM;
  * if far -> ask the LLM "does this belong to <benchmark_topic>?" — drop on "no".
So the LLM runs only on the suspicious tail, not the whole corpus.

`benchmark_topic` is a CONFIG string (kept universal: change it for another
benchmark) injected into the LLM system prompt.

  * F3.1 `TopicSelectCleaner` = topic gate + F2.1 (drop bad-code chunks).
  * F3.2 `TopicLLMCleaner`    = topic gate + F2.2 (LLM repairs bad code).

NB: `embedder: Agent` annotated on each __init__; no **kwargs.
"""

import logging

from src.agent_constructor.agent import Agent
from src.filtering.code_aware.main import CodeAwareSelectCleaner, CodeAwareLLMCleaner


logger = logging.getLogger(__name__)


class _TopicMixin:
    """Off-topic document gate (embedding pre-filter + LLM confirm)."""

    def _topic_init(self, benchmark_topic, topic_gate, llm_url, llm_model):
        self.benchmark_topic = benchmark_topic or ""
        self.topic_gate = float(topic_gate)
        self.t_llm_url = llm_url
        self.t_llm_model = llm_model
        self._topic_emb = None
        self._t_client = None
        self._kept = 0
        self._dropped = 0

    def _topic_vec(self):
        if self._topic_emb is None:
            self._topic_emb = self._embed([self.benchmark_topic])[0]
        return self._topic_emb

    def _t_client_(self):
        if self._t_client is None:
            from openai import OpenAI
            self._t_client = OpenAI(base_url=self.t_llm_url, api_key="vllm")
        return self._t_client

    def _llm_on_topic(self, text: str) -> bool:
        try:
            resp = self._t_client_().chat.completions.create(
                model=self.t_llm_model,
                temperature=0.0,
                max_tokens=3,
                messages=[
                    {"role": "system", "content":
                     f"Does the document belong to the topic: {self.benchmark_topic!r}? "
                     "Answer strictly with yes or no."},
                    {"role": "user", "content": text[:2000]},
                ],
            )
            ans = (resp.choices[0].message.content or "").strip().lower()
            return not ans.startswith("n")          # keep unless a clear "no"
        except Exception as exc:                     # noqa: BLE001
            logger.warning("_TopicMixin: LLM topic check failed (%s) — keeping", exc)
            return True

    def _document_allowed(self, doc_id, lines) -> bool:
        if not self.benchmark_topic:
            return True
        text = "\n".join(lines).strip()
        if not text:
            return True
        demb = self._embed([text[:2000]])[0]
        sim = float(demb @ self._topic_vec())
        if sim >= self.topic_gate:                   # clearly on-topic
            self._kept += 1
            return True
        ok = self._llm_on_topic(text)                # far -> LLM decides
        if ok:
            self._kept += 1
        else:
            self._dropped += 1
        return ok


class TopicSelectCleaner(_TopicMixin, CodeAwareSelectCleaner):
    """F3.1 — off-topic gate + F2.1 (drop bad-code chunks)."""

    def __init__(self, embedder: Agent, benchmark_topic: str = "",
                 topic_gate: float = 0.4, llm_url: str = None, llm_model: str = None,
                 bad_code_ratio: float = 0.5, name: str = "topic_select_filter") -> None:
        super().__init__(embedder=embedder, bad_code_ratio=bad_code_ratio, name=name)
        self._topic_init(benchmark_topic, topic_gate, llm_url, llm_model)


class TopicLLMCleaner(_TopicMixin, CodeAwareLLMCleaner):
    """F3.2 — off-topic gate + F2.2 (LLM repairs bad code)."""

    def __init__(self, embedder: Agent, benchmark_topic: str = "",
                 topic_gate: float = 0.4, llm_url: str = None, llm_model: str = None,
                 bad_code_ratio: float = 0.5, max_context_chars: int = 4000,
                 name: str = "topic_llm_filter") -> None:
        super().__init__(embedder=embedder, llm_url=llm_url, llm_model=llm_model,
                         bad_code_ratio=bad_code_ratio, max_context_chars=max_context_chars,
                         name=name)
        self._topic_init(benchmark_topic, topic_gate, llm_url, llm_model)
