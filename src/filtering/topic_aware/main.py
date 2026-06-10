"""Topic-aware cleaners (F3.1, F3.2) — F2.x + an off-topic DOCUMENT gate.

STEP 0 of filtering: drop whole documents whose topic does not match the
benchmark domain (e.g. gossip / physics injected into a data-science corpus).

This runs as ONE batch pre-pass over all docs (`_prepare_documents`), NOT
per-doc, so it scales:

  1. batch-embed every document, take cosine to the `benchmark_topic` vector;
  2. learn TWO thresholds FROM THE DATA (no hard-coded cosine) — see
     `_auto_topic_bands`: a low one (clearly off-topic -> drop, no LLM) and a
     high one (clearly on-topic -> keep, no LLM);
  3. only the few "uncertain" docs between the thresholds go to the LLM, and
     those run in parallel (`num_workers`).

`benchmark_topic` is a CONFIG string (universal: change it per benchmark). The
numeric cutoffs are derived from the corpus, controlled only by the scale-free
`band_z` multiplier (same idea as the base filter's `sep_z` / `char_z`).

  * F3.1 `TopicSelectCleaner` = topic gate + F2.1 (drop bad-code chunks).
  * F3.2 `TopicLLMCleaner`    = topic gate + F2.2 (LLM repairs bad code).

NB: `embedder: Agent` annotated on each __init__; no **kwargs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from src.agent_constructor.agent import Agent
from src.filtering.code_aware.main import CodeAwareSelectCleaner, CodeAwareLLMCleaner
from src.utils.token_tracker import get_active, set_active


logger = logging.getLogger(__name__)


class _TopicMixin:
    """Off-topic document gate: batch-embed + auto thresholds + parallel LLM."""

    def _topic_init(self, benchmark_topic, llm_url, llm_model,
                    num_workers, band_z, topic_gate):
        self.benchmark_topic = benchmark_topic or ""
        self.t_llm_url = llm_url
        self.t_llm_model = llm_model
        self.num_workers = int(num_workers)
        self.band_z = float(band_z)
        # Optional manual override of the high (keep) threshold; None -> auto.
        self.topic_gate_override = None if topic_gate is None else float(topic_gate)
        self._topic_emb = None
        self._t_client = None
        self._doc_verdict = {}          # doc_id -> keep? (filled by pre-pass)

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

    # ---- auto thresholds -------------------------------------------------

    def _embed_docs(self, texts):
        """Batch-embed every doc (one embedder call per `embed_batch_size`)."""
        bs = self.embed_batch_size
        out = []
        for s in tqdm(range(0, len(texts), bs), desc=f"{self.name}:topic-embed"):
            out.append(self._embed(texts[s:s + bs]))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)

    def _auto_topic_bands(self, cos):
        """Learn (drop, keep) cosine thresholds from the distribution itself.

        Otsu/Fisher 1-D split (same objective as the base filter's `_keep_mask`)
        finds the boundary between the off-topic and on-topic humps; the band
        around it is `band_z` within-hump std-devs wide. Returns (None, None)
        when the corpus is NOT clearly two-humped (e.g. a clean corpus) — then
        we drop nobody without the LLM."""
        m = len(cos)
        if m < 4:
            return None, None
        order = np.argsort(cos)
        cs = cos[order]
        csum = np.cumsum(cs)
        total = float(csum[-1])
        best_t, best_sep = 1, -1.0
        for t in range(1, m):
            s1 = float(csum[t - 1])
            mu1 = s1 / t
            mu2 = (total - s1) / (m - t)
            sep = t * (m - t) * (mu2 - mu1) ** 2
            if sep > best_sep:
                best_sep, best_t = sep, t
        low = cos[order[:best_t]]
        high = cos[order[best_t:]]
        mu_low, std_low = float(low.mean()), float(low.std())
        mu_high, std_high = float(high.mean()), float(high.std())
        # Separation guard: the two humps must stand apart by > band_z stds,
        # else it's one hump (no off-topic) -> don't auto-drop anyone.
        if mu_high - mu_low <= self.band_z * (std_low + std_high):
            return None, None
        topic_drop = mu_low + self.band_z * std_low      # below -> clearly off-topic
        topic_gate = mu_high - self.band_z * std_high     # above -> clearly on-topic
        if topic_drop >= topic_gate:                      # humps far apart, bands cross
            mid = 0.5 * (mu_low + mu_high)                # -> hard split, no LLM band
            topic_drop = topic_gate = mid
        return topic_drop, topic_gate

    # ---- batch pre-pass --------------------------------------------------

    def _prepare_documents(self, doc_all) -> None:
        if not self.benchmark_topic:
            self._doc_verdict = {}
            return

        doc_ids = list(doc_all.keys())
        if not doc_ids:
            self._doc_verdict = {}
            return
        texts = ["\n".join(doc_all[d]).strip()[:2000] for d in doc_ids]
        emb = self._embed_docs(texts)
        cos = emb @ self._topic_vec()

        if self.topic_gate_override is not None:          # manual single threshold
            drop_t, gate_t = float("-inf"), self.topic_gate_override
        else:
            drop_t, gate_t = self._auto_topic_bands(cos)

        verdict = {}
        if drop_t is None:                                # not two-humped -> keep all
            for d in doc_ids:
                verdict[d] = True
            self._doc_verdict = verdict
            logger.info("_TopicMixin: corpus not two-humped on topic — keeping all "
                        "%d docs, no LLM", len(doc_ids))
            return

        band = []                                         # (idx, doc_id) needing LLM
        for i, d in enumerate(doc_ids):
            c = float(cos[i])
            if c >= gate_t:
                verdict[d] = True                         # clearly on-topic
            elif c < drop_t:
                verdict[d] = False                        # clearly off-topic
            else:
                band.append((i, d))                       # uncertain -> LLM

        if band:                                          # parallel LLM on the band
            results = [True] * len(band)
            tracker = get_active()
            with ThreadPoolExecutor(max_workers=self.num_workers,
                                    initializer=set_active,
                                    initargs=(tracker,)) as ex:
                fut_to_k = {ex.submit(self._llm_on_topic, texts[i]): k
                            for k, (i, _d) in enumerate(band)}
                for fut in tqdm(as_completed(fut_to_k), total=len(fut_to_k),
                                desc=f"{self.name}:topic-llm"):
                    results[fut_to_k[fut]] = fut.result()
            for (i, d), keep in zip(band, results):
                verdict[d] = keep

        self._doc_verdict = verdict
        kept = sum(verdict.values())
        logger.info("_TopicMixin: topic gate kept %d / %d docs (dropped %d; LLM on "
                    "%d uncertain; drop<%.3f keep>=%.3f, band_z=%.2f)",
                    kept, len(verdict), len(verdict) - kept, len(band),
                    drop_t, gate_t, self.band_z)

    def _document_allowed(self, doc_id, lines) -> bool:
        return self._doc_verdict.get(doc_id, True)


class TopicSelectCleaner(_TopicMixin, CodeAwareSelectCleaner):
    """F3.1 — off-topic gate + F2.1 (drop bad-code chunks)."""

    def __init__(self, embedder: Agent, benchmark_topic: str = "",
                 llm_url: str = None, llm_model: str = None,
                 bad_code_ratio: float = 0.5, num_workers: int = 4,
                 band_z: float = 1.0, topic_gate: float = None,
                 name: str = "topic_select_filter") -> None:
        super().__init__(embedder=embedder, bad_code_ratio=bad_code_ratio, name=name)
        self._topic_init(benchmark_topic, llm_url, llm_model,
                         num_workers, band_z, topic_gate)


class TopicLLMCleaner(_TopicMixin, CodeAwareLLMCleaner):
    """F3.2 — off-topic gate + F2.2 (LLM repairs bad code)."""

    def __init__(self, embedder: Agent, benchmark_topic: str = "",
                 llm_url: str = None, llm_model: str = None,
                 bad_code_ratio: float = 0.5, max_context_chars: int = 4000,
                 num_workers: int = 4, band_z: float = 1.0,
                 topic_gate: float = None, name: str = "topic_llm_filter") -> None:
        super().__init__(embedder=embedder, llm_url=llm_url, llm_model=llm_model,
                         bad_code_ratio=bad_code_ratio, max_context_chars=max_context_chars,
                         num_workers=num_workers, name=name)
        self._topic_init(benchmark_topic, llm_url, llm_model,
                         num_workers, band_z, topic_gate)
