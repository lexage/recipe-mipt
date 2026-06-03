"""DocumentPerplexityFilter — document-level version of PerplexityFilter.

Scores each Document by average per-token NLL of its first `max_chars`
characters via the same Qwen2.5-32B used elsewhere.  Documents in the
"interesting middle" of the per-document perplexity distribution are
kept; the very low (boilerplate-heavy) and very high (broken / non-code)
tails are dropped.

Much cheaper than the chunk-level PerplexityFilter: a few thousand
documents instead of 41k chunks, and on longer windows perplexity is
better calibrated.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import openai
from tqdm import tqdm

from src.agent_constructor.core import Document
from src.agent_constructor.filters import DocumentFilter


logger = logging.getLogger(__name__)


class DocumentPerplexityFilter(DocumentFilter):
    def __init__(
        self,
        llm_url: str,
        llm_model: str,
        api_key: str = "vllm",
        low_quantile: float = 0.10,
        high_quantile: float = 0.95,
        concurrent_requests: int = 8,
        max_chars: int = 2000,
        request_timeout: float = 120.0,
        name: str = "doc_perplexity_filter",
    ) -> None:
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError(
                f"Expect 0 <= low < high <= 1, got {low_quantile}, {high_quantile}"
            )
        self.client = openai.OpenAI(base_url=llm_url, api_key=api_key,
                                    timeout=request_timeout)
        self.model = llm_model
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        self.concurrent_requests = int(concurrent_requests)
        self.max_chars = int(max_chars)
        self.name = name

    def apply(self, documents: List[Document]) -> List[Document]:
        n = len(documents)
        if n == 0:
            return []

        ppls = self._compute_perplexities(documents)
        valid_mask = np.isfinite(ppls)
        if valid_mask.sum() == 0:
            logger.warning(
                "DocumentPerplexityFilter: every document failed scoring — "
                "keeping the corpus as-is."
            )
            return list(documents)

        valid_ppls = ppls[valid_mask]
        low_thr = float(np.quantile(valid_ppls, self.low_quantile))
        high_thr = float(np.quantile(valid_ppls, self.high_quantile))

        kept: List[Document] = []
        dropped_low = dropped_high = scoring_failed = 0
        for doc, ppl in zip(documents, ppls):
            if not math.isfinite(ppl):
                kept.append(doc)
                scoring_failed += 1
                continue
            if ppl < low_thr:
                dropped_low += 1
                continue
            if ppl > high_thr:
                dropped_high += 1
                continue
            kept.append(doc)

        logger.info(
            "DocumentPerplexityFilter: kept %d / %d "
            "(dropped low=%d high=%d, scoring failed kept=%d). "
            "Thresholds: low=%.2f (q=%.2f), high=%.2f (q=%.2f).",
            len(kept), n, dropped_low, dropped_high, scoring_failed,
            low_thr, self.low_quantile, high_thr, self.high_quantile,
        )
        return kept

    def _compute_perplexities(self, documents: List[Document]) -> np.ndarray:
        n = len(documents)
        ppls = np.full(n, np.nan, dtype=np.float64)

        def _job(idx: int):
            try:
                text = (documents[idx].text or "")[: self.max_chars]
                if not text.strip():
                    return idx, None
                return idx, self._score_one(text)
            except Exception as e:
                logger.debug("DocPerplexity: doc %d failed: %s", idx, e)
                return idx, None

        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as pool:
            futures = [pool.submit(_job, i) for i in range(n)]
            for fut in tqdm(as_completed(futures), total=n,
                            desc="DocumentPerplexityFilter scoring"):
                idx, ppl = fut.result()
                if ppl is not None:
                    ppls[idx] = ppl

        return ppls

    def _score_one(self, text: str) -> float:
        response = self.client.completions.create(
            model=self.model,
            prompt=text,
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        choice = response.choices[0]
        token_logprobs = getattr(choice.logprobs, "token_logprobs", None) or []
        valid = [lp for lp in token_logprobs if isinstance(lp, (int, float))]
        if len(valid) < 2:
            raise RuntimeError("not enough logprobs returned")
        avg_neg_logprob = -sum(valid) / len(valid)
        return math.exp(avg_neg_logprob)
