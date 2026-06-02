"""PerplexityFilter — Idea №5 from the experimental fileset.

For each chunk, computes the average negative log-likelihood per token
using a hosted LLM (we use the same Qwen2.5-32B-Instruct that runs the
solver).  The filter then keeps chunks whose perplexity lands in the
"interesting middle":

    low-quantile  <=  perplexity  <=  high-quantile

The intuition:

    * very low PPL  → predictable boilerplate (imports, repeated patterns)
    * very high PPL → noise / OCR errors / non-English / broken text
    * middle        → informative, non-trivial content

The filter runs once at pipeline build time (no caching, as requested).
Uses an OpenAI-compatible /v1/completions endpoint with
`echo=True, max_tokens=0, logprobs=1` to fetch per-token log-probs of
the input prompt itself — no generation.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import openai
from tqdm import tqdm

from src.agent_constructor.core import Chunk
from src.agent_constructor.filters import Filter


logger = logging.getLogger(__name__)


class PerplexityFilter(Filter):
    """Drop chunks at the tails of the per-token perplexity distribution.

    Args:
        llm_url: OpenAI-compatible base URL (vLLM server).
        llm_model: model name as registered on the server.
        api_key: API key (use "vllm" for unauthenticated local servers).
        low_quantile: chunks below this perplexity quantile are dropped
            (boilerplate).
        high_quantile: chunks above this perplexity quantile are dropped
            (noise / non-Python).
        concurrent_requests: how many chunks to score in parallel.
        max_chars: truncate chunks longer than this before scoring
            (defensive — chunks are already ≤1000 chars by chunker).
        request_timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        llm_url: str,
        llm_model: str,
        api_key: str = "vllm",
        low_quantile: float = 0.20,
        high_quantile: float = 0.95,
        concurrent_requests: int = 8,
        max_chars: int = 4000,
        request_timeout: float = 60.0,
        name: str = "perplexity_filter",
    ) -> None:
        if not 0.0 <= low_quantile < high_quantile <= 1.0:
            raise ValueError(
                "Expect 0 <= low_quantile < high_quantile <= 1, "
                f"got {low_quantile}, {high_quantile}"
            )
        self.client = openai.OpenAI(base_url=llm_url, api_key=api_key,
                                    timeout=request_timeout)
        self.model = llm_model
        self.low_quantile = float(low_quantile)
        self.high_quantile = float(high_quantile)
        self.concurrent_requests = int(concurrent_requests)
        self.max_chars = int(max_chars)
        self.name = name

    # ------------------------------------------------------------------
    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        n = len(chunks)
        if n == 0:
            return []

        ppls = self._compute_perplexities(chunks)

        valid_mask = np.isfinite(ppls)
        if valid_mask.sum() == 0:
            logger.warning(
                "PerplexityFilter: every chunk failed scoring — "
                "keeping the corpus as-is."
            )
            return list(chunks)

        valid_ppls = ppls[valid_mask]
        low_thr = float(np.quantile(valid_ppls, self.low_quantile))
        high_thr = float(np.quantile(valid_ppls, self.high_quantile))

        kept: List[Chunk] = []
        dropped_low = dropped_high = dropped_failed = 0
        for chunk, ppl in zip(chunks, ppls):
            if not math.isfinite(ppl):
                # Scoring failed — keep the chunk by default (safer than
                # dropping content we have no info about).
                kept.append(chunk)
                dropped_failed += 1
                continue
            if ppl < low_thr:
                dropped_low += 1
                continue
            if ppl > high_thr:
                dropped_high += 1
                continue
            kept.append(chunk)

        logger.info(
            "PerplexityFilter: kept %d / %d "
            "(dropped low=%d high=%d, scoring failed=%d kept-by-default). "
            "Thresholds: low=%.2f (q=%.2f), high=%.2f (q=%.2f).",
            len(kept), n, dropped_low, dropped_high, dropped_failed,
            low_thr, self.low_quantile, high_thr, self.high_quantile,
        )
        return kept

    # ------------------------------------------------------------------
    def _compute_perplexities(self, chunks: List[Chunk]) -> np.ndarray:
        """Return an array of per-chunk perplexities (NaN where scoring failed)."""
        n = len(chunks)
        ppls = np.full(n, np.nan, dtype=np.float64)

        def _job(idx: int) -> tuple[int, Optional[float]]:
            try:
                text = (chunks[idx].text or "")[: self.max_chars]
                if not text.strip():
                    return idx, None
                return idx, self._score_one(text)
            except Exception as e:
                logger.debug("PerplexityFilter: chunk %d failed: %s", idx, e)
                return idx, None

        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as pool:
            futures = [pool.submit(_job, i) for i in range(n)]
            for fut in tqdm(as_completed(futures), total=n,
                            desc="PerplexityFilter scoring"):
                idx, ppl = fut.result()
                if ppl is not None:
                    ppls[idx] = ppl

        return ppls

    # ------------------------------------------------------------------
    def _score_one(self, text: str) -> float:
        """Return perplexity for a single chunk."""
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
        # First token has no predecessor → logprob is None; skip it.
        valid = [lp for lp in token_logprobs if isinstance(lp, (int, float))]
        if len(valid) < 2:
            raise RuntimeError("not enough logprobs returned")
        avg_neg_logprob = -sum(valid) / len(valid)
        return math.exp(avg_neg_logprob)
