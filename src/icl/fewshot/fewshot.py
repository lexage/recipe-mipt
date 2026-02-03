import numpy as np

from typing import List
from openai import OpenAI

from src.agent_constructor.core import Text
from src.agent_constructor.icl import ICLBlock
from src.agent_constructor.context_engine import Chunk


class FewShot(ICLBlock):

    def __init__(self, url: str, model_name: str, k: int = 5) -> None:
        self.model_name = model_name
        self.client = OpenAI(base_url=url, api_key='vllm')
        self.k = k

    def _ppl(self, prompt: Text):
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=100,
            logprobs=True,
        )
        all_probs = [p for p in response.choices[0].logprobs.token_logprobs]
        entropy = -np.mean(all_probs)
        return np.exp(entropy)

    def _score_pair(self, ctx: Chunk, target: Chunk):
        ppl_base = self._ppl(target.text)
        ppl_cond = self._ppl(ctx.text + "\n" + target.text)
        return ppl_base - ppl_cond

    def _score_single(self, chunk: Chunk, all_chunks: List[Chunk]):
        others = [x for x in all_chunks if x is not chunk]
        score = 0
        for t in others:
            score += self._score_pair(chunk, t)
        return score

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        if self.k >= len(chunks):
            return chunks

        scores = []
        for chunk in chunks:
            s = self._score_single(chunk, chunks)
            scores.append((s, chunk))

        scores.sort(key=lambda x: -x[0])
        best = [chunk for i, chunk in scores[:self.k]]

        return best
