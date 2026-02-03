import numpy as np

from typing import List
from openai import OpenAI

from src.agent_constructor.core import Text
from src.agent_constructor.context_engine import Chunk
from src.agent_constructor.icl import ICLBlock


class ICCL(ICLBlock):

    def __init__(self, url: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(base_url=url, api_key='vllm')

    def _eval_example(self, example: Text):
        response = self.client.completions.create(
            model=self.model_name,
            temperature=0,
            prompt=example,
            max_tokens=100,
            logprobs=True,
        )
        all_probs = [item for item in response.choices[0].logprobs.token_logprobs]
        entropy = -np.mean(all_probs)
        return np.exp(entropy)

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        valid = [
            (chunk, comp)
            for chunk in chunks
            if (comp := self._eval_example(chunk.text)) is not None and comp > 0
        ]

        return [chunk for chunk, _ in sorted(valid, key=lambda x: x[1])]
