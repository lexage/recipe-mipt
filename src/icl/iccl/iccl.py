import numpy as np

from typing import List
from openai import OpenAI

from src.agent_constructor.core import Text


class ICCL:

    def __init__(self, url: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(base_url=url, api_key='vllm')
        pass

    def _eval_example(self, example: Text):
        response = self.client.completions.create(
            model=self.model_name,
            prompt=example,
            max_tokens=100,
            logprobs=True,
        )
        all_probs = [item for item in response.choices[0].logprobs]
        entropy = -np.mean(all_probs)
        return np.exp(entropy)

    def run(self, examples: List[Text]):
        complexity = [self._eval_example(e) for e in examples]
        sorted_examples = [ex for ex, comp in zip(examples, complexity) if comp is not None and comp > 0]
        return sorted_examples