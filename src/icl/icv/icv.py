
import requests

from typing import List
from openai import OpenAI

from src.agent_constructor.core import Text


class ICV:
    def __init__(self, url: str, model_name: str, weight : float = 1.0) -> None:
        self.model_name = model_name
        self.base_url = url.rstrip('/')
        self.client = OpenAI(base_url=url, api_key='vllm')
        self.weight = weight

    def run(self, examples: List[Text]):
        
        tokens_probs = {}

        for example in examples:
            
            example_probs = self._eval_example(example)

            for token_id, logprob in example_probs.items():
                tokens_probs[token_id] = tokens_probs.get(token_id, 0) + logprob

        result_icv_load = ""
        for token_id, logprob in tokens_probs.items():
            result_icv_load = result_icv_load + str(token_id) + ':' + str(round(logprob, 3))+';'



        return {"target_tokens" : result_icv_load, "weight" : self.weight}

    def _eval_example(self, example: Text):
        response = self.client.completions.create(
            model=self.model_name,
            temperature=0,
            prompt=example,
            max_tokens=1,
            logprobs=20,
        )
        top_logprobs = response.choices[0].logprobs.top_logprobs[0]
        token_ids = self._get_token_ids(list(top_logprobs.keys()))
        return {token_id: token_logprob for token_id, token_logprob in zip(token_ids, list(top_logprobs.values()))}



    def _get_token_ids(self, tokens: list):

        token_ids = []
        base_url_without_v1 = self.base_url.rstrip('/v1').rstrip('/')
        tokenize_url = f"{base_url_without_v1}/tokenize"
        
        for token in tokens:
            response = requests.post(
                tokenize_url,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "prompt": token,
                    "add_special_tokens": True,
                    "return_token_strs": False,
                },
            )
            response.raise_for_status()
            result = response.json()
            token_ids.append(result["tokens"][0])
        
        return token_ids

