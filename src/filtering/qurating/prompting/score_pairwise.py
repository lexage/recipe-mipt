from transformers import AutoTokenizer
import numpy as np
from typing import List, Optional, Tuple
import os
import sys
sys.path.insert(0, os.getcwd())
from src.filtering.qurating.prompting.openai_util import query_openai
from src.filtering.qurating.configs.config import PairwiseComparisonConfig
# from src.agent_constructor.core import Document
from src.agent_constructor.core import Chunk
from datasets import Dataset
from operator import attrgetter


class Comparator:
    def __init__(
        self,
        template_file: str,
        config: PairwiseComparisonConfig
    ):
        self.config = config
        
        if template_file:
            with open(template_file) as f:
                self.config.template = f.read()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, use_fast=True)
        
        self.offset = 0
        self.num_examples = 0

    def __getstate__(self):
        return self.config
    
    def __setstate__(self, state):
        self.__init__(state)
        
    def extract_excerpt(self, text, index, num_tokens, token_ids=None):
        if token_ids is None:
            # heuristic for faster tokenization
            max_character_length = self.config.tokens_max * 40
            if len(text) > max_character_length:
                np.random.seed(self.config.seed + index + self.offset + 1)
                start_pos = np.random.randint(0, len(text) - max_character_length + 1)
                text = text[start_pos:start_pos + max_character_length]

            token_ids = self.tokenizer(text, truncation=False, padding=False, add_special_tokens=False).input_ids

        if len(token_ids) <= self.config.tokens_max:
            return text

        np.random.seed(self.config.seed + index + self.offset)
        start_pos = np.random.randint(0, len(token_ids) - self.config.tokens_max + 1)
        token_ids = token_ids[start_pos:start_pos + num_tokens]
        return self.tokenizer.decode(token_ids)

    def parse_generations(self, generations):
        for generation in generations:
            if generation == self.config.labels[0]:
                yield 0
            elif generation == self.config.labels[1]:
                yield 1

    def sample_num_tokens(self, indices):
        np.random.seed(self.config.seed + sum(indices) + self.offset)
        # use length tokens_max with probability probability_tokens_max, otherwise sample uniformly from [tokens_min, tokens_max]
        if self.config.probability_tokens_max > 0 and np.random.rand() < self.config.probability_tokens_max:
            return self.config.tokens_max
        else:
            return np.random.randint(self.config.tokens_min, self.config.tokens_max + 1)

    def __call__(self, examples, indices):
        num_tokens = self.sample_num_tokens(indices)

        if self.config.token_field in examples:
            texts = [self.extract_excerpt(text, index, num_tokens, token_ids) for text, token_ids, index in zip(examples[self.config.text_field], examples[self.config.token_field], indices)]
        else:
            texts = [self.extract_excerpt(text, index, num_tokens) for text, index in zip(examples[self.config.text_field], indices)]

        n = len(texts)
        votes_a = np.zeros((n, n), dtype=np.int32)
        votes_b = np.zeros((n, n), dtype=np.int32)
        predictions = np.full((n, n), -100, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                prompt = self.config.template.format(
                    text_a=texts[i],
                    text_b=texts[j],
                    label_a=self.config.labels[0],
                    label_b=self.config.labels[1])

                generations = query_openai(
                    prompt,
                    self.config.model,
                    system_prompt=self.config.system_prompt,
                    generations=self.config.generations,
                    labels=self.config.labels)

                for vote in self.parse_generations(generations):
                    if vote == 0:
                        votes_a[i, j] += 1
                    elif vote == 1:
                        votes_b[i, j] += 1

        np.divide(votes_b, votes_a + votes_b, out=predictions,
                  where=votes_a + votes_b > self.config.generations // 2)
        calibrated_predictions = np.where(
            (predictions != -100) & (predictions.T != -100),
            (predictions + (1 - predictions.T)) / 2,
            -100)

        if not self.config.flat_output_format:
            return {
                "indices": [indices],
                "examples": [examples],
                "texts": [texts],
                "votes_a": [votes_a.tolist()],
                "votes_b": [votes_b.tolist()],
                "average": [calibrated_predictions.tolist()],
            }
        else:
            indices_a, indices_b = np.where(np.triu(np.ones((n, n)), k=1))
            return {
                "index_a": indices_a,
                "index_b": indices_b,
                "texts_a": [texts[i] for i in indices_a],
                "texts_b": [texts[j] for j in indices_b],
                "comparisons_forward": predictions[indices_a, indices_b].tolist(),
                "comparisons_backward": (1-predictions[indices_b, indices_a]).tolist(),
                "comparisons_avg": calibrated_predictions[indices_a, indices_b].tolist(),
            }

    def apply(self, dataset):
        if self.config.num_examples_proportion is not None:
            offset = int(len(dataset) * self.config.num_examples_proportion_start)
            num_examples = int(len(dataset) * self.config.num_examples_proportion)
        else:
            offset = self.config.offset
            num_examples = self.config.num_examples
        self.offset = (offset // self.config.num_compare_all) * self.config.num_compare_all
        self.num_examples = (num_examples // self.config.num_compare_all) * self.config.num_compare_all

        print("Example offset", self.offset)
        print("Total number of pairwise comparisons:", self.num_examples * (self.config.num_compare_all - 1))

        return (
            dataset
                .select(range(self.offset, self.offset + self.num_examples))
                .map(self.__call__,
                     with_indices=True,
                     batched=True,
                     batch_size=self.config.num_compare_all,
                     remove_columns=dataset.column_names)
        )
        
def collect_judgement_data(
    chunks: List[Chunk],
    template_file: str,
    output_path: str
):  
    config = PairwiseComparisonConfig()
                
    dataset_dict = {
        "id": list(map(attrgetter('id'), chunks)),
        "doc_id": list(map(attrgetter('doc_id'), chunks)),
        "text": list(map(attrgetter('text'), chunks)),
        "tokens": list(map(attrgetter('tokens'), chunks)),
        "metadata": list(map(attrgetter('metadata'), chunks)),
    }
        
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset = Comparator(template_file, config).apply(dataset)
    
    print(f"Saving to {output_path}")
    dataset.save_to_disk(output_path)