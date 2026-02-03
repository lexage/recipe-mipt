import random
import heapq
import numpy as np
import hashlib

from openai import OpenAI
from typing import List, Set
from sklearn.metrics.pairwise import cosine_similarity

from src.agent_constructor.core import Text
from src.agent_constructor.agent import Agent
from src.agent_constructor.context_engine import Chunk
from src.agent_constructor.icl import ICLBlock


class Lens(ICLBlock):

    _ppl_cache = {}
    _examples_features_cache = {}

    def __init__(self, 
                 model_name: str = None, 
                 url: str = None,
                 embedding_model: Agent = None,
                 filtered_set_size: int = 10,
                 search_set_size: int = 10,
                 search_iterations: int = 1,
                 diversity_weight: float = 0.5,
                 progressive_factor: int = 2,
                 init_score_set_size: int = 20,
                 beam_size: int = 10,
                 substitution_size: int = 5) -> None:
        
        self.client = OpenAI(base_url=url, api_key='vllm')
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        self.filtered_set_size = filtered_set_size
        if search_set_size > filtered_set_size:
            print(f"search_set_size > filtered_set_size : using search_set_size = {filtered_set_size}")
            self.search_set_size = filtered_set_size
        else: 
            self.search_set_size = search_set_size
        
        self.progressive_factor = progressive_factor
        self.init_score_set_size = init_score_set_size
        self.diversity_weight = diversity_weight
        self.search_iterations = search_iterations
        self.beam_size = beam_size
        self.substitution_size = substitution_size

        self.score_set = []

    def apply(self, chunks: List[Chunk]) -> List[Chunk]:
        print("=" * 60)
        print("running lens")
        print("=" * 60)
        print(f"total input examples: {len(chunks)}")

        if not chunks:
            print("warning: empty examples list, returning empty list")
            return []

        print("\n" + "-" * 60)
        print("stage 1: filtering")
        print("-" * 60)
        informative_examples = self.filter(
            chunks=chunks,
            progressive_factor=self.progressive_factor,
            init_score_size=self.init_score_set_size,
            candidate_size=self.filtered_set_size,
        )
        print(f"filtering completed. selected examples: {len(informative_examples)}")

        # Не используется далее: validation_set возможно не нужен, если дальше не используется
        validation_set = set(chunks) - set(informative_examples)
        print(f"validation set size: {len(validation_set)}")

        print("\n" + "-" * 60)
        print("stage 2: searching for best permutation")
        print("-" * 60)
        best_permutation = self.search(
            examples=informative_examples,
            validation_set=validation_set,
            search_set_size=self.search_set_size,
            search_iterations=self.search_iterations,
            beam_size=self.beam_size,
            substitution_size=self.substitution_size,
        )
        print(f"search completed. best permutation size: {len(best_permutation)}")
        print("=" * 60)

        return best_permutation

    def filter(self, 
               examples: List[Chunk], 
               progressive_factor: int, 
               init_score_size: int, 
               candidate_size: int) -> List[Chunk]:
        
        print(f"filtering parameters:")
        print(f"   - progressive_factor: {progressive_factor}")
        print(f"   - init_score_size: {init_score_size}")
        print(f"   - candidate_size: {candidate_size}")
        print(f"   - initial examples size: {len(examples)}")
        
        informative_examples = examples.copy()
        print(f"\ninitial informative_examples size: {len(informative_examples)}")
        
        score_set = set(random.sample(examples, init_score_size))
        print(f"initialized score_set with size: {len(score_set)}")

        iteration = 0
        while len(informative_examples) > candidate_size:
            iteration += 1
            print(f"\nfiltration iteration #{iteration}")
            print(f"   current informative_examples size: {len(informative_examples)}")
            print(f"   target size: {candidate_size}")
            
            print(f"   computing information scores for {len(informative_examples)} examples...")
            scores = []
            for idx, example in enumerate(informative_examples):
                score = self._info_score(example, relative_set=score_set)
                heapq.heappush(scores, (-score, example))
                if (idx + 1) % 10 == 0 or idx == len(informative_examples) - 1:
                    print(f"      processed examples: {idx + 1}/{len(informative_examples)}")

            if len(informative_examples)/progressive_factor < candidate_size:
                print(f"   final selection: selecting top {candidate_size} examples")
                top_m = heapq.nsmallest(candidate_size, scores)
                informative_examples = [item[1] for item in top_m]
                print(f"   final size: {len(informative_examples)}")
                break

            else:
                new_size = int(len(informative_examples)/progressive_factor)
                print(f"   intermediate selection: selecting top {new_size} examples")
                top_p = heapq.nsmallest(new_size, scores)
                informative_examples = [item[1] for item in top_p]
                print(f"   new informative_examples size: {len(informative_examples)}")

            random_sample_size = init_score_size*(progressive_factor-1)
            print(f"   adding {random_sample_size} random examples to score_set")
            random_sample = random.sample(examples, random_sample_size)

            # check for unique
            old_score_set_size = len(score_set)
            score_set.update(random_sample)
            print(f"   score_set size: {old_score_set_size} -> {len(score_set)}")

        self.score_set = score_set
        print(f"\nfiltering completed in {iteration} iterations")
        print(f"   final informative_examples size: {len(informative_examples)}")
        print(f"   final score_set size: {len(self.score_set)}")
        return informative_examples

    # TO-DO: Problem with dublecates (when getting e_new)
    def search(self, 
               examples: List[Chunk], 
               validation_set: List[Text],
               search_set_size: int, 
               search_iterations: int, 
               beam_size: int, 
               substitution_size: int) -> List[Chunk]:

        print(f"search parameters:")
        print(f"   - search_iterations: {search_iterations}")
        print(f"   - beam_size: {beam_size}")
        print(f"   - substitution_size: {substitution_size}")
        print(f"   - examples size: {len(examples)}")
        print(f"   - validation_set size: {len(validation_set)}")

        print(f"\ngenerating initial permutations (beam_size={beam_size})...")
        permutations = self._get_permutations(examples, search_set_size, beam_size)
        print(f"generated {len(permutations)} initial permutations")

        for iter_idx in range(search_iterations):
            print(f"\nsearch iteration #{iter_idx + 1}/{search_iterations}")
            new_perm = []
            
            print(f"   processing {len(permutations)} permutations from beam...")
            for perm_idx, examples_set in enumerate(permutations):
                print(f"      permutation {perm_idx + 1}/{len(permutations)}: size {len(examples_set)}")
                
                # Замена примеров на основе diversity
                print(f"         generating {substitution_size} new variants via diversity...")
                for sub_idx in range(substitution_size):
                    random_example = random.choice(examples_set)
                    set_wo_random = [item for item in examples_set if item != random_example]

                    print(f"            computing diversity for example replacement {sub_idx + 1}/{substitution_size}...")
                    diversities = [(self._diversity(e_, example_set=set_wo_random), e_) for e_ in examples if e_ not in set_wo_random]
                    new_example = max(diversities, key=lambda x: x[0])[1]

                    new_set = examples_set.copy()
                    idx = new_set.index(random_example)
                    new_set[idx] = new_example

                    new_perm.append(new_set)

                # Генерация случайных перестановок
                random_perm_count = beam_size - substitution_size
                if random_perm_count > 0:
                    print(f"         generating {random_perm_count} random permutations...")
                    ranom_permutations = self._get_permutations(examples_set, search_set_size, random_perm_count)
                    new_perm.extend(ranom_permutations)
            
            print(f"   total new permutations generated: {len(new_perm)}")
            print(f"   evaluating and selecting top {beam_size} permutations...")
            permutations = self._eval(new_perm, validation_set, beam_size)
            print(f"   selected {len(permutations)} best permutations")
            # _eval ничего не делает кроме среза, но должна ранжировать new_perm по целевой метрике

        print(f"\nsearch completed. returning best permutation (size: {len(permutations[0])})")
        return permutations[0]
    
    def _diversity(self, example: Chunk, example_set: List[Chunk]):
        score = self._info_score(example, relative_set=self.score_set)
        
        if len(example_set) == 0:
            return score
        elif self.embedding_model == None:
            similarities = [random.uniform(0,1) for _ in example_set]
        else:
            example_features = self._vectorize(example.text)
            set_features = [self._vectorize(set_item.text)[0] for set_item in example_set]
            similarities = cosine_similarity(example_features, set_features)[0]

        return score - self.diversity_weight * (np.sum(similarities))

    def _vectorize(self, example: Text):
        if self.embedding_model == None:
            return []
        
        example_hash = self._hash_string(example)
        if example_hash in self._examples_features_cache:
            return self._examples_features_cache[example_hash]
        
        features = self.embedding_model.run(example)
        self._examples_features_cache[example_hash] = features

        return features

    def _eval(self, eval_set, validation_set, beam_size):
        print(f"      warning: _eval using stub (returning first {beam_size} without ranking)")
        return eval_set[:beam_size]
        # Это заглушка. Хорошо бы реализовать ранжирование permutation наборов по проверочной метрике

    def _get_permutations(self, examples:List[Chunk], search_set_size:int, beam_size:int) -> List[List[Chunk]]:
        return [random.sample(examples, search_set_size) for _ in range(beam_size)]

    def _hash_string(self, st: str):
        return int(hashlib.md5(st.encode('utf-8')).hexdigest(), 16)

    def _info_score(self, example: Chunk, relative_set: Set[Chunk]):
        if self.model_name == None:
            return random.uniform(0,1)

        score = 0
        for re in relative_set:
            
            # 1) CALCULATE PPL FOR RELATIVE EXAMPLE 
            # WITHOUT TARGET EXAMPLE AS CONTEXT
            prompt_hash = self._hash_string(re.text)

            if prompt_hash in self._ppl_cache:
                ppl = self._ppl_cache[prompt_hash]
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    temperature=0,
                    prompt=re.text,
                    max_tokens=100,
                    logprobs=True,
                )
                all_probs = [item for item in response.choices[0].logprobs.token_logprobs]
                entropy = -np.mean(all_probs)
                ppl = np.exp(entropy)
                self._ppl_cache[prompt_hash] = ppl

            # 2) CALCULATE PPL FOR RELATIVE EXAMPLE
            # WITH TARGET EXAMPLE AS CONTEXT

            pair_prompt_hash = self._hash_string(re.text+example.text)
            if pair_prompt_hash in self._ppl_cache:
                pair_ppl = self._ppl_cache[pair_prompt_hash]
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    temperature=0,
                    prompt=re.text+example.text,
                    max_tokens=100,
                    logprobs=True,
                )

                all_probs = [item for item in response.choices[0].logprobs.token_logprobs]
                entropy = -np.mean(all_probs)
                pair_ppl = np.exp(entropy)
                self._ppl_cache[pair_prompt_hash] = pair_ppl

            # DIFFERNCE BETWEEN TWO PPLs AS SCORE
            ppl = ppl - pair_ppl

            # SUM ON ALL REALTIVE EXAMPLES
            score += ppl

        return score
