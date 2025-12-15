import random
import heapq
import numpy as np

from openai import OpenAI
from typing import List, Set

from src.agent_constructor.core import Text


class Lens:

    def __init__(self, 
                 model_name: str = None, 
                 url: str = None,
                 prog_factor: int = 2,
                 init_score_set_size: int = 20,
                 candidates_num: int = 50,
                 diversity_weight: float = 0.5,
                 iter_num: int = 1,
                 beam_size: int = 10,
                 subs_size: int = 5) -> None:
        
        self.client = OpenAI(base_url=url, api_key='vllm')
        self.model_name = model_name
        self.prog_factor = prog_factor
        self.init_score_set_size = init_score_set_size
        self.candidates_num = candidates_num
        self.diversity_weight = diversity_weight
        self.iter_num = iter_num
        self.beam_size = beam_size
        self.subs_size = subs_size
        self.score_set = []

    def run(self, examples: List[Text]) -> List[Text]:

        if not examples:
            return []

        informative_examples = self.filter(
            examples=examples,
            prog_factor=self.prog_factor,
            init_score_size=self.init_score_set_size,
            candidate_size=self.candidates_num,
        )

        # Не используется далее: validation_set возможно не нужен, если дальше не используется
        validation_set = set(examples) - set(informative_examples)

        best_permutation = self.search(
            examples=informative_examples,
            validation_set=examples,
            iter_num=self.iter_num,
            beam_size=self.beam_size,
            subs_size=self.subs_size,
        )

        return best_permutation

    def filter(self, 
               examples: List[Text], 
               prog_factor: int, 
               init_score_size: int, 
               candidate_size: int):
        
        informative_examples = examples.copy()
        score_set = set(random.sample(examples, init_score_size))

        while len(informative_examples) > candidate_size:
            scores = []
            for example in informative_examples:
                score = self._info_score(example, relative_set=score_set)
                heapq.heappush(scores, (-score, example))
                # heapq можно заменить на list + sort для простоты, так как количество элементов обычно не очень большое

            if len(informative_examples)/prog_factor < candidate_size:
                top_m = heapq.nsmallest(candidate_size, scores)
                informative_examples = [item[1] for item in top_m]
                break

            else:
                top_p = heapq.nsmallest(int(len(informative_examples)/prog_factor), scores)
                informative_examples = [item[1] for item in top_p]

            random_sample = random.sample(examples, init_score_size*(prog_factor-1))

            # check for unique
            score_set.update(random_sample)

        self.score_set = score_set
        return informative_examples

    # TO-DO: Problem with dublecates (when getting e_new)
    def search(self, 
               examples: List[Text], 
               validation_set: List[Text], 
               iter_num: int, 
               beam_size: int, 
               subs_size: int):

        permutations = self._get_permutations(examples, beam_size)

        for _ in range(iter_num):
            new_perm = []
            for examples_set in permutations:
                for _ in range(subs_size):
                    random_example = random.choice(examples_set)

                    set_wo_random = [item for item in examples_set if item != random_example]

                    diversities = [(self._diversity(e_, example_set=set_wo_random), e_) for e_ in examples]
                    new_example = max(diversities, key=lambda x: x[0])[1]

                    new_set = examples_set.copy()
                    idx = new_set.index(random_example)
                    new_set[idx] = new_example

                    new_perm.append(new_set)

                for _ in range(beam_size - subs_size):
                    permutated_set = self._get_permutations(examples_set, 1)[0]
                    new_perm.append(permutated_set)
            
            permutations = self._eval(new_perm, validation_set, beam_size)
            # _eval ничего не делает кроме среза, но должна ранжировать new_perm по целевой метрике

        return permutations[0]
    
    def _diversity(self, example: Text, example_set: List[Text]):
        score = self._info_score(example, relative_set=self.score_set)

        # THIS IS PLUG
        similarities = [random.uniform(0,1) for _ in example_set]
        # Использовать случайные значения вместо реальной метрики схожести — лучше заменить на вычисление эмбеддингов или расстояния

        return score - self.diversity_weight * (sum(similarities))

    def _vectorize(exmple: Text):
        return []
        # Не используется. Можно реализовать через модель эмбеддингов или удалить

    def _eval(self, eval_set, validation_set, beam_size):
        return eval_set[:beam_size]
        # Это заглушка. Хорошо бы реализовать ранжирование permutation наборов по проверочной метрике

    def _get_permutations(self, examples, beam_size):
        return [random.sample(examples, len(examples)) for _ in range(beam_size)]
        # random.sample требует, чтобы len(examples) >= len(examples); если это не так, упадёт с ошибкой

    def _info_score(self, example: Text, relative_set: Set[Text]):
        if self.model_name == None:
            return random.uniform(0,1)
            # Лучше бросить ошибку или предупреждение, если model_name не задан, чтобы выявить ошибку конфигурации

        score = 0
        for re in relative_set:
            # 1) CALCULATE PPL FOR RELATIVE EXAMPLE 
            # WITHOUT TARGET EXAMPLE AS CONTEXT

            # this operation can be cached
            # for every re we can save it perplexity
            # also i need to devide example task and example results
            response = self.client.completions.create(
                model=self.model_name,
                prompt=re,
                max_tokens=100,
                logprobs=True,
            )
            all_probs = [item for item in response.choices[0].logprobs]
            entropy = -np.mean(all_probs)
            ppl = np.exp(entropy)

            # 2) CALCULATE PPL FOR RELATIVE EXAMPLE
            # WITH TARGET EXAMPLE AS CONTEXT
            response = self.client.completions.create(
                model=self.model_name,
                prompt=re+example,
                max_tokens=100,
                logprobs=True,
            )

            all_probs = [item for item in response.choices[0].logprobs]
            entropy = -np.mean(all_probs)

            # DIFFERNCE BETWEEN TWO PPLs AS SCORE
            ppl = ppl - np.exp(entropy)

            # SUM ON ALL REALTIVE EXAMPLES
            score += ppl

        return score
        # Функция очень медленная: для каждого примера вызывает два запроса к модели. Нужно сделать кэширование PPL и батчинг запросов
        # Возможно стоит явно отделить контекст (task) и ожидаемый результат (label) при формировании prompt
