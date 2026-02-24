import threading

from transformers import PreTrainedTokenizerFast
from typing import List


def batch_truncate(
        texts: List[str], tokenizer: PreTrainedTokenizerFast,
        max_length: int = 128, truncate_from_middle: bool = False,
        skip_special_tokens: bool = False
) -> List[str]:
    if not texts:
        return texts

    if truncate_from_middle:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=int(10**7))['input_ids']
        for idx in range(len(input_ids)):
            if len(input_ids[idx]) > max_length:
                half_length = max_length // 2
                input_ids[idx] = input_ids[idx][:half_length] + input_ids[idx][-half_length:]
    else:
        input_ids: List[List[int]] = tokenizer(texts, truncation=True, max_length=max_length)['input_ids']

    # Skip bos but keep other special tokens in the input texts
    if not skip_special_tokens:
        # bos_token_id = vllm_client.bos_token_id
        input_ids = [ids[1:] if ids[0] == tokenizer.bos_token_id else ids for ids in input_ids]

    # vllm_clinet.decode(input_ids)
    return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)


class AtomicCounter:
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

    def reset(self):
        self.value = 0
        return self.value
