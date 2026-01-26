import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (LogitsProcessor, 
                                             BatchUpdate,
                                             MoveDirectionality)


TARGET_TOKEN = "target_tokens"
WEIGHT = "weight"
DEFAULT_WEIGHT = 1.0

class ICVLogitProcessor(LogitsProcessor):

    @classmethod
    def validate_params(cls, params: SamplingParams):
        target_tokens = params.extra_args and params.extra_args.get(
            TARGET_TOKEN
        )

        weight = params.extra_args and params.extra_args.get(
            WEIGHT, DEFAULT_WEIGHT
        )

        if target_tokens is not None and not isinstance(target_tokens, str):
            raise ValueError(f"target_tokens value : {target_tokens} -- is not str")
        
        if weight is not None and not isinstance(weight, float):
            raise ValueError(f"weight value : {weight} -- is not float")

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool) -> None:
        
        self.config = vllm_config
        self.device = device
        self.is_pin_memory = is_pin_memory

        self.req_info: dict[int, tuple[dict[int, float], float]] = {}


    def is_argmax_invariant(self) -> bool:
        return False
    
    def update_state(
        self,
        batch_update: BatchUpdate | None,
    ) -> None:
        
        if not batch_update:
            return
        
        for index, params, _, _ in batch_update.added:
            
            assert params is not None
            self.validate_params(params)
            
            if params.extra_args:
                item = self._get_args(params.extra_args)
                self.req_info[index] = item
            else:
                self.req_info.pop(index, None)
            
        if self.req_info:
            
            for index in batch_update.removed:
                self.req_info.pop(index, None)
        
            for adx, bdx, direct in batch_update.moved:
                a_val = self.req_info.pop(adx, None)
                b_val = self.req_info.pop(bdx, None)
                
                if a_val is not None:
                    self.req_info[bdx] = a_val
                if direct == MoveDirectionality.SWAP and b_val is not None:
                    self.req_info[adx] = b_val

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        
        if not self.req_info:
            return logits
        
        for row_idx, (target_tokens, weight) in self.req_info.items():

            if not target_tokens or weight == 0:
                continue

            token_ids = torch.tensor(
                list(target_tokens.keys()),
                device=logits.device,
                dtype=torch.long
            )

            deltas = torch.tensor(
                list(target_tokens.values()),
                device=logits.device,
                dtype=torch.long
            )

            logits[row_idx, token_ids] += deltas

        return logits
    
    @staticmethod
    def _get_args(extra_args: dict):

        raw_tokens = extra_args.get(TARGET_TOKEN, "")
        weight = float(extra_args.get(WEIGHT, DEFAULT_WEIGHT))

        target_tokens = {}

        if raw_tokens:
            # разбиваем по ';'
            for pair in raw_tokens.split(";"):
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    token_id_str, value_str = pair.split(":")
                    token_id = int(token_id_str.strip())
                    value = float(value_str.strip())
                    target_tokens[token_id] = value
                except ValueError as e:
                    raise ValueError(
                        f"Invalid target_tokens format: '{pair}', should be '<token_id>:<float>'"
                    ) from e

        return target_tokens, weight
