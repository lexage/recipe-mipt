from typing import List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class PairwiseComparisonConfig:
    """Configuration for pairwise comparison scoring."""
    # Sampling parameters
    num_compare_all: int = 2
    num_examples: int = 100
    offset: int = 0
    num_examples_proportion: Optional[float] = None
    num_examples_proportion_start: float = 0.0
    
    # Randomness and model parameters
    seed: int = 42
    model: str = "gpt-3.5-turbo"  # Options: "gpt-3.5-turbo", "gpt-4", "claude-2"
    generations: int = 20
    
    # Prompting parameters
    template: str = ""
    labels: Tuple[str, str] = ("A", "B")
    system_prompt: str = "You are a helpful assistant."
    
    # Text processing parameters
    text_field: str = "text"
    token_field: str = "input_ids"
    tokens_min: int = 256
    tokens_max: int = 512
    probability_tokens_max: float = 0.5
    tokenizer: str = "meta-llama/Llama-2-7b-hf"
    
    # Output format
    flat_output_format: bool = False
    
    
@dataclass
class QuraterTrainingConfig:
    model: str = "princeton-nlp/Sheared-LLaMA-1.3b"
    bsz: int = 512
    seq: int = 16
    lr: float = 5e-5
    epochs: int = 2
    warmup: float = 0.1
    confidence: float = 0.5
    labeltemp: float = 1.0
    label_index: str = "all"
    suffix: str = ""
    save_steps: int = 200
    run_name: str = "trained_qurater"
    
    config_overrides: str = ""
    use_fast_tokenizer: bool = False
    max_length: int = 2048
    eval_split_size: float = 1.0
    eval_split_size_train: float = 0.1
    cache_dir: str = ".src/filtering/qurating/cache"
    single_label_ablation: int = -1
    
    log_level: str = "info"
    logging_steps: int = 1
    disable_tqdm: bool = True
    evaluation_strategy: str = "steps"
    
    load_best_model_at_end: bool = True
    metric_for_best_mode: str = "eval_validation_acc"
    greater_is_better: bool = True
    dataloader_num_workers: int = 2
    overwrite_output_dir: bool = True
    remove_unused_columns: bool = False
    report_to: List[str] = ["wandb"]
    
    do_train: bool = True
    do_eval: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    
    bf16: bool = True
    bf16_full_eval: bool = True
    
    ddp_find_unused_parameters: bool = False
    fsdp: List[str] = ["auto_wrap"]
    
    log_confidences: List[float, float] = [0.5, 0.8]
    
    
@dataclass
class EnvConfig:
    wandb_project: str = "lm-data-selection"
    wandb_mode: str = "offline"
    fsdp_sharding_strategy: str = "5"
    fsdp_state_dict_type: str = "FULL_STATE_DICT"
    
    
@dataclass
class AnnotationConfig:
    model: str = ".src/filtering/qurating/checkpoints-preferences/trained_qurater"
    output: str = ".src/filtering/qurating/datasets/annotated_data"
    tokens: int = 512
    map_batch_size: int = 512
    device_batch_size: int = 16
    num_workers: int = 1
    text_field: str = "text"
    tokens_field: str = "input_ids"
    labels: Optional[List[str]] = ["code_clarity_readability", "technical_accuracy_correctness", "educational_value_programming", "practical_utility_applicability"]
    data_files: Optional[List[str]] = field(default_factory=list)
    shard: List[int] = field(default_factory=lambda: [0, 1])
    

@dataclass
class SelectionConfig:
    """Configuration for data selection"""
    inputs: List[str] = field(default_factory=lambda: [".src/filtering/qurating/datasets/annotated_data"])
    output: str = ".src/filtering/qurating/datasets/selected_data"
    
    # Attribute paths
    attributes: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    
    # Field names
    seq_len_field: str = "input_len"
    metric_field: Optional[List[str]] = None
    reference_field: Optional[str] = None
    domain_field: Optional[str] = None
    
    # Selection parameters
    tokens: int = 5_000_000_000
    temperature: float = 1.0
    sample: bool = False
    normalize: bool = False
    select_bottom: bool = False
    margin: float = 0.1
    tokens_per_shard: int = 500_000_000
    shard_suffix: str = ""
    
    # Processing parameters
    json: bool = False
    seed: int = 42
    num_workers: Optional[int] = None