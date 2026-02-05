import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import argparse
import torch

sys.path.insert(0, os.getcwd())

from train_preference_model import train_qurater, ScriptArguments, TrainingArguments
from src.filtering.qurating.configs.training_config import QuraterTrainingConfig, EnvConfig


def run_training(train_datasets: List[str]):
    config = QuraterTrainingConfig()
    
    num_gpus = torch.cuda.device_count()
    
    gradient_accumulation_steps = config.bsz // (config.seq * num_gpus)
    
    # run_name = f"qurater_{os.path.basename(run_args.model)}_bsz{run_args.bsz}_lr{run_args.lr}_epochs{run_args.epochs}_warmup{run_args.warmup}_conf{run_args.confidence}_labeltemp{run_args.labeltemp}{run_args.suffix}"
    out_dir = f".src/filtering/qurating/checkpoints-preferences/{config.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    labels = [
        "code_clarity_readability_average",
        "technical_accuracy_correctness_average",
        "educational_value_programming_average",
        "practical_utility_applicability_average",
    ]
    
    if config.label_index == "all":
        label_field = labels
    else:
        label_index = int(config.label_index)
        label_field = [labels[label_index]]
    
    script_args = ScriptArguments(
        model_name_or_path=config.model,
        config_name=config.model,
        tokenizer_name=config.model,
        config_overrides=config.config_overrides,
        use_fast_tokenizer=config.use_fast_tokenizer,
        max_length=config.max_length,
        label_field=label_field,
        train_datasets=train_datasets,
        eval_datasets=[],
        eval_split_size=config.eval_split_size,
        eval_split_size_train=config.eval_split_size_train,
        cache_dir=config.cache_dir,
        single_label_ablation=config.single_label_ablation,
    )
    
    training_args = TrainingArguments(
        run_name=config.run_name,
        output_dir=out_dir,
        log_level=config.log_level,
        logging_steps=config.logging_steps,
        disable_tqdm=config.disable_tqdm,
        save_steps=config.save_steps,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.save_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_mode=config.metric_for_best_mode,
        greater_is_better=config.greater_is_better,
        dataloader_num_workers=config.dataloader_num_workers,
        overwrite_output_dir=config.overwrite_output_dir,
        remove_unused_columns=config.remove_unused_columns,
        report_to=config.report_to,
        
        do_train=config.do_train,
        do_eval=config.do_eval,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.seq,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.lr,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup,
        
        bf16=config.bf16,
        bf16_full_eval=config.bf16_full_eval,
        
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        fsdp=config.fsdp,
        
        label_temperature=config.labeltemp,
        confidence_threshold=config.confidence,
        log_confidences=config.log_confidences,
    )
    
    env_config = EnvConfig()
    
    os.environ["WANDB_PROJECT"] = env_config.wandb_project
    os.environ["WANDB_DIR"] = out_dir
    os.environ["WANDB_MODE"] = env_config.wandb_mode
    os.environ["FSDP_SHARDING_STRATEGY"] = env_config.fsdp_sharding_strategy
    os.environ["FSDP_STATE_DICT_TYPE"] = env_config.fsdp_state_dict_type
    os.environ["OMP_NUM_THREADS"] = str(num_gpus)
    
    train_qurater(script_args, training_args)