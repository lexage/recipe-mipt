bash scripts/run_swe_rebench_experiment_qwen_25_decrim.sh \
    best \
    react_sgr_summary_enabled_qwen2_5_decrim_new_description \
    pipeline_configs/react_sgr_summary_enabled_qwen2_5_decrim_new_description.yaml

bash scripts/run_swe_rebench_experiment_qwen_25_critic.sh \
    best \
    react_sgr_summary_enabled_qwen2_5_critic \
    pipeline_configs/react_sgr_summary_enabled_qwen2_5_critic.yaml

bash scripts/run_swe_rebench_experiment_qwen_25_enabled.sh \
    best \
    run_swe_rebench_experiment_qwen_25_enabled \
    pipeline_configs/react_sgr_summary_enabled_qwen2_5.yaml

bash scripts/run_swe_rebench_experiment_qwen_25_disabled.sh \
    best \
    run_swe_rebench_experiment_qwen_25_disabled \
    pipeline_configs/react_sgr_summary_disabled_qwen2_5.yaml
