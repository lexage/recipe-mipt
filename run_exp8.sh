#!/usr/bin/env bash
# Experiment 8: generation only (no filtration). Everything embedded as examples.
# First the baselines (pure_examples is the main comparison floor/ceiling),
# then the 6 generation configs (paraphrase / paraphrase_only / codeeval /
# code2doc / code2doc_only / code2task).
set -e
python run_ds1000.py -c test_configs_experimental_8/baseline
python run_ds1000.py -c test_configs_experimental_8/generation
