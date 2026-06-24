#!/usr/bin/env bash
# Experiment 7: API-reference filtration on the API-retriever pipeline
# (api_selector + api_retriever, as in exp4) — the pipeline where the
# "apis boosts PASS@1" effect was originally observed.
set -e
python run_ds1000.py -c test_configs_experimental_7/filtration
