#!/usr/bin/env bash
# Experiment 6: API-reference filtration (genre filter) first, then the
# oracle paraphrase generation. Mirrors run_exp5.sh.
set -e
python run_ds1000.py -c test_configs_experimental_6/filtration
python run_ds1000.py -c test_configs_experimental_6/generation
