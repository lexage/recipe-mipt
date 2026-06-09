#!/usr/bin/env bash
# Experiment 5: filtration first, then the generation baseline.
set -e
python run_ds1000.py -c test_configs_experimental_5/filtration
python run_ds1000.py -c test_configs_experimental_5/generation
