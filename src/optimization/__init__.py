"""Prompt optimization over the pipeline's optimizable prompts.

Layers, from the bottom up:

  candidates.py  a candidate is {prompt_key: text} over the OPTIMIZABLE prompts
                 of a component; applying one mutates Prompt objects in place
  split.py       seeded DS1000 train/eval split, so reported PASS@1 stays clean
  evaluator.py   one rollout: generate -> in-memory retrieve -> solve -> execute
  optimizers/    swappable search backends (upstream GEPA, random baseline)

Entry point: optimize_prompts.py at the repo root.
"""
