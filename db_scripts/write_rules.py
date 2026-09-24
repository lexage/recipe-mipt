"""Write the rule block from a config, without touching the benchmark.

Step one of the experiment: look at what the model actually writes before any
card time is spent on it.

    # one text, the seed the config carries
    python db_scripts/write_rules.py -c final_test_configs/c1_general_per_problem.yaml

    # three texts with different seeds, saved next to each other
    python db_scripts/write_rules.py -c final_test_configs/c1_general_per_problem.yaml \\
        --seeds 7,17,42 --out-dir results/rule_dumps

Each run writes its own jsonl (prompt, answer and verdict per call, then the
assembled text) and prints the block. Nothing else in the pipeline is built, so
no index, no corpus, no solver.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.prompt_rules import PromptRuleGenerator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True,
                        help="config carrying a rule_writer component")
    parser.add_argument("--seeds", default="",
                        help="comma-separated seeds; default: the config's own")
    parser.add_argument("--out-dir", default="",
                        help="where the dumps go; default: the config's "
                             "rules_path, one file, overwritten")
    parser.add_argument("--model", default="",
                        help="override the model id from the config")
    parser.add_argument("--url", default="",
                        help="override the endpoint from the config")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    components = config.get("components", {})
    if "rule_writer" not in components:
        sys.exit(f"{args.config} has no rule_writer component")
    params = dict(components["rule_writer"].get("params", {}))

    if args.model:
        params["model_name"] = args.model
    if args.url:
        params["url"] = args.url
    # Always write: reading a cached file back would defeat the point.
    params["rebuild"] = True

    stem = os.path.splitext(os.path.basename(args.config))[0]
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             or [int(params.get("seed", 7))])

    for seed in seeds:
        run = dict(params, seed=seed)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            run["rules_path"] = os.path.join(
                args.out_dir, f"{stem}_seed{seed}.jsonl")

        writer = PromptRuleGenerator(**run)
        text = writer.rules()

        print(f"\n{'=' * 70}\n{stem}  mode={writer.mode}  "
              f"problems={writer.problem_set}  seed={seed}\n"
              f"dump: {run.get('rules_path') or '(not written)'}\n{'=' * 70}")
        print(text)


if __name__ == "__main__":
    main()
