"""Write a seeded DS1000 train/eval split (experiment 4).

The augmentation agent was shown all 1000 baseline answers, so every number
measured on the full set is tuned-on data. To get one unbiased figure the agent
must be re-run against a TRAIN slice only, and the score reported on the
complement. This script produces the split file; the same JSON is later fed to

    python run_ds1000.py -c <configs> --exclude-split data/exp16/split.json

which drops the train problem_ids before evaluating. The format matches
`src/optimization/split.py` (that module is reused here, so the file is
interchangeable with the one optimize_prompts.py emits).

Usage (from the repo root):

    python scripts/make_ds1000_split.py --n-train 300 --seed 7 \
        --out data/exp16/split.json

Then filter the baseline run's per-task records down to the train ids and give
ONLY those to the augmentation agent:

    python scripts/make_ds1000_split.py --n-train 300 --seed 7 \
        --out data/exp16/split.json \
        --filter-answers results/<baseline_run>/answers.jsonl \
        --filter-out data/exp16/answers_train_only.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks import DS1000
from src.optimization import split as split_mod


def _problem_id(record: dict):
    """Best-effort problem id lookup across the shapes the runner writes."""
    for key in ("problem_id", "id", "task_id"):
        if key in record:
            return record[key]
    metadata = record.get("metadata") or {}
    return metadata.get("problem_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset", default="data/ds1000/ds1000.jsonl.gz")
    parser.add_argument("--n-train", type=int, default=300,
                        help="Tasks the augmentation agent is allowed to see.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/exp16/split.json")
    parser.add_argument("--filter-answers",
                        help="jsonl of baseline per-task records to slice down "
                             "to the train ids.")
    parser.add_argument("--filter-out",
                        help="Where to write the sliced records.")
    args = parser.parse_args()

    bench = DS1000(dataset_path=args.dataset)
    problem_ids = [
        item.metadata.get("problem_id") for item in bench.dataset._items
    ]
    split = split_mod.make_split(problem_ids, n_train=args.n_train, seed=args.seed)
    split_mod.save_split(split, args.out)
    print(f"- {len(split['train'])} train / {len(split['eval'])} held out "
          f"-> {args.out}")

    if not args.filter_answers:
        return

    if not args.filter_out:
        raise SystemExit("--filter-answers requires --filter-out")

    train_ids = set(split["train"])
    kept = dropped = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.filter_out)), exist_ok=True)
    with open(args.filter_answers, encoding="utf-8") as src, \
            open(args.filter_out, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if _problem_id(record) in train_ids:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"- answers: kept {kept}, dropped {dropped} -> {args.filter_out}")
    print("  give ONLY this file to the augmentation agent.")


if __name__ == "__main__":
    main()
