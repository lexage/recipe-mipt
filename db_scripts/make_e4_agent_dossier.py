"""Build a self-contained train-only error dossier for the E4 augmentation agent.

The first augmentation pass gave the agent the whole baseline results folder and
let it look up prompts and reference solutions in ds1000.jsonl.gz itself. For E4
that is exactly what must not happen: the dataset file holds all 1000 problems,
so an agent with access to it can see the held-out half no matter which answers
file we hand over.

This script bundles everything the agent legitimately needs into ONE file
covering the train ids only:

    problem_id, library, perturbation_type   what kind of task it was
    prompt                                   the task as the solver saw it
    reference_code                           the ground truth
    model_code                               what the solver actually produced
    passed                                   whether it was accepted

With this, the agent never has a reason to open ds1000.jsonl.gz, the baseline
results folder, or anything else — which is the only way to keep the held-out
split genuinely unseen.

    python db_scripts/make_e4_agent_dossier.py

Note `code_context` (the checker harness) is deliberately NOT included. The
first pass produced 28 documents explaining how the grader compares objects;
that is tuning to the comparison function rather than to the task, and there is
no reason to hand it over a second time.
"""

import argparse
import gzip
import json
import os
import sys


def _find_repo_root() -> str:
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        current = start
        while True:
            if os.path.isfile(os.path.join(current, "run_ds1000.py")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    sys.exit("could not find the repo root — run this from inside the repo")


REPO = _find_repo_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="data/exp16/split.json")
    parser.add_argument("--all", action="store_true",
                        help="Ignore the split and keep every task. For the "
                             "exp17 oracle probe, whose whole point is to give "
                             "the local model exactly what the manual pass had. "
                             "Anything built from this dossier is tuned on the "
                             "full test set and cannot be reported as a result.")
    parser.add_argument("--dataset", default="data/ds1000/ds1000.jsonl.gz")
    parser.add_argument("--answers", required=True,
                        help="answers.jsonl from the baseline run (all 1000; "
                             "it gets filtered here).")
    parser.add_argument("--results-csv", required=True,
                        help="results.csv from the same baseline run.")
    parser.add_argument("--out", default="data/exp16/e4_agent_dossier.jsonl")
    parser.add_argument("--failures-only", action="store_true",
                        help="Keep only the tasks the baseline got wrong.")
    args = parser.parse_args()

    os.chdir(REPO)

    if args.all:
        train = None
        print("- ALL 1000 tasks (no split) — diagnostic dossier, leaks by design")
    else:
        split = json.load(open(args.split, encoding="utf-8"))
        train = set(split["train"])
        print(f"- train ids: {len(train)}  (held out: {len(split['eval'])})")

    import pandas as pd
    verdict = pd.read_csv(args.results_csv).set_index("problem_id").score.to_dict()

    model_code = {}
    with open(args.answers, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            model_code[record["metadata"]["problem_id"]] = record["code"]

    kept = skipped = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with gzip.open(args.dataset, "rt") as src, \
            open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            item = json.loads(line)
            pid = item["metadata"]["problem_id"]
            if train is not None and pid not in train:
                skipped += 1
                continue
            passed = bool(verdict.get(pid, 0))
            if args.failures_only and passed:
                continue
            dst.write(json.dumps({
                "problem_id": pid,
                "library": item["metadata"].get("library"),
                "perturbation_type": item["metadata"].get("perturbation_type"),
                "prompt": item["prompt"],
                "reference_code": item["reference_code"],
                "model_code": model_code.get(pid, ""),
                "passed": passed,
            }, ensure_ascii=False) + "\n")
            kept += 1

    scope = set(verdict) if train is None else train
    failed = sum(1 for pid in scope if not verdict.get(pid, 0))
    print(f"- held-out problems excluded : {skipped}")
    print(f"- records written            : {kept} -> {args.out}")
    print(f"- of those tasks, failed     : {failed} / {len(scope)}")
    print("\nGive the agent this file and data/docs_database_examples.db, and")
    print("nothing else. In particular do NOT give it the dataset, the results")
    print("folder, or the list of held-out ids.")


if __name__ == "__main__":
    main()
