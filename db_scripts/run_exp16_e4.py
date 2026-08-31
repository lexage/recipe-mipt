"""Drive experiment 4 end to end: split, regenerate, evaluate on held-out only.

E4 is the only unbiased number in the exp16 battery. Every other run is scored
on the same 1000 tasks whose baseline errors the augmentation agent read, so
those figures are tuned-on data. Here the agent sees a TRAIN slice only and the
score is reported on the complement.

The middle step cannot be scripted — regenerating the corpus means re-running
the augmentation agent — so this driver works in phases and stops where your
input is needed. It is safe to re-run: each phase is skipped once its output
exists.

    python db_scripts/run_exp16_e4.py                 # start / resume
    python db_scripts/run_exp16_e4.py --n-train 300   # change the split size
    python db_scripts/run_exp16_e4.py --compare-only  # just redo the arithmetic

Phase 1  write data/exp16/split.json and answers_train_only.jsonl
Phase 2  YOU regenerate the corpus into data/exp16/docs_aug_train_only.db
Phase 3  sanity-check that corpus, then evaluate with --exclude-split
Phase 4  compare against baseline and aug_full restricted to the held-out tasks
"""

import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd

def _find_repo_root() -> str:
    """Locate the repo by its marker file rather than by where this script sits.

    Assuming a fixed depth ("my parent's parent") breaks the moment someone
    drops the script in the repo root instead of db_scripts/ — it then treats
    the directory ABOVE the repo as the root and reports everything as missing.
    Walk up from the script, then from the cwd, looking for run_ds1000.py.
    """
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        current = start
        while True:
            if os.path.isfile(os.path.join(current, "run_ds1000.py")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    sys.exit("could not find the repo root (no run_ds1000.py above this script "
             "or the current directory) — run this from inside the repo")


REPO = _find_repo_root()
sys.path.insert(0, REPO)


# --- helpers ----------------------------------------------------------------

def newest_run(results_dir: str, config_name: str):
    """Most recent completed run folder for a config, or None."""
    runs = sorted(glob.glob(os.path.join(results_dir, config_name, "*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "runtime_stats.json"))]
    return runs[-1] if runs else None


def all_runs(results_dir: str, config_name: str):
    runs = sorted(glob.glob(os.path.join(results_dir, config_name, "*")))
    return [r for r in runs if os.path.isfile(os.path.join(r, "results.csv"))]


def find_config(configs_dir: str, stem: str):
    flat = os.path.join(configs_dir, f"{stem}.yaml")
    if os.path.isfile(flat):
        return flat
    hits = glob.glob(os.path.join(configs_dir, "*", f"{stem}.yaml"))
    return hits[0] if hits else None


def held_out_score(run_dir: str, held: set) -> tuple:
    df = pd.read_csv(os.path.join(run_dir, "results.csv"))
    sub = df[df.problem_id.isin(held)]
    return sub.score.mean() * 100, len(sub)


# --- phase 1 ----------------------------------------------------------------

def phase_split(args) -> bool:
    if os.path.isfile(args.split) and not args.force_split:
        split = json.load(open(args.split, encoding="utf-8"))
        print(f"1. split exists: {len(split['train'])} train / "
              f"{len(split['eval'])} held out  ({args.split})")
        return True

    baseline = newest_run(args.results, "e0_baseline")
    if baseline is None:
        print("1. FAILED — no completed e0_baseline run under "
              f"{args.results}/e0_baseline/")
        print("   The agent needs the baseline per-task answers to work from.")
        return False

    answers = os.path.join(baseline, "answers.jsonl")
    print(f"1. building split from {answers}")
    cmd = [
        sys.executable, os.path.join(REPO, "db_scripts", "make_ds1000_split.py"),
        "--n-train", str(args.n_train), "--seed", str(args.seed),
        "--out", args.split,
        "--filter-answers", answers,
        "--filter-out", args.train_answers,
    ]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    result = subprocess.run(cmd, cwd=REPO)
    return result.returncode == 0


# --- phase 2 / 3 ------------------------------------------------------------

def instructions(args) -> None:
    dossier = args.dossier
    have = os.path.isfile(dossier)
    print(f"""
2. Regenerate the corpus. This is the manual step.

   Give the augmentation agent EXACTLY these two inputs:
     - {dossier}{'' if have else '   <-- MISSING, build it first'}
     - data/docs_database_examples.db   (the original, unmodified corpus)
   plus the first-pass prompt, adjusted to say 300 tasks instead of 1000.

   Write its output to:
     - {args.corpus}
""")
    if not have:
        print(f"""   The dossier does not exist yet. Build it with:

     python db_scripts/make_e4_agent_dossier.py \\
         --answers     results/e0_baseline/<run>/answers.jsonl \\
         --results-csv results/e0_baseline/<run>/results.csv

   It bundles prompt, reference_code, the model's answer and the verdict for
   the train ids only. Handing over {args.train_answers} alone is not enough:
   it holds generated code and nothing else, so the agent would go looking for
   prompts and reference solutions in ds1000.jsonl.gz — which contains all 1000
   problems, held-out half included.
""")
    print("""   The agent must not see the held-out tasks in any form — not their
   prompts, not their ids, not a summary of how the model did on them.
   One leaked list and this experiment measures nothing again.

   Then re-run this script; it picks up from here.
""")


def check_corpus(args) -> bool:
    """Cheap integrity checks on the regenerated corpus before spending GPU hours."""
    print(f"3. checking {args.corpus}")

    def docs(path):
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return {i: c for i, c in conn.execute(
                "SELECT id, content FROM documents")}
        finally:
            conn.close()

    base, new = docs(args.base_db), docs(args.corpus)
    missing = set(base) - set(new)
    changed = [i for i in set(base) & set(new) if base[i] != new[i]]
    added = sorted(set(new) - set(base))

    print(f"   original documents kept : {len(base) - len(missing)} / {len(base)}")
    print(f"   original documents added: {len(added)}")
    if missing:
        print(f"   !! {len(missing)} original documents are MISSING — the agent was "
              f"supposed to only append")
    if changed:
        print(f"   !! {len(changed)} original documents were MODIFIED")
    if not added:
        print("   !! nothing was added — is this the right file?")
        return False

    # Same novel-literal rule as make_exp16_dbs.py: a quoted string that appears
    # nowhere in the original corpus is a fingerprint of a task-specific detail.
    conn = sqlite3.connect(f"file:{args.base_db}?mode=ro", uri=True)
    corpus_text = "\n".join(
        [c for (c,) in conn.execute("SELECT content FROM documents")]
        + [c for (c,) in conn.execute("SELECT content FROM examples")])
    conn.close()
    literal = re.compile(r"""['"]([A-Za-z][\w \-\(\)%\.]{2,40})['"]""")
    flagged = sum(
        1 for i in added
        if any(m not in corpus_text for m in set(literal.findall(new[i] or "")))
    )
    print(f"   documents with literals absent from the base corpus: "
          f"{flagged} / {len(added)}")
    if flagged:
        print("      (these carry strings the agent could only have taken from the "
              "tasks it saw — expected on TRAIN tasks, still worth eyeballing)")
    return True


def phase_eval(args) -> bool:
    if newest_run(args.results, "e4_aug_train_only") and not args.force:
        print("3. evaluation already done, skipping (use --force to redo)")
        return True

    config = find_config(args.configs, "e4_aug_train_only")
    if config is None:
        print(f"3. FAILED — e4_aug_train_only.yaml not found under {args.configs}")
        return False

    # -c is staged through a directory: unpatched run_ds1000.py globs *.yaml and
    # silently runs nothing when handed a file path.
    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(config, os.path.join(tmp, "e4_aug_train_only.yaml"))
        cmd = [sys.executable, "run_ds1000.py", "-c", tmp,
               "--log-chunks", "--exclude-split", args.split]
        print(f"3. evaluating on the held-out tasks only\n   {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=REPO).returncode == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- phase 4 ----------------------------------------------------------------

def phase_compare(args) -> None:
    split = json.load(open(args.split, encoding="utf-8"))
    held = set(split["eval"])
    print(f"\n4. scores on the {len(held)} held-out tasks\n")

    rows = []
    for name in ("e0_baseline", "e0_aug_full", "e4_aug_train_only"):
        runs = all_runs(args.results, name)
        if not runs:
            rows.append(dict(config=name, runs=0, pass1=float("nan"), n=0))
            continue
        scores, n = [], 0
        for run in runs:
            score, n = held_out_score(run, held)
            scores.append(score)
        rows.append(dict(config=name, runs=len(scores),
                         pass1=sum(scores) / len(scores), n=n))

    table = pd.DataFrame(rows).set_index("config")
    base = table.loc["e0_baseline", "pass1"]
    table["vs_base"] = table.pass1 - base
    print(table.round(2).to_string())

    tuned = table.loc["e0_aug_full", "vs_base"]
    honest = table.loc["e4_aug_train_only", "vs_base"]
    if pd.notna(honest) and pd.notna(tuned):
        print(f"\n   tuned-on-test gain : {tuned:+.2f} pp   (e0_aug_full)")
        print(f"   unbiased gain      : {honest:+.2f} pp   (e4_aug_train_only)")
        if tuned > 0:
            print(f"   transfers          : {100 * honest / tuned:.0f}% of the "
                  f"measured gain")
        print("\n   The unbiased figure is the one to report. Note it is also a "
              "slight\n   UNDER-estimate: the train slice was 30% the size, so the "
              "corpus had\n   fewer observed failures to learn from than the full "
              "augmentation did.")
    else:
        print("\n   (run e4_aug_train_only to fill in the last row)")


# --- main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", default="results")
    parser.add_argument("--configs", default="test_configs_experimental_16")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--base-db", default="data/docs_database_examples.db")
    parser.add_argument("--corpus", default="data/exp16/docs_aug_train_only.db")
    parser.add_argument("--split", default="data/exp16/split.json")
    parser.add_argument("--train-answers",
                        default="data/exp16/answers_train_only.jsonl")
    parser.add_argument("--dossier",
                        default="data/exp16/e4_agent_dossier.jsonl",
                        help="Self-contained train-only error dossier for the "
                             "agent (see make_e4_agent_dossier.py).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run the evaluation even if results exist.")
    parser.add_argument("--force-split", action="store_true",
                        help="Rebuild the split. Invalidates any existing E4 run.")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)

    if args.compare_only:
        phase_compare(args)
        return

    if not phase_split(args):
        sys.exit(1)

    if not os.path.isfile(args.corpus):
        instructions(args)
        phase_compare(args)
        return

    if not check_corpus(args):
        sys.exit(1)

    if not phase_eval(args):
        sys.exit(1)

    phase_compare(args)


if __name__ == "__main__":
    main()
