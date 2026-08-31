"""Re-file exp16 result folders under the config that actually produced them.

run_ds1000.py names a result folder after the config FILE it was given, so a
yaml whose contents were copy-pasted from a sibling produces a run filed under
the wrong experiment. On the first pass of this battery 9 of 13 runs ended up
mislabelled, which also makes run_exp16.sh skip configs it thinks are done.

The truth is recoverable: every run snapshots the config it really used next to
its results. This script reads that snapshot, works out which experiment it is
from (corpus, presence of a system_prompt), and moves the timestamped folder
under the right name. Timestamps stay unique, so runs that turn out to be
repeats of the same config simply land side by side.

    python db_scripts/relabel_exp16_results.py            # dry run, shows the plan
    python db_scripts/relabel_exp16_results.py --apply    # actually move things

Nothing is deleted: folders are moved, and an empty parent left behind is
removed only when it really is empty.
"""

import argparse
import json
import os
import shutil
import sys

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required: pip install pyyaml")


# (corpus file, has system_prompt) -> config name
IDENTITY = {
    ("docs_base.db", False): "e0_baseline",
    ("docs_base.db", True): "e3_guide_in_prompt",
    ("docs_aug_full.db", False): "e0_aug_full",
    ("docs_only_guide.db", False): "e1_only_guide",
    ("docs_only_migr.db", False): "e1_only_migr",
    ("docs_only_howto.db", False): "e1_only_howto",
    ("docs_minus_guide.db", False): "e1_minus_guide",
    ("docs_minus_migr.db", False): "e1_minus_migr",
    ("docs_minus_howto.db", False): "e1_minus_howto",
    ("docs_minus_t1.db", False): "e2_minus_t1",
    ("docs_minus_t1_t2.db", False): "e2_minus_t1_t2",
    ("docs_base_inlined.db", False): "e6_base_inlined",
    ("docs_aug_inlined.db", False): "e6_aug_inlined",
    ("docs_aug_train_only.db", False): "e4_aug_train_only",
}

EXP16_PREFIXES = ("e0_", "e1_", "e2_", "e3_", "e4_", "e6_")


def identify(config_path: str):
    """Return the config name a snapshot belongs to, or None if unrecognised."""
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    params = config["components"]["data_base"]["params"]
    corpus = os.path.basename(params["path_to_db"])
    agent = config["components"]["agent"]["params"]
    has_prompt = bool(agent.get("system_prompt"))

    name = IDENTITY.get((corpus, has_prompt))
    if name == "e3_guide_in_prompt":
        # The two E3 variants share a corpus and differ only in prompt length.
        prompt = agent.get("system_prompt", "")
        if len(prompt) > 5000:
            name = "e3_guide_in_prompt_verbatim"
    return name, corpus, has_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the moves (default: dry run).")
    args = parser.parse_args()

    plan, unknown, ok = [], [], []
    for folder in sorted(os.listdir(args.results)):
        if not folder.startswith(EXP16_PREFIXES):
            continue
        parent = os.path.join(args.results, folder)
        if not os.path.isdir(parent):
            continue
        for stamp in sorted(os.listdir(parent)):
            run_dir = os.path.join(parent, stamp)
            config_path = os.path.join(run_dir, "config.yaml")
            if not os.path.isfile(config_path):
                unknown.append((run_dir, "no config.yaml snapshot"))
                continue
            name, corpus, has_prompt = identify(config_path)
            if name is None:
                unknown.append((run_dir, f"unrecognised corpus {corpus!r}"))
            elif name == folder:
                ok.append(run_dir)
            else:
                plan.append((run_dir, os.path.join(args.results, name, stamp),
                             folder, name, corpus))

    print(f"- already correct : {len(ok)}")
    print(f"- to be moved     : {len(plan)}")
    print(f"- unrecognised    : {len(unknown)}")

    if unknown:
        print("\nCould not identify (left untouched):")
        for run_dir, why in unknown:
            print(f"  {run_dir}  — {why}")

    if plan:
        print("\nPlan:")
        for _, _, was, now, corpus in plan:
            print(f"  {was:<28} -> {now:<28} ({corpus})")

    if not args.apply:
        print("\nDry run. Re-run with --apply to move these folders.")
        return

    for src, dst, _, _, _ in plan:
        if os.path.exists(dst):
            print(f"  !! target already exists, skipping: {dst}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"  moved {src} -> {dst}")

    # Clean up parents that are now empty, but never anything that still holds data.
    for folder in sorted(os.listdir(args.results)):
        parent = os.path.join(args.results, folder)
        if (folder.startswith(EXP16_PREFIXES) and os.path.isdir(parent)
                and not os.listdir(parent)):
            os.rmdir(parent)
            print(f"  removed empty {parent}")

    print("\nDone. Now run_exp16.sh will see the real coverage — the configs that "
          "never ran will no longer look complete.")


if __name__ == "__main__":
    main()
