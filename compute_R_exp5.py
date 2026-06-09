"""Compute metric R for experiment-5 (1 clean + 1 dirty), plus the generation
baseline PASS@1 (clean, no filter).

  R = (filter_dirty - pure_dirty) / (pure_clean - pure_dirty)
  0 = no recovery, 1 = full recovery, >1 = cleaner than original.

Run (locally, after results synced):  ./venv/bin/python compute_R_exp5.py
"""

import argparse
import csv
import datetime
import json
import os

FILTERS = ["f1v2", "code_select", "code_llm", "topic_select", "topic_llm"]
GENERATORS = ["zeroshot", "oneshot", "randomtopic", "randomword", "instruct", "codeeval"]


def pass1(stem, save_path):
    base = os.path.join(save_path, f"simple_example_{stem}")
    if not os.path.isdir(base):
        return None
    subs = sorted(os.listdir(base))
    if not subs:
        return None
    rc = os.path.join(base, subs[-1], "results.csv")
    if not os.path.exists(rc):
        return None
    rows = list(csv.DictReader(open(rc)))
    return sum(int(r["score"]) for r in rows) / len(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default="results")
    args = ap.parse_args()

    clean = pass1("pure_clean", args.save_path)
    floor = pass1("pure_dirty", args.save_path)
    out = {"pure_clean": clean, "pure_dirty": floor, "filters": {}, "generation": {}}
    L = ["R = (filter_dirty - pure_dirty) / (pure_clean - pure_dirty)", ""]
    L.append(f"pure_clean (ceiling) = {None if clean is None else round(clean,3)}")
    L.append(f"pure_dirty (floor)   = {None if floor is None else round(floor,3)}"
             + ("" if (clean is None or floor is None) else f"   (drop {floor-clean:+.3f})"))
    L.append("")
    denom = (clean - floor) if (clean is not None and floor is not None) else None
    L.append(f"{'filter':14s} {'clean':>7s} {'dirty':>7s} {'R':>7s}")
    L.append("-" * 40)
    for flt in FILTERS:
        pc = pass1(f"{flt}_clean", args.save_path)
        pd = pass1(f"{flt}_dirty", args.save_path)
        R = (pd - floor) / denom if (pd is not None and denom not in (None, 0)) else None
        out["filters"][flt] = {"clean": pc, "dirty": pd, "R": R}
        L.append(f"{flt:14s} {('—' if pc is None else f'{pc:.3f}'):>7s} "
                 f"{('—' if pd is None else f'{pd:.3f}'):>7s} "
                 f"{('—' if R is None else f'{R:.2f}'):>7s}")
    L.append("")
    L.append("=== generation baseline (clean, no filter) — база для своего метода ===")
    L.append(f"{'generator':14s} {'PASS@1':>7s} {'vs pure':>8s}")
    for g in GENERATORS:
        p = pass1(f"gen_{g}_clean", args.save_path)
        out["generation"][g] = p
        delta = "" if (p is None or clean is None) else f"{p-clean:+.3f}"
        L.append(f"{g:14s} {('—' if p is None else f'{p:.3f}'):>7s} {delta:>8s}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.save_path, "R_exp5", ts)
    os.makedirs(out_dir, exist_ok=True)
    out["timestamp"] = ts
    json.dump(out, open(os.path.join(out_dir, "R.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    summary = "\n".join(L)
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write(summary + "\n")
    print(summary)
    print(f"\nSaved -> {out_dir}/")


if __name__ == "__main__":
    main()
