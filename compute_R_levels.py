"""Compute metric R (recovery) for experiment-5: three dirt levels.

For each dirt level L (25 / 50 / 75 % junk) and each method:
  R(L) = (method_dirtyL - pure_dirtyL) / (pure_clean - pure_dirtyL)

  R = 0  -> filter did nothing (stuck at the dirty level)
  R = 1  -> filter fully recovered (back to the clean level)
  R > 1  -> filter ended up cleaner than the original corpus

Prints PASS@1 for pure / length / self_clean at every level plus R for the
filters, so we can read robustness as a CURVE over noise severity.

Run (locally, after results synced):  ./venv/bin/python compute_R_levels.py
"""

import argparse
import csv
import datetime
import json
import os

LEVELS = [("d25", "25%"), ("d50", "50%"), ("d75", "75%")]
METHODS = ["pure", "length", "self_clean"]   # pure first = the dirty floor


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
    with open(rc) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return sum(int(r["score"]) for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default="results")
    args = ap.parse_args()

    clean = pass1("pure_clean_apis", args.save_path)
    out = {"metric": "R (recovery) per dirt level", "pure_clean": clean, "levels": {}}
    lines = [
        "R (recovery) = (filter_dirty - pure_dirty) / (pure_clean - pure_dirty)",
        "0 = no recovery, 1 = full recovery, >1 = cleaner than original",
        "",
        f"pure_clean (ceiling) = {clean if clean is None else round(clean,3)}",
        "",
        f"{'level':6s} {'method':12s} {'PASS@1':>7s} {'R':>7s}",
        "-" * 36,
    ]

    if clean is None:
        lines.append("  missing pure_clean_apis — run it first")

    for lvl, pct in LEVELS:
        floor = pass1(f"pure_{lvl}", args.save_path)
        out["levels"][lvl] = {"pct": pct, "pure_dirty": floor, "methods": {}}
        denom = (clean - floor) if (clean is not None and floor is not None) else None
        for method in METHODS:
            p = pass1(f"{method}_{lvl}", args.save_path)
            R = None
            if method != "pure" and p is not None and denom not in (None, 0):
                R = (p - floor) / denom
            out["levels"][lvl]["methods"][method] = {"pass1": p, "R": R}
            ps = f"{p:.3f}" if p is not None else "—"
            Rs = f"{R:.2f}" if R is not None else ("—" if method != "pure" else "")
            lines.append(f"{pct:6s} {method:12s} {ps:>7s} {Rs:>7s}")
        lines.append("")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.save_path, "R_levels", ts)
    os.makedirs(out_dir, exist_ok=True)
    out["timestamp"] = ts
    with open(os.path.join(out_dir, "R.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"Saved -> {out_dir}/  (R.json, summary.txt)")


if __name__ == "__main__":
    main()
