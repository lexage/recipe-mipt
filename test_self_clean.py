"""Quick check of SelfConsistencyCleaner with the REAL embedder (run on server).

Builds one synthetic document dirtied at all three levels, runs the cleaner,
prints what survived and a few PASS/FAIL checks.

  ./venv/bin/python test_self_clean.py
  ./venv/bin/python test_self_clean.py --url http://0.0.0.0:7216/v1 \
                                       --model Qwen/Qwen3-Embedding-4B
"""

import argparse

from src.agents.general import EmbeddingAgent
from src.agent_constructor.core import Chunk
from src.filtering.self_clean import SelfConsistencyCleaner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://0.0.0.0:7216/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    args = ap.parse_args()

    good = [
        "numpy array supports element wise arithmetic over the array dtype int64",
        "use numpy reshape to change the array shape and the array dimensions",
        "numpy broadcasting aligns the array shapes over arrays with dtype float32",
        "create an array with numpy zeros and set the array dtype to int64",
        "numpy concatenate joins arrays along the chosen axis dimensions",
        "the axis argument controls reduction over the array dimensions",
    ]
    # inject dirt at all three levels
    lines = [
        good[0].replace("array", "ar#r@ay", 1),    # L1: symbols inside a word
        good[1].replace("reshape", "привет", 1),     # L2: foreign-word swap
        good[2].replace("broadcasting", "qwzlkj", 1),  # L2: random-word swap
        good[3].replace("zeros", "arxray", 1),       # L2: letter-corrupted word
        good[4],
        good[5],
        "Copyright (c) 2024 ACME Corporation. All rights reserved.",  # L3 boilerplate
        "Copyright (c) 2024 ACME Corporation. All rights reserved.",  # repeated
        "a8$Kd@2(fj#%^ z!9}{ q~  w=3]+[p&  x*7/.,<>?:",               # L3 garbled
        "many people enjoy walking in the park on a pleasant afternoon",  # L3 off-topic
    ]

    embedder = EmbeddingAgent(url=args.url, model_name=args.model)
    flt = SelfConsistencyCleaner(embedder=embedder, min_doc_lines=4)
    out = flt.apply([Chunk(id="d1_0", doc_id="d1", text="\n".join(lines))])
    txt = out[0].text if out else ""

    print("\n=== INPUT (%d lines) ===" % len(lines))
    for l in lines:
        print("  ", l)
    print("\n=== KEPT ===")
    for l in txt.split("\n"):
        print("  ", l)

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("\n=== CHECKS ===")
    check("L1: ar#r@ay repaired to array", "ar#r@ay" not in txt and "array supports" in txt)
    check("L1: kept domain term int64", "int64" in txt)
    check("L1: kept domain term float32", "float32" in txt)
    check("L2: removed foreign word привет", "привет" not in txt)
    check("L2: removed random word qwzlkj", "qwzlkj" not in txt)
    check("L2: removed corrupted word arxray", "arxray" not in txt)
    check("L2: kept legit words joins/along/chosen",
          all(w in txt for w in ["joins", "along", "chosen"]))
    check("L3: cut boilerplate", "Copyright" not in txt)
    check("L3: cut garbled", "Kd@2" not in txt and "Kd2" not in txt)
    check("L3: cut off-topic line", "walking" not in txt)
    check("good content survived (>=5 good lines)",
          sum(1 for k in txt.split("\n") if "numpy" in k or "axis argument" in k) >= 5)


if __name__ == "__main__":
    main()
