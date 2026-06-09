"""Quick server check of the v2 filters with the REAL embedder.

Covers what a tiny synthetic doc CAN show: F1_v2 normalisers (HTML / mojibake /
boilerplate) + code protection, and F2.1 dropping a chunk with broken code.
(L3's char model and the LLM paths are only meaningful on the full corpus / a
real run — R is the real measure.)

  ./venv/bin/python test_self_clean_v2.py
"""

import argparse

from src.agents.general import EmbeddingAgent
from src.agent_constructor.core import Chunk
from src.filtering.self_clean_v2 import SelfConsistencyCleanerV2
from src.filtering.code_aware import CodeAwareSelectCleaner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://0.0.0.0:7216/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    args = ap.parse_args()
    emb = EmbeddingAgent(url=args.url, model_name=args.model)

    lines = [
        "numpy array supports element wise arithmetic over the array dtype",
        "use numpy reshape to change the <p>array</p> shape without copying",   # HTML
        "create an array with numpy zeros and set the cafÃ© dtype value",        # mojibake
        "the axis argument controls reduction along the array dimensions",
        "numpy concatenate joins arrays along the chosen axis dimensions",
        "Copyright (c) 2024 ACME Corporation. All rights reserved.",            # boiler
        "Copyright (c) 2024 ACME Corporation. All rights reserved.",
    ]
    f1 = SelfConsistencyCleanerV2(embedder=emb, min_doc_lines=4)
    txt = (f1.apply([Chunk(id="d_0", doc_id="d", text="\n".join(lines))]) or [Chunk("","","")])[0].text
    print("=== F1_v2 KEPT ===")
    print(txt)

    def chk(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("\n=== CHECKS ===")
    chk("HTML stripped", "<p>" not in txt and "</p>" not in txt)
    chk("mojibake fixed (café)", "café" in txt and "cafÃ©" not in txt)
    chk("boilerplate cut", "Copyright" not in txt)
    chk("good lines kept (>=4)", sum(1 for l in txt.split("\n") if "numpy" in l or "axis" in l) >= 4)

    # F2.1: code protection + drop broken code
    good = Chunk(id="g_0", doc_id="g",
                 text=">>> import numpy as np\n>>> a = np.array([1,2,3])\n>>> a.sum()\n6")
    bad = Chunk(id="b_0", doc_id="b",
                text=">>> a = np.arr@y([1,2,3)\n>>> for i vvith range(:\nbroken")
    out = CodeAwareSelectCleaner(embedder=emb).apply([good, bad])
    ids = {c.id for c in out}
    chk("F2.1 good code chunk kept", "g_0" in ids)
    chk("F2.1 broken code chunk dropped", "b_0" not in ids)
    chk("F2.1 good code preserved verbatim", any("np.array([1,2,3])" in c.text for c in out))


if __name__ == "__main__":
    main()
