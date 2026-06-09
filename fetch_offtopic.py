"""Download an OFF-TOPIC corpus (Wikipedia) -> data/offtopic_docs.jsonl.

These general-knowledge articles (people, places, physics, celebrities, ...) are
the "irrelevant junk documents" the team lead asked for. make_noised_realistic.py
injects them into the dirty DB so the topic filters (F3.x) must drop them.

Run where there is internet (WSL or server):
  python fetch_offtopic.py --n 4000
  python fetch_offtopic.py --dataset wikimedia/wikipedia --config 20231101.simple --n 4000

Output: jsonl, one {"text": "..."} per line.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.simple")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--out", default="data/offtopic_docs.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"loading {args.dataset} [{args.config}] ...")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            text = text[:args.max_chars]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.n:
                break
    print(f"wrote {written} off-topic docs -> {args.out}")


if __name__ == "__main__":
    main()
