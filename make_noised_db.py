"""Make a SEPARATE heavily-noised copy of a corpus DB (for metric R = recovery).

Copies a clean corpus SQLite DB to a NEW file and INSERTS many noise documents
of several distinct types (aggressive volume), so that `pure` on this dirty DB
scores much worse. The filter then has to recover it. The clean DB is never
modified — the dirt lives in its own file, for reproducibility.

Several noise types are mixed on purpose, so neither the filters nor the metric
get tuned to one specific kind of contamination:
  char_noise   — a real doc with weird symbols inserted every few chars
  incoherent   — spliced first-sentences of several unrelated docs
  wrong_code   — one doc's title with another doc's body (description/code mismatch)
  boilerplate  — repeated template text
  lowinfo      — repetitive low-information filler
  garbled      — random characters

The first three are DERIVED from real docs and keep the source title at the
front, so they stay retrievable near the same API and actually mislead the
model (that is what drives `pure` down). All noise docs are attached to real
sections (so they share a real library) and named NOISE__<type>__<k>.

RUN ON THE SERVER (by Konstantin):
  python make_noised_db.py --in  data/docs_database_examples_apis.db \
                           --out data/docs_database_examples_apis_dirty.db \
                           --noise-mult 1.0 --seed 7
"""

import argparse
import os
import random
import re
import shutil
import sqlite3
import string

_SYMS = "§¤×÷¶∆∑≈≠#@%&*"
_SENT = re.compile(r"[^.!?\n]+[.!?]")


def char_noise(text, rng):
    out = []
    gap = rng.randint(2, 4)
    for i, ch in enumerate(text):
        out.append(ch)
        if i and i % gap == 0:
            out.append(rng.choice(_SYMS))
    return "".join(out)


def first_sents(text, n=3):
    s = _SENT.findall(text or "")
    return " ".join(x.strip() for x in s[:n]) if s else (text or "")[:200]


def make_noise(ntype, name, body, pool, rng):
    """Return noise text. `pool` = list of (name, body) real docs."""
    if ntype == "char_noise":
        return name + "\n" + char_noise(body[:1200], rng)
    if ntype == "incoherent":
        bits = [first_sents(rng.choice(pool)[1], 2) for _ in range(5)]
        return name + "\n" + " ".join(bits)
    if ntype == "wrong_code":
        other = rng.choice(pool)[1]
        return name + "\n" + other[:1000]          # title says X, body is Y
    if ntype == "boilerplate":
        return ("Copyright (c) 2024 ACME. All rights reserved.\n"
                "Navigation: Home > Docs > API. Terms | Privacy | Contact.\n"
                "Edit this page on GitHub. Was this helpful? Yes / No.\n") * 3
    if ntype == "lowinfo":
        return ("the data is the data and the data is here. " * 30)
    if ntype == "garbled":
        alpha = string.ascii_letters + string.digits + string.punctuation + "  "
        return "".join(rng.choice(alpha) for _ in range(700))
    return body


TYPES = ["char_noise", "incoherent", "wrong_code", "boilerplate", "lowinfo", "garbled"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--noise-mult", type=float, default=1.0,
                    help="noise docs as a fraction of the corpus size (1.0 = +100%)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"input DB not found: {args.inp}")
    if os.path.abspath(args.inp) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --in (keep the clean DB intact)")

    shutil.copyfile(args.inp, args.out)            # dirty starts as a copy of clean
    conn = sqlite3.connect(args.out)

    cols = [c[1] for c in conn.execute("PRAGMA table_info(documents)")]
    need = {"id", "section_id", "name", "content"}
    if not need.issubset(cols):
        raise SystemExit(f"documents table missing columns; has {cols}, need {need}")

    docs = conn.execute(
        "SELECT id, section_id, name, content FROM documents "
        "WHERE content IS NOT NULL AND length(content) > 50"
    ).fetchall()
    if not docs:
        raise SystemExit("no usable real docs found")

    pool = [(d[2] or "", d[3] or "") for d in docs]
    section_by_doc = {d[0]: d[1] for d in docs}
    section_ids = [d[1] for d in docs if d[1] is not None]
    max_id = conn.execute("SELECT MAX(id) FROM documents").fetchone()[0] or 0

    rng = random.Random(args.seed)
    n_noise = int(len(docs) * args.noise_mult)

    insert_cols = [c for c in ["id", "section_id", "name", "content", "length",
                               "created_at"] if c in cols]
    placeholders = ",".join("?" for _ in insert_cols)
    rows = []
    counts = {t: 0 for t in TYPES}
    for k in range(n_noise):
        ntype = TYPES[k % len(TYPES)]
        src = docs[rng.randrange(len(docs))]
        name = f"NOISE__{ntype}__{k}"
        content = make_noise(ntype, src[2] or "", src[3] or "", pool, rng)
        # derived types keep the source library/section; junk types pick random
        if ntype in ("char_noise", "incoherent", "wrong_code"):
            sid = src[1] if src[1] is not None else rng.choice(section_ids)
        else:
            sid = rng.choice(section_ids)
        values = {"id": max_id + 1 + k, "section_id": sid, "name": name,
                  "content": content, "length": len(content),
                  "created_at": "2024-01-01 00:00:00"}
        rows.append(tuple(values[c] for c in insert_cols))
        counts[ntype] += 1

    conn.executemany(
        f"INSERT INTO documents ({','.join(insert_cols)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    print(f"clean docs : {len(docs)}")
    print(f"noise added: {n_noise}  ({', '.join(f'{t}:{counts[t]}' for t in TYPES)})")
    print(f"total docs : {total}")
    print(f"saved dirty DB -> {args.out}")


if __name__ == "__main__":
    main()
