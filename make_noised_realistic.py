"""Realistic dirtied copy of a corpus DB (team-lead style). SCRIPT ONLY — run
on the server for reproducibility; Claude never builds the DB itself.

Two-stage dirt:

  STAGE 1 — corrupt the ORIGINAL documents in place (all of them). `--noise f`
  is the share of original WORDS made dirty, split in FIXED proportions among
  realistic in-doc corruptions:
    * mojibake  — broken encoding on a span (utf-8 bytes shown as latin-1:
                  "café" -> "cafÃ©", "—" -> "â€"")
    * html      — leaked markup/entities (<p>, </span>, &nbsp;, &amp;)
    * boiler    — repeated scraping boilerplate lines (copyright / nav / footer)
  Plus a doc-wide CHAR SUBSTITUTION ("w"->"vv", a "flown" character) applied to
  a fraction `f` of the documents (it is global by nature, not localized).

  STAGE 2 — ADD off-topic junk DOCUMENTS (from --offtopic jsonl), `--junk-frac j`
  times the number of originals. These are NOT corrupted (they are junk by
  themselves) and are NOT counted in `f`. They go under a new 'offtopic' library
  so the filter must drop them by topic (only the F3 filters do).

The clean DB is never touched; the dirty lives in its own file.

RUN ON THE SERVER:
  python make_noised_realistic.py --in data/docs_database_examples.db \
      --out data/docs_database_examples_realistic_d50.db \
      --offtopic data/offtopic_docs.jsonl --noise 0.5 --junk-frac 0.5 --seed 7
"""

import argparse
import json
import os
import random
import re
import shutil
import sqlite3

_BOILER = [
    "Copyright (c) 2024 ACME Corporation. All rights reserved.",
    "Navigation: Home > Documentation > API Reference",
    "Privacy Policy | Terms of Service | Contact Us",
    "Edit this page on GitHub. Was this helpful? Yes / No.",
]
_HTML = ["<p>", "</p>", "<div class=\"section\">", "</div>", "<span>", "</span>",
         "&nbsp;", "&amp;", "&lt;", "&gt;", "<br/>", "<code>", "</code>"]
# doc-wide "flown character" substitutions (team lead: w -> vv)
_CHAR_SUBS = [("w", "vv"), ("m", "rn"), ("l", "I"), ("o", "0")]
_WORDISH = re.compile(r"[A-Za-z]{3,}")


def mojibake(text):
    """Re-encode utf-8 as latin-1 -> classic broken-encoding artifacts."""
    try:
        return text.encode("utf-8").decode("latin-1")
    except Exception:
        return text


def corrupt_span_mojibake(words, rng):
    """Mojibake a contiguous run of words; return new words + count touched."""
    if len(words) < 2:
        return words, 0
    n = max(1, len(words) // 3)
    start = rng.randint(0, len(words) - n)
    for i in range(start, start + n):
        words[i] = mojibake(words[i])
    return words, n


def inject_html(words, rng, k):
    """Insert k html tokens at random positions among words; return count."""
    for _ in range(k):
        words.insert(rng.randint(0, len(words)), rng.choice(_HTML))
    return k


def apply_char_sub(text, rng):
    a, b = rng.choice(_CHAR_SUBS)
    return text.replace(a, b)


def corrupt_doc(content, f, rng, do_char_sub):
    """Stage-1 in-doc corruption of one original document."""
    lines = [ln for ln in (content or "").split("\n") if ln.strip()]
    if not lines:
        return content

    # doc-wide flown character (global, applied to a fraction of docs)
    if do_char_sub:
        lines = [apply_char_sub(ln, rng) for ln in lines]

    G = sum(len(ln.split()) for ln in lines)
    if G == 0:
        return "\n".join(lines)
    budget = int(f * G)                     # dirty words target (mojibake+html+boiler)
    per = max(1, budget // 3)

    # mojibake spans
    touched = 0
    out_lines = []
    for ln in lines:
        w = ln.split()
        if touched < per and len(w) >= 2 and rng.random() < 0.5:
            w, c = corrupt_span_mojibake(w, rng)
            touched += c
        out_lines.append(w)                 # list-of-words per line for now

    # html injection
    touched = 0
    for w in out_lines:
        if touched < per:
            k = min(per - touched, max(1, len(w) // 5))
            touched += inject_html(w, rng, k)
    lines = [" ".join(w) for w in out_lines]

    # boilerplate lines (count words until ~per)
    bw = 0
    while bw < per:
        bl = rng.choice(_BOILER)
        lines.insert(rng.randint(0, len(lines)), bl)
        bw += len(bl.split())

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--offtopic", default="data/offtopic_docs.jsonl",
                    help="jsonl с офтоп-документами (поле 'text'); делает fetch_offtopic.py")
    ap.add_argument("--noise", type=float, default=0.5,
                    help="доля грязных слов в ИСХОДНЫХ доках (mojibake/html/boiler)")
    ap.add_argument("--char-sub-frac", type=float, default=None,
                    help="доля доков со сквозной подменой символа (по умолч. = --noise)")
    ap.add_argument("--junk-frac", type=float, default=0.5,
                    help="офтоп-доков добавить = junk-frac * (число исходных), НЕ входит в noise")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    char_sub_frac = args.char_sub_frac if args.char_sub_frac is not None else args.noise

    if os.path.abspath(args.inp) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --in")
    shutil.copyfile(args.inp, args.out)
    conn = sqlite3.connect(args.out)
    rng = random.Random(args.seed)

    docs = conn.execute(
        "SELECT id, content FROM documents WHERE content IS NOT NULL AND length(content) > 50"
    ).fetchall()
    if not docs:
        raise SystemExit("no usable docs")
    has_len = "length" in [c[1] for c in conn.execute("PRAGMA table_info(documents)")]

    # STAGE 1: corrupt originals
    n_char = int(len(docs) * char_sub_frac)
    char_ids = set(d[0] for d in rng.sample(docs, n_char)) if n_char < len(docs) else set(d[0] for d in docs)
    for doc_id, content in docs:
        nc = corrupt_doc(content, args.noise, rng, do_char_sub=(doc_id in char_ids))
        if has_len:
            conn.execute("UPDATE documents SET content=?, length=? WHERE id=?", (nc, len(nc), doc_id))
        else:
            conn.execute("UPDATE documents SET content=? WHERE id=?", (nc, doc_id))

    # STAGE 2: add off-topic junk docs (separate, uncorrupted)
    added = 0
    if os.path.exists(args.offtopic):
        # new library + section so junk is loadable and self-identified
        cur = conn.execute("INSERT INTO libraries(name, description) VALUES('offtopic','injected off-topic junk')")
        lib_id = cur.lastrowid
        cur = conn.execute("INSERT INTO sections(library_id, name, path) VALUES(?, 'offtopic', 'offtopic')", (lib_id,))
        sec_id = cur.lastrowid
        junk = [json.loads(l) for l in open(args.offtopic, encoding="utf-8") if l.strip()]
        rng.shuffle(junk)
        target = int(len(docs) * args.junk_frac)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(documents)")]
        for i in range(min(target, len(junk))):
            txt = junk[i].get("text", "")
            if len(txt) < 50:
                continue
            name = f"offtopic_{i}"
            if has_len:
                conn.execute("INSERT INTO documents(section_id, name, content, length) VALUES(?,?,?,?)",
                             (sec_id, name, txt, len(txt)))
            else:
                conn.execute("INSERT INTO documents(section_id, name, content) VALUES(?,?,?)",
                             (sec_id, name, txt))
            added += 1
    else:
        print(f"WARNING: offtopic file not found ({args.offtopic}) — STAGE 2 skipped")

    conn.commit()
    conn.close()
    print(f"originals corrupted : {len(docs)}  (noise={args.noise}, char_sub on {len(char_ids)} docs)")
    print(f"offtopic docs added : {added}  (junk_frac={args.junk_frac})")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
