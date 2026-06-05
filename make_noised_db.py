"""Make a SEPARATE dirtied copy of a corpus DB — dirt is mixed INSIDE real docs.

For metric R (recovery). We keep every real document but inject dirt at THREE
granularities, so each cleaning level has something to act on:

  * whole junk LINES interleaved between good lines        -> cleaner level 2
  * symbols/letters inserted INTO words ("ar#r@ay")        -> cleaner level 1
  * words REPLACED by random / foreign-language tokens     -> cleaner level 3

ONE knob controls how dirty: `--noise` = the TOTAL share of dirt in the result,
measured in WORDS (tokens). The split BETWEEN the three dirt types is FIXED by
proportions (`--p-line --p-char --p-swap`, default 1:1:1) and stays the same at
every noise level — only the total grows. So `--noise 0.25/0.5/0.75` gives ~25 /
50 / 75 % dirty words with identical internal proportions.

Math (words): with good-word count G, total proportion p_line+p_char+p_swap=1
and target fraction f, the final word count is N = G/(1 - p_line*f), and we make
  char  = p_char*f*N words corrupted,
  swap  = p_swap*f*N words replaced,
  junk  = p_line*f*N words added as whole junk lines,
so dirty/N = f exactly (junk rounds to whole lines, hence ±a little).

The clean DB is never touched; the dirty lives in its own file (reproducible).

RUN ON THE SERVER (by Konstantin), e.g. the 50% level:
  python make_noised_db.py --in  data/docs_database_examples_apis.db \
                           --out data/docs_database_examples_apis_dirty_d50.db \
                           --dirty-frac 1.0 --noise 0.5 --seed 7
"""

import argparse
import os
import random
import re
import shutil
import sqlite3
import string

_BOILER = [
    "Copyright (c) 2024 ACME Corporation. All rights reserved.",
    "Navigation: Home > Documentation > API Reference",
    "Privacy Policy | Terms of Service | Contact Us",
    "Edit this page on GitHub. Was this helpful? Yes / No.",
    "Sign in to rate this page. Share on social media.",
]
_FILLER = [
    "the data is the data and the data is here for the data.",
    "this section describes the thing that describes this section.",
    "note: see the note above for the note below about this note.",
]
_FOREIGN = [
    "привет", "значение", "массив", "число", "функция",
    "das", "haus", "wert", "zahl", "wissen",
    "bonjour", "valeur", "fromage", "nombre",
    "你好", "世界", "数值", "函数",
    "مرحبا", "قيمة", "رقم",
    "こんにちは", "値", "関数",
    "gato", "casa", "valor", "número",
]
_JUNK_SYM = "!@#$%^&*~`"
_CODEISH = re.compile(r"[=(){}\[\]]|def |class |import |return ")
_WORDISH = re.compile(r"[A-Za-z]{3,}")


def garbled_line(rng, n=60):
    alpha = string.ascii_letters + string.digits + string.punctuation + "  "
    return "".join(rng.choice(alpha) for _ in range(n))


def dirt_line(rng, prose_pool, code_pool):
    """A whole junk LINE (cleaner level 2)."""
    t = rng.choice(["boiler", "garbled", "filler", "offtopic", "wrongcode"])
    if t == "boiler":
        return rng.choice(_BOILER)
    if t == "garbled":
        return garbled_line(rng)
    if t == "filler":
        return rng.choice(_FILLER)
    if t == "offtopic":
        return rng.choice(prose_pool) if prose_pool else rng.choice(_FILLER)
    return rng.choice(code_pool) if code_pool else garbled_line(rng)


def corrupt_word(word, rng, n=None):
    """Insert random junk symbols and/or letters into a word (level 1)."""
    chars = list(word)
    n = n if n is not None else rng.randint(1, 3)
    for _ in range(n):
        pos = rng.randint(0, len(chars))
        if rng.random() < 0.5:
            chars.insert(pos, rng.choice(_JUNK_SYM))
        else:
            chars.insert(pos, rng.choice(string.ascii_lowercase))
    return "".join(chars)


def swap_word(rng):
    """A replacement token: random ascii gibberish or a foreign-language word (level 3)."""
    if rng.random() < 0.5:
        return rng.choice(_FOREIGN)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9)))


def inject_dirt(content, f, p_line, p_char, p_swap, rng, prose_pool, code_pool):
    good_lines = [ln for ln in (content or "").split("\n") if ln.strip()]
    if not good_lines:
        return content

    line_tokens = [ln.split() for ln in good_lines]
    G = sum(len(toks) for toks in line_tokens)
    if G == 0:
        return content

    denom = 1.0 - p_line * f
    N = G / denom if denom > 0 else float(G)
    n_char = round(p_char * f * N)
    n_swap = round(p_swap * f * N)
    j_target = round(p_line * f * N)          # junk words to add

    # eligible word tokens (have a 3+ letter run) for char/swap
    eligible = [
        (li, ti)
        for li, toks in enumerate(line_tokens)
        for ti, t in enumerate(toks)
        if _WORDISH.search(t)
    ]
    rng.shuffle(eligible)
    n_char = min(n_char, len(eligible))
    n_swap = min(n_swap, len(eligible) - n_char)

    for li, ti in eligible[:n_char]:                       # level 1: corrupt
        line_tokens[li][ti] = corrupt_word(line_tokens[li][ti], rng)
    for li, ti in eligible[n_char:n_char + n_swap]:        # level 3: swap
        line_tokens[li][ti] = swap_word(rng)

    out = [" ".join(toks) for toks in line_tokens]

    junk_words = 0                                          # level 2: junk lines
    while junk_words < j_target:
        jl = dirt_line(rng, prose_pool, code_pool)
        out.insert(rng.randint(0, len(out)), jl)
        junk_words += len(jl.split())

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--dirty-frac", type=float, default=1.0,
                    help="доля документов, которые пачкаем (1.0 = все)")
    ap.add_argument("--noise", type=float, default=0.5,
                    help="ОБЩАЯ доля мусора в словах (0.25 / 0.5 / 0.75)")
    ap.add_argument("--p-line", type=float, default=1.0,
                    help="пропорция: целые мусорные строки (уровень 2)")
    ap.add_argument("--p-char", type=float, default=1.0,
                    help="пропорция: порча слов символами/буквами (уровень 1)")
    ap.add_argument("--p-swap", type=float, default=1.0,
                    help="пропорция: подмена слов случайными/иноязычными (уровень 3)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not 0.0 <= args.noise < 1.0:
        raise SystemExit(f"--noise must be in [0,1), got {args.noise}")
    psum = args.p_line + args.p_char + args.p_swap
    if psum <= 0:
        raise SystemExit("proportions must sum to > 0")
    p_line, p_char, p_swap = args.p_line / psum, args.p_char / psum, args.p_swap / psum

    if not os.path.exists(args.inp):
        raise SystemExit(f"input DB not found: {args.inp}")
    if os.path.abspath(args.inp) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --in (keep the clean DB intact)")

    shutil.copyfile(args.inp, args.out)
    conn = sqlite3.connect(args.out)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(documents)")]
    if not {"id", "content"}.issubset(cols):
        raise SystemExit(f"documents table missing columns; has {cols}")
    has_len = "length" in cols

    docs = conn.execute(
        "SELECT id, content FROM documents "
        "WHERE content IS NOT NULL AND length(content) > 50"
    ).fetchall()
    if not docs:
        raise SystemExit("no usable real docs found")

    rng = random.Random(args.seed)

    sample = rng.sample(docs, min(len(docs), 400))
    prose_pool, code_pool = [], []
    for _id, content in sample:
        for ln in (content or "").split("\n"):
            ln = ln.strip()
            if len(ln) < 15:
                continue
            (code_pool if _CODEISH.search(ln) else prose_pool).append(ln)
    prose_pool = prose_pool[:5000]
    code_pool = code_pool[:5000]

    n_target = int(len(docs) * args.dirty_frac)
    to_dirty = rng.sample(docs, n_target) if n_target < len(docs) else docs

    updated = 0
    for doc_id, content in to_dirty:
        new_content = inject_dirt(
            content, args.noise, p_line, p_char, p_swap, rng, prose_pool, code_pool,
        )
        if has_len:
            conn.execute("UPDATE documents SET content=?, length=? WHERE id=?",
                         (new_content, len(new_content), doc_id))
        else:
            conn.execute("UPDATE documents SET content=? WHERE id=?",
                         (new_content, doc_id))
        updated += 1

    conn.commit()
    conn.close()

    print(f"docs total      : {len(docs)}")
    print(f"docs dirtied    : {updated}  (dirty_frac={args.dirty_frac})")
    print(f"noise (total)   : {args.noise}  (доля мусорных слов в датасете)")
    print(f"proportions     : line={p_line:.2f}  char={p_char:.2f}  swap={p_swap:.2f}")
    print(f"saved dirty DB  -> {args.out}")


if __name__ == "__main__":
    main()
