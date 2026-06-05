"""Make a SEPARATE dirtied copy of a corpus DB — dirt is mixed INSIDE real docs.

For metric R (recovery). We do NOT add separate junk documents (the retriever
would just ignore them). Instead we keep every real document but inject dirt at
THREE granularities, so each cleaning level has something to act on:

  * LEVEL 3 (whole lines):   junk LINES interleaved between good lines
        (boilerplate, garbled, filler, off-topic, wrong-code).
  * LEVEL 1 (symbols/letters in a word):  inside good lines, some words get
        random JUNK SYMBOLS and/or random LETTERS inserted into them
        (e.g. "array" -> "ar#r@ay" or "arxray").
  * LEVEL 2 (whole words):   inside good lines, some words are REPLACED by a
        random token — random ascii gibberish or a real word in a random
        language (Cyrillic / CJK / Arabic / Latin-other).

After chunking each chunk therefore holds good content AND dirt together at all
three levels. A line-dropping filter can only fight level 3; a real cleaner has
to repair words (1), drop alien words (2) and drop alien lines (3).

The clean DB is never touched; the dirt lives in its own file (reproducible).

RUN ON THE SERVER (by Konstantin), e.g. the 50%-ish level:
  python make_noised_db.py --in  data/docs_database_examples_apis.db \
                           --out data/docs_database_examples_apis_dirty_d50.db \
                           --dirty-frac 1.0 --line-ratio 1.0 \
                           --char-frac 0.10 --swap-frac 0.10 --seed 7

  --line-ratio : junk LINES per good line (level 3); 1.0 ~ half the lines junk.
  --char-frac  : probability a word gets symbol/letter corruption (level 1).
  --swap-frac  : probability a word is replaced by a random/foreign word (lvl 2).
  --dirty-frac : fraction of documents to dirty (1.0 = all).
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
# Real words in several scripts/languages for level-2 word swaps.
_FOREIGN = [
    "привет", "значение", "массив", "число", "функция",      # Russian
    "das", "haus", "wert", "zahl", "wissen",                  # German
    "bonjour", "valeur", "fromage", "nombre",                 # French
    "你好", "世界", "数值", "函数",                              # Chinese
    "مرحبا", "قيمة", "رقم",                                    # Arabic
    "こんにちは", "値", "関数",                                  # Japanese
    "gato", "casa", "valor", "número",                        # Spanish
]
_JUNK_SYM = "!@#$%^&*~`"
_CODEISH = re.compile(r"[=(){}\[\]]|def |class |import |return ")
_WORD = re.compile(r"[A-Za-z]{3,}")


def garbled_line(rng, n=60):
    alpha = string.ascii_letters + string.digits + string.punctuation + "  "
    return "".join(rng.choice(alpha) for _ in range(n))


def dirt_line(rng, prose_pool, code_pool):
    """A whole junk LINE (level 3)."""
    t = rng.choice(["boiler", "garbled", "filler", "offtopic", "wrongcode"])
    if t == "boiler":
        return rng.choice(_BOILER)
    if t == "garbled":
        return garbled_line(rng)
    if t == "filler":
        return rng.choice(_FILLER)
    if t == "offtopic":
        return rng.choice(prose_pool) if prose_pool else rng.choice(_FILLER)
    return rng.choice(code_pool) if code_pool else garbled_line(rng)  # wrongcode


def corrupt_word(word, rng, n=None):
    """Insert random junk symbols and/or letters into a word (level 1)."""
    chars = list(word)
    n = n if n is not None else rng.randint(1, 3)
    for _ in range(n):
        pos = rng.randint(0, len(chars))
        if rng.random() < 0.5:
            chars.insert(pos, rng.choice(_JUNK_SYM))          # symbol noise
        else:
            chars.insert(pos, rng.choice(string.ascii_lowercase))  # letter noise
    return "".join(chars)


def swap_word(rng):
    """A replacement token: random ascii gibberish or a foreign-language word."""
    if rng.random() < 0.5:
        return rng.choice(_FOREIGN)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9)))


def corrupt_line(line, rng, char_frac, swap_frac):
    """Per-word corruption of a good line (levels 1 & 2)."""
    if char_frac <= 0 and swap_frac <= 0:
        return line

    def repl(m):
        w = m.group(0)
        r = rng.random()
        if r < swap_frac:
            return swap_word(rng)                 # level 2: alien word
        if r < swap_frac + char_frac:
            return corrupt_word(w, rng)           # level 1: junk in word
        return w

    return _WORD.sub(repl, line)


def inject_dirt(content, line_ratio, char_frac, swap_frac, rng, prose_pool, code_pool):
    good = [ln for ln in (content or "").split("\n") if ln.strip()]
    if not good:
        return content
    out = []
    acc = 0.0
    for gl in good:
        out.append(corrupt_line(gl, rng, char_frac, swap_frac))   # levels 1 & 2
        acc += line_ratio                                          # level 3
        while acc >= 1.0:
            out.append(dirt_line(rng, prose_pool, code_pool))
            acc -= 1.0
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--dirty-frac", type=float, default=1.0,
                    help="доля документов, которые пачкаем (1.0 = все)")
    ap.add_argument("--line-ratio", type=float, default=None,
                    help="грязных СТРОК на одну хорошую (уровень 3)")
    ap.add_argument("--dirt-ratio", type=float, default=None,
                    help="deprecated alias for --line-ratio")
    ap.add_argument("--char-frac", type=float, default=0.0,
                    help="вероятность порчи слова символами/буквами (уровень 1)")
    ap.add_argument("--swap-frac", type=float, default=0.0,
                    help="вероятность замены слова на случайное/иноязычное (уровень 2)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    line_ratio = args.line_ratio if args.line_ratio is not None else args.dirt_ratio
    if line_ratio is None:
        line_ratio = 1.0

    if not os.path.exists(args.inp):
        raise SystemExit(f"input DB not found: {args.inp}")
    if os.path.abspath(args.inp) == os.path.abspath(args.out):
        raise SystemExit("--out must differ from --in (keep the clean DB intact)")

    shutil.copyfile(args.inp, args.out)            # dirty starts as a copy of clean
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

    # pools of lines drawn from real docs (for off-topic / wrong-code dirt)
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
            content, line_ratio, args.char_frac, args.swap_frac,
            rng, prose_pool, code_pool,
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
    print(f"line_ratio (L3) : {line_ratio}  (грязных строк на хорошую)")
    print(f"char_frac  (L1) : {args.char_frac}  (порча слов символами/буквами)")
    print(f"swap_frac  (L2) : {args.swap_frac}  (замена слов случайными/иноязычными)")
    print(f"saved dirty DB  -> {args.out}")


if __name__ == "__main__":
    main()
