"""Tests of the JSON corpus format (src/db/json_corpus.py).

Run with pytest or directly: ``python tests/test_json_corpus.py``.
Uses the corpus shipped in the repository (data/docs_database_examples.json.gz),
cut down to a few documents.
"""

import copy
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.db.json_corpus import (  # noqa: E402
    CorpusFormatError,
    corpus_to_sqlite,
    read_json,
    resolve_sqlite_path,
    sqlite_to_corpus,
    subset_documents,
    validate,
    write_json,
)

CORPUS = os.path.join(ROOT, "data", "docs_database_examples.json.gz")
_SMALL = None


def small_corpus() -> dict:
    global _SMALL
    if _SMALL is None:
        corpus = read_json(CORPUS)
        docs = corpus["tables"]["documents"]["rows"][:50]
        keep = {row[0] for row in docs}
        corpus["tables"]["documents"]["rows"] = docs
        corpus["tables"]["examples"]["rows"] = [
            row for row in corpus["tables"]["examples"]["rows"] if row[1] in keep]
        corpus.pop("meta", None)
        _SMALL = corpus
    return copy.deepcopy(_SMALL)


def expect_error(corpus, fragment: str) -> None:
    try:
        validate(corpus)
    except CorpusFormatError as exc:
        assert fragment in str(exc), f"{fragment!r} not in {exc}"
        return
    raise AssertionError(f"no error, expected {fragment!r}")


def test_round_trip_json_sqlite_json():
    corpus = small_corpus()
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "c.db")
        corpus_to_sqlite(corpus, db)
        back = sqlite_to_corpus(db)
        assert back["tables"] == corpus["tables"]
        assert back["sqlite_sequence"] == corpus["sqlite_sequence"]


def test_gzip_output_is_deterministic():
    corpus = small_corpus()
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.json.gz"), os.path.join(tmp, "b.json.gz")
        write_json(corpus, a)
        write_json(corpus, b)
        assert open(a, "rb").read() == open(b, "rb").read()
        assert read_json(a)["tables"] == corpus["tables"]


def test_subset_keeps_rows_and_drops_examples_of_removed_documents():
    corpus = small_corpus()
    keep = [row[0] for row in corpus["tables"]["documents"]["rows"][::2]]
    sub = subset_documents(corpus, keep)
    validate(sub)
    assert [row[0] for row in sub["tables"]["documents"]["rows"]] == keep
    assert all(row[1] in set(keep) for row in sub["tables"]["examples"]["rows"])
    assert sub["tables"]["sections"] == corpus["tables"]["sections"]


def test_resolve_sqlite_path_converts_once_and_reuses():
    corpus = small_corpus()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        write_json(corpus, path)
        first = resolve_sqlite_path(path)
        mtime = os.path.getmtime(first)
        assert resolve_sqlite_path(path) == first and os.path.getmtime(first) == mtime
        with sqlite3.connect(first) as conn:
            assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 50
        assert resolve_sqlite_path(os.path.join(tmp, "x.db")) == os.path.join(tmp, "x.db")


def test_validation_errors():
    docs = lambda c: c["tables"]["documents"]["rows"]  # noqa: E731

    c = small_corpus(); c.pop("tables"); expect_error(c, "tables")
    c = small_corpus(); c["format"] = "x"; expect_error(c, "format")
    c = small_corpus(); c["version"] = 2; expect_error(c, "version")
    c = small_corpus(); c["extra"] = 1; expect_error(c, "неизвестные поля")
    c = small_corpus(); docs(c)[0].pop(); expect_error(c, "значений")
    c = small_corpus(); docs(c)[0][3] = 5; expect_error(c, "ожидалась строка")
    c = small_corpus(); docs(c)[0][0] = 0; expect_error(c, "вне диапазона")
    c = small_corpus(); docs(c)[1][0] = docs(c)[0][0]; expect_error(c, "повтор id")
    c = small_corpus(); docs(c)[0][3] = "a\x07b"; expect_error(c, "недопустимый символ")
    c = small_corpus(); docs(c)[0][1] = 999; expect_error(c, "нет секции")
    c = small_corpus(); docs(c)[0][5] = "вчера"; expect_error(c, "дата")
    c = small_corpus(); c["tables"]["examples"]["rows"][0][1] = 10 ** 6; expect_error(c, "нет документа")
    c = small_corpus(); c["schema"] = c["schema"][:-1]; expect_error(c, "schema")
    c = small_corpus(); c["sqlite_sequence"][2][1] = 1; expect_error(c, "счётчик")
    c = small_corpus()
    c["tables"]["documents"]["rows"] = []
    c["tables"]["examples"]["rows"] = []
    expect_error(c, "пуста")


def test_c1_control_characters_are_allowed():
    c = small_corpus()
    c["tables"]["documents"]["rows"][0][3] = "b'\x81\x82'\ttab\nline"
    validate(c)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"ok  {name}")
