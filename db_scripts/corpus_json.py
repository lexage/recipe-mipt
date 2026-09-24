"""Конвертация базы документов SQLite <-> JSON и проверка JSON-корпуса.

Корпус хранится в репозитории как JSON (data/docs_database_examples.json.gz);
пайплайн принимает его как path_to_db и сам собирает из него SQLite. Этот
скрипт делает то же самое руками и умеет доказать, что преобразование без
потерь.

    python db_scripts/corpus_json.py to-json data/docs_database_examples.db data/docs_database_examples.json.gz
    python db_scripts/corpus_json.py to-db data/docs_database_examples.json.gz /tmp/corpus.db
    python db_scripts/corpus_json.py validate data/docs_database_examples.json.gz
    python db_scripts/corpus_json.py compare data/docs_database_examples.db /tmp/corpus.db

compare сверяет две базы построчно по хешу содержимого каждой таблицы,
включая sqlite_sequence; код возврата 0 — базы совпадают.

Коды возврата: 0 — успех, 2 — входные данные не прошли проверку формата,
1 — базы различаются (compare) или внутренняя ошибка.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.json_corpus import (  # noqa: E402
    TABLES,
    CorpusFormatError,
    corpus_stats,
    corpus_to_sqlite,
    file_sha256,
    read_json,
    sqlite_to_corpus,
    write_json,
)


def _table_digest(conn: sqlite3.Connection, table: str) -> tuple:
    digest = hashlib.sha256()
    count = 0
    for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
        digest.update(json.dumps(list(row), ensure_ascii=False).encode("utf-8"))
        count += 1
    return count, digest.hexdigest()


def cmd_to_json(args) -> int:
    meta = {"source_sqlite": os.path.basename(args.db)}
    corpus = sqlite_to_corpus(args.db, meta=meta)
    write_json(corpus, args.json)
    print(json.dumps({"status": "ok", "json": args.json,
                      "sha256": file_sha256(args.json),
                      "bytes": os.path.getsize(args.json),
                      **corpus_stats(corpus)}, ensure_ascii=False, indent=2))
    return 0


def cmd_to_db(args) -> int:
    if os.path.exists(args.db) and not args.overwrite:
        print(f"файл уже существует: {args.db} (добавьте --overwrite)", file=sys.stderr)
        return 1
    corpus = read_json(args.json)
    corpus_to_sqlite(corpus, args.db)
    print(json.dumps({"status": "ok", "db": args.db, **corpus_stats(corpus)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args) -> int:
    corpus = read_json(args.json)
    print(json.dumps({"status": "ok", "json": args.json,
                      "sha256": file_sha256(args.json), **corpus_stats(corpus)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_compare(args) -> int:
    report = {}
    same = True
    conns = [sqlite3.connect(f"file:{path}?mode=ro", uri=True) for path in (args.a, args.b)]
    try:
        for table in (*TABLES, "sqlite_sequence"):
            (count_a, digest_a), (count_b, digest_b) = (_table_digest(c, table) for c in conns)
            equal = (count_a, digest_a) == (count_b, digest_b)
            same &= equal
            report[table] = {"rows_a": count_a, "rows_b": count_b, "identical": equal}
    finally:
        for conn in conns:
            conn.close()
    print(json.dumps({"status": "identical" if same else "different", "tables": report},
                     ensure_ascii=False, indent=2))
    return 0 if same else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("to-json", help="SQLite -> JSON (.json или .json.gz)")
    p.add_argument("db")
    p.add_argument("json")
    p.set_defaults(func=cmd_to_json)

    p = sub.add_parser("to-db", help="JSON -> SQLite")
    p.add_argument("json")
    p.add_argument("db")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_to_db)

    p = sub.add_parser("validate", help="проверить JSON-корпус")
    p.add_argument("json")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("compare", help="сравнить две SQLite-базы построчно")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CorpusFormatError as exc:
        print(json.dumps({"status": "error",
                          "error": {"type": "invalid_input", "message": str(exc)}},
                         ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
