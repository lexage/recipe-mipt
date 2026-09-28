"""JSON form of the docs database — the component's input and output format.

The component is fed and returns the corpus as JSON (ТЗ 3.5.7.1), while the
pipeline keeps working on SQLite underneath: a JSON file named as
``path_to_db`` is turned into a SQLite file once (``resolve_sqlite_path``) and
``LocalDB`` reads that file exactly as before.

The JSON carries the whole database, table by table, so the conversion is
lossless in both directions — every row of ``libraries``, ``sections``,
``documents``, ``examples`` and ``sqlite_sequence`` comes back identical,
ids and timestamps included::

    {
      "format": "filtration_generation/docs-db",
      "version": 1,
      "schema": ["CREATE TABLE libraries (...)", ..., "CREATE INDEX ..."],
      "tables": {
        "libraries": {"columns": ["id", "name", "description"], "rows": [[1, "numpy", null], ...]},
        "sections":  {...}, "documents": {...}, "examples": {...}
      },
      "sqlite_sequence": [["libraries", 16], ...],
      "meta": {...}                                  # optional, free-form provenance
    }

Every load is validated against the rules on admissible characters, formats
and ranges of the input (ТЗ 3.5.5.1); a violation raises ``CorpusFormatError``
with a message that names the offending table, row and column.

Files ending in ``.gz`` are gzip-compressed with a zero timestamp, so the same
corpus always serialises to the same bytes.
"""

import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile

from typing import Dict, Iterable, List, Optional


FORMAT_NAME = "filtration_generation/docs-db"
FORMAT_VERSION = 1

TABLES = ("libraries", "sections", "documents", "examples")

# Column order of each table, as the SQLite schema declares it.
COLUMNS = {
    "libraries": ("id", "name", "description"),
    "sections": ("id", "library_id", "name", "path"),
    "documents": ("id", "section_id", "name", "content", "length", "created_at"),
    "examples": ("id", "doc_id", "order_id", "content"),
}

# The schema of data/docs_database_examples.db, verbatim from sqlite_master.
SCHEMA = (
    "CREATE TABLE libraries (\n"
    "                id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "                name TEXT NOT NULL UNIQUE,\n"
    "                description TEXT\n"
    "            )",
    "CREATE TABLE sections (\n"
    "                id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "                library_id INTEGER REFERENCES libraries(id),\n"
    "                name TEXT NOT NULL,\n"
    "                path TEXT NOT NULL,\n"
    "                FOREIGN KEY (library_id) REFERENCES libraries(id)\n"
    "            )",
    "CREATE TABLE documents (\n"
    "                id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "                section_id INTEGER REFERENCES sections(id),\n"
    "                name TEXT NOT NULL,\n"
    "                content TEXT NOT NULL,\n"
    "                length INTEGER,\n"
    "                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,\n"
    "                FOREIGN KEY (section_id) REFERENCES sections(id)\n"
    "            )",
    "CREATE TABLE examples (\n"
    "                id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
    "                doc_id INTEGER REFERENCES documents(id),\n"
    "                order_id INTEGER,\n"
    "                content TEXT NOT NULL\n"
    "            )",
    "CREATE INDEX idx_examples_doc ON examples(doc_id)",
    "CREATE INDEX idx_documents_content ON documents(content)",
    "CREATE INDEX idx_documents_section ON documents(section_id)",
    "CREATE INDEX idx_sections_library ON sections(library_id)",
)

TOP_LEVEL_KEYS = {"format", "version", "schema", "tables", "sqlite_sequence", "meta"}
REQUIRED_KEYS = ("format", "version", "schema", "tables")

# Admissible characters: any Unicode text except C0 control characters other
# than tab / line feed / carriage return, DEL, and lone surrogates (they cannot
# be stored as UTF-8). C1 controls stay allowed: the corpus has them in a
# legitimate cp037 encoding example (examples 41878 and 42924).
_FORBIDDEN_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")

# Ranges.
MAX_ID = 2 ** 63 - 1
MAX_TEXT_CHARS = 10_000_000       # one field; the largest document is 452 404
MAX_NAME_CHARS = 1_000
MAX_ORDER_ID = 1_000_000
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

CACHE_DIR_NAME = ".sqlite_cache"


class CorpusFormatError(ValueError):
    """The JSON corpus breaks the format; the message says where and how."""


# --------------------------------------------------------------------- paths
def is_json_path(path: str) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(".json") or lowered.endswith(".json.gz")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- reading JSON
def read_json(path: str) -> dict:
    """Read and validate a JSON corpus (plain or gzip, told apart by content)."""
    if not os.path.isfile(path):
        raise CorpusFormatError(f"файл корпуса не найден: {path}")
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise CorpusFormatError(
                f"{path}: повреждённый gzip-архив ({exc})") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusFormatError(
            f"{path}: файл не в кодировке UTF-8 (байт {exc.start})") from None
    if not text.strip():
        raise CorpusFormatError(f"{path}: пустой файл")
    try:
        corpus = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorpusFormatError(
            f"{path}: некорректный JSON — {exc.msg} "
            f"(строка {exc.lineno}, столбец {exc.colno})") from None
    validate(corpus)
    return corpus


# ------------------------------------------------------------------ validation
def _where(table: str, row_index: int, column: Optional[str] = None) -> str:
    place = f"таблица {table}, строка {row_index}"
    return f"{place}, поле {column}" if column else place


def _check_int(value, table, row_index, column, low, high, nullable=False):
    if value is None:
        if nullable:
            return
        raise CorpusFormatError(f"{_where(table, row_index, column)}: пустое значение (null)")
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusFormatError(
            f"{_where(table, row_index, column)}: ожидалось целое число, "
            f"получено {type(value).__name__}")
    if not low <= value <= high:
        raise CorpusFormatError(
            f"{_where(table, row_index, column)}: значение {value} вне "
            f"диапазона [{low}, {high}]")


def _check_text(value, table, row_index, column, max_chars,
                allow_empty=True, nullable=False):
    if value is None:
        if nullable:
            return
        raise CorpusFormatError(f"{_where(table, row_index, column)}: пустое значение (null)")
    if not isinstance(value, str):
        raise CorpusFormatError(
            f"{_where(table, row_index, column)}: ожидалась строка, "
            f"получено {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise CorpusFormatError(f"{_where(table, row_index, column)}: пустая строка")
    if len(value) > max_chars:
        raise CorpusFormatError(
            f"{_where(table, row_index, column)}: длина {len(value)} больше "
            f"допустимой {max_chars}")
    bad = _FORBIDDEN_CHARS.search(value)
    if bad:
        raise CorpusFormatError(
            f"{_where(table, row_index, column)}: недопустимый символ "
            f"U+{ord(bad.group()):04X} в позиции {bad.start()}")


def _normalise_sql(statement: str) -> str:
    return " ".join(str(statement).split())


def validate(corpus) -> None:
    """Check a parsed corpus against the format; raise CorpusFormatError."""
    if not isinstance(corpus, dict):
        raise CorpusFormatError(
            f"верхний уровень JSON должен быть объектом, получено "
            f"{type(corpus).__name__}")

    for key in REQUIRED_KEYS:
        if key not in corpus:
            raise CorpusFormatError(f"нет обязательного поля «{key}»")
    unknown = sorted(set(corpus) - TOP_LEVEL_KEYS)
    if unknown:
        raise CorpusFormatError(f"неизвестные поля верхнего уровня: {', '.join(unknown)}")

    if corpus["format"] != FORMAT_NAME:
        raise CorpusFormatError(
            f"поле format = {corpus['format']!r}, ожидалось {FORMAT_NAME!r}")
    if isinstance(corpus["version"], bool) or corpus["version"] != FORMAT_VERSION:
        raise CorpusFormatError(
            f"поле version = {corpus['version']!r}, поддерживается {FORMAT_VERSION}")

    schema = corpus["schema"]
    if not isinstance(schema, list) or not all(isinstance(s, str) for s in schema):
        raise CorpusFormatError("поле schema должно быть списком строк SQL")
    if [_normalise_sql(s) for s in schema] != [_normalise_sql(s) for s in SCHEMA]:
        raise CorpusFormatError(
            "поле schema не совпадает со схемой базы документов "
            "(libraries, sections, documents, examples и их индексы)")

    if "meta" in corpus and not isinstance(corpus["meta"], dict):
        raise CorpusFormatError("поле meta должно быть объектом")

    tables = corpus["tables"]
    if not isinstance(tables, dict):
        raise CorpusFormatError("поле tables должно быть объектом")
    for name in TABLES:
        if name not in tables:
            raise CorpusFormatError(f"в tables нет таблицы «{name}»")
    unknown = sorted(set(tables) - set(TABLES))
    if unknown:
        raise CorpusFormatError(f"неизвестные таблицы: {', '.join(unknown)}")

    for name in TABLES:
        table = tables[name]
        if not isinstance(table, dict) or set(table) != {"columns", "rows"}:
            raise CorpusFormatError(
                f"таблица {name}: ожидался объект с полями columns и rows")
        if table["columns"] != list(COLUMNS[name]):
            raise CorpusFormatError(
                f"таблица {name}: столбцы {table['columns']!r}, ожидались "
                f"{list(COLUMNS[name])!r}")
        if not isinstance(table["rows"], list):
            raise CorpusFormatError(f"таблица {name}: поле rows должно быть списком")
        width = len(COLUMNS[name])
        for index, row in enumerate(table["rows"]):
            if not isinstance(row, list):
                raise CorpusFormatError(f"{_where(name, index)}: строка должна быть списком")
            if len(row) != width:
                raise CorpusFormatError(
                    f"{_where(name, index)}: {len(row)} значений, ожидалось {width}")

    _validate_rows(tables)
    _validate_sequence(corpus.get("sqlite_sequence", []), tables)


def _unique_ids(table: str, rows: List[list]) -> set:
    seen = set()
    for index, row in enumerate(rows):
        _check_int(row[0], table, index, "id", 1, MAX_ID)
        if row[0] in seen:
            raise CorpusFormatError(f"{_where(table, index, 'id')}: повтор id {row[0]}")
        seen.add(row[0])
    return seen


def _validate_rows(tables: Dict[str, dict]) -> None:
    libraries = tables["libraries"]["rows"]
    library_ids = _unique_ids("libraries", libraries)
    names = set()
    for i, (_id, name, description) in enumerate(libraries):
        _check_text(name, "libraries", i, "name", MAX_NAME_CHARS, allow_empty=False)
        if name in names:
            raise CorpusFormatError(f"{_where('libraries', i, 'name')}: повтор имени {name!r}")
        names.add(name)
        _check_text(description, "libraries", i, "description", MAX_TEXT_CHARS, nullable=True)

    sections = tables["sections"]["rows"]
    section_ids = _unique_ids("sections", sections)
    for i, (_id, library_id, name, path) in enumerate(sections):
        _check_int(library_id, "sections", i, "library_id", 1, MAX_ID, nullable=True)
        if library_id is not None and library_id not in library_ids:
            raise CorpusFormatError(
                f"{_where('sections', i, 'library_id')}: нет библиотеки с id {library_id}")
        _check_text(name, "sections", i, "name", MAX_NAME_CHARS, allow_empty=False)
        _check_text(path, "sections", i, "path", MAX_NAME_CHARS)

    documents = tables["documents"]["rows"]
    if not documents:
        raise CorpusFormatError("таблица documents пуста: в корпусе нет ни одного документа")
    document_ids = _unique_ids("documents", documents)
    for i, (_id, section_id, name, content, length, created_at) in enumerate(documents):
        _check_int(section_id, "documents", i, "section_id", 1, MAX_ID, nullable=True)
        if section_id is not None and section_id not in section_ids:
            raise CorpusFormatError(
                f"{_where('documents', i, 'section_id')}: нет секции с id {section_id}")
        _check_text(name, "documents", i, "name", MAX_NAME_CHARS, allow_empty=False)
        _check_text(content, "documents", i, "content", MAX_TEXT_CHARS)
        _check_int(length, "documents", i, "length", 0, MAX_TEXT_CHARS, nullable=True)
        if created_at is not None:
            if not isinstance(created_at, str) or not _DATETIME_RE.match(created_at):
                raise CorpusFormatError(
                    f"{_where('documents', i, 'created_at')}: ожидалась дата вида "
                    f"'ГГГГ-ММ-ДД чч:мм:сс', получено {created_at!r}")

    examples = tables["examples"]["rows"]
    _unique_ids("examples", examples)
    for i, (_id, doc_id, order_id, content) in enumerate(examples):
        _check_int(doc_id, "examples", i, "doc_id", 1, MAX_ID, nullable=True)
        if doc_id is not None and doc_id not in document_ids:
            raise CorpusFormatError(
                f"{_where('examples', i, 'doc_id')}: нет документа с id {doc_id}")
        _check_int(order_id, "examples", i, "order_id", 0, MAX_ORDER_ID, nullable=True)
        _check_text(content, "examples", i, "content", MAX_TEXT_CHARS)


def _validate_sequence(sequence, tables: Dict[str, dict]) -> None:
    if not isinstance(sequence, list):
        raise CorpusFormatError("поле sqlite_sequence должно быть списком пар [таблица, число]")
    seen = set()
    for index, entry in enumerate(sequence):
        if (not isinstance(entry, list) or len(entry) != 2
                or entry[0] not in TABLES or entry[0] in seen):
            raise CorpusFormatError(
                f"sqlite_sequence, элемент {index}: ожидалась пара "
                f"[имя таблицы, число] без повторов, получено {entry!r}")
        seen.add(entry[0])
        _check_int(entry[1], "sqlite_sequence", index, "seq", 0, MAX_ID)
        rows = tables[entry[0]]["rows"]
        top = max((row[0] for row in rows), default=0)
        if entry[1] < top:
            raise CorpusFormatError(
                f"sqlite_sequence, {entry[0]}: счётчик {entry[1]} меньше "
                f"наибольшего id {top}")


# --------------------------------------------------------------- SQLite <-> JSON
def sqlite_to_corpus(db_path: str, meta: Optional[dict] = None) -> dict:
    """Read the whole SQLite docs database into the JSON form."""
    if not os.path.isfile(db_path):
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        schema = [row[0] for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name != 'sqlite_sequence' ORDER BY rowid")]
        tables = {}
        for name in TABLES:
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')]
            if tuple(columns) != COLUMNS[name]:
                raise CorpusFormatError(
                    f"{db_path}: таблица {name} имеет столбцы {columns}, "
                    f"ожидались {list(COLUMNS[name])}")
            rows = [list(row) for row in conn.execute(
                f'SELECT {", ".join(columns)} FROM "{name}" ORDER BY rowid')]
            tables[name] = {"columns": columns, "rows": rows}
        sequence = [list(row) for row in conn.execute(
            "SELECT name, seq FROM sqlite_sequence ORDER BY rowid")]
    finally:
        conn.close()
    corpus = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "schema": schema,
        "tables": tables,
        "sqlite_sequence": sequence,
    }
    if meta:
        corpus["meta"] = meta
    validate(corpus)
    return corpus


def write_json(corpus: dict, path: str) -> None:
    """Validate and write a corpus; the file appears only once complete."""
    validate(corpus)
    payload = json.dumps(corpus, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if str(path).lower().endswith(".gz"):
        payload = gzip.compress(payload, compresslevel=9, mtime=0)
    _atomic_write_bytes(path, payload)


def corpus_to_sqlite(corpus: dict, db_path: str) -> None:
    """Build a SQLite docs database from a validated corpus (atomically)."""
    validate(corpus)
    directory = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".db", dir=directory)
    os.close(fd)
    try:
        conn = sqlite3.connect(tmp_path)
        try:
            for statement in SCHEMA:
                conn.execute(statement)
            for name in TABLES:
                columns = COLUMNS[name]
                placeholders = ", ".join("?" * len(columns))
                conn.executemany(
                    f'INSERT INTO "{name}" ({", ".join(columns)}) VALUES ({placeholders})',
                    corpus["tables"][name]["rows"])
            conn.execute("DELETE FROM sqlite_sequence")
            conn.executemany("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)",
                             corpus.get("sqlite_sequence", []))
            conn.commit()
        finally:
            conn.close()
        os.chmod(tmp_path, _default_mode())
        os.replace(tmp_path, db_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def resolve_sqlite_path(path_to_db: str, cache_dir: Optional[str] = None) -> str:
    """Path the SQLite adapter should open for ``path_to_db``.

    A ``.db`` path is returned unchanged. A JSON corpus is validated and turned
    into ``<cache_dir>/<name>-<sha256[:16]>.db`` (default cache: a
    ``.sqlite_cache`` folder next to the JSON); an existing file for the same
    content hash is reused.
    """
    if not is_json_path(path_to_db):
        return path_to_db
    if not os.path.isfile(path_to_db):
        raise CorpusFormatError(f"файл корпуса не найден: {path_to_db}")
    digest = file_sha256(path_to_db)
    base = os.path.basename(path_to_db)
    stem = base[:-len(".json.gz")] if base.lower().endswith(".json.gz") else base[:-len(".json")]
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(path_to_db)),
                                          CACHE_DIR_NAME)
    target = os.path.join(cache_dir, f"{stem}-{digest[:16]}.db")
    if not os.path.isfile(target):
        corpus_to_sqlite(read_json(path_to_db), target)
    return target


# ------------------------------------------------------------------ subsetting
def subset_documents(corpus: dict, keep_document_ids: Iterable[int],
                     meta: Optional[dict] = None) -> dict:
    """Corpus restricted to the given documents.

    Documents keep their ids, rows and order. Examples go with their document
    (an example of a removed document is removed too — the same convention as
    the hand-made API base ``docs_database_examples_apis.db``, which has no
    orphan examples). Libraries, sections and ``sqlite_sequence`` are copied
    unchanged.
    """
    keep = set(keep_document_ids)
    tables = corpus["tables"]
    documents = [row for row in tables["documents"]["rows"] if row[0] in keep]
    examples = [row for row in tables["examples"]["rows"] if row[1] in keep]
    result = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "schema": list(corpus["schema"]),
        "tables": {
            "libraries": {"columns": list(COLUMNS["libraries"]),
                          "rows": [list(r) for r in tables["libraries"]["rows"]]},
            "sections": {"columns": list(COLUMNS["sections"]),
                         "rows": [list(r) for r in tables["sections"]["rows"]]},
            "documents": {"columns": list(COLUMNS["documents"]), "rows": documents},
            "examples": {"columns": list(COLUMNS["examples"]), "rows": examples},
        },
        "sqlite_sequence": [list(entry) for entry in corpus.get("sqlite_sequence", [])],
    }
    if meta:
        result["meta"] = meta
    return result


def corpus_stats(corpus: dict) -> dict:
    """Counts used in reports: documents, examples and characters of content."""
    documents = corpus["tables"]["documents"]["rows"]
    examples = corpus["tables"]["examples"]["rows"]
    return {
        "documents": len(documents),
        "examples": len(examples),
        "document_chars": sum(len(row[3]) for row in documents),
        "example_chars": sum(len(row[3]) for row in examples),
    }


def _default_mode() -> int:
    # mkstemp создаёт файл с правами 0600; готовому файлу — обычные права по umask.
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(tmp_path, _default_mode())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
