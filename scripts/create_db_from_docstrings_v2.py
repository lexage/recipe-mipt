import sys
import argparse
import sqlite3
import logging
from pathlib import Path
from typing import List

# Настраиваем путь к корню проекта, чтобы можно было импортировать внутренние модули
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from doc_parser.universal_parser import extract_docstrings, String  # type: ignore
from scripts.extract_examples_from_docstring import (  # type: ignore
    ExampleParser,
    main as process_docstrings,
)


logger = logging.getLogger(__name__)


LIBRARY_NAMES: List[str] = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "torch",
    "tensorflow",
]


class DocstringsDBV2:
    """
    База данных c такой же структурой таблиц libraries/sections/documents/examples,
    как в create_db.py, плюс дополнительная таблица code.

    sections:
      - path  <- filepath из universal_parser
      - name  <- type (DocType) из universal_parser
    """

    def __init__(self, db_path: str = "docs_from_docstrings_v2.db") -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def setup_database(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()

        # Таблицы как в create_db.py
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS libraries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id INTEGER REFERENCES libraries(id),
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY (library_id) REFERENCES libraries(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER REFERENCES sections(id),
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                length INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (section_id) REFERENCES sections(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER REFERENCES documents(id),
                order_id INTEGER,
                content TEXT NOT NULL
            )
            """
        )

        # Дополнительная таблица code
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER REFERENCES documents(id),
                signature TEXT,
                return_info TEXT
            )
            """
        )

        # Индексы такие же, как в create_db.py (плюс индекс для code)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_examples_doc ON examples(doc_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_section ON documents(section_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sections_library ON sections(library_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_code_doc ON code(doc_id)")

        conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get_stats(self) -> None:
        """
        Выводит агрегированную статистику по библиотекам.
        Аналогично DocstringsDB.get_stats из create_db_from_docstrings.py,
        но с учётом схемы libraries/sections/documents.
        """
        conn = self.connect()
        cursor = conn.cursor()

        logger.info("Статистика по библиотекам:")
        logger.info("-" * 60)

        cursor.execute(
            """
            SELECT
                l.name AS library,
                COUNT(DISTINCT d.id) AS docs_count
            FROM libraries l
            LEFT JOIN sections s ON l.id = s.library_id
            LEFT JOIN documents d ON s.id = d.section_id
            GROUP BY l.name
            ORDER BY l.name
            """
        )

        for row in cursor.fetchall():
            logger.info(f"{row['library']:<20} docstrings: {row['docs_count']}")

        cursor.execute("SELECT COUNT(*) AS total_docs FROM documents")
        total_docs = cursor.fetchone()["total_docs"]
        cursor.execute("SELECT COUNT(*) AS total_examples FROM examples")
        total_examples = cursor.fetchone()["total_examples"]
        cursor.execute("SELECT COUNT(*) AS total_code FROM code")
        total_code = cursor.fetchone()["total_code"]

        logger.info("-" * 60)
        logger.info(f"Всего docstrings: {total_docs}")
        logger.info(f"Всего примеров:  {total_examples}")
        logger.info(f"Всего записей кода: {total_code}")


def process_library(
    library_name: str,
    site_packages_root: Path,
    db: DocstringsDBV2,
) -> None:
    """
    Обрабатывает одну библиотеку:
    - извлекает docstrings через universal_parser;
    - извлекает из них примеры через ExampleParser;
    - записывает данные в таблицы libraries/sections/documents/examples/code.
    """
    base_library_path = site_packages_root / library_name

    if not base_library_path.exists():
        logger.warning(
            "Библиотека %s не найдена по пути %s", library_name, base_library_path
        )
        return

    logger.info("Обработка библиотеки: %s", library_name)

    raw_docs: List[String] = extract_docstrings(str(base_library_path))

    parser = ExampleParser(
        enable_rst=True,
        enable_md=True,
        enable_repl=True,
        enable_free=True,
    )

    processed_docs, stats_sec, stats_ex = process_docstrings(raw_docs, parser)

    logger.info(
        "Библиотека %s: извлечено docstrings: %d, секций с примерами: %d, docstrings с примерами: %d",
        library_name,
        len(raw_docs),
        stats_sec,
        stats_ex,
    )

    conn = db.connect()
    cursor = conn.cursor()

    # libraries
    cursor.execute(
        "INSERT OR IGNORE INTO libraries (name) VALUES (?)",
        (library_name,),
    )
    cursor.execute("SELECT id FROM libraries WHERE name = ?", (library_name,))
    library_id = cursor.fetchone()[0]

    sections_cache: dict[tuple[int, str, str], int] = {}

    for doc in processed_docs:
        info = doc.information

        # sections: name <- type, path <- filepath
        section_key = (library_id, info.type, info.filepath)
        if section_key not in sections_cache:
            cursor.execute(
                """
                INSERT OR IGNORE INTO sections (library_id, name, path)
                VALUES (?, ?, ?)
                """,
                (library_id, info.type, info.filepath),
            )
            cursor.execute(
                """
                SELECT id FROM sections
                WHERE library_id = ? AND name = ? AND path = ?
                """,
                (library_id, info.type, info.filepath),
            )
            section_id = cursor.fetchone()[0]
            sections_cache[section_key] = section_id
        else:
            section_id = sections_cache[section_key]

        # documents: name — именно name объекта (без parent_class),
        # как вы хотите; для модульных docstrings name может быть пустым.
        doc_name = info.name or "<module>"

        content = doc.string or ""
        length = len(content)

        cursor.execute(
            """
            INSERT INTO documents (section_id, name, content, length)
            VALUES (?, ?, ?, ?)
            """,
            (section_id, doc_name, content, length),
        )
        doc_id = cursor.lastrowid

        # code: сигнатура и return_info
        cursor.execute(
            """
            INSERT INTO code (doc_id, signature, return_info)
            VALUES (?, ?, ?)
            """,
            (doc_id, info.signature, info.return_info),
        )

        # examples: примеры из doc.examples
        examples = getattr(doc, "examples", None) or []
        for order_id, example_content in enumerate(examples):
            cursor.execute(
                """
                INSERT INTO examples (doc_id, order_id, content)
                VALUES (?, ?, ?)
                """,
                (doc_id, order_id, example_content),
            )

    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Создание базы данных docstrings, примеров и кода "
            "со структурой, как в create_db.py (v2)"
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default="docs_from_docstrings_v2.db",
        help="Путь к файлу базы данных (по умолчанию: docs_from_docstrings_v2.db)",
    )
    parser.add_argument(
        "--site-packages",
        type=str,
        default="venv/lib/python3.12/site-packages",
        help=(
            "Путь к корню site-packages, где лежат библиотеки "
            "(по умолчанию: venv/lib/python3.12/site-packages)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db = DocstringsDBV2(args.db)
    db.setup_database()

    site_packages_root = Path(args.site_packages).resolve()
    logger.info("Используется site-packages: %s", site_packages_root)

    for lib in LIBRARY_NAMES:
        process_library(lib, site_packages_root, db)

    db.get_stats()
    db.close()


if __name__ == "__main__":
    main()

