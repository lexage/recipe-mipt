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


class DocstringsDB:
    """
    База данных с тремя основными таблицами:
    - documents: информация из universal_parser (без кода),
    - examples: примеры из extract_examples_from_docstring,
    - code: сигнатуры и return info из universal_parser.
    """

    def __init__(self, db_path: str = "docs_from_docstrings.db") -> None:
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

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                library TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT,
                parent_class TEXT,
                filepath TEXT NOT NULL,
                docstring TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER REFERENCES documents(id),
                order_id INTEGER,
                content TEXT NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER REFERENCES documents(id),
                signature TEXT,
                return_info TEXT
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_library ON documents(library)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_examples_document ON examples(document_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_code_document ON code(document_id)"
        )

        conn.commit()

    def get_stats(self) -> None:
        conn = self.connect()
        cursor = conn.cursor()

        logger.info("Статистика по библиотекам:")
        logger.info("-" * 60)
        cursor.execute(
            """
            SELECT
                library,
                COUNT(*) AS docs_count
            FROM documents
            GROUP BY library
            ORDER BY library
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

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def process_library(
    library_name: str,
    site_packages_root: Path,
    db: DocstringsDB,
) -> None:
    """
    Обрабатывает одну библиотеку:
    - извлекает docstrings через universal_parser;
    - извлекает из них примеры через ExampleParser;
    - записывает данные в три таблицы: documents, examples, code.
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

    for doc in processed_docs:
        info = doc.information

        cursor.execute(
            """
            INSERT INTO documents (
                library, type, name, parent_class, filepath, docstring
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                info.library,
                info.type,
                info.name,
                info.parent_class,
                info.filepath,
                doc.string,
            ),
        )
        document_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO code (document_id, signature, return_info)
            VALUES (?, ?, ?)
            """,
            (document_id, info.signature, info.return_info),
        )

        examples = getattr(doc, "examples", None) or []
        for order_id, example_content in enumerate(examples):
            cursor.execute(
                """
                INSERT INTO examples (document_id, order_id, content)
                VALUES (?, ?, ?)
                """,
                (document_id, order_id, example_content),
            )

    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Создание базы данных docstrings, примеров и кода "
            "на основе universal_parser и ExampleParser"
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default="docs_from_docstrings.db",
        help="Путь к файлу базы данных (по умолчанию: docs_from_docstrings.db)",
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

    db = DocstringsDB(args.db)
    db.setup_database()

    site_packages_root = Path(args.site_packages).resolve()
    logger.info("Используется site-packages: %s", site_packages_root)

    for lib in LIBRARY_NAMES:
        process_library(lib, site_packages_root, db)

    db.get_stats()
    db.close()


if __name__ == "__main__":
    main()

