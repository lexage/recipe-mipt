import os
import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.filtering.textbooks_are_all_you_need.main import EducationValueClassifierFilter


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Chunk]:

        chunks = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT d.id, d.filename, d.content, d.file_path, 
                    s.name as section, l.name as library
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
                WHERE section == 'user_guide'
            """)

            for row in cursor.fetchall():
                chunks.append(
                    Chunk(
                        id=row["id"],
                        doc_id=row["id"],
                        text=row["content"],
                        metadata={},
                    )
                )

        return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Filtering DS1000 using the filtering method from the paper.",
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/workspace/proj/grant/data/docs_database_dedup.db",
        help="Path to the data file",
    )

    parser.add_argument(
        "--result_path",
        "-r",
        type=str,
        default="/workspace/proj/grant/data/high_quality_chunks.json",
        help="Path to the file with filtered chunks",
    )

    args = parser.parse_args()

    data_path = args.data_path
    result_path = args.result_path

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    sql_client = SQLiteDocsDBAdapter(data_path)
    chunks = sql_client.get_docs()[:50]

    classifier = EducationValueClassifierFilter()

    high_quality_chunks = classifier.apply(chunks)

    output_path = Path(result_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(doc) for doc in high_quality_chunks],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(high_quality_chunks)} documents to {output_path}")


if __name__ == "__main__":

    main()
