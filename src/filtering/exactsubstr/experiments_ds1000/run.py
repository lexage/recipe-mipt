import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.agent_constructor.core import Chunk
from src.filtering.exactsubstr.main import ExactSubstrFiltrator
from src.filtering.exactsubstr.experiments_ds1000.utils import *


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
        description="Filter DS1000 dataset using the exact substring filtering method from the paper.",
    )

    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/workspace/proj/grant/data/docs_database_dedup.db",
        help="Path to the SQLite database file",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=60,
        help="Threshold length for removing short or duplicate-like chunks",
    )

    parser.add_argument(
        "--output_filtered",
        type=str,
        default="filtered_data.json",
        help="Output path for the filtered chunks in JSON format",
    )

    parser.add_argument(
        "--output_stats",
        type=str,
        default="filter_stats.json",
        help="Output path for filtering statistics in JSON format",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit the number of chunks processed (useful for testing)",
    )

    args = parser.parse_args()

    data_path = args.data_path
    threshold = args.threshold
    limit = args.limit

    sql_client = SQLiteDocsDBAdapter(data_path)
    chunks = sql_client.get_docs()
    if limit is not None:
        chunks = chunks[:limit]

    print(f"Loaded {len(chunks)} chunks.")

    filtrator = ExactSubstrFiltrator(
        threshold=threshold, enable_bytes=True, enable_tokenizer=False, tokenizer=None
    )

    start_time = time.time()
    filtrate_chunks = filtrator.apply(chunks)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Filtering completed in {elapsed_time:.2f} seconds.")
    print(f"Kept {len(filtrate_chunks)} out of {len(chunks)} chunks after filtering.")

    process_filtering_results(
        chunks=chunks,
        filtrate_chunks=filtrate_chunks,
        filtrator=filtrator,
        threshold=args.threshold,
        elapsed_time=elapsed_time,
        output_filtered=args.output_filtered,
        output_stats=args.output_stats,
    )


if __name__ == "__main__":
    main()
