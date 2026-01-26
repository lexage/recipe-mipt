import os
import sys
import argparse
import json
import sqlite3
from typing import List
from pathlib import Path
from dataclasses import asdict

project_root = Path(__file__).resolve().parents[4]  # 4 уровня вверх
sys.path.insert(0, str(project_root))

from src.filtering.exactsubstr.main import (
    main_version_3,
)
from src.agent_constructor.core import Document


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Document]:

        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT d.id, d.filename, d.content, d.file_path, 
                    s.name as section, l.name as library
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
                WHERE section == 'user_guide'
            """
            )

            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row["id"],
                        source=row["library"],
                        text=row["content"],
                        metadata={},
                    )
                )

        return documents


def main():
    parser = argparse.ArgumentParser(
        description="Filtering DS1000 using the filtering method from the paper.",
    )

    parser.add_argument(
        "--data_path",
        "-c",
        type=str,
        default="/workspace/proj/grant/data/docs_database_dedup.db",
        help="Путь к файлу с конфигурациями",
    )

    args = parser.parse_args()

    data_path = args.data_path
    
    threshold = 50

    sql_client = SQLiteDocsDBAdapter(data_path)
    documents = sql_client.get_docs()[:2]
    
    docs = [document.text for document in documents]

    for i, doc in enumerate(docs):
        print("-"*40)
        print(f"doc_{i+1} = '{doc}'")
    
    
    # len_docs = [len(doc) for doc in docs]
    # print(*len_docs)
    
    # high_quality_documents = main_version_3(
    #     docs,
    #     threshold=threshold,
    #     enable_bytes=True,
    #     enable_tokenizer=False
    # )
    
    # # for i, doc in enumerate(high_quality_documents):
    # #     print(f"doc_{i+1} = '{doc}'")
    
    # len_docs_result = [len(doc) for doc in high_quality_documents]

    # print(*len_docs_result)
    

if __name__ == "__main__":

    main()
