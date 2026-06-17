import sqlite3

from typing import List, Optional

from src.utils import DOCUMENT_SRC_DOCUMENTS, DOCUMENT_SRC_EXAMPLES
from src.agent_constructor.core import Document, Chunk
from src.utils.queries import GET_DOCUMENTS_QUERY, GET_EXAMPLES_QUERY


class SQLiteDocsDBAdapter:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db
        with sqlite3.connect(self.path_to_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            self.id_offset = cursor.fetchone()[0]

    def get_docs(self, ids: Optional[List[int]] = None) -> List[Document]:
        
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if ids:
                placeholders = ','.join('?' * len(ids))
                query = f"{GET_DOCUMENTS_QUERY} WHERE d.id IN ({placeholders})"
                cursor.execute(query, ids)
            else:
                cursor.execute(GET_DOCUMENTS_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'], 
                        source=DOCUMENT_SRC_DOCUMENTS, 
                        text=row['content'], metadata={
                            "library": row['library'],
                            "section": row['section'],
                            "doc_name": row['name'],
                        }
                    )
                )
        
        return documents
    
    def get_examples(self, ids: Optional[List[int]] = None) -> List[Document]:
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if ids:
                placeholders = ','.join('?' * len(ids))
                query = f"{GET_EXAMPLES_QUERY} WHERE e.id IN ({placeholders})"
                cursor.execute(query, ids)
            else:
                cursor.execute(GET_EXAMPLES_QUERY)
            
            for row in cursor.fetchall():
                documents.append(
                    Document(
                        id=row['id'] + self.id_offset, 
                        source=DOCUMENT_SRC_EXAMPLES, 
                        text=row['content'], 
                        metadata={
                            "doc_id": row['doc_id'],
                            "order_id": row['order_id'],
                        }
                    )
                )
        
        return documents
    
    def get_examples_by_doc_id(self, doc_id: str, order_ids: List[int]):
        examples = []
        with sqlite3.connect(self.path_to_db) as conn:
            
            cursor = conn.cursor()

            placeholders = ','.join('?' * len(order_ids))
            
            query = f"""SELECT e.id, e.order_id, e.doc_id, e.content
            FROM examples e
            WHERE e.doc_id == ?
            AND e.order_id IN ({placeholders})"""
            
            params = (doc_id, *order_ids)
            cursor.execute(query, params)

            for row in cursor.fetchall():
                examples.append(
                    Chunk(
                        id=str(row[0] + self.id_offset)+"_0",
                        doc_id=row[0] + self.id_offset,
                        text=row[3], 
                        metadata={
                            "doc_id": row[2],
                            "order_id": row[1],
                            "source": DOCUMENT_SRC_EXAMPLES,
                        }
                    )
                )
            return examples
