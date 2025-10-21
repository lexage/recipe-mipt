import sqlite3
import chromadb

from typing import List

from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.core import Document, Chunk, Text
from src.rag.corag.models import TFIDFEmbeddingFunction
from src.agents.agent_constructor.db import IDB

from tqdm import tqdm


BATCH_SIZE = 10


class SQLiteDataBase:
    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db

    def get_docs(self) -> List[Document]:
        
        documents = []
        with sqlite3.connect(self.path_to_db) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.id, d.filename, d.content, d.file_path, 
                    s.name as section, l.name as library
                FROM documents d
                JOIN sections s ON d.section_id = s.id
                JOIN libraries l ON s.library_id = l.id
            ''')
            
            for row in cursor.fetchall():
                documents.append(Document(id=row['id'], source=row['library'], text=row['content'], metadata={}))
        
        return documents


class ChromaVectorDataBase:
    def __init__(self, collection_name: str, path_to_db: str, embedding_function):
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.embedding_function = embedding_function

        client = chromadb.PersistentClient(path=path_to_db)
        
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        self.populated = collection_name in collection_names
        
        if self.populated:
            self.collection = client.get_or_create_collection(
                self.collection_name, 
                embedding_function=self.embedding_function
            )
        self.client = client

    def populate(self, chunks: List[Chunk]):
        all_texts = [chunk.text for chunk in chunks]
        self.embedding_function.fit(all_texts)
        self.collection = self.client.get_or_create_collection(
            self.collection_name, 
            embedding_function=self.embedding_function
        )

        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Vectorising"):

            batch_docs = [chunk.text for chunk in chunks[i:i+BATCH_SIZE]]
            batch_ids = [chunk.id for chunk in chunks[i:i+BATCH_SIZE]]
            self.collection.add(documents=batch_docs, ids=batch_ids)

    def search(self, queries: List[Text], top_k: int):
        results = self.collection.query(query_texts=queries, n_results=top_k)
        return results.get("ids", [[]])[0]


class LocalCoragDB(IDB):
    def __init__(
            self, 
            chunker: Chunker, 
            path_to_db: str = 'docs_database.db', 
            path_to_vector_db: str = 'vector_docs_database', 
            embedding_model: str = 'tfidf_vectorizer.pkl',
            collection_name: str = 'docs',
            ):

        self.doc_data_base = SQLiteDataBase(
            path_to_db=path_to_db
            )

        self.vector_data_base = ChromaVectorDataBase(
            collection_name=collection_name,
            path_to_db=path_to_vector_db,
            embedding_function=TFIDFEmbeddingFunction(embedding_model)
            )
        
        documents = self.doc_data_base.get_docs()
        
        self.chunks = {
            chunk.id: chunk for chunk in self._get_chunks(
                chunker=chunker, 
                documents=documents
                )
            }

        if not self.vector_data_base.populated:
            self.vector_data_base.populate(self.all_chunks())

    def _get_chunks(self, chunker: Chunker, documents: List[Document]):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks
    
    def query(self, queries: Text, top_k: int):
        chunk_ids = self.vector_data_base.search(queries=[queries], top_k=top_k)          
        return [self.chunks[chunk_id] for chunk_id in chunk_ids]
    
    def all_chunks(self):
        return list(self.chunks.values())
    
    def add_chunks(self, chunks):
        return super().add_chunks(chunks)
