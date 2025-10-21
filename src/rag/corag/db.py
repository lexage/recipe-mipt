import sqlite3
import chromadb

from src.agents.agent_constructor.chunkers import Chunker
from src.agents.agent_constructor.core import Document
from src.rag.corag.models import DummyEmbeddingFunction

from tqdm import tqdm


BATCH_SIZE = 10


class CoragDB:
    def __init__(
            self, 
            chunker: Chunker, 
            path_to_db: str = 'docs_database.db', 
            path_to_vector_db: str = 'vector_docs_database', 
            collection_name: str = 'docs',
            ):

        client = chromadb.PersistentClient(path=path_to_vector_db)

        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        embedding_function = DummyEmbeddingFunction("tfidf_vectorizer.pkl")

        documents = self._load_docs(path_to_db)
        self.chunks = {chunk.id: chunk for chunk in self._get_chunks(chunker, documents)}

        if collection_name not in collection_names:

            all_texts = [chunk.text for chunk in list(self.chunks.values())]
            embedding_function.fit(all_texts)

            collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)

            for i in tqdm(range(0, len(self.chunks), BATCH_SIZE), desc="Vectorising"):

                batch_docs = [chunk.text for chunk in list(self.chunks.values())[i:i+BATCH_SIZE]]
                batch_ids = [chunk.id for chunk in list(self.chunks.values())[i:i+BATCH_SIZE]]
                collection.add(documents=batch_docs, ids=batch_ids)
        
        else:
            embedding_function = DummyEmbeddingFunction()
            collection = client.get_or_create_collection(collection_name, embedding_function=embedding_function)
        
        self.collection = collection


    def _get_chunks(self, chunker: Chunker, documents):
        chunks = []
        for document in documents:
            doc_chunks = chunker.chunk(document)
            chunks.extend(doc_chunks)
        return chunks

    def _load_docs(self, path_to_db: str):
        conn = sqlite3.connect(path_to_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.id, d.filename, d.content, d.file_path, 
                   s.name as section, l.name as library
            FROM documents d
            JOIN sections s ON d.section_id = s.id
            JOIN libraries l ON s.library_id = l.id
        ''')
        
        documents = []
        for row in cursor.fetchall():
            documents.append(Document(id=row['id'], source=row['library'], text=row['content'], metadata={}))
        conn.close()

        return documents
    
    def search(self, queries: list, n: int):
        chunks = []
        results = self.collection.query(query_texts=queries, n_results=n)
        all_chunk_ids = results.get("ids", [])
        
        for sub_chunk_ids in all_chunk_ids:
            
            sub_chunks = []
            
            for chunk_id in sub_chunk_ids:
                sub_chunks.append(self.chunks[chunk_id])
            
            chunks.append(sub_chunks)
        
        return chunks
