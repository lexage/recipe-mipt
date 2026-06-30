import chromadb

from typing import List, Optional
from tqdm import tqdm
from chromadb.config import Settings

from src.agent_constructor.core import Text, Chunk
from src.utils.wrappers import EmbeddingFunctionWrapper


class ChromaDocsAdapter:
    
    def __init__(
            self, embedder, 
            collection_name: str, 
            path_to_db: str, 
            batch_size: int = 32,
            hnsw_space: str = "cosine",
            hnsw_m: int = 16,
            hnsw_construction_ef: int = 400,
            hnsw_search_ef: int = 200,
            threshold: float = 0.1,
            search_filter: dict = {}):
        
        self.filter = search_filter if search_filter != {} else None
        self.collection_name = collection_name
        self.path_to_db = path_to_db
        self.batch_size = batch_size
        self.threshold = threshold

        client = chromadb.PersistentClient(path=path_to_db, settings=Settings(anonymized_telemetry=False))
        
        self.collection = client.get_or_create_collection(
            self.collection_name, 
            embedding_function=EmbeddingFunctionWrapper(embedder),
            metadata={
                "hnsw:space": hnsw_space,
                "hnsw:M": hnsw_m,
                "hnsw:construction_ef": hnsw_construction_ef,
                "hnsw:search_ef": hnsw_search_ef,
            }
        )

    def _get_existing_ids(self) -> set:
        existing = self.collection.get(include=[])
        return set(existing['ids']) if existing['ids'] else set()

    def _filter_new_chunks(self, chunks: List[Chunk], existing_ids: set) -> List[Chunk]:
        return [chunk for chunk in chunks if chunk.id not in existing_ids]

    def add(self, chunks: List[Chunk]):
        existing_ids = self._get_existing_ids()
        new_chunks = self._filter_new_chunks(chunks, existing_ids)
        
        if not new_chunks:
            return
        
        for i in tqdm(range(0, len(new_chunks), self.batch_size), desc="Vectorizing"):
            batch = new_chunks[i:i + self.batch_size]
            
            batch_docs = []
            batch_ids = []
            batch_metadatas = []

            for chunk in batch:
                batch_docs.append(chunk.text)
                batch_ids.append(chunk.id)

                batch_metadatas.append({
                    **chunk.metadata,
                    "doc_id" : chunk.doc_id,
                })

            try:
                self.collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                print(f" - error adding batch : {batch_ids}, {batch_metadatas}\n - error msg : {e}")

    def get_chunks(self, ids: Optional[List[str]] = None) -> List[Chunk]:
        if ids:
            results = self.collection.get(ids=ids, include=['documents', 'metadatas'])
        else:
            results = self.collection.get(include=['documents', 'metadatas'])
        
        chunks = []
        result_ids = results['ids'] if results['ids'] else []
        documents = results['documents'] if results['documents'] else []
        metadatas = results['metadatas'] if results['metadatas'] else []
        
        for i, chunk_id in enumerate(result_ids):
            doc_id = metadatas[i].get('doc_id', None) if i < len(metadatas) else None
            
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=documents[i] if i < len(documents) else '',
                    tokens=None,
                    metadata=metadatas[i] if i < len(metadatas) else {}
                )
            )
        
        return chunks

    def search(self, queries: List[Text], top_k: int) -> List[List[Chunk]]:

        results = self.collection.query(
            query_texts=queries, 
            n_results=top_k,
            where=self.filter,
            )
    
        all_query_results = []
        
        for query_idx in range(len(results['ids'])):
            query_chunks = []
            ids = results['ids'][query_idx]
            distances = results['distances'][query_idx]
            documents = results['documents'][query_idx]
            metadatas = results['metadatas'][query_idx] if results['metadatas'] else [{}] * len(ids)
            
            for i, chunk_id in enumerate(ids):
                if distances[i] < self.threshold:
                    continue

                doc_id = metadatas[i].get('doc_id', None) if i < len(metadatas) else None
                
                query_chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        text=documents[i] if i < len(documents) else '',
                        tokens=None,
                        metadata=metadatas[i] if i < len(metadatas) else {}
                    )
                )
            
            all_query_results.append(query_chunks)
        
        return all_query_results
