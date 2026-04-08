import sqlite3
import chromadb

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm


from src.agent_constructor.core import Document, Text, Chunk
from src.agent_constructor.agent import Agent


class QdrantDocsAdapter:
    
    def __init__(self, embedder: Agent, collection_name: str, path_to_db: str):
        
        self.client = QdrantClient(path=path_to_db)
        self.embedder = embedder
        self.collection_name = collection_name
        self.batch_size = 10
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(
                    size=2560,
                    distance=Distance.COSINE,
                )
            )
        

    def add(self, chunks: List[Chunk]):
        points_count, chunks = self._filter_existing_chunks(chunks)

        if chunks == []:
            return
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Vectorizing"):
            
            id_offset = points_count + i

            batch = chunks[i:i + self.batch_size]

            payload : List[Dict] = [self._chunk_payload(chunk) for chunk in batch]
            ids = [id+id_offset for id in range(len(batch))]
            vectors = self.embedder.run([chunk.text for chunk in batch])

            points : List[PointStruct] = [
                PointStruct(
                    id=id,
                    vector=v,
                    payload=p
                ) 
                for id,p,v in zip(ids, payload, vectors)]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
    

    def search(self, query: Text, top_k: int = 1) -> List[Chunk]:
        
        embedded_query = self.embedder.run(query)[0]
        
        serach_results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            limit=top_k,
        ).points

        chunks = [
            Chunk(
                id=point.payload.get("id"),
                doc_id=point.payload.get("doc_id"),
                text=point.payload.get("text"),
                metadata=point.payload.get("metadata"),
            )
            for point in serach_results 
        ]
        
        return chunks

    def _filter_existing_chunks(self, chunks: List[Chunk]) -> Tuple[int, List[Chunk]]:

        n = self.client.count(
            collection_name=self.collection_name
        ).count

        all_points = self.client.query_points(
            collection_name=self.collection_name,
            limit=n
        ).points

        chunk_ids = [point.payload.get("id") for point in all_points]

        chunks = [chunk for chunk in chunks if not chunk.id in chunk_ids]

        return n, chunks


    @staticmethod
    def _chunk_payload(chunk: Chunk) -> Dict:
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            }
    