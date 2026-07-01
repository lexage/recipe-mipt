from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    SparseVector,
    Fusion,
    FusionQuery,
    Prefetch,
)
from fastembed import SparseTextEmbedding
from typing import List, Dict, Tuple
from tqdm import tqdm


from src.agent_constructor.core import Text, Chunk
from src.agent_constructor.agent import Agent


class QdrantDocsAdapter:
    
    def __init__(self, embedder: Agent, collection_name: str, path_to_db: str):
        
        self.client = QdrantClient(path=path_to_db)
        self.embedder = embedder
        self.collection_name = collection_name
        self.batch_size = 10
        self.sparse_embedder = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=2560,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": {}
                }
            )
        

    def add(self, chunks: List[Chunk]):
        points_count, chunks = self._filter_existing_chunks(chunks)

        if chunks == []:
            return
        
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Vectorizing"):
            id_offset = points_count + i
            batch = chunks[i:i + self.batch_size]

            points = self._points_from_batch(
                batch=batch,
                id_offset=id_offset
            )            

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
    

    def search(self, query: Text, top_k: int = 1) -> List[Chunk]:
        
        dense_query = self.embedder.run(query)[0]
        sparse_query = list(self.sparse_embedder.embed(query))[0]
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=top_k,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values
                    ),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        ).points

        chunks = []
        for point in search_results:
            metadata = (point.payload.get("metadata") or {}).copy()
            metadata["score"] = point.score
            chunks.append(
                Chunk(
                    id=point.payload.get("id"),
                    doc_id=point.payload.get("doc_id"),
                    text=point.payload.get("text"),
                    metadata=metadata,
                )
            )
        
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


    def _points_from_batch(self, batch: List[Chunk], id_offset: int) -> List[PointStruct]:
        
        if not batch:
            return []

        ids = [id+id_offset for id in range(len(batch))]
        
        payload : List[Dict] = []
        chunk_texts : List[Text] = []

        for chunk in batch:
            payload.append(self._chunk_payload(chunk))
            chunk_texts.append(chunk.text)
        
        dense_vectors = self.embedder.run(chunk_texts)
        sparse_vectors = list(self.sparse_embedder.embed(chunk_texts))

        points : List[PointStruct] = []

        for i in range(len(batch)):
            points.append(
                PointStruct(
                    id=ids[i],
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": {
                            "indices": sparse_vectors[i].indices,
                            "values": sparse_vectors[i].values, 
                        }
                    },
                    payload=payload[i]
                )
            )

        return points

    def close(self):
        self.client.close()

    @staticmethod
    def _chunk_payload(chunk: Chunk) -> Dict:
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            }
    
