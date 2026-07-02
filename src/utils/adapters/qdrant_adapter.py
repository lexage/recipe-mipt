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
from concurrent.futures import ThreadPoolExecutor


from src.agent_constructor.core import Text, Chunk
from src.agent_constructor.agent import Agent


class QdrantDocsAdapter:
    
    def __init__(self, embedder: Agent, collection_name: str, path_to_db: str,
                 embed_batch_size: int = 64, num_workers: int = 4,
                 max_embed_chars: int = 60000, embed_max_items: int = 256):

        self.client = QdrantClient(path=path_to_db)
        self.embedder = embedder
        self.collection_name = collection_name
        self.batch_size = 10
        self.embed_batch_size = int(embed_batch_size)   # texts per embedder call
        self.num_workers = int(num_workers)             # parallel embedder calls
        # Guard the embedder's context limit: truncate any single text longer than
        # max_embed_chars, and pack a request until either its total chars reach
        # max_embed_chars or it holds embed_max_items texts (whichever first), so a
        # batch never overflows the model's max token length.
        self.max_embed_chars = int(max_embed_chars)
        self.embed_max_items = int(embed_max_items)
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

        if not chunks:
            return

        bs = self.embed_batch_size
        window = bs * max(1, self.num_workers)   # chunks embedded before each upsert

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex, \
                tqdm(total=len(chunks), desc="Vectorizing") as pbar:
            for w in range(0, len(chunks), window):
                wchunks = chunks[w:w + window]
                # Truncate over-long texts so no single input exceeds the embedder
                # context; the stored payload keeps the full text.
                wtexts = [(c.text or "")[:self.max_embed_chars] for c in wchunks]

                # dense: parallel char-budgeted batched calls (order preserved)
                subs = self._char_batches(wtexts, self.max_embed_chars,
                                          self.embed_max_items)
                dense = [v for sub in ex.map(self.embedder.run, subs) for v in sub]
                # sparse: local fastembed (BM25), batched
                sparse = list(self.sparse_embedder.embed(wtexts))

                points = [
                    PointStruct(
                        id=points_count + w + j,
                        vector={
                            "dense": dense[j],
                            "sparse": {
                                "indices": sparse[j].indices,
                                "values": sparse[j].values,
                            },
                        },
                        payload=self._chunk_payload(wchunks[j]),
                    )
                    for j in range(len(wchunks))
                ]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                pbar.update(len(wchunks))
    

    def close(self):
        self.client.close()


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


    @staticmethod
    def _char_batches(texts: List[Text], max_chars: int, max_items: int) -> List[List[Text]]:
        """Group texts (in order) into sub-requests, flushing when the running
        char total would exceed max_chars or the batch reaches max_items."""
        batches: List[List[Text]] = []
        cur: List[Text] = []
        cur_len = 0
        for t in texts:
            if cur and (cur_len + len(t) > max_chars or len(cur) >= max_items):
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(t)
            cur_len += len(t)
        if cur:
            batches.append(cur)
        return batches

    @staticmethod
    def _chunk_payload(chunk: Chunk) -> Dict:
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            }
    
