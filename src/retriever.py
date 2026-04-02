from typing import List, Tuple
import numpy as np

from ingestion import Chunk
from embeddings import EmbeddingEngine
from vectorstore import FAISSVectorStore


class Retriever:
    def __init__(self, store: FAISSVectorStore, engine: EmbeddingEngine, top_k: int = 5):
        self.store = store
        self.engine = engine
        self.top_k = top_k

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Chunk, float]]:
        k = k or self.top_k
        query_vec = self.engine.embed_query(query)
        return self.store.search(query_vec, k=k)

    def retrieve_with_rerank(self, query: str, fetch_k: int = 20, return_k: int = 5) -> List[Tuple[Chunk, float]]:
        candidates = self.retrieve(query, k=fetch_k)
        if len(candidates) <= return_k:
            return candidates

        query_vec = self.engine.embed_query(query)
        chunk_vecs = self.engine.embed([c.text for c, _ in candidates])

        scores = []
        for i, (chunk, base_score) in enumerate(candidates):
            sim = float(np.dot(query_vec, chunk_vecs[i]))
            len_penalty = min(1.0, chunk.token_count / 512)
            scores.append((chunk, sim * 0.85 + base_score * 0.15 + len_penalty * 0.05))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:return_k]

    def mmr(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Chunk]:
        candidates = self.retrieve(query, k=fetch_k)
        if not candidates:
            return []

        query_vec = self.engine.embed_query(query)
        chunk_texts = [c.text for c, _ in candidates]
        vecs = self.engine.embed(chunk_texts)

        selected, remaining = [], list(range(len(candidates)))
        while len(selected) < k and remaining:
            best_idx, best_score = None, -np.inf
            for i in remaining:
                rel = float(np.dot(query_vec, vecs[i]))
                if not selected:
                    red = 0.0
                else:
                    red = max(float(np.dot(vecs[i], vecs[j])) for j in selected)
                score = lambda_mult * rel - (1 - lambda_mult) * red
                if score > best_score:
                    best_idx, best_score = i, score
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [candidates[i][0] for i in selected]
