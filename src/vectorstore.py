import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from ingestion import Chunk


class FAISSVectorStore:
    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.chunks: List[Chunk] = []

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        else:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        assert embeddings.shape[1] == self.dim
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(directory / "meta.json", "w") as f:
            json.dump({"dim": self.dim, "count": len(self.chunks)}, f)

    @classmethod
    def load(cls, directory: Path) -> "FAISSVectorStore":
        with open(directory / "meta.json") as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    def __len__(self):
        return len(self.chunks)
