import os
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))


class EmbeddingEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Union[str, List[str]], batch_size: int = DEFAULT_BATCH) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed(query)[0]

    @property
    def dim(self) -> int:
        return self.dimension
