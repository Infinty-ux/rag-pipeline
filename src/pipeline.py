from pathlib import Path
from typing import List, Optional, Union
import json

from ingestion import ingest_file, ingest_directory, Chunk
from embeddings import EmbeddingEngine
from vectorstore import FAISSVectorStore
from retriever import Retriever
from generator import Generator


class RAGPipeline:
    INDEX_DIR = Path("index")

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
        retrieval_strategy: str = "rerank",
        chunking_strategy: str = "recursive",
    ):
        self.engine = EmbeddingEngine(model_name=embedding_model)
        self.store = FAISSVectorStore(dim=self.engine.dim)
        self.retriever = Retriever(self.store, self.engine, top_k=top_k)
        self.generator = Generator(model=llm_model)
        self.retrieval_strategy = retrieval_strategy
        self.chunking_strategy = chunking_strategy
        self._indexed = False

    def index(self, source: Union[Path, str], strategy: Optional[str] = None):
        source = Path(source)
        strat = strategy or self.chunking_strategy

        if source.is_dir():
            chunks = ingest_directory(source, strategy=strat)
        else:
            chunks = ingest_file(source, strategy=strat)

        if not chunks:
            raise ValueError(f"No usable chunks from {source}")

        print(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = self.engine.embed(texts)
        self.store.add(embeddings, chunks)
        self._indexed = True
        print(f"Indexed {len(self.store)} chunks total.")

    def save_index(self, directory: Optional[Path] = None):
        directory = directory or self.INDEX_DIR
        self.store.save(directory)
        print(f"Index saved to {directory}/")

    def load_index(self, directory: Optional[Path] = None):
        directory = directory or self.INDEX_DIR
        self.store = FAISSVectorStore.load(directory)
        self.retriever.store = self.store
        self._indexed = True
        print(f"Loaded {len(self.store)} chunks from {directory}/")

    def query(self, question: str, stream: bool = False) -> dict:
        if not self._indexed:
            raise RuntimeError("No index loaded. Call .index() or .load_index() first.")

        if self.retrieval_strategy == "mmr":
            chunks = self.retriever.mmr(question, k=self.retriever.top_k)
            chunks_with_scores = [(c, 0.0) for c in chunks]
        elif self.retrieval_strategy == "rerank":
            chunks_with_scores = self.retriever.retrieve_with_rerank(question)
            chunks = [c for c, _ in chunks_with_scores]
        else:
            chunks_with_scores = self.retriever.retrieve(question)
            chunks = [c for c, _ in chunks_with_scores]

        result = self.generator.generate_with_citations(question, chunks)
        result["retrieved_chunks"] = [
            {"source": c.source, "page": c.page, "score": round(s, 4), "text": c.text[:300]}
            for c, s in chunks_with_scores
        ]
        return result
