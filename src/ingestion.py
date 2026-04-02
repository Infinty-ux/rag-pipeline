import re
from pathlib import Path
from typing import Iterator, List
from dataclasses import dataclass

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int
    token_count: int


LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def load_documents(path: Path):
    suffix = path.suffix.lower()
    loader_cls = LOADERS.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    loader = loader_cls(str(path))
    return loader.load()


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def sentence_aware_chunk(text: str, max_tokens: int = 512) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_tokens = [], [], 0
    for sent in sentences:
        t = count_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_file(path: Path, strategy: str = "recursive") -> List[Chunk]:
    docs = load_documents(path)
    chunks = []
    global_idx = 0
    for doc in docs:
        text = clean_text(doc.page_content)
        if not text:
            continue
        page = doc.metadata.get("page", 0)
        if strategy == "sentence":
            raw_chunks = sentence_aware_chunk(text)
        else:
            raw_chunks = recursive_chunk(text)

        for piece in raw_chunks:
            piece = piece.strip()
            if len(piece) < 20:
                continue
            chunks.append(Chunk(
                text=piece,
                source=str(path),
                page=page,
                chunk_index=global_idx,
                token_count=count_tokens(piece),
            ))
            global_idx += 1
    return chunks


def ingest_directory(directory: Path, strategy: str = "recursive") -> List[Chunk]:
    all_chunks = []
    for ext in LOADERS:
        for f in directory.rglob(f"*{ext}"):
            try:
                chunks = ingest_file(f, strategy=strategy)
                all_chunks.extend(chunks)
                print(f"  {f.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  Warning: {f.name} — {e}")
    return all_chunks
