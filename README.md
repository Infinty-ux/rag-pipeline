# RAG Pipeline

A production-quality **Retrieval-Augmented Generation** system that ingests PDF, TXT, and Markdown documents, builds a FAISS semantic index, and answers natural-language queries with cited sources.

## Architecture

```
Documents (PDF/TXT/MD)
        │
        ▼
  Ingestion layer
  ├─ PyPDF / TextLoader / Markdown loader
  ├─ Text cleaning
  └─ Chunking (recursive | sentence-aware)
        │
        ▼
  Embedding engine (BGE-small-en-v1.5)
        │
        ▼
  FAISS vector store (IndexFlatIP / HNSW / IVF)
        │
  Query ──► embed_query ──► similarity search
                                  │
                           Reranker / MMR
                                  │
                            Top-k chunks
                                  │
                     GPT-4o-mini with context
                                  │
                         Answer + citations
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in your OPENAI_API_KEY
```

## Usage

### Index documents

```bash
python src/cli.py index docs/ reports/annual.pdf --strategy recursive
```

Supports mixing files and directories. Chunks are saved to `index/`.

### Query

```bash
python src/cli.py query "What were the key findings in Q3?" --stream
```

### Interactive chat

```bash
python src/cli.py chat
```

### Advanced options

```bash
python src/cli.py --retrieval mmr --top-k 8 query "Summarize the risk factors"
python src/cli.py --llm-model gpt-4o query "..." --json   # full JSON output
```

## Retrieval Strategies

| Strategy | Description |
|---|---|
| `flat` | Pure cosine similarity (fastest) |
| `rerank` | Fetch 20, rerank by combined similarity + length signal |
| `mmr` | Maximal Marginal Relevance — balances relevance vs. diversity |

## Chunking Strategies

| Strategy | Description |
|---|---|
| `recursive` | LangChain RecursiveCharacterTextSplitter, token-aware |
| `sentence` | Sentence-boundary aware, respects sentence integrity |

## Project Structure

```
rag-pipeline/
├── src/
│   ├── ingestion.py    # document loading, cleaning, chunking
│   ├── embeddings.py   # SentenceTransformer wrapper (BGE)
│   ├── vectorstore.py  # FAISS index (flat/HNSW/IVF) + persistence
│   ├── retriever.py    # similarity search, reranking, MMR
│   ├── generator.py    # OpenAI chat completion with context window mgmt
│   ├── pipeline.py     # end-to-end orchestrator
│   └── cli.py          # argparse CLI
├── .env.example
└── requirements.txt
```

## Tech Stack

- **FAISS** — billion-scale approximate nearest-neighbor search
- **sentence-transformers / BGE** — state-of-the-art dense embeddings
- **LangChain** — document loaders & text splitters
- **OpenAI GPT-4o-mini** — answer generation
- **tiktoken** — token-accurate chunk sizing
