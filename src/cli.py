import argparse
import json
import sys
from pathlib import Path

from pipeline import RAGPipeline


def cmd_index(args, pipeline: RAGPipeline):
    for source in args.sources:
        print(f"Indexing {source}...")
        pipeline.index(Path(source), strategy=args.strategy)
    pipeline.save_index(Path(args.index_dir))


def cmd_query(args, pipeline: RAGPipeline):
    pipeline.load_index(Path(args.index_dir))
    result = pipeline.query(args.question, stream=args.stream)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for c in result.get("citations", []):
            print(f"  [{c['index']}] {c['source']} (page {c['page']})")
            print(f"      {c['snippet'][:120]}...")
        print("=" * 60)


def cmd_interactive(args, pipeline: RAGPipeline):
    pipeline.load_index(Path(args.index_dir))
    print("RAG Pipeline — interactive mode. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        result = pipeline.query(question, stream=True)
        print(f"\nSources: {', '.join(c['source'] for c in result['citations'][:3])}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--index-dir", default="index", help="FAISS index directory")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval", choices=["flat", "rerank", "mmr"], default="rerank")

    sub = parser.add_subparsers(dest="command")

    idx_p = sub.add_parser("index", help="Ingest and index documents")
    idx_p.add_argument("sources", nargs="+", help="Files or directories to index")
    idx_p.add_argument("--strategy", choices=["recursive", "sentence"], default="recursive")

    q_p = sub.add_parser("query", help="Ask a question")
    q_p.add_argument("question")
    q_p.add_argument("--stream", action="store_true")
    q_p.add_argument("--json", action="store_true")

    sub.add_parser("chat", help="Interactive Q&A session")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = RAGPipeline(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        retrieval_strategy=args.retrieval,
    )

    if args.command == "index":
        cmd_index(args, pipeline)
    elif args.command == "query":
        cmd_query(args, pipeline)
    elif args.command == "chat":
        cmd_interactive(args, pipeline)


if __name__ == "__main__":
    main()
