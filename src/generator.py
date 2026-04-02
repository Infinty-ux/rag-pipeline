import os
from typing import List, Optional
from openai import OpenAI
from ingestion import Chunk

SYSTEM_PROMPT = (
    "You are a precise, factual assistant. Answer the user's question using ONLY "
    "the provided context snippets. If the context does not contain enough information, "
    "say so explicitly. Do not fabricate facts."
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class Generator:
    def __init__(self, model: str = DEFAULT_MODEL, max_context_tokens: int = 3000):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.max_context_tokens = max_context_tokens

    def _build_context(self, chunks: List[Chunk]) -> str:
        parts = []
        total = 0
        for i, c in enumerate(chunks):
            entry = f"[{i+1}] (source: {c.source}, page {c.page})\n{c.text}"
            total += c.token_count
            if total > self.max_context_tokens:
                break
            parts.append(entry)
        return "\n\n---\n\n".join(parts)

    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        temperature: float = 0.2,
        stream: bool = False,
    ) -> str:
        context = self._build_context(chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        )

        if stream:
            full = []
            for chunk in self.client.chat.completions.create(**kwargs, stream=True):
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full.append(delta)
            print()
            return "".join(full)
        else:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()

    def generate_with_citations(self, query: str, chunks: List[Chunk]) -> dict:
        answer = self.generate(query, chunks)
        citations = [
            {"index": i + 1, "source": c.source, "page": c.page, "snippet": c.text[:200]}
            for i, c in enumerate(chunks[:5])
        ]
        return {"answer": answer, "citations": citations}
