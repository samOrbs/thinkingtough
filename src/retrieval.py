"""
Phase 3: Retrieval Pipeline
Query expansion, hybrid search (vector + BM25), rank fusion, context assembly.
Cost per query: ~$0.0003 (query expansion only).
"""

import os
import re
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import chromadb
from rank_bm25 import BM25Okapi
# BM25 chunks stored as plain dicts — no Chunk import needed

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
BM25_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"
COLLECTION_NAME = "toughness_training"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Lazy-loaded globals
_collection = None
_bm25_data = None


def get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = chroma_client.get_collection(COLLECTION_NAME)
    return _collection


def get_bm25():
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def embed_query(query: str) -> list[float]:
    """Embed a query using Gemini."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    return result.embeddings[0].values


def classify_query(query: str) -> str:
    """Classify query type for smarter retrieval routing."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Classify this question about a sports psychology book into ONE category.
Categories: factual, conceptual, comparative, page_specific, summary, chat

Question: {query}

Return ONLY the category name, nothing else."""
    )
    category = response.text.strip().lower().replace(" ", "_")
    valid = {"factual", "conceptual", "comparative", "page_specific", "summary", "chat"}
    return category if category in valid else "factual"


def expand_query(query: str, n_variants: int = 2) -> list[str]:
    """Generate query variants for broader retrieval."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Generate {n_variants} alternative search queries for finding
information in a book about mental toughness training for sports by James Loehr.

Original query: {query}

Return one query per line, no numbering, no explanations."""
    )
    variants = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
    return [query] + variants[:n_variants]


def hyde_query(query: str) -> str:
    """HyDE: Generate a hypothetical answer, use it as search query.
    Better for conceptual queries where the answer vocabulary differs from the question.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""You are an expert on "The New Toughness Training for Sports" by James Loehr.
Write a brief (2-3 sentence) answer to this question as if quoting from the book:

Question: {query}

Write ONLY the hypothetical answer passage, no preamble."""
    )
    return response.text.strip()


def vector_search(queries: list[str], n_per_query: int = 20) -> list[dict]:
    """Search ChromaDB with multiple queries."""
    collection = get_collection()
    all_results = []

    for query in queries:
        embedding = embed_query(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_per_query,
            include=["documents", "metadatas", "distances"]
        )
        for i, doc_id in enumerate(results["ids"][0]):
            all_results.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],
                "source": "vector"
            })

    return all_results


def bm25_search(queries: list[str], n_per_query: int = 20) -> list[dict]:
    """Search BM25 index with multiple queries."""
    bm25_data = get_bm25()
    bm25 = bm25_data["bm25"]
    chunks = bm25_data["chunks"]
    all_results = []

    for query in queries:
        tokenized = re.findall(r"\w+", query.lower())
        scores = bm25.get_scores(tokenized)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_per_query]

        for idx in top_indices:
            if scores[idx] > 0:
                chunk = chunks[idx]
                all_results.append({
                    "id": f"chunk_{idx}",
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[idx]),
                    "source": "bm25"
                })

    return all_results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge ranked result lists using RRF."""
    rrf_scores = {}
    doc_data = {}

    for results in result_lists:
        seen = {}
        for r in results:
            if r["id"] not in seen or r["score"] > seen[r["id"]]["score"]:
                seen[r["id"]] = r

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        for rank, result in enumerate(ranked):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_data[doc_id] = result

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_data[doc_id], "rrf_score": score} for doc_id, score in merged]


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank results using Gemini 2.5 Flash as a cross-encoder.
    Scores each chunk's relevance to the query, returns top-k.
    Cost: ~$0.0002 per query.
    """
    if not results:
        return results

    candidates = results[:15]  # Rerank top-15 from RRF

    # Build scoring prompt
    chunks_text = ""
    for i, r in enumerate(candidates):
        # Use original text if available (strip contextual prefix for display)
        text = r["metadata"].get("original_text", r["text"])[:300]
        page = r["metadata"].get("page_number", "?")
        chunks_text += f"\n[{i}] (Page {page}) {text}\n"

    prompt = f"""Rate the relevance of each text chunk to the query. Score 0-10 where 10 is perfectly relevant.

Query: {query}

Chunks:
{chunks_text}

Return ONLY a JSON array of scores in order, like [8, 3, 7, ...]. No explanations."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        # Parse scores
        scores_text = response.text.strip()
        # Extract JSON array from response
        match = re.search(r'\[[\d\s,\.]+\]', scores_text)
        if match:
            scores = json.loads(match.group())
        else:
            return results[:top_k]

        # Sort by reranker score
        scored = list(zip(candidates[:len(scores)], scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, s in scored[:top_k]]

    except Exception as e:
        # Fallback to RRF ordering if reranking fails
        return results[:top_k]


def assemble_context(results: list[dict], max_tokens: int = 4000, top_k: int = 5) -> tuple[str, list[dict]]:
    """Assemble retrieved chunks into a context string."""
    context_parts = []
    used_chunks = []
    total_tokens = 0

    for chunk in results[:top_k]:
        chunk_tokens = len(chunk["text"].split())
        if total_tokens + chunk_tokens > max_tokens:
            break

        page = chunk["metadata"].get("page_number", "?")
        context_parts.append(f"[Page {page}]\n{chunk['text']}")
        used_chunks.append(chunk)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts), used_chunks


def retrieve(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Complete retrieval pipeline: classify -> expand/HyDE -> search -> fuse -> rerank -> assemble."""
    # 1. Classify query type
    query_type = classify_query(query)

    # 2. Expand query + optionally use HyDE for conceptual queries
    queries = expand_query(query)
    if query_type in ("conceptual", "summary", "comparative"):
        hyde_answer = hyde_query(query)
        queries.append(hyde_answer)

    # 3. Parallel search
    vector_results = vector_search(queries)
    bm25_results = bm25_search(queries)

    # 4. Fuse
    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    # 5. Rerank top candidates
    reranked = rerank(query, fused, top_k=top_k)

    # 6. Assemble context
    context, used_chunks = assemble_context(reranked, top_k=top_k)

    return context, used_chunks


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the Ideal Performance State?"

    print(f"Query: {query}")
    print("=" * 60)

    context, chunks = retrieve(query)

    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        page = c["metadata"].get("page_number", "?")
        score = c.get("rrf_score", 0)
        print(f"  {i + 1}. Page {page} (RRF: {score:.4f}): {c['text'][:100]}...")

    print(f"\n{'=' * 60}")
    print("ASSEMBLED CONTEXT:")
    print("=" * 60)
    print(context[:1000])
