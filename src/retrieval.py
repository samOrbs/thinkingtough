"""
Phase 3: Retrieval Pipeline
Query expansion, hybrid search (vector + BM25), rank fusion, context assembly.
Cost per query: ~$0.0003 (query expansion only).
"""

import os
import re
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


def expand_query(query: str, n_variants: int = 2) -> list[str]:
    """Generate query variants for broader retrieval."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""Generate {n_variants} alternative search queries for finding
information in a book about mental toughness training for sports by James Loehr.

Original query: {query}

Return one query per line, no numbering, no explanations."""
    )
    variants = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
    return [query] + variants[:n_variants]


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
    """Complete retrieval pipeline: query -> context string."""
    # 1. Expand query
    queries = expand_query(query)

    # 2. Parallel search
    vector_results = vector_search(queries)
    bm25_results = bm25_search(queries)

    # 3. Fuse
    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    # 4. Assemble context
    context, used_chunks = assemble_context(fused, top_k=top_k)

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
