"""
Retrieval Pipeline — Multi-book hybrid search.
Query expansion, hybrid search (vector + BM25), rank fusion, context assembly.
Searches across multiple books with namespaced chunk IDs.

Usage:
    python -m src.retrieval "What is mental toughness?"
    python -m src.retrieval --book inner-game-of-tennis "What is Self 1 and Self 2?"
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
from src.books import BookConfig, load_config, get_indexed_books

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Book-keyed caches (slug -> resource)
_collections: dict[str, object] = {}
_bm25_data: dict[str, dict] = {}


def get_collection(book: BookConfig):
    global _collections
    if book.slug not in _collections:
        chroma_client = chromadb.PersistentClient(path=str(book.chroma_dir))
        _collections[book.slug] = chroma_client.get_collection(book.collection_name)
    return _collections[book.slug]


def get_bm25(book: BookConfig):
    global _bm25_data
    if book.slug not in _bm25_data:
        with open(book.bm25_path, "rb") as f:
            _bm25_data[book.slug] = pickle.load(f)
    return _bm25_data[book.slug]


def embed_query(query: str) -> list[float]:
    """Embed a query using Gemini."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    return result.embeddings[0].values


def _book_titles_str(books: list[BookConfig]) -> str:
    """Build a description string of selected books for prompts."""
    if len(books) == 1:
        return f'"{books[0].title}" by {books[0].author}'
    return ", ".join(f'"{b.title}" by {b.author}' for b in books)


def classify_query(query: str, books: list[BookConfig]) -> str:
    """Classify query type for smarter retrieval routing."""
    books_desc = _book_titles_str(books)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Classify this question about sports psychology books ({books_desc}) into ONE category.
Categories: factual, conceptual, comparative, page_specific, summary, chat

Question: {query}

Return ONLY the category name, nothing else."""
    )
    category = response.text.strip().lower().replace(" ", "_")
    valid = {"factual", "conceptual", "comparative", "page_specific", "summary", "chat"}
    return category if category in valid else "factual"


def expand_query(query: str, books: list[BookConfig], n_variants: int = 2) -> list[str]:
    """Generate query variants for broader retrieval."""
    books_desc = _book_titles_str(books)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Generate {n_variants} alternative search queries for finding
information in books about sports psychology and mental performance ({books_desc}).

Original query: {query}

Return one query per line, no numbering, no explanations."""
    )
    variants = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
    return [query] + variants[:n_variants]


def hyde_query(query: str, books: list[BookConfig]) -> str:
    """HyDE: Generate a hypothetical answer, use it as search query."""
    books_desc = _book_titles_str(books)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""You are an expert on {books_desc}.
Write a brief (2-3 sentence) answer to this question as if quoting from the books:

Question: {query}

Write ONLY the hypothetical answer passage, no preamble."""
    )
    return response.text.strip()


def vector_search(queries: list[str], books: list[BookConfig], n_per_query: int = 20) -> list[dict]:
    """Search ChromaDB across multiple books with namespaced IDs."""
    all_results = []

    for book in books:
        collection = get_collection(book)
        for query in queries:
            embedding = embed_query(query)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_per_query,
                include=["documents", "metadatas", "distances"]
            )
            for i, doc_id in enumerate(results["ids"][0]):
                meta = dict(results["metadatas"][0][i])
                # Ensure book metadata is present
                meta.setdefault("book_slug", book.slug)
                meta.setdefault("book_title", book.title)
                all_results.append({
                    "id": f"{book.slug}:{doc_id}",
                    "text": results["documents"][0][i],
                    "metadata": meta,
                    "score": 1 - results["distances"][0][i],
                    "source": "vector"
                })

    return all_results


def bm25_search(queries: list[str], books: list[BookConfig], n_per_query: int = 20) -> list[dict]:
    """Search BM25 index across multiple books with namespaced IDs."""
    all_results = []

    for book in books:
        bm25_data = get_bm25(book)
        bm25 = bm25_data["bm25"]
        chunks = bm25_data["chunks"]

        for query in queries:
            tokenized = re.findall(r"\w+", query.lower())
            scores = bm25.get_scores(tokenized)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_per_query]

            for idx in top_indices:
                if scores[idx] > 0:
                    chunk = chunks[idx]
                    meta = dict(chunk["metadata"])
                    meta.setdefault("book_slug", book.slug)
                    meta.setdefault("book_title", book.title)
                    all_results.append({
                        "id": f"{book.slug}:chunk_{idx}",
                        "text": chunk["text"],
                        "metadata": meta,
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


def rerank(query: str, results: list[dict], books: list[BookConfig], top_k: int = 5) -> list[dict]:
    """Rerank results using Gemini 2.5 Flash as a cross-encoder."""
    if not results:
        return results

    candidates = results[:15]

    chunks_text = ""
    for i, r in enumerate(candidates):
        text = r["metadata"].get("original_text", r["text"])[:300]
        page = r["metadata"].get("page_number", "?")
        book_title = r["metadata"].get("book_title", "Unknown")
        label = f"{book_title}, Page {page}" if len(books) > 1 else f"Page {page}"
        chunks_text += f"\n[{i}] ({label}) {text}\n"

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
        scores_text = response.text.strip()
        match = re.search(r'\[[\d\s,\.]+\]', scores_text)
        if match:
            scores = json.loads(match.group())
        else:
            return results[:top_k]

        scored = list(zip(candidates[:len(scores)], scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, s in scored[:top_k]]

    except Exception:
        return results[:top_k]


def assemble_context(results: list[dict], books: list[BookConfig], max_tokens: int = 4000, top_k: int = 5) -> tuple[str, list[dict]]:
    """Assemble retrieved chunks into a context string with book-aware citations."""
    multi_book = len(books) > 1
    context_parts = []
    used_chunks = []
    total_tokens = 0

    for chunk in results[:top_k]:
        chunk_tokens = len(chunk["text"].split())
        if total_tokens + chunk_tokens > max_tokens:
            break

        page = chunk["metadata"].get("page_number", "?")
        book_title = chunk["metadata"].get("book_title", "")

        if multi_book:
            header = f"[{book_title}, Page {page}]"
        else:
            header = f"[Page {page}]"

        context_parts.append(f"{header}\n{chunk['text']}")
        used_chunks.append(chunk)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts), used_chunks


def retrieve(query: str, books: list[BookConfig] = None, top_k: int = 5) -> tuple[str, list[dict]]:
    """Complete retrieval pipeline: classify -> expand/HyDE -> search -> fuse -> rerank -> assemble.

    Args:
        query: The user's question
        books: List of BookConfig to search. If None, searches all indexed books.
        top_k: Number of chunks to return
    """
    if books is None:
        books = get_indexed_books()

    if not books:
        return "No indexed books found.", []

    # 1. Classify query type
    query_type = classify_query(query, books)

    # 2. Expand query + optionally use HyDE for conceptual queries
    queries = expand_query(query, books)
    if query_type in ("conceptual", "summary", "comparative"):
        hyde_answer = hyde_query(query, books)
        queries.append(hyde_answer)

    # 3. Parallel search across all selected books
    vector_results = vector_search(queries, books)
    bm25_results = bm25_search(queries, books)

    # 4. Fuse
    fused = reciprocal_rank_fusion([vector_results, bm25_results])

    # 5. Rerank top candidates
    reranked = rerank(query, fused, books, top_k=top_k)

    # 6. Assemble context
    context, used_chunks = assemble_context(reranked, books, top_k=top_k)

    return context, used_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test retrieval pipeline")
    parser.add_argument("query", nargs="?", default="What is the Ideal Performance State?")
    parser.add_argument("--book", help="Search only this book (slug)")
    args = parser.parse_args()

    if args.book:
        books = [load_config(args.book)]
    else:
        books = get_indexed_books()

    print(f"Query: {args.query}")
    print(f"Books: {[b.title for b in books]}")
    print("=" * 60)

    context, chunks = retrieve(args.query, books)

    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        page = c["metadata"].get("page_number", "?")
        book_title = c["metadata"].get("book_title", "?")
        score = c.get("rrf_score", 0)
        print(f"  {i + 1}. [{book_title}, Page {page}] (RRF: {score:.4f}): {c['text'][:100]}...")

    print(f"\n{'=' * 60}")
    print("ASSEMBLED CONTEXT:")
    print("=" * 60)
    print(context[:1000])
