"""
Indexing Pipeline — Per-book chunking, embedding, and storage.
Chunks text, generates embeddings, stores in ChromaDB + BM25.

Usage:
    python -m src.indexing --book toughness-training
    python -m src.indexing --book inner-game-of-tennis
"""

import os
import re
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from google import genai
import chromadb
from rank_bm25 import BM25Okapi
from src.books import BookConfig, load_config

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def load_pages(book: BookConfig) -> list[tuple[int, str]]:
    """Load all markdown pages for a book, filtering front/back matter."""
    skip_pages = book.skip_page_set
    markdown_dir = book.markdown_dir

    if not markdown_dir.exists():
        raise FileNotFoundError(f"No markdown directory found: {markdown_dir}")

    pages = []
    for md_file in sorted(markdown_dir.glob("page_*.md")):
        page_num = int(md_file.stem.split("_")[1])
        if page_num in skip_pages:
            continue
        text = md_file.read_text(encoding="utf-8")
        # Remove HTML comments (page markers)
        text = re.sub(r"<!--.*?-->", "", text).strip()
        if text.strip() and len(text.split()) > 20:
            pages.append((page_num, text))
    print(f"Loaded {len(pages)} content pages for '{book.title}' (skipped front/back matter)")
    return pages


def semantic_chunk(
    pages: list[tuple[int, str]],
    book: BookConfig,
    max_tokens: int = 400,
    overlap_tokens: int = 80
) -> list[Chunk]:
    """Split pages into chunks at paragraph boundaries.
    Works ACROSS pages so chunks aren't limited to single pages.
    Tracks page ranges for citation accuracy.
    """
    # Build a flat list of (paragraph_text, page_number)
    all_paras = []
    for page_num, text in pages:
        paragraphs = re.split(r"\n\n+", text)
        for para in paragraphs:
            para = para.strip()
            if para and len(para.split()) > 3:
                all_paras.append((para, page_num))

    # Deduplicate consecutive identical paragraphs (from duplicate OCR pages)
    deduped = []
    for para, page in all_paras:
        if not deduped or para != deduped[-1][0]:
            deduped.append((para, page))
    all_paras = deduped
    print(f"  {len(all_paras)} paragraphs after deduplication")

    # Chunk across page boundaries
    chunks = []
    current_paras = []
    current_pages = set()
    current_tokens = 0

    for para, page_num in all_paras:
        para_tokens = len(para.split())

        if current_tokens + para_tokens > max_tokens and current_paras:
            chunk_text = "\n\n".join(p for p, _ in current_paras)
            page_list = sorted(current_pages)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "page_number": page_list[0],
                    "page_end": page_list[-1],
                    "pages": page_list,
                    "chunk_index": len(chunks),
                    "word_count": current_tokens,
                    "book_slug": book.slug,
                    "book_title": book.title,
                }
            ))

            # Overlap: keep last paragraph(s)
            overlap_paras = []
            overlap_count = 0
            for p, pg in reversed(current_paras):
                p_tokens = len(p.split())
                if overlap_count + p_tokens > overlap_tokens:
                    break
                overlap_paras.insert(0, (p, pg))
                overlap_count += p_tokens

            current_paras = overlap_paras
            current_pages = {pg for _, pg in current_paras}
            current_tokens = overlap_count

        current_paras.append((para, page_num))
        current_pages.add(page_num)
        current_tokens += para_tokens

    # Final chunk
    if current_paras:
        chunk_text = "\n\n".join(p for p, _ in current_paras)
        page_list = sorted(current_pages)
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                "page_number": page_list[0],
                "page_end": page_list[-1],
                "pages": page_list,
                "chunk_index": len(chunks),
                "word_count": current_tokens,
                "book_slug": book.slug,
                "book_title": book.title,
            }
        ))

    print(f"Created {len(chunks)} chunks (cross-page, deduplicated)")
    return chunks


def contextual_enrich(chunks: list[Chunk], pages: list[tuple[int, str]], book: BookConfig, batch_size: int = 20) -> list[Chunk]:
    """Anthropic Contextual Enrichment: prepend LLM-generated context to each chunk.
    This situates each chunk within the broader document, improving retrieval by ~67%.
    """
    # Build a page lookup for surrounding context
    page_texts = {num: text for num, text in pages}

    enriched = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            page_num = chunk.metadata["page_number"]
            # Get surrounding page text for context (current page + neighbors)
            surrounding = ""
            for p in range(max(page_num - 1, min(page_texts.keys())), min(page_num + 2, max(page_texts.keys()) + 1)):
                if p in page_texts:
                    surrounding += page_texts[p][:500] + "\n"

            prompt = f"""<document>
{surrounding[:1500]}
</document>

Here is a chunk from that document:
<chunk>
{chunk.text[:800]}
</chunk>

Give a short (1-2 sentence) context that situates this chunk within the book "{book.title}" by {book.author}. Focus on what topic/concept this chunk discusses and where it fits in the book's structure. Return ONLY the context sentence, nothing else."""

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                context_prefix = response.text.strip()
            except Exception as e:
                print(f"  Enrichment failed for chunk {chunk.metadata['chunk_index']}: {e}")
                context_prefix = ""

            # Prepend context to chunk text
            if context_prefix:
                enriched_text = f"{context_prefix}\n\n{chunk.text}"
            else:
                enriched_text = chunk.text

            enriched.append(Chunk(
                text=enriched_text,
                metadata={**chunk.metadata, "original_text": chunk.text}
            ))

        print(f"  Enriched batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    print(f"Enriched {len(enriched)} chunks with contextual prefixes")
    return enriched


def embed_chunks(chunks: list[Chunk], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings using gemini-embedding-001."""
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_texts = [c.text for c in chunks[i:i + batch_size]]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch_texts
        )
        all_embeddings.extend([e.values for e in result.embeddings])
        print(f"  Embedded batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    print(f"Generated {len(all_embeddings)} embeddings ({len(all_embeddings[0])} dims)")
    return all_embeddings


def store_in_chromadb(chunks: list[Chunk], embeddings: list[list[float]], book: BookConfig):
    """Store chunks and embeddings in per-book ChromaDB."""
    chroma_dir = book.chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if present
    try:
        chroma_client.delete_collection(book.collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=book.collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # ChromaDB metadata must be str/int/float/bool — convert page lists to strings
    chroma_metadatas = []
    for c in chunks:
        meta = dict(c.metadata)
        if "pages" in meta:
            meta["pages"] = ",".join(str(p) for p in meta["pages"])
        chroma_metadatas.append(meta)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=chroma_metadatas
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {chroma_dir}")


def build_bm25_index(chunks: list[Chunk], book: BookConfig):
    """Build and save BM25 keyword index.
    Stores chunks as plain dicts to avoid pickle module-path issues.
    """
    tokenized = [re.findall(r"\w+", c.text.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Convert Chunk objects to plain dicts for pickle compatibility
    chunk_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]

    bm25_path = book.bm25_path
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_dicts}, f)

    print(f"Built BM25 index over {len(chunks)} chunks, saved to {bm25_path}")


def test_retrieval(book: BookConfig, query: str = "What is mental toughness?"):
    """Quick test that retrieval works for a specific book."""
    chroma_client = chromadb.PersistentClient(path=str(book.chroma_dir))
    collection = chroma_client.get_collection(book.collection_name)

    # Embed query
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )
    query_embedding = result.embeddings[0].values

    # Vector search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nTest query: \"{query}\" (book: {book.title})")
    print("-" * 50)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\nResult {i + 1} (distance: {dist:.4f}, page {meta['page_number']}):")
        print(f"  {doc[:200]}...")


def run_indexing(book: BookConfig):
    """Run the full indexing pipeline for a book."""
    print("=" * 60)
    print(f"INDEXING: {book.title}")
    print("=" * 60)

    # 1. Load pages
    pages = load_pages(book)

    # 2. Chunk
    chunks = semantic_chunk(pages, book)

    # 3. Contextual Enrichment
    print("\nContextual enrichment (Anthropic method)...")
    chunks = contextual_enrich(chunks, pages, book)

    # 4. Embed enriched chunks
    print("\nGenerating embeddings...")
    embeddings = embed_chunks(chunks)

    # 5. Store in ChromaDB
    print("\nStoring in ChromaDB...")
    store_in_chromadb(chunks, embeddings, book)

    # 6. Build BM25 index
    print("\nBuilding BM25 index...")
    build_bm25_index(chunks, book)

    # 7. Test
    test_retrieval(book)

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index a book for RAG retrieval")
    parser.add_argument("--book", required=True, help="Book slug (e.g., toughness-training)")
    args = parser.parse_args()

    book = load_config(args.book)
    run_indexing(book)
