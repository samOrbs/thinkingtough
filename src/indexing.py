"""
Phase 2: Indexing Pipeline
Chunks text, generates embeddings, stores in ChromaDB + BM25.
Cost: ~$0.001 for embedding all chunks.
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

load_dotenv()

# Configuration
MARKDOWN_DIR = Path(__file__).parent.parent / "data" / "pages_markdown"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
BM25_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"
STRUCTURE_PATH = Path(__file__).parent.parent / "data" / "book_structure.json"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

COLLECTION_NAME = "toughness_training"


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def load_pages() -> list[tuple[int, str]]:
    """Load all markdown pages, return list of (page_number, text)."""
    pages = []
    for md_file in sorted(MARKDOWN_DIR.glob("page_*.md")):
        page_num = int(md_file.stem.split("_")[1])
        text = md_file.read_text(encoding="utf-8")
        if text.strip():
            pages.append((page_num, text))
    print(f"Loaded {len(pages)} pages")
    return pages


def semantic_chunk(
    pages: list[tuple[int, str]],
    max_tokens: int = 512,
    overlap_tokens: int = 100
) -> list[Chunk]:
    """Split pages into chunks at paragraph boundaries."""
    chunks = []

    for page_num, text in pages:
        # Remove page comment header
        text = re.sub(r"<!--.*?-->", "", text).strip()

        paragraphs = re.split(r"\n\n+", text)
        current_paras = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = len(para.split())

            if current_tokens + para_tokens > max_tokens and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "page_number": page_num,
                        "chunk_index": len(chunks),
                        "word_count": current_tokens
                    }
                ))

                # Overlap: keep last paragraph(s)
                overlap_paras = []
                overlap_count = 0
                for p in reversed(current_paras):
                    p_tokens = len(p.split())
                    if overlap_count + p_tokens > overlap_tokens:
                        break
                    overlap_paras.insert(0, p)
                    overlap_count += p_tokens

                current_paras = overlap_paras
                current_tokens = overlap_count

            current_paras.append(para)
            current_tokens += para_tokens

        # Emit final chunk for this page
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            # Merge tiny trailing chunks with previous
            if len(chunks) > 0 and current_tokens < 50 and chunks[-1].metadata["page_number"] == page_num:
                chunks[-1].text += "\n\n" + chunk_text
                chunks[-1].metadata["word_count"] += current_tokens
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        "page_number": page_num,
                        "chunk_index": len(chunks),
                        "word_count": current_tokens
                    }
                ))

    print(f"Created {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks: list[Chunk], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings using Gemini text-embedding-004."""
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


def store_in_chromadb(chunks: list[Chunk], embeddings: list[list[float]]):
    """Store chunks and embeddings in ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if present
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks]
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")


def build_bm25_index(chunks: list[Chunk]):
    """Build and save BM25 keyword index."""
    tokenized = [re.findall(r"\w+", c.text.lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Built BM25 index over {len(chunks)} chunks, saved to {BM25_PATH}")


def test_retrieval(query: str = "What is mental toughness?"):
    """Quick test that retrieval works."""
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)

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

    print(f"\nTest query: \"{query}\"")
    print("-" * 50)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\nResult {i + 1} (distance: {dist:.4f}, page {meta['page_number']}):")
        print(f"  {doc[:200]}...")


def run_indexing():
    """Run the full indexing pipeline."""
    print("=" * 60)
    print("PHASE 2: INDEXING PIPELINE")
    print("=" * 60)

    # 1. Load pages
    pages = load_pages()

    # 2. Chunk
    chunks = semantic_chunk(pages)

    # 3. Embed
    print("\nGenerating embeddings...")
    embeddings = embed_chunks(chunks)

    # 4. Store in ChromaDB
    print("\nStoring in ChromaDB...")
    store_in_chromadb(chunks, embeddings)

    # 5. Build BM25 index
    print("\nBuilding BM25 index...")
    build_bm25_index(chunks)

    # 6. Test
    test_retrieval("What is mental toughness?")
    test_retrieval("How does recovery work?")

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_indexing()
