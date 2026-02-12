"""
Book Registry: Configuration and discovery for multi-book RAG system.
All other modules import BookConfig from here.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent
BOOKS_DIR = PROJECT_ROOT / "books"


@dataclass
class BookConfig:
    slug: str
    title: str
    author: str
    total_pages: int
    skip_pages: dict = field(default_factory=dict)  # {"front": [1, 15], "back": [218, 232]}
    source_type: str = "jpg"  # "jpg" or "pdf"
    topics: str = ""
    short_description: str = ""
    pdf_path: str = ""  # relative to book dir, for PDFs

    @property
    def book_dir(self) -> Path:
        return BOOKS_DIR / self.slug

    @property
    def pages_dir(self) -> Path:
        return self.book_dir / "pages"

    @property
    def markdown_dir(self) -> Path:
        return self.book_dir / "pages_markdown"

    @property
    def chroma_dir(self) -> Path:
        return self.book_dir / "chroma_db"

    @property
    def bm25_path(self) -> Path:
        return self.book_dir / "bm25_index.pkl"

    @property
    def structure_path(self) -> Path:
        return self.book_dir / "book_structure.json"

    @property
    def collection_name(self) -> str:
        return self.slug.replace("-", "_")

    @property
    def skip_page_set(self) -> set[int]:
        """Build set of page numbers to skip from front/back ranges."""
        pages = set()
        if "front" in self.skip_pages:
            start, end = self.skip_pages["front"]
            pages |= set(range(start, end + 1))
        if "back" in self.skip_pages:
            start, end = self.skip_pages["back"]
            pages |= set(range(start, end + 1))
        return pages

    @property
    def source_pdf_path(self) -> Path:
        """Absolute path to source PDF (if source_type == 'pdf')."""
        return self.book_dir / self.pdf_path


def load_config(slug: str) -> BookConfig:
    """Load a BookConfig from books/{slug}/config.json."""
    config_path = BOOKS_DIR / slug / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found for book '{slug}' at {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    return BookConfig(
        slug=data["slug"],
        title=data["title"],
        author=data["author"],
        total_pages=data["total_pages"],
        skip_pages=data.get("skip_pages", {}),
        source_type=data.get("source_type", "jpg"),
        topics=data.get("topics", ""),
        short_description=data.get("short_description", ""),
        pdf_path=data.get("pdf_path", ""),
    )


def discover_books() -> list[BookConfig]:
    """Discover all books with config.json files."""
    books = []
    if not BOOKS_DIR.exists():
        return books
    for config_path in sorted(BOOKS_DIR.glob("*/config.json")):
        slug = config_path.parent.name
        try:
            books.append(load_config(slug))
        except Exception as e:
            print(f"Warning: failed to load config for '{slug}': {e}")
    return books


def get_indexed_books() -> list[BookConfig]:
    """Return only books that have been indexed (chroma_db/ exists)."""
    return [b for b in discover_books() if b.chroma_dir.exists()]
