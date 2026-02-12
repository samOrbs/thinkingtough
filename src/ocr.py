"""
OCR Pipeline — Multi-book support with auto-detection for PDFs.
Converts JPG page images or PDF pages to clean markdown using Gemini 2.0 Flash.

Usage:
    python -m src.ocr toughness-training          # JPG book
    python -m src.ocr inner-game-of-tennis         # PDF book (auto-detects text vs scanned)
    python -m src.ocr inner-game-of-tennis 1 50    # PDF, pages 1-50 only
"""

import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from src.books import BookConfig, load_config

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

OCR_PROMPT = """Transcribe this scanned book page to clean text. Follow these rules:
1. Preserve all paragraph breaks
2. Preserve headings and subheadings (mark with # markdown syntax)
3. Preserve any bullet points or numbered lists
4. Preserve italics and bold where visible
5. If the page is blank or contains only images/diagrams, describe what you see briefly
6. Do NOT add any commentary — only the text that appears on the page
7. If there is a page number, note it at the top as: <!-- Page X -->"""


def ocr_page(image_path: Path) -> str:
    """OCR a single page image using Gemini 2.0 Flash vision."""
    image_data = image_path.read_bytes()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            OCR_PROMPT,
            genai.types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
        ]
    )
    return response.text


def clean_text(raw_text: str) -> str:
    """Clean OCR output: fix common artifacts."""
    text = raw_text

    # Rejoin hyphenated words split across lines
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "\u2014").replace("\u2013", "\u2013")

    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def detect_structure(page_num: int, text: str) -> dict:
    """Detect chapter headings and section structure from page text."""
    info = {"page_number": page_num, "headings": []}

    # Look for chapter headings (common patterns)
    chapter_match = re.search(
        r"^#?\s*(Chapter\s+\d+|CHAPTER\s+\d+)", text, re.MULTILINE
    )
    if chapter_match:
        info["chapter_start"] = chapter_match.group(1)

    # Collect all markdown headings
    for match in re.finditer(r"^(#{1,3})\s+(.+)$", text, re.MULTILINE):
        info["headings"].append({
            "level": len(match.group(1)),
            "text": match.group(2).strip()
        })

    return info


def pdf_is_text_based(pdf_path: Path, sample_pages: int = 5) -> bool:
    """Auto-detect whether a PDF has extractable text or is scanned images.
    Returns True if text-based, False if scanned/image PDF.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    text_pages = 0
    check_count = min(sample_pages, len(doc))

    # Sample from the middle of the book (skip cover pages)
    start = min(5, len(doc) - check_count)
    for i in range(start, start + check_count):
        page = doc[i]
        text = page.get_text().strip()
        if len(text.split()) > 30:
            text_pages += 1

    doc.close()
    is_text = text_pages >= check_count * 0.6
    print(f"  PDF auto-detect: {text_pages}/{check_count} sampled pages have text → {'text-based' if is_text else 'scanned/image'}")
    return is_text


def pdf_extract_text(book: BookConfig) -> tuple[list, list]:
    """Extract text directly from a text-based PDF. Cost: $0."""
    import fitz  # PyMuPDF

    pdf_path = book.source_pdf_path
    output_dir = book.markdown_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    structure = []
    failed = []

    print(f"Extracting text from {len(doc)} PDF pages (text-based, $0 cost)...")

    for page_num in range(1, len(doc) + 1):
        output_path = output_dir / f"page_{page_num:04d}.md"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  SKIP page {page_num}: already processed")
            existing_text = output_path.read_text(encoding="utf-8")
            structure.append(detect_structure(page_num, existing_text))
            continue

        try:
            page = doc[page_num - 1]  # 0-indexed in PyMuPDF
            raw_text = page.get_text()
            cleaned = clean_text(raw_text)

            # Add page comment
            cleaned = f"<!-- Page {page_num} -->\n\n{cleaned}"
            output_path.write_text(cleaned, encoding="utf-8")

            page_info = detect_structure(page_num, cleaned)
            structure.append(page_info)

            word_count = len(cleaned.split())
            print(f"  OK   page {page_num}: {word_count} words")

        except Exception as e:
            print(f"  FAIL page {page_num}: {e}")
            failed.append({"page": page_num, "error": str(e)})

    doc.close()
    return structure, failed


def pdf_to_images_and_ocr(book: BookConfig, start_page: int = 1, end_page: int = None) -> tuple[list, list]:
    """For scanned PDFs: render pages to images then OCR with Gemini Vision."""
    import fitz  # PyMuPDF

    pdf_path = book.source_pdf_path
    pages_dir = book.pages_dir
    output_dir = book.markdown_dir
    pages_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    if end_page is None:
        end_page = len(doc)

    structure = []
    failed = []

    print(f"Rendering PDF pages {start_page}-{end_page} to images + OCR...")

    for page_num in range(start_page, end_page + 1):
        image_path = pages_dir / f"page_{page_num:04d}.jpg"
        output_path = output_dir / f"page_{page_num:04d}.md"

        # Render page to image if not already done
        if not image_path.exists():
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=200)
            pix.save(str(image_path))

        # OCR if not already done
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  SKIP page {page_num}: already processed")
            existing_text = output_path.read_text(encoding="utf-8")
            structure.append(detect_structure(page_num, existing_text))
            continue

        try:
            raw_text = ocr_page(image_path)
            cleaned = clean_text(raw_text)
            output_path.write_text(cleaned, encoding="utf-8")

            page_info = detect_structure(page_num, cleaned)
            structure.append(page_info)

            word_count = len(cleaned.split())
            print(f"  OK   page {page_num}: {word_count} words")
            time.sleep(0.3)

        except Exception as e:
            print(f"  FAIL page {page_num}: {e}")
            failed.append({"page": page_num, "error": str(e)})
            time.sleep(1)

    doc.close()
    return structure, failed


def run_ocr(book: BookConfig, start_page: int = 1, end_page: int = None):
    """Run OCR pipeline for a book. Auto-detects PDF type."""
    if end_page is None:
        end_page = book.total_pages

    if book.source_type == "pdf":
        pdf_path = book.source_pdf_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if pdf_is_text_based(pdf_path):
            structure, failed = pdf_extract_text(book)
        else:
            structure, failed = pdf_to_images_and_ocr(book, start_page, end_page)
    else:
        # JPG-based OCR (original path)
        structure, failed = run_ocr_jpg(book, start_page, end_page)

    # Save structure
    book.structure_path.parent.mkdir(parents=True, exist_ok=True)
    with open(book.structure_path, "w") as f:
        json.dump({
            "book": book.slug,
            "total_pages": end_page - start_page + 1,
            "pages": structure,
            "failed": failed
        }, f, indent=2)

    print(f"\nDone! {len(structure)} pages processed, {len(failed)} failed.")
    if failed:
        print(f"Failed pages: {[f['page'] for f in failed]}")

    return structure, failed


def run_ocr_jpg(book: BookConfig, start_page: int = 1, end_page: int = 232) -> tuple[list, list]:
    """Run OCR on JPG page images (original path)."""
    pages_dir = book.pages_dir
    output_dir = book.markdown_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    structure = []
    failed = []

    print(f"Starting OCR for {book.title} pages {start_page}-{end_page}...")

    for i in range(start_page, end_page + 1):
        image_path = pages_dir / f"page_{i:04d}.jpg"
        output_path = output_dir / f"page_{i:04d}.md"

        if not image_path.exists():
            print(f"  SKIP page {i}: image not found")
            continue

        # Skip if already processed
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  SKIP page {i}: already processed")
            existing_text = output_path.read_text(encoding="utf-8")
            structure.append(detect_structure(i, existing_text))
            continue

        try:
            raw_text = ocr_page(image_path)
            clean = clean_text(raw_text)
            output_path.write_text(clean, encoding="utf-8")

            page_info = detect_structure(i, clean)
            structure.append(page_info)

            word_count = len(clean.split())
            print(f"  OK   page {i}: {word_count} words")

            # Rate limiting: small delay to avoid API throttling
            time.sleep(0.3)

        except Exception as e:
            print(f"  FAIL page {i}: {e}")
            failed.append({"page": i, "error": str(e)})
            time.sleep(1)

    return structure, failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr <book-slug> [start_page] [end_page]")
        print("Example: python -m src.ocr toughness-training")
        print("         python -m src.ocr inner-game-of-tennis")
        sys.exit(1)

    slug = sys.argv[1]
    book = load_config(slug)

    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    run_ocr(book, start, end)
