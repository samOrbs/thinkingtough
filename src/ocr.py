"""
Phase 1: OCR Pipeline
Converts 232 JPG page images to clean markdown using Gemini 2.0 Flash.
Cost: ~$0.06 for all 232 pages.
"""

import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Configuration
PAGES_DIR = Path(__file__).parent.parent / "pages"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "pages_markdown"
STRUCTURE_FILE = Path(__file__).parent.parent / "data" / "book_structure.json"

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
    text = text.replace("\u2014", "—").replace("\u2013", "–")

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


def run_ocr(start_page: int = 1, end_page: int = 232):
    """Run OCR pipeline on all pages (or a range)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    structure = []
    failed = []

    print(f"Starting OCR for pages {start_page}-{end_page}...")

    for i in range(start_page, end_page + 1):
        image_path = PAGES_DIR / f"page_{i:04d}.jpg"
        output_path = OUTPUT_DIR / f"page_{i:04d}.md"

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

    # Save structure
    STRUCTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STRUCTURE_FILE, "w") as f:
        json.dump({
            "total_pages": end_page - start_page + 1,
            "pages": structure,
            "failed": failed
        }, f, indent=2)

    print(f"\nDone! {len(structure)} pages processed, {len(failed)} failed.")
    if failed:
        print(f"Failed pages: {[f['page'] for f in failed]}")

    return structure, failed


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 232
    run_ocr(start, end)
