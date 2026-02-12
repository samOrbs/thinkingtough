"""
Generation Layer — Dynamic system prompt for single or multi-book mode.
System prompt, LLM generation with Gemini, streaming support, response validation.
Cost per query: ~$0.001 with Gemini 2.5 Flash.
"""

import os
import re
from dotenv import load_dotenv
from google import genai
from src.books import BookConfig

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def build_system_prompt(books: list[BookConfig]) -> str:
    """Build a dynamic system prompt based on selected books."""
    if len(books) == 1:
        book = books[0]
        book_desc = f'the book "{book.title}" by {book.author}'
        citation_rule = "ALWAYS cite specific pages when referencing the book: [Page X]"
    else:
        book_list = "\n".join(f'- "{b.title}" by {b.author}' for b in books)
        book_desc = f"the following books:\n{book_list}"
        citation_rule = "ALWAYS cite with book title and page: [Book Title, Page X]"

    return f"""You are a deep expert on {book_desc}. You have comprehensive knowledge of the content and can discuss any concept, technique, or idea in detail.

## Your Capabilities
- Answer any question about the books' content with specific page references
- Explain concepts in depth, connecting ideas across chapters
- Compare and contrast different ideas within and across books
- Help the user understand training techniques and mental performance principles

## Rules
1. {citation_rule}
2. If the retrieved context doesn't contain the answer, say so honestly — do not fabricate information
3. Distinguish between what the books say and your own analysis
4. When the user asks about something NOT in the books, clearly state that
5. Use the conversation history to maintain context and avoid repeating yourself
6. Be conversational but precise — match the depth of your answer to the question
7. When quoting, use quotation marks and cite the page"""


def build_prompt(query: str, context: str, history: list[dict] = None, books: list[BookConfig] = None) -> str:
    """Build the full prompt with context and history."""
    if books is None:
        books = []

    system_prompt = build_system_prompt(books) if books else build_system_prompt([])

    parts = [system_prompt]

    if history:
        parts.append("\n## Recent Conversation")
        for turn in history[-10:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['content'][:500]}")

    source_label = "Retrieved Context from the Books" if len(books) > 1 else "Retrieved Context from the Book"
    parts.append(f"\n## {source_label}\n{context}")
    parts.append(f"\n## Current Question\n{query}")

    return "\n\n".join(parts)


def validate_response(response_text: str, used_chunks: list[dict], books: list[BookConfig] = None) -> dict:
    """Validate that citations in the response actually exist in retrieved context.
    Supports both [Page X] and [Book Title, Page X] formats.
    """
    multi_book = books and len(books) > 1

    # Extract all citations from response
    cited_pages = set()
    # Match [Page X] format
    for match in re.finditer(r'\[Page\s+(\d+)\]', response_text):
        cited_pages.add(int(match.group(1)))
    # Match [Book Title, Page X] format
    for match in re.finditer(r'\[[^]]+,\s*Page\s+(\d+)\]', response_text):
        cited_pages.add(int(match.group(1)))

    if not cited_pages:
        return {"valid_pages": set(), "hallucinated_pages": set(), "warning": None}

    # Build set of pages actually present in retrieved context
    context_pages = set()
    for chunk in used_chunks:
        meta = chunk.get("metadata", {})
        if "page_number" in meta:
            context_pages.add(int(meta["page_number"]))
        if "page_end" in meta:
            start = int(meta["page_number"])
            end = int(meta["page_end"])
            context_pages.update(range(start, end + 1))
        if "pages" in meta and isinstance(meta["pages"], str):
            for p in meta["pages"].split(","):
                p = p.strip()
                if p.isdigit():
                    context_pages.add(int(p))

    valid = cited_pages & context_pages
    hallucinated = cited_pages - context_pages

    warning = None
    if hallucinated:
        pages_str = ", ".join(str(p) for p in sorted(hallucinated))
        warning = (
            f"\n\n> **Note:** Page(s) {pages_str} cited above were not in the retrieved "
            f"context — these references may be approximate."
        )

    return {
        "valid_pages": valid,
        "hallucinated_pages": hallucinated,
        "warning": warning
    }


def generate(query: str, context: str, history: list[dict] = None, books: list[BookConfig] = None) -> str:
    """Generate a response using Gemini 2.5 Flash."""
    prompt = build_prompt(query, context, history, books)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def generate_streaming(query: str, context: str, history: list[dict] = None, books: list[BookConfig] = None):
    """Generate a streaming response using Gemini 2.5 Flash."""
    prompt = build_prompt(query, context, history, books)

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text
