"""
Phase 4: Generation Layer
System prompt, LLM generation with Gemini, streaming support, response validation.
Cost per query: ~$0.001 with Gemini 2.5 Flash.
"""

import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a deep expert on the book "The New Toughness Training for Sports" by James E. Loehr. You have comprehensive knowledge of the entire book and can discuss any concept, technique, or idea from it in detail.

## Your Capabilities
- Answer any question about the book's content with specific page references
- Explain concepts in depth, connecting ideas across chapters
- Compare and contrast different ideas within the book
- Help the user understand training techniques and mental toughness principles

## Rules
1. ALWAYS cite specific pages when referencing the book: [Page X]
2. If the retrieved context doesn't contain the answer, say so honestly — do not fabricate information about the book
3. Distinguish between what the book says and your own analysis
4. When the user asks about something NOT in the book, clearly state that
5. Use the conversation history to maintain context and avoid repeating yourself
6. Be conversational but precise — match the depth of your answer to the question
7. When quoting the book, use quotation marks and cite the page"""


def build_prompt(query: str, context: str, history: list[dict] = None) -> str:
    """Build the full prompt with context and history."""
    parts = [SYSTEM_PROMPT]

    if history:
        parts.append("\n## Recent Conversation")
        for turn in history[-10:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            parts.append(f"{role}: {turn['content'][:500]}")

    parts.append(f"\n## Retrieved Context from the Book\n{context}")
    parts.append(f"\n## Current Question\n{query}")

    return "\n\n".join(parts)


def validate_response(response_text: str, used_chunks: list[dict]) -> dict:
    """Validate that [Page X] citations in the response actually exist in retrieved context.
    Returns dict with 'valid_pages', 'hallucinated_pages', and optional 'warning'.
    """
    # Extract all [Page X] citations from response
    cited_pages = set()
    for match in re.finditer(r'\[Page\s+(\d+)\]', response_text):
        cited_pages.add(int(match.group(1)))

    if not cited_pages:
        return {"valid_pages": set(), "hallucinated_pages": set(), "warning": None}

    # Build set of pages actually present in retrieved context
    context_pages = set()
    for chunk in used_chunks:
        meta = chunk.get("metadata", {})
        # Single page
        if "page_number" in meta:
            context_pages.add(int(meta["page_number"]))
        # Page range end
        if "page_end" in meta:
            start = int(meta["page_number"])
            end = int(meta["page_end"])
            context_pages.update(range(start, end + 1))
        # Comma-separated pages string (from ChromaDB)
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


def generate(query: str, context: str, history: list[dict] = None) -> str:
    """Generate a response using Gemini 2.5 Flash."""
    prompt = build_prompt(query, context, history)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def generate_streaming(query: str, context: str, history: list[dict] = None):
    """Generate a streaming response using Gemini 2.5 Flash."""
    prompt = build_prompt(query, context, history)

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text
