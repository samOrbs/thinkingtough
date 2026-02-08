"""
Phase 4: Generation Layer
System prompt, LLM generation with Gemini, streaming support.
Cost per query: ~$0.001 with Gemini 2.0 Flash.
"""

import os
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


def generate(query: str, context: str, history: list[dict] = None) -> str:
    """Generate a response using Gemini 2.0 Flash."""
    prompt = build_prompt(query, context, history)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


def generate_streaming(query: str, context: str, history: list[dict] = None):
    """Generate a streaming response using Gemini 2.0 Flash."""
    prompt = build_prompt(query, context, history)

    response = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=prompt
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text
