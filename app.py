"""
Toughness Training Book Expert — Chainlit Chat UI
Run with: chainlit run app.py
"""

import chainlit as cl
from src.retrieval import retrieve
from src.generation import generate_streaming
from src.memory import create_session, save_turn, get_history


@cl.on_chat_start
async def start():
    session_id = create_session()
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content="Welcome! I'm an expert on **The New Toughness Training for Sports** "
                "by James E. Loehr. Ask me anything about the book — mental toughness, "
                "the Ideal Performance State, recovery techniques, training cycles, "
                "or any concept from the 232 pages."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    history = get_history(session_id)

    # Save user message
    save_turn(session_id, "user", message.content)

    # Retrieve relevant context
    context, sources = retrieve(message.content)

    # Stream response
    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    for token in generate_streaming(message.content, context, history):
        full_response += token
        await msg.stream_token(token)

    await msg.update()

    # Add source references
    if sources:
        source_pages = sorted(set(
            str(s["metadata"].get("page_number", "?")) for s in sources
        ))
        elements = []
        for source in sources:
            page = source["metadata"].get("page_number", "?")
            elements.append(
                cl.Text(
                    name=f"Page {page}",
                    content=source["text"][:500],
                    display="side"
                )
            )
        msg.elements = elements
        await msg.update()

    # Save assistant response
    save_turn(session_id, "assistant", full_response)
