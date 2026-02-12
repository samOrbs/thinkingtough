"""
Sports Psychology Book Expert — Chainlit Chat UI
Multi-book RAG with toggle switches per book.
Run with: chainlit run app.py
"""

import chainlit as cl
from chainlit.input_widget import Switch
from src.books import get_indexed_books, load_config, BookConfig
from src.retrieval import retrieve
from src.generation import generate_streaming, validate_response
from src.memory import create_session, save_turn, get_history, save_selected_books


@cl.on_chat_start
async def start():
    session_id = create_session()
    cl.user_session.set("session_id", session_id)

    # Discover indexed books
    indexed_books = get_indexed_books()
    cl.user_session.set("all_books", indexed_books)
    cl.user_session.set("selected_books", indexed_books)  # All on by default

    # Create toggle settings for each book
    if indexed_books:
        settings = await cl.ChatSettings(
            [
                Switch(
                    id=book.slug,
                    label=f"{book.title}",
                    initial=True,
                    description=f"by {book.author} — {book.short_description}",
                )
                for book in indexed_books
            ]
        ).send()

    # Build welcome message
    if len(indexed_books) == 0:
        book_list = "No indexed books found. Run the indexing pipeline first."
    elif len(indexed_books) == 1:
        book_list = f"**{indexed_books[0].title}** by {indexed_books[0].author}"
    else:
        book_list = "\n".join(
            f"- **{b.title}** by {b.author}" for b in indexed_books
        )

    await cl.Message(
        content=f"Welcome! I'm an expert on sports psychology and mental performance. "
                f"I can answer questions from these books:\n\n{book_list}\n\n"
                f"Use the **settings gear** to toggle which books to search. "
                f"Ask me anything!"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    all_books = cl.user_session.get("all_books")
    session_id = cl.user_session.get("session_id")

    # Filter to enabled books
    selected = [book for book in all_books if settings.get(book.slug, True)]
    cl.user_session.set("selected_books", selected)

    # Persist selection
    save_selected_books(session_id, [b.slug for b in selected])

    if selected:
        names = ", ".join(f"**{b.title}**" for b in selected)
        await cl.Message(content=f"Now searching: {names}").send()
    else:
        await cl.Message(content="No books selected. Please enable at least one book in settings.").send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    selected_books = cl.user_session.get("selected_books", [])
    history = get_history(session_id)

    if not selected_books:
        await cl.Message(content="No books selected. Please enable at least one book in the settings gear.").send()
        return

    # Save user message
    save_turn(session_id, "user", message.content)

    # Retrieve relevant context from selected books
    context, sources = retrieve(message.content, selected_books)

    # Stream response
    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    for token in generate_streaming(message.content, context, history, selected_books):
        full_response += token
        await msg.stream_token(token)

    await msg.update()

    # Validate citations against retrieved context
    validation = validate_response(full_response, sources, selected_books)
    if validation["warning"]:
        full_response += validation["warning"]
        msg.content = full_response
        await msg.update()

    # Add source references with book title
    if sources:
        multi_book = len(selected_books) > 1
        elements = []
        for source in sources:
            page = source["metadata"].get("page_number", "?")
            book_title = source["metadata"].get("book_title", "")

            if multi_book:
                name = f"{book_title}, p.{page}"
            else:
                name = f"Page {page}"

            elements.append(
                cl.Text(
                    name=name,
                    content=source["text"][:500],
                    display="side"
                )
            )
        msg.elements = elements
        await msg.update()

    # Save assistant response
    save_turn(session_id, "assistant", full_response)
