# ==============================================================================
# LAYER: Frontend UI Layer (Gradio Web Application using gr.Blocks)
# Responsibility: User interface presentation, interactive tabs, form controls,
# event handlers, and formatting responses received from the Backend API layer.
# ==============================================================================

import os
import sys
from typing import List, Tuple, Dict, Any
import gradio as gr

# Ensure current directory is in Python path for local module imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import backend API layer functions
import backend
import seed_notes

# --- INITIALIZATION ---
# On application launch, automatically seed the database with the 8 full-stack notes
# if the ChromaDB vector database is currently empty.
init_seed_result = seed_notes.seed_database(force=False)
print(f"[Init] {init_seed_result['message']}")


# --- HELPER FORMATTING FUNCTIONS ---

def format_status_badge(status_code: int, status_text: str, message: str) -> str:
    """
    Renders an HTML/Markdown status banner styled with HTTP status code.
    Green for 2xx, Yellow/Orange for 404, Red for 400/500.
    """
    if 200 <= status_code < 300:
        bg_color = "#e6f4ea"
        border_color = "#34a853"
        badge_color = "#137333"
        icon = "✅"
    elif status_code == 404:
        bg_color = "#fef7e0"
        border_color = "#f9ab00"
        badge_color = "#b06000"
        icon = "⚠️"
    else:
        bg_color = "#fce8e6"
        border_color = "#ea4335"
        badge_color = "#c5221f"
        icon = "❌"

    return f"""
    <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; margin-right: 8px;">
            HTTP {status_code} {status_text}
        </span>
        <span style="color: #202124; font-size: 0.95em;">{icon} {message}</span>
    </div>
    """


def format_search_results_markdown(results: List[Dict[str, Any]]) -> str:
    """
    Formats the top semantic search results into clean, readable Markdown cards.
    Displays Rank, Topic, Similarity Distance, and Note Content.
    """
    if not results:
        return "*No results to display.*"

    md_blocks = []
    for idx, item in enumerate(results, 1):
        topic = item.get("topic", "General")
        distance = item.get("distance", 0.0)
        content = item.get("content", "")
        chunk_idx = item.get("chunk_index", 1)
        total_chunks = item.get("total_chunks", 1)

        # In cosine distance: 0.0 = identical, 1.0 = orthogonal.
        # Lower distance indicates higher semantic relevance.
        card = f"""
### Rank #{idx} &nbsp; | &nbsp; **Topic:** `{topic}`
> **Distance:** `{distance}` *(Cosine distance: lower value = higher semantic similarity)*  
> **Chunk:** `{chunk_idx} of {total_chunks}`

{content}

---
"""
        md_blocks.append(card)

    return "\n".join(md_blocks)


def format_topics_table(topics_summary: List[Dict[str, Any]]) -> List[List[Any]]:
    """Converts topic summary dictionaries into table rows for gr.Dataframe."""
    rows = []
    for item in topics_summary:
        rows.append([
            item.get("topic", ""),
            item.get("chunks_count", 0),
            item.get("sample_preview", "")
        ])
    return rows


def format_all_chunks_table(all_notes: List[Dict[str, Any]]) -> List[List[Any]]:
    """Converts note chunk dictionaries into table rows for gr.Dataframe."""
    rows = []
    for item in all_notes:
        rows.append([
            item.get("id", ""),
            item.get("topic", ""),
            f"{item.get('chunk_index', 1)}/{item.get('total_chunks', 1)}",
            item.get("content", "")
        ])
    return rows


# --- EVENT HANDLER FUNCTIONS ---

def handle_add_note(topic: str, content: str) -> Tuple[str, str, str]:
    """
    Invokes the backend API to validate, chunk (~300 chars), and persist a note.
    Returns:
        status_html, cleared_topic (if success), cleared_content (if success)
    """
    response = backend.add_note_api(topic=topic, content=content)
    status_code = response.get("status_code", 500)
    status_text = response.get("status_text", "Error")

    if status_code == 201:
        msg = response.get("message", "Note saved.")
        status_html = format_status_badge(status_code, status_text, msg)
        # Clear form inputs on success
        return status_html, "", ""
    else:
        err = response.get("error", "An error occurred.")
        status_html = format_status_badge(status_code, status_text, err)
        # Preserve user input on failure so they can fix it
        return status_html, topic, content


def handle_search(query: str, top_k: int = 3) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Invokes the backend API to query ChromaDB for top matching chunks.
    Returns:
        status_html, formatted_results_markdown, raw_results_for_rag
    """
    response = backend.search_notes_api(query=query, top_k=int(top_k))
    status_code = response.get("status_code", 500)
    status_text = response.get("status_text", "Error")

    if status_code == 200:
        results = response.get("data", [])
        status_html = format_status_badge(status_code, status_text, response.get("message", "Search completed."))
        results_md = format_search_results_markdown(results)
        return status_html, results_md, results
    else:
        err = response.get("error", "Search failed.")
        status_html = format_status_badge(status_code, status_text, err)
        return status_html, "*No matching notes found.*", []


def handle_generate_rag_answer(
    query: str,
    raw_results: List[Dict[str, Any]],
    user_api_key: str
) -> Tuple[str, str]:
    """
    STRETCH GOAL:
    Sends top retrieved chunks + user query to an LLM API to generate a synthesized answer.
    """
    if not raw_results:
        status_html = format_status_badge(
            400, "Bad Request",
            "Please run a search first to retrieve context chunks before generating an AI answer."
        )
        return status_html, ""

    response = backend.generate_rag_answer_api(
        query=query,
        retrieved_chunks=raw_results,
        user_api_key=user_api_key
    )

    status_code = response.get("status_code", 500)
    status_text = response.get("status_text", "Error")

    if status_code == 200:
        data = response.get("data", {})
        answer = data.get("answer", "")
        model = data.get("model", "LLM")
        status_html = format_status_badge(
            status_code, status_text,
            f"AI answer generated successfully using {model}."
        )
        return status_html, answer
    else:
        err = response.get("error", "Failed to generate AI answer.")
        status_html = format_status_badge(status_code, status_text, err)
        return status_html, ""


def handle_refresh_all_notes() -> Tuple[str, str, str, List[List[Any]], List[List[Any]]]:
    """
    Fetches stored notes count and table data from the backend.
    """
    response = backend.get_all_notes_api()
    status_code = response.get("status_code", 200)
    status_text = response.get("status_text", "OK")

    if status_code == 200:
        data = response.get("data", {})
        total_chunks = data.get("total_chunks", 0)
        total_topics = data.get("total_topics", 0)
        topics_table = format_topics_table(data.get("topics_summary", []))
        chunks_table = format_all_chunks_table(data.get("all_notes", []))
        status_html = format_status_badge(status_code, status_text, response.get("message", "Notes loaded."))
        return (
            status_html,
            f"## 📚 {total_chunks} Total Chunks Stored",
            f"## 🏷️ {total_topics} Unique Topics",
            topics_table,
            chunks_table
        )
    else:
        status_html = format_status_badge(status_code, status_text, response.get("error", "No notes found."))
        return (
            status_html,
            "## 📚 0 Total Chunks Stored",
            "## 🏷️ 0 Unique Topics",
            [],
            []
        )


def handle_reseed_db() -> Tuple[str, str, str, List[List[Any]], List[List[Any]]]:
    """
    Resets the database and seeds the 8 Full-Stack Layers Guide notes.
    """
    response = backend.reseed_database_api()
    status_code = response.get("status_code", 201)
    status_text = response.get("status_text", "Created")
    status_html = format_status_badge(status_code, status_text, response.get("message", "Database re-seeded."))
    
    # Reload notes overview
    _, stat_chunks, stat_topics, topics_table, chunks_table = handle_refresh_all_notes()
    return status_html, stat_chunks, stat_topics, topics_table, chunks_table


def handle_clear_db() -> Tuple[str, str, str, List[List[Any]], List[List[Any]]]:
    """
    Clears all stored notes from the database.
    """
    response = backend.clear_database_api()
    status_code = response.get("status_code", 200)
    status_text = response.get("status_text", "OK")
    status_html = format_status_badge(status_code, status_text, response.get("message", "Database cleared."))
    return (
        status_html,
        "## 📚 0 Total Chunks Stored",
        "## 🏷️ 0 Unique Topics",
        [],
        []
    )


# --- GRADIO UI CONSTRUCTION ---

# Custom CSS for styling
CUSTOM_CSS = """
.app-header { text-align: center; margin-bottom: 20px; }
.app-title { font-size: 2.2rem; font-weight: 700; color: #1a73e8; margin-bottom: 4px; }
.app-subtitle { font-size: 1.05rem; color: #5f6368; }
.status-container { margin-top: 10px; margin-bottom: 10px; }
.stat-box { background: #f8f9fa; border: 1px solid #dadce0; border-radius: 8px; padding: 12px; text-align: center; }
"""

with gr.Blocks(title="Semantic Search for Course Notes") as demo:
    # State variable to store top retrieved chunks for RAG answer generation
    retrieved_chunks_state = gr.State([])

    # Header section
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">🧠 Semantic Search for Course Notes</div>
        <div class="app-subtitle">
            Local Vector Search powered by <b>ChromaDB</b> (all-MiniLM-L6-v2 embeddings) & <b>Gradio</b>
        </div>
    </div>
    """)

    with gr.Tabs() as tabs:
        # ======================================================================
        # TAB 1: ADD NOTES
        # ======================================================================
        with gr.Tab("Add notes", id="tab_add"):
            gr.Markdown("### 📝 Add New Course Note")
            gr.Markdown(
                "Paste a paragraph of notes with a topic label. "
                "Notes longer than **300 characters** are automatically split into manageable chunks before being embedded and saved into ChromaDB."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    topic_input = gr.Textbox(
                        label="Topic Label",
                        placeholder="e.g. Flask, Redis, GraphQL, React...",
                        lines=1
                    )
                with gr.Column(scale=3):
                    content_input = gr.Textbox(
                        label="Notes Paragraph",
                        placeholder="Paste course notes here...",
                        lines=6
                    )

            with gr.Row():
                save_btn = gr.Button("💾 Save Note", variant="primary", scale=2)
                clear_inputs_btn = gr.Button("🧹 Clear Form", scale=1)

            add_status_output = gr.HTML(label="Status")

            # Quick sample loader buttons for easy testing
            gr.Markdown("#### 💡 Quick Test Examples (Click to Fill)")
            with gr.Row():
                example_btn_1 = gr.Button("Paste GraphQL Note (~350 chars - triggers 2 chunks)")
                example_btn_2 = gr.Button("Paste Kubernetes Note (~220 chars - 1 chunk)")

            example_btn_1.click(
                fn=lambda: (
                    "GraphQL",
                    "GraphQL is an open-source data query and manipulation language for APIs, created by Facebook in 2012. "
                    "Unlike traditional REST APIs that require multiple round-trips to separate endpoints, GraphQL allows clients to request exactly the data structures they need in a single round-trip, preventing over-fetching and under-fetching."
                ),
                inputs=[],
                outputs=[topic_input, content_input]
            )

            example_btn_2.click(
                fn=lambda: (
                    "Kubernetes",
                    "Kubernetes is an open-source container orchestration engine for automating deployment, scaling, and management of containerized applications. It groups application containers into logical units called pods for easy discovery."
                ),
                inputs=[],
                outputs=[topic_input, content_input]
            )

            # Wire Save Button
            save_btn.click(
                fn=handle_add_note,
                inputs=[topic_input, content_input],
                outputs=[add_status_output, topic_input, content_input]
            )

            # Wire Clear Form Button
            clear_inputs_btn.click(
                fn=lambda: ("", "", ""),
                inputs=[],
                outputs=[add_status_output, topic_input, content_input]
            )

        # ======================================================================
        # TAB 2: SEARCH
        # ======================================================================
        with gr.Tab("Search", id="tab_search"):
            gr.Markdown("### 🔍 Semantic Similarity Search")
            gr.Markdown(
                "Type a natural language question. The app converts your query into an embedding vector "
                "and retrieves the **top 3 closest chunks** based on cosine semantic distance (even when words don't match exactly)."
            )

            with gr.Row():
                search_query_input = gr.Textbox(
                    label="Search Question / Query",
                    placeholder="e.g. which database is fastest for caching?",
                    scale=4,
                    lines=1
                )
                search_btn = gr.Button("🔎 Search", variant="primary", scale=1)

            # Sample questions for quick exploration
            gr.Markdown("#### 💡 Try Sample Questions:")
            with gr.Row():
                sample_q1 = gr.Button("which database is fastest for caching?")
                sample_q2 = gr.Button("how to build user interfaces in javascript?")
                sample_q3 = gr.Button("what tool runs asynchronous background tasks?")
                sample_q4 = gr.Button("relational database with acid transactions")

            sample_q1.click(fn=lambda: "which database is fastest for caching?", inputs=[], outputs=[search_query_input])
            sample_q2.click(fn=lambda: "how to build user interfaces in javascript?", inputs=[], outputs=[search_query_input])
            sample_q3.click(fn=lambda: "what tool runs asynchronous background tasks?", inputs=[], outputs=[search_query_input])
            sample_q4.click(fn=lambda: "relational database with acid transactions", inputs=[], outputs=[search_query_input])

            search_status_output = gr.HTML(label="Search Status")
            search_results_output = gr.Markdown(label="Top Matching Chunks")

            # Wire Search Button and Enter key
            search_btn.click(
                fn=handle_search,
                inputs=[search_query_input],
                outputs=[search_status_output, search_results_output, retrieved_chunks_state]
            )
            search_query_input.submit(
                fn=handle_search,
                inputs=[search_query_input],
                outputs=[search_status_output, search_results_output, retrieved_chunks_state]
            )

            # --- STRETCH GOAL: RAG CHATBOT SECTION ---
            with gr.Accordion("✨ Stretch Goal: Generate AI Answer (RAG)", open=True):
                gr.Markdown(
                    "Send the **top 3 retrieved chunks** plus your question to an LLM API to generate a synthesized answer grounded in your course notes."
                )
                with gr.Row():
                    user_api_key_input = gr.Textbox(
                        label="LLM API Key (Optional if GEMINI_API_KEY / OPENAI_API_KEY is set in environment)",
                        placeholder="Paste your Gemini (AIza...) or OpenAI (sk-...) API key here",
                        type="password",
                        scale=3
                    )
                    generate_rag_btn = gr.Button("🤖 Generate AI Answer", variant="secondary", scale=1)

                rag_status_output = gr.HTML(label="RAG Status")
                rag_answer_output = gr.Markdown(label="AI Synthesized Answer")

                generate_rag_btn.click(
                    fn=handle_generate_rag_answer,
                    inputs=[search_query_input, retrieved_chunks_state, user_api_key_input],
                    outputs=[rag_status_output, rag_answer_output]
                )

        # ======================================================================
        # TAB 3: ALL NOTES
        # ======================================================================
        with gr.Tab("All notes", id="tab_all"):
            gr.Markdown("### 📊 Course Notes Overview & Topics")

            with gr.Row():
                stat_total_chunks = gr.Markdown("## 📚 8 Total Chunks Stored")
                stat_total_topics = gr.Markdown("## 🏷️ 8 Unique Topics")

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Notes List", variant="primary")
                reseed_btn = gr.Button("🌱 Re-seed 8 Default Notes")
                clear_btn = gr.Button("🗑️ Clear Database", variant="stop")

            all_status_output = gr.HTML(label="Database Status")

            gr.Markdown("#### 🏷️ Table of Topics")
            topics_dataframe = gr.Dataframe(
                headers=["Topic", "Chunks Count", "Sample Preview"],
                datatype=["str", "number", "str"],
                interactive=False
            )

            gr.Markdown("#### 📄 All Stored Note Chunks")
            chunks_dataframe = gr.Dataframe(
                headers=["Chunk ID", "Topic", "Chunk Index", "Content"],
                datatype=["str", "str", "str", "str"],
                interactive=False
            )

            # Wire Refresh Button
            refresh_btn.click(
                fn=handle_refresh_all_notes,
                inputs=[],
                outputs=[all_status_output, stat_total_chunks, stat_total_topics, topics_dataframe, chunks_dataframe]
            )

            # Wire Re-seed Button
            reseed_btn.click(
                fn=handle_reseed_db,
                inputs=[],
                outputs=[all_status_output, stat_total_chunks, stat_total_topics, topics_dataframe, chunks_dataframe]
            )

            # Wire Clear Button
            clear_btn.click(
                fn=handle_clear_db,
                inputs=[],
                outputs=[all_status_output, stat_total_chunks, stat_total_topics, topics_dataframe, chunks_dataframe]
            )

    # Initial load of Tab 3 data on start
    demo.load(
        fn=handle_refresh_all_notes,
        inputs=[],
        outputs=[all_status_output, stat_total_chunks, stat_total_topics, topics_dataframe, chunks_dataframe]
    )


# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    print("==================================================================")
    print("🚀 Launching Semantic Search for Course Notes Gradio App")
    print("🌐 Open URL: http://127.0.0.1:7860 in your web browser")
    print("==================================================================")
    # Launch Gradio local web server
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        share=False
    )
