# 🧠 Semantic Search for Course Notes

A simple, local, beginner-friendly Semantic Search web application for course notes built with **Gradio** and **ChromaDB**.

Unlike traditional keyword search (which only finds exact word matches like `Ctrl+F`), this app understands the **meaning** of your questions and notes. For instance, searching for *"which database is fastest for caching?"* automatically retrieves notes about **Redis**, even if the words "database" or "fastest" aren't explicitly typed in the same phrasing!

---

## 🏛️ Architecture & Clean Layer Separation

The application follows a strict three-layer separation of concerns, with each file marked with a comment indicating its layer:

```
notes_search/
├── app.py            # LAYER: Frontend UI Layer (Gradio Web UI with gr.Blocks)
├── backend.py        # LAYER: Backend API / Business Logic Layer (Validation, Chunking, HTTP Codes, LLM RAG)
├── vector_store.py   # LAYER: Database Layer (ChromaDB Vector Store persistence and similarity search)
├── seed_notes.py     # LAYER: Data Seeding Layer (Full-Stack Technology Notes Seeder)
├── requirements.txt  # Pinned dependencies for reproducible setup
└── README.md         # Documentation, conceptual guides, and setup instructions
```

### Layer Responsibilities
1. **Frontend Layer (`app.py`)**:
   - Built with Gradio `gr.Blocks`.
   - Organizes features into 3 clean tabs: **"Add notes"**, **"Search"**, and **"All notes"**.
   - Collects user inputs, displays search results as formatted cards, visualizes tables using `gr.Dataframe`, and renders colored HTTP status code banners.
2. **Backend API Layer (`backend.py`)**:
   - Performs strict input validation.
   - Splits long notes into coherent chunks of ~300 characters without breaking words.
   - Returns standardized responses with HTTP status codes:
     - `200 OK`: Successful search, notes retrieval, or database clear.
     - `201 Created`: Note successfully validated, chunked, and saved into ChromaDB.
     - `400 Bad Request`: Missing topic, empty query, or missing LLM API key.
     - `404 Not Found`: Vector database is empty or no matching notes found.
   - Powers the RAG stretch goal by calling LLM APIs (Gemini or OpenAI).
3. **Database Layer (`vector_store.py`)**:
   - Manages ChromaDB persistent local storage (`chromadb.PersistentClient(path="./chroma_db")`).
   - Uses ChromaDB's default embedding function (`all-MiniLM-L6-v2` ONNX model), which runs completely locally on CPU without needing any external API keys.
   - Exposes `add_note_chunk`, `add_note_chunks`, `search(query, n_results=3)`, and `count()`.
4. **Data Seeding Layer (`seed_notes.py`)**:
   - Automatically seeds 8 canonical notes from the Full-Stack Layers Guide upon first launch or when manually triggered.

---

## 💡 Conceptual Explanations (In Simple Words)

### 1. What is an Embedding?
Imagine trying to place every book in a giant library onto shelves not by alphabetical title, but by **meaning**. Books about baking cookies, making pastry, and baking bread are placed on the exact same shelf, while books about quantum physics are placed far across the room.

In computer science, an **embedding** is a way of turning a piece of text into a **list of numbers** (called a *vector*).
- An embedding model reads a sentence and outputs an array of numbers (e.g., `[0.23, -0.89, 0.45, ... 384 numbers]`).
- These numbers act as coordinates that position the text inside a high-dimensional "meaning space".
- Sentences that mean similar things will have coordinates that are very close to each other in this space.

---

### 2. Why can "fast cache" find a note about Redis even though the words differ?
If you use standard keyword search (`Ctrl + F`) for the query `"fast cache"`:
- The computer looks for the literal letters `f-a-s-t` and `c-a-c-h-e`.
- If your note says: *"Redis stores key-value pairs directly in RAM, delivering sub-millisecond read latency"*, keyword search may fail or rank it low because the word *"fast"* wasn't typed—the author wrote *"sub-millisecond"*!

**How Semantic Search Solves This:**
1. The embedding model has read billions of web pages during training. It has learned that:
   - *"fast"* means the same thing as *"high-speed"*, *"low latency"*, and *"sub-millisecond"*.
   - *"cache"* is strongly tied to *"RAM storage"*, *"in-memory store"*, and *"Redis"*.
2. When the user types `"which database is fastest for caching?"`, ChromaDB converts that query into an embedding vector.
3. ChromaDB then measures the mathematical distance (cosine distance) between the query vector and all the stored note vectors.
4. Because the *meaning* of Redis and caching are closely aligned, their vectors point in nearly the exact same direction. The distance between them is the smallest, placing **Redis at Rank #1**!

---

### 3. How does adding an LLM after the search step turn this into a RAG chatbot?
**RAG** stands for **Retrieval-Augmented Generation**. It combines the accuracy of search with the natural fluency of a Large Language Model (LLM):

1. **Step 1: Retrieve (What this app does)**
   - The user asks: *"which database is fastest for caching?"*
   - ChromaDB searches the database and retrieves the top 3 most relevant note chunks (e.g., the Redis note).
2. **Step 2: Augment (Preparing the prompt)**
   - Instead of asking an LLM to answer from its general memory (which might hallucinate or be outdated), we build a prompt that includes the retrieved notes:
     ```text
     Context from Course Notes:
     - Redis is an in-memory data store delivering sub-millisecond read latency...

     Question: which database is fastest for caching?
     Please answer using ONLY the context above.
     ```
3. **Step 3: Generate (The LLM answers)**
   - The LLM reads the context and writes a friendly, accurate answer:
     > *"Based on your course notes, Redis is the fastest database for caching because it stores key-value pairs directly in RAM with sub-millisecond latency."*

This turns a static search engine into an intelligent, truthful study assistant!

---

## 📋 Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12 installed on your laptop.
- **Operating System**: Windows, macOS, or Linux.
- **Hardware**: No GPU required; ChromaDB's default embedding model runs efficiently on CPU.
- **Internet Connection**: Needed only once during initial setup to download packages and the lightweight (~80MB) local embedding model.

---

## 🛠️ Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "notes_search"
   ```

2. **(Recommended) Create and activate a Python virtual environment:**
   - On Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - On macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the pinned requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run Each Part

### 1. Run the Main Web Application
Launch the Gradio interface:
```bash
python app.py
```
On startup:
- The app checks if the vector database is empty.
- If empty, it **automatically seeds the 8 notes** from the Full-Stack Layers Guide.
- Opens a local web server at:
  👉 **http://127.0.0.1:7860**

Open this URL in any web browser.

---

### 2. (Optional) Run the Standalone Seeder Script
You can manually view, test, or re-seed the notes from your terminal anytime:
```bash
python seed_notes.py
```
Output:
```text
--- Starting ChromaDB Seeder ---
Status : success
Message: Successfully seeded 8 full-stack technology notes into ChromaDB.
Total stored chunks: 8
```

---

### 3. (Optional) Test Backend & Database Functions
You can test the backend logic directly using Python:
```bash
python -c "import backend; print(backend.search_notes_api('which database is fastest for caching?'))"
```

---

## 🧭 Using the Application Features

### Tab 1: "Add notes"
1. Type a topic name into **Topic Label** (e.g. `GraphQL`).
2. Paste your notes into **Notes Paragraph**.
   - If the text is longer than **300 characters**, it is automatically divided into multiple ~300 character chunks so each chunk stays focused.
3. Click **"💾 Save Note"**.
4. The system validates the input, chunks the text, stores it in ChromaDB, and displays a green `HTTP 201 Created` status badge.

### Tab 2: "Search"
1. Type any natural language question into the search box:
   - Example: *"which database is fastest for caching?"*
   - Example: *"how to build user interfaces in javascript?"*
   - Example: *"what tool runs asynchronous background tasks?"*
2. Click **"🔎 Search"** (or press Enter).
3. The app displays an `HTTP 200 OK` banner and displays the **top 3 matching chunks**, each showing:
   - **Rank** (#1, #2, #3)
   - **Topic** label
   - **Cosine Distance** (e.g., `0.6170`, lower value = closer semantic match)
   - **Note Content**

### Tab 3: "All notes"
1. View overall metrics: **Total Chunks Stored** and **Unique Topics**.
2. View the **Table of Topics** summarizing each topic, its chunk count, and a preview.
3. View the **All Stored Note Chunks** table showing raw chunk IDs and content.
4. Use the control buttons:
   - **🔄 Refresh Notes List**: Reloads latest database state.
   - **🌱 Re-seed 8 Default Notes**: Restores the 8 full-stack layer notes.
   - **🗑️ Clear Database**: Empties the database to test empty-state handling.

---

## ✨ Stretch Goal: LLM RAG Answer Generation

The Search tab includes an interactive accordion: **"✨ Stretch Goal: Generate AI Answer (RAG)"**.

When you click **"🤖 Generate AI Answer"**, the application:
1. Gathers the **top 3 retrieved chunks** from the search step.
2. Sends the chunks and your question to an LLM (supports Google Gemini or OpenAI).
3. Displays a synthesized answer grounded strictly in your notes.

### Setting Up the API Key
You can provide an API key in two ways:
- **Option A (Environment Variable - Recommended)**:
  - For Google Gemini (free API keys available at [Google AI Studio](https://aistudio.google.com/)):
    ```bash
    # Windows PowerShell:
    $env:GEMINI_API_KEY="your-gemini-api-key"

    # macOS / Linux:
    export GEMINI_API_KEY="your-gemini-api-key"
    ```
  - For OpenAI:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
- **Option B (Directly in UI)**:
  - Paste your key into the password box in the Gradio UI under the Search tab.

*(Note: If no API key is provided, the semantic search still works 100% locally with zero external dependencies!)*

---

## 📦 Pinned Open-Source Libraries

All tools used are 100% free and open-source:
| Package | Pinned Version | Purpose |
| :--- | :--- | :--- |
| `gradio` | `6.31.0` | Web UI, responsive layouts, blocks, event binding |
| `chromadb` | `1.5.9` | Embedded vector database with ONNX embedding engine |
| `requests` | `2.34.2` | Lightweight HTTP client for stretch goal LLM API calls |
| `pandas` | `3.0.6` | Data structures and tabular data representation |
| `pydantic` | `2.14.1` | Data validation and schema consistency |

