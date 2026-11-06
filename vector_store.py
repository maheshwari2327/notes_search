# ==============================================================================
# LAYER: Database Layer (ChromaDB Persistent Vector Store)
# Responsibility: Handles persistent vector storage, embedding generation via
# Chroma's default embedding function, and similarity search queries.
# ==============================================================================

import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

# Define the local persistence path for ChromaDB.
# Using an absolute path based on this file's directory ensures consistency
# regardless of where the script is executed from.
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "course_notes"

# Singleton reference for the persistent client and collection
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def get_collection() -> chromadb.Collection:
    """
    Initializes and returns the persistent ChromaDB collection.
    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 ONNX model)
    which runs locally on the CPU with zero external API key requirements.
    """
    global _client, _collection

    if _collection is None:
        # Initialize persistent client pointing to ./chroma_db
        _client = chromadb.PersistentClient(path=DB_DIR)

        # Chroma's default embedding function: sentence-transformers/all-MiniLM-L6-v2
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        # Get or create the vector collection configured with cosine similarity distance
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=default_ef,
            metadata={"hnsw:space": "cosine"}  # Cosine distance: 0.0 means identical, lower = more similar
        )

    return _collection


def add_note_chunk(topic: str, chunk_text: str, chunk_index: int = 0, total_chunks: int = 1, note_id: Optional[str] = None) -> str:
    """
    Stores a single text chunk into ChromaDB with metadata.

    Args:
        topic: The category or subject label (e.g., 'Redis', 'Flask').
        chunk_text: The textual content of the chunk (approx 300 chars).
        chunk_index: Index of this chunk within the parent note.
        total_chunks: Total number of chunks the parent note was split into.
        note_id: Optional explicit ID. If omitted, a unique UUID is generated.

    Returns:
        The ID string of the newly added document chunk.
    """
    collection = get_collection()

    # Generate a unique ID if not provided
    doc_id = note_id or f"{topic.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

    # Add document and metadata into the ChromaDB vector store
    collection.add(
        ids=[doc_id],
        documents=[chunk_text],
        metadatas=[{
            "topic": topic.strip(),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "char_count": len(chunk_text)
        }]
    )

    return doc_id


def add_note_chunks(topic: str, chunks: List[str]) -> List[str]:
    """
    Stores multiple chunks for a given topic in a single batch.

    Args:
        topic: The topic label.
        chunks: List of chunk text strings.

    Returns:
        List of generated chunk ID strings.
    """
    collection = get_collection()
    total_chunks = len(chunks)
    ids = []
    documents = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        doc_id = f"{topic.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}_c{idx+1}"
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "topic": topic.strip(),
            "chunk_index": idx + 1,
            "total_chunks": total_chunks,
            "char_count": len(chunk)
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return ids


def search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Performs semantic vector similarity search against stored notes.

    Args:
        query: Natural language search question or phrase.
        n_results: Maximum number of closest chunks to retrieve (default: 3).

    Returns:
        List of result dictionaries containing 'id', 'topic', 'content', and 'distance'.
    """
    collection = get_collection()
    total_stored = collection.count()

    # If the collection is empty, return an empty list immediately
    if total_stored == 0:
        return []

    # Ensure n_results does not exceed total stored items
    limit = min(n_results, total_stored)

    # Query ChromaDB collection
    query_results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    docs = query_results.get("documents", [[]])[0]
    metas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]
    ids = query_results.get("ids", [[]])[0]

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        distance_val = distances[i] if i < len(distances) else 0.0

        formatted_results.append({
            "id": ids[i] if i < len(ids) else f"result_{i}",
            "topic": meta.get("topic", "General"),
            "chunk_index": meta.get("chunk_index", 1),
            "total_chunks": meta.get("total_chunks", 1),
            "content": docs[i],
            "distance": round(float(distance_val), 4)  # Lower is closer
        })

    return formatted_results


def count() -> int:
    """
    Returns the total number of stored chunks in the vector database.
    """
    collection = get_collection()
    return collection.count()


def get_all_notes() -> List[Dict[str, Any]]:
    """
    Retrieves all stored chunks and their metadata from the collection.

    Returns:
        List of all note dictionaries containing id, topic, chunk info, and content.
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    all_data = collection.get(include=["documents", "metadatas"])
    items = []
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        items.append({
            "id": ids[i],
            "topic": meta.get("topic", "General"),
            "chunk_index": meta.get("chunk_index", 1),
            "total_chunks": meta.get("total_chunks", 1),
            "content": docs[i]
        })

    # Sort items alphabetically by topic then chunk_index
    items.sort(key=lambda x: (x["topic"].lower(), x["chunk_index"]))
    return items


def reset_db() -> bool:
    """
    Deletes all entries in the collection and re-creates an empty collection.
    Useful for testing, clearing, or re-seeding data.
    """
    global _client, _collection
    if _client is not None:
        try:
            _client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
        # Reinitialize fresh collection
        get_collection()
        return True
    return False

