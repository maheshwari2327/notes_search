# ==============================================================================
# LAYER: Backend API / Business Logic Layer
# Responsibility: Input validation, text chunking (~300 chars), handling business
# logic, enforcing HTTP status codes (200, 201, 400, 404), and LLM RAG integration.
# ==============================================================================

import os
import re
import json
from typing import List, Dict, Any, Optional
import requests

from vector_store import add_note_chunks, search, count, get_all_notes, reset_db
import seed_notes


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    """
    Splits long text into coherent chunks of approximately `chunk_size` characters (about 300 chars).
    Splits along word boundaries whenever possible, and breaks ultra-long tokens if needed.

    Args:
        text: The raw input notes text.
        chunk_size: Target maximum characters per chunk (default: 300).

    Returns:
        List of text chunks.
    """
    text = text.strip()
    if not text:
        return []

    # If already under or equal to chunk_size, keep as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    words = text.split(" ")
    current_chunk = ""

    for word in words:
        # If an individual word exceeds chunk_size, split it into slices
        while len(word) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(word[:chunk_size])
            word = word[chunk_size:]

        if not word:
            continue

        if not current_chunk:
            current_chunk = word
        elif len(current_chunk) + 1 + len(word) <= chunk_size:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def add_note_api(topic: str, content: str) -> Dict[str, Any]:
    """
    Backend API endpoint to validate and add notes into the vector database.
    Splits long content into chunks of ~300 characters.

    HTTP Status Codes:
        - 400 Bad Request: Missing or blank topic or content.
        - 201 Created: Note successfully validated, chunked, and saved.

    Returns:
        dict: Standardized API response containing status_code, message, and data.
    """
    # 1. Input Validation: Check for empty or whitespace-only topic
    clean_topic = (topic or "").strip()
    if not clean_topic:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": "Validation Error: Topic label is required and cannot be empty.",
            "data": None
        }

    # 2. Input Validation: Check for empty or whitespace-only content
    clean_content = (content or "").strip()
    if not clean_content:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": "Validation Error: Notes content is required and cannot be empty.",
            "data": None
        }

    # 3. Text Chunking: Break note into chunks of ~300 characters
    chunks = chunk_text(clean_content, chunk_size=300)
    if not chunks:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": "Validation Error: No valid text found after parsing.",
            "data": None
        }

    # 4. Persistence: Store chunks into ChromaDB
    try:
        created_ids = add_note_chunks(topic=clean_topic, chunks=chunks)
    except Exception as e:
        return {
            "status_code": 500,
            "status_text": "Internal Server Error",
            "error": f"Database Error: Failed to store note chunks: {str(e)}",
            "data": None
        }

    # 5. Return 201 Created on success
    return {
        "status_code": 201,
        "status_text": "Created",
        "message": (
            f"Successfully saved {len(chunks)} chunk(s) under topic '{clean_topic}'. "
            f"Total database chunks: {count()}."
        ),
        "data": {
            "topic": clean_topic,
            "chunks_count": len(chunks),
            "chunks": chunks,
            "chunk_ids": created_ids
        }
    }


def search_notes_api(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Backend API endpoint to search notes using semantic vector similarity.

    HTTP Status Codes:
        - 400 Bad Request: Query string is missing or whitespace.
        - 404 Not Found: Database is empty or no matches exist.
        - 200 OK: Search executed successfully and returned matching chunks.

    Returns:
        dict: Standardized API response containing status_code, message, and top chunks.
    """
    # 1. Input Validation: Check for empty query
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": "Validation Error: Search query cannot be empty.",
            "data": []
        }

    # 2. Database Check: Verify notes exist in the vector store
    total_stored = count()
    if total_stored == 0:
        return {
            "status_code": 404,
            "status_text": "Not Found",
            "error": "Not Found: Vector database is empty. Please add notes or click 'Seed Sample Notes'.",
            "data": []
        }

    # 3. Vector Similarity Search
    try:
        results = search(clean_query, n_results=top_k)
    except Exception as e:
        return {
            "status_code": 500,
            "status_text": "Internal Server Error",
            "error": f"Search Error: Failed to execute vector query: {str(e)}",
            "data": []
        }

    if not results:
        return {
            "status_code": 404,
            "status_text": "Not Found",
            "error": f"Not Found: No matching notes found for '{clean_query}'.",
            "data": []
        }

    # 4. Return 200 OK with top matching results
    return {
        "status_code": 200,
        "status_text": "OK",
        "message": f"Retrieved top {len(results)} relevant chunk(s) for query: '{clean_query}'.",
        "data": results
    }


def get_all_notes_api() -> Dict[str, Any]:
    """
    Backend API endpoint to retrieve all stored notes, total chunk count,
    and a summary table of topics with their chunk counts.

    HTTP Status Codes:
        - 404 Not Found: Database is completely empty.
        - 200 OK: Notes found and retrieved.

    Returns:
        dict: Standardized API response containing status_code, message, and overview data.
    """
    total_chunks = count()
    if total_chunks == 0:
        return {
            "status_code": 404,
            "status_text": "Not Found",
            "error": "Not Found: Vector database has no notes stored yet.",
            "data": {
                "total_chunks": 0,
                "topics_summary": [],
                "all_notes": []
            }
        }

    all_notes = get_all_notes()

    # Aggregate chunk counts and content previews per topic
    topics_map: Dict[str, Dict[str, Any]] = {}
    for item in all_notes:
        top = item["topic"]
        if top not in topics_map:
            topics_map[top] = {
                "topic": top,
                "chunks_count": 0,
                "sample_preview": item["content"][:120] + ("..." if len(item["content"]) > 120 else "")
            }
        topics_map[top]["chunks_count"] += 1

    topics_summary = list(topics_map.values())
    topics_summary.sort(key=lambda x: x["topic"].lower())

    return {
        "status_code": 200,
        "status_text": "OK",
        "message": f"Database contains {total_chunks} chunk(s) across {len(topics_summary)} topic(s).",
        "data": {
            "total_chunks": total_chunks,
            "total_topics": len(topics_summary),
            "topics_summary": topics_summary,
            "all_notes": all_notes
        }
    }


def reseed_database_api() -> Dict[str, Any]:
    """
    Backend API endpoint to reset and re-seed the 8 Full-Stack Layers Guide notes.

    HTTP Status Codes:
        - 201 Created: Successfully reset and seeded notes.
    """
    result = seed_notes.seed_database(force=True)
    return {
        "status_code": 201,
        "status_text": "Created",
        "message": result["message"],
        "data": {
            "seeded_count": result["seeded_count"],
            "total_chunks": result["total_count"]
        }
    }


def clear_database_api() -> Dict[str, Any]:
    """
    Backend API endpoint to delete all notes from the vector database.

    HTTP Status Codes:
        - 200 OK: Collection cleared.
    """
    reset_db()
    return {
        "status_code": 200,
        "status_text": "OK",
        "message": "All notes have been cleared from the ChromaDB vector database.",
        "data": {
            "total_chunks": 0
        }
    }


def generate_rag_answer_api(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    user_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    STRETCH GOAL:
    Sends top 3 retrieved chunks plus the user's question to an LLM API to generate
    a synthesized, grounded answer. Reads API key from environment variables
    (GEMINI_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY) or uses the provided key.

    HTTP Status Codes:
        - 400 Bad Request: Missing API key or empty chunks.
        - 200 OK: LLM generated an answer successfully.
    """
    # 1. Check for retrieved context
    if not retrieved_chunks:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": "No context chunks provided to generate an answer.",
            "data": None
        }

    # 2. Resolve API Key from user input or environment variables
    api_key = (
        (user_api_key or "").strip()
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not api_key:
        return {
            "status_code": 400,
            "status_text": "Bad Request",
            "error": (
                "Missing API Key: To generate an AI answer, please set GEMINI_API_KEY "
                "or OPENAI_API_KEY as an environment variable, or enter it in the API settings box."
            ),
            "data": None
        }

    # 3. Format Context from the retrieved chunks
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[{idx}] Topic: {c.get('topic', 'General')} (Distance: {c.get('distance', 'N/A')})\n{c.get('content', '')}"
        )
    formatted_context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert teaching assistant for a Full-Stack Web Development course. "
        "Answer the student's question accurately, concisely, and clearly using ONLY the provided course notes context. "
        "Explicitly mention the relevant technology topics in your explanation. "
        "If the answer cannot be found in the context, state that the course notes do not contain that information."
    )

    user_prompt = (
        f"Context from Course Notes:\n{formatted_context}\n\n"
        f"Student Question: {query}\n\n"
        "Your Answer:"
    )

    # 4. Call either Gemini API or OpenAI API based on key shape / config
    is_gemini = not api_key.startswith("sk-")

    try:
        if is_gemini:
            # Google Gemini REST API call
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\n{user_prompt}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 400
                }
            }
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code != 200:
                return {
                    "status_code": resp.status_code,
                    "status_text": "Gemini API Error",
                    "error": f"Gemini API returned error: {resp.text}",
                    "data": None
                }

            result_json = resp.json()
            candidates = result_json.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                answer_text = parts[0].get("text", "") if parts else "No text generated."
            else:
                answer_text = "No answer candidates returned by Gemini."

            return {
                "status_code": 200,
                "status_text": "OK",
                "message": "Answer generated successfully using Google Gemini 1.5 Flash.",
                "data": {"answer": answer_text, "model": "gemini-1.5-flash"}
            }

        else:
            # OpenAI REST API call
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 400
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code != 200:
                return {
                    "status_code": resp.status_code,
                    "status_text": "OpenAI API Error",
                    "error": f"OpenAI API returned error: {resp.text}",
                    "data": None
                }

            result_json = resp.json()
            answer_text = result_json["choices"][0]["message"]["content"]

            return {
                "status_code": 200,
                "status_text": "OK",
                "message": "Answer generated successfully using OpenAI gpt-4o-mini.",
                "data": {"answer": answer_text, "model": "gpt-4o-mini"}
            }

    except Exception as e:
        return {
            "status_code": 500,
            "status_text": "LLM Request Error",
            "error": f"Failed to connect to LLM API: {str(e)}",
            "data": None
        }
