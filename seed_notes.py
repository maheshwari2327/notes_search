# ==============================================================================
# LAYER: Data Seeding Layer (Full-Stack Technology Notes Seeder)
# Responsibility: Provides 8 canonical course notes covering Full-Stack Layers
# and populates ChromaDB on the first application launch or on manual command.
# ==============================================================================

import sys
import os

# Ensure the parent directory is in sys.path so modules can be imported directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from vector_store import count, add_note_chunks, reset_db

# 8 concise notes covering the modern Full-Stack Architecture Layers:
# 1. Frontend UI: React
# 2. Lightweight Web Framework: Flask
# 3. Modern Async REST APIs: FastAPI
# 4. In-Memory Caching & Performance: Redis
# 5. Persistent Relational Storage: PostgreSQL
# 6. Asynchronous Background Tasks: Celery
# 7. Reverse Proxy & Web Server: Nginx
# 8. Containerization & Deployment: Docker
SAMPLE_NOTES = [
    {
        "topic": "React",
        "content": (
            "React is a declarative, component-based frontend JavaScript library for building "
            "interactive user interfaces. It uses a virtual DOM to efficiently update and "
            "re-render UI components when application state changes."
        )
    },
    {
        "topic": "Flask",
        "content": (
            "Flask is a lightweight Python WSGI web micro-framework for the backend API layer. "
            "It provides flexible routing, request handling, and template rendering without "
            "imposing rigid database or authentication choices."
        )
    },
    {
        "topic": "FastAPI",
        "content": (
            "FastAPI is a modern, high-performance web framework for building REST APIs with Python 3.8+. "
            "It provides native async/await coroutines, automatic data validation using Pydantic, "
            "and interactive OpenAPI Swagger documentation out of the box."
        )
    },
    {
        "topic": "Redis",
        "content": (
            "Redis is an open-source, in-memory data store used as a high-speed cache and message broker. "
            "Because it stores all key-value pairs directly in RAM rather than disk, it delivers "
            "sub-millisecond read and write latency for rapid data retrieval."
        )
    },
    {
        "topic": "PostgreSQL",
        "content": (
            "PostgreSQL is an advanced, enterprise-grade open-source relational database management system. "
            "It guarantees ACID transactions, handles complex SQL queries, and natively supports "
            "JSON document storage, indexing, and foreign key relations."
        )
    },
    {
        "topic": "Celery",
        "content": (
            "Celery is an asynchronous distributed task queue and background job scheduler for Python apps. "
            "It offloads long-running workloads, batch processing, and transactional emails outside the "
            "synchronous web request-response lifecycle."
        )
    },
    {
        "topic": "Nginx",
        "content": (
            "Nginx is a high-performance HTTP web server and reverse proxy. It sits in front of backend "
            "application servers to manage load balancing, SSL/TLS certificate termination, "
            "HTTP caching, and fast static file delivery with an event-driven architecture."
        )
    },
    {
        "topic": "Docker",
        "content": (
            "Docker is a containerization platform that packages software code, system runtimes, and "
            "libraries into standardized portable containers. This guarantees reproducible execution "
            "across local developer laptops, staging servers, and production clouds."
        )
    }
]


def seed_database(force: bool = False) -> dict:
    """
    Seeds the ChromaDB vector database with the 8 full-stack layer notes.

    Args:
        force: If True, clears existing notes before seeding.
               If False, only seeds if the database currently has 0 chunks.

    Returns:
        dict: Seeding status and count information.
    """
    current_count = count()

    # If already seeded and force is False, do not duplicate
    if current_count > 0 and not force:
        return {
            "status": "skipped",
            "message": f"Database already contains {current_count} chunk(s). Skipping seed.",
            "seeded_count": 0,
            "total_count": current_count
        }

    # If force is requested, clear existing collection first
    if force and current_count > 0:
        reset_db()

    # Add each note into ChromaDB
    total_added = 0
    for note in SAMPLE_NOTES:
        # Each sample note is concise (~200 chars), so it fits neatly in one chunk
        add_note_chunks(topic=note["topic"], chunks=[note["content"]])
        total_added += 1

    return {
        "status": "success",
        "message": f"Successfully seeded {total_added} full-stack technology notes into ChromaDB.",
        "seeded_count": total_added,
        "total_count": count()
    }


if __name__ == "__main__":
    print("--- Starting ChromaDB Seeder ---")
    result = seed_database(force=True)
    print(f"Status : {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Total stored chunks: {result['total_count']}")

