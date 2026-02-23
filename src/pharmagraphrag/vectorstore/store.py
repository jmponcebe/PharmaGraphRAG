"""ChromaDB vector store operations.

Manages the ChromaDB collection that stores drug-label embeddings.
Provides functions to create/reset the collection, add chunks, and
perform similarity search with optional metadata filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from pharmagraphrag.config import DATA_DIR
from pharmagraphrag.vectorstore.chunker import TextChunk
from pharmagraphrag.vectorstore.embedder import embed_texts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "drug_labels"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"

# ---------------------------------------------------------------------------
# Client management
# ---------------------------------------------------------------------------

_client: chromadb.ClientAPI | None = None


def get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client (cached).

    Args:
        persist_dir: Directory for ChromaDB storage.
            Defaults to ``data/chroma``.

    Returns:
        ChromaDB PersistentClient.
    """
    global _client

    if persist_dir is None:
        persist_dir = CHROMA_PERSIST_DIR

    if _client is None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialising ChromaDB at {}", persist_dir)
        _client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    return _client


def get_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get or create the drug-labels collection.

    Args:
        client: ChromaDB client (uses default if None).
        collection_name: Name of the collection.

    Returns:
        ChromaDB Collection instance.
    """
    if client is None:
        client = get_client()

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Collection '{}' — {} documents",
        collection_name,
        collection.count(),
    )
    return collection


def reset_collection(
    client: chromadb.ClientAPI | None = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Delete and recreate the collection (fresh start).

    Args:
        client: ChromaDB client.
        collection_name: Name of the collection.

    Returns:
        Empty ChromaDB Collection.
    """
    if client is None:
        client = get_client()

    try:
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '{}'", collection_name)
    except (ValueError, Exception) as exc:
        # ChromaDB raises NotFoundError (or ValueError in older versions)
        if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
            pass  # collection didn't exist
        else:
            raise

    return get_collection(client, collection_name)


# ---------------------------------------------------------------------------
# Add documents
# ---------------------------------------------------------------------------


def add_chunks(
    chunks: list[TextChunk],
    collection: chromadb.Collection | None = None,
    batch_size: int = 100,
) -> int:
    """Embed and add text chunks to ChromaDB.

    Args:
        chunks: List of TextChunk objects to add.
        collection: Target collection (uses default if None).
        batch_size: How many chunks to process per batch.

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    if collection is None:
        collection = get_collection()

    total_added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        texts = [c.text for c in batch]
        ids = [c.doc_id for c in batch]
        metadatas = [c.metadata for c in batch]

        # Generate embeddings
        embeddings = embed_texts(texts)

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_added += len(batch)
        logger.info(
            "  Added batch {}/{} ({} chunks)",
            i // batch_size + 1,
            (len(chunks) + batch_size - 1) // batch_size,
            len(batch),
        )

    logger.info("Total chunks in collection: {}", collection.count())
    return total_added


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    query: str,
    n_results: int = 5,
    collection: chromadb.Collection | None = None,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Perform similarity search on the vector store.

    Args:
        query: Natural language query string.
        n_results: Number of results to return.
        collection: Target collection (uses default if None).
        where: Optional metadata filter (e.g., {"drug_name": "WARFARIN"}).
        where_document: Optional document text filter.

    Returns:
        List of dicts with keys: id, text, metadata, distance.
    """
    if collection is None:
        collection = get_collection()

    # Embed the query
    query_embedding = embed_texts([query])[0]

    # Early return if collection is empty
    doc_count = collection.count()
    if doc_count == 0:
        return []

    # Build query kwargs
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, doc_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where
    if where_document is not None:
        query_kwargs["where_document"] = where_document

    results = collection.query(**query_kwargs)

    # Flatten results into a list of dicts
    output: list[dict[str, Any]] = []
    if results["ids"] and results["ids"][0]:
        for idx in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx] if results["documents"] else "",
                    "metadata": results["metadatas"][0][idx] if results["metadatas"] else {},
                    "distance": results["distances"][0][idx] if results["distances"] else None,
                }
            )

    return output


def search_by_drug(
    query: str,
    drug_name: str,
    n_results: int = 5,
    collection: chromadb.Collection | None = None,
) -> list[dict[str, Any]]:
    """Similarity search filtered to a specific drug.

    Args:
        query: Natural language query string.
        drug_name: Drug name to filter by (exact match, uppercase).
        n_results: Number of results to return.
        collection: Target collection.

    Returns:
        List of result dicts.
    """
    return search(
        query=query,
        n_results=n_results,
        collection=collection,
        where={"drug_name": drug_name.upper()},
    )


def format_vector_context(results: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Format search results as a text block for the LLM prompt.

    Args:
        results: Output from :func:`search`.
        max_chars: Maximum total characters.

    Returns:
        Formatted context string.
    """
    if not results:
        return "No relevant text context found."

    parts: list[str] = []
    total = 0

    for i, r in enumerate(results, 1):
        text = r["text"]
        meta = r.get("metadata", {})
        drug = meta.get("drug_name", "Unknown")
        section = meta.get("section", "unknown").replace("_", " ").title()
        distance = r.get("distance", "?")

        header = f"[Source {i}: {drug} — {section} (distance: {distance:.4f})]"
        block = f"{header}\n{text}\n"

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "…")
            break

        parts.append(block)
        total += len(block)

    return "\n".join(parts)
