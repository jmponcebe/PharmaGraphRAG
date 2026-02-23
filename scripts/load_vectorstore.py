"""Load DailyMed drug labels into ChromaDB vector store.

Usage:
    uv run python scripts/load_vectorstore.py
"""

from __future__ import annotations

from loguru import logger

from pharmagraphrag.vectorstore.chunker import chunk_all_labels
from pharmagraphrag.vectorstore.store import add_chunks, reset_collection


def main() -> None:
    """Chunk all DailyMed labels and load into ChromaDB."""
    logger.info("=== Loading DailyMed labels into ChromaDB ===")

    # 1. Chunk all drug labels
    chunks = chunk_all_labels(chunk_size=1000, chunk_overlap=200)
    logger.info("Created {} chunks from DailyMed labels", len(chunks))

    if not chunks:
        logger.error("No chunks created — check that data/raw/dailymed/ has JSON files")
        return

    # 2. Reset and populate collection
    collection = reset_collection()
    added = add_chunks(chunks, collection=collection, batch_size=50)

    logger.info("=== Done! Added {} chunks to ChromaDB ===", added)
    logger.info("Collection now has {} documents", collection.count())

    # 3. Quick stats
    drug_names = {c.drug_name for c in chunks}
    sections = {c.section for c in chunks}
    logger.info("Drugs: {}", len(drug_names))
    logger.info("Sections: {}", sections)


if __name__ == "__main__":
    main()
