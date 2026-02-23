"""Embedding generation using sentence-transformers.

Wraps the sentence-transformers library to produce dense vector embeddings
for text chunks. Uses the model configured in settings (default:
all-MiniLM-L6-v2, 384-dimensional embeddings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sentence_transformers import SentenceTransformer

from pharmagraphrag.config import get_settings

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Module-level cache for the model (loaded once)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def get_model(model_name: str | None = None) -> SentenceTransformer:
    """Return a cached SentenceTransformer model.

    Args:
        model_name: HuggingFace model name, or None to use config default.

    Returns:
        Loaded SentenceTransformer instance.
    """
    global _model  # noqa: PLW0603

    if model_name is None:
        model_name = get_settings().embedding_model

    if _model is None or _model.get_sentence_embedding_dimension() is None:
        logger.info("Loading embedding model: {}", model_name)
        _model = SentenceTransformer(model_name)
        dim = _model.get_sentence_embedding_dimension()
        logger.info("Model loaded — embedding dimension: {}", dim)

    return _model


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
    normalize: bool = True,
) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: Input strings to embed.
        model_name: Override the default model.
        batch_size: Texts per inference batch.
        show_progress: Show a progress bar.
        normalize: L2-normalise embeddings (recommended for cosine similarity).

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    model = get_model(model_name)

    logger.info("Embedding {} texts (batch_size={})", len(texts), batch_size)

    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
    )

    # Convert numpy array → list of lists (ChromaDB expects this)
    return embeddings.tolist()


def embed_single(
    text: str,
    model_name: str | None = None,
    normalize: bool = True,
) -> list[float]:
    """Embed a single text string.

    Convenience wrapper around :func:`embed_texts` for query-time embedding.

    Args:
        text: Input text.
        model_name: Override the default model.
        normalize: L2-normalise the embedding.

    Returns:
        Embedding vector as a list of floats.
    """
    result = embed_texts([text], model_name=model_name, normalize=normalize)
    return result[0]


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Return the embedding dimension of the configured model."""
    model = get_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    assert dim is not None
    return int(dim)
