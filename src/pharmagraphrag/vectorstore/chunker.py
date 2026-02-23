"""Text chunking for DailyMed drug labels.

Splits drug label sections into overlapping chunks suitable for embedding
and vector search. Each chunk carries metadata (drug name, section, index)
so retrieved results can be traced back to the original source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from pharmagraphrag.config import DATA_RAW_DIR

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

DAILYMED_DIR = DATA_RAW_DIR / "dailymed"

# Sections ordered by relevance for the RAG use-case
SECTION_PRIORITY: list[str] = [
    "drug_interactions",
    "adverse_reactions",
    "warnings_and_cautions",
    "contraindications",
    "boxed_warning",
    "indications_and_usage",
    "dosage_and_administration",
    "clinical_pharmacology",
    "mechanism_of_action",
    "pharmacodynamics",
    "overdosage",
    "warnings",
]


@dataclass
class TextChunk:
    """A single text chunk with metadata."""

    text: str
    drug_name: str
    section: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Unique identifier for this chunk."""
        return f"{self.drug_name}__{self.section}__{self.chunk_index}"


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove section number prefixes."""
    # Remove leading section numbers like "7 DRUG INTERACTIONS"
    text = re.sub(r"^\d+(\.\d+)?\s+", "", text.strip())
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Split *text* into overlapping chunks by character count.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = _clean_text(text)

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence boundary (period, question mark, etc.)
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_question = chunk.rfind("? ")
            last_excl = chunk.rfind("! ")
            best_break = max(last_period, last_question, last_excl)
            if best_break > chunk_size // 2:
                end = start + best_break + 2  # include the period + space
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - chunk_overlap

    return chunks


def load_drug_label(filepath: Path) -> dict | None:
    """Load a single DailyMed JSON file.

    Returns:
        Parsed dictionary or None if the file can't be read.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load {}: {}", filepath.name, exc)
        return None


def chunk_drug_label(
    label: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    sections: list[str] | None = None,
) -> list[TextChunk]:
    """Chunk all sections of a drug label.

    Args:
        label: Parsed DailyMed label dictionary.
        chunk_size: Max chars per chunk.
        chunk_overlap: Overlap between chunks.
        sections: Which sections to include (default: all available).

    Returns:
        List of TextChunk objects with metadata.
    """
    drug_name: str = label.get("drug_name", "UNKNOWN")
    label_sections: dict[str, str] = label.get("sections", {})

    if sections is None:
        # Use priority order, but include any extra sections present
        ordered_keys = [s for s in SECTION_PRIORITY if s in label_sections]
        extra_keys = [s for s in label_sections if s not in SECTION_PRIORITY]
        sections = ordered_keys + extra_keys

    all_chunks: list[TextChunk] = []

    for section_name in sections:
        raw_text = label_sections.get(section_name, "")
        if not raw_text or not raw_text.strip():
            continue

        text_chunks = chunk_text(raw_text, chunk_size, chunk_overlap)

        for idx, chunk_text_str in enumerate(text_chunks):
            # Build a prefix for context
            prefix = f"Drug: {drug_name} | Section: {section_name.replace('_', ' ').title()}"
            full_text = f"{prefix}\n{chunk_text_str}"

            tc = TextChunk(
                text=full_text,
                drug_name=drug_name,
                section=section_name,
                chunk_index=idx,
                metadata={
                    "drug_name": drug_name,
                    "section": section_name,
                    "chunk_index": str(idx),
                    "generic_names": ", ".join(label.get("generic_names", [])),
                    "brand_names": ", ".join(label.get("brand_names", [])),
                    "route": ", ".join(label.get("route", [])),
                },
            )
            all_chunks.append(tc)

    return all_chunks


def chunk_all_labels(
    dailymed_dir: Path | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    sections: list[str] | None = None,
) -> list[TextChunk]:
    """Chunk every DailyMed label JSON in the given directory.

    Args:
        dailymed_dir: Path to the directory containing DailyMed JSON files.
        chunk_size: Max chars per chunk.
        chunk_overlap: Overlap between chunks.
        sections: Which sections to include.

    Returns:
        All TextChunk objects across all drugs.
    """
    if dailymed_dir is None:
        dailymed_dir = DAILYMED_DIR

    json_files = sorted(dailymed_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in {}", dailymed_dir)
        return []

    logger.info("Chunking {} drug labels from {}", len(json_files), dailymed_dir)

    all_chunks: list[TextChunk] = []
    for filepath in json_files:
        label = load_drug_label(filepath)
        if label is None:
            continue

        chunks = chunk_drug_label(
            label,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sections=sections,
        )
        all_chunks.extend(chunks)
        logger.debug("  {} → {} chunks", label.get("drug_name", filepath.stem), len(chunks))

    logger.info("Total chunks created: {}", len(all_chunks))
    return all_chunks
