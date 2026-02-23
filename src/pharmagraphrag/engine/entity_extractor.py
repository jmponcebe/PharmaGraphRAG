"""Entity extraction from user questions.

Identifies drug names and adverse event terms in a natural-language
question using:
  1. Exact substring matching against known drug names.
  2. Fuzzy matching (rapidfuzz) for misspellings / partial names.
  3. Optional adverse-event extraction via substring matching.

The known-drug catalogue is loaded lazily — first from a DailyMed summary
file on disk, then enriched from Neo4j if available.

Usage:
    from pharmagraphrag.engine.entity_extractor import extract_entities
    result = extract_entities("What are the side effects of ibuprofen?")
    # result.drugs == ["IBUPROFEN"]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger
from rapidfuzz import fuzz, process

from pharmagraphrag.config import DATA_RAW_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum score (0-100) for fuzzy matching to accept a candidate
FUZZY_THRESHOLD = 80

# Minimum drug-name length to consider (avoid matching "A", "AS", etc.)
MIN_DRUG_NAME_LENGTH = 3

# Common stop words that should NOT be extracted even if they match
STOP_WORDS: set[str] = {
    "THE",
    "AND",
    "FOR",
    "WITH",
    "THIS",
    "THAT",
    "FROM",
    "HAVE",
    "HAS",
    "ARE",
    "WAS",
    "WERE",
    "BEEN",
    "WILL",
    "CAN",
    "MAY",
    "SHOULD",
    "WHAT",
    "WHICH",
    "WHEN",
    "WHERE",
    "HOW",
    "WHO",
    "WHY",
    "NOT",
    "DOES",
    "DRUG",
    "DRUGS",
    "EFFECT",
    "EFFECTS",
    "SIDE",
    "TAKE",
    "TAKING",
    "USE",
    "USED",
    "CAUSE",
    "CAUSES",
    "INTERACTION",
    "INTERACTIONS",
    "ADVERSE",
    "EVENT",
    "EVENTS",
    "REPORT",
    "REPORTS",
    "PATIENT",
    "PATIENTS",
    "DOSE",
    "RISK",
    "TREATMENT",
    # Spanish stop words
    "QUE",
    "LOS",
    "LAS",
    "DEL",
    "UNA",
    "CON",
    "POR",
    "PARA",
    "UNO",
    "COMO",
    "MAS",
    "SUS",
    "HAY",
    "SER",
    "TIENE",
    "ENTRE",
    "CUANDO",
    "EFECTOS",
    "ADVERSOS",
    "SECUNDARIOS",
    "MEDICAMENTO",
    "MEDICAMENTOS",
    "INTERACCIONES",
    "TOMAR",
    "FARMACO",
    "FARMACOS",
}


@dataclass
class ExtractedEntities:
    """Result of entity extraction from a user question."""

    drugs: list[str] = field(default_factory=list)
    """Drug names found (uppercase, normalised)."""

    adverse_events: list[str] = field(default_factory=list)
    """Adverse event names found (uppercase, normalised)."""

    raw_query: str = ""
    """The original user question."""

    def is_empty(self) -> bool:
        """Return True when no entities were extracted."""
        return not self.drugs and not self.adverse_events


# ---------------------------------------------------------------------------
# Known-drug catalogue
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_known_drugs_from_disk() -> set[str]:
    """Load drug names from the DailyMed JSON files on disk.

    Returns a set of uppercase drug names. Includes both the primary
    drug_name and any generic/brand names found in the label files.
    """
    dailymed_dir = DATA_RAW_DIR / "dailymed"
    drugs: set[str] = set()

    if not dailymed_dir.exists():
        logger.warning("DailyMed directory not found: {}", dailymed_dir)
        return drugs

    for json_file in sorted(dailymed_dir.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            with open(json_file, encoding="utf-8") as f:
                label = json.load(f)
            drug_name = label.get("drug_name", "")
            if drug_name:
                drugs.add(drug_name.upper().strip())
            for gn in label.get("generic_names", []):
                name = gn.upper().strip()
                if len(name) >= MIN_DRUG_NAME_LENGTH:
                    drugs.add(name)
            for bn in label.get("brand_names", []):
                name = bn.upper().strip()
                if len(name) >= MIN_DRUG_NAME_LENGTH:
                    drugs.add(name)
        except (json.JSONDecodeError, KeyError):
            continue

    logger.info("Loaded {} known drug names from disk", len(drugs))
    return drugs


def _load_known_drugs_from_neo4j() -> set[str]:
    """Load drug names from Neo4j (best-effort, non-blocking)."""
    try:
        from pharmagraphrag.graph.queries import search_drugs

        # Grab a large batch — search_drugs with empty query returns all
        names = search_drugs("", limit=50_000)
        drugs = {n.upper().strip() for n in names if len(n) >= MIN_DRUG_NAME_LENGTH}
        logger.info("Loaded {} drug names from Neo4j", len(drugs))
        return drugs
    except Exception as exc:
        logger.debug("Could not load drugs from Neo4j: {}", exc)
        return set()


_cached_all_drugs: set[str] | None = None


def get_known_drugs(use_neo4j: bool = True) -> set[str]:
    """Return the full set of known drug names.

    Merges disk + Neo4j catalogues. Cached after first call.

    Args:
        use_neo4j: Whether to attempt loading from Neo4j.

    Returns:
        Set of uppercase drug names.
    """
    global _cached_all_drugs
    if _cached_all_drugs is not None:
        return _cached_all_drugs

    drugs = _load_known_drugs_from_disk()
    if use_neo4j:
        drugs |= _load_known_drugs_from_neo4j()

    # Filter out stop words that snuck in
    drugs -= STOP_WORDS

    _cached_all_drugs = drugs
    return drugs


def reset_cache() -> None:
    """Clear the known-drugs cache (useful in tests)."""
    global _cached_all_drugs
    _cached_all_drugs = None
    _load_known_drugs_from_disk.cache_clear()


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def _normalize_query(text: str) -> str:
    """Normalise a query for matching: uppercase, collapse whitespace."""
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _exact_match(query_upper: str, known_drugs: set[str]) -> list[str]:
    """Find drug names that appear as substrings in the query.

    Longer names are matched first to avoid partial overlaps
    (e.g. "INSULIN GLARGINE" before "INSULIN").
    """
    found: list[str] = []
    sorted_drugs = sorted(known_drugs, key=len, reverse=True)

    for drug in sorted_drugs:
        if len(drug) < MIN_DRUG_NAME_LENGTH:
            continue
        # Use word-boundary-aware search to avoid matching inside words
        pattern = r"(?<![A-Z])" + re.escape(drug) + r"(?![A-Z])"
        if re.search(pattern, query_upper):
            # Make sure the match isn't already covered by a longer name
            already_covered = any(drug in longer for longer in found if drug != longer)
            if not already_covered:
                found.append(drug)

    return found


def _fuzzy_match(
    query_tokens: list[str],
    known_drugs: set[str],
    threshold: int = FUZZY_THRESHOLD,
) -> list[str]:
    """Find drugs via fuzzy token matching.

    For each meaningful token (or bigram) in the query, check whether
    it closely matches a known drug name.
    """
    if not known_drugs:
        return []

    candidates: list[str] = []
    drugs_list = list(known_drugs)

    # Single tokens
    for token in query_tokens:
        if len(token) < MIN_DRUG_NAME_LENGTH or token in STOP_WORDS:
            continue
        match = process.extractOne(
            token,
            drugs_list,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if match:
            candidates.append(match[0])

    # Bigrams (for multi-word drugs like "INSULIN GLARGINE")
    for i in range(len(query_tokens) - 1):
        bigram = f"{query_tokens[i]} {query_tokens[i + 1]}"
        if any(t in STOP_WORDS for t in [query_tokens[i], query_tokens[i + 1]]):
            continue
        match = process.extractOne(
            bigram,
            drugs_list,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if match:
            candidates.append(match[0])

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_entities(
    question: str,
    *,
    use_neo4j: bool = True,
    fuzzy: bool = True,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> ExtractedEntities:
    """Extract drug names (and optionally adverse events) from a question.

    Strategy:
      1. Exact substring matching against all known drugs.
      2. Fuzzy matching per token / bigram if ``fuzzy=True``.
      3. Deduplicate and return.

    Args:
        question: User question in natural language.
        use_neo4j: Include drugs from Neo4j in the catalogue.
        fuzzy: Enable fuzzy matching as fallback.
        fuzzy_threshold: Minimum rapidfuzz score (0-100).

    Returns:
        ExtractedEntities with recognised drugs list.
    """
    known_drugs = get_known_drugs(use_neo4j=use_neo4j)
    query_upper = _normalize_query(question)
    tokens = query_upper.split()

    # 1) Exact substring matching
    drugs = _exact_match(query_upper, known_drugs)

    # 2) Fuzzy fallback (only for tokens not already matched)
    if fuzzy and not drugs:
        fuzzy_drugs = _fuzzy_match(tokens, known_drugs, threshold=fuzzy_threshold)
        drugs.extend(fuzzy_drugs)

    # Deduplicate, preserving order
    seen: set[str] = set()
    unique_drugs: list[str] = []
    for d in drugs:
        if d not in seen:
            seen.add(d)
            unique_drugs.append(d)

    result = ExtractedEntities(
        drugs=unique_drugs,
        raw_query=question,
    )

    logger.debug(
        "Extracted entities from '{}': drugs={}",
        question[:80],
        unique_drugs,
    )

    return result
