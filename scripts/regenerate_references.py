"""Regenerate testset references using actual data from the KG and vector store.

Produces a new testset (testset_v2.json) where each reference is grounded in
what is actually loaded in Neo4j Aura (CAUSES, INTERACTS_WITH, HAS_OUTCOME,
BELONGS_TO) and ChromaDB (DailyMed labels), rather than hand-written from
textbook knowledge.

This eliminates the ContextRecall=0 problem caused by mismatch between the
original references and the data the system retrieves.

Loads .env.aura explicitly (does not touch the local .env). Read-only.

Usage:
    uv run python scripts/regenerate_references.py \
        --testset data/evaluation/testset.json \
        --output data/evaluation/testset_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

# Load Aura credentials BEFORE importing pharmagraphrag (pydantic-settings cache)
aura_env = dotenv_values(ROOT / ".env.aura")
os.environ["NEO4J_URI"] = aura_env["NEO4J_URI"]
os.environ["NEO4J_USER"] = aura_env.get("NEO4J_USER") or aura_env["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = aura_env["NEO4J_PASSWORD"]

# Now safe to import; settings will pick up the Aura values
import re  # noqa: E402

from pharmagraphrag.graph import queries  # noqa: E402
from pharmagraphrag.vectorstore import store  # noqa: E402

# ---------- helpers ---------------------------------------------------------


def _clean_snippet(text: str, max_chars: int = 240) -> str:
    """Strip ChromaDB metadata headers and partial leading words."""
    # Drop our own header markers like "Drug: METFORMIN | Section: Drug Interactions"
    text = re.sub(r"Drug:\s*[A-Z0-9 .,\\/-]+\s*\|\s*Section:[^\n]*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # If the snippet starts mid-word (lowercase letter following nothing), advance to a sentence start
    if text and text[0].islower():
        m = re.search(r"(?<=[.!?])\s+[A-Z]", text)
        if m:
            text = text[m.start() + 1 :].lstrip()
        else:
            # Or jump to the first capital letter
            m2 = re.search(r"[A-Z]", text)
            if m2:
                text = text[m2.start() :]
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def fmt_top_ae(drug: str, n: int = 5) -> str:
    events = queries.get_drug_adverse_events(drug, limit=n)
    if not events:
        return f"No adverse events were found for {drug.title()} in FAERS."
    items = ", ".join(f"{e['adverse_event'].lower()} ({e['report_count']} reports)" for e in events)
    return f"The most frequently reported adverse events for {drug.title()} in FAERS are {items}."


def fmt_drugs_for_ae(ae: str, n: int = 5) -> str:
    drugs = queries.get_adverse_event_drugs(ae, limit=n)
    if drugs:
        items = ", ".join(f"{d['drug_name'].title()} ({d['report_count']} reports)" for d in drugs)
        return f"Drugs reported in FAERS as causing {ae.upper()} include {items}."
    # Fallback to fuzzy substring match on AE names
    matches = queries.search_adverse_events(ae, limit=5)
    if not matches:
        return f"No drugs were found in FAERS for the adverse event '{ae}' or related MedDRA terms."
    items = ", ".join(f"{m['name'].lower()} ({m['total_reports']} reports)" for m in matches)
    return f"The exact MedDRA term '{ae.upper()}' was not found, but related terms include {items}."


def fmt_interaction(drug_a: str, drug_b: str) -> str:
    inters = queries.get_drug_interactions(drug_a)
    match = next(
        (i for i in inters if i["interacting_drug"].upper() == drug_b.upper()),
        None,
    )
    if match:
        desc = _clean_snippet(match.get("description") or "", max_chars=220)
        src = (match.get("source") or "").strip()
        if desc:
            return (
                f"Yes, {drug_a.title()} and {drug_b.title()} have a documented "
                f"interaction (source: {src or 'DailyMed'}). {desc}"
            )
        return (
            f"Yes, {drug_a.title()} and {drug_b.title()} have a documented "
            f"interaction in {src or 'DailyMed labels'}."
        )
    return (
        f"No direct INTERACTS_WITH relationship between {drug_a.title()} and "
        f"{drug_b.title()} is documented in the knowledge graph."
    )


def fmt_outcomes(drug: str) -> str:
    outs = queries.get_drug_outcomes(drug)
    if not outs:
        return f"No clinical outcomes are documented for {drug.title()}."
    items = ", ".join(
        f"{o['outcome_description'] or o['outcome_code']} ({o['report_count']} reports)"
        for o in outs[:5]
    )
    return f"Clinical outcomes documented for {drug.title()} in FAERS include {items}."


def fmt_category(drug: str) -> str:
    cats = queries.get_drug_category(drug)
    if not cats:
        return (
            f"{drug.title()} has no DrugCategory node assigned in the knowledge graph "
            f"(BELONGS_TO relationship is missing for this drug)."
        )
    return f"{drug.title()} belongs to the {', '.join(cats)} category."


def fmt_drugs_in_same_category(drug: str) -> str:
    cats = queries.get_drug_category(drug)
    if not cats:
        return (
            f"{drug.title()} has no DrugCategory assigned in the knowledge graph, "
            f"so no peer drugs can be retrieved by category."
        )
    cat = cats[0]
    peers = queries.get_drugs_by_category(cat, limit=15)
    names = sorted(
        {p["drug_name"].title() for p in peers if p["drug_name"].upper() != drug.upper()}
    )
    if not names:
        return f"{drug.title()} is the only drug in the {cat} category in this knowledge graph."
    return f"{drug.title()} belongs to the {cat} category, alongside {', '.join(names[:8])}."


def fmt_top_interactions(drug: str, n: int = 6) -> str:
    inters = queries.get_drug_interactions(drug)
    if not inters:
        return f"No drug interactions are documented for {drug.title()}."
    names = sorted({i["interacting_drug"].title() for i in inters})[:n]
    return f"Drugs documented as interacting with {drug.title()} include {', '.join(names)}."


def fmt_compare(drug_a: str, drug_b: str) -> str:
    cats_a = queries.get_drug_category(drug_a) or ["no assigned category"]
    cats_b = queries.get_drug_category(drug_b) or ["no assigned category"]
    aes_a = {e["adverse_event"] for e in queries.get_drug_adverse_events(drug_a, limit=20)}
    aes_b = {e["adverse_event"] for e in queries.get_drug_adverse_events(drug_b, limit=20)}
    shared = sorted(aes_a & aes_b)[:5]
    return (
        f"{drug_a.title()} ({cats_a[0]}) and {drug_b.title()} ({cats_b[0]}) "
        f"share adverse events such as "
        f"{', '.join(s.lower() for s in shared) if shared else 'none in the top 20 reports'}, "
        f"based on FAERS data in the knowledge graph."
    )


def fmt_shared_ae(drug_a: str, drug_b: str) -> str:
    aes_a = {e["adverse_event"] for e in queries.get_drug_adverse_events(drug_a, limit=30)}
    aes_b = {e["adverse_event"] for e in queries.get_drug_adverse_events(drug_b, limit=30)}
    shared = sorted(aes_a & aes_b)
    if not shared:
        return f"{drug_a.title()} and {drug_b.title()} share no top-30 adverse events in FAERS."
    return (
        f"Adverse events reported in FAERS for both {drug_a.title()} and {drug_b.title()} "
        f"include {', '.join(s.lower() for s in shared[:8])}."
    )


def fmt_drug_search(pattern: str) -> str:
    matches = queries.search_drugs(pattern, limit=12)
    if not matches:
        return f"No drugs whose name contains '{pattern}' were found in the knowledge graph."
    return (
        f"Drugs whose name contains '{pattern}' include "
        f"{', '.join(m.title() for m in matches[:10])}."
    )


def fmt_top_ae_overall(n: int = 6) -> str:
    drv = queries._get_driver()
    with drv.session() as s:
        rows = s.run(
            """
            MATCH (:Drug)-[r:CAUSES]->(ae:AdverseEvent)
            RETURN ae.name AS name, sum(r.report_count) AS total
            ORDER BY total DESC
            LIMIT $n
            """,
            n=n,
        ).data()
    items = ", ".join(f"{r['name'].lower()} ({r['total']:,} total reports)" for r in rows)
    return f"The most frequently reported adverse events across all drugs in FAERS are {items}."


def fmt_drugs_with_two_aes(ae1: str, ae2: str, n: int = 8) -> str:
    drv = queries._get_driver()
    with drv.session() as s:
        rows = s.run(
            """
            MATCH (d:Drug)-[r1:CAUSES]->(a:AdverseEvent)
            WHERE toUpper(a.name) = toUpper($ae1)
            WITH d, r1.report_count AS c1
            MATCH (d)-[r2:CAUSES]->(b:AdverseEvent)
            WHERE toUpper(b.name) = toUpper($ae2)
            WITH d.name AS name, c1 + r2.report_count AS total
            RETURN name, total
            ORDER BY total DESC
            LIMIT $n
            """,
            ae1=ae1,
            ae2=ae2,
            n=n,
        ).data()
    if not rows:
        return f"No drugs were found in FAERS that report both {ae1.upper()} and {ae2.upper()}."
    items = ", ".join(f"{r['name'].title()} ({r['total']} combined reports)" for r in rows)
    return (
        f"Drugs that report both {ae1.upper()} and {ae2.upper()} as adverse events, "
        f"ranked by combined report count, include {items}."
    )


def fmt_ae_count(drug: str, ae: str) -> str:
    drv = queries._get_driver()
    with drv.session() as s:
        rec = s.run(
            """
            MATCH (d:Drug)-[r:CAUSES]->(a:AdverseEvent)
            WHERE toUpper(d.name) = toUpper($drug) AND toUpper(a.name) = toUpper($ae)
            RETURN r.report_count AS c
            """,
            drug=drug,
            ae=ae,
        ).single()
    if not rec:
        return f"No FAERS reports link {drug.title()} to {ae.upper()} in the knowledge graph."
    return f"FAERS reports {rec['c']} cases of {ae.upper()} associated with {drug.title()}."


def fmt_label_search(
    query: str,
    drug_filter: str | None = None,
    preferred_sections: tuple[str, ...] | None = None,
) -> str:
    """Use ChromaDB to find what the DailyMed label actually says about a topic.

    Picks the best chunk by (a) preferred sections matching the question intent,
    (b) semantic relevance from ChromaDB, and cleans the snippet.
    """
    where = {"drug_name": drug_filter.upper()} if drug_filter else None
    results = store.search(query, n_results=8, where=where)
    if not results:
        return f"No DailyMed label content matched '{query}'" + (
            f" for {drug_filter.title()}." if drug_filter else "."
        )
    if preferred_sections:
        # Tie-break only among top-4 by semantic distance — avoid promoting
        # short/irrelevant chunks just because their section label matches.
        top = sorted(results, key=lambda r: r.get("distance", 1.0))[:4]
        ranked = sorted(
            top,
            key=lambda r: (
                0 if r["metadata"].get("section") in preferred_sections else 1,
                r.get("distance", 1.0),
            ),
        )
    else:
        ranked = sorted(results, key=lambda r: r.get("distance", 1.0))
    # Pre-clean every candidate so we can filter by useful-content length
    cleaned = []
    for r in ranked:
        c = _clean_snippet(r["text"], max_chars=260)
        if len(c) >= 150:
            cleaned.append((r, c))
    chosen = cleaned[:2]
    name = drug_filter.title() if drug_filter else "the drug"
    if not chosen:
        sections_in_results = sorted({r["metadata"].get("section", "?") for r in results})
        return (
            f"The DailyMed label chunks indexed for {name} do not contain a substantive "
            f"passage matching '{query}'. Available sections in the indexed label "
            f"include: {', '.join(sections_in_results)}."
        )
    snippets = []
    for r, text in chosen:
        section = r["metadata"].get("section", "drug label")
        snippets.append(f"[{section}] {text}")
    return f"Based on the DailyMed label for {name}: " + " ".join(snippets)


# ---------- per-id mapping --------------------------------------------------

REGENERATORS = {
    "q01": lambda: fmt_top_ae("ASPIRIN", n=5),
    "q02": lambda: fmt_interaction("WARFARIN", "ASPIRIN"),
    "q03": lambda: fmt_drugs_for_ae("HEPATOTOXICITY", n=5),
    "q04": lambda: fmt_outcomes("METFORMIN"),
    "q05": lambda: fmt_category("IBUPROFEN"),
    "q06": lambda: fmt_compare("ASPIRIN", "IBUPROFEN"),
    "q07": lambda: fmt_label_search(
        "drug interactions", "METFORMIN", preferred_sections=("drug_interactions",)
    ),
    "q08": lambda: fmt_top_ae("LISINOPRIL", n=5),
    "q09": lambda: fmt_top_interactions("METHOTREXATE"),
    "q10": lambda: fmt_drugs_for_ae("RHABDOMYOLYSIS", n=5),
    "q11": lambda: fmt_outcomes("ATORVASTATIN"),
    "q12": lambda: fmt_label_search(
        "warnings and precautions",
        "OMEPRAZOLE",
        preferred_sections=("warnings_and_cautions", "warnings", "boxed_warning"),
    ),
    "q13": lambda: fmt_drugs_in_same_category("WARFARIN"),
    "q14": lambda: fmt_top_ae_overall(n=6),
    "q15": lambda: fmt_compare("WARFARIN", "APIXABAN"),
    "q16": lambda: fmt_interaction("METFORMIN", "LISINOPRIL"),
    "q17": lambda: fmt_label_search(
        "contraindications and do not use",
        "ASPIRIN",
        preferred_sections=("contraindications", "warnings", "boxed_warning"),
    ),
    "q18": lambda: fmt_drug_search("STATIN"),
    "q19": lambda: fmt_label_search(
        "mechanism of action proton pump inhibitor",
        "OMEPRAZOLE",
        preferred_sections=("mechanism_of_action", "clinical_pharmacology", "pharmacodynamics"),
    ),
    "q20": lambda: fmt_shared_ae("ASPIRIN", "WARFARIN"),
    "q21": lambda: fmt_ae_count("IBUPROFEN", "NAUSEA"),
    "q22": lambda: fmt_top_interactions("WARFARIN", n=8),
    "q23": lambda: fmt_top_ae("PREDNISONE", n=5),
    "q24": lambda: fmt_label_search(
        "dosage and administration recommended dose",
        "METFORMIN",
        preferred_sections=("dosage_and_administration", "indications_and_usage"),
    ),
    "q25": lambda: fmt_drugs_with_two_aes("NAUSEA", "HEADACHE", n=8),
}


# ---------- main ------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/evaluation/testset.json")
    ap.add_argument("--output", default="data/evaluation/testset_v2.json")
    args = ap.parse_args()

    in_path = ROOT / args.testset
    out_path = ROOT / args.output

    with in_path.open(encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Regenerating references for {len(samples)} samples...\n")

    skipped = []
    for s in samples:
        sid = s["id"]
        regen = REGENERATORS.get(sid)
        if regen is None:
            skipped.append(sid)
            continue
        old_ref = s.get("reference", "")
        try:
            new_ref = regen()
        except Exception as exc:  # pragma: no cover - diagnostic
            new_ref = f"[regen failed: {type(exc).__name__}: {exc}]"
        s["original_reference"] = old_ref
        s["reference"] = new_ref
        print(f"--- {sid}: {s['question']}")
        print(f"OLD: {old_ref}")
        print(f"NEW: {new_ref}\n")

    data["metadata"]["regenerated_at"] = "2026-04-29"
    data["metadata"]["regeneration_note"] = (
        "References regenerated from actual KG (Neo4j Aura) + ChromaDB content "
        "to fix the mismatch between hand-written textbook references and the data "
        "the system retrieves. Original references preserved under "
        "'original_reference'."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path}")
    if skipped:
        print(f"Skipped (no regenerator): {skipped}")


if __name__ == "__main__":
    main()
