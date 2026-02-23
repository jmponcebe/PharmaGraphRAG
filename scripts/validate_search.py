"""Quick validation of ChromaDB similarity search."""

from pharmagraphrag.vectorstore.store import (
    format_vector_context,
    search,
    search_by_drug,
)


def main() -> None:
    print("=" * 60)
    print("QUERY 1: What are the side effects of warfarin?")
    print("=" * 60)
    results = search("What are the side effects of warfarin?", n_results=3)
    for r in results:
        m = r["metadata"]
        print(f"  [{m['drug_name']} / {m['section']}] distance={r['distance']:.4f}")
        print(f"  {r['text'][:150]}...")
        print()

    print("=" * 60)
    print("QUERY 2: Does metformin interact with alcohol?")
    print("=" * 60)
    results = search("Does metformin interact with alcohol?", n_results=3)
    for r in results:
        m = r["metadata"]
        print(f"  [{m['drug_name']} / {m['section']}] distance={r['distance']:.4f}")
        print(f"  {r['text'][:150]}...")
        print()

    print("=" * 60)
    print("QUERY 3: search_by_drug for LISINOPRIL")
    print("=" * 60)
    results = search_by_drug(
        "blood pressure medication side effects", drug_name="LISINOPRIL", n_results=3
    )
    for r in results:
        m = r["metadata"]
        print(f"  [{m['drug_name']} / {m['section']}] distance={r['distance']:.4f}")
        print(f"  {r['text'][:150]}...")
        print()

    print("=" * 60)
    print("QUERY 4: format_vector_context output")
    print("=" * 60)
    results = search("Can I take ibuprofen with aspirin?", n_results=3)
    print(format_vector_context(results, max_chars=800))


if __name__ == "__main__":
    main()
