"""Extract demo data from Neo4j for the demo dataset."""

import json
from pathlib import Path

from neo4j import GraphDatabase

# DailyMed drugs
dailymed_dir = Path("data/raw/dailymed")
dm_drugs = []
for f in sorted(dailymed_dir.glob("*.json")):
    if f.name.startswith("_"):
        continue
    dm_drugs.append(f.stem.upper().replace("_", " "))

print(f"DailyMed drugs: {len(dm_drugs)}")

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pharmagraphrag"))

# Find DailyMed drugs that also have FAERS data
demo_data = {"drugs": [], "adverse_events": [], "causes": [], "outcomes": [], "has_outcome": []}

with driver.session() as s:
    for drug_name in dm_drugs:
        # Get top 15 adverse events per drug
        result = s.run(
            """
            MATCH (d:Drug {name: $name})-[r:CAUSES]->(ae:AdverseEvent)
            RETURN d.name AS drug, ae.name AS event, r.report_count AS report_count
            ORDER BY r.report_count DESC
            LIMIT 15
            """,
            name=drug_name,
        )
        records = [dict(r) for r in result]
        if records:
            demo_data["drugs"].append(drug_name)
            for rec in records:
                if rec["event"] not in demo_data["adverse_events"]:
                    demo_data["adverse_events"].append(rec["event"])
                demo_data["causes"].append(
                    {
                        "drug_name": rec["drug"],
                        "event_name": rec["event"],
                        "report_count": rec["report_count"],
                    }
                )

        # Get outcomes
        result = s.run(
            """
            MATCH (d:Drug {name: $name})-[r:HAS_OUTCOME]->(o:Outcome)
            RETURN d.name AS drug, o.code AS code, o.description AS description,
                   r.report_count AS report_count
            """,
            name=drug_name,
        )
        for rec in result:
            demo_data["has_outcome"].append(
                {
                    "drug_name": rec["drug"],
                    "outc_cod": rec["code"],
                    "description": rec["description"],
                    "report_count": rec["report_count"],
                }
            )

    # Get all outcomes
    result = s.run("MATCH (o:Outcome) RETURN o.code AS code, o.description AS description")
    demo_data["outcomes"] = [dict(r) for r in result]

driver.close()

print(f"Drugs with FAERS data: {len(demo_data['drugs'])}")
print(f"Adverse events: {len(demo_data['adverse_events'])}")
print(f"CAUSES relationships: {len(demo_data['causes'])}")
print(f"HAS_OUTCOME relationships: {len(demo_data['has_outcome'])}")
print(f"Outcomes: {len(demo_data['outcomes'])}")

# Save
output = Path("data/demo/faers_graph.json")
output.parent.mkdir(parents=True, exist_ok=True)
with open(output, "w", encoding="utf-8") as f:
    json.dump(demo_data, f, indent=2)

print(f"\nSaved to {output}")
print(f"File size: {output.stat().st_size / 1024:.1f} KB")
