"""Smoke test for Neo4j Aura connection.

Loads .env.aura explicitly, connects, and runs a simple count query.
Does not modify any data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import dotenv_values
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
env = dotenv_values(ROOT / ".env.aura")

uri = env.get("NEO4J_URI")
user = env.get("NEO4J_USER") or env.get("NEO4J_USERNAME")
pwd = env.get("NEO4J_PASSWORD")
db = env.get("NEO4J_DATABASE") or "neo4j"

if not all([uri, user, pwd]):
    print("Missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env.aura")
    sys.exit(1)

print(f"Connecting to {uri} as user='{user}' db='{db}'")

try:
    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
        driver.verify_connectivity()
        with driver.session(database=db) as session:
            n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            labels = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label").value()
            rels = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            ).value()
        print(f"OK. Nodes={n:,} Relationships={r:,}")
        print(f"Labels: {labels}")
        print(f"Relationship types: {rels}")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    sys.exit(2)
