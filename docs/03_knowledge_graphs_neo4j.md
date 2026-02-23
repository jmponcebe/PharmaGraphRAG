# Knowledge Graphs y Neo4j

## ¿Qué es un grafo?

Un **grafo** es una estructura de datos con dos componentes:

- **Nodos** (vértices): Representan entidades (cosas)
- **Aristas** (edges/relaciones): Representan conexiones entre entidades

```
  [Ana] ---amiga de---> [Bob] ---trabaja en---> [Google]
    │                     │
    └──estudia en──→ [MIT] ←──estudia en──┘
```

Los grafos son naturales para representar **relaciones entre cosas**. Mucho mejor que las tablas SQL para ciertos problemas.

### Grafo vs Base de datos relacional (SQL)

Imagina que quieres saber: *"Amigos de amigos de Ana que trabajan en Google"*

**En SQL** (muchas tablas + JOINs complejos):
```sql
SELECT p3.name
FROM friendships f1
JOIN friendships f2 ON f1.friend_id = f2.person_id
JOIN employment e ON f2.friend_id = e.person_id
JOIN companies c ON e.company_id = c.id
WHERE f1.person_id = (SELECT id FROM people WHERE name='Ana')
AND c.name = 'Google'
```

**En Cypher** (Neo4j):
```cypher
MATCH (ana:Person {name: "Ana"})-[:FRIEND]->()-[:FRIEND]->(fof)-[:WORKS_AT]->(g:Company {name: "Google"})
RETURN fof.name
```

Para problemas con **muchas relaciones y navegación**, los grafos son más naturales y eficientes.

---

## ¿Qué es un Knowledge Graph (KG)?

Un **Knowledge Graph** es un grafo donde:
- Los nodos representan **entidades del mundo real** (fármacos, enfermedades, personas)
- Las relaciones representan **hechos** sobre esas entidades
- Cada relación tiene un **tipo** y puede tener **propiedades**

Ejemplos famosos:
- **Google Knowledge Graph**: Cuando buscas "Albert Einstein" y te sale un panel con datos estructurados
- **Wikidata**: Los datos detrás de Wikipedia, en formato de grafo
- **SNOMED CT**: Terminología médica en formato de grafo

### Nuestro Knowledge Graph farmacéutico

```
(:Drug {name: "WARFARIN"})
  -[:CAUSES {report_count: 1200}]->
(:AdverseEvent {name: "HAEMORRHAGE"})

(:Drug {name: "WARFARIN"})
  -[:INTERACTS_WITH {source: "dailymed"}]->
(:Drug {name: "ASPIRIN"})

(:Drug {name: "WARFARIN"})
  -[:HAS_OUTCOME {report_count: 50}]->
(:Outcome {code: "DE", name: "Death"})

(:Drug {name: "WARFARIN"})
  -[:BELONGS_TO]->
(:DrugCategory {name: "Vitamin K Antagonists"})
```

---

## Neo4j: La base de datos de grafos

### ¿Qué es Neo4j?
Neo4j es la base de datos de grafos más popular del mundo. Es a los grafos lo que MySQL es a las tablas SQL.

- **Modelo**: Grafos de propiedades (nodos y relaciones con propiedades)
- **Lenguaje**: Cypher (como SQL pero para grafos)
- **Licencia**: Community (gratis) y Enterprise (pago)
- **Almacenamiento**: En disco, optimizado para traversals
- **Interfaz**: Neo4j Browser (web en `localhost:7474`)

### ¿Por qué Neo4j y no SQL?

Para nuestros datos:

```
SQL: 4 tablas (drugs, adverse_events, outcomes, categories)
     + 4 tablas de relación (drug_causes_event, drug_interacts_drug, ...)
     = 8 tablas con JOINs complejos

Neo4j: Nodos y relaciones directas, consultas intuitivas
```

---

## Cypher: El lenguaje de consultas

### Sintaxis visual — "Dibujas" el patrón que buscas

```
(nodo)                    → Un nodo
(d:Drug)                  → Un nodo con etiqueta Drug, variable "d"
(d:Drug {name: "ASPIRIN"}) → Un Drug con nombre ASPIRIN
-[r:CAUSES]->             → Una relación de tipo CAUSES, dirección →
-[:INTERACTS_WITH]-       → Relación sin dirección (bidireccional)
```

### Las cláusulas fundamentales

#### MATCH — Buscar un patrón
```cypher
-- Encuentra todos los fármacos
MATCH (d:Drug)
RETURN d.name
LIMIT 10
```

#### WHERE — Filtrar
```cypher
-- Fármacos que empiezan con "WAR"
MATCH (d:Drug)
WHERE d.name STARTS WITH "WAR"
RETURN d.name
```

#### RETURN — Qué devolver
```cypher
-- Nombre del fármaco y número de efectos que causa
MATCH (d:Drug)-[r:CAUSES]->(ae:AdverseEvent)
WHERE d.name = "ASPIRIN"
RETURN ae.name, r.report_count
ORDER BY r.report_count DESC
LIMIT 10
```

### Consultas de nuestro proyecto (queries.py)

#### 1. Información de un fármaco
```cypher
MATCH (d:Drug {name: "WARFARIN"})
RETURN d
```

#### 2. Efectos adversos de un fármaco
```cypher
MATCH (d:Drug {name: "WARFARIN"})-[r:CAUSES]->(ae:AdverseEvent)
RETURN ae.name, r.report_count
ORDER BY r.report_count DESC
LIMIT 20
```
Resultado:
```
HAEMORRHAGE          | 1,200
DRUG INTERACTION     |   800
INTERNATIONAL NORMALISED RATIO INCREASED | 750
DEATH                |   500
...
```

#### 3. Interacciones de un fármaco
```cypher
MATCH (d:Drug {name: "WARFARIN"})-[r:INTERACTS_WITH]-(other:Drug)
RETURN other.name, r.description
```
Nota: `-[r:INTERACTS_WITH]-` (sin flecha) busca en ambas direcciones.

#### 4. Outcomes clínicos
```cypher
MATCH (d:Drug {name: "WARFARIN"})-[r:HAS_OUTCOME]->(o:Outcome)
RETURN o.name, r.report_count
ORDER BY r.report_count DESC
```

#### 5. Multi-hop: Fármacos que comparten efectos
```cypher
-- "¿Qué otros fármacos causan los mismos efectos que WARFARIN?"
MATCH (d:Drug {name: "WARFARIN"})-[:CAUSES]->(ae:AdverseEvent)<-[:CAUSES]-(other:Drug)
WHERE other.name <> "WARFARIN"
RETURN other.name, COUNT(ae) AS shared_effects
ORDER BY shared_effects DESC
LIMIT 10
```

Esta consulta hace un **traversal de 2 saltos**:
```
WARFARIN → CAUSES → HAEMORRHAGE ← CAUSES ← ASPIRIN
WARFARIN → CAUSES → NAUSEA      ← CAUSES ← IBUPROFEN
```

#### 6. Contexto completo para GraphRAG (get_drug_full_context)
```cypher
-- Todo lo que sabemos de un fármaco en una sola consulta
MATCH (d:Drug {name: $drug_name})

OPTIONAL MATCH (d)-[r:CAUSES]->(ae:AdverseEvent)
WITH d, ae, r ORDER BY r.report_count DESC LIMIT 15

OPTIONAL MATCH (d)-[:INTERACTS_WITH]-(other:Drug)
OPTIONAL MATCH (d)-[:HAS_OUTCOME]->(o:Outcome)
OPTIONAL MATCH (d)-[:BELONGS_TO]->(cat:DrugCategory)

RETURN d, collect(DISTINCT ae) AS events, 
       collect(DISTINCT other) AS interactions,
       collect(DISTINCT o) AS outcomes,
       collect(DISTINCT cat) AS categories
```

---

## Nuestro esquema Neo4j en detalle

### Nodos

| Etiqueta | Propiedades | Cantidad | Ejemplo |
|----------|------------|----------|---------|
| **Drug** | name, pharmacologic_class, source | 4,998 | `{name: "WARFARIN", pharmacologic_class: "Vitamin K Antagonists"}` |
| **AdverseEvent** | name | 6,863 | `{name: "HAEMORRHAGE"}` |
| **Outcome** | code, name | 7 | `{code: "DE", name: "Death"}` |
| **DrugCategory** | name | 32 | `{name: "HMG-CoA Reductase Inhibitors"}` |

### Relaciones

| Tipo | De → A | Propiedades | Cantidad | Significado |
|------|--------|------------|----------|-------------|
| **CAUSES** | Drug → AdverseEvent | report_count | 365,360 | "Fármaco X causó efecto Y (N reportes)" |
| **INTERACTS_WITH** | Drug ↔ Drug | source, description | 193 | "Fármaco X interactúa con fármaco Y" |
| **HAS_OUTCOME** | Drug → Outcome | report_count | 15,759 | "Fármaco X tuvo resultado clínico Y" |
| **BELONGS_TO** | Drug → DrugCategory | — | 47 | "Fármaco X pertenece a categoría Y" |

### Constraints e Indexes (schema.py)

**Constraints** (restricciones):
```cypher
-- Cada Drug tiene un nombre único
CREATE CONSTRAINT drug_name_unique IF NOT EXISTS
FOR (d:Drug) REQUIRE d.name IS UNIQUE

-- Cada AdverseEvent tiene un nombre único  
CREATE CONSTRAINT ae_name_unique IF NOT EXISTS
FOR (ae:AdverseEvent) REQUIRE ae.name IS UNIQUE

-- (similar para Outcome.code y DrugCategory.name)
```

**Indexes** (índices para búsqueda rápida):
```cypher
-- Índice de texto para buscar fármacos por nombre parcial
CREATE TEXT INDEX drug_name_text IF NOT EXISTS
FOR (d:Drug) ON (d.name)
```

Sin índices, buscar un fármaco requeriría escanear los 4,998 nodos. Con el índice, es instantáneo.

---

## Cómo se cargan los datos (loader.py)

### Paso 1: Cargar FAERS → Neo4j

```python
# 1. Leer Parquet (DRUG + REAC)
drug_df = pd.read_parquet("data/processed/faers/2024Q3/DRUG.parquet")
reac_df = pd.read_parquet("data/processed/faers/2024Q3/REAC.parquet")

# 2. JOIN por primaryid (el ID del reporte)
merged = drug_df.merge(reac_df, on="primaryid")
# Resultado: fármaco + reacción + primaryid

# 3. Agregar: contar pares (fármaco, reacción)
counts = merged.groupby(["drugname", "pt"]).size().reset_index(name="report_count")
# Ejemplo: ("WARFARIN", "HAEMORRHAGE") → 1,200

# 4. Filtrar: solo pares con ≥3 reportes (evitar ruido)

# 5. Batch upsert en Neo4j
for batch in batches_of(counts, 5000):
    session.run("""
        UNWIND $data AS row
        MERGE (d:Drug {name: row.drugname})
        MERGE (ae:AdverseEvent {name: row.pt})
        MERGE (d)-[r:CAUSES]->(ae)
        SET r.report_count = row.report_count
    """, data=batch)
```

**Conceptos clave**:

- **MERGE vs CREATE**: `MERGE` es un "upsert" — crea el nodo si no existe, o lo reutiliza si ya existe. Esto evita duplicados y hace el proceso idempotente.
- **UNWIND**: Toma una lista y la "desenrolla" en filas individuales. Permite insertar miles de registros en una sola query (batch).
- **Batch de 5000**: No insertar todo de golpe (podría quedarse sin memoria). Dividir en lotes de 5000 registros.

### Paso 2: Cargar DailyMed → Neo4j

```python
# 1. Para cada JSON de DailyMed
for json_file in dailymed_dir.glob("*.json"):
    label = json.load(open(json_file))
    
    # 2. Enriquecer nodo Drug con metadata
    session.run("""
        MERGE (d:Drug {name: $name})
        SET d.pharmacologic_class = $pharm_class,
            d.source = "dailymed"
    """, name=label["drug_name"], ...)
    
    # 3. Crear relaciones INTERACTS_WITH
    # Lee la sección drug_interactions del texto
    # Busca nombres de fármacos conocidos en el texto
    # Para cada match: MERGE (d)-[:INTERACTS_WITH]->(other)
    
    # 4. Crear relaciones BELONGS_TO
    # Si pharm_class_epc tiene una clase:
    # MERGE (cat:DrugCategory {name: "Vitamin K Antagonists"})
    # MERGE (d)-[:BELONGS_TO]->(cat)
```

---

## Cómo se usa Neo4j en GraphRAG (queries.py)

El módulo `queries.py` tiene funciones que ejecutan Cypher y devuelven los resultados como diccionarios Python:

```python
def get_drug_adverse_events(drug_name: str, driver, limit: int = 20):
    """Obtiene los top-N efectos adversos de un fármaco."""
    query = """
    MATCH (d:Drug {name: $name})-[r:CAUSES]->(ae:AdverseEvent)
    RETURN ae.name AS event, r.report_count AS count
    ORDER BY r.report_count DESC
    LIMIT $limit
    """
    with driver.session() as session:
        result = session.run(query, name=drug_name, limit=limit)
        return [dict(record) for record in result]
```

Y `format_graph_context()` convierte los resultados en texto legible para el LLM:

```python
def format_graph_context(drug_info, adverse_events, interactions, outcomes):
    """Formatea datos del grafo como texto para el prompt del LLM."""
    context = f"Drug: {drug_info['name']}\n"
    context += f"Category: {drug_info.get('pharmacologic_class', 'Unknown')}\n\n"
    
    context += "Top Adverse Events:\n"
    for ae in adverse_events:
        context += f"  - {ae['event']} ({ae['count']} reports)\n"
    
    context += "\nKnown Interactions:\n"
    for inter in interactions:
        context += f"  - {inter['drug']} ({inter['description']})\n"
    
    return context
```

---

## Neo4j en Docker

Usamos Neo4j en un contenedor Docker por conveniencia:

```yaml
# docker-compose.yml
services:
  neo4j:
    image: neo4j:5-community
    container_name: pharmagraphrag-neo4j
    ports:
      - "7474:7474"    # Neo4j Browser (interfaz web)
      - "7687:7687"    # Protocolo Bolt (conexión desde Python)
    environment:
      NEO4J_AUTH: neo4j/pharmagraphrag
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data    # Persistir datos entre reinicios
```

- **Puerto 7474**: Interfaz web para explorar el grafo visualmente
- **Puerto 7687**: Puerto Bolt para la conexión desde Python (`neo4j` driver)
- **APOC**: Plugin con funciones extra (estadísticas, algoritmos, etc.)
- **Volume**: Los datos del grafo se persisten aunque reinicies el contenedor

### Conexión desde Python

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "pharmagraphrag")
)

with driver.session() as session:
    result = session.run("MATCH (d:Drug) RETURN count(d) AS total")
    print(result.single()["total"])  # 4998
```

---

## Ejercicios para practicar (en Neo4j Browser)

Abre `http://localhost:7474`, conéctate con `neo4j`/`pharmagraphrag`, y prueba:

### Nivel básico
```cypher
-- 1. ¿Cuántos nodos Drug hay?
MATCH (d:Drug) RETURN count(d)

-- 2. ¿Cuáles son los 7 tipos de outcome?
MATCH (o:Outcome) RETURN o.code, o.name

-- 3. Buscar un fármaco por nombre
MATCH (d:Drug {name: "METFORMIN"}) RETURN d
```

### Nivel intermedio
```cypher
-- 4. Top 10 fármacos con más efectos adversos diferentes
MATCH (d:Drug)-[:CAUSES]->(ae:AdverseEvent)
RETURN d.name, count(ae) AS num_effects
ORDER BY num_effects DESC
LIMIT 10

-- 5. ¿Qué categorías farmacológicas tenemos?
MATCH (d:Drug)-[:BELONGS_TO]->(cat:DrugCategory)
RETURN cat.name, count(d) AS drugs
ORDER BY drugs DESC

-- 6. Fármacos con más reportes de muerte
MATCH (d:Drug)-[r:HAS_OUTCOME]->(o:Outcome {code: "DE"})
RETURN d.name, r.report_count
ORDER BY r.report_count DESC
LIMIT 10
```

### Nivel avanzado
```cypher
-- 7. Pares de fármacos que comparten más de 50 efectos adversos
MATCH (d1:Drug)-[:CAUSES]->(ae:AdverseEvent)<-[:CAUSES]-(d2:Drug)
WHERE d1.name < d2.name  -- evitar duplicados (A,B) y (B,A)
WITH d1.name AS drug1, d2.name AS drug2, count(ae) AS shared
WHERE shared > 50
RETURN drug1, drug2, shared
ORDER BY shared DESC
LIMIT 20

-- 8. Camino más corto entre dos fármacos
MATCH path = shortestPath(
  (d1:Drug {name: "ASPIRIN"})-[*..5]-(d2:Drug {name: "METFORMIN"})
)
RETURN path
```
