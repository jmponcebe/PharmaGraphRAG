# Embeddings y Vector Search

## El problema: las máquinas no entienden texto

Cuando tú lees "warfarin interacciones peligrosas" y "anticoagulant drug risks", **entiendes** que hablan de lo mismo. Las máquinas no — para ellas son cadenas de caracteres completamente diferentes.

¿Cómo hacemos que una máquina "entienda" el significado? **Embeddings**.

---

## ¿Qué es un embedding?

Un **embedding** es una lista de números (un vector) que captura el **significado** de un texto.

```
"warfarin causes bleeding"  →  [0.12, -0.34, 0.56, ..., 0.78]   (384 números)
"anticoagulant hemorrhage"  →  [0.11, -0.33, 0.55, ..., 0.77]   (384 números, ¡similares!)
"pizza is delicious"        →  [0.89, 0.45, -0.12, ..., -0.56]  (384 números, ¡diferentes!)
```

### ¿Por qué funciona?

Un modelo de embedding (como all-MiniLM-L6-v2) fue **entrenado** con millones de pares de textos:
- "El perro corre" ↔ "The dog runs" → deben tener vectores cercanos
- "El perro corre" ↔ "La bolsa sube" → deben tener vectores lejanos

Después de ver millones de ejemplos, el modelo "aprende" a representar significado como números. No entiende realmente, pero las representaciones son tan buenas que **funcionan** como si entendiera.

### Analogía: coordenadas GPS

Piensa en las coordenadas GPS:
- Madrid: (40.42, -3.70)
- Barcelona: (41.39, 2.17)
- Tokyo: (35.68, 139.69)

Madrid y Barcelona están geográficamente cerca → sus coordenadas son similares.
Madrid y Tokyo están lejos → sus coordenadas son muy diferentes.

Los embeddings hacen lo mismo pero en un **espacio de significado** de 384 dimensiones (en vez de 2 dimensiones geográficas). Textos con significado similar tienen coordenadas cercanas en este espacio.

---

## Dimensionalidad

### ¿Por qué 384 dimensiones?

Cada dimensión captura un **aspecto del significado**. No son interpretables individualmente (no es que "dimensión 73 = medicamentos"), pero combinadas capturan matices:

| Dimensiones | Modelo | Calidad | Velocidad | Tamaño |
|-------------|--------|---------|-----------|--------|
| 384 | all-MiniLM-L6-v2 | Buena | Muy rápido | 80MB |
| 768 | all-mpnet-base-v2 | Mejor | Más lento | 420MB |
| 1536 | OpenAI ada-002 | Excelente | API (lenta) | N/A (cloud) |
| 3072 | OpenAI text-embedding-3-large | Mejor | API (lenta) | N/A (cloud) |

Nosotros usamos 384 dim porque:
- Es **gratuito** y **local** (no dependemos de APIs)
- Es **rápido** (cabe en CPU, sin GPU)
- Es **suficiente** para nuestro caso de uso (medicamentos, no poesía abstracta)

### ¿Cómo se ve un embedding real?

```python
from pharmagraphrag.vectorstore.embedder import embed_single

vec = embed_single("warfarin bleeding risk")
print(f"Dimensiones: {len(vec)}")  # 384
print(f"Primeros 10 valores: {vec[:10]}")
# [-0.0284, 0.0651, -0.0023, 0.0412, -0.0891, 0.0334, ...]
```

Son números decimales pequeños (entre -1 y 1 aprox.) que no tienen significado individual.

---

## Similitud: ¿cómo se comparan embeddings?

### Similitud coseno

La métrica más usada para comparar embeddings es la **similitud coseno**: mide el ángulo entre dos vectores.

```
                    ↑ dim 2
                    │
              vec_A │  /  vec_B
                    │ /θ
                    │/____→ dim 1
```

- **θ = 0°** → vectores idénticos → coseno = 1.0
- **θ = 90°** → completamente no relacionados → coseno = 0.0
- **θ = 180°** → significados opuestos → coseno = -1.0

### Distancia coseno (lo que usa ChromaDB)

ChromaDB usa **distancia coseno = 1 - similitud coseno**:

| Distancia | Significado | Ejemplo |
|-----------|-------------|---------|
| 0.0 | Idéntico | Misma frase |
| 0.1-0.3 | Muy similar | "warfarin bleeding" vs "anticoagulant hemorrhage" |
| 0.3-0.5 | Relacionado | "warfarin bleeding" vs "blood thinning drugs" |
| 0.5-0.7 | Vagamente relacionado | "warfarin bleeding" vs "hospital treatment" |
| 0.7-1.0 | No relacionado | "warfarin bleeding" vs "pizza recipe" |

### Ejemplo real de nuestro proyecto

```
Query: "What are the side effects of warfarin?"

Resultado 1: WARFARIN / adverse_reactions  → distancia 0.27 (¡muy relevante!)
Resultado 2: WARFARIN / adverse_reactions  → distancia 0.38 (relevante)
Resultado 3: WARFARIN / adverse_reactions  → distancia 0.40 (relevante)
```

La búsqueda encontró correctamente las secciones de adverse_reactions de warfarin.

---

## sentence-transformers: El modelo de embedding

### ¿Qué es sentence-transformers?

Es una librería Python que proporciona modelos pre-entrenados para generar embeddings de texto. Desarrollada por la Universidad de Darmstadt.

### all-MiniLM-L6-v2: Nuestro modelo

| Propiedad | Valor |
|-----------|-------|
| Nombre | all-MiniLM-L6-v2 |
| Dimensión | 384 |
| Tamaño | ~80MB |
| Velocidad | ~14,000 frases/segundo (CPU) |
| Entrenamiento | 1 billón de pares de frases |
| Longitud máxima | 256 tokens (~200 palabras) |
| Licencia | Apache 2.0 (gratuito) |

### ¿Cómo funciona internamente?

```
Input: "Warfarin drug interactions"
         ↓
[Tokenización]
  "warfarin" → token_id 42195
  "drug"     → token_id 3672
  "interactions" → token_id 11467
         ↓
[Modelo Transformer (6 capas)]
  Cada token pasa por 6 capas de atención
  que capturan relaciones entre palabras
         ↓
[Mean Pooling]
  Promedia las representaciones de todos los tokens
  en un solo vector de 384 dimensiones
         ↓
[Normalización L2]
  Escala el vector para que tenga longitud 1
  (facilita el cálculo de coseno)
         ↓
Output: [0.12, -0.34, 0.56, ..., 0.78]  (384 floats)
```

### embedder.py — Nuestro código

```python
# Cargar modelo (se cachea, solo se carga una vez)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB download

# Generar embeddings para múltiples textos
embeddings = model.encode(
    ["warfarin interactions", "aspirin side effects"],
    normalize_embeddings=True,    # L2 normalize
    show_progress_bar=True,       # Barra de progreso
    batch_size=64,                # Procesar 64 textos a la vez
)
# Resultado: numpy array de shape (2, 384)
```

---

## Text Chunking: ¿Por qué dividir los textos?

### El problema

La sección "drug_interactions" de warfarin tiene **6,477 caracteres** (~1,500 palabras). Pero:

1. **El modelo tiene límite**: all-MiniLM-L6-v2 solo procesa ~256 tokens (~200 palabras). Texto más largo se **trunca** (se pierde información).

2. **Precisión de búsqueda**: Si embedes todo el texto de 6,477 chars como un solo vector, ese vector representa el "promedio" de todo el contenido. Si buscas algo específico (ej: "warfarin aspirin interaction"), el vector "promedio" no será tan cercano como un chunk que hable específicamente de aspirina.

3. **Contexto para el LLM**: Es mejor darle al LLM 3-5 chunks relevantes que un texto gigante donde solo el 10% es relevante.

### La solución: Chunking con overlap

```
Texto original (6477 chars):
┌──────────────────────────────────────────────────────────────┐
│ Warfarin interacts with many drugs. Antibiotics can increase │
│ the effect of warfarin. NSAIDs like aspirin increase bleed-  │
│ ing risk. CYP2C9 inhibitors may increase warfarin levels... │
│ Avoid concurrent use with herbal supplements like ginkgo...  │
│ Vitamin K-rich foods can decrease warfarin effectiveness...  │
└──────────────────────────────────────────────────────────────┘

Después de chunk_text(chunk_size=1000, chunk_overlap=200):

Chunk 1 (1000 chars):
┌─────────────────────────────────────────────────┐
│ Warfarin interacts with many drugs. Antibiotics  │
│ can increase the effect of warfarin. NSAIDs like │
│ aspirin increase bleeding risk...                │
│                              ┌──── overlap ─────┐│
└──────────────────────────────┤                  │┘
Chunk 2 (1000 chars):          │                  │
┌──────────────────────────────┤                  │
│...aspirin increase bleeding  │risk. CYP2C9      ││
│ inhibitors may increase warf │arin levels...    ││
│ Avoid concurrent use with he │rbal supplements  ││
│                              └──── overlap ─────┘│
│                              ┌──── overlap ─────┐│
└──────────────────────────────┤                  │┘
Chunk 3 (800 chars):           │                  │
┌──────────────────────────────┤                  │
│...herbal supplements like gin│kgo. Vitamin K-   ││
│ rich foods can decrease warfa│rin effectiveness  ││
└──────────────────────────────└──────────────────┘┘
```

### ¿Por qué overlap?

Sin overlap, si buscas "aspirin bleeding" y la frase relevante está cortada entre el final del chunk 1 y el inicio del chunk 2, **ninguno de los dos chunks** la contendría completa.

Con overlap de 200 chars, esa información aparece en **ambos chunks**, así que la búsqueda la encuentra.

### chunker.py — Nuestro código

```python
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    # 1. Limpiar texto (quitar números de sección, colapsar espacios)
    text = _clean_text(text)
    
    # 2. Si el texto cabe en un chunk, devolverlo entero
    if len(text) <= chunk_size:
        return [text]
    
    # 3. Dividir con overlap, intentando cortar en fin de frase
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Intentar cortar en un punto (fin de oración)
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            end = start + last_period + 2
            chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - chunk_overlap  # Retroceder 200 chars para overlap
    
    return chunks
```

### Metadata en cada chunk

Cada chunk lleva metadata que nos permite filtrar y citar fuentes:

```python
TextChunk(
    text="Drug: WARFARIN | Section: Drug Interactions\nWarfarin interacts...",
    drug_name="WARFARIN",
    section="drug_interactions",
    chunk_index=0,
    metadata={
        "drug_name": "WARFARIN",
        "section": "drug_interactions",
        "chunk_index": "0",
        "generic_names": "WARFARIN SODIUM",
        "brand_names": "Coumadin",
        "route": "ORAL",
    }
)
```

La metadata permite hacer búsquedas filtradas:
```python
# Buscar solo en chunks de WARFARIN
results = search_by_drug("bleeding risk", drug_name="WARFARIN")
```

---

## ChromaDB: La base de datos de vectores

### ¿Qué es ChromaDB?

ChromaDB es una base de datos especializada en almacenar y buscar **vectores** (embeddings). Es el equivalente de Neo4j para datos vectoriales.

| Propiedad | Valor |
|-----------|-------|
| Tipo | Base de datos vectorial, embedded |
| Storage | SQLite + HNSW index |
| Distancia | Coseno (por defecto) |
| API | Python nativa |
| Persistencia | Directorio en disco (`data/chroma/`) |

### ¿Por qué ChromaDB?

| Opción | Ventajas | Desventajas |
|--------|---------|-------------|
| **ChromaDB** ✅ | Gratis, embedded (no necesita servidor), fácil | Menos escalable |
| Pinecone | Escalable, managed | Pago, depende de cloud |
| Qdrant | Rápido, features avanzadas | Requiere servidor separado |
| FAISS | Muy rápido | Solo índice, sin metadata filtering |
| pgvector | PostgreSQL native | Requiere PostgreSQL |

Para un proyecto portfolio, ChromaDB es ideal: **simple, gratis, y suficiente**.

### Estructura interna

```
data/chroma/
├── chroma.sqlite3          # Metadata SQLite (documentos, IDs, metadata)
└── {collection_id}/
    ├── data_level0.bin     # Vectores (embeddings)
    ├── header.bin          # Cabecera del índice HNSW
    ├── index_metadata.pickle
    ├── length.bin
    └── link_lists.bin      # Estructura del grafo HNSW
```

### HNSW: Cómo busca rápido

**HNSW = Hierarchical Navigable Small World** — Un algoritmo que permite buscar el vector más cercano sin comparar con TODOS los vectores.

Imagina que tienes 5,654 chunks. Sin índice, cada búsqueda compararía con los 5,654 → lento.

HNSW crea un grafo de navegación en múltiples niveles:

```
Nivel 3 (pocos nodos):     A ─── F
                           │     │
Nivel 2 (más nodos):      A ─ C ─ F ─ H
                           │   │   │   │
Nivel 1 (más nodos):     A B C D E F G H I J
                           │ │ │ │ │ │ │ │ │ │
Nivel 0 (todos):          a b c d e f g h i j k l m n o ...
```

Para buscar, empiezas arriba (nivel 3, pocos nodos, saltos grandes) y bajas (nivel 0, todos los nodos, búsqueda fina). Esto reduce la búsqueda de O(n) a O(log n).

### store.py — Nuestro código

```python
# 1. Crear cliente persistente
client = chromadb.PersistentClient(path="data/chroma/")

# 2. Obtener o crear colección
collection = client.get_or_create_collection(
    name="drug_labels",
    metadata={"hnsw:space": "cosine"}  # Usar distancia coseno
)

# 3. Añadir chunks con embeddings
collection.upsert(
    ids=["WARFARIN__drug_interactions__0", ...],
    embeddings=[[0.12, -0.34, ...], ...],     # Vectores 384-dim
    documents=["Drug: WARFARIN | ...", ...],    # Textos originales
    metadatas=[{"drug_name": "WARFARIN", ...}]  # Metadata para filtrado
)

# 4. Buscar por similitud
results = collection.query(
    query_embeddings=[[0.15, -0.28, ...]],  # Vector de la pregunta
    n_results=5,
    where={"drug_name": "WARFARIN"},  # Filtro opcional por metadata
    include=["documents", "metadatas", "distances"]
)
```

---

## El flujo completo: De pregunta a resultados

```
Usuario: "¿Warfarin causa sangrado?"
         │
         ▼
[embed_single("warfarin causa sangrado")]
         │
         ▼
Vector de query: [0.15, -0.28, 0.61, ...]  (384 floats)
         │
         ▼
[ChromaDB.query(query_embedding, n_results=5)]
         │
         ▼
HNSW busca los 5 vectores más cercanos
         │
         ▼
Resultados:
┌──────────────────────────────────────────────────────┐
│ 1. WARFARIN/adverse_reactions (dist: 0.27)           │
│    "Bleeding is the main risk of warfarin therapy.   │
│     Hemorrhage can occur at any site..."             │
│                                                      │
│ 2. WARFARIN/boxed_warning (dist: 0.35)               │
│    "WARNING: BLEEDING RISK. Warfarin can cause major │
│     or fatal bleeding..."                            │
│                                                      │
│ 3. WARFARIN/warnings (dist: 0.42)                    │
│    "Hemorrhage: Monitor INR regularly. Risk factors  │
│     include hypertension, cerebrovascular disease..."│
└──────────────────────────────────────────────────────┘
```

Este contexto se le pasará al LLM junto con el contexto del grafo para generar la respuesta final.

---

## Métricas de nuestro Vector Store

| Métrica | Valor |
|---------|-------|
| Chunks totales | 5,654 |
| Drugs indexados | 88 |
| Secciones de etiqueta | 12 |
| Dimensión embedding | 384 |
| Modelo | all-MiniLM-L6-v2 |
| Tamaño chunk | 1000 chars |
| Overlap | 200 chars |
| Distancia | Coseno |
| Tiempo de carga | ~2.5 minutos |
| Tiempo de búsqueda | ~50ms por query |

---

## Comparación con búsqueda tradicional

| | Búsqueda por keywords (LIKE '%warfarin%') | Búsqueda vectorial |
|---|---|---|
| "warfarin bleeding" | ✅ Encuentra textos con "warfarin" y "bleeding" | ✅ Encuentra textos con esas palabras |
| "anticoagulant hemorrhage risk" | ❌ No hay match (no tiene "warfarin") | ✅ Entiende que es lo mismo |
| "blood thinner dangers" | ❌ No menciona warfarin | ✅ Semánticamente relacionado |
| Velocidad (5K docs) | Rápido | Rápido |
| Velocidad (5M docs) | Lento (escaneo completo) | Rápido (HNSW index) |
| Configuración | Simple | Necesita modelo de embedding |

La búsqueda vectorial es **estrictamente superior** para textos en lenguaje natural, pero requiere un modelo de embedding y más recursos.
