# Arquitectura y Conceptos: GraphRAG

## ¿Qué problema resuelve este proyecto?

Imagina que eres un médico y quieres saber: *"¿Puedo recetar ibuprofeno a un paciente que ya toma warfarina?"*

**Opción 1: Google** → Resultados genéricos, no basados en datos reales, posibles errores.

**Opción 2: Buscar en la FDA directamente** → Los datos existen (millones de reportes), pero están en CSVs gigantes, sin estructura, y no puedes hacer preguntas en lenguaje natural.

**Opción 3: PharmaGraphRAG** → Combina datos reales de la FDA + etiquetas de medicamentos + IA para darte una respuesta fundamentada, con fuentes, en lenguaje natural.

---

## ¿Qué es RAG?

**RAG = Retrieval-Augmented Generation** (Generación Aumentada por Recuperación)

Un LLM (como ChatGPT, Gemini, Llama) es muy bueno generando texto, pero tiene dos problemas graves:

1. **Alucinaciones**: Se inventa cosas que suenan plausibles pero son falsas
2. **Datos desactualizados**: Solo sabe lo que estaba en su entrenamiento (puede tener meses o años de retraso)

**RAG soluciona esto** dándole al LLM **contexto relevante** antes de que genere la respuesta:

```
SIN RAG:
  Usuario: "¿Warfarin interactúa con aspirin?"
  LLM: "Sí, pueden interactuar" (genérico, sin datos, puede alucinar)

CON RAG:
  Usuario: "¿Warfarin interactúa con aspirin?"
      ↓
  [1. RETRIEVAL] Buscar en nuestras bases de datos:
      - Neo4j: WARFARIN -[INTERACTS_WITH]-> ASPIRIN ✓
      - Neo4j: WARFARIN -[CAUSES]-> HAEMORRHAGE (report_count: 1200)
      - ChromaDB: "Concurrent use of warfarin and aspirin increases
                   the risk of serious hemorrhage..."
      ↓
  [2. AUGMENTED] Pasarle ese contexto al LLM:
      "Basándote SOLO en esta información, responde:"
      + contexto del grafo + contexto del texto
      ↓
  [3. GENERATION] El LLM genera:
      "Sí, warfarin y aspirin interactúan. Según los datos de FAERS,
       hay 1,200 reportes de hemorragia asociados a warfarin.
       La etiqueta del medicamento advierte que 'el uso concurrente
       aumenta el riesgo de hemorragia grave'."
```

### ¿Por qué es mejor?
- **Fundamentado en datos reales** (no se inventa nada)
- **Con fuentes citables** (puedes verificar)
- **Actualizable** (solo tienes que actualizar tus datos, no reentrenar el LLM)

---

## ¿Qué es GraphRAG?

**GraphRAG** es RAG pero usando un **grafo de conocimiento** como una de las fuentes de datos. Esto es lo que hace especial a nuestro proyecto.

### RAG normal (solo vector)
```
Pregunta → Buscar en ChromaDB → Chunks de texto → LLM → Respuesta
```

**Problema**: Solo tienes texto. No sabes las relaciones entre entidades. Si preguntas "¿Qué otros fármacos causan los mismos efectos que aspirin?", tendrías que tener un chunk de texto que diga exactamente eso.

### GraphRAG (grafo + vector)
```
Pregunta → Extraer entidades (ASPIRIN)
         → Buscar en Neo4j: ASPIRIN → [CAUSES] → HEADACHE ← [CAUSES] ← IBUPROFEN
         → Buscar en ChromaDB: "aspirin side effects..."
         → Combinar ambos contextos
         → LLM → Respuesta fundamentada
```

**Ventaja**: El grafo te da **relaciones estructuradas** que serían imposibles de extraer solo de texto. Puedes hacer consultas multi-hop (saltar de un nodo a otro a otro).

---

## Arquitectura de PharmaGraphRAG

```
┌─────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                         │
│                                                          │
│  FDA FAERS (CSV)        DailyMed (API)                  │
│  816K reportes          88 etiquetas de medicamentos     │
│       │                        │                         │
│       ▼                        ▼                         │
│  clean_faers.py         ingest_dailymed.py              │
│  (normalizar,           (descargar, extraer             │
│   deduplicar,            secciones)                     │
│   Parquet)                                              │
└─────────┬──────────────────────┬────────────────────────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│   NEO4J (KG)    │    │   CHROMADB      │
│                 │    │   (Vector Store) │
│ 4,998 Drugs     │    │                 │
│ 6,863 Adverse   │    │ 5,654 chunks    │
│   Events        │    │ 384-dim vectors │
│ 365K CAUSES     │    │ cosine search   │
│ 193 INTERACTS   │    │                 │
└────────┬────────┘    └────────┬────────┘
         │                      │
         │   ┌──────────────┐   │
         └──►│ QUERY ENGINE │◄──┘
             │              │
             │ 1. Extraer   │
             │    entidades │
             │ 2. Buscar    │
             │    en grafo  │
             │ 3. Buscar    │
             │    en vector │
             │ 4. Combinar  │
             │    contextos │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │     LLM      │
             │ (Gemini /    │
             │  Ollama)     │
             │              │
             │ Genera       │
             │ respuesta    │
             │ con fuentes  │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │   INTERFAZ   │
             │              │
             │ FastAPI      │
             │ (endpoints)  │
             │              │
             │ Streamlit    │
             │ (dashboard)  │
             └──────────────┘
```

---

## Los dos tipos de retrieval: ¿por qué necesitamos ambos?

### Neo4j (Knowledge Graph) — Datos estructurados

**Qué almacena**: Relaciones entre entidades con datos cuantitativos.

```
ASPIRIN --[CAUSES {report_count: 500}]--> STOMACH BLEEDING
ASPIRIN --[INTERACTS_WITH]--> WARFARIN
ASPIRIN --[HAS_OUTCOME {report_count: 50}]--> DEATH
```

**Para qué es bueno**:
- Preguntas sobre **relaciones**: "¿Con qué interactúa warfarin?"
- Preguntas de **agregación**: "¿Cuáles son los efectos adversos más reportados?"
- **Traversal multi-hop**: "¿Qué fármacos causan los mismos efectos que aspirin?"
- Datos **cuantitativos**: "¿Cuántos reportes de muerte tiene el metformin?"

**Para qué NO es bueno**:
- Preguntas vagas: "¿Es seguro el ibuprofeno para personas mayores?"
- Contexto detallado: "¿Qué dice la etiqueta sobre la dosis?"
- Búsqueda por significado: "medicamentos para la presión arterial"

### ChromaDB (Vector Store) — Datos no estructurados

**Qué almacena**: Fragmentos de texto de las etiquetas de medicamentos, convertidos en vectores numéricos.

```
"Warfarin acts by inhibiting the synthesis of vitamin K-dependent
 clotting factors, which include Factors II, VII, IX, and X..."
                    ↓ embedding
 [0.12, -0.34, 0.56, ..., 0.78]  (384 números)
```

**Para qué es bueno**:
- Búsqueda por **significado** (no palabras exactas)
- Encontrar **contexto explicativo** (mecanismos, advertencias, dosis)
- Preguntas en **lenguaje natural**
- Proporcionar **texto citable** al LLM

**Para qué NO es bueno**:
- Relaciones exactas entre entidades
- Datos cuantitativos (no sabe que hay 500 reportes)
- Consultas multi-hop

### La combinación: 1 + 1 = 3

| Pregunta | Solo Neo4j | Solo ChromaDB | **GraphRAG (ambos)** |
|----------|-----------|---------------|---------------------|
| "¿Warfarin interactúa con aspirin?" | ✅ Sí, hay relación INTERACTS_WITH | ✅ Sí, el texto lo menciona | ✅✅ Sí, con datos + explicación detallada |
| "¿Cuántos reportes de sangrado tiene warfarin?" | ✅ 1,200 reportes | ❌ No tiene datos numéricos | ✅ 1,200 reportes + contexto sobre el riesgo |
| "¿Es seguro mezclar ibuprofeno con alcohol?" | ❌ No hay nodo "alcohol" | ✅ La etiqueta menciona el riesgo | ✅ Contexto relevante de varias fuentes |
| "¿Qué fármacos causan náuseas?" | ✅ Lista completa | ⚠️ Parcial (solo texto) | ✅ Lista completa + explicaciones |

---

## Flujo de una pregunta (lo que construiremos en la Semana 2)

```
Usuario: "¿Puedo tomar ibuprofeno con warfarina?"

Paso 1 - ENTITY EXTRACTION
  Input:  "¿Puedo tomar ibuprofeno con warfarina?"
  Output: entidades = ["IBUPROFEN", "WARFARIN"]
  Cómo:   El LLM o un NER extrae los nombres de fármacos

Paso 2 - GRAPH RETRIEVAL (Neo4j)
  Para cada entidad, buscar en el grafo:
  - get_drug_info("IBUPROFEN")     → Drug node con propiedades
  - get_drug_info("WARFARIN")      → Drug node con propiedades
  - get_drug_interactions("WARFARIN") → INTERACTS_WITH IBUPROFEN? (buscar)
  - get_drug_adverse_events("IBUPROFEN") → top N efectos adversos
  - get_drug_adverse_events("WARFARIN")  → top N efectos adversos
  Resultado: texto estructurado con relaciones

Paso 3 - VECTOR RETRIEVAL (ChromaDB)
  search("ibuprofen warfarin interaction risk", n_results=5)
  Resultado: 5 chunks de texto relevantes de las etiquetas

Paso 4 - CONTEXT MERGING
  Combinar el contexto del grafo + contexto del vector en un
  prompt coherente, eliminando duplicados

Paso 5 - LLM GENERATION
  Prompt: "Basándote SOLO en este contexto, responde:"
  + contexto combinado
  + pregunta del usuario
  
  LLM responde con: explicación + datos + fuentes
```

---

## Conceptos clave para entender bien

### 1. Grounding (Fundamentación)
El principio de que la respuesta del LLM debe estar **basada en datos verificables**, no en su conocimiento general. Si le dices "responde SOLO basándote en este contexto" y le das el contexto correcto, el LLM no alucina (o alucina mucho menos).

### 2. Retrieval vs Generation
- **Retrieval** = buscar información relevante (no genera texto nuevo)
- **Generation** = crear texto nuevo basado en la información encontrada
- RAG = primero busco, luego genero

### 3. Embeddings como "fingerprints" de significado
Cada texto se convierte en una "huella dactilar" numérica. Textos con significado similar tienen huellas similares. Esto permite buscar por concepto, no por palabras exactas.

### 4. Knowledge Graph como "mapa de relaciones"
El grafo es como un mapa donde los nodos son entidades (fármacos, efectos) y las flechas son relaciones (causa, interactúa). Puedes navegar el mapa para descubrir conexiones que no serían obvias solo leyendo texto.

---

## ¿Dónde encaja cada tecnología?

| Tecnología | Rol | Analogía |
|-----------|-----|---------|
| **Neo4j** | Almacena relaciones entre entidades | El mapa de una ciudad (calles conectan lugares) |
| **ChromaDB** | Almacena texto buscable por significado | Una biblioteca con un bibliotecario que entiende lo que buscas |
| **sentence-transformers** | Convierte texto → números (embeddings) | Un traductor que convierte idioma humano → idioma de máquina |
| **Gemini/Ollama** | Genera respuestas en lenguaje natural | Un experto que lee toda la información y te da un resumen |
| **FastAPI** | Expone todo como API web | La recepcionista que recibe tu pregunta y te devuelve la respuesta |
| **Streamlit** | Interfaz visual para el usuario | La pantalla bonita donde escribes tu pregunta |
| **Docker** | Empaqueta todo para que funcione en cualquier PC | Una caja que contiene todo lo necesario para que funcione |
