# PharmaGraphRAG — Plan de Proyecto

> GraphRAG para consulta de interacciones farmacológicas y efectos adversos.
> Duración estimada: 3 semanas

---

## Concepto

Sistema de pregunta-respuesta que combina un knowledge graph de fármacos con RAG para responder preguntas sobre interacciones farmacológicas y efectos adversos. El usuario pregunta en lenguaje natural y obtiene respuestas fundamentadas en datos de la FDA.

**Preguntas ejemplo:**
- "¿Qué efectos adversos tiene el ibuprofeno?"
- "¿Qué fármacos interactúan con metformina?"
- "¿Hay reportes de eventos adversos cuando se combina warfarina con aspirina?"

---

## Datos

| Fuente | Qué contiene | Formato | URL |
|--------|-------------|---------|-----|
| FDA FAERS | Reportes de eventos adversos (fármacos, reacciones, outcomes) | CSV (quarterly) | https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| DailyMed | Etiquetas de fármacos (interacciones, contraindicaciones, warnings) | XML/API | https://dailymed.nlm.nih.gov/dailymed/ |

**Scope de datos**: Empezar con 1-2 trimestres de FAERS + ~100-500 fármacos más comunes de DailyMed. Expandir si hay tiempo.

---

## Arquitectura

```
┌─────────────────────────────────────────────────┐
│                  DATA LAYER                      │
│  FDA FAERS (CSV) ──→ Limpieza ──→ Parquet       │
│  DailyMed (API)  ──→ Extracción ──→ Textos      │
└────────────┬────────────────────────┬────────────┘
             │                        │
             ▼                        ▼
┌────────────────────┐  ┌────────────────────────┐
│  KNOWLEDGE GRAPH   │  │     VECTOR STORE       │
│  Neo4j             │  │     ChromaDB           │
│                    │  │                        │
│  Drug ──causes──→  │  │  Embeddings de drug    │
│    AdverseEvent    │  │  labels (chunks)       │
│  Drug ──interacts  │  │                        │
│    ──→ Drug        │  │  Embedding model:      │
│  Drug ──belongs──→ │  │  sentence-transformers │
│    Category        │  │                        │
└────────┬───────────┘  └───────────┬────────────┘
         │                          │
         │     ┌────────────┐       │
         └────→│ QUERY      │←──────┘
               │ ENGINE     │
               │            │
               │ 1. Parse question         │
               │ 2. Entity extraction      │
               │ 3. Graph traversal        │
               │ 4. Vector similarity      │
               │ 5. Merge context          │
               │ 6. LLM answer generation  │
               └──────┬─────┘
                      │
                      ▼
         ┌────────────────────┐
         │   SERVING LAYER    │
         │   FastAPI + Streamlit │
         └────────────────────┘
```

---

## Tech Stack

| Componente | Tecnología | Razón |
|------------|-----------|-------|
| Lenguaje | Python 3.11+ | Stack principal |
| Knowledge Graph | Neo4j (Docker) | Nuevo skill, demandado, visual |
| Vector Store | ChromaDB | Ligero, embebido, sin infra extra |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Gratuito, rápido, buena calidad |
| LLM (principal) | Google Gemini API (free tier) | Gratis, buena calidad |
| LLM (backup) | Ollama + Llama 3 / Mistral | 100% local, sin dependencia |
| API | FastAPI | Consistencia con DengueMLOps |
| UI | Streamlit | Rápido para demo, ya conoces |
| Contenedores | Docker Compose (Neo4j + app + Ollama) | Todo reproducible |
| CI/CD | GitHub Actions | Consistencia con DengueMLOps |
| Testing | pytest | Tests de pipeline + API |

---

## Fases y Timeline

### Semana 1: Datos + Knowledge Graph

**Día 1-2: Setup + ingestión de datos**
- [ ] Crear repo GitHub con estructura del proyecto
- [ ] Descargar FAERS (1-2 trimestres recientes)
- [ ] Script de descarga/limpieza de FAERS → Parquet
- [ ] Script de ingestión desde DailyMed API (top 100-500 fármacos)

**Día 3-4: Construcción del Knowledge Graph**
- [ ] Levantar Neo4j en Docker
- [ ] Definir schema del grafo: nodos (Drug, AdverseEvent, DrugCategory) y relaciones (CAUSES, INTERACTS_WITH, BELONGS_TO)
- [ ] Script de carga de FAERS → Neo4j (Cypher)
- [ ] Script de carga de interacciones DailyMed → Neo4j
- [ ] Queries Cypher de validación

**Día 5: Vector store**
- [ ] Chunking de textos de drug labels (DailyMed)
- [ ] Generar embeddings con sentence-transformers
- [ ] Cargar en ChromaDB
- [ ] Tests de similarity search

### Semana 2: GraphRAG Engine + LLM

**Día 1-2: Query engine**
- [ ] Entity extraction de la pregunta del usuario (NER básico o regex + fuzzy matching sobre fármacos conocidos)
- [ ] Graph traversal: dado un fármaco, obtener entidades relacionadas de Neo4j
- [ ] Vector search: obtener chunks relevantes de ChromaDB
- [ ] Context merging: combinar info del grafo + chunks en un prompt

**Día 3-4: Integración LLM**
- [ ] Integrar Gemini API (google-generativeai SDK)
- [ ] Integrar Ollama como fallback
- [ ] Prompt engineering para respuesta fundamentada (con citas de los datos)
- [ ] Evaluación básica: 10-20 preguntas de test con respuestas esperadas

**Día 5: FastAPI**
- [ ] Endpoint POST /query con pregunta → respuesta + fuentes
- [ ] Endpoint GET /drug/{name} con info del grafo
- [ ] Health check + docs OpenAPI

### Semana 3: UI + Docker + Polish

**Día 1-2: Streamlit**
- [ ] Interfaz de chat para preguntas
- [ ] Visualización del subgrafo relevante (pyvis o streamlit-agraph)
- [ ] Mostrar fuentes/evidencia usada para la respuesta

**Día 3-4: Docker + testing**
- [ ] Docker Compose: Neo4j + ChromaDB + FastAPI + Streamlit
- [ ] Tests: pipeline de datos, query engine, API
- [ ] README completo con screenshots, arquitectura, instrucciones de setup

**Día 5: CI/CD + lanzamiento**
- [ ] GitHub Actions: lint + test + build
- [ ] Post de LinkedIn sobre el proyecto
- [ ] Actualizar CV si procede

---

## Entregables

1. **Repo GitHub público** con código limpio, README con screenshots, y CI verde
2. **Demo funcional** en local con Docker Compose
3. **Knowledge Graph** con datos reales de la FDA
4. **Query engine** que combina graph traversal + vector search + LLM
5. **API REST** documentada (FastAPI)
6. **Dashboard** interactivo (Streamlit)

---

## Diferenciador vs RAG estándar

La mayoría de proyectos RAG de portfolio son "PDF chatbot con LangChain". Este se diferencia porque:

1. **Knowledge Graph real** — no solo embeddings, sino estructura de relaciones
2. **Dominio relevante** — farmacología con datos de la FDA, no un toy example
3. **Background del autor** — experiencia previa en KG (BASF/NTT DATA) hace la elección técnica creíble
4. **Dual retrieval** — graph traversal + vector search, no solo uno
5. **Stack moderno** — Neo4j + ChromaDB + Gemini/Ollama + FastAPI

---

## Skills que demuestra

- [x] GenAI / LLM integration (gap #1 del mercado)
- [x] RAG architecture
- [x] Knowledge graphs (refuerza experiencia previa)
- [x] Neo4j (nuevo skill, demandado)
- [x] Vector databases
- [x] Python, FastAPI, Docker (refuerza stack)
- [x] Prompt engineering
