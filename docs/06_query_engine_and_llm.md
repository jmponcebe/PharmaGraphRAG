# Query Engine y LLM: El Cerebro de PharmaGraphRAG

## Resumen

En el documento anterior construimos las bases de datos (Neo4j y ChromaDB). Ahora conectamos todo con un **motor de consultas** que extrae entidades, recupera contexto de ambas fuentes, y genera respuestas con un LLM.

Este es el flujo completo:

```
Pregunta del usuario
    |
    v
[1] Entity Extraction  -->  "IBUPROFEN", "WARFARIN"
    |
    v
[2] Graph Retrieval    -->  Relaciones de Neo4j
    |
    v
[3] Vector Retrieval   -->  Chunks de ChromaDB
    |
    v
[4] Prompt Assembly    -->  SYSTEM_PROMPT + contexto + pregunta
    |
    v
[5] LLM Generation    -->  Respuesta fundamentada
```

---

## 1. Entity Extraction: Identificar farmacos en texto libre

**Archivo**: `src/pharmagraphrag/engine/entity_extractor.py`

### El problema

El usuario escribe: *"What are the side effects of ibuprofen and warfarin?"*

Necesitamos extraer `["IBUPROFEN", "WARFARIN"]` de esa cadena de texto.

### La solucion: Exact match + Fuzzy match

En vez de usar un modelo NER costoso, usamos una estrategia de dos fases:

#### Fase 1: Exact substring match

Recorremos la lista de farmacos conocidos y comprobamos si aparecen como subcadena en la pregunta (en mayusculas):

```python
def _exact_match(query_upper: str, known_drugs: set[str]) -> list[str]:
    """Busca nombres exactos de farmacos en la pregunta."""
    found = []
    for drug in known_drugs:
        if len(drug) >= MIN_DRUG_NAME_LENGTH and drug in query_upper:
            # Verificar que no es un stop word
            if drug not in STOP_WORDS:
                found.append(drug)
    return found
```

**Ventajas**: Rapido, sin dependencias externas, sin falsos positivos.
**Limitaciones**: No detecta errores tipograficos ("ibuprofn").

#### Fase 2: Fuzzy matching con rapidfuzz

Si el exact match no encuentra nada, usamos **rapidfuzz** (libreria en Rust, muy rapida) para matching difuso:

```python
from rapidfuzz import fuzz, process

def _fuzzy_match(query_upper, known_drugs, threshold=80):
    """Matching difuso para capturar errores tipograficos."""
    words = query_upper.split()
    for word in words:
        if len(word) < MIN_DRUG_NAME_LENGTH:
            continue
        # Buscar el farmaco mas parecido
        result = process.extractOne(
            word, known_drugs,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        if result:
            drug_name, score, _ = result
            # "IBUPROFN" -> "IBUPROFEN" (score: 89)
            found.append(drug_name)
```

**FUZZY_THRESHOLD = 80**: Un score de 80/100 significa que el texto debe ser al menos 80% similar. Esto captura errores tipograficos pero evita falsos positivos.

### Catalogo de farmacos conocidos

Los farmacos se cargan de dos fuentes, con cache:

1. **Disco** (`data/raw/dailymed/_summary.json`): Los 88 farmacos descargados de DailyMed
2. **Neo4j** (opcional): Los 4,998 farmacos del knowledge graph

```python
@lru_cache(maxsize=1)
def get_known_drugs(use_neo4j: bool = True) -> set[str]:
    """Carga el catalogo de farmacos conocidos (con cache)."""
    drugs = _load_known_drugs_from_disk()  # Siempre disponible
    if use_neo4j:
        drugs |= _load_known_drugs_from_neo4j()  # Si Neo4j esta activo
    return drugs
```

El `@lru_cache` asegura que solo se carga una vez por sesion.

### Resultado: ExtractedEntities

```python
@dataclass
class ExtractedEntities:
    """Entidades extraidas de una pregunta."""
    drugs: list[str] = field(default_factory=list)
    adverse_events: list[str] = field(default_factory=list)
    raw_query: str = ""
```

---

## 2. Dual Retrieval: Buscar en grafo y vectores

**Archivo**: `src/pharmagraphrag/engine/retriever.py`

### Por que "dual"?

Cada fuente aporta algo diferente:

| Fuente | Aporta | Ejemplo |
|--------|--------|---------|
| **Neo4j** (grafo) | Relaciones estructuradas, datos cuantitativos | "IBUPROFEN causa NAUSEA (500 reportes)" |
| **ChromaDB** (vectores) | Texto explicativo, contexto clinico | "Ibuprofen should be avoided in patients with..." |

### Graph Retrieval

Para cada farmaco extraido, consultamos Neo4j:

```python
def _retrieve_graph(drugs: list[str]) -> tuple[str, dict, list[str]]:
    """Recupera contexto del knowledge graph para cada farmaco."""
    from pharmagraphrag.graph.queries import get_drug_full_context

    for drug in drugs:
        ctx = get_drug_full_context(drug)
        # ctx contiene:
        #   drug_info:      propiedades del nodo Drug
        #   adverse_events: top-N efectos adversos con report_count
        #   interactions:   farmacos que interactuan
        #   outcomes:       resultados (muerte, hospitalizacion, etc.)
```

El resultado es texto estructurado tipo:

```
Drug: IBUPROFEN
  Adverse events (top 10):
    - NAUSEA (report_count: 500)
    - HEADACHE (report_count: 350)
    ...
  Interactions:
    - WARFARIN: concurrent use may increase bleeding risk
  Outcomes:
    - HOSPITALIZATION (report_count: 120)
```

### Vector Retrieval

Busqueda semantica en ChromaDB, filtrada por farmacos:

```python
def _retrieve_vector(drugs, query, n_results=5, max_chars=4000):
    """Recupera chunks de texto relevantes de ChromaDB."""
    from pharmagraphrag.vectorstore.store import search

    results = search(
        query=query,
        n_results=n_results,
        drug_names=drugs if drugs else None,  # Filtro por farmaco
    )
    # Truncar a max_chars para no exceder el contexto del LLM
```

### RetrievedContext

```python
@dataclass
class RetrievedContext:
    """Contexto combinado de ambas fuentes."""
    graph_context: str = ""       # Texto estructurado del grafo
    vector_context: str = ""      # Texto de los chunks
    graph_raw: dict = field(...)  # Datos crudos del grafo
    vector_raw: list = field(...) # Chunks crudos
    drugs_found: list = field(...)# Farmacos encontrados en el grafo
```

---

## 3. Prompt Assembly: Preparar el prompt para el LLM

**Archivo**: `src/pharmagraphrag/engine/query_engine.py`

### System Prompt

El system prompt define el comportamiento del LLM:

```python
SYSTEM_PROMPT = (
    "You are a pharmaceutical knowledge assistant. Answer the user's question "
    "about drug interactions and adverse events based ONLY on the provided "
    "context.\n\n"
    "Rules:\n"
    "1. Be factual - cite specific drugs, adverse events, and report counts.\n"
    "2. If the context does not contain enough information, say so explicitly.\n"
    "3. Organise your answer with bullet points or numbered lists.\n"
    "4. Mention the data source (FAERS reports, DailyMed label) when relevant.\n"
)
```

**Clave**: "based ONLY on the provided context". Esto es **grounding** -- el LLM no debe inventar informacion.

### User Prompt

Combina el contexto recuperado con la pregunta:

```python
CONTEXT_TEMPLATE = (
    "GRAPH CONTEXT (structured relationships from the knowledge graph):\n"
    "{graph_context}\n\n"
    "TEXT CONTEXT (from drug labels via semantic search):\n"
    "{text_context}\n"
)

USER_TEMPLATE = "USER QUESTION: {question}\n"
```

### process_query: El orquestador

```python
def process_query(question, *, use_graph=True, use_vector=True, ...):
    """Procesa una pregunta completa por el pipeline GraphRAG."""

    # 1. Extraer entidades
    entities = extract_entities(question)

    # 2. Recuperar contexto dual
    context = retrieve_context(
        drugs=entities.drugs,
        query=question,
        use_graph=use_graph,
        use_vector=use_vector,
    )

    # 3. Ensamblar prompt
    system_prompt = SYSTEM_PROMPT
    user_prompt = _build_user_prompt(question, context)

    return QueryResult(
        question=question,
        entities=entities,
        context=context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
```

---

## 4. LLM Client: Gemini + Ollama + Fallback

**Archivo**: `src/pharmagraphrag/llm/client.py`

### Arquitectura del cliente

```
generate_answer(system_prompt, user_prompt)
    |
    +-- provider == "gemini"?
    |   +-- SI --> _generate_gemini() --> OK? --> return
    |   |                                  NO? --> _generate_ollama() (fallback)
    |   +-- NO --> _generate_ollama() --> OK? --> return
    |                                      NO? --> return error
```

### Gemini (google-genai SDK)

```python
def _generate_gemini(system_prompt, user_prompt, model=None, api_key=None):
    """Llama a Google Gemini via el SDK google-genai."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,         # Baja: respuestas mas deterministas
            max_output_tokens=2048,  # Limite de longitud
        ),
    )
    return LLMResponse(text=response.text, model=model, provider="gemini")
```

**Nota importante**: Usamos `google-genai` (>= 1.64.0), NO `google-generativeai` que esta deprecated.

**Temperature = 0.3**: Valor bajo para respuestas consistentes y factuales (no creativas).

### Ollama (local)

```python
def _generate_ollama(system_prompt, user_prompt, model=None, base_url=None):
    """Llama a un modelo local via Ollama."""
    import ollama as ollama_sdk

    client = ollama_sdk.Client(host=base_url)

    response = client.chat(
        model=model or "llama3:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.3},
    )
    return LLMResponse(text=response["message"]["content"], ...)
```

### Cadena de fallback

```python
def generate_answer(system_prompt, user_prompt, *, provider=None, model=None):
    """Genera una respuesta con el proveedor configurado + fallback."""
    provider = provider or settings.llm_provider  # "gemini" o "ollama"

    if provider == "gemini":
        response = _generate_gemini(system_prompt, user_prompt)
        if not response.ok:
            logger.warning("Gemini fallo, intentando Ollama...")
            response = _generate_ollama(system_prompt, user_prompt)
    else:
        response = _generate_ollama(system_prompt, user_prompt)

    return response
```

Si Gemini falla (API key invalida, rate limit, error de red), automaticamente intenta Ollama. Si ambos fallan, devuelve `LLMResponse(ok=False, error="...")`.

### LLMResponse

```python
@dataclass
class LLMResponse:
    text: str = ""          # Texto generado
    model: str = ""         # "gemini-2.0-flash" o "llama3:8b"
    provider: str = ""      # "gemini" o "ollama"
    usage: dict = field(...)# Tokens utilizados
    error: str | None = None# Mensaje de error si fallo

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)
```

---

## 5. Decisiones de diseno

### Por que rapidfuzz y no un LLM para entity extraction?

| Aspecto | rapidfuzz | LLM-based NER |
|---------|-----------|---------------|
| Velocidad | ~1ms | ~500ms+ |
| Coste | Gratis | Consume tokens |
| Fiabilidad | Determinista | Puede variar |
| Dependencias | Solo rapidfuzz | API call |

Para un catalogo de farmacos cerrado (nombres conocidos), fuzzy matching es mas eficiente y fiable.

### Por que google-genai y no google-generativeai?

El SDK `google-generativeai` esta **deprecated**. Google recomienda migrar a `google-genai` (>= 1.64.0), que usa una API diferente:

```python
# VIEJO (deprecated)
import google.generativeai as genai
genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("...")

# NUEVO (actual)
from google import genai
client = genai.Client(api_key=key)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="...",
    config=types.GenerateContentConfig(system_instruction="..."),
)
```

### Por que fallback automatico?

En produccion, un solo punto de fallo es inaceptable. El fallback Gemini -> Ollama garantiza disponibilidad:

- **Gemini caido**: Ollama responde localmente
- **Sin internet**: Ollama funciona offline
- **Rate limit de Gemini**: Ollama absorbe el exceso

---

## 6. Testing

37 tests para el engine + 14 tests para el LLM = **51 tests** en total.

### Como probamos sin llamar a APIs reales

Usamos **mocking** extensivo:

```python
@patch("pharmagraphrag.llm.client._generate_gemini")
def test_generate_answer_gemini(mock_gemini):
    """Test que Gemini se llama correctamente."""
    mock_gemini.return_value = LLMResponse(
        text="Ibuprofen can cause nausea.",
        model="gemini-2.0-flash",
        provider="gemini",
    )
    result = generate_answer("system", "user")
    assert result.ok
    assert "nausea" in result.text
```

### Tests de fallback

```python
@patch("pharmagraphrag.llm.client._generate_ollama")
@patch("pharmagraphrag.llm.client._generate_gemini")
def test_fallback_gemini_to_ollama(mock_gemini, mock_ollama):
    """Si Gemini falla, debe intentar Ollama."""
    mock_gemini.return_value = LLMResponse(error="API key invalid")
    mock_ollama.return_value = LLMResponse(text="Fallback OK", ...)

    result = generate_answer("system", "user")
    assert result.ok
    assert result.provider == "ollama"
```

---

## Resumen

| Componente | Archivo | Funcion principal |
|------------|---------|-------------------|
| Entity Extraction | `engine/entity_extractor.py` | `extract_entities()` |
| Dual Retrieval | `engine/retriever.py` | `retrieve_context()` |
| Query Orchestrator | `engine/query_engine.py` | `process_query()` |
| LLM Client | `llm/client.py` | `generate_answer()` |

El query engine es el **corazon** del sistema: conecta la extraccion de entidades con la recuperacion de informacion y la generacion de respuestas. Sin el, tendriamos datos en Neo4j y ChromaDB pero ningun modo de consultarlos de forma inteligente.
