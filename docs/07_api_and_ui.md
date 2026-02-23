# API REST y Streamlit UI: La Capa de Presentacion

## Resumen

Tenemos un pipeline GraphRAG completo (entity extraction -> retrieval -> prompt -> LLM). Ahora necesitamos dos cosas:

1. **FastAPI**: Una API REST para que cualquier cliente pueda hacer consultas
2. **Streamlit**: Una interfaz de chat visual para usuarios finales

```
   Cliente HTTP          Usuario
       |                    |
       v                    v
   +--------+        +----------+
   | FastAPI |        | Streamlit|
   | (REST)  |        | (Chat)  |
   +----+----+        +----+----+
        |                  |
        +--------+---------+
                 |
        Query Engine + LLM
```

---

## 1. FastAPI: API REST

**Archivos**:
- `src/pharmagraphrag/api/main.py` -- Endpoints
- `src/pharmagraphrag/api/models.py` -- Schemas Pydantic v2

### Por que FastAPI?

| Caracteristica | Flask | FastAPI |
|----------------|-------|---------|
| Validacion automatica | No | Si (Pydantic) |
| Documentacion auto | No | Si (OpenAPI/Swagger) |
| Type hints | Opcionales | Integrados |
| Performance | Medio | Alto (Starlette) |
| Async support | Parcial | Nativo |

FastAPI genera documentacion interactiva automaticamente en `/docs` (Swagger UI) y `/redoc`.

### Schemas con Pydantic v2

Cada endpoint tiene modelos de request y response bien definidos:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Body para POST /query."""
    question: str = Field(
        ...,                    # Requerido
        min_length=3,           # Validacion automatica
        max_length=2000,
        description="Pregunta en lenguaje natural.",
        examples=["What are the side effects of ibuprofen?"],
    )
    use_graph: bool = Field(True, description="Incluir contexto del knowledge graph.")
    use_vector: bool = Field(True, description="Incluir contexto del vector store.")
    use_llm: bool = Field(True, description="Generar respuesta con LLM.")
    n_results: int = Field(5, ge=1, le=20, description="Resultados de busqueda vectorial.")
```

**Ventaja de Pydantic v2**: Si el usuario envia `question: ""` (vacia), FastAPI devuelve automaticamente un error 422 con detalle del problema. No necesitas escribir codigo de validacion.

### Endpoint POST /query

El endpoint principal que ejecuta todo el pipeline:

```python
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Procesa una pregunta por el pipeline GraphRAG completo."""
    from pharmagraphrag.engine.query_engine import process_query
    from pharmagraphrag.llm.client import generate_answer

    # 1. Entity extraction + retrieval
    result = process_query(
        req.question,
        use_graph=req.use_graph,
        use_vector=req.use_vector,
        n_vector_results=req.n_results,
    )

    # 2. Construir lista de fuentes
    sources = []
    for drug in result.context.drugs_found:
        sources.append(SourceInfo(type="graph", drug=drug, ...))
    for vr in result.context.vector_raw:
        sources.append(SourceInfo(type="vector", ...))

    # 3. Generar respuesta LLM (si se pide)
    if req.use_llm:
        llm_resp = generate_answer(
            system_prompt=result.system_prompt,
            user_prompt=result.user_prompt,
        )
        answer = llm_resp.text

    return QueryResponse(
        question=req.question,
        answer=answer,
        drugs_extracted=result.entities.drugs,
        sources=sources,
        ...
    )
```

**Ejemplo de uso** con curl:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the side effects of ibuprofen?"}'
```

### Endpoint GET /drug/{name}

Consulta directa al knowledge graph:

```python
@app.get("/drug/{name}", response_model=DrugInfoResponse)
def get_drug(name: str) -> DrugInfoResponse:
    """Informacion completa de un farmaco desde el grafo."""
    from pharmagraphrag.graph.queries import get_drug_full_context

    ctx = get_drug_full_context(name)
    drug_info = ctx.get("drug_info") or {}
    if not drug_info:
        raise HTTPException(status_code=404, detail=f"Drug '{name}' not found.")

    return DrugInfoResponse(
        name=drug_info.get("name", name.upper()),
        adverse_events=ctx.get("adverse_events", []),
        interactions=ctx.get("interactions", []),
        outcomes=ctx.get("outcomes", []),
        ...
    )
```

**Ejemplo**: `GET /drug/ibuprofen` devuelve un JSON con efectos adversos, interacciones, outcomes, etc.

### Endpoint GET /health

Verifica que Neo4j y ChromaDB estan operativos:

```python
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check de los servicios."""
    # Probar Neo4j
    neo4j_status = "ok" if search_drugs("ASPIRIN", limit=1) else "empty"

    # Probar ChromaDB
    coll = get_collection()
    chroma_status = f"ok ({coll.count()} docs)"

    return HealthResponse(status="ok", version=__version__, ...)
```

Util para Docker healthchecks y monitorizacion.

### Por que endpoints sincronos?

El driver de Neo4j para Python es **sincrono**. Usar `async def` con `await` requeriria un driver async (que no usamos). Los endpoints sincronos son mas simples y correctos en este caso.

---

## 2. Streamlit UI: Interfaz de Chat

**Archivos**:
- `src/pharmagraphrag/ui/app.py` -- App principal
- `src/pharmagraphrag/ui/components.py` -- Componentes reutilizables

### Por que Streamlit?

Streamlit convierte scripts de Python en apps web interactivas con minimo codigo. No necesitas HTML, CSS, ni JavaScript.

```python
import streamlit as st

st.title("Mi App")
nombre = st.text_input("Tu nombre")
if nombre:
    st.write(f"Hola, {nombre}!")
```

Eso genera una web completa con input y output.

### Estructura de la UI

```
+--------------------------------------------------+
|  SIDEBAR              |  MAIN AREA               |
|                       |                           |
|  Logo + titulo        |  Titulo                   |
|  ---                  |  ---                      |
|  Configuracion:       |  Chat history:            |
|  [x] Knowledge Graph  |  User: "Efectos de..."   |
|  [x] Vector Search    |  Bot:  "Segun FAERS..."  |
|  [x] Generar con LLM  |                           |
|  Resultados: [5]      |  [grafo interactivo]      |
|                       |  [fuentes expandibles]    |
|  ---                  |                           |
|  Drug Explorer:       |  ---                      |
|  [buscar farmaco]     |  [campo de chat]          |
|  -> Ver detalle       |                           |
+--------------------------------------------------+
```

### Session State: Persistir datos entre reruns

Streamlit re-ejecuta el script completo cada vez que el usuario interactua. Para mantener la conversacion, usamos **session state**:

```python
def _init_session() -> None:
    """Inicializa el estado de la sesion."""
    if "messages" not in st.session_state:
        st.session_state.messages = []     # Historial de chat
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "use_graph": True,
            "use_vector": True,
            "use_llm": True,
            "n_results": 5,
        }
```

`st.session_state` persiste entre reruns. Sin el, la conversacion se perderia con cada interaccion.

### ChatMessage: Estructura de cada mensaje

```python
@dataclass
class ChatMessage:
    role: str                     # "user" o "assistant"
    content: str                  # Texto del mensaje
    sources_graph: dict = field(...)   # Datos del grafo
    sources_vector: list = field(...)  # Chunks del vector
    drugs_extracted: list = field(...) # Farmacos detectados
    drugs_found: list = field(...)     # Farmacos en el grafo
    llm_provider: str = ""        # "gemini" o "ollama"
    llm_model: str = ""           # "gemini-2.0-flash"
    error: str | None = None
```

### Chat Input y Display

```python
# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        if msg.role == "assistant":
            render_sources(msg.sources_graph, msg.sources_vector)

# Input del usuario
if prompt := st.chat_input("Pregunta sobre farmacos..."):
    # Agregar mensaje del usuario
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))

    # Procesar y agregar respuesta
    result = _process_question(prompt)
    st.session_state.messages.append(result)
```

`st.chat_message` renderiza mensajes con el estilo de chat (burbuja, icono segun rol).
`st.chat_input` muestra un campo de texto fijo en la parte inferior.

---

## 3. Componentes Visuales

**Archivo**: `src/pharmagraphrag/ui/components.py`

### Grafo Interactivo (streamlit-agraph)

Visualizamos el subgrafo relevante como un **grafo interactivo** con nodos y aristas:

```python
def render_graph(graph_raw: dict) -> None:
    """Renderiza el knowledge graph con streamlit-agraph."""
    from streamlit_agraph import Config, Edge, Node, agraph

    nodes_map = {}
    edges = []

    for drug_name, ctx in graph_raw.items():
        # Nodo del farmaco (verde, central)
        nodes_map[f"drug_{drug_name}"] = Node(
            id=f"drug_{drug_name}",
            label=drug_name,
            size=30,
            color="#4CAF50",  # Verde
        )

        # Nodos de efectos adversos (rojo)
        for ae in ctx.get("adverse_events", [])[:10]:
            nodes_map[f"ae_{ae['adverse_event']}"] = Node(
                label=ae["adverse_event"],
                color="#F44336",  # Rojo
            )
            edges.append(Edge(
                source=f"drug_{drug_name}",
                target=f"ae_{ae['adverse_event']}",
                label=str(ae["report_count"]),
            ))

    # Renderizar
    agraph(nodes=list(nodes_map.values()), edges=edges, config=Config(...))
```

**Tipos de nodos** (por color y forma):
- **Verde** (circulo): Farmacos
- **Rojo** (circulo): Eventos adversos
- **Azul** (circulo): Farmacos con interaccion
- **Morado** (diamante): Outcomes
- **Naranja** (triangulo): Categorias

### Panel de Fuentes

Muestra las evidencias que fundamentaron la respuesta:

```python
def render_sources(graph_raw, vector_raw):
    """Renderiza las fuentes/evidencia de la respuesta."""
    # Fuentes del grafo
    with st.expander("Knowledge Graph Sources"):
        for drug, ctx in graph_raw.items():
            st.markdown(f"**{drug}**")
            for ae in ctx.get("adverse_events", [])[:5]:
                st.markdown(f"- {ae['adverse_event']} ({ae['report_count']} reportes)")

    # Fuentes vectoriales
    with st.expander("Vector Sources"):
        for chunk in vector_raw:
            meta = chunk.get("metadata", {})
            st.markdown(f"**{meta.get('drug_name')}** - {meta.get('section')}")
            st.caption(chunk.get("text", "")[:300])
```

`st.expander` crea secciones colapsables -- el usuario puede explorar las fuentes sin que ocupen espacio en pantalla siempre.

### Drug Explorer (Sidebar)

Permite buscar farmacos individualmente sin hacer una pregunta:

```python
def render_drug_explorer():
    """Widget de exploracion de farmacos en la sidebar."""
    drug_search = st.sidebar.text_input("Buscar farmaco")
    if drug_search:
        results = search_drugs(drug_search)
        for drug in results:
            if st.sidebar.button(drug["name"]):
                render_drug_detail(drug["name"])
```

---

## 4. Docker: Empaquetando todo

### Dockerfile.api (Multi-stage)

```dockerfile
# Stage 1: Builder
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 2: Runtime (imagen mas pequena)
FROM python:3.13-slim AS runtime
RUN useradd --uid 1000 appuser     # Usuario no-root
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
USER appuser
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["/app/.venv/bin/uvicorn", "pharmagraphrag.api.main:app", "--host", "0.0.0.0"]
```

**Multi-stage**: El builder instala dependencias, el runtime solo copia los binarios.
**Non-root**: El contenedor corre como `appuser` (seguridad).
**Healthcheck**: Docker verifica que la API responde.

### docker-compose.yml

```yaml
services:
  neo4j:
    image: neo4j:5-community
    ports: ["7474:7474", "7687:7687"]
    healthcheck:
      test: wget --spider http://localhost:7474
      interval: 10s

  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    depends_on:
      neo4j: { condition: service_healthy }

  ui:
    build: { context: ., dockerfile: docker/Dockerfile.ui }
    ports: ["8501:8501"]
    depends_on:
      neo4j: { condition: service_healthy }

  ollama:  # Opcional
    image: ollama/ollama:latest
    profiles: ["ollama"]
```

`depends_on: { condition: service_healthy }` asegura que la API no arranca hasta que Neo4j este listo.

---

## 5. Testing

### Tests de API (13 tests)

Usamos `TestClient` de FastAPI, que simula requests HTTP sin levantar un servidor:

```python
from fastapi.testclient import TestClient
from pharmagraphrag.api.main import app

client = TestClient(app)

@patch("pharmagraphrag.api.main.process_query")
@patch("pharmagraphrag.api.main.generate_answer")
def test_query_endpoint(mock_llm, mock_engine):
    """Test POST /query con mocks."""
    mock_engine.return_value = QueryResult(...)
    mock_llm.return_value = LLMResponse(text="Test answer", ...)

    response = client.post("/query", json={"question": "Side effects of ibuprofen?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
```

### Tests de UI (14 tests)

Streamlit es dificil de testear porque depende de `st.session_state` y componentes visuales. Soluciones:

```python
# Helper para simular session_state (dict con acceso por atributo)
class _DictLike(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value

# Mock del modulo streamlit-agraph (evita import errors)
@patch.dict("sys.modules", {"streamlit_agraph": mock_module})
def test_render_graph():
    """Test que render_graph llama a agraph correctamente."""
    import importlib
    importlib.reload(components)  # Recargar con el mock
    components.render_graph(sample_graph_data)
    mock_agraph.assert_called_once()
```

---

## 6. CI/CD: GitHub Actions

El pipeline se ejecuta en cada push:

```yaml
jobs:
  lint-and-test:
    strategy:
      matrix:
        python-version: ["3.11", "3.13"]  # Test en 2 versiones
    steps:
      - uv sync --extra dev --extra ui
      - uv run ruff check src/ tests/    # Lint
      - uv run ruff format --check       # Formato
      - uv run pytest --cov              # Tests + coverage

  docker-build:
    needs: lint-and-test                  # Solo si lint+tests pasan
    steps:
      - docker buildx build -f docker/Dockerfile.api .
      - docker buildx build -f docker/Dockerfile.ui .
```

---

## Resumen

| Componente | Archivo | Tecnologia |
|------------|---------|------------|
| API REST | `api/main.py` | FastAPI + Pydantic v2 |
| Schemas | `api/models.py` | Pydantic BaseModel |
| Chat UI | `ui/app.py` | Streamlit |
| Grafo visual | `ui/components.py` | streamlit-agraph |
| Fuentes | `ui/components.py` | st.expander |
| Docker API | `docker/Dockerfile.api` | Multi-stage + non-root |
| Docker UI | `docker/Dockerfile.ui` | Multi-stage + non-root |
| Orquestacion | `docker-compose.yml` | 4 servicios |
| CI/CD | `.github/workflows/ci.yml` | GitHub Actions |

La capa de presentacion es lo que convierte un pipeline de datos en un **producto utilizable**. Sin ella, tendriamos funciones excelentes pero ningun modo accesible de usarlas.
