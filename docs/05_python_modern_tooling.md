# Python Moderno: uv, pyproject.toml y Estructura de Proyecto

## El ecosistema de Python moderno

Python ha evolucionado mucho en cómo gestiona proyectos. Históricamente era un caos (setup.py, requirements.txt, virtualenv, pip...). Hoy hay un estándar más limpio.

```
Antes (2015-2020):             Ahora (2024+):
setup.py                       pyproject.toml (todo en uno)
setup.cfg
requirements.txt               uv.lock (lockfile automático)
requirements-dev.txt
MANIFEST.in

virtualenv / venv              uv (gestor de entorno + paquetes)
pip install
pip freeze
```

---

## uv: El gestor de paquetes moderno

### ¿Qué es uv?

**uv** es un gestor de paquetes y entornos Python, escrito en Rust. Es un reemplazo de `pip`, `pip-tools`, `virtualenv`, y `pyenv` — todo en uno.

Creado por **Astral** (la misma empresa que hace **ruff**, el linter que usamos).

### ¿Por qué uv y no pip?

| Característica | pip | uv |
|---------------|-----|-----|
| Velocidad | Lento (Python) | 10-100x más rápido (Rust) |
| Resolución de deps | Básica | Avanzada (resolver conflictos) |
| Lockfile | No (necesitas pip-freeze) | Sí (uv.lock automático) |
| Virtualenv | Manejo separado | Integrado |
| Reproducibilidad | Difícil | Garantizada (lockfile) |
| Multiplataforma | Sí | Sí |

### Comandos básicos de uv

```bash
# Crear proyecto nuevo (hace init + virtualenv + pyproject.toml)
uv init mi-proyecto

# Instalar dependencias del pyproject.toml
uv sync

# Instalar con extras (dev tools, UI)
uv sync --extra dev --extra ui

# Añadir una dependencia
uv add pandas
uv add --dev pytest  # Solo para desarrollo

# Ejecutar un script con el entorno del proyecto
uv run python scripts/load_graph.py
uv run pytest

# Ver el entorno
uv python list      # Versiones de Python disponibles
```

### ¿Qué pasa cuando ejecutas `uv sync`?

```
1. Lee pyproject.toml (dependencias declaradas)
2. Resuelve versiones compatibles de TODAS las dependencias
   (incluyendo dependencias transitivas)
3. Crea/actualiza .venv/ (virtualenv)
4. Instala los paquetes en .venv/
5. Genera/actualiza uv.lock (versiones exactas resueltas)
```

### ¿Qué es .venv/?

Un **virtual environment** (entorno virtual) es una copia aislada de Python donde se instalan paquetes **solo para este proyecto**. Sin virtualenv, todos los paquetes se instalan globalmente y pueden entrar en conflicto.

```
C:\Users\ponce\workspace\PharmaGraphRAG\
├── .venv/                        # Entorno virtual (creado por uv)
│   ├── Scripts/                  # (Windows) o bin/ (Linux)
│   │   ├── python.exe            # Python del entorno
│   │   ├── pip.exe               # pip del entorno
│   │   └── pytest.exe            # Herramientas instaladas
│   └── Lib/site-packages/        # Paquetes instalados
│       ├── pandas/
│       ├── neo4j/
│       ├── chromadb/
│       └── ... (170 paquetes)
```

Cuando haces `uv run pytest`, uv automáticamente usa el Python de `.venv/`.

### uv.lock: Reproducibilidad

El lockfile `uv.lock` registra las **versiones exactas** de TODOS los paquetes:

```toml
# uv.lock (ejemplo simplificado)
[[package]]
name = "pandas"
version = "2.2.3"
dependencies = [
    { name = "numpy", version = "2.1.2" },
    { name = "python-dateutil", version = "2.9.0" },
]

[[package]]
name = "numpy"
version = "2.1.2"
```

**¿Por qué importa?** Si alguien clona tu repo y ejecuta `uv sync`, obtendrá **exactamente** las mismas versiones que tú. Sin lockfile, `pip install pandas` podría instalar una versión diferente y romper cosas.

---

## pyproject.toml: La configuración centralizada

Antes de pyproject.toml, un proyecto Python necesitaba 5-6 archivos de configuración. Ahora **todo va en uno**:

### Secciones de nuestro pyproject.toml

#### 1. Metadata del proyecto
```toml
[project]
name = "pharmagraphrag"
version = "0.1.0"
description = "GraphRAG system for drug interactions and adverse events"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Jose María Ponce Bernabé" }
]
```

#### 2. Dependencias
```toml
dependencies = [
    # Data
    "pandas>=2.0",
    "pyarrow>=14.0",        # Para Parquet
    "httpx>=0.27",          # HTTP client moderno

    # Graph
    "neo4j>=5.0",           # Driver Python para Neo4j

    # Vector Store
    "chromadb>=0.5",
    "sentence-transformers>=3.0",

    # LLM
    "google-generativeai>=0.8",  # Gemini API
    "ollama>=0.3",               # Ollama local

    # API
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",

    # Config
    "pydantic-settings>=2.0",
    "loguru>=0.7",
]
```

#### 3. Dependencias opcionales (extras)
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit>=4.0",
]
ui = [
    "streamlit>=1.40",
]
```

Con `uv sync --extra dev --extra ui` se instalan las deps base + dev + ui.

#### 4. Build system
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Esto dice: "para construir el paquete instalable, usa hatchling". Es lo que permite que `pip install .` o `uv sync` funcionen.

#### 5. Configuración de herramientas
```toml
# Ruff (linter + formatter)
[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]  # Reglas activadas

# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

# MyPy (type checker)
[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
```

### ¿Por qué es mejor que archivos separados?

| Antes | Ahora (pyproject.toml) |
|-------|----------------------|
| setup.py (dependencias) | `[project.dependencies]` |
| setup.cfg (metadata) | `[project]` |
| requirements.txt (versiones) | `uv.lock` |
| pytest.ini (config pytest) | `[tool.pytest.ini_options]` |
| .flake8 (linting) | `[tool.ruff]` |
| mypy.ini (type checking) | `[tool.mypy]` |
| **6 archivos** | **1 archivo** |

---

## src Layout: ¿Por qué la subcarpeta?

### La estructura
```
PharmaGraphRAG/
├── src/
│   └── pharmagraphrag/    # ← Paquete Python
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       ├── graph/
│       └── vectorstore/
├── tests/
└── pyproject.toml
```

### ¿Por qué no poner el código directamente en src/?

**Pregunta**: ¿Por qué `src/pharmagraphrag/config.py` y no `src/config.py`?

**Respuesta**: La carpeta `pharmagraphrag/` es el **paquete Python**. El nombre del paquete **es** `pharmagraphrag`.

```python
# Esto funciona porque el paquete se llama "pharmagraphrag"
from pharmagraphrag.config import get_settings
from pharmagraphrag.vectorstore.store import search
```

Si fuera `src/config.py`, tendrías:
```python
from config import get_settings  # ¿De qué proyecto? Ambiguo y propenso a conflictos
```

### ¿Por qué src/ y no directamente pharmagraphrag/?

```
# Opción A: Flat layout (sin src/)
PharmaGraphRAG/
├── pharmagraphrag/     # ← Paquete
├── tests/
└── pyproject.toml

# Opción B: src/ layout (la nuestra)
PharmaGraphRAG/
├── src/
│   └── pharmagraphrag/ # ← Paquete
├── tests/
└── pyproject.toml
```

La carpeta `src/` existe para **prevenir imports accidentales**:

**Sin src/**: Si estás en el directorio `PharmaGraphRAG/` y haces `python -c "import pharmagraphrag"`, Python podría importar desde el directorio actual en vez del paquete instalado. Esto causa bugs sutiles donde el código funciona en tu PC pero no cuando lo instalas.

**Con src/**: El paquete **solo** se puede importar si está instalado correctamente (con `uv sync`). Más seguro y reproducible.

El **src layout es el estándar recomendado** por PyPA (Python Packaging Authority) y herramientas modernas.

---

## Testing con pytest

### ¿Qué es pytest?

Un framework de testing para Python. Más simple y potente que el `unittest` estándar.

### Filosofía: "un test es una función que empieza con test_"

```python
# tests/test_vectorstore.py

def test_chunk_text_empty_returns_empty():
    """chunk_text con texto vacío devuelve lista vacía."""
    assert chunk_text("") == []

def test_chunk_text_short_returns_single():
    """Texto corto cabe en un solo chunk."""
    result = chunk_text("Hello world", chunk_size=1000)
    assert len(result) == 1
```

Ejecutar: `uv run pytest` → pytest descubre TODOS los archivos `test_*.py` y ejecuta TODAS las funciones `test_*`.

### Fixtures: Setup reutilizable

```python
@pytest.fixture()
def sample_label() -> dict:
    """Devuelve un label DailyMed de ejemplo para tests."""
    return {
        "drug_name": "WARFARIN",
        "generic_names": ["WARFARIN SODIUM"],
        "brand_names": ["Coumadin"],
        "route": ["ORAL"],
        "sections": {
            "drug_interactions": "Warfarin interacts with many drugs. " * 20,
            "adverse_reactions": "Bleeding is a common side effect. " * 10,
        },
    }

def test_chunk_drug_label(sample_label):
    """Usa el fixture sample_label automáticamente."""
    chunks = chunk_drug_label(sample_label)
    assert len(chunks) > 0
```

Pytest inyecta `sample_label` automáticamente cuando un test lo pide como parámetro. Esto evita repetir código de setup.

### Mocking: Simular dependencias externas

```python
from unittest.mock import patch

@patch("httpx.get")  # Reemplaza httpx.get con un mock
def test_fetch_drug_label(mock_get):
    """Test que simula la respuesta de la API."""
    # Configurar qué devuelve el mock
    mock_get.return_value.json.return_value = {
        "results": [{"openfda": {"generic_name": ["ASPIRIN"]}}]
    }
    
    # Ejecutar la función (que internamente llama a httpx.get)
    result = fetch_drug_label("aspirin")
    
    # Verificar
    assert result["drug_name"] == "ASPIRIN"
    mock_get.assert_called_once()  # Se llamó exactamente 1 vez
```

**¿Por qué mocks?**
- Tests **rápidos** (no esperan respuestas HTTP)
- Tests **fiables** (no dependen de que la API esté disponible)
- Tests **reproducibles** (siempre la misma respuesta)
- **No gastan quota** de la API

### tmp_path: Directorio temporal

```python
def test_save_file(tmp_path):
    """tmp_path da un directorio temporal que se limpia solo."""
    filepath = tmp_path / "test.json"
    filepath.write_text('{"name": "test"}')
    
    result = load_drug_label(filepath)
    assert result["name"] == "test"
    # tmp_path se elimina automáticamente después del test
```

### Nuestros 142 tests

| Archivo | Tests | Qué cubren |
|---------|-------|-----------|
| test_download_faers.py | 2 | URLs de descarga, skip de archivos existentes |
| test_clean_faers.py | 13 | Normalización nombres, deduplicación, mapping outcomes |
| test_ingest_dailymed.py | 12 | Parsing API, guardar JSON, manejo errores (mocked HTTP) |
| test_vectorstore.py | 35 | Chunking, embeddings, ChromaDB add/search/filter |
| test_engine.py | 37 | Entity extraction, retrieval, prompt assembly |
| test_llm.py | 14 | Gemini, Ollama, fallback chain (mocked APIs) |
| test_api.py | 13 | FastAPI endpoints, TestClient |
| test_ui.py | 14 | Streamlit components, session state |
| **Total** | **142** | |

---

## Herramientas de calidad de código

### ruff: Linter + Formatter

**Linter**: Analiza el código sin ejecutarlo para encontrar errores y malas prácticas.
**Formatter**: Reformatea el código para que sea consistente (indentación, espacios, etc.).

```bash
# Verificar errores de estilo
uv run ruff check src/ tests/

# Formatear código automáticamente
uv run ruff format src/ tests/
```

Reglas que tenemos activadas:
- **E**: Errores de estilo PEP 8
- **F**: Errores de pyflakes (variables no usadas, imports no usados)
- **I**: Orden de imports (isort)
- **N**: Convenciones de naming
- **W**: Warnings
- **UP**: Modernizar código (usar f-strings en vez de .format(), etc.)

### mypy: Type checker

Python es dinámicamente tipado, pero con **type hints** puedes añadir tipos opcionales:

```python
# Sin type hints (funciona, pero ¿qué es text? ¿qué devuelve?)
def chunk_text(text, chunk_size, chunk_overlap):
    ...

# Con type hints (claro qué espera y qué devuelve)
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    ...
```

mypy verifica estáticamente que los tipos son correctos:
```bash
uv run mypy src/
```

### pre-commit: Hooks automáticos

Pre-commit ejecuta verificaciones **antes de cada commit git**:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff        # Lint
      - id: ruff-format # Format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
```

Flujo:
```
git commit -m "feat: add feature"
    ↓
pre-commit ejecuta:
  ✅ ruff check → sin errores
  ✅ ruff format → ya formateado
  ✅ trailing whitespace → ok
    ↓
commit exitoso
```

Si hay errores, el commit se **bloquea** hasta que los corrijas.

---

## Docker: Contenedores

### ¿Qué es Docker?

Docker empaqueta una aplicación con **todo lo que necesita** (sistema operativo, librerías, configuración) en un contenedor que funciona igual en cualquier máquina.

```
Sin Docker:
  "Funciona en mi PC" → "No funciona en la tuya" (diferentes versiones, configs, OS)

Con Docker:
  "Funciona en mi PC" → "Funciona en TODAS las PCs" (mismo contenedor)
```

### Nuestro docker-compose.yml

```yaml
services:
  neo4j:
    image: neo4j:5-community    # Imagen oficial de Neo4j
    container_name: pharmagraphrag-neo4j
    ports:
      - "7474:7474"              # Puerto del browser
      - "7687:7687"              # Puerto Bolt
    environment:
      NEO4J_AUTH: neo4j/pharmagraphrag
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data         # Persistir datos
    healthcheck:
      test: wget --no-verbose --tries=1 --spider http://localhost:7474
```

- **`image`**: Qué software ejecutar (descarga automáticamente)
- **`ports`**: Mapear puertos del contenedor a tu PC (7474:7474 = accesible en localhost:7474)
- **`environment`**: Variables de entorno (contraseña, plugins)
- **`volumes`**: Datos que persisten aunque reinicies el contenedor
- **`healthcheck`**: Verifica que el servicio está sano

### CI/CD: GitHub Actions

```yaml
# .github/workflows/ci.yml (simplificado)
name: CI
on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --extra dev --extra ui
      - run: uv run ruff check src/ tests/
      - run: uv run ruff format --check src/ tests/
      - run: uv run pytest --cov --cov-report=xml
      - uses: actions/upload-artifact@v4
        with: { name: coverage, path: coverage.xml }

  docker-build:
    needs: lint-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: docker/setup-buildx-action@v3
      - run: docker buildx build -f docker/Dockerfile.api .
      - run: docker buildx build -f docker/Dockerfile.ui .
```

Cada vez que haces `git push`, GitHub **automáticamente**:
1. Crea máquinas virtuales (Python 3.11 y 3.13)
2. Instala uv y dependencias
3. Ejecuta linting (ruff check + format)
4. Ejecuta 142 tests con cobertura
5. Construye imágenes Docker (multi-stage con Buildx + GHA cache)
6. Te dice si algo falló (❌) o todo está bien (✅)

---

## Resumen de conceptos

| Concepto | ¿Qué es? | ¿Dónde lo usamos? |
|----------|----------|-------------------|
| **uv** | Gestor de paquetes y entornos (reemplazo de pip + venv) | Todo el proyecto |
| **pyproject.toml** | Configuración centralizada | Raíz del proyecto |
| **uv.lock** | Versiones exactas de dependencias | Raíz del proyecto |
| **.venv/** | Entorno virtual con paquetes aislados | Auto-generado por uv |
| **src layout** | Código fuente bajo src/paquete/ | src/pharmagraphrag/ |
| **pytest** | Framework de testing | tests/ |
| **fixtures** | Setup reutilizable para tests | En funciones con @pytest.fixture |
| **mocking** | Simular dependencias externas | Tests de ingest_dailymed |
| **ruff** | Linter + formatter (Rust, rápido) | pyproject.toml config |
| **mypy** | Type checker estático | pyproject.toml config |
| **pre-commit** | Hooks automáticos antes de commit | .pre-commit-config.yaml |
| **Docker** | Contenedores reproducibles | docker-compose.yml |
| **GitHub Actions** | CI/CD automático | .github/workflows/ci.yml |
| **type hints** | Anotaciones de tipos opcionales | Todo el código src/ |
| **Parquet** | Formato columnar eficiente | data/processed/ |
