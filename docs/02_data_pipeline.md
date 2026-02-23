# Pipeline de Datos: FAERS y DailyMed

## Visión general

Antes de tener un sistema inteligente, necesitamos **datos**. Nuestro pipeline descarga, limpia y estructura datos de dos fuentes oficiales de la FDA:

```
FDA FAERS (web)          DailyMed / openFDA (API)
     │                          │
     ▼                          ▼
download_faers.py        ingest_dailymed.py
     │                          │
     ▼                          ▼
data/raw/faers/          data/raw/dailymed/
 2024Q3/*.txt              aspirin.json
 2024Q4/*.txt              warfarin.json
     │                      ...88 archivos
     ▼
clean_faers.py
     │
     ▼
data/processed/faers/
 2024Q3/*.parquet
 2024Q4/*.parquet
```

---

## Fuente 1: FDA FAERS

### ¿Qué es FAERS?

**FDA Adverse Event Reporting System** — Un sistema donde médicos, pacientes y fabricantes reportan **efectos adversos** de medicamentos a la FDA.

Ejemplo de un reporte:
> "Paciente de 65 años tomando WARFARIN reportó HEMORRAGIA CEREBRAL. Resultado: HOSPITALIZACIÓN."

La FDA publica estos reportes trimestralmente como archivos CSV (realmente ficheros delimitados por `$`).

### ¿Qué datos descargamos?

Descargamos 2 trimestres: **2024Q3** y **2024Q4** (~135MB en total).

Cada trimestre tiene 5 tablas:

| Tabla | Qué contiene | Ejemplo | Filas (aprox) |
| ------- | ------------- | --------- | --------------- |
| **DEMO** | Datos demográficos del paciente | Edad: 65, Sexo: F, País: US | 408K |
| **DRUG** | Medicamentos que tomaba | WARFARIN, 5mg, oral | 1.95M |
| **REAC** | Reacciones adversas reportadas | HAEMORRHAGE, NAUSEA | 1.4M |
| **OUTC** | Resultado clínico | Death (DE), Hospitalization (HO) | 300K |
| **INDI** | Indicación (por qué tomaba el fármaco) | ATRIAL FIBRILLATION | 550K |

### download_faers.py — Paso a paso

```python
# 1. Construye la URL de descarga
url = f"https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip"
# Ejemplo: faers_ascii_2024Q3.zip

# 2. Descarga el ZIP con barra de progreso
# Usa httpx (como requests pero más moderno) con streaming
# para no cargar todo en memoria

# 3. Extrae los archivos del ZIP
# Cada ZIP contiene archivos .txt (CSV delimitados por $)
# Los guarda en data/raw/faers/2024Q3/

# 4. Es idempotente: si el archivo ya existe, no lo vuelve a descargar
```

**Conceptos clave**:

- **Streaming download**: Descargar archivo grande sin cargar todo en RAM. Va leyendo trozos (chunks) y escribiéndolos en disco progresivamente.
- **Idempotencia**: Si ejecutas el script 2 veces, la segunda vez no hace nada (el archivo ya existe). Esto es importante en pipelines de datos — deberías poder re-ejecutar sin efectos secundarios.

### clean_faers.py — Paso a paso

Los datos crudos son un desastre:

```
"primaryid$caseid$caseversion$i_f_code$event_dt$mfr_dt..."
"123456$654321$1$I$20240101$..."
"WARFARIN  (5MG)$PRIMARY SUSPECT DRUG$..."
"warfarin  (5mg)$Primary Suspect Drug$..."  ← duplicado con diferente caso
```

Problemas:

1. **Delimitador `$`** en vez de coma (CSV normal)
2. **Duplicados**: El mismo reporte puede tener varias versiones
3. **Inconsistencia**: "WARFARIN", "warfarin", "Warfarin (5mg)" son el mismo fármaco
4. **Valores nulos**: Campos vacíos, formatos inconsistentes

#### ¿Qué hace el limpiado?

```python
# 1. Leer el archivo $-delimitado
df = pd.read_csv(filepath, sep="$", encoding="latin1", low_memory=False)

# 2. Normalizar nombres de fármacos
def normalize_drug_name(name: str) -> str:
    name = name.upper()              # warfarin → WARFARIN
    name = name.strip()              # "  WARFARIN  " → "WARFARIN"
    name = re.sub(r"\(.*?\)", "", name)  # "WARFARIN (5MG)" → "WARFARIN"
    name = re.sub(r"\s+", " ", name) # "WARFARIN   SODIUM" → "WARFARIN SODIUM"
    return name.strip()

# 3. Deduplicar: quedarse con la última versión de cada reporte
df = df.sort_values("caseversion").drop_duplicates(
    subset=["primaryid", "drug_seq"], keep="last"
)

# 4. Mapear códigos de outcome a nombres legibles
OUTCOME_MAP = {
    "DE": "Death",
    "HO": "Hospitalization",
    "LT": "Life-Threatening",
    "DS": "Disability",
    "CA": "Congenital Anomaly",
    "RI": "Required Intervention",
    "OT": "Other Serious",
}

# 5. Guardar como Parquet (formato columnar eficiente)
df.to_parquet(output_path)
```

**Conceptos clave**:

- **Normalización**: Convertir datos inconsistentes a un formato uniforme. Sin esto, "warfarin" y "WARFARIN" serían dos fármacos diferentes en el grafo.

- **Deduplicación**: FAERS tiene múltiples versiones del mismo reporte (actualizaciones). Nos quedamos con la más reciente.

- **Parquet vs CSV**: Parquet es un formato binario **columnar** (guarda columna por columna, no fila por fila). Ventajas:
  - **Mucho más pequeño** (compresión): 100MB CSV → ~20MB Parquet
  - **Mucho más rápido** de leer (especialmente si solo necesitas algunas columnas)
  - **Preserva tipos** (int, float, string, datetime) — CSV todo es texto
  - **Estándar en data engineering** (Spark, Pandas, DuckDB, etc.)

### Resultado del pipeline FAERS

```
data/raw/faers/
├── 2024Q3/
│   ├── DEMO24Q3.txt    (demográficos)
│   ├── DRUG24Q3.txt    (medicamentos)
│   ├── REAC24Q3.txt    (reacciones)
│   ├── OUTC24Q3.txt    (outcomes)
│   └── INDI24Q3.txt    (indicaciones)
└── 2024Q4/
    └── (misma estructura)

data/processed/faers/
├── 2024Q3/
│   ├── DEMO.parquet    (limpios, deduplicados)
│   ├── DRUG.parquet
│   ├── REAC.parquet
│   ├── OUTC.parquet
│   └── INDI.parquet
└── 2024Q4/
    └── (misma estructura)
```

**Números finales**: 816K reportes únicos, 3.9M entradas de fármacos, 2.8M reacciones.

---

## Fuente 2: DailyMed

### ¿Qué es DailyMed?

La base de datos oficial de **etiquetas de medicamentos** de la FDA. Cada medicamento aprobado tiene una etiqueta con secciones como:

- **Drug Interactions**: Con qué otros fármacos interactúa
- **Adverse Reactions**: Efectos secundarios conocidos
- **Warnings**: Advertencias de seguridad
- **Contraindications**: Cuándo NO usarlo
- **Dosage**: Cómo administrarlo
- etc.

### ¿Por qué necesitamos DailyMed si ya tenemos FAERS?

| | FAERS | DailyMed |
| --- | --- | --- |
| **Tipo de dato** | Reportes individuales (cuantitativo) | Texto oficial de etiquetas (cualitativo) |
| **Ejemplo** | "500 reportes de sangrado con warfarin" | "Warfarin puede causar sangrado grave. Monitorear INR regularmente." |
| **Uso en el proyecto** | Nodos y relaciones en Neo4j | Chunks de texto en ChromaDB |

FAERS te dice **cuántas veces** ocurrió algo. DailyMed te explica **por qué** y **qué hacer al respecto**.

### ingest_dailymed.py — Paso a paso

```python
# 1. Lista de ~200 fármacos más comunes (curada manualmente)
TOP_DRUGS = [
    "aspirin", "ibuprofen", "acetaminophen", "warfarin",
    "metformin", "lisinopril", "atorvastatin", ...
]

# 2. Para cada fármaco, consultar la API de openFDA
url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug}"
response = httpx.get(url)
data = response.json()

# 3. Extraer las secciones de la etiqueta
SECTIONS = [
    "drug_interactions",        # Interacciones
    "adverse_reactions",        # Efectos adversos
    "warnings_and_cautions",    # Advertencias
    "contraindications",        # Contraindicaciones
    "boxed_warning",            # Advertencia grave (caja negra)
    "indications_and_usage",    # Para qué se usa
    "dosage_and_administration", # Dosis
    "clinical_pharmacology",    # Farmacología
    "mechanism_of_action",      # Cómo funciona
    "pharmacodynamics",         # Farmacodinámica
    "overdosage",               # Sobredosis
    "warnings",                 # Advertencias (formato antiguo)
]

# 4. Guardar como JSON individual
# data/raw/dailymed/warfarin.json

# 5. Rate limiting: esperar 0.3s entre llamadas
# (la API de openFDA tiene límites de uso)
```

### Estructura de un archivo DailyMed

```json
{
  "drug_name": "WARFARIN",
  "generic_names": ["WARFARIN SODIUM"],
  "brand_names": ["Coumadin"],
  "manufacturer": ["Bristol-Myers Squibb"],
  "product_type": ["HUMAN PRESCRIPTION DRUG"],
  "route": ["ORAL"],
  "substance_name": ["WARFARIN SODIUM"],
  "pharm_class_epc": ["Vitamin K Inhibitor [EPC]"],
  "pharm_class_moa": ["Vitamin K Antagonists [MoA]"],
  "sections": {
    "drug_interactions": "7 DRUG INTERACTIONS Concomitant use of drugs...",
    "adverse_reactions": "6 ADVERSE REACTIONS The following serious...",
    "warnings_and_cautions": "5 WARNINGS AND PRECAUTIONS Tissue necrosis...",
    "contraindications": "4 CONTRAINDICATIONS Warfarin sodium is...",
    "boxed_warning": "WARNING: BLEEDING RISK...",
    "indications_and_usage": "1 INDICATIONS AND USAGE...",
    "dosage_and_administration": "2 DOSAGE AND ADMINISTRATION...",
    "clinical_pharmacology": "12 CLINICAL PHARMACOLOGY...",
    "mechanism_of_action": "12.1 Mechanism of Action...",
    "pharmacodynamics": "12.2 Pharmacodynamics...",
    "overdosage": "10 OVERDOSAGE..."
  }
}
```

**Resultado**: 88 archivos JSON, uno por fármaco. 71 tienen sección de drug_interactions, 75 tienen adverse_reactions.

---

## Conceptos importantes de Data Engineering

### ETL (Extract, Transform, Load)

Nuestro pipeline sigue el patrón clásico de data engineering:

| Fase | Qué hace | En nuestro proyecto |
| ------ | --------- | ------------------- |
| **Extract** | Obtener datos de la fuente | `download_faers.py`, `ingest_dailymed.py` |
| **Transform** | Limpiar, normalizar, deduplicar | `clean_faers.py` |
| **Load** | Cargar en el sistema de destino | `load_graph.py`, `load_vectorstore.py` |

### Formato Parquet — ¿Por qué no CSV?

Imagina una tabla con 3 columnas y 1 millón de filas:

**CSV (row-oriented)**: Guarda fila por fila

```
nombre,edad,ciudad
Ana,25,Madrid
Bob,30,Barcelona
... (1 millón más)
```

Para leer solo la columna "edad", tiene que leer TODAS las filas.

**Parquet (column-oriented)**: Guarda columna por columna

```
[nombres]: Ana, Bob, Carlos, ...
[edades]: 25, 30, 28, ...
[ciudades]: Madrid, Barcelona, Sevilla, ...
```

Para leer solo "edad", lee un bloque contiguo → **100x más rápido**.

Además, los valores de una misma columna son similares, así que **comprime mucho mejor**.

### Idempotencia en pipelines

Un pipeline idempotente produce el **mismo resultado** sin importar cuántas veces lo ejecutes. Esto es crucial porque:

1. **Errores**: Si el proceso falla a la mitad, puedes reiniciarlo sin miedo
2. **Actualizaciones**: Puedes añadir nuevos datos y re-ejecutar
3. **Testing**: Resultados reproducibles

Nuestro pipeline es idempotente:

- `download_faers.py`: Si el ZIP ya existe, no lo descarga de nuevo
- `clean_faers.py`: Sobreescribe los Parquet → mismo resultado siempre
- `load_graph.py`: Usa MERGE (upsert) en vez de CREATE → no duplica nodos

### APIs y Rate Limiting

La API de openFDA (DailyMed) permite cierto número de llamadas por minuto. Si las excedes, te bloquea. Por eso:

```python
time.sleep(0.3)  # Esperar 300ms entre llamadas
```

Esto es **rate limiting** (limitación de velocidad). Es cortesía y buena práctica cuando consumes APIs de terceros.

---

## Cómo los tests validan el pipeline

### Tests de download_faers.py (2 tests)

```python
def test_build_download_url():
    """Verifica que se construye la URL correcta."""
    url = build_download_url("2024Q3")
    assert "faers_ascii_2024Q3.zip" in url

def test_download_file_skips_existing(tmp_path):
    """Verifica que no re-descarga si el archivo existe."""
    # Crea un archivo falso en tmp_path
    # Ejecuta download_file
    # Verifica que no hizo ninguna petición HTTP
```

### Tests de clean_faers.py (13 tests)

```python
def test_normalize_drug_name():
    assert normalize_drug_name("warfarin (5mg)") == "WARFARIN"
    assert normalize_drug_name("  ASPIRIN  ") == "ASPIRIN"

def test_clean_removes_duplicates():
    """Verifica que solo queda la última versión de cada reporte."""
    # Datos con reporte duplicado (versión 1 y 2)
    # Después de limpiar, solo queda versión 2

def test_outcome_mapping():
    """Verifica que 'DE' se convierte en 'Death'."""
```

### Tests de ingest_dailymed.py (12 tests)

```python
@patch("httpx.get")  # Simula las llamadas HTTP (no llama a la API real)
def test_fetch_drug_label(mock_get):
    """Verifica que la respuesta de la API se parsea correctamente."""
    mock_get.return_value = MockResponse(json={"results": [...]})
    label = fetch_drug_label("aspirin")
    assert label["drug_name"] == "ASPIRIN"
```

**Concepto clave**: Los tests de `ingest_dailymed.py` usan **mocks** — simulan las respuestas de la API para no hacer llamadas reales durante los tests. Esto es más rápido, más fiable, y no gasta quota de la API.
