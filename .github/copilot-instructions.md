# Professional Profile — Copilot Instructions

## About Me
- **Name**: Jose María Ponce Bernabé
- **Location**: Murcia, Spain
- **Languages**: Spanish (native), English (B2 — FCE), French (B2 — DELF)
- **Email**: jmponcebe@gmail.com
- **GitHub**: github.com/jmponcebe
- **LinkedIn**: linkedin.com/in/jmponcebe

## Professional Background

### Current Status (2025)
Recently graduated from **Master in Data Science and Cloud Data Engineering** at **Universidad de Castilla-La Mancha (CIDaeN)**. Looking for ML Engineer, MLOps Engineer, or Data Engineer roles.

### Education
1. **MSc Data Science & Cloud Data Engineering** — UCLM (CIDaeN), Oct 2024 – Feb 2026
   - Thesis: DengueMLOps — End-to-end MLOps pipeline for dengue prediction in Brazil
   - Focus: ML, Cloud (AWS), MLOps, Data Pipelines
2. **MSc Bioinformatics** — University of Murcia, Sep 2020 – Sep 2023, GPA 9.03/10
   - Stats, ML (R, caret, keras), HPC (SLURM), Docker, Unix, Git
   - Thesis: Ontology of units of measurement in the chemical industry
3. **BSc Biotechnology** — University of Murcia, Sep 2015 – Jun 2020, GPA 7.35/10
   - Erasmus+ at Université Paris-Est Créteil (2017–2018)

### Work Experience
1. **Semantic Solutions Developer** — NTT DATA (SEMBU), Jan 2023 – Mar 2024, Barcelona
   - Python microservices for knowledge graph construction (Flask, RDFLib, Owlready2)
   - Data pipelines for RDF/OWL → GraphDB, Elasticsearch/OpenSearch
   - SPARQL/GraphQL APIs, SHACL validation, FAIR/DCAT standards
2. **Knowledge Engineer / Ontologist** — BASF Digital Solutions, Jun 2021 – Dec 2022, Madrid
   - Co-developed GOMO (Governance Operational Model for Ontologies), published ISWC 2022
   - Production ontologies with Protégé, Python (RDFLib, kglab, morph-kgc)
   - RML/YARRRML data-to-graph transformations, Streamlit internal tools, Graphistry visualizations
3. **Research Intern (Bioinformatics)** — University of Murcia, Oct 2020 – Jun 2021
4. **Research Intern (Agriculture)** — CEBAS-CSIC, Oct 2018 – Jan 2019

### Publication
- "Ontology Management in an Industrial Environment: The BASF GOMO" — ISWC 2022

### Certification
- AWS Academy Cloud Foundations (Jan 2025)

## Key Technical Skills
- **Python** (3+ years production): microservices, APIs, data pipelines, ML
- **MLOps**: MLflow, Evidently, Prefect, GitHub Actions, Docker, CI/CD
- **Cloud**: AWS (S3, ECR, ECS/Fargate, CloudFormation IaC)
- **ML**: XGBoost, Scikit-learn, TensorFlow, Optuna, Pandas, NumPy
- **Apps**: FastAPI, Streamlit, Flask, Pydantic, Plotly
- **Data**: SQL, Spark/PySpark, Parquet, MongoDB, ETL, Elasticsearch, GraphDB, Databricks
- **Semantic Tech**: RDF, OWL, SPARQL, SHACL, Knowledge Graphs
- **Tools**: Git, GitHub, Docker Compose, Linux/Unix, LaTeX

## TFM Project: DengueMLOps
End-to-end MLOps pipeline for dengue alert prediction in Brazil:
- **Data**: 4.5M weekly records from Mosqlimate API, 5,500+ municipalities (2010–2025)
- **Features**: 15 engineered features with zero target leakage, climate lags based on vector biology
- **Model**: XGBoost with balanced sample weights, Optuna-tuned (40 trials), macro_f1=0.39
- **Tracking**: 5 MLflow experiments, 90+ runs, Model Registry with champion alias
- **Serving**: FastAPI REST API + Streamlit dashboard with Brazil choropleth map
- **Containers**: Multi-image Docker, Compose orchestration, health checks
- **CI/CD**: GitHub Actions — tests/lint/build on push, ECR/ECS deploy on tags
- **Cloud**: AWS ECS/Fargate + S3 + ECR, CloudFormation IaC, automated deploy scripts
- **Monitoring**: Evidently data drift (KS + chi²), prediction logging with auto-flush
- **Tests**: 88 unit tests, 97% coverage on feature engineering
- **Repo**: https://github.com/jmponcebe/DengueMLOps

## Unique Value Proposition
Combination of domain science (biotechnology), knowledge engineering (ontologies, knowledge graphs), and modern ML operations. Can bridge the gap between domain experts and ML infrastructure. Understands data quality, metadata standards, and scalable data modeling from both academic and enterprise perspectives.

## Career Hub Structure
This repo is organized as a career management hub:
- `cv/cv_en.tex` — English CV (LaTeX, Palatino, A4)
- `cv/cv_es.tex` — Spanish CV (LaTeX, Palatino, A4)
- `cover-letters/templates/` — Reusable cover letter templates (EN/ES)
- `cover-letters/applications/` — Company-specific cover letters (YYYY-MM_company_role.tex)
- `linkedin/profile_en.md` — LinkedIn profile content (English)
- `linkedin/profile_es.md` — LinkedIn profile content (Spanish)
- `linkedin/posts/` — Draft LinkedIn posts
- `applications/tracker.md` — Application tracking (status, company, notes)
- `interview-prep/` — Study notes (ML fundamentals, system design, Spark)
- `assets/` — LinkedIn banner, icons
- `scripts/generate_banner.py` — LinkedIn banner generator (Pillow)
- `.github/copilot-instructions.md` — This file

## Code Style for LaTeX
- Use `\CVSubheading`, `\CVItem`, `\CVItemListStart/End` macros consistently
- Keep bullet points concise: start with action verb, include measurable impact
- Spanish CV bullets must use action verbs (first person past tense: "Diseñé", "Desarrollé", etc.)
- One page for CV (two max if needed for academic version)
- Use fontawesome5 for contact icons
- Compile with pdflatex (palatino font, standard packages)
- Cover letters use `\newcommand` variables for easy per-application customization
- Spanish numbers: comma as decimal separator (4,5M), dot as thousands separator (5.500)

## Target Roles
- ML Engineer
- MLOps Engineer
- Data Engineer (ML/AI focus)
- Platform Engineer (ML)

## Key Messages for Applications
1. **Not a fresh grad**: 3 years production Python experience (BASF + NTT DATA)
2. **End-to-end builder**: From data ingestion to cloud deployment and monitoring
3. **Domain versatile**: Worked in chemical industry (BASF), consulting (NTT DATA), epidemiology (TFM)
4. **Research + Industry**: Published at ISWC, but focused on practical engineering
5. **Trilingual**: ES/EN/FR — valuable for international teams
