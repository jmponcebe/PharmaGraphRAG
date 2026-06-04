# Deployment Reference — Full Details

## Live Architecture

| Service | Platform | URL | Cost |
|---|---|---|---|
| Chat UI | Streamlit Community Cloud | <https://pharmagraphrag.streamlit.app> | $0 |
| API + ChromaDB | Google Cloud Run (us-central1) | pharmagraphrag-api-893694384146.us-central1.run.app | $0 (free tier) |
| Knowledge Graph | Neo4j Aura Free | Managed instance (11.9K nodes, 381K rels) | $0 (200K nodes limit) |

## Streamlit Cloud

- Reads `API_URL` from `st.secrets`, switches to HTTP mode (calls Cloud Run API instead of local imports)
- Uses `uv sync` from `uv.lock`
- Main file: `src/pharmagraphrag/ui/app.py`

## Cloud Run

- Docker image with CPU-only PyTorch + baked-in ChromaDB + pre-cached embedding model
- `docker/Dockerfile.cloudrun` uses multi-stage build
- `min_instances=0` (scale to zero), `max_instances=2`
- Cold start ~50s, warm ~4.5s
- 4 env vars: GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

## Neo4j Aura

- Free tier: 200K nodes, 400K rels
- Data migrated via `scripts/migrate_neo4j.py`
- Auto-pauses after 3 days inactivity

## CD Pipeline

1. **Trigger**: GitHub Actions `deploy.yml` on version tags (`v*`)
2. **Flow**: `deploy.yml` reuses CI workflow for tests → authenticates GCP → runs `gcloud builds submit --config=cloudbuild.yaml`
3. **Cloud Build steps** (`cloudbuild.yaml`):
   - Download ChromaDB snapshot from `gs://pharmagraphrag-data/chroma/chroma/`
   - Multi-stage Docker build (`docker/Dockerfile.cloudrun`)
   - Push image to GCR (`gcr.io/pharmagraphrag/api:{tag}` + `latest`)
   - Deploy to Cloud Run (min_instances=0, max_instances=2)
4. **Service Account**: `github-cd@pharmagraphrag.iam.gserviceaccount.com` (roles: run.admin, storage.admin, iam.serviceAccountUser, cloudbuild.builds.editor, logging.viewer, viewer)
5. **GCS Bucket**: `gs://pharmagraphrag-data` (us-central1) — stores ChromaDB embeddings (99.6 MiB)

## Kubernetes (GKE Autopilot, on-demand)

- Helm 3 chart in `helm/pharmagraphrag/`, raw manifests in `k8s/`
- HPA on CPU/memory, startup probes tuned for ~50s embedding-model cold start
- LoadBalancer for UI, ClusterIP for API, optional Ingress + GKE managed cert
- Deployed via `deploy-gke.yml` on `v*-k8s` tags → GCR → Helm upgrade
- On-demand cluster strategy: create cluster, demo, destroy (avoid idle costs)

## Versions

- v1.0.0 (initial), v1.0.3 (current)

## Key Files

- `cloudbuild.yaml` — Cloud Build config
- `.github/workflows/deploy.yml` — CD workflow (Cloud Run)
- `.github/workflows/deploy-gke.yml` — CD workflow (GKE)
- `docker/Dockerfile.cloudrun` — Cloud Run image
- `scripts/migrate_neo4j.py` — Migrate data between Neo4j instances
- `scripts/setup_demo.py` — Load demo data into any Neo4j instance
