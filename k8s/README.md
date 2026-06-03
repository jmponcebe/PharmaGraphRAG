# Kubernetes manifests

Production-grade Kubernetes manifests for PharmaGraphRAG. Two ways to deploy:

1. **Raw manifests** (this folder) — apply directly with `kubectl apply -f k8s/`.
2. **Helm chart** ([../helm/pharmagraphrag/](../helm/pharmagraphrag/)) — recommended for real environments.

## What's deployed

| Resource | Purpose |
|---|---|
| `namespace.yaml` | Isolates all resources in `pharmagraphrag` namespace |
| `configmap.yaml` | Non-secret config (LLM model, Neo4j user, Chroma path, internal API URL) |
| `secret.example.yaml` | Template only — real Secret created via `kubectl create secret` |
| `api-deployment.yaml` | FastAPI Deployment (1 replica baseline, ChromaDB baked-in image) |
| `api-service.yaml` | ClusterIP exposing API on port 8000 |
| `api-hpa.yaml` | HPA: scale 1→3 on CPU 70% / memory 80% |
| `ui-deployment.yaml` | Streamlit Deployment |
| `ui-service.yaml` | LoadBalancer Service (external IP) for UI |
| `ui-hpa.yaml` | HPA: scale 1→2 on CPU 75% |
| `ingress.yaml` | Optional: GKE Ingress + managed cert (requires domain) |
| `kustomization.yaml` | Apply everything at once via Kustomize |

## Quick deploy (GKE Autopilot)

Assumes `gcloud`, `kubectl` and a GCP project with billing are set up.

```bash
# 1. Create cluster (~5 min)
gcloud container clusters create-auto pharmagraphrag-autopilot \
  --region=us-central1 --project=pharmagraphrag

# 2. Get credentials
gcloud container clusters get-credentials pharmagraphrag-autopilot \
  --region=us-central1 --project=pharmagraphrag

# 3. Create namespace
kubectl apply -f k8s/namespace.yaml

# 4. Create the real Secret (DO NOT use secret.example.yaml directly)
kubectl create secret generic pharmagraphrag-secrets \
  --namespace=pharmagraphrag \
  --from-literal=GEMINI_API_KEY="$GEMINI_API_KEY" \
  --from-literal=NEO4J_URI="$NEO4J_URI" \
  --from-literal=NEO4J_PASSWORD="$NEO4J_PASSWORD"

# 5. Apply everything else
kubectl apply -k k8s/

# 6. Watch pods come up
kubectl -n pharmagraphrag get pods -w

# 7. Get UI external IP (~1-2 min for LB provisioning)
kubectl -n pharmagraphrag get svc pharmagraphrag-ui
```

## Local validation with kind

```bash
kind create cluster --name pgrag
kubectl apply -k k8s/
# Port-forward instead of LoadBalancer:
kubectl -n pharmagraphrag port-forward svc/pharmagraphrag-ui 8501:80
```

## Cost & cleanup

GKE Autopilot bills per pod CPU/memory plus a small cluster management fee.
For a portfolio demo:

- Run cluster ~2 hours → ~$1-2 total
- **Always destroy after screenshots** to avoid surprise bills:

```bash
helm uninstall pharmagraphrag -n pharmagraphrag      # if installed via Helm
kubectl delete -k k8s/                                # if applied raw
gcloud container clusters delete pharmagraphrag-autopilot \
  --region=us-central1 --project=pharmagraphrag --quiet
```

The cluster is **on-demand by design**: manifests live in this repo, you can `helm install`
in 5 minutes whenever you need the live demo (e.g. before an interview screen-share).

## Why this exists

PharmaGraphRAG's primary cloud deployment is **Cloud Run** (lower cost for low-traffic
demos, scales to zero). The Kubernetes path was added as part of the
[portfolio upgrade roadmap](../README.md#deployment-options) to demonstrate
production-grade orchestration patterns:

- Stateless API + UI Deployments with resource requests/limits
- Liveness, readiness and startup probes tuned for the embedding-model cold start
- HorizontalPodAutoscalers on CPU and memory
- ConfigMap + Secret separation
- Helm packaging with parameterized values
- CI/CD via GitHub Actions to GKE
