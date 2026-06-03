# PharmaGraphRAG Helm chart

A Helm 3 chart that packages PharmaGraphRAG (API + UI) for Kubernetes.

## TL;DR

```bash
# Install / upgrade (uses an existing Secret by default)
helm upgrade --install pharmagraphrag ./helm/pharmagraphrag \
  --namespace pharmagraphrag --create-namespace \
  --set secrets.create=true \
  --set secrets.values.GEMINI_API_KEY="$GEMINI_API_KEY" \
  --set secrets.values.NEO4J_URI="$NEO4J_URI" \
  --set secrets.values.NEO4J_PASSWORD="$NEO4J_PASSWORD"

# Uninstall (frees up GKE costs)
helm uninstall pharmagraphrag -n pharmagraphrag
```

## What it deploys

- `Deployment` + `Service` + `HPA` for the FastAPI **API**.
- `Deployment` + `Service` (LoadBalancer by default) + `HPA` for the Streamlit **UI**.
- `ConfigMap` with non-secret config (LLM model, Neo4j user, Chroma path, inter-service URL).
- `Secret` with credentials (created by the chart in dev, or referenced from an existing one in prod).
- Optional `Ingress` + GKE `ManagedCertificate` for custom-domain HTTPS.
- Production-grade probes: startup probe sized for ~50s embedding-model cold start, plus liveness + readiness.

## Key values

| Key | Default | Notes |
|---|---|---|
| `image.registry` | `gcr.io/pharmagraphrag` | Change to your registry |
| `image.tag` | `latest` | Falls back to `Chart.appVersion` if empty |
| `api.replicaCount` / `ui.replicaCount` | `1` | Ignored when `autoscaling.enabled` |
| `api.autoscaling.{min,max}Replicas` | `1` / `3` | CPU 70%, memory 80% targets |
| `ui.autoscaling.{min,max}Replicas` | `1` / `2` | CPU 75% target |
| `ui.service.type` | `LoadBalancer` | Use `ClusterIP` if exposing via Ingress |
| `secrets.create` | `false` | `true` to let the chart create the Secret (dev only) |
| `secrets.existingSecret` | `pharmagraphrag-secrets` | Name when `create=false` |
| `ingress.enabled` | `false` | Set `true` + provide `ingress.host` to use GKE Ingress |

## Render without installing

```bash
helm template demo ./helm/pharmagraphrag --namespace pharmagraphrag
```

## Lint

```bash
helm lint ./helm/pharmagraphrag
```

## Why a chart and not just raw manifests?

Both are checked in:

- [`k8s/`](../../k8s) — raw manifests, useful for understanding what gets created.
- [`helm/pharmagraphrag/`](.) — Helm chart, recommended for real deploys: parameterization,
  upgrade/rollback, environment-specific values files, NOTES output, secret indirection.

CI/CD ([`deploy-gke.yml`](../../.github/workflows/deploy-gke.yml)) uses the Helm chart.
