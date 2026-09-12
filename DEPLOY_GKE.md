# Deploying to GKE

This mirrors the existing Fly.io deployment (see `DEPLOY.md`) onto a real
Kubernetes cluster on GCP — same image, same `/health` check, same
`ANTHROPIC_API_KEY` secret pattern, same `/data` volume for the SQLite file.

**Architecture note:** the Deployment runs a single replica on purpose. The
app writes to one SQLite file on a `ReadWriteOnce` volume, and SQLite only
supports one writer at a time — a second replica would either serialize on
file locks or corrupt the database, and RWO wouldn't even let a second pod
mount the volume from a different node. Real horizontal scaling (and an HPA)
would require externalizing state first, e.g. migrating to Cloud SQL for
Postgres. That's a deliberate next step, not an oversight — see the comment
in `k8s/deployment.yaml`.

## Cost

- GKE's per-cluster management fee is waived for one zonal cluster per
  billing account — you still pay for the underlying Compute Engine VM(s).
  A single `e2-small` node running 24/7 is roughly $12-15/month.
- The Service is `ClusterIP` (no public IP), so there's no separate
  Load Balancer charge (~$18/mo) unless you add an Ingress later.
- New Google Cloud billing accounts get $300 in credit, good for 90 days.

## One-time setup (do this yourself — it's your GCP account/billing)

```bash
# 1. Authenticate the gcloud CLI (opens a browser)
gcloud auth login

# 2. Create (or select) a project — pick a globally-unique ID
gcloud projects create fitness-agent-<your-suffix> --name="fitness-agent"
gcloud config set project fitness-agent-<your-suffix>

# 3. Link billing (requires a billing account already set up in the console —
#    console.cloud.google.com/billing — this is the one step gcloud can't do
#    non-interactively)
gcloud billing accounts list
gcloud billing projects link fitness-agent-<your-suffix> \
  --billing-account=<BILLING_ACCOUNT_ID>

# 4. Enable the APIs this deploy needs
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com

# 5. Artifact Registry repo for the image
gcloud artifacts repositories create fitness-agent \
  --repository-format=docker \
  --location=us-central1

# 6. The cluster itself — zonal, smallest reasonable node pool.
#    (Autopilot is simpler to operate but bills per-pod-resource rather than
#    per-node, which usually costs more for a single always-on 256Mi pod —
#    zonal standard is the cheaper choice here.)
gcloud container clusters create fitness-agent \
  --zone=us-central1-a \
  --num-nodes=1 \
  --machine-type=e2-small \
  --disk-size=30

# 7. Point kubectl at it
gcloud container clusters get-credentials fitness-agent --zone=us-central1-a
```

## Manual first deploy

```bash
# Build and push the image (reuses the existing Dockerfile as-is)
gcloud auth configure-docker us-central1-docker.pkg.dev
IMAGE=us-central1-docker.pkg.dev/<PROJECT_ID>/fitness-agent/fitness-agent:v1
docker build -t "$IMAGE" .
docker push "$IMAGE"

# Namespace, config, storage, service first
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/service.yaml

# Secret — from the imperative command, not a checked-in file (see
# k8s/secret.example.yaml for why)
kubectl create secret generic fitness-agent-secrets \
  --namespace fitness-agent \
  --from-literal=ANTHROPIC_API_KEY=<your real key> \
  --dry-run=client -o yaml | kubectl apply -f -

# Deployment — patch in the real image first since deployment.yaml's
# `image: fitness-agent` is a placeholder CI fills in via `kubectl set image`
sed "s|image: fitness-agent|image: $IMAGE|" k8s/deployment.yaml | kubectl apply -f -

kubectl rollout status deployment/fitness-agent -n fitness-agent
```

## Verify it's live

```bash
kubectl get pods -n fitness-agent
kubectl port-forward -n fitness-agent svc/fitness-agent 8000:80
curl http://localhost:8000/health
# {"status": "ok"}
```

## Wiring up `deploy-gke.yml` (CI/CD via Workload Identity Federation)

The included `.github/workflows/deploy-gke.yml` builds, pushes, and rolls
out on manual trigger (`workflow_dispatch`) — not on every push, since each
deploy has a brief availability gap (see the architecture note above). It
authenticates via **Workload Identity Federation**, not a downloaded
service-account JSON key, so there's no long-lived credential sitting in a
GitHub secret:

```bash
# Create a deploy service account with just the roles it needs
gcloud iam service-accounts create fitness-agent-deployer

PROJECT_NUMBER=$(gcloud projects describe <PROJECT_ID> --format='value(projectNumber)')

for ROLE in roles/container.developer roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:fitness-agent-deployer@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="$ROLE"
done

# Workload Identity Pool + Provider scoped to this one GitHub repo
gcloud iam workload-identity-pools create github \
  --location=global

gcloud iam workload-identity-pools providers create-oidc github \
  --location=global \
  --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='chpham92/fitness-agent'"

gcloud iam service-accounts add-iam-policy-binding \
  fitness-agent-deployer@<PROJECT_ID>.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/chpham92/fitness-agent"
```

Then, in the GitHub repo settings:

- **Settings → Secrets and variables → Actions → Secrets:**
  - `GCP_WIF_PROVIDER` — `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github/providers/github`
  - `GCP_DEPLOY_SA` — `fitness-agent-deployer@<PROJECT_ID>.iam.gserviceaccount.com`
- **Settings → Secrets and variables → Actions → Variables:**
  - `GCP_PROJECT_ID`, `GCP_REGION` (`us-central1`), `GKE_CLUSTER_NAME` (`fitness-agent`)

Run it from the Actions tab → "Deploy to GKE" → Run workflow.

## Tearing down (avoid ongoing charges)

```bash
gcloud container clusters delete fitness-agent --zone=us-central1-a
gcloud artifacts repositories delete fitness-agent --location=us-central1
```
