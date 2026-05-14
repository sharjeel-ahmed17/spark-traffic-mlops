# Kubernetes Manifests for Spark Traffic MLOps

## Prerequisites

- Kubernetes cluster 1.25+
- kubectl configured with cluster access
- Docker registry for storing images
- Optional: Ingress controller (nginx) for external access

## Quick Start

### 1. Build and push Docker image

```bash
docker build -t your-docker-registry/spark-traffic:latest .
docker push your-docker-registry/spark-traffic:latest
```

Update `kustomization.yaml` with your registry:

```yaml
images:
  - name: spark-traffic
    newName: your-docker-registry/spark-traffic
    newTag: latest
```

### 2. Apply manifests

Using Kustomize (recommended):

```bash
kubectl apply -k .
```

Or apply individual files:

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f mlflow-deployment.yaml
kubectl apply -f mlflow-service.yaml
kubectl apply -f fastapi-deployment.yaml
kubectl apply -f fastapi-service.yaml
kubectl apply -f streamlit-deployment.yaml
kubectl apply -f streamlit-service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f services.yaml
```

### 3. Verify deployment

```bash
kubectl get pods -n spark-traffic
kubectl get services -n spark-traffic
kubectl get ingress -n spark-traffic
```

## Services

| Service | Type | Port | Description |
|---------|------|------|-------------|
| mlflow-service | ClusterIP | 5000 | MLflow tracking server |
| fastapi-service | ClusterIP | 8000 | FastAPI inference API |
| streamlit-service | LoadBalancer | 80 | Streamlit dashboard |
| ingress | HTTP | 80/443 | External routing |

## Scaling

HPA is configured for FastAPI and Streamlit:
- FastAPI: 2-5 replicas, scales on CPU/memory utilization
- Streamlit: 2-5 replicas, scales on CPU utilization

## Configuration

Edit `configmap.yaml` to configure:
- MLFLOW_TRACKING_URI: MLflow server address
- API_URL: FastAPI service address

## Storage

The PVC (`models-pvc`) provides shared storage for model artifacts. Adjust size in `pvc.yaml` as needed.

## Cleanup

```bash
kubectl delete -k .
```


To deploy:
  # Update image in kustomization.yaml with your registry
  kubectl apply -k k8s/