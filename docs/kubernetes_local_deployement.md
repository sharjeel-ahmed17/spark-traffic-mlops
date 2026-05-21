- install kubectl
- create folder for k8s
- create deployement.yaml and services.yaml files
- start minikube

- minikube start --driver=docker
- minikube image load <your_image_name>:latest

- minikube image ls | grep <your_image_name>

## set secret varaibles ;

```
kubectl create secret generic mlflow-secrets \
 --from-literal=tracking_uri='YOUR_MLFLOW_TRACKING_URI' \
 --from-literal=username='YOUR_DAGS_HUB_USERNAME' \
 --from-literal=password='YOUR_DAGS_HUB_TOKEN'
```

## apply kubernetes minifest

```

kubectl apply -f k8s/deployment.yaml

kubectl apply -f k8s/service.yaml
```

- minikube tunnel
- kubectl get pods
- kubectl get service ship-fuel-prediction-service
