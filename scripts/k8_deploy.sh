#!/bin/bash

kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml \
    -f k8s/mlflow-deployment.yaml -f k8s/mlflow-pvc.yaml \
    -f k8s/mlflow-service.yaml -f k8s/postgres.yaml \
    -f k8s/resourcequota.yaml