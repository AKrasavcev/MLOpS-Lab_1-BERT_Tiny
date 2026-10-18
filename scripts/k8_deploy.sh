#!/bin/bash

kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml \
    -f k8s/mlflow.yaml -f k8s/postgres.yaml \
    -f k8s/resourcequota.yaml