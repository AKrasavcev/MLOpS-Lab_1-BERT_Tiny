#!/bin/bash
# Make an image available inside Minikube (containerd, so it can't see the host's Docker images).
#   scripts/minikube_image.sh <image> <dockerfile>
# Prefers the CI-built image from GHCR; builds locally only if it can't be pulled (e.g. a [skip ci] commit).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${1:?image}"
DOCKERFILE="${2:?dockerfile}"

if minikube image ls | grep -qF "$IMAGE"; then
    echo "$IMAGE already in Minikube"
    exit 0
fi
if ! docker pull "$IMAGE"; then
    echo "::warning::$IMAGE not in the registry, building it locally from $DOCKERFILE"
    docker build -f "$DOCKERFILE" --build-arg GIT_COMMIT="$(git rev-parse --short=8 HEAD)" -t "$IMAGE" .
fi
minikube image load "$IMAGE"
echo "loaded $IMAGE into Minikube"
