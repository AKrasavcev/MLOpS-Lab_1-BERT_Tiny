#!/bin/bash

gh auth login --hostname github.com --git-protocol ssh --web \
    --scopes write:packages,read:packages
    
OWNER=$(gh api user --jq '.login | ascii_downcase')
gh auth token | docker login ghcr.io -u "$OWNER" --password-stdin
export IMAGE_NAME=bert-tiny
export COMMIT_SHA=$(git rev-parse --short=12 HEAD)
export IMAGE=ghcr.io/$OWNER/$IMAGE_NAME:sha-$COMMIT_SHA
docker build -f docker/Dockerfile.cpu -t $IMAGE .           # or docker tag bert-tiny:latest $IMAGE
docker push $IMAGE