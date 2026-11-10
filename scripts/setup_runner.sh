#!/bin/bash
# One-time setup of the self-hosted GitHub Actions runner used by CD (.github/workflows/cd.yml).
# It runs on this WSL machine, so jobs can reach Minikube, kubectl and Docker.
#   1. GitHub -> repo Settings -> Actions -> Runners -> New self-hosted runner (Linux x64): copy the token.
#   2. bash scripts/setup_runner.sh <TOKEN>
# Installs into ~/actions-runner with labels self-hosted,linux,minikube and starts it as a systemd service.
set -euo pipefail
TOKEN="${1:?usage: setup_runner.sh <registration-token>}"
REPO_URL="${REPO_URL:-https://github.com/AKrasavcev/MLOpS-Lab_1-BERT_Tiny}"
RUNNER_VERSION="${RUNNER_VERSION:-2.329.0}"
DIR="$HOME/actions-runner"

# kubectl on PATH (Minikube ships one; the CD jobs also add it themselves if missing).
if ! command -v kubectl >/dev/null; then
    minikube kubectl -- version --client >/dev/null
    mkdir -p "$HOME/.local/bin"
    ln -sf "$(ls -d "$HOME"/.minikube/cache/linux/amd64/*/kubectl | sort -V | tail -1)" "$HOME/.local/bin/kubectl"
    echo "linked kubectl into ~/.local/bin"
fi

mkdir -p "$DIR" && cd "$DIR"
if [ ! -x ./config.sh ]; then
    curl -fsSL -o runner.tgz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
    tar xzf runner.tgz && rm runner.tgz
fi
./config.sh --unattended --replace --url "$REPO_URL" --token "$TOKEN" \
    --name "$(hostname)-minikube" --labels minikube --work _work

# minikube detects WSL through WSL_DISTRO_NAME. systemd services don't get it, and without it minikube
# expects the API server at the node IP instead of WSL's 127.0.0.1 port mapping ("kubeconfig: Misconfigured").
if [ -n "${WSL_DISTRO_NAME:-}" ] && ! grep -q '^WSL_DISTRO_NAME=' .env 2>/dev/null; then
    echo "WSL_DISTRO_NAME=$WSL_DISTRO_NAME" >> .env
fi

# Service runs as the current user, so it uses this user's ~/.kube/config, ~/.minikube and docker group.
sudo ./svc.sh install "$USER"
sudo ./svc.sh start
sudo ./svc.sh status | head -5
