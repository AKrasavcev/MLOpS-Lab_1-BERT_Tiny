#!/bin/bash
uv venv .venv && source .venv/bin/activate
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r requirements.txt -r requirements-train.txt -r requirements-server.txt -r requirements-dev.txt
