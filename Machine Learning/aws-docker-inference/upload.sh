#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

export AWS_PROFILE="${AWS_PROFILE:-uade-valorar}"

aws s3 cp s3://observatorio-inmobiliario/models ./models --recursive
