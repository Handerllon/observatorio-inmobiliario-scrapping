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

echo "\n*** Note: sam build needs docker server running in the host ***\n"
sam build && sam deploy --guided
