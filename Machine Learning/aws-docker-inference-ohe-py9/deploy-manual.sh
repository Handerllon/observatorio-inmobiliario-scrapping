#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Carga variables de entorno locales (incluido ACCOUNT_ID) si existe .env
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

export AWS_PROFILE="${AWS_PROFILE:-uade-valorar}"
REGION="us-east-2"
ACCOUNT_ID=${ACCOUNT_ID:-$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query Account --output text)}
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/valorar-inference-ohe-py9"
IMAGE_TAG="latest"
STACK_NAME="valorar-inference-py9"

echo "Building Docker image for x86_64..."
docker build --platform linux/amd64 -t lambda-docker-image-py9 .

echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Tagging image..."
docker tag lambda-docker-image-py9:latest $ECR_REPO:$IMAGE_TAG

echo "Pushing image to ECR..."
docker push $ECR_REPO:$IMAGE_TAG

echo "Deploying with SAM..."
sam deploy --stack-name $STACK_NAME --region $REGION --capabilities CAPABILITY_IAM --resolve-s3

echo "Deployment complete!"
