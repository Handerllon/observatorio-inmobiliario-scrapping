#!/bin/bash
set -e

export AWS_PROFILE=uade-valorar
REGION="us-east-2"
ECR_REPO="650532183679.dkr.ecr.us-east-2.amazonaws.com/valorar-inference-ohe-py12"
IMAGE_TAG="latest"
STACK_NAME="valorar-inference-py12"

echo "Building Docker image for x86_64..."
docker build --platform linux/amd64 -t lambda-docker-image-py12 .

echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 650532183679.dkr.ecr.$REGION.amazonaws.com

echo "Tagging image..."
docker tag lambda-docker-image-py12:latest $ECR_REPO:$IMAGE_TAG

echo "Pushing image to ECR..."
docker push $ECR_REPO:$IMAGE_TAG

echo "Deploying with SAM..."
sam deploy --stack-name $STACK_NAME --region $REGION --capabilities CAPABILITY_IAM --resolve-s3

echo "Deployment complete!"
