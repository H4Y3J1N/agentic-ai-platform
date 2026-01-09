#!/bin/bash
# 무중단 배포 스크립트
set -e
SERVICE_NAME="ecommerce-ai"
NAMESPACE="agentic-ai"
IMAGE_TAG="${1:-latest}"

echo "🚀 Zero-Downtime Deployment: $SERVICE_NAME:$IMAGE_TAG"
echo "1. Pre-deployment checks"
kubectl get deployment $SERVICE_NAME -n $NAMESPACE

echo "2. DB Migration (Backward Compatible)"
python scripts/migrations/run_migrations.py

echo "3. Build & Push Image"
docker build -t $SERVICE_NAME:$IMAGE_TAG .

echo "4. Update Deployment"
kubectl set image deployment/$SERVICE_NAME api=$SERVICE_NAME:$IMAGE_TAG -n $NAMESPACE

echo "5. Wait for Rollout"
kubectl rollout status deployment/$SERVICE_NAME -n $NAMESPACE --timeout=10m

echo "✅ Deployment Complete!"
