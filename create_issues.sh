#!/bin/bash

# Create GitHub labels for clinical-rag-assistant
REPO="lal-jaouni/clinical-rag-assistant"

echo "Creating labels..."
gh label create "phase-1" --color "0E8A16" --description "Infrastructure setup" -R "$REPO" 2>/dev/null || true
gh label create "phase-2" --color "1F883D" --description "Data ingestion" -R "$REPO" 2>/dev/null || true
gh label create "phase-3" --color "2E7D32" --description "Embedding and retrieval" -R "$REPO" 2>/dev/null || true
gh label create "phase-4" --color "3D7839" --description "Generation and API" -R "$REPO" 2>/dev/null || true
gh label create "infrastructure" --color "D73A49" --description "Docker, DB, deployment" -R "$REPO" 2>/dev/null || true
gh label create "evaluation" --color "A371F7" --description "Metrics and testing" -R "$REPO" 2>/dev/null || true
gh label create "documentation" --color "0366D6" --description "Docs and guides" -R "$REPO" 2>/dev/null || true

echo "Labels created successfully!"
