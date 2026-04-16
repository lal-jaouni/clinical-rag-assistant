#!/bin/bash
set -e

REPO="lal-jaouni/clinical-rag-assistant"

echo "Setting up GitHub repository for clinical-rag-assistant..."
echo "Repository: $REPO"
echo ""

# Create labels
echo "Creating labels..."
gh label create "phase-1" --color "0E8A16" --description "Infrastructure setup" -R "$REPO" 2>/dev/null || echo "  (phase-1 label already exists)"
gh label create "phase-2" --color "1F883D" --description "Data ingestion" -R "$REPO" 2>/dev/null || echo "  (phase-2 label already exists)"
gh label create "phase-3" --color "2E7D32" --description "Embedding and retrieval" -R "$REPO" 2>/dev/null || echo "  (phase-3 label already exists)"
gh label create "phase-4" --color "3D7839" --description "Generation and API" -R "$REPO" 2>/dev/null || echo "  (phase-4 label already exists)"
gh label create "phase-5" --color "40876C" --description "Evaluation" -R "$REPO" 2>/dev/null || echo "  (phase-5 label already exists)"
gh label create "infrastructure" --color "D73A49" --description "Docker, DB, deployment" -R "$REPO" 2>/dev/null || echo "  (infrastructure label already exists)"
gh label create "evaluation" --color "A371F7" --description "Metrics and testing" -R "$REPO" 2>/dev/null || echo "  (evaluation label already exists)"
gh label create "documentation" --color "0366D6" --description "Docs and guides" -R "$REPO" 2>/dev/null || echo "  (documentation label already exists)"

echo "Labels created/verified!"
echo ""
echo "To create GitHub issues, run:"
echo "  python3 /home/laith/workspaces/clinical-rag-assistant/create_github_issues.py"
echo ""
