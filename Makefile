# Clinical RAG Assistant -- common developer commands
#
# Run `make help` to list targets.

SHELL := /bin/bash

.PHONY: help install dev-install up down logs health pull-ollama test lint format clean ui api

help:
	@echo "Clinical RAG -- Makefile targets:"
	@echo "  make install        Install package (prod deps only)"
	@echo "  make dev-install    Install with dev + eval extras"
	@echo "  make up             Start Postgres + Ollama via docker compose"
	@echo "  make down           Stop containers (preserves data volumes)"
	@echo "  make logs           Tail docker-compose logs"
	@echo "  make health         Run Phase 1 health check"
	@echo "  make pull-ollama    Pull the default Ollama LLM (ollama/llama3.1:8b)"
	@echo "  make test           Run pytest"
	@echo "  make lint           Run ruff + mypy"
	@echo "  make format         Auto-format with ruff"
	@echo "  make api            Start FastAPI backend"
	@echo "  make ui             Start Streamlit demo UI"
	@echo "  make clean          Remove build artifacts + __pycache__"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,eval]"

up:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example -- edit before running again."; fi
	docker compose up -d
	@echo ""
	@echo "Waiting for Postgres to be healthy..."
	@until docker compose ps postgres --format json | grep -q '"Health":"healthy"'; do sleep 1; done
	@echo "Postgres ready. Run 'make health' to verify."

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

health:
	python scripts/health_check.py

api:
	PYTHONPATH=src python -m api.app

ui:
	PYTHONPATH=src streamlit run src/ui/app.py

pull-ollama:
	docker exec clinical-rag-ollama ollama pull llama3.1:8b

test:
	pytest tests/

lint:
	ruff check src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
