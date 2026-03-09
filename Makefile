.PHONY: help install test lint format clean run-ui docs docs-serve docs-clean mmore-eval-up mmore-eval-down mmore-eval-rebuild mmore-eval-logs mmore-eval-status mmore-eval-run sync-thesis sync-thesis-figures sync-thesis-tables sync-thesis-prompts sync-thesis-dry-run sync-thesis-commit sync-paper sync-paper-dry-run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e .[dev]

test:  ## Run all tests
	pytest

test-fast:  ## Run only fast tests (skip slow integration tests)
	pytest -m "not slow"

test-cov:  ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint:  ## Check code quality with ruff and mypy
	ruff check .
	mypy .

format:  ## Format code with ruff
	ruff format .
	ruff check --fix .

clean:  ## Clean cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-outputs:  ## Clean all files in the outputs folder
	rm -rf outputs/*
	@echo "✓ Outputs folder cleaned"

clean-results:  ## Clean all files in the results folder
	rm -rf results/*
	@echo "✓ Results folder cleaned"

clean-all: clean clean-outputs clean-results  ## Clean everything (cache, outputs, and results)
	@echo "✓ Full cleanup complete"

run-ui:  ## Start the Streamlit UI
	streamlit run src/ui/streamlit_app.py

run-mcp:  ## Start the standalone Prusa MCP server
	./services/prusa_mcp_server/run.sh

docker-up:  ## Start Docker services and host service
	@echo "Starting host service in background..."
	@pkill -f "python.*host_service.py" 2>/dev/null || true
	@nohup python services/host_service.py > /tmp/host_service.log 2>&1 &
	@sleep 1
	@if curl -s http://localhost:9999/ > /dev/null 2>&1; then \
		echo "✓ Host service running on http://localhost:9999"; \
	else \
		echo "⚠ Host service may have failed to start. Check /tmp/host_service.log"; \
	fi
	./docker-compose-wrapper.sh up -d

docker-down:  ## Stop Docker services and host service
	./docker-compose-wrapper.sh down
	@echo "Stopping host service..."
	@pkill -f "python.*host_service.py" 2>/dev/null || true
	@echo "✓ Host service stopped"

docker-rebuild:  ## Rebuild and restart all Docker services
	./docker-compose-wrapper.sh down
	./docker-compose-wrapper.sh up -d --build

docker-rebuild-chatbot:  ## Rebuild only the chatbot service
	./docker-compose-wrapper.sh up -d --build chatbot

docker-rebuild-mmore:  ## Rebuild only the MMORE RAG service
	./docker-compose-wrapper.sh up -d --build mmore-rag-service

docker-rebuild-prusa:  ## Rebuild only the Prusa MCP server
	./docker-compose-wrapper.sh up -d --build prusa-mcp-server

docker-restart:  ## Restart Docker services without rebuilding
	docker-compose restart

# MMORE RAG Evaluation (runs in parallel with main services)
mmore-eval-up:  ## Start MMORE RAG service for evaluation (port 8001)
	docker compose -f docker-compose.mmore-eval.yml up -d

mmore-eval-down:  ## Stop MMORE RAG evaluation service
	docker compose -f docker-compose.mmore-eval.yml down

mmore-eval-rebuild:  ## Rebuild and restart MMORE RAG evaluation service
	docker compose -f docker-compose.mmore-eval.yml up -d --build

mmore-eval-logs:  ## Show logs for MMORE RAG evaluation service
	docker compose -f docker-compose.mmore-eval.yml logs -f

mmore-eval-status:  ## Show status of MMORE RAG evaluation service
	docker compose -f docker-compose.mmore-eval.yml ps

mmore-eval-run:  ## Run evaluation with MMORE eval service (uses port 8001). Usage: make mmore-eval-run ARGS="--problem beams2d --samples 1"
	MMORE_RAG_URL=http://localhost:8001 python benchmarks/evaluations/evaluate_agent.py --mmore $(ARGS)

# Build documentation
.PHONY: docs
docs:
	python docs/generate_api_docs.py
	cd docs && $(MAKE) dirhtml

# Generate API documentation only
.PHONY: docs-api
docs-api:  ## Generate API documentation from source code
	python docs/generate_api_docs.py

# Serve documentation with live reload (recommended for development)
.PHONY: docs-watch
docs-watch:
	python docs/generate_api_docs.py
	sphinx-autobuild -b dirhtml docs/source docs/build/dirhtml

# Serve documentation locally
.PHONY: docs-serve
docs-serve:
	cd docs/build/dirhtml && python -m http.server 8001

docs-clean:  ## Clean documentation build files
	cd docs && $(MAKE) clean

install-docs:  ## Install documentation dependencies
	pip install -e ".[docs]"

check-all: lint test  ## Run all checks (lint + test)
	@echo "\n✓ All checks passed!"

# Thesis sync targets
sync-thesis:  ## Sync all content (figures, tables, prompts) to thesis submodule
	python scripts/sync_thesis.py --all

sync-thesis-figures:  ## Sync only figures to thesis submodule
	python scripts/sync_thesis.py --figures

sync-thesis-tables:  ## Sync only tables to thesis submodule
	python scripts/sync_thesis.py --tables

sync-thesis-prompts:  ## Sync only prompts to thesis submodule
	python scripts/sync_thesis.py --prompts

sync-thesis-dry-run:  ## Preview what would be synced (no changes)
	python scripts/sync_thesis.py --all --dry-run

sync-thesis-commit:  ## Sync all content and auto-commit in thesis submodule
	python scripts/sync_thesis.py --all --auto-commit

# Paper sync targets
sync-paper:  ## Sync prompts to paper submodule
	python scripts/sync_paper.py --prompts

sync-paper-dry-run:  ## Preview what would be synced to paper
	python scripts/sync_paper.py --prompts --dry-run
