.PHONY: help install test lint format clean run-ui docs docs-serve docs-clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	conda env update -f environment.yml

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
	./prusa_mcp_server/run.sh

docker-up:  ## Start Docker services
	./docker-compose-wrapper.sh up -d

docker-down:  ## Stop Docker services
	./docker-compose-wrapper.sh down

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
	docker-compose -f docker-compose.mcp.yml restart

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
