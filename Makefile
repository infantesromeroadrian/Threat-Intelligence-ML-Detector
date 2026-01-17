# =============================================================================
# Threat Intelligence Aggregator - Makefile
# Simplified Docker commands and development tasks
# =============================================================================

.PHONY: help build up down logs clean test lint format install-local

# Default target
help:
	@echo "Threat Intelligence Aggregator - Available Commands"
	@echo ""
	@echo "🐳 Docker Commands (Production):"
	@echo "  make build         - Build Docker images"
	@echo "  make up            - Start all services"
	@echo "  make down          - Stop all services"
	@echo "  make restart       - Restart all services"
	@echo "  make logs          - View logs (all services)"
	@echo "  make logs-api      - View API logs"
	# @echo "  make logs-dash     - View Dashboard logs" # No dashboard yet
	@echo ""
	@echo "🔧 Docker Commands (Development):"
	@echo "  make dev-build     - Build development images"
	@echo "  make dev-up        - Start services in dev mode (hot-reload)"
	@echo "  make dev-down      - Stop dev services"
	@echo "  make dev-logs      - View dev logs"
	@echo ""
	@echo "🧹 Cleanup Commands:"
	@echo "  make clean         - Stop services and remove containers"
	@echo "  make clean-all     - Remove containers, volumes, and images"
	@echo "  make prune         - Clean Docker system (caution!)"
	@echo ""
	@echo "🔨 CLI Commands:"
	@echo "  make cli CMD='...' - Run CLI command in container"
	@echo "  make shell         - Open shell in API container"
	@echo ""
	@echo "📥 Data Scraping Commands:"
	@echo "  make scrape-data   - Scrape CVEs (7 days) + OTX (24h) with notifications"
	@echo "  make scrape-quick  - Quick scrape (1 day, 12h, no notifications)"
	@echo "  make scrape-nvd    - Scrape NVD CVEs only (7 days)"
	@echo "  make scrape-otx    - Scrape OTX threats only (24 hours)"
	@echo "  make check-db      - Check database statistics"
	@echo ""
	@echo "💻 Local Development (without Docker):"
	@echo "  make install-local - Install dependencies locally with uv"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting (ruff)"
	@echo "  make format        - Format code (ruff)"
	@echo "  make typecheck     - Run type checking (mypy)"
	@echo "  make pre-commit    - Run pre-commit hooks"

# =============================================================================
# Docker Commands - Production
# =============================================================================

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart: down up

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

# logs-dash:
# 	docker-compose logs -f dashboard  # No dashboard service yet

ps:
	docker-compose ps

# =============================================================================
# Docker Commands - Development
# =============================================================================

dev-build:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

dev-up:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-down:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

dev-logs:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# =============================================================================
# CLI Commands
# =============================================================================

cli:
	docker-compose run --rm cli python -m src.threat_intelligence_aggregator.infrastructure.cli.commands $(CMD)

shell:
	docker-compose exec api /bin/bash

# shell-dash:
# 	docker-compose exec dashboard /bin/bash  # No dashboard service yet

# Example CLI commands
scrape-cves:
	docker-compose run --rm cli python -m src.threat_intelligence_aggregator.infrastructure.cli.commands scrape-cves --days 7

extract-iocs:
	docker-compose run --rm cli python -m src.threat_intelligence_aggregator.infrastructure.cli.commands extract-iocs

discover-topics:
	docker-compose run --rm cli python -m src.threat_intelligence_aggregator.infrastructure.cli.commands discover-topics --num-topics 10

# =============================================================================
# Cleanup Commands
# =============================================================================

clean:
	docker-compose down -v --remove-orphans

clean-all:
	docker-compose down -v --rmi all --remove-orphans

prune:
	@echo "⚠️  This will remove ALL unused Docker data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker system prune -af --volumes; \
	fi

# =============================================================================
# Local Development (without Docker)
# =============================================================================

install-local:
	@echo "📦 Installing dependencies with uv..."
	uv sync
	@echo "📥 Downloading spaCy model..."
	python -m spacy download en_core_web_sm
	@echo "🔧 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Installation complete!"

test:
	pytest

test-cov:
	pytest --cov=src/threat_intelligence_aggregator --cov-report=html --cov-report=term

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src/threat_intelligence_aggregator --strict --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# =============================================================================
# ML Model Training
# =============================================================================

train-models:
	@echo "🧠 Training ML models (LDA + Word2Vec)..."
	python scripts/train_ml_models.py --all

train-lda:
	@echo "🧠 Training LDA topic model..."
	python scripts/train_ml_models.py --model lda

train-word2vec:
	@echo "🧠 Training Word2Vec similarity model..."
	python scripts/train_ml_models.py --model word2vec

# =============================================================================
# Data Scraping & Population
# =============================================================================

scrape-data:
	@echo "📥 Scraping threat intelligence data..."
	docker exec threat-intel-api python scripts/scrape_and_populate.py --days 7 --hours 24 --notify

scrape-quick:
	@echo "📥 Quick scrape (1 day, no notifications)..."
	docker exec threat-intel-api python scripts/scrape_and_populate.py --days 1 --hours 12

scrape-nvd:
	@echo "📥 Scraping NVD CVEs only..."
	docker exec threat-intel-api python scripts/scrape_and_populate.py --days 7 --skip-otx

scrape-otx:
	@echo "📥 Scraping OTX threats only..."
	docker exec threat-intel-api python scripts/scrape_and_populate.py --hours 24 --skip-nvd

check-db:
	@echo "🔍 Checking database statistics..."
	@docker exec threat-intel-api sqlite3 /app/data/threat_intel.db "SELECT 'CVEs: ' || COUNT(*) FROM cves; SELECT 'IOCs: ' || COUNT(*) FROM iocs; SELECT 'Threats: ' || COUNT(*) FROM threat_intel; SELECT 'Alerts: ' || COUNT(*) FROM alerts;"

# =============================================================================
# Health Checks
# =============================================================================

health:
	@echo "🔍 Checking API health..."
	@curl -f http://localhost:8000/health || echo "❌ API not responding"
	# @echo ""
	# @echo "🔍 Checking Dashboard..."
	# @curl -f http://localhost:8501 || echo "❌ Dashboard not responding"  # No dashboard yet

# =============================================================================
# Requirements Generation
# =============================================================================

requirements:
	@echo "📦 Regenerating requirements files..."
	uv pip compile pyproject.toml -o requirements.txt --python-version 3.10
	uv pip compile pyproject.toml --extra dev -o requirements-dev.txt --python-version 3.10
	@echo "✅ Requirements updated!"
