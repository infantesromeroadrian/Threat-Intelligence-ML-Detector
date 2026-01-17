# 🛡️ Threat Intelligence Aggregator

**Automated threat intelligence collection, analysis, and alerting system.**

Combines NLP techniques (NER, LDA, BERT, Word2Vec) to aggregate, extract, and analyze cybersecurity threat intelligence from multiple sources.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DRIVING ADAPTERS                           │
│          FastAPI | CLI | Streamlit Dashboard                │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     DOMAIN                                  │
│  Entities: CVE, IOC, ThreatIntel, Topic, Alert             │
│  Pure Python, No Infrastructure Dependencies               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 DRIVEN ADAPTERS                             │
│  NVD/OTX Scrapers | spaCy NER | LDA | BERT | Word2Vec     │
│  SQLite | Slack | Email                                    │
└─────────────────────────────────────────────────────────────┘
```

**Architecture Style**: Hexagonal (Ports & Adapters)  
**Language**: Python 3.10+  
**Framework**: FastAPI, Streamlit

---

## 📦 Installation

### Option 1: Using Docker (Recommended for Production)

```bash
# Build image
docker build -t threat-intel-aggregator .

# Run container
docker run -p 8000:8000 -p 8501:8501 threat-intel-aggregator
```

### Option 2: Local Development

#### Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

#### With `uv` (Fast)
```bash
# Clone repository
git clone <repo-url>
cd AI-RedTeam-Course

# Install dependencies
uv pip install -r requirements-dev.txt --python 3.10

# Install spaCy model
python -m spacy download en_core_web_sm

# Setup pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

#### With `pip` (Traditional)
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Setup pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

---

## 🔑 Configuration

Edit `.env` file with your settings:

```bash
# Required: NVD API Key (free, register at https://nvd.nist.gov/developers/request-an-api-key)
NVD_API_KEY="your-api-key-here"

# Optional: AlienVault OTX API Key (free, register at https://otx.alienvault.com/)
OTX_API_KEY="your-otx-key-here"

# Optional: Slack webhook for alerts
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Database (default: SQLite)
DATABASE_URL="sqlite:///./threat_intel.db"

# Logging
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

---

## 🚀 Usage

### 1. Start API Server

```bash
# Development mode (auto-reload)
uvicorn src.threat_intelligence_aggregator.infrastructure.api.main:app --reload

# Production mode
uvicorn src.threat_intelligence_aggregator.infrastructure.api.main:app --host 0.0.0.0 --port 8000
```

API will be available at:
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

### 2. Start Dashboard

```bash
streamlit run src/threat_intelligence_aggregator/frontend/app.py
```

Dashboard: http://localhost:8501

### 3. CLI Usage

```bash
# Scrape recent CVEs
python -m src.threat_intelligence_aggregator.infrastructure.cli.commands scrape-cves --days 7

# Extract IOCs from documents
python -m src.threat_intelligence_aggregator.infrastructure.cli.commands extract-iocs

# Discover topics with LDA
python -m src.threat_intelligence_aggregator.infrastructure.cli.commands discover-topics --num-topics 10

# Generate alerts
python -m src.threat_intelligence_aggregator.infrastructure.cli.commands generate-alerts
```

---

## 📊 Features

### ✅ Implemented

- [x] **Domain Layer**: Pure Python entities (CVE, IOC, ThreatIntel, Topic, Alert)
- [x] **Ports**: Abstract interfaces for all infrastructure adapters
- [x] **Configuration**: Pydantic Settings with environment variables
- [x] **Logging**: Structured logging with structlog

### 🚧 In Progress (Session 2)

- [ ] **CVE Scraper**: NVD API integration
- [ ] **IOC Extractor**: spaCy NER for IP, domain, hash, CVE extraction
- [ ] **Topic Modeler**: LDA for discovering emerging threats
- [ ] **SQLite Repository**: Persistence layer

### 📋 Planned (Session 3+)

- [ ] **BERT Classifier**: Severity classification
- [ ] **Word2Vec**: Similarity search
- [ ] **OTX Scraper**: AlienVault threat feed integration
- [ ] **Alert System**: Intelligent alerting with Slack/Email
- [ ] **FastAPI Endpoints**: Full REST API
- [ ] **Streamlit Dashboard**: Interactive visualization with pyLDAvis

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/threat_intelligence_aggregator --cov-report=html

# Run specific test types
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests

# Type checking
mypy src/threat_intelligence_aggregator --strict --ignore-missing-imports

# Linting & Formatting
ruff check .             # Check linting
ruff format .            # Format code
```

---

## 📂 Project Structure

```
src/threat_intelligence_aggregator/
├── domain/                   # ❤️ Core business logic (no external deps)
│   ├── entities/            # CVE, IOC, ThreatIntel, Topic, Alert
│   ├── ports/               # Abstract interfaces (Protocol)
│   └── services/            # Domain services
├── application/             # 🎯 Use cases
│   ├── use_cases/           # Scrape, extract, analyze
│   └── dtos/                # Data transfer objects
├── infrastructure/          # 🔌 External adapters
│   ├── adapters/
│   │   ├── scrapers/        # NVD, OTX
│   │   ├── ml_models/       # spaCy, LDA, BERT, Word2Vec
│   │   ├── repositories/    # SQLite
│   │   └── notifiers/       # Slack, Email
│   ├── api/                 # FastAPI
│   ├── cli/                 # CLI commands
│   └── config/              # Settings, logging
├── frontend/                # 📊 Streamlit dashboard
└── tests/                   # 🧪 Unit, integration, e2e
```

---

## 🛠️ Development

### Code Quality Standards

- **Type Hints**: 100% coverage (mypy --strict)
- **Formatting**: `ruff format` (replaces black + isort)
- **Linting**: `ruff check` (replaces flake8, pylint)
- **Pre-commit**: Automated quality gates
- **Testing**: pytest + hypothesis (property-based testing)
- **Architecture**: Hexagonal - domain has ZERO infrastructure dependencies

### Adding Dependencies

```bash
# 1. Edit pyproject.toml (add package to dependencies or [project.optional-dependencies])

# 2. Regenerate requirements files
uv pip compile pyproject.toml -o requirements.txt --python-version 3.10
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt --python-version 3.10

# 3. Reinstall
uv pip install -r requirements-dev.txt --python 3.10
```

### Pre-commit Hooks

Automatically run before each commit:
- Trailing whitespace removal
- YAML/JSON/TOML validation
- Secret detection
- Ruff formatting + linting
- mypy type checking

```bash
# Run manually on all files
pre-commit run --all-files
```

---

## 📈 Roadmap

### Session 1 ✅ (Completed)
- Setup: Project structure, dependencies, configuration
- Domain: Entities (CVE, IOC, ThreatIntel, Topic, Alert)
- Domain: Ports (11 interfaces)

### Session 2 🚧 (In Progress)
- Domain services
- NVD CVE scraper (mock data)
- NER IOC extractor (spaCy)
- SQLite repositories

### Session 3 📅 (Planned)
- LDA topic modeler
- BERT severity classifier
- Word2Vec similarity
- OTX scraper

### Session 4 📅 (Planned)
- FastAPI REST API
- CLI commands
- Alert system

### Session 5 📅 (Planned)
- Streamlit dashboard
- Docker deployment
- Tests + Documentation

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

This is an educational project from the **AI-RedTeam-Course**.

For issues or suggestions, please open an issue on GitHub.

---

## 🔗 Related Resources

- [NVD API](https://nvd.nist.gov/developers)
- [AlienVault OTX](https://otx.alienvault.com/)
- [MITRE ATT&CK](https://attack.mitre.org/)
- [spaCy Documentation](https://spacy.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Version**: 0.1.0  
**Status**: 🚧 In Development  
**Last Updated**: 2026-01-17
