# 📦 Dependency Management

## Overview

This project uses **`pyproject.toml`** as the **single source of truth** for dependencies, following PEP 621 standards.

**Generated files for Docker/CI/CD:**
- `requirements.txt` - Production dependencies (125 packages)
- `requirements-dev.txt` - Development dependencies (151 packages)

---

## 🔄 Dependency Workflow

```
pyproject.toml (source of truth)
    │
    ├─► uv pip compile → requirements.txt (pinned versions)
    ├─► uv pip compile --extra dev → requirements-dev.txt
    │
    └─► uv sync → .venv/ + uv.lock (local development)
```

---

## 📝 Files Explained

| File | Purpose | Tracked in Git? |
|------|---------|-----------------|
| `pyproject.toml` | **Source of truth** - declares dependencies | ✅ Yes |
| `requirements.txt` | **Docker/CI** - pinned production deps | ✅ Yes (generated) |
| `requirements-dev.txt` | **Docker dev** - pinned dev deps | ✅ Yes (generated) |
| `uv.lock` | **Local dev** - exact versions from uv sync | ✅ Yes (if using uv) |
| `poetry.lock` | Poetry lockfile (not used) | ❌ No |
| `Pipfile.lock` | Pipenv lockfile (not used) | ❌ No |

---

## 🚀 Installation Methods

### For Docker (Production)

```dockerfile
# Dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

### For Local Development

#### Option A: Using `uv` (Recommended)

```bash
# 1. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies (creates .venv/ and uv.lock)
uv sync

# 3. Activate environment
source .venv/bin/activate

# 4. Install spaCy model
python -m spacy download en_core_web_sm
```

#### Option B: Using `pip`

```bash
# 1. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Install spaCy model
python -m spacy download en_core_web_sm
```

---

## 🔧 Adding New Dependencies

### Step 1: Edit `pyproject.toml`

**For production dependency:**
```toml
[project]
dependencies = [
    "new-package>=1.0.0",
]
```

**For development dependency:**
```toml
[project.optional-dependencies]
dev = [
    "new-dev-package>=2.0.0",
]
```

### Step 2: Regenerate Requirements Files

```bash
# Regenerate production requirements
uv pip compile pyproject.toml -o requirements.txt --python-version 3.10

# Regenerate dev requirements
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt --python-version 3.10
```

### Step 3: Reinstall

**With uv:**
```bash
uv sync
```

**With pip:**
```bash
pip install -r requirements-dev.txt
```

---

## 🧪 Verifying Dependencies

```bash
# Check installed packages
pip list

# Check for security vulnerabilities (if installed)
pip-audit

# Verify project can be installed
pip install -e .

# Run tests
pytest
```

---

## 📊 Current Dependencies (Summary)

### Production (125 packages)
- **Web**: FastAPI, uvicorn, httpx
- **ML/NLP**: transformers, torch, spacy, gensim, scikit-learn
- **Data**: pandas, numpy, scipy
- **Visualization**: streamlit, plotly, pyldavis
- **Database**: sqlalchemy
- **Config**: pydantic, pydantic-settings, python-dotenv
- **Logging**: structlog
- **Scraping**: requests, beautifulsoup4, lxml
- **CLI**: typer, rich

### Development (+26 packages)
- **Testing**: pytest, pytest-cov, pytest-asyncio, hypothesis
- **Type Checking**: mypy, types-*
- **Linting**: ruff
- **Pre-commit**: pre-commit

---

## 🐳 Docker Multi-Stage Build Example

```dockerfile
# Stage 1: Base image with Python
FROM python:3.10-slim AS base
WORKDIR /app

# Stage 2: Install dependencies
FROM base AS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Copy application code
FROM dependencies AS app
COPY src/ ./src/
COPY .env.example .env

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["uvicorn", "src.threat_intelligence_aggregator.infrastructure.api.main:app", "--host", "0.0.0.0"]
```

---

## ⚠️ Important Notes

### Why Pin Versions in Docker?

`requirements.txt` has **exact versions** (e.g., `numpy==2.2.6`) to ensure:
- ✅ **Reproducible builds** - same versions every time
- ✅ **Avoid surprises** - no unexpected breaking changes
- ✅ **Security auditing** - know exactly what's installed

### Why Use `uv`?

- ⚡ **10-100x faster** than pip
- 🔒 **Better dependency resolution** - fewer conflicts
- 📦 **Modern tooling** - built in Rust, replaces pip-tools
- 🎯 **Single command** - `uv sync` does everything

### Migration from Older Setup

If you previously used `requirements.txt` manually:
```bash
# Old way (manual)
pip freeze > requirements.txt

# New way (from pyproject.toml)
uv pip compile pyproject.toml -o requirements.txt
```

---

## 🔗 References

- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [pip-tools (alternative)](https://github.com/jazzband/pip-tools)
- [Poetry (alternative)](https://python-poetry.org/)

---

**Last Updated**: 2026-01-17  
**Generated with**: `uv pip compile`
