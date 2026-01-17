# Agent Guidelines for ML Engineer Codebase

## Project Overview

Full-stack **SPAM & PHISHING Detection System** with:
- **Backend**: Python 3.10+ (FastAPI + scikit-learn ML models)
- **Frontend**: React 19 + TypeScript 5.9+ (Vite)
- **Architecture**: Domain-Driven Design with clean separation of concerns

---

## Build, Lint & Test Commands

### Backend (Python)
```bash
# Location: src/backend/
cd src/backend

# Install dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/application/use_cases/test_classify_email.py

# Run single test function
pytest tests/unit/application/use_cases/test_classify_email.py::TestClassifyEmailUseCase::test_execute_calls_classifier

# Run tests by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Linting & formatting
ruff check .                    # Lint
ruff check --fix .              # Auto-fix lint issues
ruff format .                   # Format code

# Type checking
mypy spam_detector/

# Start API server (development)
uvicorn spam_detector.infrastructure.api:app --reload
# Or use entry point:
spam-detector-api
```

### Frontend (TypeScript/React)
```bash
# Location: src/frontend/
cd src/frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Linting
npm run lint

# Preview production build
npm run preview
```

### Docker (Full Stack)
```bash
# From project root
docker-compose up --build
# API: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## Code Style Guidelines

### Python Style (PEP 8 + Modern Python)

#### Naming Conventions
- **Modules/files**: `snake_case.py`
- **Packages**: `snake_case`
- **Functions/methods**: `snake_case`
- **Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- Use descriptive names: `load_customer_data`, not `ld` or `do_stuff`

#### Imports
```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local (absolute from project root)
from spam_detector.domain.entities import Email
from spam_detector.application.use_cases import ClassifyEmailUseCase
```
- Use absolute imports from project root
- Group imports: stdlib → third-party → local
- NO wildcard imports (`from x import *`)

#### Formatting
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Target ≤ 88 chars (Black style), hard limit 100
- **Tool**: `ruff format` (replaces Black/isort)
- Spaces around operators: `total = price * quantity + tax`

#### Type Hints
```python
from __future__ import annotations

def load_data(path: str) -> list[dict[str, Any]]:
    """Load data from file."""
    ...

def process(items: list[str] | None = None) -> None:
    """Process items."""
    if items is None:
        items = []
```
- Use modern syntax: `list[str]`, `dict[str, Any]` (not `List`, `Dict`)
- Use `|` for unions: `str | None` (not `Optional[str]`)
- Always add `-> None` for void functions
- Type hints required for function parameters and return types

#### Functions & Methods
- Keep functions **under 20-30 lines**
- One responsibility per function
- Avoid mutable defaults:
  ```python
  # ✅ Good
  def process(items: list[str] | None = None) -> None:
      if items is None:
          items = []
  
  # ❌ Bad
  def process(items: list[str] = []) -> None:
      ...
  ```

#### Error Handling
```python
# ✅ Catch specific exceptions
try:
    value = int(raw)
except ValueError:
    logger.error("❌ Invalid integer: %s", raw)
    return None

# ❌ Avoid bare except
try:
    value = int(raw)
except:  # DON'T DO THIS
    pass
```
- Never swallow exceptions silently
- Log errors with context
- Use exceptions for exceptional cases, not control flow

#### Logging
```python
import logging

logger = logging.getLogger(__name__)

logger.info("ℹ️ Processing started")
logger.warning("⚠️ Deprecated feature used")
logger.error("❌ Failed to load model: %s", error)
```
- Use emoji prefixes: `ℹ️` (info), `⚠️` (warning), `❌` (error)
- NO `print()` in application code (tests/scripts OK)
- Never log secrets, API keys, or PII

#### Docstrings
```python
def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize a list of scores to [0.0, 1.0].

    Args:
        scores: List of raw scores

    Returns:
        Normalized scores in [0.0, 1.0] range

    Raises:
        ValueError: If scores is empty or contains negatives
    """
    ...
```
- Required for public functions, classes, modules
- Short summary + optional Args/Returns/Raises sections

---

### TypeScript/React Style

#### Naming Conventions
- **Files/components**: `PascalCase.tsx` (components), `camelCase.ts` (utilities)
- **Functions/variables**: `camelCase`
- **Types/interfaces**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`

#### Imports
```typescript
// React & third-party
import { useState } from 'react';
import { QueryClient } from '@tanstack/react-query';

// Local types
import type { ClassifyResponse } from './types';

// Local components/hooks
import { Header } from './components/Header';
import { useClassifyEmail } from './hooks/useClassifyEmail';
```

#### Type Safety
- Always use TypeScript, avoid `any`
- Define interfaces for API responses
- Use `type` for unions/aliases, `interface` for objects

#### Components
- Functional components with hooks
- Extract logic into custom hooks
- Keep components under 200 lines

---

## Architecture & Project Structure

### Backend (Domain-Driven Design)
```
src/backend/spam_detector/
├── domain/           # Business logic (entities, ports, services)
├── application/      # Use cases, orchestration
├── infrastructure/   # Adapters (API, ML, storage, CLI)
├── config/          # Configuration management
└── utils/           # Shared utilities
```

**Key Principles**:
- **Single Responsibility**: One clear purpose per module/class/function
- **Files under ~300 lines**: Split if exceeding ~400 lines
- **Layered architecture**: Domain → Application → Infrastructure
- **No circular dependencies**: Use interfaces/ports to decouple

### Domain Layer
- Pure business logic, no framework dependencies
- Entities: `Email`, `ClassificationResult`, `SinglePrediction`
- Ports: Interfaces (`IOutputFormatter`, `IMLModel`)
- Services: Domain logic (`EmailClassifierService`)

### Application Layer
- Use cases orchestrate domain logic
- Example: `ClassifyEmailUseCase`
- Thin layer, delegates to domain services

### Infrastructure Layer
- Adapters implement domain ports
- External concerns: FastAPI routes, ML models, CLI
- MLflow integration, model loading

---

## Testing Standards

### Test Structure
```python
# tests/unit/application/use_cases/test_classify_email.py

@pytest.fixture
def mock_service():
    """Mock dependencies."""
    return Mock()

class TestClassifyEmailUseCase:
    """Test ClassifyEmailUseCase."""

    def test_execute_calls_classifier(self, use_case, mock_service):
        """Should call classifier service with Email entity."""
        use_case.execute("Test email")
        
        mock_service.classify.assert_called_once()
```

**Naming**: `test_<behavior>_<context>` or `test_should_<behavior>`

**Markers**:
- `@pytest.mark.unit` - Fast, no I/O
- `@pytest.mark.integration` - Files/network
- `@pytest.mark.slow` - Skip with `-m "not slow"`

**Coverage**: Target ≥80% (configured in pytest.ini)

---

## File Size Limits

- **Source files**: ~300 lines target, split if >400 lines
- **Functions/methods**: 20-30 lines ideal, extract helpers if >30
- **Test files**: Prefer multiple small files over one giant file
- **Classes**: Avoid "God classes" - apply Single Responsibility Principle

---

## Security & Best Practices

### MUST
- Externalize configuration (env vars, config files)
- Never hardcode secrets, API keys, credentials
- Use structured logging (no `print()` in app code)
- Write tests for new features and bug fixes
- Apply type hints consistently
- Keep functions/classes small and focused
- Prefer composition over inheritance

### MUST NOT
- Commit secrets or `.env` files with real credentials
- Use wildcard imports (`from x import *`)
- Create God files/classes mixing unrelated concerns
- Swallow exceptions silently without logging
- Log sensitive data (PII, passwords, tokens)
- Use mutable default arguments

---

## MLOps & ML Practices

- **Reproducibility**: Record seeds, data versions, model configs
- **Artifact tracking**: MLflow for models, metrics, experiments
- **Model versioning**: Store models with timestamps/versions
- **Separation of concerns**: Data preprocessing → Training → Inference
- **Evaluation**: Clear metrics, test splits, validation

---

## Cursor Rules Reference

This project includes 22 Cursor rules in `.cursor/rules/`. Key rules:
- `07_clean_code_principles.mdc` - Structure, size, responsibilities
- `08_cod_style.mdc` - Python style (PEP 8 + modern)
- `09_general_rules.mdc` - ML/MLOps engineering practices
- `01_security_baseline_rule.mdc` - Security/privacy
- `19_python_logging_rule.mdc` - Logging structure

Refer to these for comprehensive guidance.

---

## Quick Reference

| Task | Command |
|------|---------|
| Run all Python tests | `cd src/backend && pytest` |
| Run specific test | `pytest path/to/test.py::TestClass::test_method` |
| Format Python | `cd src/backend && ruff format .` |
| Lint Python | `cd src/backend && ruff check --fix .` |
| Type check | `cd src/backend && mypy spam_detector/` |
| Run backend API | `cd src/backend && uvicorn spam_detector.infrastructure.api:app --reload` |
| Run frontend dev | `cd src/frontend && npm run dev` |
| Build frontend | `cd src/frontend && npm run build` |
| Lint frontend | `cd src/frontend && npm run lint` |
| Full stack (Docker) | `docker-compose up --build` |

---

**Version**: 1.0  
**Last Updated**: 2026-01-08  
**Maintained by**: ML Engineer Team
