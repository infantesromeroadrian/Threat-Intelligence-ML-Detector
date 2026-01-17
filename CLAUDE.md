# Code Review Rules - Gentleman-AI Standards

> Pre-commit validation rules for AI-powered code review
> Based on 25+ years of Principal AI/ML Architect & Security Engineer experience

## 🚨 CRITICAL SECURITY RULES (Auto-Reject)

### Secrets Management
**Never commit secrets to git. This is non-negotiable.**

❌ **REJECT:**
- API keys, tokens, passwords hardcoded in code
- .env files committed to repository
- Database credentials in source files
- OAuth secrets in configuration

✅ **REQUIRE:**
- Secrets loaded from environment variables
- .env listed in .gitignore
- Secret manager usage (Vault, AWS SM, doppler)

**Example:**
```python
# ❌ REJECT - Hardcoded secret
api_key = "sk-1234567890abcdef"

# ✅ ACCEPT - Environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
```

---

## 🐍 Python Standards

### Dependency Management

❌ **REJECT:**
- `pip install` without virtual environment
- requirements.txt without lock file
- Mixing conda and pip
- poetry.lock or Pipfile.lock (use uv instead)

✅ **REQUIRE:**
- pyproject.toml with project metadata
- uv.lock for reproducible builds
- Virtual environment activated (check $VIRTUAL_ENV)

### Type Safety

❌ **REJECT:**
- `any` type annotations
- Missing type hints on public functions
- Ignoring mypy errors
- Duck typing in domain layer

✅ **REQUIRE:**
- Full type hints on all public APIs
- mypy --strict compliance
- Generic types properly bounded

**Example:**
```python
# ❌ REJECT
def process_data(data):
    return data + 1

def get_user(user_id: any) -> any:
    pass

# ✅ ACCEPT
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(value=lambda x: x.value + 1)

def get_user(user_id: int) -> User | None:
    pass
```

### Code Quality

❌ **REJECT:**
- `print()` statements for debugging (use logging)
- Bare `except:` clauses without exception type
- Code that doesn't pass `ruff format` and `ruff check`
- Magic numbers without constants
- Functions longer than 50 lines

✅ **REQUIRE:**
- Structured logging with `structlog` or `logging`
- Specific exception types
- ruff-compliant code
- Named constants for magic values
- Small, focused functions

**Example:**
```python
# ❌ REJECT
try:
    result = risky_operation()
except:
    print("Error occurred")

# ✅ ACCEPT
import structlog

logger = structlog.get_logger()

try:
    result = risky_operation()
except ValueError as e:
    logger.error("invalid_value", error=str(e))
    raise
except ConnectionError as e:
    logger.warning("connection_failed", error=str(e), retry=True)
    raise
```

### Testing

❌ **REJECT:**
- New features without tests
- Tests not co-located with source
- Tests without descriptive names
- Mock-heavy tests that don't test real behavior

✅ **REQUIRE:**
- pytest tests for all new features
- Tests in same directory as source (`src/module.py` → `src/module_test.py`)
- Descriptive test names that explain behavior
- Property-based tests with `hypothesis` for complex logic

---

## 🤖 ML/AI Standards

### Model Development

❌ **REJECT:**
- No train/validation/test split
- Only accuracy metric for imbalanced datasets
- No seed fixing (irreproducible experiments)
- Training on full dataset without holdout
- No baseline comparison

✅ **REQUIRE:**
- Proper train/val/test splits (no data leakage)
- Appropriate metrics (F1, precision, recall for imbalanced data)
- Fixed random seeds for reproducibility
- Baseline model for comparison
- Feature importance analysis

**Example violations:**
```python
# ❌ REJECT - No split
X, y = load_data()
model.fit(X, y)
print(f"Accuracy: {model.score(X, y)}")  # Training accuracy!

# ✅ ACCEPT - Proper evaluation
from sklearn.model_selection import train_test_split

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split train into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
val_score = model.score(X_val, y_val)
test_score = model.score(X_test, y_test)  # Never seen during training
```

### Production ML

❌ **REJECT:**
- Jupyter notebooks in production code
- Models without versioning
- No drift detection
- Hardcoded model paths
- Missing model metadata

✅ **REQUIRE:**
- Scripts/modules instead of notebooks
- MLflow or W&B experiment tracking
- Drift detection configured
- Model registry usage
- Model cards with metadata

### ML Pipelines

❌ **REJECT:**
- Airflow for new projects (legacy, complex)
- Untestable pipeline steps
- No data versioning
- Pipeline without monitoring

✅ **REQUIRE:**
- Dagster for orchestration
- Each step testable in isolation
- DVC or similar for data versioning
- Metrics logged at each pipeline stage

---

## 🏗️ Architecture Standards

### Hexagonal Architecture (for production projects)

**Domain layer MUST be pure - no infrastructure dependencies.**

❌ **REJECT in domain/:**
- pandas, numpy imports
- torch, tensorflow imports
- sklearn imports
- Database models (SQLAlchemy, etc.)
- API clients
- File I/O operations

✅ **REQUIRE:**
- Pure Python entities
- Business logic only
- Type hints with standard library types
- No side effects

**Example:**
```python
# ❌ REJECT - domain/user.py
import pandas as pd
from sqlalchemy import Column, Integer, String

class User:
    def load_from_db(self, user_id: int) -> pd.DataFrame:
        # DB access in domain!
        pass

# ✅ ACCEPT - domain/user.py
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class User:
    id: int
    email: str
    created_at: datetime

    def is_active(self) -> bool:
        """Pure business logic"""
        return (datetime.now() - self.created_at).days < 365
```

### Microservices

❌ **REJECT:**
- Multiple responsibilities in one service
- Untyped API contracts
- No health check endpoint
- Missing circuit breakers

✅ **REQUIRE:**
- Single responsibility per service
- Pydantic models for API contracts
- `/health` and `/ready` endpoints
- Timeout and retry logic

---

## 🐳 Infrastructure Standards

### Docker

❌ **REJECT:**
- Single-stage Dockerfile
- Running as root user
- Outdated base images
- No health check in Dockerfile
- Copying entire project (use .dockerignore)

✅ **REQUIRE:**
- Multi-stage builds (builder → runtime)
- Official base images (python:3.11-slim, not python:latest)
- Non-root user in runtime
- HEALTHCHECK instruction
- Minimal final image

**Example:**
```dockerfile
# ❌ REJECT
FROM python:latest
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]

# ✅ ACCEPT
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache

# Stage 2: Runtime
FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
USER appuser
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "-m", "src.main"]
```

### CI/CD

❌ **REJECT:**
- Manual deployment steps
- No automated tests
- Deploying without rollback plan
- Secrets in CI config

✅ **REQUIRE:**
- Automated test suite in pipeline
- Documented rollback procedure
- Secrets from CI secret store
- Deployment approval gates for production

---

## 📋 RAG Pipeline Standards

❌ **REJECT:**
- Arbitrary chunk size without justification
- No retrieval metrics
- Embeddings without evaluation
- No ground truth dataset
- Missing hallucination detection

✅ **REQUIRE:**
- Chunking strategy documented (size, overlap rationale)
- Precision@k, Recall@k measured
- Embedding model benchmarked against alternatives
- RAGAS or similar evaluation framework
- Guardrails on input/output
- Drift monitoring for embeddings

---

## 🎯 Review Response Format

**CRITICAL:** Your first line MUST be `STATUS: PASSED` or `STATUS: FAILED`.

### Format for FAILED Review:

```
STATUS: FAILED

Violations:

1. **src/api/routes.py:23** - Security Critical
   Issue: API key hardcoded in source
   Fix: Move to environment variable with os.getenv()

2. **src/domain/user.py:15** - Architecture
   Issue: pandas imported in domain layer
   Fix: Move DataFrame operations to infrastructure/repositories/

3. **src/ml/train.py:67** - ML Standards
   Issue: Training on full dataset without holdout
   Fix: Add train_test_split with test_size=0.2

4. **src/main.py:145** - Code Quality
   Issue: Bare except clause
   Fix: Catch specific exceptions (ValueError, ConnectionError)
```

### Format for PASSED Review:

```
STATUS: PASSED

All files comply with Gentleman-AI standards.

Summary:
- Security: ✓ No hardcoded secrets
- Architecture: ✓ Hexagonal structure maintained  
- Code Quality: ✓ ruff checks passing
- Testing: ✓ New features have tests
```

---

## 🔍 What to Check

For each file, verify:

1. **Security First**
   - No secrets in code
   - .env in .gitignore if used
   - Proper authentication/authorization

2. **Architecture**
   - Correct layer separation
   - No business logic in infrastructure
   - Single responsibility

3. **Code Quality**
   - Type hints present
   - No debugging artifacts (print statements)
   - Specific exception handling
   - ruff compliant

4. **Testing**
   - New features have tests
   - Tests are meaningful
   - No mock-heavy tests

5. **ML Specific (if applicable)**
   - Proper train/test split
   - Appropriate metrics
   - Reproducibility (seeds)
   - Experiment tracking

---

## 📊 Priority Levels

| Priority | Type | Action |
|----------|------|--------|
| 🚨 **Critical** | Security violations | Auto-reject, cannot commit |
| ❌ **Error** | Architecture/Quality violations | Must fix before commit |
| ⚠️ **Warning** | Best practices | Should fix, but can commit |

**Critical issues always result in `STATUS: FAILED`.**

---

## Response Guidelines

- Be specific: Include file, line number, issue, and fix
- Be actionable: Tell exactly what to change
- Be concise: No lengthy explanations
- Be consistent: Use the exact format above
- Focus on violations: Don't list things that are correct unless STATUS: PASSED

**Remember:** You are enforcing standards that prevent production incidents, security breaches, and technical debt. Be strict but fair.
