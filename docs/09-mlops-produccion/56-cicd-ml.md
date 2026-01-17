# CI/CD para Machine Learning

## 1. Introduccion a CI/CD para ML

### Diferencias con CI/CD Tradicional

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CI/CD TRADICIONAL vs CI/CD para ML                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CI/CD TRADICIONAL (Software)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  [Codigo] ──► [Build] ──► [Unit Tests] ──► [Integration] ──► [Deploy] │
│  │                                                                     │   │
│  │  Artefactos: binarios, containers                                   │   │
│  │  Tests: deterministas, rapidos                                       │   │
│  │  Validacion: tests pasan/fallan                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  CI/CD para ML                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  [Codigo] ──► [Build]───────────────────────────────┐              │   │
│  │     +                                                │              │   │
│  │  [Datos] ──► [Data Validation] ──► [Train] ──► [Model Validation]  │   │
│  │     +                                   │              │             │   │
│  │  [Config] ──────────────────────────────┘              │             │   │
│  │                                                        ▼             │   │
│  │                                              [Register Model]        │   │
│  │                                                        │             │   │
│  │                                                        ▼             │   │
│  │                          [Deploy to Staging] ──► [A/B Test] ──► [Prod]│
│  │                                                                     │   │
│  │  Artefactos: codigo + datos + modelos + config                      │   │
│  │  Tests: code tests + data tests + model tests                        │   │
│  │  Validacion: metricas > threshold, sin regression                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline CI/CD para ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE CI/CD PARA ML                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRIGGER: Push to main / PR / Schedule / Data change                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: CODE QUALITY                                               │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │   │
│  │ │  Lint   │ │  Type   │ │ Format  │ │Security │                    │   │
│  │ │  ruff   │ │  mypy   │ │  ruff   │ │ bandit  │                    │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘                    │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │ Pass                                    │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: UNIT TESTS                                                 │   │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │   │
│  │ │ Domain Tests    │ │ Service Tests   │ │ Utils Tests     │        │   │
│  │ │ (pure logic)    │ │ (mocked deps)   │ │                 │        │   │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │ Pass                                    │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: DATA VALIDATION                                            │   │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │   │
│  │ │ Schema Check    │ │ Distribution    │ │ Data Quality    │        │   │
│  │ │ (great_expect.) │ │ Check           │ │ Metrics         │        │   │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │ Pass                                    │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: MODEL TRAINING                                             │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐│   │
│  │ │ Train model with tracked experiment (MLflow)                    ││   │
│  │ │ Log: params, metrics, artifacts, model                          ││   │
│  │ └─────────────────────────────────────────────────────────────────┘│   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: MODEL VALIDATION                                           │   │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │   │
│  │ │ Metrics Check   │ │ Regression Test │ │ Bias/Fairness   │        │   │
│  │ │ (F1 > 0.8)      │ │ (vs baseline)   │ │ Check           │        │   │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │ Pass                                    │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 6: REGISTER & DEPLOY                                          │   │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │   │
│  │ │ Register to     │ │ Deploy to       │ │ Smoke Tests     │        │   │
│  │ │ Model Registry  │ │ Staging         │ │ on Staging      │        │   │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘        │   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                   │ Manual Approval (optional)              │
│                                   ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 7: PRODUCTION                                                 │   │
│  │ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │   │
│  │ │ Promote to      │ │ Canary/Blue-    │ │ Monitor         │        │   │
│  │ │ Production      │ │ Green Deploy    │ │ Metrics         │        │   │
│  │ └─────────────────┘ └─────────────────┘ └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. GitHub Actions para ML

### Estructura del Repositorio

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ESTRUCTURA DE REPOSITORIO ML                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ml-project/                                                                │
│  ├── .github/                                                               │
│  │   └── workflows/                                                         │
│  │       ├── ci.yml              # CI: lint, test, validate                │
│  │       ├── train.yml           # Training pipeline                       │
│  │       ├── deploy.yml          # Deployment pipeline                     │
│  │       └── scheduled-train.yml # Scheduled retraining                    │
│  │                                                                          │
│  ├── src/                                                                   │
│  │   ├── data/                   # Data loading, validation                │
│  │   ├── features/               # Feature engineering                     │
│  │   ├── models/                 # Model definitions                       │
│  │   ├── training/               # Training scripts                        │
│  │   └── serving/                # Serving code                            │
│  │                                                                          │
│  ├── tests/                                                                 │
│  │   ├── unit/                   # Unit tests                              │
│  │   ├── integration/            # Integration tests                       │
│  │   ├── data/                   # Data validation tests                   │
│  │   └── model/                  # Model validation tests                  │
│  │                                                                          │
│  ├── configs/                                                               │
│  │   ├── training/               # Training configs                        │
│  │   └── serving/                # Serving configs                         │
│  │                                                                          │
│  ├── pyproject.toml                                                        │
│  ├── Dockerfile                                                             │
│  └── README.md                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workflow de CI Completo

```yaml
# .github/workflows/ci.yml
name: ML CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # ==========================================================================
  # STAGE 1: Code Quality
  # ==========================================================================
  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv pip install --system -e ".[dev]"

      - name: Run ruff linter
        run: ruff check src/ tests/

      - name: Run ruff formatter check
        run: ruff format --check src/ tests/

      - name: Run mypy
        run: mypy src/ --strict

      - name: Run bandit (security)
        run: bandit -r src/ -ll

  # ==========================================================================
  # STAGE 2: Unit Tests
  # ==========================================================================
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=term-missing \
            --junitxml=junit.xml \
            -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: unit-test-results
          path: junit.xml

  # ==========================================================================
  # STAGE 3: Data Validation
  # ==========================================================================
  data-validation:
    name: Data Validation
    runs-on: ubuntu-latest
    needs: unit-tests

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Download test data
        run: |
          # Descargar datos de prueba (S3, GCS, etc.)
          # aws s3 cp s3://bucket/test-data/ data/ --recursive
          echo "Using synthetic test data"

      - name: Run data validation tests
        run: |
          pytest tests/data/ \
            -v \
            --tb=short

      - name: Run Great Expectations
        run: |
          python -m src.data.validate_data \
            --checkpoint test_data_checkpoint

  # ==========================================================================
  # STAGE 4: Model Tests (sin entrenamiento completo)
  # ==========================================================================
  model-tests:
    name: Model Tests
    runs-on: ubuntu-latest
    needs: data-validation

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Run model unit tests
        run: |
          pytest tests/model/ \
            -v \
            -m "not slow" \
            --tb=short

      - name: Test model inference
        run: |
          python -m src.models.test_inference \
            --model-path models/baseline.pkl \
            --test-samples 100

  # ==========================================================================
  # STAGE 5: Integration Tests
  # ==========================================================================
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: model-tests

    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Run integration tests
        env:
          REDIS_URL: redis://localhost:6379
        run: |
          pytest tests/integration/ \
            -v \
            --tb=short

  # ==========================================================================
  # STAGE 6: Build Docker Image
  # ==========================================================================
  build-image:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/ml-service:${{ github.sha }}
            ghcr.io/${{ github.repository }}/ml-service:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Workflow de Training

```yaml
# .github/workflows/train.yml
name: ML Training Pipeline

on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name'
        required: true
        default: 'threat_detection'
      config_file:
        description: 'Training config file'
        required: true
        default: 'configs/training/default.yaml'
  schedule:
    # Reentrenar semanalmente
    - cron: '0 0 * * 0'

env:
  PYTHON_VERSION: "3.11"
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  train:
    name: Train Model
    runs-on: ubuntu-latest
    # Usar runner con GPU si es necesario
    # runs-on: [self-hosted, gpu]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e "."

      - name: Configure MLflow
        run: |
          echo "MLFLOW_TRACKING_URI=${{ env.MLFLOW_TRACKING_URI }}" >> $GITHUB_ENV

      - name: Download training data
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          aws s3 cp s3://ml-data-bucket/training/ data/training/ --recursive

      - name: Run training
        id: training
        run: |
          python -m src.training.train \
            --config ${{ github.event.inputs.config_file || 'configs/training/default.yaml' }} \
            --experiment-name ${{ github.event.inputs.experiment_name || 'threat_detection' }} \
            --output-dir ./outputs

          # Guardar run_id para siguientes steps
          echo "run_id=$(cat outputs/run_id.txt)" >> $GITHUB_OUTPUT

      - name: Upload training artifacts
        uses: actions/upload-artifact@v4
        with:
          name: training-outputs
          path: outputs/

  validate:
    name: Validate Model
    runs-on: ubuntu-latest
    needs: train

    steps:
      - uses: actions/checkout@v4

      - name: Download training artifacts
        uses: actions/download-artifact@v4
        with:
          name: training-outputs
          path: outputs/

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Validate model metrics
        id: validate
        run: |
          python -m src.validation.validate_model \
            --model-path outputs/model/ \
            --metrics-file outputs/metrics.json \
            --thresholds configs/validation/thresholds.yaml

      - name: Check for regression
        run: |
          python -m src.validation.check_regression \
            --new-metrics outputs/metrics.json \
            --baseline-model models:/threat_detector/Production

      - name: Validate model fairness
        run: |
          python -m src.validation.check_fairness \
            --model-path outputs/model/ \
            --test-data data/test/

  register:
    name: Register Model
    runs-on: ubuntu-latest
    needs: validate
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Download training artifacts
        uses: actions/download-artifact@v4
        with:
          name: training-outputs
          path: outputs/

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install uv
          uv pip install --system -e "."

      - name: Register model to MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ env.MLFLOW_TRACKING_URI }}
        run: |
          python -m src.registry.register_model \
            --model-path outputs/model/ \
            --model-name threat_detector \
            --stage Staging \
            --metrics-file outputs/metrics.json

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: model-v${{ github.run_number }}
          name: Model Release v${{ github.run_number }}
          body: |
            ## Model Metrics
            - Accuracy: ${{ steps.training.outputs.accuracy }}
            - F1 Score: ${{ steps.training.outputs.f1 }}
            - Recall: ${{ steps.training.outputs.recall }}
          files: |
            outputs/metrics.json
            outputs/model_card.md
```

## 3. Testing de Modelos

### Data Tests

```python
"""
Tests de validacion de datos con Great Expectations y pytest.
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest


class TestDataSchema:
    """Tests de schema de datos."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Datos de prueba."""
        return pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "ip_address": ["1.2.3.4", "5.6.7.8", "9.10.11.12"],
            "bytes_sent": [1000, 2000, 1500],
            "bytes_received": [500, 1000, 750],
            "is_malicious": [0, 1, 0],
            "timestamp": pd.date_range("2024-01-01", periods=3)
        })

    def test_required_columns_exist(self, sample_data: pd.DataFrame):
        """Verifica que existen todas las columnas requeridas."""
        required_columns = [
            "user_id", "ip_address", "bytes_sent",
            "bytes_received", "is_malicious", "timestamp"
        ]

        missing = set(required_columns) - set(sample_data.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"

    def test_no_null_in_required_columns(self, sample_data: pd.DataFrame):
        """Verifica que no hay nulls en columnas requeridas."""
        required_non_null = ["user_id", "ip_address", "is_malicious"]

        for col in required_non_null:
            null_count = sample_data[col].isnull().sum()
            assert null_count == 0, f"Column {col} has {null_count} nulls"

    def test_data_types(self, sample_data: pd.DataFrame):
        """Verifica tipos de datos."""
        expected_types = {
            "user_id": "object",
            "bytes_sent": "int64",
            "bytes_received": "int64",
            "is_malicious": "int64",
        }

        for col, expected_type in expected_types.items():
            actual_type = str(sample_data[col].dtype)
            assert actual_type == expected_type, \
                f"Column {col}: expected {expected_type}, got {actual_type}"


class TestDataDistribution:
    """Tests de distribucion de datos."""

    @pytest.fixture
    def training_data(self) -> pd.DataFrame:
        """Datos de entrenamiento."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame({
            "bytes_sent": np.random.exponential(1000, n),
            "bytes_received": np.random.exponential(500, n),
            "is_malicious": np.random.choice([0, 1], n, p=[0.95, 0.05])
        })

    def test_target_class_balance(self, training_data: pd.DataFrame):
        """Verifica balance de clases."""
        class_counts = training_data["is_malicious"].value_counts(normalize=True)

        # Para deteccion de amenazas, esperamos desbalance
        # pero no extremo (al menos 1% de positivos)
        positive_rate = class_counts.get(1, 0)

        assert positive_rate >= 0.01, \
            f"Too few positive samples: {positive_rate:.2%}"
        assert positive_rate <= 0.5, \
            f"Suspiciously high positive rate: {positive_rate:.2%}"

    def test_feature_ranges(self, training_data: pd.DataFrame):
        """Verifica rangos de features."""
        # bytes_sent debe ser >= 0
        assert (training_data["bytes_sent"] >= 0).all(), \
            "Negative bytes_sent values found"

        # bytes_sent no deberia tener outliers extremos
        q99 = training_data["bytes_sent"].quantile(0.99)
        assert q99 < 1e9, f"Extreme outliers in bytes_sent: q99={q99}"

    def test_no_data_leakage(self, training_data: pd.DataFrame):
        """Verifica que no hay data leakage obvio."""
        # Correlacion perfecta con target = leakage
        for col in training_data.columns:
            if col != "is_malicious" and training_data[col].dtype in ['int64', 'float64']:
                corr = training_data[col].corr(training_data["is_malicious"])
                assert abs(corr) < 0.99, \
                    f"Possible leakage: {col} has {corr:.2f} correlation with target"


class TestGreatExpectations:
    """Tests usando Great Expectations."""

    def test_data_expectations(self):
        """Ejecuta expectativas de Great Expectations."""
        # Crear contexto
        context = gx.get_context()

        # Definir expectativas
        validator = context.sources.pandas_default.read_dataframe(
            pd.DataFrame({
                "user_id": ["u1", "u2", "u3"],
                "bytes_sent": [1000, 2000, 1500],
                "is_malicious": [0, 1, 0]
            })
        )

        # Ejecutar expectativas
        validator.expect_column_values_to_not_be_null("user_id")
        validator.expect_column_values_to_be_between(
            "bytes_sent", min_value=0, max_value=1e9
        )
        validator.expect_column_values_to_be_in_set(
            "is_malicious", [0, 1]
        )

        # Validar
        results = validator.validate()
        assert results.success, f"Expectations failed: {results}"


# =============================================================================
# DATA VALIDATION SCRIPT
# =============================================================================

def validate_training_data(data_path: str) -> Dict[str, Any]:
    """
    Script de validacion de datos de entrenamiento.

    Usado en CI/CD para validar datos antes de entrenar.

    Args:
        data_path: Path a los datos

    Returns:
        Dict con resultados de validacion
    """
    import json

    df = pd.read_parquet(data_path)

    results = {
        "passed": True,
        "checks": [],
        "warnings": []
    }

    # Check 1: Schema
    required_cols = ["user_id", "features", "label"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        results["passed"] = False
        results["checks"].append({
            "name": "schema_check",
            "passed": False,
            "message": f"Missing columns: {missing}"
        })
    else:
        results["checks"].append({
            "name": "schema_check",
            "passed": True
        })

    # Check 2: No nulls en label
    null_labels = df["label"].isnull().sum()
    if null_labels > 0:
        results["passed"] = False
        results["checks"].append({
            "name": "null_labels",
            "passed": False,
            "message": f"{null_labels} null labels found"
        })
    else:
        results["checks"].append({
            "name": "null_labels",
            "passed": True
        })

    # Check 3: Sample size minimo
    min_samples = 1000
    if len(df) < min_samples:
        results["passed"] = False
        results["checks"].append({
            "name": "sample_size",
            "passed": False,
            "message": f"Only {len(df)} samples, need {min_samples}"
        })
    else:
        results["checks"].append({
            "name": "sample_size",
            "passed": True
        })

    # Check 4: Class balance (warning only)
    if "label" in df.columns:
        positive_rate = df["label"].mean()
        if positive_rate < 0.01:
            results["warnings"].append({
                "name": "class_imbalance",
                "message": f"Very low positive rate: {positive_rate:.2%}"
            })

    return results
```

### Model Tests

```python
"""
Tests de validacion de modelos.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


class TestModelBehavior:
    """Tests de comportamiento del modelo."""

    @pytest.fixture
    def trained_model(self) -> BaseEstimator:
        """Carga modelo entrenado."""
        return joblib.load("models/threat_detector.pkl")

    @pytest.fixture
    def test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Datos de test."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_model_output_shape(
        self,
        trained_model: BaseEstimator,
        test_data: Tuple
    ):
        """Verifica shape del output."""
        X, _ = test_data

        predictions = trained_model.predict(X)
        assert predictions.shape == (len(X),), \
            f"Expected shape ({len(X)},), got {predictions.shape}"

    def test_model_output_values(
        self,
        trained_model: BaseEstimator,
        test_data: Tuple
    ):
        """Verifica que outputs son validos."""
        X, _ = test_data

        predictions = trained_model.predict(X)

        # Para clasificacion binaria
        assert set(np.unique(predictions)).issubset({0, 1}), \
            f"Invalid predictions: {np.unique(predictions)}"

    def test_model_probabilities(
        self,
        trained_model: BaseEstimator,
        test_data: Tuple
    ):
        """Verifica probabilidades."""
        X, _ = test_data

        if hasattr(trained_model, 'predict_proba'):
            probas = trained_model.predict_proba(X)

            # Shape correcto
            assert probas.shape == (len(X), 2), \
                f"Expected shape ({len(X)}, 2), got {probas.shape}"

            # Probabilidades suman 1
            sums = probas.sum(axis=1)
            assert np.allclose(sums, 1.0), \
                "Probabilities don't sum to 1"

            # Entre 0 y 1
            assert (probas >= 0).all() and (probas <= 1).all(), \
                "Probabilities outside [0, 1]"

    def test_model_deterministic(
        self,
        trained_model: BaseEstimator,
        test_data: Tuple
    ):
        """Verifica que modelo es determinista."""
        X, _ = test_data

        pred1 = trained_model.predict(X)
        pred2 = trained_model.predict(X)

        assert np.array_equal(pred1, pred2), \
            "Model predictions are not deterministic"


class TestModelMetrics:
    """Tests de metricas del modelo."""

    @pytest.fixture
    def model_and_data(self) -> Tuple[BaseEstimator, np.ndarray, np.ndarray]:
        """Modelo y datos de test."""
        model = joblib.load("models/threat_detector.pkl")
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")
        return model, X_test, y_test

    def test_minimum_accuracy(self, model_and_data: Tuple):
        """Verifica accuracy minima."""
        model, X, y = model_and_data
        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        min_accuracy = 0.85

        assert accuracy >= min_accuracy, \
            f"Accuracy {accuracy:.2%} below minimum {min_accuracy:.2%}"

    def test_minimum_recall(self, model_and_data: Tuple):
        """
        Verifica recall minimo.

        CRITICO para ciberseguridad: preferimos falsos positivos
        a falsos negativos.
        """
        model, X, y = model_and_data
        y_pred = model.predict(X)

        recall = recall_score(y, y_pred)
        min_recall = 0.80  # Muy importante para seguridad

        assert recall >= min_recall, \
            f"Recall {recall:.2%} below minimum {min_recall:.2%}"

    def test_minimum_precision(self, model_and_data: Tuple):
        """Verifica precision minima."""
        model, X, y = model_and_data
        y_pred = model.predict(X)

        precision = precision_score(y, y_pred)
        min_precision = 0.70

        assert precision >= min_precision, \
            f"Precision {precision:.2%} below minimum {min_precision:.2%}"

    def test_no_regression_vs_baseline(self, model_and_data: Tuple):
        """Verifica que no hay regresion vs baseline."""
        model, X, y = model_and_data
        y_pred = model.predict(X)

        # Cargar metricas del baseline
        baseline_metrics = {
            "accuracy": 0.90,
            "f1": 0.85,
            "recall": 0.82
        }

        current_metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "recall": recall_score(y, y_pred)
        }

        # Permitir 2% de degradacion
        tolerance = 0.02

        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics[metric]
            min_allowed = baseline_value - tolerance

            assert current_value >= min_allowed, \
                f"Regression in {metric}: {current_value:.2%} < {min_allowed:.2%}"


class TestModelRobustness:
    """Tests de robustez del modelo."""

    @pytest.fixture
    def model(self) -> BaseEstimator:
        return joblib.load("models/threat_detector.pkl")

    def test_handles_edge_values(self, model: BaseEstimator):
        """Verifica que maneja valores extremos."""
        # Valores muy grandes
        X_large = np.array([[1e10] * 10])
        pred_large = model.predict(X_large)
        assert pred_large.shape == (1,)

        # Valores muy pequenos
        X_small = np.array([[1e-10] * 10])
        pred_small = model.predict(X_small)
        assert pred_small.shape == (1,)

        # Ceros
        X_zeros = np.zeros((1, 10))
        pred_zeros = model.predict(X_zeros)
        assert pred_zeros.shape == (1,)

    def test_handles_batch_sizes(self, model: BaseEstimator):
        """Verifica que maneja diferentes tamanos de batch."""
        for batch_size in [1, 10, 100, 1000]:
            X = np.random.randn(batch_size, 10)
            pred = model.predict(X)
            assert pred.shape == (batch_size,), \
                f"Failed for batch_size={batch_size}"

    def test_latency_acceptable(self, model: BaseEstimator):
        """Verifica latencia de inferencia."""
        import time

        X = np.random.randn(100, 10)

        # Warm up
        model.predict(X)

        # Medir
        times = []
        for _ in range(10):
            start = time.time()
            model.predict(X)
            times.append(time.time() - start)

        avg_time_ms = np.mean(times) * 1000
        max_latency_ms = 50  # 50ms para batch de 100

        assert avg_time_ms < max_latency_ms, \
            f"Latency {avg_time_ms:.2f}ms exceeds {max_latency_ms}ms"


# =============================================================================
# MODEL VALIDATION SCRIPT
# =============================================================================

def validate_model(
    model_path: str,
    test_data_path: str,
    thresholds_path: str
) -> Dict[str, Any]:
    """
    Script de validacion de modelo para CI/CD.

    Args:
        model_path: Path al modelo
        test_data_path: Path a datos de test
        thresholds_path: Path a config de thresholds

    Returns:
        Dict con resultados de validacion
    """
    import yaml

    # Cargar modelo y datos
    model = joblib.load(model_path)
    X_test = np.load(f"{test_data_path}/X_test.npy")
    y_test = np.load(f"{test_data_path}/y_test.npy")

    # Cargar thresholds
    with open(thresholds_path) as f:
        thresholds = yaml.safe_load(f)

    # Predecir
    y_pred = model.predict(X_test)

    # Calcular metricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # Validar contra thresholds
    results = {
        "passed": True,
        "metrics": metrics,
        "validations": []
    }

    for metric_name, threshold in thresholds.items():
        if metric_name in metrics:
            passed = metrics[metric_name] >= threshold
            results["validations"].append({
                "metric": metric_name,
                "value": metrics[metric_name],
                "threshold": threshold,
                "passed": passed
            })
            if not passed:
                results["passed"] = False

    return results
```

## 4. Deployment Automatizado

### Workflow de Deploy

```yaml
# .github/workflows/deploy.yml
name: Deploy Model

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    if: ${{ github.event.inputs.environment == 'staging' }}
    environment: staging

    steps:
      - uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Kubernetes
        run: |
          # Update image tag
          kubectl set image deployment/ml-service \
            ml-service=ghcr.io/${{ github.repository }}/ml-service:${{ github.event.inputs.model_version }} \
            -n staging

          # Wait for rollout
          kubectl rollout status deployment/ml-service -n staging --timeout=5m

      - name: Run smoke tests
        run: |
          STAGING_URL=${{ secrets.STAGING_URL }}

          # Health check
          curl -f ${STAGING_URL}/health || exit 1

          # Sample prediction
          curl -X POST ${STAGING_URL}/predict \
            -H "Content-Type: application/json" \
            -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}' \
            || exit 1

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    if: ${{ github.event.inputs.environment == 'production' }}
    environment: production
    # Requiere aprobacion manual

    steps:
      - uses: actions/checkout@v4

      - name: Validate model in registry
        run: |
          python -m src.registry.validate_model \
            --model-name threat_detector \
            --version ${{ github.event.inputs.model_version }} \
            --required-stage Staging

      - name: Deploy canary (10%)
        run: |
          kubectl apply -f k8s/canary.yaml

          # Wait and monitor
          sleep 300  # 5 min de canary

          # Check error rate
          ERROR_RATE=$(curl -s prometheus:9090/api/v1/query?query=... | jq '.data.result[0].value[1]')
          if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
            echo "Error rate too high, rolling back"
            kubectl rollout undo deployment/ml-service-canary -n production
            exit 1
          fi

      - name: Promote to production
        run: |
          kubectl set image deployment/ml-service \
            ml-service=ghcr.io/${{ github.repository }}/ml-service:${{ github.event.inputs.model_version }} \
            -n production

          kubectl rollout status deployment/ml-service -n production --timeout=10m

      - name: Update model registry
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python -m src.registry.promote_model \
            --model-name threat_detector \
            --version ${{ github.event.inputs.model_version }} \
            --stage Production

      - name: Notify team
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Model deployed to production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "Model *threat_detector v${{ github.event.inputs.model_version }}* deployed to production"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## 5. Scheduled Retraining

```yaml
# .github/workflows/scheduled-train.yml
name: Scheduled Retraining

on:
  schedule:
    # Cada domingo a las 00:00 UTC
    - cron: '0 0 * * 0'
  workflow_dispatch:

jobs:
  check-data-drift:
    name: Check Data Drift
    runs-on: ubuntu-latest
    outputs:
      should_retrain: ${{ steps.drift.outputs.should_retrain }}

    steps:
      - uses: actions/checkout@v4

      - name: Check drift metrics
        id: drift
        run: |
          # Consultar metricas de drift de monitoreo
          DRIFT_SCORE=$(curl -s ${{ secrets.MONITORING_URL }}/drift-score)

          if (( $(echo "$DRIFT_SCORE > 0.1" | bc -l) )); then
            echo "should_retrain=true" >> $GITHUB_OUTPUT
          else
            echo "should_retrain=false" >> $GITHUB_OUTPUT
          fi

  retrain:
    name: Retrain Model
    needs: check-data-drift
    if: needs.check-data-drift.outputs.should_retrain == 'true'
    uses: ./.github/workflows/train.yml
    with:
      experiment_name: scheduled_retrain
      config_file: configs/training/default.yaml
    secrets: inherit
```

## 6. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESUMEN: CI/CD para ML                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DIFERENCIAS CON CI/CD TRADICIONAL                                          │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Artefactos: codigo + datos + modelo + config                             │
│  • Tests: code + data + model tests                                         │
│  • Validacion: metricas > threshold, no solo pass/fail                      │
│  • Deploy: incluye model registry y versionado                              │
│                                                                             │
│  STAGES DEL PIPELINE                                                        │
│  ───────────────────────────────────────────────────────────────────────   │
│  1. Code Quality: lint, format, types, security                             │
│  2. Unit Tests: logica pura, mocked dependencies                            │
│  3. Data Validation: schema, distribution, quality                          │
│  4. Training: con experiment tracking                                        │
│  5. Model Validation: metricas, regression, fairness                        │
│  6. Register & Deploy: model registry, staging, prod                        │
│                                                                             │
│  TIPOS DE TESTS                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Data Tests: schema, nulls, distribution, leakage                         │
│  • Model Tests: output shape, values, determinism, latency                  │
│  • Metric Tests: accuracy, recall, precision vs thresholds                  │
│  • Regression Tests: comparar vs baseline                                   │
│                                                                             │
│  HERRAMIENTAS                                                               │
│  ───────────────────────────────────────────────────────────────────────   │
│  • GitHub Actions / GitLab CI / Jenkins                                     │
│  • Great Expectations para data validation                                  │
│  • pytest para todos los tests                                              │
│  • MLflow para tracking y registry                                          │
│                                                                             │
│  TRIGGERS                                                                   │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Push/PR: CI completo                                                     │
│  • Schedule: retraining periodico                                           │
│  • Data change: trigger por nuevos datos                                    │
│  • Drift detection: trigger automatico si hay drift                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Monitoreo y Drift Detection - data drift, model drift, concept drift
