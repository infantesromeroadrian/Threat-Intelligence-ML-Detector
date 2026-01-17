# Introduccion a MLOps

## 1. Que es MLOps

### Definicion

**MLOps (Machine Learning Operations)** es un conjunto de practicas que combina Machine Learning, DevOps e Ingenieria de Datos para desplegar y mantener sistemas de ML en produccion de forma confiable y eficiente.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps = ML + DevOps + Data                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐              │
│     │   Machine   │      │   DevOps    │      │    Data     │              │
│     │  Learning   │  +   │ Practices   │  +   │ Engineering │              │
│     └─────────────┘      └─────────────┘      └─────────────┘              │
│           │                    │                    │                       │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│                                ▼                                            │
│           ┌─────────────────────────────────────────┐                       │
│           │              MLOps                      │                       │
│           │  • Automatizacion de pipelines         │                       │
│           │  • Versionado de datos y modelos       │                       │
│           │  • Monitoreo continuo                  │                       │
│           │  • CI/CD para ML                       │                       │
│           │  • Gobernanza y reproducibilidad       │                       │
│           └─────────────────────────────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Por que Necesitamos MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROBLEMA: "It works on my laptop"                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  87% de los proyectos de ML nunca llegan a produccion                      │
│                                                                             │
│  Razones principales:                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. DEUDA TECNICA OCULTA                                            │   │
│  │     Solo ~5% del codigo real de ML es el modelo                     │   │
│  │                                                                     │   │
│  │     ┌─────────────────────────────────────────────────────────┐    │   │
│  │     │  Configuracion  │  Gestion de Datos  │  Feature Eng.   │    │   │
│  │     │─────────────────┼────────────────────┼─────────────────│    │   │
│  │     │  Monitoreo      │  ██ ML CODE ██     │  Serving        │    │   │
│  │     │─────────────────┼────────────────────┼─────────────────│    │   │
│  │     │  Testing        │  Validacion        │  Infraestructura│    │   │
│  │     └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. REPRODUCIBILIDAD                                                │   │
│  │     - Versiones de librerias diferentes                             │   │
│  │     - Datos de entrenamiento no versionados                         │   │
│  │     - Hiperparametros no documentados                               │   │
│  │     - Seeds aleatorias no fijadas                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. DRIFT Y DEGRADACION                                             │   │
│  │     - Distribucion de datos cambia en produccion                    │   │
│  │     - Modelo entrena con datos de 2023, sirve en 2025               │   │
│  │     - Sin monitoreo, nadie detecta la degradacion                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Ciclo de Vida de ML

### Ciclo Completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CICLO DE VIDA ML                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐                                                         │
│    │  1. PROBLEMA │ ◄───────────────────────────────────────────────────┐  │
│    │   DEFINICION │                                                     │  │
│    └──────┬───────┘                                                     │  │
│           │                                                             │  │
│           ▼                                                             │  │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │  │
│    │  2. DATA     │─────►│  3. FEATURE  │─────►│  4. MODEL    │        │  │
│    │  COLLECTION  │      │  ENGINEERING │      │  TRAINING    │        │  │
│    └──────────────┘      └──────────────┘      └──────┬───────┘        │  │
│                                                       │                │  │
│                                                       ▼                │  │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │  │
│    │  7. MONITOR  │◄─────│  6. DEPLOY   │◄─────│  5. EVALUATE │        │  │
│    │  & FEEDBACK  │      │              │      │  & VALIDATE  │        │  │
│    └──────┬───────┘      └──────────────┘      └──────────────┘        │  │
│           │                                                             │  │
│           └─────────────────────────────────────────────────────────────┘  │
│                              Ciclo Continuo                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detalle de Cada Fase

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import hashlib
from datetime import datetime


class MLLifecyclePhase(Enum):
    """Fases del ciclo de vida ML."""
    PROBLEM_DEFINITION = "problem_definition"
    DATA_COLLECTION = "data_collection"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class MLProject:
    """
    Representacion de un proyecto ML con tracking de fases.

    Attributes:
        name: Nombre del proyecto
        problem_type: clasificacion, regresion, clustering, etc.
        success_metrics: Metricas que definen exito
        baseline: Baseline a superar
    """
    name: str
    problem_type: str
    success_metrics: Dict[str, float]
    baseline: Optional[Dict[str, float]] = None
    current_phase: MLLifecyclePhase = MLLifecyclePhase.PROBLEM_DEFINITION

    def define_success_criteria(
        self,
        metric_name: str,
        threshold: float,
        direction: str = "higher_is_better"
    ) -> None:
        """Define criterios de exito medibles."""
        self.success_metrics[metric_name] = {
            "threshold": threshold,
            "direction": direction,
            "achieved": False
        }

    def check_readiness_for_production(self) -> Dict[str, bool]:
        """
        Verifica si el proyecto cumple criterios para produccion.

        Returns:
            Dict con checklist de requisitos
        """
        checklist = {
            "metrics_defined": len(self.success_metrics) > 0,
            "baseline_established": self.baseline is not None,
            "baseline_beaten": self._check_baseline_beaten(),
            "data_versioned": False,  # Verificar externamente
            "model_versioned": False,  # Verificar externamente
            "tests_passing": False,    # Verificar externamente
            "monitoring_configured": False  # Verificar externamente
        }
        return checklist

    def _check_baseline_beaten(self) -> bool:
        """Verifica si el modelo supera el baseline."""
        if self.baseline is None:
            return False
        # Logica de comparacion aqui
        return True


@dataclass
class DataVersion:
    """
    Control de versiones para datasets.

    En ciberseguridad, crucial para:
    - Reproducir entrenamientos de detectores de malware
    - Auditar que datos se usaron para decisiones
    - Detectar data poisoning
    """
    dataset_name: str
    version: str
    creation_date: datetime
    row_count: int
    column_count: int
    schema_hash: str
    storage_path: str

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pandas DataFrame
        dataset_name: str,
        storage_path: str
    ) -> "DataVersion":
        """Crea version desde DataFrame."""
        schema_str = str(df.dtypes.to_dict())
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:12]

        return cls(
            dataset_name=dataset_name,
            version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            creation_date=datetime.now(),
            row_count=len(df),
            column_count=len(df.columns),
            schema_hash=schema_hash,
            storage_path=storage_path
        )


@dataclass
class ModelVersion:
    """
    Control de versiones para modelos.

    Attributes:
        model_name: Nombre del modelo
        version: Version semantica o hash
        training_data_version: Version de datos de entrenamiento
        hyperparameters: Hiperparametros usados
        metrics: Metricas de evaluacion
    """
    model_name: str
    version: str
    training_data_version: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    framework: str
    model_path: str
    created_at: datetime

    def to_model_card(self) -> str:
        """Genera model card en formato texto."""
        card = f"""
# Model Card: {self.model_name}

## Version: {self.version}
Created: {self.created_at.isoformat()}

## Training Data
Version: {self.training_data_version}

## Hyperparameters
{self._format_dict(self.hyperparameters)}

## Metrics
{self._format_dict(self.metrics)}

## Framework
{self.framework}

## Model Artifacts
Path: {self.model_path}
"""
        return card

    def _format_dict(self, d: Dict) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in d.items())
```

## 3. ML vs Software Tradicional

### Diferencias Fundamentales

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           SOFTWARE TRADICIONAL vs MACHINE LEARNING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOFTWARE TRADICIONAL                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    Input ────► [ Codigo/Reglas ] ────► Output                      │   │
│  │                      │                                              │   │
│  │                      │                                              │   │
│  │              Determinista                                           │   │
│  │              Debuggeable paso a paso                                │   │
│  │              Comportamiento predecible                              │   │
│  │              Version = codigo                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MACHINE LEARNING                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    Input ────► [ Modelo Aprendido ] ────► Output (probabilistico)  │   │
│  │                      ▲                                              │   │
│  │                      │                                              │   │
│  │                  [ Datos ]                                          │   │
│  │                                                                     │   │
│  │              Probabilistico                                         │   │
│  │              "Black box" parcial                                    │   │
│  │              Comportamiento depende de datos                        │   │
│  │              Version = codigo + datos + modelo + config             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tabla Comparativa

```
┌──────────────────────┬────────────────────────┬────────────────────────────┐
│      Aspecto         │  Software Tradicional  │     Machine Learning       │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Artefacto principal  │ Codigo                 │ Codigo + Datos + Modelo    │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Testing              │ Unit tests, integracion│ + Data tests, Model tests  │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Versionado           │ Git (codigo)           │ Git + DVC/MLflow (datos,   │
│                      │                        │ modelos, experimentos)     │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ CI/CD                │ Build -> Test -> Deploy│ + Train -> Validate ->     │
│                      │                        │ Register -> Deploy         │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Monitoreo            │ Logs, metricas sistema │ + Data drift, Model drift, │
│                      │                        │ Prediction distribution    │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Degradacion          │ Bugs, memory leaks     │ + Concept drift, Stale     │
│                      │                        │ features, Feedback loops   │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Rollback             │ Deploy version anterior│ + Reentrenar con datos     │
│                      │                        │ anteriores, A/B test       │
├──────────────────────┼────────────────────────┼────────────────────────────┤
│ Debugging            │ Stack traces, logs     │ + Feature importance,      │
│                      │                        │ SHAP, ejemplos fallidos    │
└──────────────────────┴────────────────────────┴────────────────────────────┘
```

### Codigo: Diferencias en Testing

```python
import pytest
from typing import List, Tuple
import numpy as np
from sklearn.base import BaseEstimator


# ==============================================================================
# SOFTWARE TRADICIONAL: Testing Determinista
# ==============================================================================

def calculate_risk_score(
    failed_logins: int,
    unusual_hours: bool,
    new_device: bool
) -> float:
    """Calcula risk score con reglas fijas."""
    score = 0.0

    if failed_logins > 3:
        score += 0.3
    if failed_logins > 10:
        score += 0.3

    if unusual_hours:
        score += 0.2

    if new_device:
        score += 0.2

    return min(score, 1.0)


class TestRiskScoreTradicional:
    """Tests para software tradicional: deterministas."""

    def test_no_risk_factors_returns_zero(self):
        """Sin factores de riesgo = score 0."""
        assert calculate_risk_score(0, False, False) == 0.0

    def test_all_risk_factors_returns_one(self):
        """Todos los factores = score maximo 1.0."""
        assert calculate_risk_score(15, True, True) == 1.0

    def test_specific_scenario(self):
        """Escenario especifico = resultado exacto."""
        # Determinista: siempre el mismo resultado
        assert calculate_risk_score(5, True, False) == 0.5


# ==============================================================================
# MACHINE LEARNING: Testing Probabilistico
# ==============================================================================

class MLRiskScorer:
    """Risk scorer basado en ML."""

    def __init__(self, model: BaseEstimator, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict_risk(self, features: np.ndarray) -> float:
        """Predice riesgo (probabilistico)."""
        proba = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        return float(proba)

    def is_high_risk(self, features: np.ndarray) -> bool:
        """Clasificacion binaria."""
        return self.predict_risk(features) >= self.threshold


class TestMLRiskScorer:
    """
    Tests para ML: NO deterministas, basados en metricas y propiedades.
    """

    @pytest.fixture
    def trained_model(self) -> MLRiskScorer:
        """Fixture: modelo entrenado para tests."""
        # En la practica, cargar modelo versionado
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Simular entrenamiento
        X_train = np.random.randn(100, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        model.fit(X_train, y_train)
        return MLRiskScorer(model)

    def test_output_in_valid_range(self, trained_model: MLRiskScorer):
        """Output siempre entre 0 y 1."""
        for _ in range(100):
            features = np.random.randn(3)
            score = trained_model.predict_risk(features)
            assert 0.0 <= score <= 1.0

    def test_model_not_random(self, trained_model: MLRiskScorer):
        """Mismo input = mismo output (seed fija)."""
        features = np.array([1.0, 0.5, -0.3])
        score1 = trained_model.predict_risk(features)
        score2 = trained_model.predict_risk(features)
        assert score1 == score2

    def test_minimum_recall_on_test_set(self, trained_model: MLRiskScorer):
        """
        CRITICO: Recall minimo en casos de alto riesgo.
        En ciberseguridad, preferimos falsos positivos a falsos negativos.
        """
        # Test set con ground truth
        X_test = np.array([
            [2.0, 1.0, 0.5],   # Alto riesgo real
            [1.5, 0.8, 0.3],   # Alto riesgo real
            [-1.0, -0.5, 0.1], # Bajo riesgo real
            [-0.5, -0.3, 0.0], # Bajo riesgo real
        ])
        y_true = np.array([1, 1, 0, 0])

        y_pred = [trained_model.is_high_risk(x) for x in X_test]

        # Calcular recall para clase positiva
        true_positives = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp)
        actual_positives = sum(y_true)
        recall = true_positives / actual_positives if actual_positives > 0 else 0

        # Recall minimo aceptable: 80%
        assert recall >= 0.8, f"Recall {recall:.2%} bajo minimo 80%"

    def test_calibration_approximate(self, trained_model: MLRiskScorer):
        """
        Calibracion: probabilidad predicha ~= frecuencia real.
        Si predice 0.7, deberia acertar ~70% de las veces.
        """
        # Test simplificado de calibracion
        predictions = []
        actuals = []

        for _ in range(100):
            features = np.random.randn(3)
            prob = trained_model.predict_risk(features)
            actual = features[0] > 0  # Ground truth simulado

            predictions.append(prob)
            actuals.append(actual)

        # Verificar calibracion en bins
        # (En practica, usar calibration_curve de sklearn)
        high_conf_preds = [p for p in predictions if p > 0.7]
        # Test basico: deberia haber predicciones en distintos rangos
        assert len(high_conf_preds) > 0 or len(predictions) > 50


# ==============================================================================
# DATA TESTS: Validacion de datos de entrada
# ==============================================================================

class DataValidator:
    """Valida datos antes de inferencia."""

    def __init__(
        self,
        expected_features: List[str],
        feature_ranges: dict[str, Tuple[float, float]]
    ):
        self.expected_features = expected_features
        self.feature_ranges = feature_ranges

    def validate(self, data: dict) -> Tuple[bool, List[str]]:
        """
        Valida datos de entrada.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Verificar features requeridas
        for feature in self.expected_features:
            if feature not in data:
                errors.append(f"Missing feature: {feature}")

        # Verificar rangos
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in data:
                value = data[feature]
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"Feature {feature}={value} outside range [{min_val}, {max_val}]"
                    )

        # Verificar tipos (no NaN, no infinitos)
        for feature, value in data.items():
            if isinstance(value, float):
                if np.isnan(value):
                    errors.append(f"Feature {feature} is NaN")
                if np.isinf(value):
                    errors.append(f"Feature {feature} is infinite")

        return len(errors) == 0, errors


class TestDataValidation:
    """Tests de validacion de datos."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        return DataValidator(
            expected_features=["failed_logins", "session_duration", "requests_per_min"],
            feature_ranges={
                "failed_logins": (0, 1000),
                "session_duration": (0, 86400),  # Max 24 horas en segundos
                "requests_per_min": (0, 10000),
            }
        )

    def test_valid_data_passes(self, validator: DataValidator):
        """Datos validos pasan validacion."""
        data = {
            "failed_logins": 5,
            "session_duration": 3600,
            "requests_per_min": 100
        }
        is_valid, errors = validator.validate(data)
        assert is_valid
        assert len(errors) == 0

    def test_missing_feature_fails(self, validator: DataValidator):
        """Feature faltante = fallo."""
        data = {
            "failed_logins": 5,
            "session_duration": 3600,
            # Falta requests_per_min
        }
        is_valid, errors = validator.validate(data)
        assert not is_valid
        assert any("Missing feature" in e for e in errors)

    def test_out_of_range_fails(self, validator: DataValidator):
        """Valor fuera de rango = fallo."""
        data = {
            "failed_logins": -1,  # Negativo invalido
            "session_duration": 3600,
            "requests_per_min": 100
        }
        is_valid, errors = validator.validate(data)
        assert not is_valid

    def test_nan_value_fails(self, validator: DataValidator):
        """NaN = fallo."""
        data = {
            "failed_logins": float('nan'),
            "session_duration": 3600,
            "requests_per_min": 100
        }
        is_valid, errors = validator.validate(data)
        assert not is_valid
        assert any("NaN" in e for e in errors)
```

## 4. Niveles de Madurez MLOps

### Modelo de Madurez

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NIVELES DE MADUREZ MLOps                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  NIVEL 0: Manual / Ad-hoc                                                   │
│  ═══════════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Caracteristicas:                                                   │   │
│  │  • Data scientists trabajan en notebooks                            │   │
│  │  • Modelo se "tira por encima de la pared" a ops                   │   │
│  │  • Sin versionado de datos ni modelos                               │   │
│  │  • Despliegue manual y esporadico                                   │   │
│  │  • Sin monitoreo en produccion                                      │   │
│  │                                                                     │   │
│  │  Workflow:                                                          │   │
│  │  [Notebook] ──manual──> [pickle file] ──email──> [Ops team]        │   │
│  │                                                                     │   │
│  │  Problemas:                                                         │   │
│  │  • "Funciona en mi maquina"                                         │   │
│  │  • No reproducible                                                  │   │
│  │  • Semanas/meses para desplegar                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                                    │                                        │
│                                    ▼                                        │
│                                                                             │
│  NIVEL 1: ML Pipeline Automation                                            │
│  ═══════════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Caracteristicas:                                                   │   │
│  │  • Pipeline de entrenamiento automatizado                           │   │
│  │  • Versionado de datos y modelos                                    │   │
│  │  • Experiment tracking (MLflow, W&B)                                │   │
│  │  • Entrenamiento reproducible                                       │   │
│  │  • Despliegue semi-automatico                                       │   │
│  │                                                                     │   │
│  │  Workflow:                                                          │   │
│  │  [Data] ──pipeline──> [Train] ──> [Validate] ──> [Register]        │   │
│  │           automatico     │           │              │               │   │
│  │                          └───────────┴──────────────┘               │   │
│  │                               Tracking & Versioning                 │   │
│  │                                                                     │   │
│  │  Mejoras vs Nivel 0:                                                │   │
│  │  • Reproducibilidad                                                 │   │
│  │  • Dias para desplegar (no semanas)                                 │   │
│  │  • Comparacion de experimentos                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                                    │                                        │
│                                    ▼                                        │
│                                                                             │
│  NIVEL 2: CI/CD for ML (Full MLOps)                                         │
│  ═══════════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Caracteristicas:                                                   │   │
│  │  • CI/CD automatico para codigo, datos y modelos                    │   │
│  │  • Testing automatizado (unit, integracion, modelo)                 │   │
│  │  • Feature store para reutilizacion                                 │   │
│  │  • Monitoreo completo (drift, performance)                          │   │
│  │  • Reentrenamiento automatico basado en triggers                    │   │
│  │                                                                     │   │
│  │  Workflow:                                                          │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │                                                            │    │   │
│  │  │  [Feature Store] ◄──────────────────────────────────┐     │    │   │
│  │  │        │                                             │     │    │   │
│  │  │        ▼                                             │     │    │   │
│  │  │  [CI Pipeline]──>[Train]──>[Test]──>[Register]──>[CD]│     │    │   │
│  │  │        │              ▲                        │     │     │    │   │
│  │  │        │              │                        ▼     │     │    │   │
│  │  │  [Data Tests]    [Retrain Trigger]◄────[Monitoring]  │     │    │   │
│  │  │                                                      │     │    │   │
│  │  └──────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │  Mejoras vs Nivel 1:                                                │   │
│  │  • Minutos/horas para desplegar                                     │   │
│  │  • Autorecuperacion ante drift                                      │   │
│  │  • Gobernanza completa                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementacion Nivel 1: Pipeline Basico

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import hashlib


class PipelineStep(ABC):
    """Paso abstracto de pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del paso."""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el paso.

        Args:
            context: Datos del contexto del pipeline

        Returns:
            Contexto actualizado
        """
        pass

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """Valida inputs requeridos."""
        return True


@dataclass
class PipelineRun:
    """Registro de ejecucion de pipeline."""
    run_id: str
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    steps_completed: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


class MLPipeline:
    """
    Pipeline de ML con tracking integrado.

    Implementa Nivel 1 de madurez MLOps:
    - Pasos encadenados
    - Tracking de metricas y artefactos
    - Reproducibilidad via config hash
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.steps: List[PipelineStep] = []
        self.runs: List[PipelineRun] = []

    def add_step(self, step: PipelineStep) -> "MLPipeline":
        """Anade paso al pipeline (fluent interface)."""
        self.steps.append(step)
        return self

    def _generate_run_id(self) -> str:
        """Genera ID unico para la ejecucion."""
        config_str = json.dumps(self.config, sort_keys=True)
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.name}_{config_str}_{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def run(self, initial_context: Optional[Dict[str, Any]] = None) -> PipelineRun:
        """
        Ejecuta el pipeline completo.

        Args:
            initial_context: Contexto inicial (datos, config, etc.)

        Returns:
            PipelineRun con resultado de la ejecucion
        """
        run = PipelineRun(
            run_id=self._generate_run_id(),
            pipeline_name=self.name,
            start_time=datetime.now()
        )

        context = initial_context or {}
        context["config"] = self.config
        context["run_id"] = run.run_id

        try:
            for step in self.steps:
                print(f"[{run.run_id}] Executing step: {step.name}")

                if not step.validate_inputs(context):
                    raise ValueError(f"Invalid inputs for step: {step.name}")

                context = step.execute(context)
                run.steps_completed.append(step.name)

                # Extraer metricas si las hay
                if "metrics" in context:
                    run.metrics.update(context["metrics"])

                print(f"[{run.run_id}] Completed step: {step.name}")

            run.status = "success"

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            print(f"[{run.run_id}] Pipeline failed: {e}")
            raise

        finally:
            run.end_time = datetime.now()
            self.runs.append(run)

        return run


# Ejemplo de pasos concretos para pipeline de ciberseguridad

class DataIngestionStep(PipelineStep):
    """Paso de ingesta de datos de logs de seguridad."""

    @property
    def name(self) -> str:
        return "data_ingestion"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Carga datos de logs."""
        import pandas as pd

        data_path = context["config"]["data_path"]

        # Simular carga de logs de seguridad
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="H"),
            "source_ip": [f"192.168.1.{i % 255}" for i in range(1000)],
            "dest_ip": [f"10.0.0.{i % 255}" for i in range(1000)],
            "bytes_sent": [abs(hash(f"sent_{i}")) % 10000 for i in range(1000)],
            "bytes_received": [abs(hash(f"recv_{i}")) % 10000 for i in range(1000)],
            "is_malicious": [i % 20 == 0 for i in range(1000)]  # 5% malicioso
        })

        context["raw_data"] = df
        context["data_version"] = DataVersion.from_dataframe(
            df, "security_logs", data_path
        )

        return context


class FeatureEngineeringStep(PipelineStep):
    """Extrae features para deteccion de anomalias."""

    @property
    def name(self) -> str:
        return "feature_engineering"

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        return "raw_data" in context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Genera features de seguridad."""
        import pandas as pd
        import numpy as np

        df = context["raw_data"]

        # Features agregadas por IP origen
        features = df.groupby("source_ip").agg({
            "bytes_sent": ["mean", "std", "max"],
            "bytes_received": ["mean", "std", "max"],
            "dest_ip": "nunique",  # Numero de destinos unicos
            "is_malicious": "max"  # Target
        }).reset_index()

        # Aplanar columnas multi-nivel
        features.columns = [
            "_".join(col).strip("_") for col in features.columns
        ]

        context["features"] = features
        context["feature_columns"] = [
            c for c in features.columns
            if c not in ["source_ip", "is_malicious_max"]
        ]

        return context


class ModelTrainingStep(PipelineStep):
    """Entrena modelo de deteccion."""

    @property
    def name(self) -> str:
        return "model_training"

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        return "features" in context and "feature_columns" in context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Entrena modelo."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, f1_score

        df = context["features"]
        feature_cols = context["feature_columns"]
        config = context["config"]

        X = df[feature_cols].fillna(0)
        y = df["is_malicious_max"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.get("test_size", 0.2),
            random_state=config.get("random_state", 42),
            stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=config.get("random_state", 42)
        )

        model.fit(X_train, y_train)

        # Evaluacion
        y_pred = model.predict(X_test)

        context["model"] = model
        context["X_test"] = X_test
        context["y_test"] = y_test
        context["metrics"] = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }

        return context


class ModelValidationStep(PipelineStep):
    """Valida modelo contra criterios de aceptacion."""

    @property
    def name(self) -> str:
        return "model_validation"

    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        return "model" in context and "metrics" in context

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Valida metricas contra thresholds."""
        config = context["config"]
        metrics = context["metrics"]

        min_recall = config.get("min_recall", 0.8)
        min_precision = config.get("min_precision", 0.5)

        validation_passed = (
            metrics["recall"] >= min_recall and
            metrics["precision"] >= min_precision
        )

        context["validation_passed"] = validation_passed
        context["validation_details"] = {
            "recall_check": metrics["recall"] >= min_recall,
            "precision_check": metrics["precision"] >= min_precision,
            "thresholds": {"min_recall": min_recall, "min_precision": min_precision}
        }

        if not validation_passed:
            raise ValueError(
                f"Model validation failed. Metrics: {metrics}, "
                f"Required: recall>={min_recall}, precision>={min_precision}"
            )

        return context


# Uso del pipeline
def run_security_ml_pipeline():
    """Ejecuta pipeline de ML para seguridad."""

    config = {
        "data_path": "/data/security_logs/",
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10,
        "min_recall": 0.75,
        "min_precision": 0.5
    }

    pipeline = MLPipeline("security_anomaly_detection", config)

    pipeline.add_step(DataIngestionStep())
    pipeline.add_step(FeatureEngineeringStep())
    pipeline.add_step(ModelTrainingStep())
    pipeline.add_step(ModelValidationStep())

    run = pipeline.run()

    print(f"\nPipeline Run: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Steps: {run.steps_completed}")
    print(f"Metrics: {run.metrics}")

    return run
```

## 5. Ecosistema de Herramientas MLOps

### Mapa de Herramientas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ECOSISTEMA MLOps                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA MANAGEMENT                    EXPERIMENT TRACKING                     │
│  ┌─────────────────────────┐       ┌─────────────────────────┐             │
│  │ • DVC (Data Version     │       │ • MLflow                │             │
│  │   Control)              │       │ • Weights & Biases      │             │
│  │ • Delta Lake            │       │ • Neptune.ai            │             │
│  │ • LakeFS                │       │ • Comet ML              │             │
│  │ • Pachyderm             │       │ • ClearML               │             │
│  └─────────────────────────┘       └─────────────────────────┘             │
│                                                                             │
│  FEATURE STORES                    ORCHESTRATION                            │
│  ┌─────────────────────────┐       ┌─────────────────────────┐             │
│  │ • Feast                 │       │ • Dagster *             │             │
│  │ • Tecton                │       │ • Airflow (legacy)      │             │
│  │ • Hopsworks             │       │ • Prefect               │             │
│  │ • Amazon SageMaker FS   │       │ • Kubeflow Pipelines    │             │
│  └─────────────────────────┘       │ • Argo Workflows        │             │
│                                    └─────────────────────────┘             │
│                                                                             │
│  MODEL SERVING                     MONITORING                               │
│  ┌─────────────────────────┐       ┌─────────────────────────┐             │
│  │ • Triton Inference      │       │ • Evidently             │             │
│  │ • TorchServe            │       │ • WhyLabs               │             │
│  │ • BentoML               │       │ • Alibi Detect          │             │
│  │ • Seldon Core           │       │ • Fiddler               │             │
│  │ • KServe                │       │ • Arize                 │             │
│  │ • vLLM (LLMs)           │       │ • Prometheus + Grafana  │             │
│  └─────────────────────────┘       └─────────────────────────┘             │
│                                                                             │
│  CI/CD FOR ML                      PLATFORMS (End-to-End)                   │
│  ┌─────────────────────────┐       ┌─────────────────────────┐             │
│  │ • GitHub Actions        │       │ • Kubeflow              │             │
│  │ • GitLab CI/CD          │       │ • MLflow + Databricks   │             │
│  │ • CML (DVC)             │       │ • Amazon SageMaker      │             │
│  │ • Jenkins               │       │ • Google Vertex AI      │             │
│  │                         │       │ • Azure ML              │             │
│  └─────────────────────────┘       └─────────────────────────┘             │
│                                                                             │
│  * Recomendado sobre Airflow para nuevos proyectos                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Seleccion de Herramientas por Caso de Uso

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict


class TeamSize(Enum):
    SOLO = "solo"
    SMALL = "small"      # 2-5
    MEDIUM = "medium"    # 5-20
    LARGE = "large"      # 20+


class InfrastructureType(Enum):
    LOCAL = "local"
    CLOUD_SINGLE = "cloud_single"  # Un solo cloud
    MULTI_CLOUD = "multi_cloud"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class MLUseCase(Enum):
    BATCH_PREDICTION = "batch"
    REAL_TIME = "real_time"
    STREAMING = "streaming"
    EDGE = "edge"
    LLM = "llm"


@dataclass
class MLOpsToolRecommendation:
    """Recomendacion de herramientas MLOps."""

    team_size: TeamSize
    infrastructure: InfrastructureType
    use_case: MLUseCase
    budget: str  # "low", "medium", "high"

    def get_recommendations(self) -> Dict[str, List[str]]:
        """
        Genera recomendaciones de herramientas.

        Returns:
            Dict con categoria -> lista de herramientas
        """
        recs = {
            "experiment_tracking": [],
            "orchestration": [],
            "feature_store": [],
            "model_serving": [],
            "monitoring": [],
            "data_versioning": []
        }

        # Experiment Tracking
        if self.budget == "low":
            recs["experiment_tracking"] = ["MLflow (self-hosted)", "ClearML (free tier)"]
        elif self.team_size in [TeamSize.MEDIUM, TeamSize.LARGE]:
            recs["experiment_tracking"] = ["Weights & Biases", "Neptune.ai"]
        else:
            recs["experiment_tracking"] = ["MLflow", "Weights & Biases"]

        # Orchestration
        recs["orchestration"] = ["Dagster"]  # Siempre Dagster sobre Airflow
        if self.infrastructure == InfrastructureType.CLOUD_SINGLE:
            recs["orchestration"].append("Cloud-native (Step Functions, Cloud Composer)")

        # Feature Store
        if self.team_size == TeamSize.SOLO:
            recs["feature_store"] = ["Feast (local)", "Sin feature store (pandas)"]
        elif self.budget == "high":
            recs["feature_store"] = ["Tecton", "Hopsworks"]
        else:
            recs["feature_store"] = ["Feast"]

        # Model Serving
        if self.use_case == MLUseCase.LLM:
            recs["model_serving"] = ["vLLM", "TGI (HuggingFace)"]
        elif self.use_case == MLUseCase.REAL_TIME:
            recs["model_serving"] = ["Triton Inference Server", "BentoML"]
        elif self.use_case == MLUseCase.EDGE:
            recs["model_serving"] = ["ONNX Runtime", "TensorFlow Lite"]
        else:
            recs["model_serving"] = ["BentoML", "FastAPI custom"]

        # Monitoring
        if self.budget == "low":
            recs["monitoring"] = ["Evidently", "Alibi Detect"]
        else:
            recs["monitoring"] = ["WhyLabs", "Arize", "Evidently"]

        # Data Versioning
        recs["data_versioning"] = ["DVC"]
        if self.infrastructure in [InfrastructureType.CLOUD_SINGLE, InfrastructureType.MULTI_CLOUD]:
            recs["data_versioning"].append("Delta Lake / LakeFS")

        return recs

    def generate_stack_recommendation(self) -> str:
        """Genera recomendacion completa en texto."""
        recs = self.get_recommendations()

        output = f"""
# MLOps Stack Recommendation

## Context
- Team Size: {self.team_size.value}
- Infrastructure: {self.infrastructure.value}
- Use Case: {self.use_case.value}
- Budget: {self.budget}

## Recommended Stack

### Experiment Tracking
{chr(10).join(f'- {tool}' for tool in recs['experiment_tracking'])}

### Orchestration
{chr(10).join(f'- {tool}' for tool in recs['orchestration'])}

### Feature Store
{chr(10).join(f'- {tool}' for tool in recs['feature_store'])}

### Model Serving
{chr(10).join(f'- {tool}' for tool in recs['model_serving'])}

### Monitoring
{chr(10).join(f'- {tool}' for tool in recs['monitoring'])}

### Data Versioning
{chr(10).join(f'- {tool}' for tool in recs['data_versioning'])}
"""
        return output


# Ejemplo para equipo de ciberseguridad
def recommend_security_ml_stack():
    """Recomienda stack para equipo de seguridad."""

    recommendation = MLOpsToolRecommendation(
        team_size=TeamSize.SMALL,
        infrastructure=InfrastructureType.HYBRID,  # Cloud + on-prem por compliance
        use_case=MLUseCase.REAL_TIME,  # Deteccion en tiempo real
        budget="medium"
    )

    print(recommendation.generate_stack_recommendation())

    # Consideraciones especiales para seguridad
    print("""
## Security-Specific Considerations

### Data Sensitivity
- Usar Feast on-premise para features sensibles
- DVC con storage encriptado
- MLflow con autenticacion

### Compliance
- Audit trail completo con MLflow
- Model cards obligatorios
- Data lineage con DVC

### Real-time Requirements
- Triton con batching adaptativo
- Latencia < 100ms para alertas
- Fallback a reglas si modelo falla

### Adversarial Considerations
- Monitoreo de distribution shift (posible data poisoning)
- A/B testing para validar robustez
- Model versioning para rollback rapido
""")
```

## 6. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESUMEN: Introduccion a MLOps                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DEFINICION                                                                 │
│  ───────────────────────────────────────────────────────────────────────   │
│  MLOps = Practicas para desplegar y mantener ML en produccion              │
│  Combina: Machine Learning + DevOps + Data Engineering                      │
│                                                                             │
│  CICLO DE VIDA ML                                                           │
│  ───────────────────────────────────────────────────────────────────────   │
│  Problema → Datos → Features → Entrenamiento → Evaluacion →                │
│  Despliegue → Monitoreo → (vuelta al inicio)                               │
│                                                                             │
│  ML vs SOFTWARE TRADICIONAL                                                 │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Software: Version = codigo                                               │
│  • ML: Version = codigo + datos + modelo + config                           │
│  • Testing ML: probabilistico, no determinista                              │
│  • Degradacion ML: drift, no solo bugs                                      │
│                                                                             │
│  NIVELES DE MADUREZ                                                         │
│  ───────────────────────────────────────────────────────────────────────   │
│  Nivel 0: Manual (notebooks, sin tracking)                                  │
│  Nivel 1: Pipeline automatizado + tracking                                  │
│  Nivel 2: CI/CD completo + monitoreo + auto-retraining                     │
│                                                                             │
│  HERRAMIENTAS CLAVE                                                         │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Tracking: MLflow, W&B                                                    │
│  • Orquestacion: Dagster (no Airflow)                                       │
│  • Feature Store: Feast                                                     │
│  • Serving: Triton, BentoML, vLLM                                           │
│  • Monitoreo: Evidently, WhyLabs                                            │
│  • Versionado: DVC                                                          │
│                                                                             │
│  CONSIDERACIONES CIBERSEGURIDAD                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Datos sensibles: compliance, encriptacion                                │
│  • Tiempo real: latencia critica para alertas                               │
│  • Adversarial: monitoreo de data poisoning                                 │
│  • Audit trail: gobernanza completa                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Experiment Tracking con MLflow - tracking, projects, models, registry
