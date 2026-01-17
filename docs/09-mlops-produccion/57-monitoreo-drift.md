# Monitoreo y Deteccion de Drift

## 1. Tipos de Drift

### Por que los Modelos se Degradan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POR QUE LOS MODELOS SE DEGRADAN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRENAMIENTO                         PRODUCCION (6 meses despues)         │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Distribucion de datos:                Distribucion de datos:               │
│  ┌───────────────────────┐             ┌───────────────────────┐           │
│  │     ████████          │             │         ████████████  │           │
│  │   ██████████████      │      →      │       ████████████████│           │
│  │ ████████████████████  │             │     ████████████████  │           │
│  │██████████████████████ │             │   ████████████        │           │
│  └───────────────────────┘             └───────────────────────┘           │
│  Mean: 50, Std: 10                     Mean: 70, Std: 15                   │
│                                        ← DATA DRIFT                         │
│                                                                             │
│  Relacion X → Y:                       Relacion X → Y:                      │
│  ┌───────────────────────┐             ┌───────────────────────┐           │
│  │       /               │             │    \                  │           │
│  │      /                │      →      │     \                 │           │
│  │     /                 │             │      \                │           │
│  │    /                  │             │       \               │           │
│  └───────────────────────┘             └───────────────────────┘           │
│  Correlacion positiva                  Correlacion negativa!               │
│                                        ← CONCEPT DRIFT                      │
│                                                                             │
│  CAUSAS COMUNES:                                                            │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Cambios en comportamiento de usuarios                                    │
│  • Nuevos tipos de amenazas (ciberseguridad)                               │
│  • Cambios estacionales                                                     │
│  • Cambios en fuentes de datos                                             │
│  • Bugs en pipelines de datos                                              │
│  • Cambios en features upstream                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Taxonomia de Drift

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIPOS DE DRIFT                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DATA DRIFT (Covariate Shift)                                            │
│  ═══════════════════════════════════════════════════════════════════════   │
│  • QUE ES: Cambio en distribucion de features P(X)                          │
│  • RELACION: P(Y|X) se mantiene, solo cambia P(X)                          │
│  • EJEMPLO: Antes 80% trafico diurno, ahora 50% diurno                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Training:        Production:        Feature "hora del dia"        │   │
│  │  P(X_train)       P(X_prod)          cambia, pero la relacion      │   │
│  │    ▄▄▄▄             ▄▄▄▄▄▄          hora → fraude se mantiene       │   │
│  │   ██████▄         ▄███████                                          │   │
│  │  █████████       █████████                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. CONCEPT DRIFT                                                           │
│  ═══════════════════════════════════════════════════════════════════════   │
│  • QUE ES: Cambio en relacion entre features y target P(Y|X)               │
│  • EJEMPLO: Nuevos patrones de fraude que el modelo no conoce              │
│  • MAS PELIGROSO: El modelo apredio relaciones obsoletas                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Training:        Production:                                       │   │
│  │  Fraude = login   Fraude = transacciones   La DEFINICION de         │   │
│  │  desde paises     de alto valor en         fraude cambio            │   │
│  │  exoticos         horario nocturno                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. PREDICTION DRIFT (Label Drift)                                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│  • QUE ES: Cambio en distribucion de predicciones P(Y_pred)                │
│  • SINTOMA: No necesariamente implica degradacion                          │
│  • UTIL: Indicador temprano de otros tipos de drift                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Training:           Production:                                    │   │
│  │  5% positivos        15% positivos     Mas predicciones positivas   │   │
│  │  ▓░░░░░░░░░░░░░░░░   ▓▓▓░░░░░░░░░░░░   (puede ser real o drift)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4. MODEL DRIFT (Performance Degradation)                                   │
│  ═══════════════════════════════════════════════════════════════════════   │
│  • QUE ES: Degradacion de metricas del modelo en produccion                │
│  • CAUSA: Resultado de data drift o concept drift                          │
│  • DETECCION: Requiere ground truth (labels retrasados)                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  F1 Score over time:                                                │   │
│  │  0.95 ─────────┐                                                    │   │
│  │                 \                                                   │   │
│  │  0.85 ───────────────\                                              │   │
│  │                       \                                             │   │
│  │  0.75 ─────────────────────\   ← Performance degradation           │   │
│  │        Jan    Feb    Mar   Apr                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Metodos de Deteccion de Drift

### Metodos Estadisticos

```python
"""
Metodos estadisticos para deteccion de drift.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DriftType(Enum):
    """Tipos de drift."""
    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass
class DriftResult:
    """Resultado de deteccion de drift."""
    feature: str
    test_name: str
    statistic: float
    p_value: float
    threshold: float
    drift_detected: DriftType
    details: Optional[Dict[str, Any]] = None


class StatisticalDriftDetector:
    """
    Detector de drift usando tests estadisticos.

    Metodos:
    - KS Test: Para features numericas continuas
    - Chi-squared: Para features categoricas
    - PSI: Population Stability Index
    - Wasserstein Distance: Distancia entre distribuciones
    """

    def __init__(
        self,
        warning_threshold: float = 0.05,
        drift_threshold: float = 0.01
    ):
        """
        Args:
            warning_threshold: p-value para warning
            drift_threshold: p-value para drift
        """
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold

    def ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftResult:
        """
        Kolmogorov-Smirnov test para features numericas.

        H0: Las dos muestras vienen de la misma distribucion.

        Ventajas:
        - No asume ninguna distribucion especifica
        - Sensible a cambios en forma, location y scale

        Args:
            reference: Datos de referencia (training)
            current: Datos actuales (production)
            feature_name: Nombre del feature

        Returns:
            DriftResult con estadistico y p-value
        """
        statistic, p_value = stats.ks_2samp(reference, current)

        if p_value < self.drift_threshold:
            drift_type = DriftType.DRIFT
        elif p_value < self.warning_threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT

        return DriftResult(
            feature=feature_name,
            test_name="Kolmogorov-Smirnov",
            statistic=statistic,
            p_value=p_value,
            threshold=self.drift_threshold,
            drift_detected=drift_type,
            details={
                "reference_mean": float(np.mean(reference)),
                "current_mean": float(np.mean(current)),
                "reference_std": float(np.std(reference)),
                "current_std": float(np.std(current))
            }
        )

    def chi_squared_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature"
    ) -> DriftResult:
        """
        Chi-squared test para features categoricas.

        Compara frecuencias observadas vs esperadas.

        Args:
            reference: Categorias de referencia
            current: Categorias actuales
        """
        # Obtener todas las categorias
        all_categories = set(reference) | set(current)

        # Contar frecuencias
        ref_counts = pd.Series(reference).value_counts()
        cur_counts = pd.Series(current).value_counts()

        # Asegurar mismas categorias
        ref_freq = np.array([ref_counts.get(c, 0) for c in all_categories])
        cur_freq = np.array([cur_counts.get(c, 0) for c in all_categories])

        # Normalizar a frecuencias esperadas
        ref_freq_norm = ref_freq / ref_freq.sum() * cur_freq.sum()

        # Chi-squared test
        # Evitar division por cero
        ref_freq_norm = np.maximum(ref_freq_norm, 1e-10)

        statistic, p_value = stats.chisquare(cur_freq, ref_freq_norm)

        if p_value < self.drift_threshold:
            drift_type = DriftType.DRIFT
        elif p_value < self.warning_threshold:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT

        return DriftResult(
            feature=feature_name,
            test_name="Chi-Squared",
            statistic=statistic,
            p_value=p_value,
            threshold=self.drift_threshold,
            drift_detected=drift_type
        )

    def psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        n_bins: int = 10
    ) -> DriftResult:
        """
        Population Stability Index.

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Interpretacion:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change (warning)
        - PSI >= 0.2: Significant change (drift)

        Args:
            reference: Datos de referencia
            current: Datos actuales
            n_bins: Numero de bins para discretizar
        """
        # Crear bins basados en referencia
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calcular frecuencias
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Convertir a proporciones (evitar ceros)
        ref_pct = (ref_hist + 1) / (len(reference) + n_bins)
        cur_pct = (cur_hist + 1) / (len(current) + n_bins)

        # Calcular PSI
        psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        # Determinar drift
        if psi_value >= 0.2:
            drift_type = DriftType.DRIFT
        elif psi_value >= 0.1:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT

        return DriftResult(
            feature=feature_name,
            test_name="PSI",
            statistic=psi_value,
            p_value=-1,  # PSI no tiene p-value
            threshold=0.2,
            drift_detected=drift_type,
            details={
                "psi_per_bin": list(zip(
                    bins[:-1].tolist(),
                    ((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).tolist()
                ))
            }
        )

    def wasserstein_distance(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "feature",
        threshold: float = 0.1
    ) -> DriftResult:
        """
        Wasserstein Distance (Earth Mover's Distance).

        Mide el "trabajo" necesario para transformar una distribucion en otra.

        Ventajas:
        - Considera la geometria del espacio de distribuciones
        - Mas robusto que KS para algunas distribuciones
        """
        # Normalizar para comparabilidad
        ref_normalized = (reference - reference.mean()) / (reference.std() + 1e-10)
        cur_normalized = (current - current.mean()) / (current.std() + 1e-10)

        distance = stats.wasserstein_distance(ref_normalized, cur_normalized)

        if distance >= threshold:
            drift_type = DriftType.DRIFT
        elif distance >= threshold * 0.5:
            drift_type = DriftType.WARNING
        else:
            drift_type = DriftType.NO_DRIFT

        return DriftResult(
            feature=feature_name,
            test_name="Wasserstein",
            statistic=distance,
            p_value=-1,
            threshold=threshold,
            drift_detected=drift_type
        )

    def detect_drift_all_features(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str]
    ) -> List[DriftResult]:
        """
        Detecta drift en todas las features.

        Args:
            reference_df: DataFrame de referencia
            current_df: DataFrame actual
            numerical_features: Lista de features numericas
            categorical_features: Lista de features categoricas

        Returns:
            Lista de DriftResult
        """
        results = []

        # Features numericas
        for feature in numerical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                ref = reference_df[feature].dropna().values
                cur = current_df[feature].dropna().values

                # KS Test
                results.append(self.ks_test(ref, cur, feature))

                # PSI
                results.append(self.psi(ref, cur, f"{feature}_psi"))

        # Features categoricas
        for feature in categorical_features:
            if feature in reference_df.columns and feature in current_df.columns:
                ref = reference_df[feature].dropna().values
                cur = current_df[feature].dropna().values

                results.append(self.chi_squared_test(ref, cur, feature))

        return results
```

## 3. Evidently: Framework de Monitoreo

### Configuracion de Evidently

```python
"""
Monitoreo de drift con Evidently AI.

Evidently es un framework open-source para monitoreo de ML.
"""
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    RegressionPreset,
    ClassificationPreset
)
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ColumnCorrelationsMetric,
    DatasetMissingValuesMetric,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import (
    DataDriftTestPreset,
    DataQualityTestPreset,
    DataStabilityTestPreset,
)
from evidently.tests import (
    TestColumnDrift,
    TestShareOfDriftedColumns,
    TestNumberOfMissingValues,
)
import pandas as pd
from typing import Optional, Dict, Any
import json


class EvidentlyMonitor:
    """
    Monitor de ML usando Evidently.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
        numerical_features: Optional[list] = None,
        categorical_features: Optional[list] = None
    ):
        """
        Inicializa monitor.

        Args:
            reference_data: Datos de referencia (training data)
            target_column: Nombre de columna target
            prediction_column: Nombre de columna de predicciones
            numerical_features: Lista de features numericas
            categorical_features: Lista de features categoricas
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.prediction_column = prediction_column

        # Column mapping
        self.column_mapping = ColumnMapping()
        self.column_mapping.target = target_column
        self.column_mapping.prediction = prediction_column

        if numerical_features:
            self.column_mapping.numerical_features = numerical_features

        if categorical_features:
            self.column_mapping.categorical_features = categorical_features

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte de data drift.

        Args:
            current_data: Datos actuales a comparar
            output_path: Path para guardar HTML report

        Returns:
            Dict con metricas de drift
        """
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DataDriftTable()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)

        # Extraer metricas como dict
        result = report.as_dict()
        return result

    def generate_full_report(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genera reporte completo incluyendo calidad de datos.
        """
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            ColumnCorrelationsMetric(),
            DatasetMissingValuesMetric(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)

        return report.as_dict()

    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
        drift_share_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Ejecuta tests de drift con pass/fail.

        Args:
            current_data: Datos actuales
            drift_share_threshold: Porcentaje de features con drift para fallar

        Returns:
            Dict con resultados de tests
        """
        test_suite = TestSuite(tests=[
            DataDriftTestPreset(),
            TestShareOfDriftedColumns(lt=drift_share_threshold),
            TestNumberOfMissingValues(eq=0),
        ])

        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        results = test_suite.as_dict()

        # Resumen
        summary = {
            "all_passed": results["summary"]["all_passed"],
            "total_tests": results["summary"]["total"],
            "passed": results["summary"]["success"],
            "failed": results["summary"]["failed"],
            "tests": []
        }

        for test in results["tests"]:
            summary["tests"].append({
                "name": test["name"],
                "status": test["status"],
                "description": test.get("description", "")
            })

        return summary

    def monitor_classification_performance(
        self,
        current_data: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Monitorea performance de modelo de clasificacion.

        Requiere que current_data tenga target y predicciones.
        """
        report = Report(metrics=[
            ClassificationPreset(),
            TargetDriftPreset(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        if output_path:
            report.save_html(output_path)

        return report.as_dict()


# =============================================================================
# EJEMPLO DE USO PARA CIBERSEGURIDAD
# =============================================================================

def security_drift_monitoring_example():
    """
    Ejemplo de monitoreo para detector de amenazas.
    """
    import numpy as np

    # Simular datos de referencia (training)
    np.random.seed(42)
    n_reference = 10000

    reference_data = pd.DataFrame({
        "bytes_sent": np.random.exponential(1000, n_reference),
        "bytes_received": np.random.exponential(500, n_reference),
        "session_duration": np.random.exponential(300, n_reference),
        "num_requests": np.random.poisson(50, n_reference),
        "is_weekend": np.random.choice([0, 1], n_reference, p=[0.7, 0.3]),
        "hour_of_day": np.random.randint(0, 24, n_reference),
        "country": np.random.choice(["US", "UK", "DE", "FR", "Other"], n_reference),
        "is_malicious": np.random.choice([0, 1], n_reference, p=[0.95, 0.05])
    })

    # Simular datos de produccion CON DRIFT
    n_current = 5000

    # Introducir data drift: mas trafico de otros paises
    current_data = pd.DataFrame({
        "bytes_sent": np.random.exponential(1500, n_current),  # Drift: mayor media
        "bytes_received": np.random.exponential(500, n_current),
        "session_duration": np.random.exponential(200, n_current),  # Drift: menor
        "num_requests": np.random.poisson(70, n_current),  # Drift: mayor
        "is_weekend": np.random.choice([0, 1], n_current, p=[0.5, 0.5]),  # Drift
        "hour_of_day": np.random.randint(0, 24, n_current),
        "country": np.random.choice(
            ["US", "UK", "DE", "FR", "Other", "CN", "RU"],  # Nuevas categorias
            n_current,
            p=[0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]  # Distribucion diferente
        ),
        "is_malicious": np.random.choice([0, 1], n_current, p=[0.90, 0.10])  # Mas malicioso
    })

    # Crear monitor
    monitor = EvidentlyMonitor(
        reference_data=reference_data,
        target_column="is_malicious",
        numerical_features=["bytes_sent", "bytes_received", "session_duration", "num_requests", "hour_of_day"],
        categorical_features=["is_weekend", "country"]
    )

    # Generar reportes
    print("=== DATA DRIFT REPORT ===")
    drift_report = monitor.generate_data_drift_report(
        current_data,
        output_path="drift_report.html"
    )

    # Verificar drift por feature
    metrics = drift_report.get("metrics", [])
    for metric in metrics:
        if "drift" in metric.get("metric", "").lower():
            print(f"Metric: {metric.get('metric')}")
            result = metric.get("result", {})
            if "drift_share" in result:
                print(f"  Drift share: {result['drift_share']:.2%}")
            if "dataset_drift" in result:
                print(f"  Dataset drift: {result['dataset_drift']}")

    # Ejecutar tests
    print("\n=== DRIFT TESTS ===")
    test_results = monitor.run_drift_tests(current_data)
    print(f"All tests passed: {test_results['all_passed']}")
    print(f"Passed: {test_results['passed']}/{test_results['total_tests']}")

    for test in test_results["tests"]:
        status_emoji = "PASS" if test["status"] == "SUCCESS" else "FAIL"
        print(f"  [{status_emoji}] {test['name']}")

    return drift_report, test_results
```

## 4. Alibi Detect

### Deteccion Avanzada de Drift

```python
"""
Deteccion de drift con Alibi Detect.

Alibi Detect ofrece metodos mas avanzados:
- Drift detectors basados en deep learning
- Outlier detection
- Adversarial detection
"""
from alibi_detect.cd import (
    KSDrift,
    ChiSquareDrift,
    MMDDrift,
    TabularDrift,
    ClassifierDrift
)
from alibi_detect.saving import save_detector, load_detector
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier


class AlibiDriftDetector:
    """
    Detector de drift usando Alibi Detect.
    """

    def __init__(self, p_val: float = 0.05):
        """
        Args:
            p_val: p-value threshold para drift
        """
        self.p_val = p_val
        self.detectors: Dict[str, Any] = {}

    def fit_ks_detector(
        self,
        reference_data: np.ndarray,
        feature_names: Optional[list] = None,
        detector_name: str = "ks"
    ) -> None:
        """
        Entrena detector KS para features numericas.

        Args:
            reference_data: Datos de referencia (n_samples, n_features)
            feature_names: Nombres de features
            detector_name: Nombre para guardar detector
        """
        detector = KSDrift(
            reference_data,
            p_val=self.p_val,
            preprocess_fn=None,  # Opcional: preprocessing
            data_type="tabular"
        )

        self.detectors[detector_name] = {
            "detector": detector,
            "feature_names": feature_names,
            "type": "KS"
        }

    def fit_chi_squared_detector(
        self,
        reference_data: np.ndarray,
        categories_per_feature: list,
        feature_names: Optional[list] = None,
        detector_name: str = "chi2"
    ) -> None:
        """
        Entrena detector Chi-squared para features categoricas.

        Args:
            reference_data: Datos de referencia (codificados como enteros)
            categories_per_feature: Lista de numero de categorias por feature
            feature_names: Nombres de features
            detector_name: Nombre para guardar detector
        """
        detector = ChiSquareDrift(
            reference_data,
            p_val=self.p_val,
            categories_per_feature=categories_per_feature
        )

        self.detectors[detector_name] = {
            "detector": detector,
            "feature_names": feature_names,
            "type": "ChiSquared"
        }

    def fit_mmd_detector(
        self,
        reference_data: np.ndarray,
        feature_names: Optional[list] = None,
        detector_name: str = "mmd"
    ) -> None:
        """
        Entrena detector MMD (Maximum Mean Discrepancy).

        MMD es mas robusto que KS para datos multidimensionales.
        Usa kernel methods para comparar distribuciones.

        Args:
            reference_data: Datos de referencia
            feature_names: Nombres de features
            detector_name: Nombre para guardar detector
        """
        detector = MMDDrift(
            reference_data,
            p_val=self.p_val,
            backend="pytorch",  # "tensorflow" o "pytorch"
            n_permutations=100
        )

        self.detectors[detector_name] = {
            "detector": detector,
            "feature_names": feature_names,
            "type": "MMD"
        }

    def fit_classifier_detector(
        self,
        reference_data: np.ndarray,
        feature_names: Optional[list] = None,
        detector_name: str = "classifier"
    ) -> None:
        """
        Entrena detector basado en clasificador.

        Idea: Si un clasificador puede distinguir entre datos de referencia
        y datos actuales, hay drift.

        Ventajas:
        - Puede capturar relaciones complejas
        - Indica QUE features contribuyen al drift
        """
        detector = ClassifierDrift(
            reference_data,
            p_val=self.p_val,
            backend="sklearn",  # Usa sklearn
            model=RandomForestClassifier(n_estimators=50, max_depth=5),
            preds_type="probs"
        )

        self.detectors[detector_name] = {
            "detector": detector,
            "feature_names": feature_names,
            "type": "Classifier"
        }

    def predict(
        self,
        current_data: np.ndarray,
        detector_name: str = "ks"
    ) -> Dict[str, Any]:
        """
        Detecta drift en datos actuales.

        Args:
            current_data: Datos a evaluar
            detector_name: Nombre del detector a usar

        Returns:
            Dict con resultados de drift
        """
        if detector_name not in self.detectors:
            raise ValueError(f"Detector {detector_name} not found")

        detector_info = self.detectors[detector_name]
        detector = detector_info["detector"]

        preds = detector.predict(current_data)

        result = {
            "detector_type": detector_info["type"],
            "is_drift": bool(preds["data"]["is_drift"]),
            "p_val": float(preds["data"]["p_val"]),
            "threshold": self.p_val,
            "distance": preds["data"].get("distance"),
        }

        # Feature-level drift si esta disponible
        if "is_drift" in preds["data"] and isinstance(preds["data"]["is_drift"], (list, np.ndarray)):
            feature_names = detector_info.get("feature_names")
            if feature_names:
                result["feature_drift"] = {
                    name: bool(drift)
                    for name, drift in zip(feature_names, preds["data"]["is_drift"])
                }

        return result

    def save(self, path: str, detector_name: str) -> None:
        """Guarda detector a disco."""
        if detector_name in self.detectors:
            save_detector(self.detectors[detector_name]["detector"], path)

    def load(self, path: str, detector_name: str) -> None:
        """Carga detector desde disco."""
        detector = load_detector(path)
        self.detectors[detector_name] = {
            "detector": detector,
            "type": "loaded"
        }


# =============================================================================
# DETECTOR TABULAR COMPLETO
# =============================================================================

class TabularDriftMonitor:
    """
    Monitor completo para datos tabulares.

    Combina multiples metodos de deteccion.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        p_val: float = 0.05
    ):
        """
        Args:
            reference_data: DataFrame de referencia
            numerical_features: Lista de features numericas
            categorical_features: Lista de features categoricas
            p_val: Threshold de p-value
        """
        self.reference_data = reference_data
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.p_val = p_val

        # Preparar datos
        self.ref_numerical = reference_data[numerical_features].values.astype(np.float32)
        self.ref_categorical = reference_data[categorical_features].values

        # Obtener categorias
        self.categories_per_feature = [
            reference_data[col].nunique()
            for col in categorical_features
        ]

        # Crear detectores
        self._create_detectors()

    def _create_detectors(self):
        """Crea detectores para numericas y categoricas."""
        # KS para numericas
        self.ks_detector = KSDrift(
            self.ref_numerical,
            p_val=self.p_val
        )

        # Chi-squared para categoricas (si hay)
        if len(self.categorical_features) > 0:
            # Codificar categoricas como enteros
            self.cat_encoders = {}
            ref_cat_encoded = np.zeros_like(self.ref_categorical, dtype=np.int64)

            for i, col in enumerate(self.categorical_features):
                unique_vals = self.reference_data[col].unique()
                self.cat_encoders[col] = {v: idx for idx, v in enumerate(unique_vals)}
                ref_cat_encoded[:, i] = [
                    self.cat_encoders[col].get(v, -1)
                    for v in self.ref_categorical[:, i]
                ]

            self.chi_detector = ChiSquareDrift(
                ref_cat_encoded,
                p_val=self.p_val,
                categories_per_feature=self.categories_per_feature
            )
        else:
            self.chi_detector = None

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta drift en todas las features.

        Args:
            current_data: DataFrame actual

        Returns:
            Dict con resultados detallados
        """
        results = {
            "numerical_drift": None,
            "categorical_drift": None,
            "summary": {
                "any_drift": False,
                "drifted_features": []
            }
        }

        # Drift en numericas
        cur_numerical = current_data[self.numerical_features].values.astype(np.float32)
        ks_result = self.ks_detector.predict(cur_numerical)

        results["numerical_drift"] = {
            "is_drift": bool(ks_result["data"]["is_drift"]),
            "p_val": float(ks_result["data"]["p_val"]),
            "feature_drift": {}
        }

        if hasattr(ks_result["data"], "p_val") and isinstance(ks_result["data"]["p_val"], (list, np.ndarray)):
            for i, feat in enumerate(self.numerical_features):
                p = ks_result["data"]["p_val"][i]
                is_drift = p < self.p_val
                results["numerical_drift"]["feature_drift"][feat] = {
                    "p_val": float(p),
                    "is_drift": is_drift
                }
                if is_drift:
                    results["summary"]["drifted_features"].append(feat)

        # Drift en categoricas
        if self.chi_detector is not None:
            cur_cat_encoded = np.zeros(
                (len(current_data), len(self.categorical_features)),
                dtype=np.int64
            )

            for i, col in enumerate(self.categorical_features):
                cur_cat_encoded[:, i] = [
                    self.cat_encoders[col].get(v, 0)  # 0 para categorias nuevas
                    for v in current_data[col].values
                ]

            chi_result = self.chi_detector.predict(cur_cat_encoded)

            results["categorical_drift"] = {
                "is_drift": bool(chi_result["data"]["is_drift"]),
                "p_val": float(chi_result["data"]["p_val"])
            }

        # Resumen
        results["summary"]["any_drift"] = (
            results["numerical_drift"]["is_drift"] or
            (results["categorical_drift"] is not None and results["categorical_drift"]["is_drift"])
        )

        return results
```

## 5. Sistema de Alertas y Dashboard

### Pipeline de Monitoreo

```python
"""
Sistema de monitoreo y alertas.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alertas."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alerta de drift."""
    timestamp: datetime
    alert_type: str
    severity: AlertSeverity
    feature: Optional[str]
    metric_value: float
    threshold: float
    message: str
    metadata: Optional[Dict[str, Any]] = None


class DriftMonitoringPipeline:
    """
    Pipeline de monitoreo continuo de drift.

    Componentes:
    - Detectores de drift
    - Sistema de alertas
    - Logging y metricas
    """

    def __init__(
        self,
        alert_handlers: Optional[List[Callable[[DriftAlert], None]]] = None
    ):
        """
        Args:
            alert_handlers: Funciones para manejar alertas
        """
        self.alert_handlers = alert_handlers or []
        self.alerts_history: List[DriftAlert] = []

        # Thresholds
        self.thresholds = {
            "data_drift_psi": 0.2,
            "data_drift_ks_pvalue": 0.01,
            "prediction_drift": 0.1,
            "performance_degradation": 0.05
        }

    def add_alert_handler(self, handler: Callable[[DriftAlert], None]) -> None:
        """Agrega handler de alertas."""
        self.alert_handlers.append(handler)

    def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        metric_value: float,
        threshold: float,
        message: str,
        feature: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> DriftAlert:
        """Crea y procesa una alerta."""
        alert = DriftAlert(
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            feature=feature,
            metric_value=metric_value,
            threshold=threshold,
            message=message,
            metadata=metadata
        )

        self.alerts_history.append(alert)

        # Notificar handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        return alert

    def check_data_drift(
        self,
        drift_results: Dict[str, Any]
    ) -> List[DriftAlert]:
        """
        Procesa resultados de drift y genera alertas.

        Args:
            drift_results: Resultados de detector de drift

        Returns:
            Lista de alertas generadas
        """
        alerts = []

        # Check PSI
        if "psi" in drift_results:
            psi_value = drift_results["psi"]
            if psi_value >= self.thresholds["data_drift_psi"]:
                severity = AlertSeverity.CRITICAL if psi_value >= 0.3 else AlertSeverity.WARNING

                alert = self.create_alert(
                    alert_type="data_drift_psi",
                    severity=severity,
                    metric_value=psi_value,
                    threshold=self.thresholds["data_drift_psi"],
                    message=f"Data drift detected: PSI = {psi_value:.3f}",
                    feature=drift_results.get("feature"),
                    metadata={"drift_type": "psi"}
                )
                alerts.append(alert)

        # Check KS
        if "ks_pvalue" in drift_results:
            p_value = drift_results["ks_pvalue"]
            if p_value < self.thresholds["data_drift_ks_pvalue"]:
                alert = self.create_alert(
                    alert_type="data_drift_ks",
                    severity=AlertSeverity.WARNING,
                    metric_value=p_value,
                    threshold=self.thresholds["data_drift_ks_pvalue"],
                    message=f"Data drift detected: KS p-value = {p_value:.4f}",
                    feature=drift_results.get("feature"),
                    metadata={"drift_type": "ks"}
                )
                alerts.append(alert)

        return alerts

    def check_performance_drift(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> List[DriftAlert]:
        """
        Verifica degradacion de performance.

        Args:
            current_metrics: Metricas actuales
            baseline_metrics: Metricas baseline

        Returns:
            Lista de alertas
        """
        alerts = []

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline_metrics:
                continue

            baseline_value = baseline_metrics[metric_name]
            degradation = baseline_value - current_value

            if degradation > self.thresholds["performance_degradation"]:
                severity = (
                    AlertSeverity.CRITICAL
                    if degradation > 0.1
                    else AlertSeverity.WARNING
                )

                alert = self.create_alert(
                    alert_type="performance_degradation",
                    severity=severity,
                    metric_value=degradation,
                    threshold=self.thresholds["performance_degradation"],
                    message=f"Performance degradation in {metric_name}: {degradation:.2%} drop",
                    metadata={
                        "metric": metric_name,
                        "current": current_value,
                        "baseline": baseline_value
                    }
                )
                alerts.append(alert)

        return alerts

    def get_alerts_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Resumen de alertas recientes.

        Args:
            hours: Horas hacia atras para incluir

        Returns:
            Dict con resumen
        """
        cutoff = datetime.now().timestamp() - (hours * 3600)

        recent_alerts = [
            a for a in self.alerts_history
            if a.timestamp.timestamp() > cutoff
        ]

        summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": {
                "critical": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in recent_alerts if a.severity == AlertSeverity.INFO])
            },
            "by_type": {},
            "recent_critical": []
        }

        for alert in recent_alerts:
            if alert.alert_type not in summary["by_type"]:
                summary["by_type"][alert.alert_type] = 0
            summary["by_type"][alert.alert_type] += 1

            if alert.severity == AlertSeverity.CRITICAL:
                summary["recent_critical"].append({
                    "timestamp": alert.timestamp.isoformat(),
                    "message": alert.message
                })

        return summary


# =============================================================================
# HANDLERS DE ALERTAS
# =============================================================================

def slack_alert_handler(alert: DriftAlert) -> None:
    """Envia alerta a Slack."""
    import requests

    webhook_url = "SLACK_WEBHOOK_URL"  # Desde env/secrets

    color = {
        AlertSeverity.CRITICAL: "danger",
        AlertSeverity.WARNING: "warning",
        AlertSeverity.INFO: "good"
    }[alert.severity]

    payload = {
        "attachments": [{
            "color": color,
            "title": f"ML Alert: {alert.alert_type}",
            "text": alert.message,
            "fields": [
                {"title": "Severity", "value": alert.severity.value, "short": True},
                {"title": "Metric", "value": f"{alert.metric_value:.4f}", "short": True},
                {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                {"title": "Feature", "value": alert.feature or "N/A", "short": True},
            ],
            "ts": alert.timestamp.timestamp()
        }]
    }

    # requests.post(webhook_url, json=payload)
    logger.info(f"Slack alert: {alert.message}")


def prometheus_metrics_handler(alert: DriftAlert) -> None:
    """Expone metricas para Prometheus."""
    from prometheus_client import Counter, Gauge

    # Counters
    drift_alerts_total = Counter(
        'ml_drift_alerts_total',
        'Total drift alerts',
        ['alert_type', 'severity']
    )

    drift_alerts_total.labels(
        alert_type=alert.alert_type,
        severity=alert.severity.value
    ).inc()


def log_alert_handler(alert: DriftAlert) -> None:
    """Log estructurado de alertas."""
    log_entry = {
        "timestamp": alert.timestamp.isoformat(),
        "alert_type": alert.alert_type,
        "severity": alert.severity.value,
        "feature": alert.feature,
        "metric_value": alert.metric_value,
        "threshold": alert.threshold,
        "message": alert.message
    }

    if alert.severity == AlertSeverity.CRITICAL:
        logger.critical(json.dumps(log_entry))
    elif alert.severity == AlertSeverity.WARNING:
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))
```

## 6. Resumen

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RESUMEN: Monitoreo y Drift Detection                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIPOS DE DRIFT                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Data Drift: Cambia P(X), relacion X→Y igual                             │
│  • Concept Drift: Cambia P(Y|X), la relacion cambia                        │
│  • Prediction Drift: Cambia distribucion de predicciones                   │
│  • Model Drift: Degradacion de metricas del modelo                         │
│                                                                             │
│  METODOS DE DETECCION                                                       │
│  ───────────────────────────────────────────────────────────────────────   │
│  • KS Test: Features numericas continuas                                   │
│  • Chi-Squared: Features categoricas                                       │
│  • PSI: Population Stability Index (industry standard)                     │
│  • MMD: Maximum Mean Discrepancy (multidimensional)                        │
│  • Classifier-based: Entrenar clasificador ref vs current                  │
│                                                                             │
│  HERRAMIENTAS                                                               │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Evidently: Reports HTML, test suites, facil de usar                     │
│  • Alibi Detect: Metodos avanzados, deep learning                          │
│  • WhyLabs: SaaS, monitoring completo                                      │
│  • Custom: scipy.stats + prometheus                                        │
│                                                                             │
│  PIPELINE DE MONITOREO                                                      │
│  ───────────────────────────────────────────────────────────────────────   │
│  1. Recolectar datos de produccion                                         │
│  2. Comparar con referencia (training data)                                │
│  3. Calcular metricas de drift                                             │
│  4. Generar alertas si excede thresholds                                   │
│  5. Trigger retraining si es necesario                                     │
│                                                                             │
│  THRESHOLDS COMUNES                                                         │
│  ───────────────────────────────────────────────────────────────────────   │
│  • PSI: < 0.1 OK, 0.1-0.2 warning, > 0.2 drift                            │
│  • KS p-value: > 0.05 OK, < 0.01 drift                                     │
│  • Performance: degradacion > 5% = warning                                  │
│                                                                             │
│  CIBERSEGURIDAD                                                             │
│  ───────────────────────────────────────────────────────────────────────   │
│  • Drift puede indicar nuevos ataques (concept drift)                      │
│  • O data poisoning (drift intencional)                                    │
│  • Retraining rapido es critico                                            │
│  • Monitorear: features de red, comportamiento, temporales                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** A/B Testing de Modelos - canary deployments, shadow mode, multi-armed bandits
