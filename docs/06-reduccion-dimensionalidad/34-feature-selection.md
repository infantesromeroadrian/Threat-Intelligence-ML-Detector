# Feature Selection: Métodos de Selección de Características

## Taxonomía de Métodos de Feature Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE SELECTION                                    │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│      FILTER         │      WRAPPER        │           EMBEDDED              │
│   (Independiente    │   (Usa modelo       │     (Durante el                 │
│    del modelo)      │    como caja negra) │      entrenamiento)             │
├─────────────────────┼─────────────────────┼─────────────────────────────────┤
│ • Correlación       │ • RFE               │ • Lasso (L1)                    │
│ • Mutual Info       │ • SFS/SBS           │ • Ridge (L2)                    │
│ • Chi-cuadrado      │ • Genetic Algorithm │ • ElasticNet                    │
│ • ANOVA F-test      │ • Exhaustive Search │ • Tree-based importance         │
│ • Variance Thresh.  │                     │ • Permutation importance        │
└─────────────────────┴─────────────────────┴─────────────────────────────────┘
          │                    │                         │
          ▼                    ▼                         ▼
       RÁPIDO              COSTOSO                  BALANCEADO
    Sin interacción      Considera interacción    Modelo-específico
```

## 1. Métodos Filter (Filtrado)

Los métodos filter evalúan cada feature independientemente usando estadísticas.

### 1.1 Variance Threshold

Elimina features con varianza menor a un umbral. Features constantes o casi constantes no aportan información.

```python
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, List

class VarianceSelector:
    """
    Selector de features basado en varianza.
    Útil como primer paso para eliminar features constantes.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: Varianza mínima requerida.
                       0.0 = elimina solo constantes
                       0.1 = elimina features con var < 0.1
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)

    def fit_transform(self, X: np.ndarray,
                      feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Aplica selección y retorna features seleccionadas.
        """
        X_selected = self.selector.fit_transform(X)

        # Máscaras de features seleccionadas
        mask = self.selector.get_support()

        if feature_names:
            selected_names = [f for f, m in zip(feature_names, mask) if m]
        else:
            selected_names = [f"feature_{i}" for i in range(X_selected.shape[1])]

        # Info de varianzas
        variances = self.selector.variances_

        print(f"Features originales: {X.shape[1]}")
        print(f"Features seleccionadas: {X_selected.shape[1]}")
        print(f"Features eliminadas: {X.shape[1] - X_selected.shape[1]}")

        return X_selected, selected_names

    def get_variance_report(self, X: np.ndarray,
                           feature_names: List[str]) -> None:
        """Muestra reporte de varianzas."""
        self.selector.fit(X)
        variances = self.selector.variances_

        # Ordenar por varianza
        sorted_idx = np.argsort(variances)

        print("\nVarianza por feature (ordenado):")
        print("-" * 50)
        for idx in sorted_idx[:10]:  # Top 10 menor varianza
            status = "❌" if variances[idx] < self.threshold else "✓"
            print(f"{status} {feature_names[idx]}: {variances[idx]:.6f}")


# Ejemplo: Dataset de malware con features binarias
np.random.seed(42)
n_samples = 1000
feature_names = [
    'api_createfile', 'api_writefile', 'api_deletefile',
    'api_regsetvalue', 'api_socket', 'api_connect',
    'has_section_execute', 'has_section_write',
    'is_packed',  # Casi siempre 0
    'is_signed',  # Siempre 0 en malware
    'entropy_high', 'imports_count_suspicious'
]

# Crear dataset con algunas features casi constantes
X = np.random.binomial(1, 0.5, (n_samples, len(feature_names))).astype(float)
X[:, 9] = 0  # is_signed siempre 0
X[:, 8] = np.random.binomial(1, 0.02, n_samples)  # is_packed casi siempre 0

selector = VarianceSelector(threshold=0.05)
selector.get_variance_report(X, feature_names)
X_selected, selected_names = selector.fit_transform(X, feature_names)
```

### 1.2 Correlación de Pearson (para regresión)

Mide relación lineal entre feature y target continuo.

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple

class CorrelationSelector:
    """
    Selección basada en correlación de Pearson.
    Para targets continuos (regresión).
    """

    def __init__(self,
                 threshold: float = 0.1,
                 remove_correlated: bool = True,
                 correlation_threshold: float = 0.9):
        """
        Args:
            threshold: Correlación mínima con target.
            remove_correlated: Si True, elimina features correlacionadas entre sí.
            correlation_threshold: Umbral para eliminar features redundantes.
        """
        self.threshold = threshold
        self.remove_correlated = remove_correlated
        self.correlation_threshold = correlation_threshold

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'CorrelationSelector':
        """
        Calcula correlaciones y selecciona features.
        """
        n_features = X.shape[1]

        # 1. Correlación con target
        self.correlations_with_target_ = np.array([
            stats.pearsonr(X[:, i], y)[0] for i in range(n_features)
        ])

        # 2. Máscara inicial: correlación > threshold
        self.selected_mask_ = np.abs(self.correlations_with_target_) >= self.threshold

        # 3. Eliminar features correlacionadas entre sí
        if self.remove_correlated:
            self._remove_redundant_features(X, feature_names)

        self.feature_names_ = feature_names
        return self

    def _remove_redundant_features(self, X: np.ndarray,
                                   feature_names: List[str]) -> None:
        """Elimina features altamente correlacionadas entre sí."""
        # Matriz de correlación solo de features seleccionadas
        selected_indices = np.where(self.selected_mask_)[0]

        if len(selected_indices) < 2:
            return

        X_selected = X[:, selected_indices]
        corr_matrix = np.corrcoef(X_selected.T)

        # Encontrar pares altamente correlacionados
        to_remove = set()
        n_selected = len(selected_indices)

        for i in range(n_selected):
            if i in to_remove:
                continue
            for j in range(i + 1, n_selected):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Eliminar la que tiene menor correlación con target
                    idx_i = selected_indices[i]
                    idx_j = selected_indices[j]

                    if abs(self.correlations_with_target_[idx_i]) < \
                       abs(self.correlations_with_target_[idx_j]):
                        to_remove.add(i)
                        print(f"Eliminando {feature_names[idx_i]} "
                              f"(correlacionada con {feature_names[idx_j]})")
                    else:
                        to_remove.add(j)
                        print(f"Eliminando {feature_names[idx_j]} "
                              f"(correlacionada con {feature_names[idx_i]})")

        # Actualizar máscara
        for i in to_remove:
            self.selected_mask_[selected_indices[i]] = False

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica selección."""
        return X[:, self.selected_mask_]

    def get_report(self) -> pd.DataFrame:
        """Genera reporte de correlaciones."""
        report = pd.DataFrame({
            'feature': self.feature_names_,
            'correlation_target': self.correlations_with_target_,
            'abs_correlation': np.abs(self.correlations_with_target_),
            'selected': self.selected_mask_
        })
        return report.sort_values('abs_correlation', ascending=False)


# Ejemplo: Predicción de tiempo de infección de malware
np.random.seed(42)
n_samples = 500

# Features de comportamiento de malware
file_ops = np.random.poisson(10, n_samples)  # Operaciones de archivo
network_conns = np.random.poisson(5, n_samples)  # Conexiones de red
registry_mods = np.random.poisson(8, n_samples)  # Modificaciones de registro
process_spawn = np.random.poisson(3, n_samples)  # Procesos creados
api_calls = file_ops * 5 + network_conns * 3  # Correlacionada con otras

# Target: tiempo hasta detección (minutos)
y = 100 - 5 * file_ops - 10 * network_conns + np.random.normal(0, 10, n_samples)

X = np.column_stack([file_ops, network_conns, registry_mods, process_spawn, api_calls])
feature_names = ['file_ops', 'network_conns', 'registry_mods', 'process_spawn', 'api_calls']

selector = CorrelationSelector(threshold=0.2, remove_correlated=True)
selector.fit(X, y, feature_names)
print("\nReporte de correlaciones:")
print(selector.get_report())
```

### 1.3 Mutual Information (para cualquier relación)

Captura relaciones no lineales entre features y target.

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest
import numpy as np
from typing import List, Tuple

class MutualInfoSelector:
    """
    Selección basada en Mutual Information.
    Captura relaciones no lineales.

    MI = 0: Features independientes
    MI alto: Features dependientes (lineal o no lineal)
    """

    def __init__(self,
                 task: str = 'classification',
                 k: int = 10,
                 n_neighbors: int = 3,
                 random_state: int = 42):
        """
        Args:
            task: 'classification' o 'regression'
            k: Número de features a seleccionar
            n_neighbors: Vecinos para estimación de MI (más = más preciso, más lento)
        """
        self.task = task
        self.k = k
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        score_func = mutual_info_classif if task == 'classification' \
                     else mutual_info_regression

        self.selector = SelectKBest(
            score_func=lambda X, y: score_func(
                X, y,
                n_neighbors=n_neighbors,
                random_state=random_state
            ),
            k=k
        )

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Ajusta y transforma, retornando features seleccionadas.
        """
        X_selected = self.selector.fit_transform(X, y)

        self.scores_ = self.selector.scores_
        self.selected_mask_ = self.selector.get_support()

        if feature_names:
            self.feature_names_ = feature_names
            selected_names = [f for f, m in zip(feature_names, self.selected_mask_) if m]
        else:
            selected_names = [f"feature_{i}" for i, m in enumerate(self.selected_mask_) if m]

        return X_selected, selected_names

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Retorna ranking de features por MI score."""
        if not hasattr(self, 'feature_names_'):
            raise ValueError("Primero llama a fit_transform con feature_names")

        ranking = list(zip(self.feature_names_, self.scores_))
        return sorted(ranking, key=lambda x: x[1], reverse=True)


# Ejemplo: Clasificación de familias de malware
from sklearn.datasets import make_classification

# Dataset con relaciones no lineales
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=5,
    n_clusters_per_class=3,
    random_state=42
)

feature_names = [f"behavior_{i}" for i in range(20)]

mi_selector = MutualInfoSelector(task='classification', k=8)
X_selected, selected_names = mi_selector.fit_transform(X, y, feature_names)

print("Ranking por Mutual Information:")
for name, score in mi_selector.get_ranking():
    selected = "✓" if name in selected_names else ""
    print(f"  {name}: {score:.4f} {selected}")
```

### 1.4 Chi-Cuadrado (para clasificación con features no negativas)

Ideal para features categóricas o counts.

```python
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np
from typing import List

class Chi2Selector:
    """
    Selección basada en test Chi-cuadrado.

    IMPORTANTE: Solo para features no negativas (counts, binarias).
    Test de independencia entre feature y clase.
    """

    def __init__(self, k: int = 10, alpha: float = 0.05):
        """
        Args:
            k: Número de features a seleccionar
            alpha: Nivel de significancia para p-value
        """
        self.k = k
        self.alpha = alpha
        self.selector = SelectKBest(score_func=chi2, k=k)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'Chi2Selector':
        """
        Ajusta el selector.

        Args:
            X: Features (deben ser no negativas)
            y: Labels
        """
        # Verificar que no hay valores negativos
        if np.any(X < 0):
            raise ValueError("Chi-cuadrado requiere features no negativas. "
                           "Considera usar MinMaxScaler primero.")

        self.selector.fit(X, y)
        self.chi2_scores_ = self.selector.scores_
        self.p_values_ = self.selector.pvalues_
        self.feature_names_ = feature_names
        self.selected_mask_ = self.selector.get_support()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)

    def get_report(self):
        """Genera reporte con chi2 scores y p-values."""
        import pandas as pd

        report = pd.DataFrame({
            'feature': self.feature_names_,
            'chi2_score': self.chi2_scores_,
            'p_value': self.p_values_,
            'significant': self.p_values_ < self.alpha,
            'selected': self.selected_mask_
        })
        return report.sort_values('chi2_score', ascending=False)


# Ejemplo: Features de malware basadas en API calls (counts)
np.random.seed(42)
n_samples = 500

# Simular counts de API calls por categoría
api_categories = {
    'file_apis': np.random.poisson(15, n_samples),      # Informativa
    'network_apis': np.random.poisson(8, n_samples),     # Informativa
    'registry_apis': np.random.poisson(10, n_samples),   # Informativa
    'process_apis': np.random.poisson(5, n_samples),     # Algo informativa
    'memory_apis': np.random.poisson(20, n_samples),     # Ruido
    'gui_apis': np.random.poisson(3, n_samples),         # Ruido
    'crypto_apis': np.random.poisson(2, n_samples),      # Muy informativa
    'debug_apis': np.random.poisson(1, n_samples),       # Ruido
}

X = np.column_stack(list(api_categories.values()))
feature_names = list(api_categories.keys())

# Labels: 0=benigno, 1=malware (correlacionado con algunas features)
y = (api_categories['file_apis'] > 18) | \
    (api_categories['crypto_apis'] > 3) | \
    (api_categories['network_apis'] > 10)
y = y.astype(int)

chi2_selector = Chi2Selector(k=4)
chi2_selector.fit(X, y, feature_names)
print("Reporte Chi-cuadrado:")
print(chi2_selector.get_report())
```

### 1.5 ANOVA F-test (para clasificación)

Compara medias entre grupos. Asume distribución normal.

```python
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np

class ANOVASelector:
    """
    Selección basada en ANOVA F-test.

    Compara varianza entre grupos vs dentro de grupos.
    F alto = diferencias significativas entre clases.

    Asumpciones:
    - Features continuas
    - Distribución aproximadamente normal
    - Homogeneidad de varianzas
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=k)

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list = None):
        X_selected = self.selector.fit_transform(X, y)

        self.f_scores_ = self.selector.scores_
        self.p_values_ = self.selector.pvalues_
        self.selected_mask_ = self.selector.get_support()

        if feature_names:
            selected_names = [f for f, m in zip(feature_names, self.selected_mask_) if m]
        else:
            selected_names = list(range(X_selected.shape[1]))

        return X_selected, selected_names


# Ejemplo rápido
from sklearn.datasets import load_iris

iris = load_iris()
anova = ANOVASelector(k=2)
X_selected, selected_names = anova.fit_transform(
    iris.data, iris.target,
    feature_names=iris.feature_names
)
print(f"Features seleccionadas: {selected_names}")
print(f"F-scores: {dict(zip(iris.feature_names, anova.f_scores_))}")
```

---

## 2. Métodos Wrapper

Los wrappers usan un modelo como "caja negra" para evaluar subconjuntos de features.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           WRAPPER METHODS                                 │
│                                                                          │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│    │  Subset de  │────▶│   Entrenar  │────▶│  Evaluar    │              │
│    │  Features   │     │   Modelo    │     │  Métricas   │              │
│    └─────────────┘     └─────────────┘     └─────────────┘              │
│           ▲                                       │                      │
│           │                                       │                      │
│           └───────────────────────────────────────┘                      │
│                     Iterar hasta convergencia                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Recursive Feature Elimination (RFE)

Elimina features iterativamente basándose en importancia del modelo.

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import List, Tuple

class RFESelector:
    """
    Recursive Feature Elimination.

    Proceso:
    1. Entrenar modelo con todas las features
    2. Obtener importancias/coeficientes
    3. Eliminar feature menos importante
    4. Repetir hasta n_features_to_select

    Con CV (RFECV): Encuentra automáticamente el número óptimo.
    """

    def __init__(self,
                 estimator=None,
                 n_features_to_select: int = None,
                 use_cv: bool = True,
                 cv: int = 5,
                 scoring: str = 'accuracy'):
        """
        Args:
            estimator: Modelo base (debe tener coef_ o feature_importances_)
            n_features_to_select: Número de features (None con CV = auto)
            use_cv: Si True, usa cross-validation para encontrar óptimo
            cv: Número de folds
            scoring: Métrica para CV
        """
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.n_features = n_features_to_select
        self.use_cv = use_cv
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'RFESelector':
        """
        Ejecuta RFE.
        """
        self.feature_names_ = feature_names

        if self.use_cv:
            self.selector_ = RFECV(
                estimator=self.estimator,
                step=1,
                cv=StratifiedKFold(self.cv),
                scoring=self.scoring,
                n_jobs=-1
            )
        else:
            self.selector_ = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features,
                step=1
            )

        self.selector_.fit(X, y)

        self.selected_mask_ = self.selector_.support_
        self.ranking_ = self.selector_.ranking_

        if self.use_cv:
            self.n_features_optimal_ = self.selector_.n_features_
            self.cv_results_ = self.selector_.cv_results_

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector_.transform(X)

    def get_selected_features(self) -> List[str]:
        """Retorna nombres de features seleccionadas."""
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]

    def get_ranking(self) -> List[Tuple[str, int]]:
        """
        Retorna ranking de features.
        1 = seleccionada, 2 = segunda en eliminarse, etc.
        """
        ranking = list(zip(self.feature_names_, self.ranking_))
        return sorted(ranking, key=lambda x: x[1])

    def plot_cv_results(self):
        """Visualiza resultados de CV si use_cv=True."""
        if not self.use_cv:
            raise ValueError("CV no fue usado")

        import matplotlib.pyplot as plt

        n_features_range = range(1, len(self.feature_names_) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(n_features_range, self.cv_results_['mean_test_score'], 'b-')
        plt.fill_between(
            n_features_range,
            self.cv_results_['mean_test_score'] - self.cv_results_['std_test_score'],
            self.cv_results_['mean_test_score'] + self.cv_results_['std_test_score'],
            alpha=0.2
        )
        plt.axvline(x=self.n_features_optimal_, color='r', linestyle='--',
                   label=f'Óptimo: {self.n_features_optimal_} features')
        plt.xlabel('Número de features')
        plt.ylabel(f'Score ({self.scoring})')
        plt.title('RFECV: Score vs Número de Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt


# Ejemplo: Detección de malware
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=25,
    n_informative=8,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

feature_names = [
    'entropy', 'file_size', 'num_sections', 'imports_count', 'exports_count',
    'has_debug', 'has_tls', 'has_resources', 'section_entropy_mean', 'section_entropy_max',
    'api_file_count', 'api_network_count', 'api_registry_count', 'api_process_count',
    'string_count', 'url_count', 'ip_count', 'suspicious_strings',
    'packed_score', 'obfuscation_score', 'code_section_size', 'data_section_size',
    'timestamp_valid', 'checksum_valid', 'certificate_present'
]

rfe = RFESelector(use_cv=True, scoring='f1')
rfe.fit(X, y, feature_names)

print(f"Número óptimo de features: {rfe.n_features_optimal_}")
print(f"\nFeatures seleccionadas: {rfe.get_selected_features()}")
print(f"\nRanking completo:")
for name, rank in rfe.get_ranking():
    status = "✓" if rank == 1 else ""
    print(f"  {rank}. {name} {status}")
```

### 2.2 Sequential Feature Selection (SFS/SBS)

Añade o elimina features una a una de forma greedy.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from typing import List

class SequentialSelector:
    """
    Sequential Feature Selection.

    Forward (SFS): Empieza vacío, añade la mejor feature una a una.
    Backward (SBS): Empieza con todas, elimina la peor una a una.

    Ventaja vs RFE: Considera interacciones entre features.
    Desventaja: Más costoso computacionalmente.
    """

    def __init__(self,
                 estimator=None,
                 direction: str = 'forward',
                 n_features_to_select: int = 'auto',
                 cv: int = 5,
                 scoring: str = 'accuracy'):
        """
        Args:
            direction: 'forward' (SFS) o 'backward' (SBS)
            n_features_to_select: Número, 'auto' (mitad), o float (proporción)
        """
        self.estimator = estimator or LogisticRegression(max_iter=1000)
        self.direction = direction
        self.n_features = n_features_to_select
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'SequentialSelector':
        """Ejecuta selección secuencial."""

        self.feature_names_ = feature_names

        self.selector_ = SequentialFeatureSelector(
            self.estimator,
            n_features_to_select=self.n_features,
            direction=self.direction,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1
        )

        self.selector_.fit(X, y)
        self.selected_mask_ = self.selector_.get_support()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector_.transform(X)

    def get_selected_features(self) -> List[str]:
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]


# Comparar Forward vs Backward
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=15,
                          n_informative=5, random_state=42)
feature_names = [f"feat_{i}" for i in range(15)]

print("=== Forward Selection (SFS) ===")
sfs = SequentialSelector(direction='forward', n_features_to_select=5)
sfs.fit(X, y, feature_names)
print(f"Seleccionadas: {sfs.get_selected_features()}")

print("\n=== Backward Selection (SBS) ===")
sbs = SequentialSelector(direction='backward', n_features_to_select=5)
sbs.fit(X, y, feature_names)
print(f"Seleccionadas: {sbs.get_selected_features()}")
```

### 2.3 Búsqueda con Algoritmos Genéticos

Para espacios de búsqueda muy grandes donde métodos secuenciales son insuficientes.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Callable
import random

class GeneticFeatureSelector:
    """
    Selección de features usando Algoritmo Genético.

    Cromosoma = máscara binaria de features [1,0,1,1,0,...]
    Fitness = score de CV del modelo con esas features

    Útil cuando:
    - Muchas features (>50)
    - Interacciones complejas
    - Métodos secuenciales son muy lentos
    """

    def __init__(self,
                 estimator=None,
                 population_size: int = 50,
                 generations: int = 30,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 5,
                 cv: int = 3,
                 scoring: str = 'accuracy',
                 min_features: int = 1,
                 random_state: int = 42):
        """
        Args:
            population_size: Número de individuos por generación
            generations: Número de generaciones
            crossover_rate: Probabilidad de crossover
            mutation_rate: Probabilidad de mutación por gen
            tournament_size: Tamaño del torneo para selección
            min_features: Mínimo de features activas
        """
        self.estimator = estimator or RandomForestClassifier(n_estimators=50, n_jobs=-1)
        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.cv = cv
        self.scoring = scoring
        self.min_features = min_features
        self.random_state = random_state

    def _init_population(self, n_features: int) -> np.ndarray:
        """Inicializa población aleatoria."""
        population = np.random.randint(0, 2, (self.pop_size, n_features))

        # Asegurar mínimo de features activas
        for i in range(self.pop_size):
            if population[i].sum() < self.min_features:
                indices = np.random.choice(n_features, self.min_features, replace=False)
                population[i, indices] = 1

        return population

    def _fitness(self, chromosome: np.ndarray, X: np.ndarray,
                 y: np.ndarray) -> float:
        """Evalúa fitness de un cromosoma."""
        if chromosome.sum() == 0:
            return 0.0

        X_selected = X[:, chromosome.astype(bool)]

        try:
            scores = cross_val_score(
                self.estimator, X_selected, y,
                cv=self.cv, scoring=self.scoring, n_jobs=-1
            )
            # Penalizar ligeramente por número de features (parsimonia)
            penalty = 0.001 * chromosome.sum() / len(chromosome)
            return scores.mean() - penalty
        except:
            return 0.0

    def _tournament_selection(self, population: np.ndarray,
                             fitness_scores: np.ndarray) -> np.ndarray:
        """Selección por torneo."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        winner_idx = indices[np.argmax(fitness_scores[indices])]
        return population[winner_idx].copy()

    def _crossover(self, parent1: np.ndarray,
                   parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover de un punto."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1, child2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutación bit a bit."""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]

        # Asegurar mínimo de features
        if mutated.sum() < self.min_features:
            zero_indices = np.where(mutated == 0)[0]
            to_activate = np.random.choice(
                zero_indices,
                self.min_features - int(mutated.sum()),
                replace=False
            )
            mutated[to_activate] = 1

        return mutated

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'GeneticFeatureSelector':
        """
        Ejecuta el algoritmo genético.
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        n_features = X.shape[1]
        self.feature_names_ = feature_names

        # Inicializar
        population = self._init_population(n_features)

        self.history_ = []
        self.best_chromosome_ = None
        self.best_fitness_ = -np.inf

        print(f"Iniciando GA: {self.pop_size} individuos, {self.generations} generaciones")

        for gen in range(self.generations):
            # Evaluar fitness
            fitness_scores = np.array([
                self._fitness(chrom, X, y) for chrom in population
            ])

            # Tracking
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_mean_fitness = fitness_scores.mean()

            if gen_best_fitness > self.best_fitness_:
                self.best_fitness_ = gen_best_fitness
                self.best_chromosome_ = population[gen_best_idx].copy()

            self.history_.append({
                'generation': gen,
                'best_fitness': gen_best_fitness,
                'mean_fitness': gen_mean_fitness,
                'best_n_features': population[gen_best_idx].sum()
            })

            if gen % 5 == 0:
                print(f"Gen {gen}: Best={gen_best_fitness:.4f}, "
                      f"Mean={gen_mean_fitness:.4f}, "
                      f"Features={population[gen_best_idx].sum()}")

            # Nueva generación
            new_population = []

            # Elitismo: mantener el mejor
            new_population.append(population[gen_best_idx])

            while len(new_population) < self.pop_size:
                # Selección
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutación
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = np.array(new_population[:self.pop_size])

        self.selected_mask_ = self.best_chromosome_.astype(bool)
        print(f"\nMejor solución: {self.best_fitness_:.4f} con "
              f"{self.best_chromosome_.sum()} features")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_mask_]

    def get_selected_features(self) -> List[str]:
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]


# Ejemplo con dataset grande
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=400,
    n_features=50,
    n_informative=10,
    n_redundant=10,
    random_state=42
)
feature_names = [f"feature_{i}" for i in range(50)]

ga_selector = GeneticFeatureSelector(
    population_size=30,
    generations=20,
    mutation_rate=0.05,
    random_state=42
)
ga_selector.fit(X, y, feature_names)

print(f"\nFeatures seleccionadas ({len(ga_selector.get_selected_features())}):")
print(ga_selector.get_selected_features())
```

---

## 3. Métodos Embedded

Los métodos embedded realizan selección durante el entrenamiento del modelo.

### 3.1 Regularización L1 (Lasso)

L1 produce coeficientes exactamente cero, eliminando features.

```python
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List

class LassoSelector:
    """
    Feature Selection con regularización L1 (Lasso).

    L1 penalty = λ * Σ|w_i|

    Efecto: Produce coeficientes exactamente 0 para features no informativas.

    Para regresión: Lasso
    Para clasificación: LogisticRegression con penalty='l1'
    """

    def __init__(self,
                 task: str = 'classification',
                 alpha: float = None,  # None = usar CV
                 cv: int = 5,
                 max_iter: int = 10000):
        """
        Args:
            task: 'classification' o 'regression'
            alpha: Fuerza de regularización (None = encontrar con CV)
        """
        self.task = task
        self.alpha = alpha
        self.cv = cv
        self.max_iter = max_iter
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'LassoSelector':
        """
        Ajusta el modelo Lasso/LogReg con L1.
        """
        self.feature_names_ = feature_names

        # Estandarizar (importante para Lasso)
        X_scaled = self.scaler.fit_transform(X)

        if self.task == 'regression':
            if self.alpha is None:
                self.model_ = LassoCV(cv=self.cv, max_iter=self.max_iter)
            else:
                self.model_ = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        else:  # classification
            if self.alpha is None:
                # Buscar mejor C (C = 1/alpha)
                from sklearn.model_selection import GridSearchCV
                base_model = LogisticRegression(
                    penalty='l1', solver='saga', max_iter=self.max_iter
                )
                param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
                self.model_ = GridSearchCV(base_model, param_grid, cv=self.cv)
            else:
                self.model_ = LogisticRegression(
                    penalty='l1', solver='saga',
                    C=1/self.alpha, max_iter=self.max_iter
                )

        self.model_.fit(X_scaled, y)

        # Obtener coeficientes
        if hasattr(self.model_, 'best_estimator_'):
            self.coef_ = self.model_.best_estimator_.coef_.ravel()
        else:
            self.coef_ = self.model_.coef_.ravel()

        self.selected_mask_ = self.coef_ != 0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_mask_]

    def get_selected_features(self) -> List[str]:
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]

    def get_coefficients(self) -> List[tuple]:
        """Retorna features con sus coeficientes, ordenadas por importancia."""
        coef_pairs = list(zip(self.feature_names_, self.coef_))
        return sorted(coef_pairs, key=lambda x: abs(x[1]), reverse=True)


# Ejemplo: Selección para clasificación de malware
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500, n_features=30,
    n_informative=8, n_redundant=5,
    random_state=42
)
feature_names = [f"feat_{i}" for i in range(30)]

lasso = LassoSelector(task='classification')
lasso.fit(X, y, feature_names)

print(f"Features seleccionadas: {len(lasso.get_selected_features())} de {len(feature_names)}")
print(f"\nTop 10 por coeficiente:")
for name, coef in lasso.get_coefficients()[:10]:
    status = "✓" if coef != 0 else "✗"
    print(f"  {status} {name}: {coef:.4f}")
```

### 3.2 ElasticNet (L1 + L2)

Combina L1 (sparsity) y L2 (estabilidad).

```python
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class ElasticNetSelector:
    """
    Feature Selection con ElasticNet (L1 + L2).

    Penalty = α * λ * Σ|w_i| + (1-α) * λ * Σw_i²

    α (l1_ratio): Balance entre L1 y L2
        - α=1: Puro Lasso (máxima sparsity)
        - α=0: Puro Ridge (sin selección)
        - α=0.5: Balanceado

    Ventajas sobre Lasso puro:
    - Más estable con features correlacionadas
    - Selecciona grupos de features relacionadas
    """

    def __init__(self,
                 task: str = 'regression',
                 l1_ratio: float = 0.5,
                 cv: int = 5):
        self.task = task
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list) -> 'ElasticNetSelector':

        self.feature_names_ = feature_names
        X_scaled = self.scaler.fit_transform(X)

        if self.task == 'regression':
            self.model_ = ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv,
                max_iter=10000
            )
        else:
            # Para clasificación: SGDClassifier con elasticnet penalty
            self.model_ = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                l1_ratio=self.l1_ratio,
                max_iter=10000,
                random_state=42
            )

        self.model_.fit(X_scaled, y)
        self.coef_ = self.model_.coef_.ravel()
        self.selected_mask_ = self.coef_ != 0

        return self

    def get_selected_features(self):
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]


# Comparar diferentes l1_ratios
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=20,
                       n_informative=5, noise=10, random_state=42)
feature_names = [f"f{i}" for i in range(20)]

print("Efecto de l1_ratio en selección:")
print("-" * 50)

for l1_ratio in [0.1, 0.5, 0.9, 1.0]:
    en = ElasticNetSelector(task='regression', l1_ratio=l1_ratio)
    en.fit(X, y, feature_names)
    n_selected = len(en.get_selected_features())
    print(f"l1_ratio={l1_ratio}: {n_selected} features seleccionadas")
```

### 3.3 Tree-Based Feature Importance

Random Forest y Gradient Boosting proporcionan importancias de features.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import numpy as np
from typing import List, Tuple

class TreeBasedSelector:
    """
    Feature Selection basada en importancia de árboles.

    Tipos de importancia:
    1. MDI (Mean Decrease Impurity): Reducción de impureza en splits.
       - Rápido pero sesgado hacia features de alta cardinalidad

    2. Permutation Importance: Caída de score al permutar feature.
       - Más robusto pero más lento
       - Captura importancia real para predicción
    """

    def __init__(self,
                 estimator=None,
                 method: str = 'both',  # 'mdi', 'permutation', 'both'
                 n_estimators: int = 100,
                 threshold: float = None,  # None = usar media
                 cv: int = 5):
        """
        Args:
            method: 'mdi', 'permutation', o 'both'
            threshold: Umbral de importancia para selección
        """
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        self.method = method
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> 'TreeBasedSelector':
        """
        Calcula importancias y selecciona features.
        """
        self.feature_names_ = feature_names

        # Entrenar modelo
        self.estimator.fit(X, y)

        # MDI (built-in)
        self.mdi_importance_ = self.estimator.feature_importances_

        # Permutation Importance (más costoso)
        if self.method in ['permutation', 'both']:
            perm_result = permutation_importance(
                self.estimator, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            self.perm_importance_ = perm_result.importances_mean
            self.perm_importance_std_ = perm_result.importances_std

        # Seleccionar features
        if self.method == 'mdi':
            importance = self.mdi_importance_
        elif self.method == 'permutation':
            importance = self.perm_importance_
        else:  # both - usar promedio normalizado
            mdi_norm = self.mdi_importance_ / self.mdi_importance_.max()
            perm_norm = self.perm_importance_ / (self.perm_importance_.max() + 1e-10)
            importance = (mdi_norm + perm_norm) / 2

        self.importance_ = importance

        # Umbral: media si no se especifica
        threshold = self.threshold or np.mean(importance)
        self.selected_mask_ = importance >= threshold

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_mask_]

    def get_importance_report(self) -> List[Tuple[str, dict]]:
        """Genera reporte detallado de importancias."""
        report = []
        for i, name in enumerate(self.feature_names_):
            entry = {
                'feature': name,
                'mdi': self.mdi_importance_[i],
                'selected': self.selected_mask_[i]
            }
            if hasattr(self, 'perm_importance_'):
                entry['permutation'] = self.perm_importance_[i]
                entry['perm_std'] = self.perm_importance_std_[i]
            if hasattr(self, 'importance_'):
                entry['combined'] = self.importance_[i]

            report.append(entry)

        return sorted(report, key=lambda x: x.get('combined', x['mdi']), reverse=True)

    def plot_importances(self, top_n: int = 15):
        """Visualiza importancias."""
        import matplotlib.pyplot as plt

        report = self.get_importance_report()[:top_n]

        names = [r['feature'] for r in report]
        mdi = [r['mdi'] for r in report]

        fig, axes = plt.subplots(1, 2 if hasattr(self, 'perm_importance_') else 1,
                                figsize=(12, 6))

        if not hasattr(self, 'perm_importance_'):
            axes = [axes]

        # MDI
        colors = ['green' if r['selected'] else 'red' for r in report]
        axes[0].barh(names, mdi, color=colors, alpha=0.7)
        axes[0].set_xlabel('MDI Importance')
        axes[0].set_title('Mean Decrease Impurity')
        axes[0].invert_yaxis()

        # Permutation
        if hasattr(self, 'perm_importance_'):
            perm = [r['permutation'] for r in report]
            perm_std = [r['perm_std'] for r in report]
            axes[1].barh(names, perm, xerr=perm_std, color=colors, alpha=0.7)
            axes[1].set_xlabel('Permutation Importance')
            axes[1].set_title('Permutation Importance')
            axes[1].invert_yaxis()

        plt.tight_layout()
        return plt


# Ejemplo: Features de ejecutables PE
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=8, n_redundant=4, n_clusters_per_class=3,
    random_state=42
)

feature_names = [
    'entropy', 'file_size', 'num_sections', 'imports_count',
    'exports_count', 'has_debug', 'has_tls', 'has_resources',
    'virtual_size', 'raw_size', 'section_align', 'file_align',
    'subsystem', 'dll_chars', 'timestamp', 'checksum',
    'code_size', 'data_size', 'header_size', 'image_base'
]

tree_selector = TreeBasedSelector(method='both')
tree_selector.fit(X, y, feature_names)

print("Importancia de Features (MDI + Permutation):")
print("-" * 60)
for entry in tree_selector.get_importance_report()[:10]:
    status = "✓" if entry['selected'] else ""
    print(f"{status:2} {entry['feature']:20} MDI={entry['mdi']:.4f} "
          f"Perm={entry.get('permutation', 0):.4f}")
```

---

## 4. Comparación de Métodos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPARACIÓN DE MÉTODOS                                   │
├──────────────┬───────────────┬──────────────────┬───────────────────────────┤
│   MÉTODO     │  VELOCIDAD    │   CONSIDERA      │      MEJOR PARA           │
│              │               │  INTERACCIONES   │                           │
├──────────────┼───────────────┼──────────────────┼───────────────────────────┤
│ Variance     │  ★★★★★        │       No         │ Eliminar constantes       │
│ Correlation  │  ★★★★★        │       No         │ Regresión lineal          │
│ Chi-cuadrado │  ★★★★★        │       No         │ Features categóricas      │
│ Mutual Info  │  ★★★★☆        │  Parcialmente    │ Relaciones no lineales    │
│ ANOVA        │  ★★★★★        │       No         │ Clasificación simple      │
├──────────────┼───────────────┼──────────────────┼───────────────────────────┤
│ RFE          │  ★★★☆☆        │       Sí         │ Features ordenadas        │
│ SFS/SBS      │  ★★☆☆☆        │       Sí         │ Subconjuntos óptimos      │
│ Genetic      │  ★☆☆☆☆        │       Sí         │ Búsqueda global           │
├──────────────┼───────────────┼──────────────────┼───────────────────────────┤
│ Lasso        │  ★★★★☆        │ Durante training │ Modelos lineales          │
│ ElasticNet   │  ★★★★☆        │ Durante training │ Features correlacionadas  │
│ Tree Impor.  │  ★★★★☆        │ Durante training │ Modelos de árbol          │
└──────────────┴───────────────┴──────────────────┴───────────────────────────┘
```

### Pipeline Recomendado

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def create_feature_selection_pipeline(n_features_final: int = 20):
    """
    Pipeline de selección de features en 3 etapas:
    1. Eliminar constantes (Filter)
    2. Selección basada en importancia de RF (Embedded)
    3. Modelo final
    """
    return Pipeline([
        # Etapa 1: Eliminar features constantes
        ('variance', VarianceThreshold(threshold=0.01)),

        # Etapa 2: Escalar
        ('scaler', StandardScaler()),

        # Etapa 3: Selección por importancia
        ('selector', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=n_features_final,
            threshold=-np.inf  # Seleccionar top n_features
        )),

        # Etapa 4: Modelo final
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])


# Uso
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=100,
                          n_informative=15, random_state=42)

pipeline = create_feature_selection_pipeline(n_features_final=15)
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
print(f"F1-Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

---

## 5. Feature Selection en Ciberseguridad

### 5.1 Selección para Detección de Malware

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict

class MalwareFeatureSelector:
    """
    Feature Selection especializada para detección de malware.

    Considera:
    - Features estáticas (PE headers, imports, strings)
    - Features dinámicas (API calls, network, registry)
    - Balance entre precisión y costo de extracción
    """

    def __init__(self, max_features: int = 50):
        self.max_features = max_features

        # Costo de extracción de features (0-1)
        # Features estáticas son baratas, dinámicas son caras
        self.feature_costs = {}

    def set_feature_costs(self, costs: Dict[str, float]):
        """Define costo de extracción por feature."""
        self.feature_costs = costs

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str],
            cost_weight: float = 0.1) -> 'MalwareFeatureSelector':
        """
        Selecciona features considerando importancia y costo.

        Args:
            cost_weight: Peso del costo en la selección (0=ignorar, 1=priorizar)
        """
        self.feature_names_ = feature_names

        # 1. Random Forest para importancia base
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = rf.feature_importances_

        # 2. Ajustar por costo de extracción
        adjusted_importance = importance.copy()
        for i, name in enumerate(feature_names):
            cost = self.feature_costs.get(name, 0.5)  # Default: costo medio
            # Penalizar features caras
            adjusted_importance[i] = importance[i] * (1 - cost_weight * cost)

        # 3. Seleccionar top features
        top_indices = np.argsort(adjusted_importance)[-self.max_features:]

        self.selected_mask_ = np.zeros(len(feature_names), dtype=bool)
        self.selected_mask_[top_indices] = True

        self.importance_ = importance
        self.adjusted_importance_ = adjusted_importance

        return self

    def get_selected_features(self) -> List[str]:
        return [f for f, m in zip(self.feature_names_, self.selected_mask_) if m]

    def get_report(self):
        """Genera reporte con análisis de costo-beneficio."""
        import pandas as pd

        costs = [self.feature_costs.get(f, 0.5) for f in self.feature_names_]

        report = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.importance_,
            'cost': costs,
            'adjusted_importance': self.adjusted_importance_,
            'selected': self.selected_mask_
        })

        return report.sort_values('adjusted_importance', ascending=False)


# Ejemplo de uso
np.random.seed(42)

# Simular dataset de malware
n_samples = 1000
n_features = 40

feature_names = [
    # Features estáticas (baratas)
    'entropy', 'file_size', 'num_sections', 'imports_count', 'exports_count',
    'virtual_size', 'code_size', 'header_size', 'section_entropy_max',
    'pe_timestamp', 'pe_checksum', 'subsystem', 'dll_characteristics',
    'has_debug', 'has_tls', 'has_certificate', 'resource_count',
    'import_dlls_count', 'string_count', 'suspicious_strings',
    # Features dinámicas (caras - requieren sandbox)
    'api_file_create', 'api_file_write', 'api_file_delete',
    'api_registry_set', 'api_registry_delete', 'api_network_connect',
    'api_network_send', 'api_process_create', 'api_process_inject',
    'api_memory_allocate', 'dns_queries', 'http_requests',
    'mutex_created', 'service_created', 'driver_loaded',
    'crypto_api_calls', 'evasion_techniques', 'persistence_methods',
    'sandbox_detection', 'runtime_seconds'
]

# Definir costos
feature_costs = {
    # Estáticas: bajo costo
    **{f: 0.1 for f in feature_names[:20]},
    # Dinámicas: alto costo
    **{f: 0.8 for f in feature_names[20:]}
}

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Seleccionar
selector = MalwareFeatureSelector(max_features=15)
selector.set_feature_costs(feature_costs)
selector.fit(X, y, feature_names, cost_weight=0.3)

print("Features seleccionadas (considerando costo):")
print("-" * 50)
report = selector.get_report()
print(report[report['selected']][['feature', 'importance', 'cost', 'selected']])
```

---

## 6. Mejores Prácticas

### 6.1 Evitar Data Leakage

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# ❌ INCORRECTO: Feature selection antes del split
# selector = SelectKBest(k=10).fit(X, y)  # Usa todo el dataset
# X_selected = selector.transform(X)
# scores = cross_val_score(model, X_selected, y, cv=5)  # Data leakage!

# ✓ CORRECTO: Feature selection dentro del pipeline
pipeline = Pipeline([
    ('selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', RandomForestClassifier())
])

# La selección se hace en cada fold, usando solo datos de train
scores = cross_val_score(pipeline, X, y, cv=5)
```

### 6.2 Combinación de Métodos

```python
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def robust_feature_selection(X, y, feature_names, n_select=20):
    """
    Combinación robusta de métodos:
    1. Filter: Eliminar constantes
    2. Filter: Mutual Information
    3. Wrapper: RFE con Random Forest
    4. Consenso: Features seleccionadas por múltiples métodos
    """
    from sklearn.feature_selection import mutual_info_classif

    n_features = X.shape[1]

    # 1. Variance Threshold
    var_selector = VarianceThreshold(threshold=0.01)
    var_mask = var_selector.fit(X).get_support()

    # 2. Mutual Information (top 50%)
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_threshold = np.percentile(mi_scores, 50)
    mi_mask = mi_scores >= mi_threshold

    # 3. RFE
    rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42),
              n_features_to_select=n_select)
    rfe.fit(X, y)
    rfe_mask = rfe.support_

    # 4. Consenso: al menos 2 de 3 métodos
    consensus_score = var_mask.astype(int) + mi_mask.astype(int) + rfe_mask.astype(int)
    consensus_mask = consensus_score >= 2

    # Si tenemos más de n_select, ordenar por MI y tomar top
    if consensus_mask.sum() > n_select:
        indices = np.where(consensus_mask)[0]
        top_indices = indices[np.argsort(mi_scores[indices])[-n_select:]]
        final_mask = np.zeros(n_features, dtype=bool)
        final_mask[top_indices] = True
    else:
        final_mask = consensus_mask

    selected_features = [f for f, m in zip(feature_names, final_mask) if m]

    return final_mask, selected_features, {
        'variance': var_mask.sum(),
        'mutual_info': mi_mask.sum(),
        'rfe': rfe_mask.sum(),
        'consensus': consensus_mask.sum(),
        'final': final_mask.sum()
    }
```

---

## 7. Resumen

| Escenario | Método Recomendado |
|-----------|-------------------|
| Dataset grande, primera pasada | Variance + Correlation |
| Relaciones no lineales | Mutual Information |
| Features categóricas | Chi-cuadrado |
| Necesito ranking | RFE con CV |
| Muchas features (>100) | Algoritmo Genético o Lasso |
| Modelo final es RF/XGBoost | Tree-based importance |
| Features correlacionadas | ElasticNet |
| Producción con costo | Selección con costo-beneficio |

**Reglas de Oro:**
1. Siempre usar pipeline para evitar data leakage
2. Combinar múltiples métodos para robustez
3. Considerar el costo de extracción en producción
4. Validar con cross-validation, no solo train score
