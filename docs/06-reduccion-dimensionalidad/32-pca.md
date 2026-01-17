# PCA (Principal Component Analysis)

## Introduccion

**PCA** (Analisis de Componentes Principales) es la tecnica de reduccion de dimensionalidad mas utilizada. Transforma los datos a un nuevo sistema de coordenadas donde los ejes (componentes principales) estan ordenados por la cantidad de varianza que capturan.

```
PCA: IDEA INTUITIVA
===================

Datos originales (2D):         Despues de PCA:
       y                            PC2
       │    ●●                       │
       │   ●●●●                      │    ●
       │  ●●●●●●                     │   ●●●
       │ ●●●●●●●●                    │  ●●●●●
       │●●●●●●●●●●              ─────┼●●●●●●●●──── PC1
       └──────────── x               │  ●●●●●
                                     │   ●●●
                                     │    ●

PC1 captura la MAXIMA varianza (direccion de mayor dispersion)
PC2 es ORTOGONAL a PC1 y captura la siguiente mayor varianza

Si los datos estan muy "estirados" en PC1,
podemos descartar PC2 con poca perdida de info.
```

## Fundamentos Matematicos

### Varianza y Covarianza

```
COVARIANZA
==========

Cov(X, Y) = E[(X - μ_x)(Y - μ_y)]

    Cov > 0: Cuando X sube, Y tiende a subir
    Cov < 0: Cuando X sube, Y tiende a bajar
    Cov ≈ 0: X e Y no tienen relacion lineal


MATRIZ DE COVARIANZA
====================

Para datos X con d features:

         ┌                              ┐
         │ Var(x₁)   Cov(x₁,x₂) ... │
    Σ =  │ Cov(x₂,x₁) Var(x₂)   ... │
         │ ...       ...        ... │
         └                              ┘

    Matriz d × d, simetrica
    Diagonal = varianzas de cada feature
    Fuera de diagonal = covarianzas entre features
```

### Eigenvectors y Eigenvalues

```
EIGENVECTORS Y EIGENVALUES
==========================

Para la matriz de covarianza Σ:

    Σ · v = λ · v

Donde:
    v = eigenvector (direccion)
    λ = eigenvalue (magnitud/varianza en esa direccion)


Propiedades:
- Los eigenvectors son ORTOGONALES entre si
- Los eigenvalues indican cuanta VARIANZA hay en cada direccion
- Ordenados: λ₁ >= λ₂ >= λ₃ >= ...

          │
    λ₁    │●●●●●●●●●●●●●●●●●●●●  (mayor varianza)
          │
    λ₂    │●●●●●●●●●
          │
    λ₃    │●●●●
          │
    λ₄    │●●
          │
          └─────────────────────
```

### Algoritmo PCA

```
ALGORITMO PCA
=============

Input: X ∈ ℝⁿˣᵈ (n muestras, d features)
Output: X_reduced ∈ ℝⁿˣᵏ (n muestras, k componentes)

1. CENTRAR datos (restar media):
   X_centered = X - mean(X)

2. CALCULAR matriz de covarianza:
   Σ = (1/n) · X_centered.T · X_centered

3. CALCULAR eigenvectors y eigenvalues de Σ:
   Σ · V = V · Λ
   donde V = [v₁, v₂, ..., vₐ], Λ = diag(λ₁, λ₂, ..., λₐ)

4. ORDENAR eigenvectors por eigenvalue descendente

5. SELECCIONAR top k eigenvectors:
   W = [v₁, v₂, ..., vₖ]

6. PROYECTAR datos:
   X_reduced = X_centered · W


Varianza explicada por componente i:
    explained_variance_ratio_i = λᵢ / Σλⱼ
```

---

## Implementacion

### PCA desde Cero

```python
import numpy as np
from typing import Tuple


class PCAFromScratch:
    """
    Implementacion de PCA desde cero para entender el algoritmo.
    """

    def __init__(self, n_components: int | None = None):
        """
        Args:
            n_components: Numero de componentes a retener.
                         None = todos
        """
        self.n_components = n_components
        self.components_ = None          # Eigenvectors (direcciones)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> 'PCAFromScratch':
        """
        Calcula componentes principales.

        Args:
            X: Datos [n_samples, n_features]
        """
        n_samples, n_features = X.shape

        # 1. Centrar datos
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 2. Matriz de covarianza
        cov_matrix = np.cov(X_centered.T)

        # 3. Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Ordenar por eigenvalue descendente
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Seleccionar componentes
        if self.n_components is None:
            self.n_components = n_features

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / eigenvalues.sum()
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyecta datos al espacio reducido.
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit y transform en un paso."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruye datos originales (aproximacion).
        """
        return X_reduced @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calcula error de reconstruccion (MSE).
        """
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        return np.mean((X - X_reconstructed) ** 2)


# Verificar contra sklearn
def verify_implementation():
    """Verifica que nuestra implementacion coincide con sklearn."""
    from sklearn.decomposition import PCA as SklearnPCA

    np.random.seed(42)
    X = np.random.randn(100, 10)

    # Nuestra implementacion
    pca_scratch = PCAFromScratch(n_components=3)
    X_scratch = pca_scratch.fit_transform(X)

    # Sklearn
    pca_sklearn = SklearnPCA(n_components=3)
    X_sklearn = pca_sklearn.fit_transform(X)

    # Los resultados pueden diferir en signo (eigenvectors no son unicos en signo)
    # Comparamos varianza explicada
    print("Varianza explicada (scratch):", pca_scratch.explained_variance_ratio_)
    print("Varianza explicada (sklearn):", pca_sklearn.explained_variance_ratio_)
    print("Diferencia maxima:", np.max(np.abs(
        pca_scratch.explained_variance_ratio_ - pca_sklearn.explained_variance_ratio_
    )))


if __name__ == "__main__":
    verify_implementation()
```

### PCA con Sklearn

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca_analysis(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    n_components: int | None = None,
    variance_threshold: float = 0.95
) -> dict:
    """
    Analisis PCA completo.

    Args:
        X: Datos [n_samples, n_features]
        feature_names: Nombres de features
        n_components: Componentes a retener (None = automatico)
        variance_threshold: Varianza minima a retener si n_components=None

    Returns:
        Diccionario con resultados del analisis
    """
    # Estandarizar (importante para PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA completo primero para analizar
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Varianza acumulada
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Determinar n_components si no se especifica
    if n_components is None:
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # PCA final
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    # Loadings (contribucion de cada feature a cada componente)
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )

    return {
        'X_reduced': X_reduced,
        'pca': pca,
        'scaler': scaler,
        'n_components': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance[:n_components],
        'total_variance_retained': cumulative_variance[n_components - 1],
        'loadings': loadings,
        'feature_importance': get_feature_importance(pca, feature_names)
    }


def get_feature_importance(
    pca: PCA,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Calcula importancia de features basada en loadings.
    """
    # Importancia = suma de |loading| * varianza_explicada
    importance = np.zeros(len(feature_names))

    for i, (component, var) in enumerate(zip(
        pca.components_, pca.explained_variance_ratio_
    )):
        importance += np.abs(component) * var

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)


def plot_variance_explained(pca: PCA, figsize: tuple = (12, 4)):
    """Grafica varianza explicada."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    n = len(pca.explained_variance_ratio_)
    x = range(1, n + 1)

    # Individual
    axes[0].bar(x, pca.explained_variance_ratio_, alpha=0.7)
    axes[0].set_xlabel('Componente Principal')
    axes[0].set_ylabel('Varianza Explicada')
    axes[0].set_title('Varianza por Componente')

    # Acumulada
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(x, cumulative, 'bo-')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90%')
    axes[1].set_xlabel('Numero de Componentes')
    axes[1].set_ylabel('Varianza Acumulada')
    axes[1].set_title('Varianza Acumulada')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pca_variance.png')
    print("Grafico guardado: pca_variance.png")


def plot_loadings_heatmap(loadings: pd.DataFrame, top_n: int = 20):
    """Grafica heatmap de loadings."""
    # Seleccionar top features por importancia total
    importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
    top_features = importance.head(top_n).index

    plt.figure(figsize=(10, 8))
    plt.imshow(loadings.loc[top_features].values, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Loading')
    plt.xticks(range(len(loadings.columns)), loadings.columns)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Componente Principal')
    plt.ylabel('Feature')
    plt.title('PCA Loadings (Top Features)')
    plt.tight_layout()
    plt.savefig('pca_loadings.png')
    print("Grafico guardado: pca_loadings.png")
```

---

## Eleccion del Numero de Componentes

### Metodos

```
METODOS PARA ELEGIR N_COMPONENTS
================================

1. UMBRAL DE VARIANZA (mas comun)
   Retener componentes hasta alcanzar X% de varianza
   Tipico: 90%, 95%, 99%

   cumulative_variance >= 0.95 → n_components


2. ELBOW METHOD (Scree Plot)
   Buscar "codo" donde la curva se aplana

       Varianza
          │\
          │ \
          │  \
          │   \_____  ← codo aqui
          │
          └──────────
            Componentes


3. KAISER CRITERION
   Retener componentes con eigenvalue > 1
   (Solo si datos estan estandarizados)


4. PARALLEL ANALYSIS
   Comparar eigenvalues reales vs aleatorios
   Retener donde real > aleatorio


5. VALIDACION CRUZADA
   Probar diferentes n y elegir por performance en tarea downstream
```

### Implementacion

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def find_optimal_components(
    X: np.ndarray,
    y: np.ndarray | None = None,
    method: str = 'variance',
    threshold: float = 0.95,
    max_components: int | None = None
) -> dict:
    """
    Encuentra numero optimo de componentes PCA.

    Args:
        X: Datos
        y: Labels (para metodo 'cv')
        method: 'variance', 'elbow', 'kaiser', 'cv'
        threshold: Umbral para metodo variance
        max_components: Maximo a considerar

    Returns:
        Diccionario con analisis
    """
    if max_components is None:
        max_components = min(X.shape)

    # PCA completo
    pca = PCA(n_components=max_components)
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    variance_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance_ratio)

    if method == 'variance':
        # Primer componente que alcanza threshold
        n_optimal = np.argmax(cumulative >= threshold) + 1
        reason = f"First to reach {threshold*100}% variance"

    elif method == 'elbow':
        # Buscar punto de mayor curvatura
        # Segunda derivada de la curva
        diff1 = np.diff(variance_ratio)
        diff2 = np.diff(diff1)
        n_optimal = np.argmax(diff2) + 2  # +2 por los diffs
        reason = "Maximum curvature (elbow)"

    elif method == 'kaiser':
        # Eigenvalues > 1 (datos deben estar estandarizados)
        n_optimal = np.sum(eigenvalues > 1)
        if n_optimal == 0:
            n_optimal = 1
        reason = "Eigenvalues > 1"

    elif method == 'cv':
        # Cross-validation con clasificador
        if y is None:
            raise ValueError("Se necesita y para metodo cv")

        scores = []
        for n in range(1, max_components + 1):
            pca_n = PCA(n_components=n)
            X_reduced = pca_n.fit_transform(X)

            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            score = cross_val_score(clf, X_reduced, y, cv=3).mean()
            scores.append(score)

        n_optimal = np.argmax(scores) + 1
        reason = f"Best CV score: {max(scores):.4f}"

    else:
        raise ValueError(f"Metodo desconocido: {method}")

    return {
        'method': method,
        'n_optimal': n_optimal,
        'reason': reason,
        'variance_at_optimal': cumulative[n_optimal - 1],
        'eigenvalues': eigenvalues,
        'variance_ratio': variance_ratio,
        'cumulative_variance': cumulative
    }


def parallel_analysis(X: np.ndarray, n_iterations: int = 100) -> dict:
    """
    Parallel Analysis para determinar numero de componentes.
    Compara eigenvalues reales vs datos aleatorios.
    """
    n_samples, n_features = X.shape

    # PCA en datos reales
    pca_real = PCA()
    pca_real.fit(X)
    real_eigenvalues = pca_real.explained_variance_

    # Eigenvalues de datos aleatorios (multiples iteraciones)
    random_eigenvalues = np.zeros((n_iterations, min(n_samples, n_features)))

    for i in range(n_iterations):
        X_random = np.random.randn(n_samples, n_features)
        pca_random = PCA()
        pca_random.fit(X_random)
        random_eigenvalues[i] = pca_random.explained_variance_

    # Percentil 95 de eigenvalues aleatorios
    threshold_eigenvalues = np.percentile(random_eigenvalues, 95, axis=0)

    # Componentes donde real > aleatorio
    n_components = np.sum(real_eigenvalues > threshold_eigenvalues)

    return {
        'n_components': max(1, n_components),
        'real_eigenvalues': real_eigenvalues,
        'threshold_eigenvalues': threshold_eigenvalues,
        'comparison': real_eigenvalues > threshold_eigenvalues
    }
```

---

## Variantes de PCA

### Kernel PCA (No Lineal)

```
KERNEL PCA
==========

PCA estandar: Solo captura relaciones LINEALES

Kernel PCA: Captura relaciones NO LINEALES usando kernel trick

    Datos originales          PCA                  Kernel PCA
         ●●●                   │                       │
        ●   ●                  │─────●●●●●            │  ●●●●●
       ●     ●                 │                      │●●
      ●       ●                │                       ●
       ●     ●                 └─────────             └─────────
        ●   ●               (no puede separar)      (si puede!)
         ●●●

    Datos no linealmente       Proyeccion lineal     Kernel proyecta
    separables                 falla                 a espacio donde
                                                     es lineal


Kernels comunes:
- RBF (Gaussian): K(x,y) = exp(-γ||x-y||²)
- Polynomial: K(x,y) = (x·y + c)^d
- Sigmoid: K(x,y) = tanh(α·x·y + c)
```

```python
from sklearn.decomposition import KernelPCA
import numpy as np


def kernel_pca_analysis(
    X: np.ndarray,
    n_components: int = 2,
    kernels: list[str] = ['linear', 'rbf', 'poly']
) -> dict:
    """
    Compara diferentes kernels para PCA.
    """
    results = {}

    for kernel in kernels:
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=10 if kernel == 'rbf' else None,
            degree=3 if kernel == 'poly' else None
        )

        X_reduced = kpca.fit_transform(X)

        results[kernel] = {
            'X_reduced': X_reduced,
            'model': kpca
        }

    return results


# Ejemplo: Datos no lineales
def demo_kernel_pca():
    """Demuestra ventaja de Kernel PCA en datos no lineales."""
    np.random.seed(42)

    # Crear datos en forma de circulo
    n = 500
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.choice([1, 2], n)  # Dos circulos concentricos
    X = np.column_stack([
        r * np.cos(theta) + np.random.randn(n) * 0.1,
        r * np.sin(theta) + np.random.randn(n) * 0.1
    ])
    y = (r == 2).astype(int)

    # Comparar PCA vs Kernel PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
    X_kpca = kpca.fit_transform(X)

    print("PCA lineal no puede separar circulos concentricos")
    print("Kernel PCA (RBF) puede proyectarlos a espacio separable")

    return X_pca, X_kpca, y
```

### Incremental PCA (Para Big Data)

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np


def incremental_pca_large_dataset(
    data_generator,
    n_components: int,
    batch_size: int = 1000
) -> IncrementalPCA:
    """
    PCA incremental para datasets que no caben en memoria.

    Args:
        data_generator: Generador que yield batches de datos
        n_components: Componentes a calcular
        batch_size: Tamano de cada batch

    Returns:
        Modelo IncrementalPCA ajustado
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    for batch in data_generator:
        ipca.partial_fit(batch)

    return ipca


# Ejemplo con archivo grande
def pca_from_file(
    filepath: str,
    n_components: int = 50,
    chunk_size: int = 10000
) -> IncrementalPCA:
    """
    Aplica PCA a archivo CSV grande.
    """
    import pandas as pd

    ipca = IncrementalPCA(n_components=n_components)

    # Leer en chunks
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Preprocesar chunk
        X_chunk = chunk.select_dtypes(include=[np.number]).values
        X_chunk = np.nan_to_num(X_chunk)  # Manejar NaN

        ipca.partial_fit(X_chunk)

    return ipca
```

### Sparse PCA

```python
from sklearn.decomposition import SparsePCA
import numpy as np


def sparse_pca_analysis(
    X: np.ndarray,
    n_components: int = 10,
    alpha: float = 1.0
) -> dict:
    """
    Sparse PCA: Componentes con pocos features no-cero.
    Util para interpretabilidad.

    Args:
        X: Datos
        n_components: Componentes
        alpha: Regularizacion L1 (mayor = mas sparse)
    """
    spca = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=42
    )

    X_reduced = spca.fit_transform(X)

    # Contar features no-cero por componente
    sparsity = []
    for i, component in enumerate(spca.components_):
        n_nonzero = np.sum(component != 0)
        sparsity.append({
            'component': i + 1,
            'n_nonzero': n_nonzero,
            'sparsity': 1 - n_nonzero / len(component)
        })

    return {
        'X_reduced': X_reduced,
        'model': spca,
        'components': spca.components_,
        'sparsity_info': sparsity
    }
```

---

## Aplicacion: PCA para Analisis de Malware

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class MalwarePCAAnalyzer:
    """
    Analiza features de malware usando PCA.
    Visualiza familias y detecta outliers.
    """

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None):
        """
        Ajusta PCA a features de malware.

        Args:
            X: Features [n_samples, n_features]
            feature_names: Nombres de features
        """
        self.feature_names = feature_names or [
            f'f{i}' for i in range(X.shape[1])
        ]

        # Estandarizar y aplicar PCA
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforma nuevos datos."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def analyze_components(self) -> pd.DataFrame:
        """
        Analiza que features contribuyen mas a cada componente.
        """
        if not self.is_fitted:
            raise ValueError("Modelo no ajustado")

        # Top features por componente
        analysis = []

        for i, component in enumerate(self.pca.components_[:10]):
            # Top 5 features positivas y negativas
            top_pos_idx = np.argsort(component)[-5:][::-1]
            top_neg_idx = np.argsort(component)[:5]

            for idx in top_pos_idx:
                analysis.append({
                    'component': f'PC{i+1}',
                    'feature': self.feature_names[idx],
                    'loading': component[idx],
                    'direction': 'positive'
                })

            for idx in top_neg_idx:
                analysis.append({
                    'component': f'PC{i+1}',
                    'feature': self.feature_names[idx],
                    'loading': component[idx],
                    'direction': 'negative'
                })

        return pd.DataFrame(analysis)

    def visualize_families(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        label_names: dict | None = None
    ):
        """
        Visualiza familias de malware en espacio PCA 2D.
        """
        X_pca = self.transform(X)[:, :2]

        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            name = label_names.get(label, str(label)) if label_names else str(label)
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=name,
                alpha=0.6,
                s=50
            )

        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('Malware Families in PCA Space')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('malware_pca.png', dpi=150, bbox_inches='tight')
        print("Grafico guardado: malware_pca.png")

    def detect_outliers(
        self,
        X: np.ndarray,
        threshold_std: float = 3.0
    ) -> np.ndarray:
        """
        Detecta muestras anomalas basado en reconstruccion PCA.
        Muestras que no se reconstruyen bien son outliers.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)

        # Error de reconstruccion por muestra
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

        # Outliers = error > threshold
        mean_error = reconstruction_error.mean()
        std_error = reconstruction_error.std()
        threshold = mean_error + threshold_std * std_error

        is_outlier = reconstruction_error > threshold

        return is_outlier, reconstruction_error


# Demo
def demo_malware_pca():
    """Demo de PCA para analisis de malware."""
    np.random.seed(42)

    # Simular features de malware
    # 3 familias con diferentes caracteristicas
    n_per_family = 200
    n_features = 100

    # Familia 1: Alta entropia, muchas API calls
    family1 = np.random.randn(n_per_family, n_features)
    family1[:, :20] += 3  # Features 0-19 altas

    # Familia 2: Baja entropia, pocas API calls
    family2 = np.random.randn(n_per_family, n_features)
    family2[:, 20:40] += 3  # Features 20-39 altas

    # Familia 3: Caracteristicas mixtas
    family3 = np.random.randn(n_per_family, n_features)
    family3[:, 40:60] += 3  # Features 40-59 altas

    X = np.vstack([family1, family2, family3])
    labels = np.array([0] * n_per_family + [1] * n_per_family + [2] * n_per_family)

    # Analizar
    print("=== Malware PCA Analysis ===\n")

    analyzer = MalwarePCAAnalyzer(n_components=20)
    analyzer.fit(X)

    # Varianza explicada
    print("Varianza explicada (top 5 componentes):")
    for i, var in enumerate(analyzer.pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var*100:.2f}%")

    print(f"\nVarianza total (20 comp): "
          f"{sum(analyzer.pca.explained_variance_ratio_)*100:.2f}%")

    # Visualizar
    label_names = {0: 'Ransomware', 1: 'Trojan', 2: 'Worm'}
    analyzer.visualize_families(X, labels, label_names)

    # Detectar outliers
    is_outlier, errors = analyzer.detect_outliers(X)
    print(f"\nOutliers detectados: {is_outlier.sum()} / {len(X)}")

    return analyzer


if __name__ == "__main__":
    demo_malware_pca()
```

---

## Resumen

```
PCA CHEATSHEET
==============

CUANDO USAR PCA:
├── Reducir dimensionalidad antes de modelar
├── Visualizar datos de alta dimension
├── Eliminar multicolinealidad
├── Comprimir datos manteniendo varianza
└── Preprocesamiento para algoritmos sensibles a dimension

CUANDO NO USAR PCA:
├── Relaciones no lineales importantes
├── Features tienen significado de dominio critico
├── Interpretabilidad es prioritaria
└── Datos muy sparse (mejor usar Sparse PCA o NMF)

PASOS:
1. ESTANDARIZAR datos (obligatorio!)
2. Ajustar PCA
3. Elegir n_components (90-95% varianza)
4. Transformar datos
5. Evaluar reconstruccion si es necesario

VARIANTES:
├── Kernel PCA: relaciones no lineales
├── Incremental PCA: big data
├── Sparse PCA: interpretabilidad
└── Randomized PCA: muy alta dimension
```

### Puntos Clave

1. **Estandarizar** antes de PCA (siempre!)
2. **Varianza explicada** determina n_components
3. **Loadings** muestran contribucion de features
4. **Kernel PCA** para relaciones no lineales
5. **Incremental PCA** para datos que no caben en memoria
6. **Reconstruccion** mide calidad de la reduccion
7. En ciberseguridad: analisis de malware, visualizacion de ataques
