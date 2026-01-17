# Introduccion a la Reduccion de Dimensionalidad

## Que es la Dimensionalidad

La **dimensionalidad** de un dataset se refiere al numero de features (variables) que describen cada observacion. Un dataset con 100 features tiene dimensionalidad 100.

```
DIMENSIONALIDAD
===============

Dataset con 3 features:
    [temperatura, humedad, presion]
    Dimensionalidad = 3

Dataset de red con 50 features:
    [bytes_in, bytes_out, packets, ports, flags, ...]
    Dimensionalidad = 50

Dataset de malware con 1000 features:
    [api_calls, strings, imports, sections, entropy, ...]
    Dimensionalidad = 1000


Visualizacion:

1D:  ────●────●──●─────●──────
           puntos en una linea

2D:       ●
        ●   ●
          ●    ●
        ●     ●
           puntos en un plano

3D:        ●
          /|\
         ● | ●
          \|/
           ●
        puntos en espacio 3D

100D:  ??? (imposible visualizar)
```

---

## La Maldicion de la Dimensionalidad

### Concepto

```
CURSE OF DIMENSIONALITY
=======================

A medida que aumentan las dimensiones:

1. El espacio se vuelve ENORME
2. Los datos se vuelven DISPERSOS
3. Las distancias pierden SIGNIFICADO
4. Se necesitan EXPONENCIALMENTE mas datos


Ejemplo - Volumen del espacio:

    Dim    Puntos para cubrir 10% del espacio
    ───    ────────────────────────────────
     1     10 puntos
     2     100 puntos (10²)
     3     1,000 puntos (10³)
    10     10,000,000,000 puntos (10¹⁰)
   100     10¹⁰⁰ puntos (imposible!)


El espacio crece exponencialmente,
pero nuestros datos son FINITOS.
```

### Problemas Especificos

```
PROBLEMA 1: DISTANCIAS PIERDEN SIGNIFICADO
==========================================

En alta dimension, todos los puntos estan "igual de lejos".

    Dimension    dist_max / dist_min
    ─────────    ───────────────────
        2             ~3
       10             ~1.5
      100             ~1.1
     1000             ~1.01

    Todos los vecinos estan a distancia similar!
    K-NN y clustering fallan.


PROBLEMA 2: DATOS DISPERSOS (Sparse)
====================================

Con n=1000 muestras:

    En 2D:  Densidad alta, puntos cercanos
            ●●●●●●
            ●●●●●●
            ●●●●●●

    En 100D: Densidad ~0, puntos aislados
             ●                    ●

                    ●
                           ●
                 ●


PROBLEMA 3: OVERFITTING
=======================

    Features > Samples  →  Overfitting casi garantizado

    El modelo memoriza ruido en lugar de patrones.
    Cada feature adicional es una oportunidad de sobreajustar.
```

### Demostracion Matematica

```python
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_curse_of_dimensionality():
    """
    Demuestra como las distancias convergen en alta dimension.
    """
    np.random.seed(42)
    n_points = 1000

    results = []

    for dim in [2, 5, 10, 50, 100, 500, 1000]:
        # Generar puntos aleatorios en hipercubo [0,1]^dim
        points = np.random.rand(n_points, dim)

        # Calcular distancias al origen
        distances = np.sqrt((points ** 2).sum(axis=1))

        # Ratio max/min
        ratio = distances.max() / distances.min()

        # Estadisticas
        results.append({
            'dim': dim,
            'mean_dist': distances.mean(),
            'std_dist': distances.std(),
            'ratio_max_min': ratio,
            'cv': distances.std() / distances.mean()  # Coef. variacion
        })

        print(f"Dim={dim:4d}: mean={distances.mean():.3f}, "
              f"std={distances.std():.3f}, ratio={ratio:.3f}")

    return results


def show_distance_concentration():
    """
    Muestra como las distancias se concentran en alta dimension.
    """
    np.random.seed(42)
    n_points = 1000

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, dim in zip(axes, [2, 50, 500]):
        points = np.random.rand(n_points, dim)
        distances = np.sqrt((points ** 2).sum(axis=1))

        ax.hist(distances, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(distances.mean(), color='red', linestyle='--',
                   label=f'Mean: {distances.mean():.2f}')
        ax.set_title(f'Dimension = {dim}')
        ax.set_xlabel('Distancia al origen')
        ax.legend()

    plt.tight_layout()
    plt.savefig('distance_concentration.png')
    print("Grafico guardado: distance_concentration.png")


if __name__ == "__main__":
    print("=== Curse of Dimensionality ===\n")
    demonstrate_curse_of_dimensionality()
```

---

## Por Que Reducir Dimensiones

### Beneficios

```
BENEFICIOS DE REDUCIR DIMENSIONALIDAD
=====================================

1. VISUALIZACION
   100D → 2D para plotear y entender datos
   Ver clusters, outliers, estructura

2. REDUCIR OVERFITTING
   Menos features = menos oportunidad de memorizar
   Mejor generalizacion

3. ACELERAR ENTRENAMIENTO
   Menos features = menos computo
   O(n * d²) → O(n * k²) donde k << d

4. ELIMINAR RUIDO
   Features irrelevantes añaden ruido
   Reduccion puede filtrar ruido

5. MITIGAR CURSE OF DIMENSIONALITY
   Distancias vuelven a ser significativas
   Algoritmos basados en distancia funcionan

6. REDUCIR ALMACENAMIENTO
   Menos features = menos memoria
   Importante para datasets grandes
```

### Cuando Reducir

```
CUANDO REDUCIR DIMENSIONALIDAD
==============================

SI:
├── Tienes muchas features (>50-100)
├── Sospechas features redundantes
├── Modelo sufre overfitting
├── Necesitas visualizar datos
├── Entrenamiento muy lento
└── Algoritmo basado en distancia (KNN, SVM-RBF)

NO:
├── Features son pocas y bien seleccionadas
├── Modelo funciona bien sin reducir
├── Interpretabilidad es critica
├── Features tienen significado de dominio importante
└── Ya tienes features ingenierizadas


REGLA PRACTICA:
    n_samples / n_features > 10  →  Probablemente OK
    n_samples / n_features < 10  →  Considerar reducir
```

---

## Taxonomia de Metodos

### Vision General

```
METODOS DE REDUCCION DE DIMENSIONALIDAD
=======================================

                    Reduccion de Dimensionalidad
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
    Feature Selection                   Feature Extraction
    (seleccionar subset)                (crear nuevas features)
            │                                   │
    ┌───────┼───────┐               ┌───────────┼───────────┐
    │       │       │               │           │           │
  Filter  Wrapper Embedded        Lineal    No Lineal    Autoencoders
    │       │       │               │           │           │
 Corr    RFE     Lasso           PCA       t-SNE        AE
 MI      SFS     Ridge           LDA       UMAP         VAE
 Chi2    Genetic ElasticNet      FA        Isomap


Feature Selection: Elige features originales
Feature Extraction: Crea combinaciones de features
```

### Comparativa

| Metodo | Tipo | Lineal | Preserva | Uso Principal |
|--------|------|--------|----------|---------------|
| PCA | Extraction | Si | Varianza global | Preprocesamiento |
| LDA | Extraction | Si | Separabilidad clases | Clasificacion |
| t-SNE | Extraction | No | Vecindarios locales | Visualizacion |
| UMAP | Extraction | No | Estructura global+local | Visualizacion |
| Filter | Selection | - | Features originales | Rapido, baseline |
| Wrapper | Selection | - | Features originales | Optimo pero lento |
| Embedded | Selection | - | Features originales | Durante entrenamiento |

---

## Feature Selection vs Feature Extraction

### Diferencia Fundamental

```
FEATURE SELECTION
=================

Entrada: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

Proceso: Evaluar importancia de cada feature

Salida:  [f2, f5, f7]  (subset de originales)

Caracteristicas:
+ Features interpretables (son las originales)
+ Reduce costo de adquisicion de datos
- Puede perder interacciones entre features


FEATURE EXTRACTION
==================

Entrada: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

Proceso: Crear combinaciones/transformaciones

Salida:  [PC1, PC2, PC3]  (nuevas features)

         PC1 = 0.3*f1 + 0.5*f2 - 0.2*f3 + ...
         PC2 = 0.1*f1 - 0.4*f2 + 0.6*f3 + ...
         PC3 = ...

Caracteristicas:
+ Captura interacciones
+ Puede comprimir mas
- Pierde interpretabilidad
- Necesita todas las features originales
```

### Cuando Usar Cada Uno

```
FEATURE SELECTION:
├── Interpretabilidad importante
├── Quieres reducir costo de adquisicion
├── Features tienen significado de dominio
├── Sospechas features irrelevantes
└── Modelo final debe ser explicable

FEATURE EXTRACTION:
├── Solo importa performance
├── Visualizacion es el objetivo
├── Features muy correlacionadas
├── Quieres maxima compresion
└── Datos de imagen/texto/señal
```

---

## Metricas de Evaluacion

### Para Feature Extraction

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def evaluate_dimensionality_reduction(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    y: np.ndarray | None = None
) -> dict:
    """
    Evalua calidad de la reduccion de dimensionalidad.

    Args:
        X_original: Datos originales [n_samples, n_features]
        X_reduced: Datos reducidos [n_samples, n_components]
        y: Labels (opcional, para metricas supervisadas)

    Returns:
        Diccionario con metricas
    """
    metrics = {}

    # 1. Varianza explicada (si PCA)
    # Se calcula durante PCA

    # 2. Reconstruction error
    # Solo si tenemos metodo de reconstruccion

    # 3. Trustworthiness: Vecinos en espacio reducido deben ser
    #    vecinos en espacio original
    from sklearn.manifold import trustworthiness
    metrics['trustworthiness'] = trustworthiness(
        X_original, X_reduced, n_neighbors=5
    )

    # 4. Si tenemos labels: separabilidad de clases
    if y is not None:
        # Silhouette en espacio reducido
        metrics['silhouette_reduced'] = silhouette_score(X_reduced, y)

        # Performance de clasificador simple
        knn = KNeighborsClassifier(n_neighbors=5)
        metrics['knn_accuracy'] = cross_val_score(
            knn, X_reduced, y, cv=5
        ).mean()

    # 5. Ratio de compresion
    metrics['compression_ratio'] = X_original.shape[1] / X_reduced.shape[1]

    return metrics


def variance_explained_analysis(X: np.ndarray, max_components: int = 50) -> dict:
    """
    Analiza varianza explicada por numero de componentes.
    Ayuda a elegir numero optimo de componentes PCA.
    """
    n_components = min(max_components, X.shape[1], X.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Encontrar componentes para diferentes umbrales
    thresholds = [0.80, 0.90, 0.95, 0.99]
    components_needed = {}

    for thresh in thresholds:
        n = np.argmax(cumulative_variance >= thresh) + 1
        components_needed[f'{int(thresh*100)}%'] = n

    return {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cumulative_variance,
        'components_for_threshold': components_needed,
        'total_components': n_components
    }
```

### Para Feature Selection

```python
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
    SelectKBest
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def evaluate_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    selected_indices: list[int],
    cv: int = 5
) -> dict:
    """
    Evalua calidad de feature selection.
    """
    X_selected = X[:, selected_indices]

    # Modelo con features seleccionadas vs todas
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    score_all = cross_val_score(model, X, y, cv=cv).mean()
    score_selected = cross_val_score(model, X_selected, y, cv=cv).mean()

    return {
        'n_features_original': X.shape[1],
        'n_features_selected': len(selected_indices),
        'reduction_ratio': len(selected_indices) / X.shape[1],
        'accuracy_all_features': score_all,
        'accuracy_selected': score_selected,
        'accuracy_change': score_selected - score_all,
        'efficiency_gain': (
            (score_selected / score_all) *
            (X.shape[1] / len(selected_indices))
        )
    }
```

---

## Implementacion Basica de Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


class DimensionalityReductionPipeline:
    """
    Pipeline para comparar metodos de reduccion.
    """

    def __init__(
        self,
        target_dim: int | None = None,
        variance_threshold: float = 0.95
    ):
        self.target_dim = target_dim
        self.variance_threshold = variance_threshold
        self.results = {}

    def compare_methods(
        self,
        X: np.ndarray,
        y: np.ndarray,
        methods: list[str] = ['pca', 'select_k', 'mutual_info']
    ) -> pd.DataFrame:
        """
        Compara diferentes metodos de reduccion.
        """
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = []
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Baseline: todas las features
        baseline_score = cross_val_score(model, X_scaled, y, cv=5).mean()
        results.append({
            'method': 'baseline (all)',
            'n_features': X.shape[1],
            'accuracy': baseline_score,
            'time': 0
        })

        # Determinar target dimension
        if self.target_dim is None:
            # Usar PCA para determinar
            pca_full = PCA()
            pca_full.fit(X_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            self.target_dim = np.argmax(cumvar >= self.variance_threshold) + 1
            print(f"Target dimension (95% variance): {self.target_dim}")

        import time

        for method in methods:
            start = time.time()

            if method == 'pca':
                reducer = PCA(n_components=self.target_dim)
                X_reduced = reducer.fit_transform(X_scaled)

            elif method == 'select_k':
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=self.target_dim
                )
                X_reduced = selector.fit_transform(X_scaled, y)

            elif method == 'mutual_info':
                # Ranking por mutual information
                mi_scores = mutual_info_classif(X_scaled, y)
                top_k = np.argsort(mi_scores)[-self.target_dim:]
                X_reduced = X_scaled[:, top_k]

            elapsed = time.time() - start

            # Evaluar
            score = cross_val_score(model, X_reduced, y, cv=5).mean()

            results.append({
                'method': method,
                'n_features': X_reduced.shape[1],
                'accuracy': score,
                'time': elapsed
            })

        self.results = pd.DataFrame(results)
        return self.results

    def recommend(self) -> str:
        """
        Recomienda mejor metodo basado en accuracy vs complejidad.
        """
        if self.results is None or len(self.results) == 0:
            return "Ejecuta compare_methods primero"

        df = self.results[self.results['method'] != 'baseline (all)']

        # Score ponderado: accuracy - penalizacion por tiempo
        df['score'] = df['accuracy'] - 0.01 * df['time']

        best = df.loc[df['score'].idxmax()]

        return f"Recomendado: {best['method']} (acc={best['accuracy']:.4f})"


# Ejemplo
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Dataset sintetico
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        n_classes=2,
        random_state=42
    )

    print("=== Comparativa de Reduccion de Dimensionalidad ===\n")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"(20 informativas, 30 redundantes, 50 ruido)\n")

    pipeline = DimensionalityReductionPipeline(variance_threshold=0.95)
    results = pipeline.compare_methods(X, y)

    print(results.to_string(index=False))
    print(f"\n{pipeline.recommend()}")
```

---

## Aplicaciones en Ciberseguridad

```
DONDE SE USA EN CIBERSEGURIDAD
==============================

1. ANALISIS DE MALWARE
   Features: API calls, strings, imports, entropy, sections...
   Tipico: 1000+ features
   Reduccion: PCA para clasificacion, t-SNE para visualizar familias

2. DETECCION DE INTRUSIONES (IDS)
   Features: Estadisticas de flujo, puertos, flags, bytes...
   Tipico: 50-200 features (NSL-KDD tiene 41, CICIDS tiene 80+)
   Reduccion: Feature selection para interpretabilidad

3. ANALISIS DE LOGS
   Features: TF-IDF de mensajes, conteos, timestamps...
   Tipico: 10000+ features (vocabulario)
   Reduccion: PCA/SVD, embeddings

4. THREAT INTELLIGENCE
   Features: IOCs, TTPs, relaciones...
   Tipico: Grafos de alta dimension
   Reduccion: Node embeddings, graph reduction

5. BEHAVIORAL ANALYTICS (UEBA)
   Features: Actividades de usuario, horarios, recursos...
   Tipico: 100+ features
   Reduccion: Autoencoders para anomaly detection
```

---

## Resumen

```
CHECKLIST REDUCCION DE DIMENSIONALIDAD
======================================

1. ¿NECESITO REDUCIR?
   [ ] n_features > 50-100
   [ ] n_samples / n_features < 10
   [ ] Modelo tiene overfitting
   [ ] Entrenamiento muy lento

2. ¿QUE METODO?
   [ ] Interpretabilidad → Feature Selection
   [ ] Solo performance → PCA / Autoencoder
   [ ] Visualizacion → t-SNE / UMAP
   [ ] Clasificacion → LDA

3. ¿CUANTAS DIMENSIONES?
   [ ] PCA: 90-95% varianza explicada
   [ ] t-SNE/UMAP: 2-3 para visualizacion
   [ ] Feature Selection: validar con CV

4. EVALUAR
   [ ] Comparar accuracy antes/despues
   [ ] Verificar no perder demasiada info
   [ ] Visualizar si es posible
```

### Puntos Clave

1. **Curse of dimensionality**: espacios de alta dimension son problematicos
2. **Feature Selection**: mantiene features originales (interpretable)
3. **Feature Extraction**: crea nuevas features (mas compresion)
4. **PCA**: baseline para reduccion lineal
5. **t-SNE/UMAP**: para visualizacion no lineal
6. **Evaluar siempre**: comparar performance antes y despues
7. En ciberseguridad: fundamental para malware, IDS, logs
