# t-SNE y UMAP: Visualizacion No Lineal

## Introduccion

**t-SNE** y **UMAP** son tecnicas de reduccion de dimensionalidad **no lineales** disenadas principalmente para **visualizacion**. A diferencia de PCA, pueden capturar estructuras complejas y preservar vecindarios locales.

```
PCA vs t-SNE/UMAP
=================

PCA (lineal):                t-SNE/UMAP (no lineal):

    Datos 3D                      Datos 3D
       ●●●                           ●●●
      ●●●●●                         ●●●●●
     ●●●●●●●                       ●●●●●●●
         │                              │
         ▼                              ▼
    Proyeccion 2D                 Proyeccion 2D
    ─────────────                 ─────────────
      ●●●●●●●●●                    ●●●   ●●●
    (estructura                    (clusters
     aplastada)                    preservados)


PCA preserva: Varianza GLOBAL
t-SNE/UMAP preservan: Vecindarios LOCALES
```

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Concepto

```
t-SNE: IDEA CENTRAL
===================

1. En espacio ORIGINAL (alta dim):
   - Calcular probabilidad de que j sea vecino de i
   - Puntos cercanos → probabilidad alta
   - Puntos lejanos → probabilidad baja

   P_ij = exp(-||x_i - x_j||² / 2σ²) / Σ exp(...)


2. En espacio REDUCIDO (2D/3D):
   - Definir probabilidades similares
   - Usar distribucion t-Student (colas pesadas)

   Q_ij = (1 + ||y_i - y_j||²)^(-1) / Σ (...)


3. OPTIMIZAR posiciones y_i para que Q ≈ P
   - Minimizar divergencia KL entre P y Q
   - KL(P || Q) = Σ P_ij log(P_ij / Q_ij)


Resultado:
- Vecinos cercanos en alta dim → cercanos en 2D
- Vecinos lejanos en alta dim → lejanos en 2D
- Clusters se preservan
```

### Por Que t-Student

```
DISTRIBUCION t-STUDENT vs GAUSSIAN
==================================

Problema del "crowding":
- En alta dimension hay mucho espacio
- En 2D hay poco espacio
- Los puntos se "amontonan" en el centro

Solucion: Usar t-Student con colas pesadas

    Probabilidad
         │
    Gauss│\
         │ \___           t-Student
         │     \____  ←── colas mas pesadas
         └────────────────── Distancia

t-Student permite que puntos moderadamente lejanos
queden MAS separados en 2D, evitando el crowding.
```

### Hiperparametros

```
HIPERPARAMETROS t-SNE
=====================

1. PERPLEXITY (5-50, tipico 30)
   - Numero efectivo de vecinos considerados
   - Bajo: estructura muy local (clusters pequenos)
   - Alto: estructura mas global (clusters grandes)

   Perplexity = 5:           Perplexity = 50:
     ●   ●   ●                  ●●●
    ● ● ● ● ●                  ●●●●●
     ●   ●   ●                  ●●●
   (muy fragmentado)         (clusters unidos)


2. LEARNING_RATE (10-1000, tipico 200)
   - Paso de optimizacion
   - Muy bajo: convergencia lenta
   - Muy alto: clusters pueden colapsar

3. N_ITER (250-5000, tipico 1000)
   - Iteraciones de optimizacion
   - Mas = mejor (hasta converger)

4. EARLY_EXAGGERATION (default 12)
   - Amplifica diferencias al inicio
   - Ayuda a separar clusters


REGLA PRACTICA:
    perplexity ≈ n_samples^0.5 / 3
    Pero siempre probar varios valores!
```

### Implementacion

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List


def tsne_visualization(
    X: np.ndarray,
    labels: np.ndarray | None = None,
    perplexities: List[int] = [5, 30, 50],
    n_iter: int = 1000,
    random_state: int = 42
) -> dict:
    """
    Aplica t-SNE con diferentes perplexities para comparar.

    Args:
        X: Datos [n_samples, n_features]
        labels: Etiquetas para colorear
        perplexities: Lista de valores de perplexity
        n_iter: Iteraciones

    Returns:
        Diccionario con embeddings
    """
    results = {}

    for perp in perplexities:
        print(f"Ejecutando t-SNE con perplexity={perp}...")

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=n_iter,
            learning_rate='auto',
            init='pca',
            random_state=random_state,
            n_jobs=-1
        )

        embedding = tsne.fit_transform(X)

        results[perp] = {
            'embedding': embedding,
            'kl_divergence': tsne.kl_divergence_
        }

    return results


def plot_tsne_comparison(
    results: dict,
    labels: np.ndarray | None = None,
    label_names: dict | None = None,
    figsize: tuple = (15, 5)
):
    """
    Grafica comparacion de t-SNE con diferentes perplexities.
    """
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, (perp, data) in zip(axes, results.items()):
        embedding = data['embedding']

        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=10
            )
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.6,
                s=10
            )

        ax.set_title(f'Perplexity = {perp}\nKL = {data["kl_divergence"]:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('tsne_comparison.png', dpi=150)
    print("Grafico guardado: tsne_comparison.png")


class TSNEAnalyzer:
    """
    Analizador t-SNE con utilidades adicionales.
    """

    def __init__(
        self,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42
    ):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding_ = None
        self.tsne_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta t-SNE y retorna embedding."""
        self.tsne_ = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            learning_rate='auto',
            init='pca',
            random_state=self.random_state
        )

        self.embedding_ = self.tsne_.fit_transform(X)
        return self.embedding_

    def find_optimal_perplexity(
        self,
        X: np.ndarray,
        perplexity_range: List[int] = [5, 10, 20, 30, 50, 100]
    ) -> int:
        """
        Encuentra perplexity optimo por KL divergence.
        (Menor KL = mejor fit)
        """
        best_perp = perplexity_range[0]
        best_kl = float('inf')

        for perp in perplexity_range:
            if perp >= X.shape[0]:
                continue

            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                n_iter=500,  # Menos iter para busqueda
                random_state=self.random_state
            )
            tsne.fit_transform(X)

            if tsne.kl_divergence_ < best_kl:
                best_kl = tsne.kl_divergence_
                best_perp = perp

            print(f"Perplexity={perp}: KL={tsne.kl_divergence_:.4f}")

        print(f"\nMejor perplexity: {best_perp}")
        return best_perp
```

### Limitaciones de t-SNE

```
LIMITACIONES t-SNE
==================

1. NO PRESERVA DISTANCIAS GLOBALES
   - Distancias entre clusters no son significativas
   - Solo vecindarios locales son confiables

   Cluster A ●●●     ●●● Cluster B
              ↑       ↑
         Esta distancia NO significa nada!


2. NO ES DETERMINISTICO
   - Diferentes runs = diferentes resultados
   - Usar random_state para reproducibilidad


3. SOLO PARA VISUALIZACION
   - No tiene transform() para nuevos datos
   - No usar para reduccion antes de ML


4. COMPUTACIONALMENTE COSTOSO
   - O(n²) en memoria
   - O(n² log n) en tiempo
   - Problematico para n > 10,000


5. SENSIBLE A HIPERPARAMETROS
   - Perplexity cambia drasticamente el resultado
   - Siempre probar multiples valores
```

---

## UMAP (Uniform Manifold Approximation and Projection)

### Concepto

```
UMAP: IDEA CENTRAL
==================

Similar a t-SNE pero basado en topologia:

1. Construir grafo de vecinos en alta dimension
   - Conectar cada punto con sus k vecinos mas cercanos
   - Pesos = similitud (fuzzy set membership)

2. Construir grafo similar en baja dimension
   - Optimizar para que grafos sean similares

3. Minimizar diferencia (cross-entropy)


DIFERENCIAS CON t-SNE:
─────────────────────

t-SNE:                          UMAP:
- Probabilidades gaussianas     - Membership fuzzy sets
- t-Student en baja dim         - Similar en baja dim
- Solo vecindarios locales      - Equilibra local/global
- Lento O(n²)                   - Rapido O(n^1.14)
- Solo visualizacion            - Puede hacer transform()
```

### Hiperparametros

```
HIPERPARAMETROS UMAP
====================

1. N_NEIGHBORS (5-50, tipico 15)
   - Numero de vecinos para construir grafo
   - Bajo: estructura muy local
   - Alto: estructura mas global
   - Similar a perplexity en t-SNE

2. MIN_DIST (0.0-1.0, tipico 0.1)
   - Distancia minima entre puntos en embedding
   - Bajo: clusters muy compactos
   - Alto: puntos mas dispersos

   min_dist = 0.0:              min_dist = 0.5:
       ●●●                         ●  ●  ●
       ●●●                        ●  ●  ●
       ●●●                         ●  ●  ●
   (muy compacto)               (mas disperso)

3. N_COMPONENTS (tipico 2)
   - Dimensiones del embedding
   - UMAP funciona bien con 2, 3, o mas

4. METRIC (default 'euclidean')
   - Distancia para calcular vecinos
   - Opciones: 'cosine', 'manhattan', 'correlation', etc.


REGLA PRACTICA:
    n_neighbors ≈ sqrt(n_samples)
    min_dist = 0.1 para clusters
    min_dist = 0.5 para ver distribucion
```

### Implementacion

```python
import numpy as np
import umap
import matplotlib.pyplot as plt
from typing import List


def umap_visualization(
    X: np.ndarray,
    labels: np.ndarray | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42
) -> dict:
    """
    Aplica UMAP para visualizacion.

    Args:
        X: Datos [n_samples, n_features]
        labels: Etiquetas
        n_neighbors: Vecinos para grafo
        min_dist: Distancia minima en embedding
        metric: Metrica de distancia

    Returns:
        Diccionario con embedding y modelo
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    embedding = reducer.fit_transform(X)

    return {
        'embedding': embedding,
        'model': reducer
    }


def compare_umap_params(
    X: np.ndarray,
    labels: np.ndarray | None = None,
    n_neighbors_list: List[int] = [5, 15, 50],
    min_dist_list: List[float] = [0.0, 0.1, 0.5]
) -> dict:
    """
    Compara UMAP con diferentes parametros.
    """
    results = {}

    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            key = f"nn={n_neighbors}_md={min_dist}"
            print(f"Ejecutando UMAP: {key}...")

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )

            embedding = reducer.fit_transform(X)
            results[key] = embedding

    return results


def plot_umap_grid(
    results: dict,
    labels: np.ndarray | None = None,
    figsize: tuple = (15, 10)
):
    """
    Grafica grid de resultados UMAP.
    """
    n_plots = len(results)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (key, embedding) in zip(axes, results.items()):
        if labels is not None:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=10
            )
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.6,
                s=10
            )

        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])

    # Ocultar axes vacios
    for ax in axes[len(results):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig('umap_grid.png', dpi=150)
    print("Grafico guardado: umap_grid.png")


class UMAPAnalyzer:
    """
    Analizador UMAP con capacidad de transformar nuevos datos.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = 'euclidean',
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state

        self.model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )

    def fit(self, X: np.ndarray) -> 'UMAPAnalyzer':
        """Ajusta UMAP."""
        self.model.fit(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta y transforma."""
        return self.model.fit_transform(X)

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transforma NUEVOS datos al embedding existente.
        (t-SNE no puede hacer esto!)
        """
        return self.model.transform(X_new)

    def inverse_transform(self, X_embedded: np.ndarray) -> np.ndarray:
        """
        Aproxima datos originales desde embedding.
        (Util para interpretacion)
        """
        return self.model.inverse_transform(X_embedded)
```

### Ventajas de UMAP sobre t-SNE

```
UMAP vs t-SNE
=============

                        t-SNE           UMAP
                        ─────           ────
Velocidad               Lento O(n²)     Rapido O(n^1.14)
Escalabilidad          <10K puntos      >100K puntos
Transform nuevos       NO              SI
Preserva global        NO              Parcialmente
Reproducibilidad       Baja            Alta
Parametros sensibles   Muy             Moderadamente
Uso para ML            No recomendado  Posible (con cuidado)
```

---

## Comparativa Practica

### Codigo de Comparacion

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import time


def compare_methods(
    X: np.ndarray,
    labels: np.ndarray | None = None,
    figsize: tuple = (15, 5)
):
    """
    Compara PCA, t-SNE y UMAP.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    methods = [
        ('PCA', PCA(n_components=2)),
        ('t-SNE', TSNE(n_components=2, perplexity=30, random_state=42)),
        ('UMAP', umap.UMAP(n_components=2, random_state=42))
    ]

    for ax, (name, method) in zip(axes, methods):
        start = time.time()
        embedding = method.fit_transform(X)
        elapsed = time.time() - start

        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap='tab10',
                alpha=0.6,
                s=10
            )
        else:
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                alpha=0.6,
                s=10
            )

        ax.set_title(f'{name}\n({elapsed:.2f}s)')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150)
    print("Grafico guardado: method_comparison.png")


def benchmark_scalability():
    """
    Benchmark de escalabilidad de cada metodo.
    """
    np.random.seed(42)

    sizes = [100, 500, 1000, 2000, 5000]
    n_features = 50

    results = {method: [] for method in ['PCA', 't-SNE', 'UMAP']}

    for n in sizes:
        X = np.random.randn(n, n_features)
        print(f"\nn = {n}")

        # PCA
        start = time.time()
        PCA(n_components=2).fit_transform(X)
        results['PCA'].append(time.time() - start)
        print(f"  PCA: {results['PCA'][-1]:.3f}s")

        # t-SNE
        if n <= 2000:  # t-SNE es muy lento para n grande
            start = time.time()
            TSNE(n_components=2, perplexity=min(30, n-1)).fit_transform(X)
            results['t-SNE'].append(time.time() - start)
            print(f"  t-SNE: {results['t-SNE'][-1]:.3f}s")
        else:
            results['t-SNE'].append(None)
            print(f"  t-SNE: SKIPPED (too slow)")

        # UMAP
        start = time.time()
        umap.UMAP(n_components=2).fit_transform(X)
        results['UMAP'].append(time.time() - start)
        print(f"  UMAP: {results['UMAP'][-1]:.3f}s")

    return sizes, results
```

---

## Aplicacion: Visualizacion de Malware

```python
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class MalwareVisualizer:
    """
    Visualiza familias de malware usando t-SNE y UMAP.
    """

    def __init__(self, method: str = 'umap'):
        """
        Args:
            method: 'tsne' o 'umap'
        """
        self.method = method
        self.scaler = StandardScaler()
        self.reducer = None
        self.embedding_ = None

    def fit_transform(
        self,
        X: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce dimensiones para visualizacion.
        """
        # Estandarizar
        X_scaled = self.scaler.fit_transform(X)

        if self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=2,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'euclidean'),
                random_state=42
            )
        else:  # tsne
            self.reducer = TSNE(
                n_components=2,
                perplexity=kwargs.get('perplexity', 30),
                n_iter=kwargs.get('n_iter', 1000),
                random_state=42
            )

        self.embedding_ = self.reducer.fit_transform(X_scaled)
        return self.embedding_

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """Transforma nuevos datos (solo UMAP)."""
        if self.method != 'umap':
            raise ValueError("transform() solo disponible para UMAP")

        X_scaled = self.scaler.transform(X_new)
        return self.reducer.transform(X_scaled)

    def plot_families(
        self,
        labels: np.ndarray,
        family_names: dict | None = None,
        title: str = "Malware Families",
        highlight_samples: np.ndarray | None = None,
        figsize: tuple = (12, 10)
    ):
        """
        Visualiza familias de malware.

        Args:
            labels: Array de etiquetas de familia
            family_names: Diccionario {label: nombre}
            title: Titulo del grafico
            highlight_samples: Indices de muestras a resaltar
        """
        if self.embedding_ is None:
            raise ValueError("Ejecuta fit_transform primero")

        plt.figure(figsize=figsize)

        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            name = family_names.get(label, f'Family {label}') if family_names else f'Family {label}'

            plt.scatter(
                self.embedding_[mask, 0],
                self.embedding_[mask, 1],
                c=[color],
                label=name,
                alpha=0.6,
                s=30
            )

        # Resaltar muestras especificas
        if highlight_samples is not None:
            plt.scatter(
                self.embedding_[highlight_samples, 0],
                self.embedding_[highlight_samples, 1],
                c='red',
                marker='*',
                s=200,
                label='Highlighted',
                edgecolors='black',
                linewidths=1
            )

        plt.title(title, fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        filename = f'malware_{self.method}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Grafico guardado: {filename}")

    def find_similar_samples(
        self,
        sample_idx: int,
        n_neighbors: int = 10
    ) -> np.ndarray:
        """
        Encuentra muestras similares en el espacio de embedding.
        """
        if self.embedding_ is None:
            raise ValueError("Ejecuta fit_transform primero")

        sample = self.embedding_[sample_idx]
        distances = np.sqrt(np.sum((self.embedding_ - sample) ** 2, axis=1))

        # Excluir la muestra misma
        distances[sample_idx] = np.inf

        nearest_indices = np.argsort(distances)[:n_neighbors]
        return nearest_indices


# Demo
def demo_malware_visualization():
    """Demo de visualizacion de malware."""
    np.random.seed(42)

    # Simular 5 familias de malware
    n_per_family = 200
    n_features = 100

    families = []
    labels = []

    for i in range(5):
        # Cada familia tiene caracteristicas diferentes
        family_data = np.random.randn(n_per_family, n_features)
        # Anadir offset distintivo
        family_data[:, i*20:(i+1)*20] += 3

        families.append(family_data)
        labels.extend([i] * n_per_family)

    X = np.vstack(families)
    labels = np.array(labels)

    family_names = {
        0: 'Ransomware',
        1: 'Trojan',
        2: 'Worm',
        3: 'Spyware',
        4: 'Adware'
    }

    print("=== Malware Visualization Demo ===\n")

    # UMAP
    print("Generando visualizacion UMAP...")
    viz_umap = MalwareVisualizer(method='umap')
    viz_umap.fit_transform(X, n_neighbors=15, min_dist=0.1)
    viz_umap.plot_families(labels, family_names, title="Malware Families (UMAP)")

    # t-SNE
    print("Generando visualizacion t-SNE...")
    viz_tsne = MalwareVisualizer(method='tsne')
    viz_tsne.fit_transform(X, perplexity=30)
    viz_tsne.plot_families(labels, family_names, title="Malware Families (t-SNE)")

    # Encontrar muestras similares
    print("\nMuestras similares al sample 0 (UMAP):")
    similar = viz_umap.find_similar_samples(0, n_neighbors=5)
    for idx in similar:
        print(f"  Sample {idx}: {family_names[labels[idx]]}")

    return viz_umap, viz_tsne


if __name__ == "__main__":
    demo_malware_visualization()
```

---

## Mejores Practicas

```
MEJORES PRACTICAS t-SNE / UMAP
==============================

1. PREPROCESAMIENTO
   [ ] Estandarizar datos (StandardScaler)
   [ ] Reducir con PCA primero si dim > 50-100
   [ ] Eliminar outliers extremos

2. SELECCION DE PARAMETROS
   [ ] Probar multiples valores de perplexity/n_neighbors
   [ ] Usar min_dist bajo (0.0-0.1) para clusters compactos
   [ ] Aumentar n_iter si no converge

3. INTERPRETACION
   [ ] NO interpretar distancias entre clusters
   [ ] Solo confiar en vecindarios locales
   [ ] Ejecutar multiples veces para verificar estabilidad

4. ESCALABILIDAD
   [ ] t-SNE: < 10,000 puntos
   [ ] UMAP: < 100,000 puntos
   [ ] PCA primero para datasets muy grandes

5. PARA ML
   [ ] No usar t-SNE para features de modelo
   [ ] UMAP es mejor opcion si necesitas transform()
   [ ] Pero aun asi, validar con cross-validation
```

---

## Resumen

```
CUANDO USAR CADA METODO
=======================

PCA:
├── Reduccion para ML (input de otro modelo)
├── Analisis de varianza
├── Interpretacion de features
└── Siempre como baseline

t-SNE:
├── Visualizacion de clusters
├── Explorar estructura local
├── Datasets < 10K puntos
└── Cuando tiempo no importa

UMAP:
├── Visualizacion (mas rapido que t-SNE)
├── Datasets grandes (10K-100K)
├── Necesitas transform() para nuevos datos
├── Preservar algo de estructura global
└── Embedding para clustering posterior


CHECKLIST VISUALIZACION
=======================

1. [ ] Estandarizar datos
2. [ ] Si dim > 50: PCA primero a ~50
3. [ ] Probar t-SNE Y UMAP
4. [ ] Variar hiperparametros
5. [ ] Colorear por labels conocidos
6. [ ] NO interpretar distancias globales
7. [ ] Verificar estabilidad (multiples runs)
```

### Puntos Clave

1. **t-SNE** y **UMAP** son para **visualizacion**, no para ML
2. **Perplexity/n_neighbors** controla estructura local vs global
3. **No interpretar distancias entre clusters**
4. **UMAP** es mas rapido y puede transformar nuevos datos
5. **Siempre estandarizar** antes de aplicar
6. **PCA primero** si hay muchas dimensiones
7. En ciberseguridad: visualizar familias de malware, clusters de ataques