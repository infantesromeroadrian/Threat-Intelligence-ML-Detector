# DBSCAN (Density-Based Spatial Clustering)

## 1. ¿Qué es DBSCAN?

### Concepto: Clustering Basado en Densidad

```
┌────────────────────────────────────────────────────────────────┐
│  DBSCAN = Density-Based Spatial Clustering of Applications     │
│           with Noise                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA CENTRAL:                                                 │
│    Clusters = Regiones de ALTA DENSIDAD de puntos              │
│    separadas por regiones de BAJA DENSIDAD                     │
│                                                                │
│       ●●●●●●●                                                  │
│      ●●●●●●●●           Cluster 1 (alta densidad)              │
│       ●●●●●●                                                   │
│                                                                │
│              ✗ ← Outlier (baja densidad)                       │
│                                                                │
│                    ○○○○○                                       │
│                   ○○○○○○○      Cluster 2 (alta densidad)       │
│                    ○○○○○                                       │
│                                                                │
│  VENTAJAS CLAVE:                                               │
│    • NO requiere especificar número de clusters (K)            │
│    • Detecta clusters de CUALQUIER FORMA                       │
│    • Identifica automáticamente OUTLIERS                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### DBSCAN vs K-Means

```
┌─────────────────────────────────────────────────────────────────┐
│  K-MEANS                          DBSCAN                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Solo clusters esféricos          Cualquier forma               │
│                                                                 │
│      ┌───┐    ┌───┐               ●●●●●●●●●●                    │
│      │●●●│    │○○○│               ●●●●●●●●●                     │
│      │●●●│    │○○○│                                             │
│      └───┘    └───┘                    ○○○○○○                   │
│                                       ○○○○○○○○                  │
│                                        ○○○○○                    │
│                                                                 │
│  Requiere K                       No requiere K                 │
│                                   (lo descubre solo)            │
│                                                                 │
│  No detecta outliers              Detecta outliers              │
│  (todos pertenecen a un cluster)  (etiqueta como -1)            │
│                                                                 │
│        ●●●                              ●●●                     │
│       ●●★●● ← centroide                ●●●●●                    │
│        ●●●     alejado                  ●●●                     │
│              ✗                               ✗ ← outlier        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Conceptos Fundamentales

### Parámetros de DBSCAN

```
┌────────────────────────────────────────────────────────────────┐
│  DOS PARÁMETROS CLAVE                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ε (eps):  Radio de vecindad                                   │
│  ─────────────────────────────                                 │
│  "¿Qué tan lejos puede estar un punto para ser vecino?"        │
│                                                                │
│            ┌─────┐                                             │
│            │  ●  │ ← puntos dentro del círculo                 │
│       ●    │ ●●● │   son vecinos                               │
│            │●★●● │                                             │
│            │  ●  │   ★ = punto central                         │
│            └─────┘   radio = ε                                 │
│                                                                │
│  min_samples:  Mínimo de vecinos para ser "core point"         │
│  ──────────────────────────────────────────────────            │
│  "¿Cuántos vecinos necesita un punto para formar un cluster?"  │
│                                                                │
│  Si min_samples = 4:                                           │
│                                                                │
│     ●●●●●  → 5 vecinos → ES core point ✓                       │
│       ★                                                        │
│                                                                │
│     ●●     → 2 vecinos → NO es core point ✗                    │
│      ★                                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Tipos de Puntos

```
┌────────────────────────────────────────────────────────────────┐
│  TRES TIPOS DE PUNTOS                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. CORE POINT (Punto núcleo)                                  │
│     ─────────────────────────                                  │
│     • Tiene ≥ min_samples vecinos dentro de ε                  │
│     • Forma el "núcleo" del cluster                            │
│                                                                │
│         ●●●●●                                                  │
│        ●●★●●●  ← ★ es core point (muchos vecinos)              │
│         ●●●●                                                   │
│                                                                │
│  2. BORDER POINT (Punto frontera)                              │
│     ─────────────────────────────                              │
│     • NO tiene min_samples vecinos                             │
│     • PERO está dentro de ε de un core point                   │
│     • Pertenece al cluster pero en el borde                    │
│                                                                │
│         ●●●●                                                   │
│        ●●★●●   ○ ← ○ es border point (vecino de core)          │
│         ●●●                                                    │
│                                                                │
│  3. NOISE POINT (Outlier)                                      │
│     ────────────────────                                       │
│     • NO tiene min_samples vecinos                             │
│     • NO está cerca de ningún core point                       │
│     • Se etiqueta como -1 (no pertenece a ningún cluster)      │
│                                                                │
│         ●●●●                                                   │
│        ●●★●●            ✗ ← ✗ es noise (muy lejos)             │
│         ●●●                                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización de Tipos

```
           CLUSTER                           OUTLIERS
    ┌─────────────────────┐
    │    ●●●●●●●●         │
    │   ●●●●●●●●●●   ○    │                    ✗
    │    ●●●●●●●●    ↑    │
    │   ●●●●●●●●●●  border│        ✗
    │    ●●●●●●●●         │
    │        ↑            │                         ✗
    │      core           │
    └─────────────────────┘

    ● = Core points (muchos vecinos)
    ○ = Border points (pocos vecinos pero cerca de core)
    ✗ = Noise/Outliers (aislados)
```

## 3. Algoritmo DBSCAN

### Pasos del Algoritmo

```
┌────────────────────────────────────────────────────────────────┐
│  ALGORITMO DBSCAN                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Para cada punto P no visitado:                             │
│     a. Marcar P como visitado                                  │
│     b. Encontrar vecinos de P (puntos a distancia ≤ ε)         │
│     c. Si |vecinos| < min_samples:                             │
│        → Marcar P como NOISE (temporalmente)                   │
│     d. Si |vecinos| ≥ min_samples:                             │
│        → P es CORE POINT                                       │
│        → Crear nuevo cluster C                                 │
│        → Añadir P a C                                          │
│        → Expandir cluster (paso 2)                             │
│                                                                │
│  2. EXPANDIR CLUSTER:                                          │
│     Para cada vecino V de P:                                   │
│     a. Si V no visitado:                                       │
│        → Marcar V como visitado                                │
│        → Si V es core point, añadir sus vecinos                │
│     b. Si V no pertenece a ningún cluster:                     │
│        → Añadir V al cluster C                                 │
│                                                                │
│  3. Los puntos que quedan como NOISE son outliers              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización del Proceso

```
PASO 1: Identificar core points (min_samples=4, ε=radio del círculo)
───────────────────────────────────────────────────────────────────

    ●  ●   ●                    ★  ★   ★
      ●  ●    ●        →         ★  ★    ○    (★ = core, ○ = border)
    ●   ●  ●                    ★   ★  ★

              ✗                           ✗   (✗ = noise)


PASO 2: Expandir desde core points (conectar clusters)
───────────────────────────────────────────────────────

    ★──★───★                   ┌─────────────┐
      ╲ ╱                      │ ★──★───★    │
       ★──★────○        →      │   ╲ ╱       │  = UN cluster
      ╱                        │    ★──★────○│
    ★───★──★                   │   ╱         │
                               │ ★───★──★    │
              ✗                └─────────────┘
                                         ✗ (sigue siendo noise)


RESULTADO FINAL:
────────────────

    Cluster 0: todos los puntos conectados
    Noise (-1): puntos aislados
```

## 4. Implementación en Python

### Código Básico

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generar datos con diferentes formas
np.random.seed(42)

# Dos clusters con forma de media luna
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.05)

# Añadir outliers
outliers = np.random.uniform(-2, 3, (20, 2))
X = np.vstack([X, outliers])

# IMPORTANTE: Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(
    eps=0.3,          # Radio de vecindad
    min_samples=5,    # Mínimo de vecinos para core point
    metric='euclidean'
)

labels = dbscan.fit_predict(X_scaled)

# Resultados
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"Clusters encontrados: {n_clusters}")
print(f"Puntos de ruido (outliers): {n_noise}")
print(f"Core samples: {len(dbscan.core_sample_indices_)}")
```

### Visualización

```python
import matplotlib.pyplot as plt

def plot_dbscan(X, labels, title="DBSCAN Clustering"):
    plt.figure(figsize=(12, 8))

    # Colores: -1 (noise) en negro
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Outliers en negro con X
            color = 'black'
            marker = 'x'
            size = 50
            alpha = 0.6
        else:
            marker = 'o'
            size = 50
            alpha = 0.8

        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=[color], marker=marker, s=size,
                   alpha=alpha, label=f'Cluster {label}' if label != -1 else 'Noise')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_dbscan(X_scaled, labels)
```

## 5. Elegir los Parámetros

### El Problema

```
┌────────────────────────────────────────────────────────────────┐
│  ELEGIR eps Y min_samples ES CRÍTICO                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  eps MUY PEQUEÑO:              eps MUY GRANDE:                 │
│                                                                │
│    ●  ●  ●  ●                     ●●●●●●●●●●                   │
│    ●  ●  ●  ●                    ●●●●●●●●●●●                   │
│    ●  ●  ●  ●                     ●●●●●●●●●                    │
│                                                                │
│   Muchos clusters pequeños        Un solo cluster gigante      │
│   o todo es noise                                              │
│                                                                │
│  min_samples MUY BAJO:         min_samples MUY ALTO:           │
│                                                                │
│    Todo es cluster              Todo es noise                  │
│    (incluso outliers)           (nada tiene suficientes        │
│                                  vecinos)                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Método del K-Distance Graph

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTODO K-DISTANCE PARA ELEGIR eps                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Elegir k = min_samples (o min_samples - 1)                 │
│                                                                │
│  2. Para cada punto, calcular distancia al k-ésimo vecino      │
│                                                                │
│  3. Ordenar distancias de menor a mayor y graficar             │
│                                                                │
│  4. Buscar el "codo" en el gráfico                             │
│                                                                │
│  k-distance                                                    │
│     │                                                          │
│     │                               ╱                          │
│     │                          ╱───╱                           │
│     │                     ____╱                                │
│     │               _____╱                                     │
│     │          ____╱  ← CODO (eps óptimo)                      │
│     │_________╱                                                │
│     └──────────────────────────────── puntos (ordenados)       │
│                                                                │
│  El codo indica donde la densidad cambia significativamente    │
│  eps ≈ valor de k-distance en el codo                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código: Encontrar eps Óptimo

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

def find_optimal_eps(X, min_samples=5):
    """Encuentra eps óptimo usando k-distance graph"""

    # Calcular distancias al k-ésimo vecino
    k = min_samples
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)

    # Distancia al k-ésimo vecino (última columna)
    k_distances = distances[:, -1]

    # Ordenar
    k_distances_sorted = np.sort(k_distances)

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-')
    plt.xlabel('Puntos ordenados')
    plt.ylabel(f'Distancia al {k}-ésimo vecino')
    plt.title('K-Distance Graph para elegir eps')
    plt.grid(True, alpha=0.3)

    # Añadir línea horizontal sugerida (heurística)
    # Buscar el punto de máxima curvatura
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(k_distances_sorted, sigma=len(k_distances)//50)
    second_derivative = np.diff(np.diff(smoothed))
    elbow_idx = np.argmax(second_derivative) + 2
    suggested_eps = k_distances_sorted[elbow_idx]

    plt.axhline(y=suggested_eps, color='r', linestyle='--',
                label=f'eps sugerido ≈ {suggested_eps:.3f}')
    plt.legend()
    plt.show()

    return suggested_eps

# Uso
optimal_eps = find_optimal_eps(X_scaled, min_samples=5)
print(f"eps sugerido: {optimal_eps:.3f}")
```

### Guía para min_samples

```
┌────────────────────────────────────────────────────────────────┐
│  REGLAS PARA min_samples                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  REGLA GENERAL:                                                │
│    min_samples ≥ dimensiones + 1                               │
│    (mínimo para definir un hiperplano)                         │
│                                                                │
│  REGLAS PRÁCTICAS:                                             │
│                                                                │
│    • Datos 2D: min_samples = 4-5                               │
│    • Datos alta dimensión: min_samples = 2 × dim               │
│    • Datos con mucho ruido: aumentar min_samples               │
│    • Clusters pequeños esperados: reducir min_samples          │
│                                                                │
│  EJEMPLO:                                                      │
│    Dataset con 10 features:                                    │
│    min_samples = 10 + 1 = 11 (mínimo)                          │
│    min_samples = 2 × 10 = 20 (conservador)                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Búsqueda de Parámetros

```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def search_dbscan_params(X, eps_range, min_samples_range):
    """Busca mejores parámetros para DBSCAN"""

    best_score = -1
    best_params = None
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            # Ignorar si todo es noise o un solo cluster
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            if n_clusters >= 2 and n_noise < len(X) * 0.5:
                # Calcular silhouette solo para puntos no-noise
                mask = labels != -1
                if mask.sum() > 0:
                    score = silhouette_score(X[mask], labels[mask])
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)

    return best_params, results

# Uso
eps_range = np.arange(0.1, 1.0, 0.1)
min_samples_range = range(3, 10)

best_params, results = search_dbscan_params(X_scaled, eps_range, min_samples_range)
print(f"Mejores parámetros: eps={best_params[0]:.2f}, min_samples={best_params[1]}")
```

## 6. HDBSCAN: Versión Mejorada

### Limitaciones de DBSCAN

```
┌────────────────────────────────────────────────────────────────┐
│  PROBLEMA DE DBSCAN: Clusters con diferente densidad           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Datos con densidad variable:                                  │
│                                                                │
│      ●●●●●●●●                      ○  ○                        │
│     ●●●●●●●●●●                    ○    ○                       │
│      ●●●●●●●●                      ○  ○                        │
│     ●●●●●●●●●●                                                 │
│      ●●●●●●●●                                                  │
│                                                                │
│    Cluster DENSO                Cluster DISPERSO               │
│                                                                │
│  Con un solo eps:                                              │
│    - eps pequeño: detecta denso, disperso es todo noise        │
│    - eps grande: une ambos en un solo cluster                  │
│                                                                │
│  SOLUCIÓN: HDBSCAN (Hierarchical DBSCAN)                       │
│    - No requiere eps fijo                                      │
│    - Maneja clusters de diferente densidad                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### HDBSCAN en Python

```python
# pip install hdbscan
import hdbscan

# HDBSCAN no requiere eps
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,     # Tamaño mínimo de cluster
    min_samples=3,          # Como DBSCAN
    cluster_selection_epsilon=0.0,
    metric='euclidean'
)

labels = clusterer.fit_predict(X_scaled)

# HDBSCAN proporciona probabilidades de pertenencia
probabilities = clusterer.probabilities_

print(f"Clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Noise: {(labels == -1).sum()}")

# Visualizar con intensidad por probabilidad
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                     c=labels, cmap='viridis',
                     alpha=probabilities)
plt.colorbar(scatter)
plt.title('HDBSCAN Clustering')
plt.show()
```

## 7. Ejemplo Práctico: Detección de Anomalías en Red

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Simular tráfico de red
np.random.seed(42)

# Tráfico normal (cluster grande y denso)
normal = np.column_stack([
    np.random.normal(100, 20, 800),     # bytes
    np.random.normal(5, 1, 800),        # paquetes/seg
    np.random.normal(443, 10, 800),     # puerto (HTTPS)
])

# Tráfico de backup (cluster pequeño, diferente patrón)
backup = np.column_stack([
    np.random.normal(5000, 500, 100),   # muchos bytes
    np.random.normal(50, 10, 100),      # muchos paquetes
    np.random.normal(22, 1, 100),       # SSH
])

# Anomalías (outliers dispersos)
anomalias = np.column_stack([
    np.random.uniform(0, 10000, 30),    # bytes aleatorios
    np.random.uniform(0, 100, 30),      # paquetes aleatorios
    np.random.uniform(1, 65535, 30),    # puertos aleatorios
])

# Combinar
X = np.vstack([normal, backup, anomalias])
tipos = ['Normal']*800 + ['Backup']*100 + ['Anomalía']*30

print(f"Total conexiones: {len(X)}")
print(f"  Normal: 800")
print(f"  Backup: 100")
print(f"  Anomalías: 30")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar eps óptimo
print("\nBuscando eps óptimo...")
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

# Visualizar k-distance
plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.xlabel('Puntos ordenados')
plt.ylabel(f'Distancia al {k}-ésimo vecino')
plt.title('K-Distance Graph')
plt.grid(True, alpha=0.3)
plt.show()

# DBSCAN
eps = 0.5  # Ajustar según el gráfico
dbscan = DBSCAN(eps=eps, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Resultados
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()

print(f"\nResultados DBSCAN (eps={eps}):")
print(f"  Clusters detectados: {n_clusters}")
print(f"  Puntos de ruido (anomalías): {n_noise}")

# Analizar
df = pd.DataFrame(X, columns=['bytes', 'paquetes', 'puerto'])
df['cluster'] = labels
df['tipo_real'] = tipos

print("\n" + "=" * 60)
print("ANÁLISIS DE DETECCIÓN DE ANOMALÍAS")
print("=" * 60)

# Análisis por cluster
for cluster in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cluster
    n = mask.sum()

    if cluster == -1:
        print(f"\n🚨 ANOMALÍAS DETECTADAS ({n} conexiones):")
    else:
        print(f"\nCluster {cluster} ({n} conexiones):")

    # Estadísticas
    print(f"  Bytes promedio: {df[mask]['bytes'].mean():,.0f}")
    print(f"  Paquetes/seg: {df[mask]['paquetes'].mean():.1f}")
    print(f"  Puerto más común: {df[mask]['puerto'].mode().values[0]:.0f}")

    # Composición real
    print(f"  Tipos reales:")
    for tipo in df[mask]['tipo_real'].value_counts().items():
        print(f"    - {tipo[0]}: {tipo[1]} ({tipo[1]/n*100:.1f}%)")

# Métricas de detección
anomalias_detectadas = ((df['cluster'] == -1) & (df['tipo_real'] == 'Anomalía')).sum()
anomalias_totales = (df['tipo_real'] == 'Anomalía').sum()
falsos_positivos = ((df['cluster'] == -1) & (df['tipo_real'] != 'Anomalía')).sum()

print("\n" + "=" * 60)
print("MÉTRICAS DE DETECCIÓN")
print("=" * 60)
print(f"Anomalías reales detectadas: {anomalias_detectadas}/{anomalias_totales} "
      f"({anomalias_detectadas/anomalias_totales*100:.1f}%)")
print(f"Falsos positivos: {falsos_positivos}")
print(f"Precision: {anomalias_detectadas/(anomalias_detectadas+falsos_positivos)*100:.1f}%"
      if (anomalias_detectadas+falsos_positivos) > 0 else "N/A")

# Visualización
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))

# Clusters normales
for cluster in sorted(set(labels)):
    if cluster == -1:
        continue
    mask = labels == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               alpha=0.6, s=50, label=f'Cluster {cluster}')

# Anomalías
mask = labels == -1
plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
           c='red', marker='x', s=100, linewidths=2,
           label=f'Anomalías ({mask.sum()})')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Detección de Anomalías con DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 8. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE DBSCAN                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ NO requiere especificar número de clusters                  │
│  ✓ Encuentra clusters de CUALQUIER forma                       │
│  ✓ Detecta OUTLIERS automáticamente                            │
│  ✓ Robusto al ruido                                            │
│  ✓ Solo dos parámetros (eps, min_samples)                      │
│  ✓ No asume distribución de datos                              │
│  ✓ Eficiente con índices espaciales (O(n log n))              │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE DBSCAN                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Sensible a la elección de eps y min_samples                 │
│  ✗ Mal desempeño con densidades muy diferentes                 │
│  ✗ No funciona bien en alta dimensión (curse of dim)           │
│  ✗ Lento para datasets muy grandes sin optimización            │
│  ✗ No asigna nuevos puntos (necesita re-entrenar)              │
│  ✗ Resultados pueden variar con orden de datos                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 9. Cuándo Usar DBSCAN

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Detección de anomalías / outliers                           │
│  ✓ No sabes cuántos clusters hay                               │
│  ✓ Clusters con formas irregulares                             │
│  ✓ Datos geoespaciales                                         │
│  ✓ Segmentación de imágenes                                    │
│  ✓ Agrupación de comportamientos en logs                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Clusters tienen densidades muy diferentes (usar HDBSCAN)    │
│  ✗ Datos de muy alta dimensión (>20 features)                  │
│  ✗ Necesitas asignar nuevos puntos frecuentemente              │
│  ✗ Clusters esféricos y sabes K (usar K-Means, más rápido)     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 10. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  DBSCAN - RESUMEN                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Clusters = regiones de alta densidad                        │
│    Outliers = puntos en regiones de baja densidad              │
│                                                                │
│  PARÁMETROS:                                                   │
│    • eps: radio de vecindad                                    │
│    • min_samples: mínimo de vecinos para core point            │
│                                                                │
│  TIPOS DE PUNTOS:                                              │
│    • Core: ≥ min_samples vecinos                               │
│    • Border: vecino de core pero < min_samples vecinos         │
│    • Noise: ni core ni border (etiqueta -1)                    │
│                                                                │
│  ELEGIR PARÁMETROS:                                            │
│    • eps: k-distance graph (buscar codo)                       │
│    • min_samples: dim + 1 o 2 × dim                            │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de anomalías en tráfico                         │
│    • Identificar comportamientos anómalos                      │
│    • Filtrar ruido en logs                                     │
│    • Agrupar eventos de seguridad                              │
│                                                                │
│  ALTERNATIVAS:                                                 │
│    • HDBSCAN: para densidades variables                        │
│    • OPTICS: para visualizar estructura de densidad            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Clustering Jerárquico
