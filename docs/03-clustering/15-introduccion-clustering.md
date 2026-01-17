# Introducción al Clustering

## 1. ¿Qué es Clustering?

### Aprendizaje No Supervisado

```
┌────────────────────────────────────────────────────────────────┐
│  SUPERVISADO vs NO SUPERVISADO                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  SUPERVISADO (Clasificación):                                  │
│    • Tenemos etiquetas (y)                                     │
│    • Objetivo: predecir etiqueta de nuevos datos               │
│    • Ejemplo: email → SPAM o HAM                               │
│                                                                │
│    Datos:  X₁ → y₁ (SPAM)                                      │
│            X₂ → y₂ (HAM)                                       │
│            X₃ → y₃ (SPAM)                                      │
│                                                                │
│  NO SUPERVISADO (Clustering):                                  │
│    • NO tenemos etiquetas                                      │
│    • Objetivo: descubrir ESTRUCTURA en los datos               │
│    • Ejemplo: agrupar malware similar sin saber familias       │
│                                                                │
│    Datos:  X₁ → ?                                              │
│            X₂ → ?     →  Algoritmo descubre grupos             │
│            X₃ → ?                                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Concepto de Clustering

```
┌────────────────────────────────────────────────────────────────┐
│  CLUSTERING = Agrupar datos similares                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ANTES (datos sin estructura):     DESPUÉS (clusters):         │
│                                                                │
│       ●  ○    ●                    ┌───────┐                   │
│    ○    ●  ○     ●                 │ ●  ●  │    ┌───────┐      │
│      ●    ○   ●                    │●    ● │    │ ○  ○  │      │
│    ○   ●    ○    ●                 │  ●  ● │    │○    ○ │      │
│       ○  ●    ○                    └───────┘    │  ○  ○ │      │
│                                    Cluster 1    └───────┘      │
│                                                 Cluster 2      │
│                                                                │
│  El algoritmo encuentra grupos de puntos SIMILARES             │
│  sin que nadie le diga cuántos grupos hay o cuáles son         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Aplicaciones en Ciberseguridad

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO EN CIBERSEGURIDAD                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DETECCIÓN DE ANOMALÍAS                                     │
│     • Tráfico normal forma clusters densos                     │
│     • Tráfico anómalo queda FUERA de los clusters              │
│     • Detectar intrusiones sin firmas conocidas                │
│                                                                │
│  2. AGRUPACIÓN DE MALWARE                                      │
│     • Agrupar muestras por comportamiento similar              │
│     • Descubrir nuevas familias de malware                     │
│     • Análisis de variantes                                    │
│                                                                │
│  3. SEGMENTACIÓN DE RED                                        │
│     • Identificar tipos de dispositivos                        │
│     • Agrupar patrones de uso                                  │
│     • Detectar dispositivos comprometidos                      │
│                                                                │
│  4. ANÁLISIS DE LOGS                                           │
│     • Agrupar eventos similares                                │
│     • Reducir ruido en alertas                                 │
│     • Identificar patrones de ataque                           │
│                                                                │
│  5. THREAT INTELLIGENCE                                        │
│     • Agrupar IOCs relacionados                                │
│     • Identificar campañas de ataque                           │
│     • Correlacionar amenazas                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. Tipos de Clustering

### Taxonomía

```
┌────────────────────────────────────────────────────────────────┐
│  TIPOS DE ALGORITMOS DE CLUSTERING                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. PARTITIONAL (Particional)                                  │
│     ───────────────────────                                    │
│     • Divide datos en K grupos disjuntos                       │
│     • K se especifica a priori                                 │
│     • Ejemplos: K-Means, K-Medoids                             │
│                                                                │
│  2. HIERARCHICAL (Jerárquico)                                  │
│     ──────────────────────────                                 │
│     • Crea jerarquía de clusters (árbol)                       │
│     • No requiere especificar K                                │
│     • Tipos: Aglomerativo (bottom-up), Divisivo (top-down)     │
│                                                                │
│  3. DENSITY-BASED (Basado en densidad)                         │
│     ─────────────────────────────────                          │
│     • Clusters = regiones de alta densidad                     │
│     • Detecta clusters de forma arbitraria                     │
│     • Ejemplos: DBSCAN, HDBSCAN, OPTICS                        │
│                                                                │
│  4. MODEL-BASED (Basado en modelo)                             │
│     ─────────────────────────────                              │
│     • Asume que datos vienen de distribuciones                 │
│     • Encuentra parámetros de las distribuciones               │
│     • Ejemplos: Gaussian Mixture Models (GMM)                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Comparación Visual

```
PARTICIONAL (K-Means):          Solo encuentra clusters esféricos
                                Requiere especificar K
    ┌─────┐     ┌─────┐
    │ ●●● │     │ ○○○ │
    │ ●●● │     │ ○○○ │
    └─────┘     └─────┘


JERÁRQUICO:                     Crea dendrograma
                                Flexible para elegir K después
         ┌──────────┐
      ┌──┴──┐    ┌──┴──┐
     ┌┴┐   ┌┴┐  ┌┴┐   ┌┴┐
     ●●    ●●   ○○    ○○


BASADO EN DENSIDAD (DBSCAN):    Encuentra formas arbitrarias
                                Detecta outliers
    ●●●●●●●●
     ●●●●●●●
      ●●●●●               ○
                      ○○○○○○
          ✗ (outlier)  ○○○○○○○


BASADO EN MODELO (GMM):         Clusters probabilísticos
                                Soft assignment
    ╭───────╮   ╭───────╮
   ╱  ●●●●   ╲ ╱  ○○○○   ╲
  │   ●●●●    ╳    ○○○○   │
   ╲  ●●●●   ╱ ╲  ○○○○   ╱
    ╰───────╯   ╰───────╯
       Puede pertenecer a ambos con probabilidad
```

## 3. Métricas de Distancia

### Distancias Comunes

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTRICAS DE DISTANCIA                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  La elección de distancia AFECTA los resultados                │
│                                                                │
│  1. EUCLIDIANA (L2)                                            │
│     d(x,y) = √(Σ(xᵢ - yᵢ)²)                                    │
│     • La más común                                             │
│     • "Línea recta" entre puntos                               │
│     • Sensible a escala                                        │
│                                                                │
│  2. MANHATTAN (L1)                                             │
│     d(x,y) = Σ|xᵢ - yᵢ|                                        │
│     • Suma de diferencias absolutas                            │
│     • Menos sensible a outliers                                │
│     • "Caminar por calles de ciudad"                           │
│                                                                │
│  3. COSENO                                                     │
│     d(x,y) = 1 - (x·y)/(||x|| ||y||)                          │
│     • Mide ángulo entre vectores                               │
│     • Ignora magnitud                                          │
│     • Ideal para texto (TF-IDF)                                │
│                                                                │
│  4. MINKOWSKI                                                  │
│     d(x,y) = (Σ|xᵢ - yᵢ|^p)^(1/p)                             │
│     • Generaliza Euclidiana (p=2) y Manhattan (p=1)            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización de Distancias

```
EUCLIDIANA vs MANHATTAN:

  y
  │      B
  │     ╱│
  │   ╱  │  Manhattan: |x₂-x₁| + |y₂-y₁| = 3 + 4 = 7
  │ ╱    │
  A──────┘  Euclidiana: √(3² + 4²) = 5
  │
  └─────────── x

COSENO (para texto):

  Documento 1: "seguridad red firewall" → [1, 1, 1, 0, 0]
  Documento 2: "seguridad red vpn"      → [1, 1, 0, 1, 0]
  Documento 3: "gato perro animal"      → [0, 0, 0, 0, 1]

  Coseno(Doc1, Doc2) = alto (similares)
  Coseno(Doc1, Doc3) = 0 (completamente diferentes)
```

### Código: Calcular Distancias

```python
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_distances
)
import numpy as np

# Dos puntos de ejemplo
X = np.array([[1, 2, 3]])
Y = np.array([[4, 5, 6]])

print("Distancia Euclidiana:", euclidean_distances(X, Y)[0, 0])
print("Distancia Manhattan:", manhattan_distances(X, Y)[0, 0])
print("Distancia Coseno:", cosine_distances(X, Y)[0, 0])

# Para clustering con distancia específica
from sklearn.cluster import KMeans, AgglomerativeClustering

# K-Means siempre usa Euclidiana
kmeans = KMeans(n_clusters=3)

# Jerárquico puede usar diferentes métricas
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    metric='manhattan',  # o 'euclidean', 'cosine'
    linkage='average'
)
```

## 4. Evaluación de Clustering

### Métricas Internas (sin etiquetas reales)

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTRICAS INTERNAS - No requieren ground truth                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SILHOUETTE SCORE                                           │
│     ────────────────                                           │
│     Mide qué tan bien cada punto encaja en su cluster          │
│                                                                │
│     s(i) = (b(i) - a(i)) / max(a(i), b(i))                    │
│                                                                │
│     a(i) = distancia promedio a puntos del MISMO cluster       │
│     b(i) = distancia promedio al cluster MÁS CERCANO           │
│                                                                │
│     Rango: [-1, 1]                                             │
│       1  = perfectamente asignado                              │
│       0  = en la frontera                                      │
│      -1  = probablemente mal asignado                          │
│                                                                │
│  2. CALINSKI-HARABASZ INDEX                                    │
│     ───────────────────────                                    │
│     Ratio de dispersión entre-clusters / dentro-clusters       │
│     Mayor = mejor                                              │
│                                                                │
│  3. DAVIES-BOULDIN INDEX                                       │
│     ─────────────────────                                      │
│     Promedio de similitud entre clusters                       │
│     Menor = mejor (clusters más separados)                     │
│                                                                │
│  4. INERTIA (para K-Means)                                     │
│     ──────                                                     │
│     Suma de distancias cuadradas al centroide                  │
│     Menor = clusters más compactos                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Métricas Externas (con etiquetas reales)

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTRICAS EXTERNAS - Requieren ground truth                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Si tienes etiquetas reales para comparar:                     │
│                                                                │
│  1. ADJUSTED RAND INDEX (ARI)                                  │
│     • Mide similitud entre clustering y etiquetas reales       │
│     • Rango: [-1, 1], 1 = perfecto                             │
│     • Ajustado por azar                                        │
│                                                                │
│  2. NORMALIZED MUTUAL INFORMATION (NMI)                        │
│     • Información compartida entre clusters y etiquetas        │
│     • Rango: [0, 1], 1 = perfecto                              │
│                                                                │
│  3. HOMOGENEITY                                                │
│     • Cada cluster contiene solo miembros de una clase         │
│                                                                │
│  4. COMPLETENESS                                               │
│     • Todos los miembros de una clase están en el mismo cluster│
│                                                                │
│  5. V-MEASURE                                                  │
│     • Media armónica de homogeneity y completeness             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código: Evaluar Clustering

```python
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans
import numpy as np

# Datos de ejemplo
X = np.random.randn(300, 5)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Métricas internas (no necesitan ground truth)
print("MÉTRICAS INTERNAS:")
print(f"  Silhouette Score: {silhouette_score(X, labels):.3f}")
print(f"  Calinski-Harabasz: {calinski_harabasz_score(X, labels):.1f}")
print(f"  Davies-Bouldin: {davies_bouldin_score(X, labels):.3f}")
print(f"  Inertia: {kmeans.inertia_:.1f}")

# Si tienes etiquetas reales (ej: para validación)
y_true = np.array([0]*100 + [1]*100 + [2]*100)  # Etiquetas simuladas

print("\nMÉTRICAS EXTERNAS:")
print(f"  Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.3f}")
print(f"  NMI: {normalized_mutual_info_score(y_true, labels):.3f}")
```

## 5. Elegir el Número de Clusters (K)

### Método del Codo (Elbow Method)

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTODO DEL CODO                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Graficar Inertia vs K y buscar el "codo"                      │
│                                                                │
│  Inertia                                                       │
│     │                                                          │
│     │ \                                                        │
│     │  \                                                       │
│     │   \                                                      │
│     │    ╲___                                                  │
│     │        ╲____                                             │
│     │             ╲_________                                   │
│     │                       ──────────                         │
│     └──────────────────────────────────── K                    │
│         1   2   3   4   5   6   7   8                          │
│                   ↑                                            │
│                 CODO                                           │
│            (K óptimo ≈ 4)                                      │
│                                                                │
│  El codo indica donde añadir más clusters                      │
│  ya no reduce significativamente la inertia                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Método Silhouette

```
┌────────────────────────────────────────────────────────────────┐
│  ANÁLISIS DE SILHOUETTE                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Graficar Silhouette Score vs K                                │
│  Elegir K con MAYOR silhouette                                 │
│                                                                │
│  Silhouette                                                    │
│     │                                                          │
│     │           ●                                              │
│     │       ●       ●                                          │
│     │   ●               ●                                      │
│     │ ●                     ●                                  │
│     │                           ●                              │
│     └──────────────────────────────── K                        │
│         2   3   4   5   6   7   8                              │
│                 ↑                                              │
│            MÁXIMO                                              │
│         (K óptimo = 4)                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código: Encontrar K Óptimo

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Datos
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [10, 0],
])

# Probar diferentes valores de K
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Método del codo
ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Número de clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Método del Codo')

# Silhouette
ax2.plot(K_range, silhouettes, 'ro-')
ax2.set_xlabel('Número de clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Análisis Silhouette')

plt.tight_layout()
plt.show()

# Mejor K según silhouette
best_k = K_range[np.argmax(silhouettes)]
print(f"\nMejor K según Silhouette: {best_k}")
print(f"Silhouette Score: {max(silhouettes):.3f}")
```

## 6. Preprocesamiento para Clustering

### Pasos Importantes

```
┌────────────────────────────────────────────────────────────────┐
│  PREPROCESAMIENTO CRÍTICO PARA CLUSTERING                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. ESCALADO (OBLIGATORIO para la mayoría)                     │
│     ───────────────────────────────────                        │
│     Sin escalar:                                               │
│       Feature A: rango [0, 1]                                  │
│       Feature B: rango [0, 1000000]                            │
│       → Feature B domina completamente la distancia            │
│                                                                │
│     Solución: StandardScaler o MinMaxScaler                    │
│                                                                │
│  2. REDUCCIÓN DE DIMENSIONALIDAD                               │
│     ────────────────────────────                               │
│     Muchas features → "Maldición de la dimensionalidad"        │
│     En alta dimensión, las distancias pierden significado      │
│                                                                │
│     Solución: PCA, t-SNE, UMAP antes de clustering             │
│                                                                │
│  3. MANEJO DE OUTLIERS                                         │
│     ──────────────────                                         │
│     K-Means es muy sensible a outliers                         │
│     Un outlier puede arrastrar un centroide                    │
│                                                                │
│     Solución: Detectar/eliminar outliers o usar DBSCAN         │
│                                                                │
│  4. SELECCIÓN DE FEATURES                                      │
│     ─────────────────────                                      │
│     Features irrelevantes añaden ruido                         │
│                                                                │
│     Solución: Análisis de correlación, feature selection       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código: Pipeline de Preprocesamiento

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import numpy as np

# Pipeline completo
clustering_pipeline = Pipeline([
    ('scaler', StandardScaler()),       # 1. Escalar
    ('pca', PCA(n_components=10)),      # 2. Reducir dimensión
    ('kmeans', KMeans(n_clusters=5))    # 3. Clustering
])

# Datos de alta dimensión
X = np.random.randn(1000, 100)

# Fit pipeline completo
labels = clustering_pipeline.fit_predict(X)

print(f"Clusters encontrados: {len(np.unique(labels))}")
print(f"Distribución: {np.bincount(labels)}")
```

## 7. Visualización de Clusters

### Técnicas de Visualización

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualizar_clusters(X, labels, titulo="Clusters"):
    """Visualiza clusters usando PCA o t-SNE"""

    # Si hay más de 2 dimensiones, reducir
    if X.shape[1] > 2:
        # PCA para reducir a 2D
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title(titulo)
    plt.show()

# Uso
# visualizar_clusters(X, labels, "K-Means Clustering")
```

### Visualización con t-SNE (para alta dimensión)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualizar_tsne(X, labels, perplexity=30):
    """t-SNE para visualización de alta dimensión"""

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=labels, cmap='tab10',
                         alpha=0.7, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Visualización t-SNE de Clusters')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

# visualizar_tsne(X, labels)
```

## 8. Ejemplo: Segmentación de Tráfico de Red

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Simular datos de tráfico de red
np.random.seed(42)
n_samples = 1000

# Generar 4 tipos de tráfico
trafico_web = np.column_stack([
    np.random.normal(80, 10, 300),      # puerto destino ~80
    np.random.normal(500, 100, 300),    # bytes
    np.random.normal(10, 2, 300),       # paquetes
    np.random.normal(0.5, 0.1, 300),    # duración
])

trafico_ssh = np.column_stack([
    np.random.normal(22, 1, 200),       # puerto ~22
    np.random.normal(200, 50, 200),     # bytes
    np.random.normal(5, 1, 200),        # paquetes
    np.random.normal(30, 10, 200),      # duración larga
])

trafico_dns = np.column_stack([
    np.random.normal(53, 1, 300),       # puerto ~53
    np.random.normal(100, 20, 300),     # bytes pequeños
    np.random.normal(2, 0.5, 300),      # pocos paquetes
    np.random.normal(0.1, 0.02, 300),   # muy corto
])

trafico_sospechoso = np.column_stack([
    np.random.uniform(1024, 65535, 200),  # puertos altos aleatorios
    np.random.exponential(10000, 200),    # muchos bytes
    np.random.poisson(100, 200),          # muchos paquetes
    np.random.uniform(0.01, 0.1, 200),    # muy rápido
])

# Combinar
X = np.vstack([trafico_web, trafico_ssh, trafico_dns, trafico_sospechoso])
tipos_reales = ['Web']*300 + ['SSH']*200 + ['DNS']*300 + ['Sospechoso']*200

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar K óptimo
print("Buscando número óptimo de clusters...")
silhouettes = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouettes.append(score)
    print(f"  K={k}: Silhouette={score:.3f}")

best_k = range(2, 8)[np.argmax(silhouettes)]
print(f"\nMejor K: {best_k}")

# Clustering final
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Analizar clusters
print("\n" + "=" * 60)
print("ANÁLISIS DE CLUSTERS DE TRÁFICO")
print("=" * 60)

df = pd.DataFrame(X, columns=['puerto', 'bytes', 'paquetes', 'duracion'])
df['cluster'] = labels
df['tipo_real'] = tipos_reales

for cluster in range(best_k):
    mask = df['cluster'] == cluster
    print(f"\nCluster {cluster} ({mask.sum()} conexiones):")
    print(f"  Puerto promedio: {df[mask]['puerto'].mean():.0f}")
    print(f"  Bytes promedio: {df[mask]['bytes'].mean():.0f}")
    print(f"  Paquetes promedio: {df[mask]['paquetes'].mean():.1f}")
    print(f"  Duración promedio: {df[mask]['duracion'].mean():.2f}s")
    print(f"  Tipos reales: {df[mask]['tipo_real'].value_counts().to_dict()}")
```

## 9. Resumen de Algoritmos

```
┌────────────────────────────────────────────────────────────────┐
│  CUÁNDO USAR CADA ALGORITMO                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  K-MEANS                                                       │
│    ✓ Clusters esféricos, tamaño similar                        │
│    ✓ Conoces K aproximadamente                                 │
│    ✓ Dataset grande (escalable)                                │
│    ✗ Formas irregulares, outliers                              │
│                                                                │
│  DBSCAN                                                        │
│    ✓ Formas arbitrarias                                        │
│    ✓ Detectar outliers                                         │
│    ✓ No conoces K                                              │
│    ✗ Densidad variable entre clusters                          │
│                                                                │
│  JERÁRQUICO                                                    │
│    ✓ Quieres explorar diferentes K                             │
│    ✓ Visualizar relaciones (dendrograma)                       │
│    ✓ Datasets pequeños/medianos                                │
│    ✗ Datasets muy grandes (lento)                              │
│                                                                │
│  GMM                                                           │
│    ✓ Clusters con diferentes formas/tamaños                    │
│    ✓ Necesitas probabilidad de pertenencia                     │
│    ✓ Clusters que se solapan                                   │
│    ✗ Muchos clusters, alta dimensión                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 10. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  CLUSTERING - RESUMEN                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Agrupar datos similares SIN etiquetas                       │
│    Aprendizaje no supervisado                                  │
│                                                                │
│  TIPOS PRINCIPALES:                                            │
│    • Particional: K-Means                                      │
│    • Jerárquico: Aglomerativo/Divisivo                         │
│    • Densidad: DBSCAN, HDBSCAN                                 │
│    • Modelo: GMM                                               │
│                                                                │
│  MÉTRICAS DE EVALUACIÓN:                                       │
│    Sin etiquetas: Silhouette, Calinski-Harabasz, Davies-Bouldin│
│    Con etiquetas: ARI, NMI, V-Measure                          │
│                                                                │
│  ELEGIR K:                                                     │
│    • Método del codo (inertia)                                 │
│    • Análisis Silhouette                                       │
│                                                                │
│  PREPROCESAMIENTO:                                             │
│    • ESCALAR siempre (StandardScaler)                          │
│    • Reducir dimensión si necesario (PCA)                      │
│    • Manejar outliers                                          │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de anomalías                                    │
│    • Agrupación de malware                                     │
│    • Segmentación de tráfico                                   │
│    • Análisis de comportamiento                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** K-Means
