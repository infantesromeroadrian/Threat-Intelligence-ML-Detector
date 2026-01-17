# Clustering Jerárquico

## 1. ¿Qué es Clustering Jerárquico?

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  CLUSTERING JERÁRQUICO                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Crea una JERARQUÍA de clusters (árbol)                        │
│  NO requiere especificar K de antemano                         │
│                                                                │
│  DOS ENFOQUES:                                                 │
│                                                                │
│  AGLOMERATIVO (Bottom-Up):                                     │
│  ─────────────────────────                                     │
│  • Empieza: cada punto es un cluster                           │
│  • Iteración: une los dos clusters más cercanos                │
│  • Termina: todo en un solo cluster                            │
│                                                                │
│       A  B  C  D  E       →      (AB) (CDE)     →   ((AB)(CDE))│
│       │  │  │  │  │              ╱  ╲  │  ╲           │        │
│       cada uno solo          se unen los más        un solo    │
│                              cercanos              cluster     │
│                                                                │
│  DIVISIVO (Top-Down):                                          │
│  ────────────────────                                          │
│  • Empieza: todos en un cluster                                │
│  • Iteración: divide el cluster más heterogéneo                │
│  • Termina: cada punto es un cluster                           │
│                                                                │
│  (Aglomerativo es mucho más común)                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Dendrograma

```
┌────────────────────────────────────────────────────────────────┐
│  DENDROGRAMA = Visualización del proceso jerárquico            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Distancia                                                     │
│     │                                                          │
│  5  │              ┌─────────────────┐                         │
│     │              │                 │                         │
│  4  │        ┌─────┘           ┌─────┘                         │
│     │        │                 │                               │
│  3  │   ┌────┘            ┌────┘                               │
│     │   │                 │                                    │
│  2  │ ┌─┘           ┌─────┘                                    │
│     │ │             │                                          │
│  1  │─┘       ┌─────┘                                          │
│     │         │                                                │
│  0  └─────────┴──────────────────────────                      │
│       A   B   C   D   E   F   G   H                            │
│                                                                │
│  Cómo leer:                                                    │
│    • Eje Y = distancia a la que se unen clusters               │
│    • Líneas verticales = clusters                              │
│    • Líneas horizontales = unión de clusters                   │
│    • Cortar horizontalmente da diferentes K                    │
│                                                                │
│  Corte en distancia 3:                                         │
│    → Cluster 1: {A, B}                                         │
│    → Cluster 2: {C, D, E}                                      │
│    → Cluster 3: {F, G, H}                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. Métodos de Linkage

### Tipos de Linkage

```
┌────────────────────────────────────────────────────────────────┐
│  LINKAGE = Cómo medir distancia entre CLUSTERS                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SINGLE LINKAGE (Mínimo)                                    │
│     ────────────────────────                                   │
│     Distancia = mínima entre cualquier par de puntos           │
│                                                                │
│        Cluster A        Cluster B                              │
│         ●  ●              ○  ○                                 │
│        ●    ●←──────────→○    ○                                │
│         ●  ●         d    ○  ○                                 │
│                       ↑                                        │
│               Distancia más corta                              │
│                                                                │
│     Pro: Encuentra clusters alargados                          │
│     Con: "Efecto cadena" - une clusters que no deberían        │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  2. COMPLETE LINKAGE (Máximo)                                  │
│     ─────────────────────────                                  │
│     Distancia = máxima entre cualquier par de puntos           │
│                                                                │
│        ●←────────────────────→○                                │
│         ↖                   ↗                                  │
│          ●  ●          ○  ○                                    │
│         ●    ●        ○    ○                                   │
│          ●  ●          ○  ○                                    │
│               d = distancia más larga                          │
│                                                                │
│     Pro: Clusters compactos y esféricos                        │
│     Con: Sensible a outliers                                   │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  3. AVERAGE LINKAGE (UPGMA)                                    │
│     ───────────────────────                                    │
│     Distancia = promedio de todas las distancias               │
│                                                                │
│          ●  ●          ○  ○                                    │
│         ●ₐ  ●ᵦ   ⟷    ○ᵧ  ○δ                                  │
│          ●  ●          ○  ○                                    │
│                                                                │
│     d = (d(a,γ) + d(a,δ) + d(β,γ) + d(β,δ) + ...) / n         │
│                                                                │
│     Pro: Balance entre single y complete                       │
│     Con: Puede ser lento                                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  4. WARD'S METHOD                                              │
│     ─────────────────                                          │
│     Minimiza el incremento de varianza al unir clusters        │
│                                                                │
│     Unir clusters que aumenten MENOS la varianza intra-cluster │
│                                                                │
│     Pro: Tiende a crear clusters de tamaño similar             │
│     Con: Asume clusters esféricos                              │
│     ★ Más usado en la práctica                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Comparación Visual

```
SINGLE LINKAGE:              COMPLETE LINKAGE:           WARD:
(une más fácilmente)         (clusters compactos)        (tamaño similar)

     ●●●●●●●●●●                  ●●●   ●●●                 ●●●●   ●●●●
    ●●●●●●●●●●●                 ●●●●● ●●●●●               ●●●●● ●●●●●
     ●●●●●●●●●●                  ●●●   ●●●                 ●●●●   ●●●●
          ↓
    Puede crear "cadenas"      Clusters bien             Clusters
    de un solo cluster         separados                 balanceados
```

## 3. Algoritmo Aglomerativo

### Pasos

```
┌────────────────────────────────────────────────────────────────┐
│  ALGORITMO CLUSTERING AGLOMERATIVO                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ENTRADA:                                                      │
│    • N puntos                                                  │
│    • Método de linkage                                         │
│    • Métrica de distancia                                      │
│                                                                │
│  ALGORITMO:                                                    │
│                                                                │
│  1. Inicializar: cada punto es un cluster (N clusters)         │
│                                                                │
│  2. Calcular matriz de distancias entre todos los clusters     │
│                                                                │
│  3. MIENTRAS número de clusters > 1:                           │
│     a. Encontrar los dos clusters más cercanos                 │
│     b. Unirlos en un nuevo cluster                             │
│     c. Actualizar matriz de distancias                         │
│                                                                │
│  4. Construir dendrograma                                      │
│                                                                │
│  COMPLEJIDAD:                                                  │
│    • Tiempo: O(n³) o O(n² log n) con optimizaciones            │
│    • Espacio: O(n²) para matriz de distancias                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ejemplo Paso a Paso

```
DATOS: A, B, C, D, E (5 puntos)

PASO 0: Matriz de distancias inicial
─────────────────────────────────────
        A    B    C    D    E
   A    0    2    6    10   9
   B    2    0    5    9    8
   C    6    5    0    4    5
   D    10   9    4    0    3
   E    9    8    5    3    0

PASO 1: Unir A y B (distancia = 2, la mínima)
──────────────────────────────────────────────
Clusters: {A,B}, {C}, {D}, {E}

        AB   C    D    E
  AB    0    5    9    8     ← Recalcular distancias
   C    5    0    4    5       (usando linkage elegido)
   D    9    4    0    3
   E    8    5    3    0

PASO 2: Unir D y E (distancia = 3)
──────────────────────────────────
Clusters: {A,B}, {C}, {D,E}

        AB   C    DE
  AB    0    5    8
   C    5    0    4
  DE    8    4    0

PASO 3: Unir C y DE (distancia = 4)
───────────────────────────────────
Clusters: {A,B}, {C,D,E}

        AB   CDE
  AB    0    5
 CDE    5    0

PASO 4: Unir AB y CDE (distancia = 5)
─────────────────────────────────────
Cluster final: {A,B,C,D,E}

DENDROGRAMA:
                    ┌────────┐
          5         │        │
                ┌───┘    ┌───┘
          4     │   ┌────┘
                │   │
          3     │ ┌─┘
          2   ┌─┘ │
              │   │
          0   A B C D E
```

## 4. Implementación en Python

### Código Básico

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generar datos
np.random.seed(42)
X = np.vstack([
    np.random.randn(30, 2) + [0, 0],
    np.random.randn(30, 2) + [5, 5],
    np.random.randn(30, 2) + [10, 0],
])

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering Jerárquico con sklearn
agg = AgglomerativeClustering(
    n_clusters=3,          # Si conoces K
    # n_clusters=None,     # Si no conoces K
    # distance_threshold=5, # Cortar en distancia
    linkage='ward',        # 'single', 'complete', 'average', 'ward'
    metric='euclidean'
)

labels = agg.fit_predict(X_scaled)

print(f"Clusters encontrados: {len(set(labels))}")
print(f"Distribución: {np.bincount(labels)}")
```

### Crear Dendrograma

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def plot_dendrogram(X, method='ward', title="Dendrograma"):
    """Crea y visualiza un dendrograma"""

    # Calcular linkage
    Z = linkage(X, method=method)

    # Crear figura
    plt.figure(figsize=(12, 8))

    # Dendrograma
    dendrogram(
        Z,
        truncate_mode='level',    # 'lastp' o 'level' para grandes datasets
        p=30,                      # Mostrar máximo 30 hojas
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True
    )

    plt.xlabel('Índice de muestra (o tamaño del cluster)')
    plt.ylabel('Distancia')
    plt.title(f'{title} (linkage={method})')
    plt.tight_layout()
    plt.show()

    return Z

# Crear dendrograma
Z = plot_dendrogram(X_scaled, method='ward')
```

### Elegir Número de Clusters

```python
from scipy.cluster.hierarchy import fcluster

def analizar_cortes(Z, X, max_k=10):
    """Analiza diferentes cortes del dendrograma"""

    from sklearn.metrics import silhouette_score

    scores = []

    for k in range(2, max_k + 1):
        # Cortar dendrograma para obtener k clusters
        labels = fcluster(Z, k, criterion='maxclust')

        # Calcular silhouette
        score = silhouette_score(X, labels)
        scores.append((k, score))
        print(f"K={k}: Silhouette={score:.3f}")

    # Mejor K
    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"\nMejor K según Silhouette: {best_k}")

    # Visualizar
    plt.figure(figsize=(10, 5))
    plt.plot([s[0] for s in scores], [s[1] for s in scores], 'bo-')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Análisis de Cortes del Dendrograma')
    plt.grid(True, alpha=0.3)
    plt.show()

    return best_k

best_k = analizar_cortes(Z, X_scaled)
```

### Cortar el Dendrograma

```python
from scipy.cluster.hierarchy import fcluster

# Método 1: Por número de clusters
labels = fcluster(Z, t=3, criterion='maxclust')  # 3 clusters

# Método 2: Por distancia
labels = fcluster(Z, t=5.0, criterion='distance')  # Cortar en distancia 5

# Método 3: Por inconsistencia (automático)
labels = fcluster(Z, t=1.5, criterion='inconsistent')

# Visualizar resultado
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustering Jerárquico')
plt.show()
```

## 5. Comparación de Linkages

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def comparar_linkages(X, n_clusters=3):
    """Compara diferentes métodos de linkage"""

    linkages = ['single', 'complete', 'average', 'ward']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, method in enumerate(linkages):
        # Dendrograma
        Z = linkage(X, method=method)
        ax_dend = axes[0, i]
        dendrogram(Z, ax=ax_dend, truncate_mode='level', p=10)
        ax_dend.set_title(f'{method.upper()} Linkage')
        ax_dend.set_xlabel('')

        # Clustering
        if method == 'ward':
            metric = 'euclidean'
        else:
            metric = 'euclidean'

        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=method,
            metric=metric
        )
        labels = agg.fit_predict(X)

        ax_cluster = axes[1, i]
        scatter = ax_cluster.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        ax_cluster.set_title(f'Clusters ({method})')

    plt.tight_layout()
    plt.show()

# Uso
comparar_linkages(X_scaled)
```

## 6. Ejemplo Práctico: Agrupación de Malware

```python
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Simular características de muestras de malware
np.random.seed(42)

# Familia 1: Ransomware (cifrado, pocos archivos, rápido)
ransomware = np.column_stack([
    np.random.normal(0.9, 0.1, 40),    # uso_crypto (alto)
    np.random.normal(100, 20, 40),     # archivos_modificados
    np.random.normal(5, 1, 40),        # tiempo_ejecucion (seg)
    np.random.normal(0.1, 0.05, 40),   # trafico_red (bajo)
    np.random.normal(0.8, 0.1, 40),    # persistencia
])

# Familia 2: Spyware (poco cifrado, monitoreo largo, red moderada)
spyware = np.column_stack([
    np.random.normal(0.2, 0.1, 35),    # uso_crypto (bajo)
    np.random.normal(10, 5, 35),       # archivos_modificados (pocos)
    np.random.normal(3600, 600, 35),   # tiempo_ejecucion (largo)
    np.random.normal(0.5, 0.1, 35),    # trafico_red (moderado)
    np.random.normal(0.9, 0.05, 35),   # persistencia (alto)
])

# Familia 3: Botnet (poco cifrado, ejecución variable, mucha red)
botnet = np.column_stack([
    np.random.normal(0.3, 0.1, 45),    # uso_crypto
    np.random.normal(5, 2, 45),        # archivos_modificados
    np.random.normal(7200, 1000, 45),  # tiempo_ejecucion (muy largo)
    np.random.normal(0.9, 0.1, 45),    # trafico_red (alto)
    np.random.normal(0.95, 0.03, 45),  # persistencia (muy alto)
])

# Familia 4: Cryptominer (alto CPU, larga duración, moderada red)
cryptominer = np.column_stack([
    np.random.normal(0.7, 0.15, 30),   # uso_crypto (alto, para mining)
    np.random.normal(3, 1, 30),        # archivos_modificados (pocos)
    np.random.normal(86400, 10000, 30),# tiempo_ejecucion (muy largo)
    np.random.normal(0.3, 0.1, 30),    # trafico_red (pool connection)
    np.random.normal(0.85, 0.1, 30),   # persistencia
])

# Combinar
X = np.vstack([ransomware, spyware, botnet, cryptominer])
familias_reales = (['Ransomware']*40 + ['Spyware']*35 +
                   ['Botnet']*45 + ['Cryptominer']*30)

features = ['uso_crypto', 'archivos_mod', 'tiempo_ejec', 'trafico_red', 'persistencia']

print(f"Total muestras de malware: {len(X)}")
print(f"Features: {features}")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear dendrograma
print("\nCreando dendrograma...")
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(Z, truncate_mode='level', p=15,
           leaf_rotation=90, leaf_font_size=8)
plt.xlabel('Muestras de Malware')
plt.ylabel('Distancia (Ward)')
plt.title('Dendrograma de Familias de Malware')
plt.axhline(y=10, color='r', linestyle='--', label='Corte sugerido')
plt.legend()
plt.tight_layout()
plt.show()

# Analizar diferentes números de clusters
print("\nAnalizando diferentes cortes...")
for k in range(2, 7):
    labels = fcluster(Z, k, criterion='maxclust')
    score = silhouette_score(X_scaled, labels)
    print(f"  K={k}: Silhouette={score:.3f}")

# Clustering final
n_clusters = 4  # Sabemos que hay 4 familias
labels = fcluster(Z, n_clusters, criterion='maxclust')

# Análisis de resultados
print("\n" + "=" * 70)
print("ANÁLISIS DE FAMILIAS DE MALWARE DETECTADAS")
print("=" * 70)

df = pd.DataFrame(X, columns=features)
df['cluster'] = labels
df['familia_real'] = familias_reales

for cluster in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cluster
    n = mask.sum()

    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster} ({n} muestras)")
    print(f"{'='*50}")

    # Características promedio
    print("\nCaracterísticas promedio:")
    for feat in features:
        val = df[mask][feat].mean()
        std = df[mask][feat].std()
        print(f"  {feat:15}: {val:.2f} (±{std:.2f})")

    # Composición real
    print("\nFamilias reales en este cluster:")
    for familia, count in df[mask]['familia_real'].value_counts().items():
        pct = count / n * 100
        print(f"  {familia}: {count} ({pct:.1f}%)")

    # Interpretación automática
    avg_crypto = df[mask]['uso_crypto'].mean()
    avg_tiempo = df[mask]['tiempo_ejec'].mean()
    avg_red = df[mask]['trafico_red'].mean()
    avg_archivos = df[mask]['archivos_mod'].mean()

    print("\n📋 Interpretación:")
    if avg_crypto > 0.7 and avg_tiempo < 100:
        print("  → Comportamiento típico de RANSOMWARE")
        print("    (alto cifrado, ejecución rápida)")
    elif avg_red > 0.7 and avg_tiempo > 5000:
        print("  → Comportamiento típico de BOTNET")
        print("    (mucho tráfico de red, larga duración)")
    elif avg_tiempo > 50000:
        print("  → Comportamiento típico de CRYPTOMINER")
        print("    (ejecución muy prolongada)")
    elif avg_tiempo > 1000 and avg_red > 0.3 and avg_red < 0.7:
        print("  → Comportamiento típico de SPYWARE")
        print("    (ejecución larga, monitoreo)")

# Matriz de confusión
print("\n" + "=" * 70)
print("MATRIZ: CLUSTER vs FAMILIA REAL")
print("=" * 70)
confusion = pd.crosstab(df['cluster'], df['familia_real'])
print(confusion)

# Visualización con PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))

# Plot por cluster detectado
for cluster in sorted(set(labels)):
    mask = labels == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               alpha=0.7, s=60, label=f'Cluster {cluster}')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
plt.title('Clustering Jerárquico de Familias de Malware')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 7. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DEL CLUSTERING JERÁRQUICO                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ NO requiere especificar K de antemano                       │
│  ✓ Dendrograma permite explorar diferentes K                   │
│  ✓ Visualización intuitiva de relaciones                       │
│  ✓ Determinístico (mismo resultado siempre)                    │
│  ✓ Flexible con diferentes linkages                            │
│  ✓ Puede usar cualquier métrica de distancia                   │
│  ✓ Revela estructura jerárquica de los datos                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DEL CLUSTERING JERÁRQUICO                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Lento para datasets grandes O(n²) a O(n³)                   │
│  ✗ Requiere O(n²) memoria para matriz de distancias            │
│  ✗ Una vez hecha una unión, no se puede deshacer               │
│  ✗ Sensible a outliers (especialmente single linkage)          │
│  ✗ No asigna nuevos puntos directamente                        │
│  ✗ Puede ser difícil elegir el corte correcto                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. Cuándo Usar Clustering Jerárquico

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ No sabes cuántos clusters hay                               │
│  ✓ Quieres explorar la estructura de los datos                 │
│  ✓ Datos tienen estructura jerárquica natural                  │
│  ✓ Dataset pequeño/mediano (< 10,000 muestras)                 │
│  ✓ Taxonomías (clasificación de especies, malware, etc.)       │
│  ✓ Análisis exploratorio                                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Dataset muy grande (> 10,000)                               │
│  ✗ Necesitas asignar nuevos puntos frecuentemente              │
│  ✗ Memoria limitada                                            │
│  ✗ Clusters esféricos y conoces K (usar K-Means)               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 9. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  CLUSTERING JERÁRQUICO - RESUMEN                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Crear jerarquía de clusters (árbol/dendrograma)             │
│    No requiere K de antemano                                   │
│                                                                │
│  TIPOS:                                                        │
│    • Aglomerativo (bottom-up): más común                       │
│    • Divisivo (top-down): menos usado                          │
│                                                                │
│  LINKAGES:                                                     │
│    • Single: mínima distancia (clusters alargados)             │
│    • Complete: máxima distancia (clusters compactos)           │
│    • Average: promedio de distancias                           │
│    • Ward: minimiza varianza (★ más recomendado)               │
│                                                                │
│  DENDROGRAMA:                                                  │
│    • Eje Y = distancia de unión                                │
│    • Cortar horizontalmente define K                           │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Agrupación de malware por familia                         │
│    • Taxonomías de amenazas                                    │
│    • Análisis de similitud entre ataques                       │
│    • Correlación de IOCs                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Gaussian Mixture Models
