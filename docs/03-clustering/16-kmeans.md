# K-Means Clustering

## 1. ¿Qué es K-Means?

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  K-MEANS = Particionar datos en K clusters                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  OBJETIVO: Minimizar la distancia de cada punto a su           │
│            centroide (centro del cluster)                      │
│                                                                │
│  ENTRADA:                                                      │
│    • Dataset X con n puntos                                    │
│    • Número de clusters K (tú lo decides)                      │
│                                                                │
│  SALIDA:                                                       │
│    • K centroides (centros de los clusters)                    │
│    • Asignación de cada punto a un cluster                     │
│                                                                │
│       ●  ●                  ★ = Centroide                      │
│     ●   ●  ●                ● = Puntos del cluster             │
│    ●  ★    ●                                                   │
│      ●   ●                  Cada punto pertenece al cluster    │
│    ●   ●                    cuyo centroide está MÁS CERCA      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Función Objetivo

```
┌────────────────────────────────────────────────────────────────┐
│  FUNCIÓN DE COSTE (INERTIA)                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²                                     │
│                                                                │
│  Donde:                                                        │
│    Cᵢ = cluster i                                              │
│    μᵢ = centroide del cluster i                                │
│    x = cada punto del cluster                                  │
│                                                                │
│  OBJETIVO: Minimizar J (suma de distancias al cuadrado)        │
│                                                                │
│  Cluster compacto:              Cluster disperso:              │
│       ●●●                          ●        ●                  │
│      ●★●●                              ★                       │
│       ●●                           ●         ●                 │
│    J bajo (bueno)                  J alto (malo)               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. El Algoritmo K-Means

### Pasos del Algoritmo

```
┌────────────────────────────────────────────────────────────────┐
│  ALGORITMO K-MEANS                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PASO 1: INICIALIZACIÓN                                        │
│    Elegir K puntos aleatorios como centroides iniciales        │
│                                                                │
│  PASO 2: ASIGNACIÓN                                            │
│    Asignar cada punto al centroide más cercano                 │
│                                                                │
│  PASO 3: ACTUALIZACIÓN                                         │
│    Recalcular centroides como la media de sus puntos           │
│                                                                │
│  PASO 4: REPETIR                                               │
│    Repetir pasos 2-3 hasta que los centroides no cambien       │
│    (o cambien menos que un umbral)                             │
│                                                                │
│  CONVERGENCIA:                                                 │
│    El algoritmo SIEMPRE converge                               │
│    Pero puede quedar en un mínimo LOCAL                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización Paso a Paso

```
ITERACIÓN 0 (Inicialización):
────────────────────────────────
    ●  ●  ○  ○
  ●   ●    ○   ○
   ●  ●  ○   ○  ○
    ●   ○  ○

    ★₁           ★₂      ← Centroides aleatorios

ITERACIÓN 1 (Asignación + Actualización):
────────────────────────────────────────
    ●  ●  │  ○  ○
  ●   ●   │   ○   ○      ← Puntos asignados al centroide más cercano
   ●  ●   │ ○   ○  ○
    ●     │  ○  ○
          │
     ★₁   │    ★₂        ← Centroides recalculados (media)

ITERACIÓN 2:
─────────────
    ●  ●    │  ○  ○
  ●   ●     │   ○   ○
   ●  ●     │○   ○  ○    ← Frontera se ajusta
    ●       │ ○  ○
            │
      ★₁    │   ★₂       ← Centroides se mueven

CONVERGENCIA:
─────────────
    ●  ●    │  ○  ○
  ●   ●     │   ○   ○
   ● ★₁●    │  ★₂ ○  ○   ← Centroides ya no cambian
    ●       │ ○  ○
```

### Pseudocódigo

```python
def kmeans(X, K, max_iter=100):
    # 1. Inicializar centroides aleatoriamente
    centroides = random_sample(X, K)

    for _ in range(max_iter):
        # 2. Asignar cada punto al centroide más cercano
        labels = []
        for x in X:
            distancias = [distancia(x, c) for c in centroides]
            labels.append(argmin(distancias))

        # 3. Recalcular centroides
        nuevos_centroides = []
        for k in range(K):
            puntos_cluster = X[labels == k]
            nuevos_centroides.append(mean(puntos_cluster))

        # 4. Verificar convergencia
        if centroides == nuevos_centroides:
            break
        centroides = nuevos_centroides

    return centroides, labels
```

## 3. Inicialización: K-Means++

### El Problema de la Inicialización Aleatoria

```
┌────────────────────────────────────────────────────────────────┐
│  PROBLEMA: Inicialización afecta el resultado                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  MALA inicialización:           BUENA inicialización:          │
│                                                                │
│    ●●●        ○○○                 ●●●        ○○○               │
│   ●●●●       ○○○○                ●●●●       ○○○○               │
│    ●●●        ○○○                 ●●●        ○○○               │
│                                                                │
│   ★₁ ★₂                            ★₁          ★₂              │
│   (ambos en el mismo lado)        (uno en cada grupo)          │
│                                                                │
│   Resultado: Un cluster vacío     Resultado: Correcto          │
│   o muy desbalanceado                                          │
│                                                                │
│  SOLUCIÓN: K-Means++ (inicialización inteligente)              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Algoritmo K-Means++

```
┌────────────────────────────────────────────────────────────────┐
│  K-MEANS++ INICIALIZACIÓN                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Elegir primer centroide uniformemente al azar              │
│                                                                │
│  2. Para cada punto x, calcular D(x) = distancia al            │
│     centroide más cercano ya elegido                           │
│                                                                │
│  3. Elegir siguiente centroide con probabilidad                │
│     proporcional a D(x)²                                       │
│     (puntos lejanos tienen más probabilidad)                   │
│                                                                │
│  4. Repetir 2-3 hasta tener K centroides                       │
│                                                                │
│  RESULTADO:                                                    │
│    Centroides tienden a estar bien distribuidos                │
│    Mucho mejor que aleatorio puro                              │
│                                                                │
│  En sklearn: init='k-means++' (default)                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Múltiples Inicializaciones

```python
from sklearn.cluster import KMeans

# n_init = número de veces que se ejecuta con diferentes inicializaciones
# Se queda con la mejor (menor inertia)
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',   # Inicialización inteligente
    n_init=10,          # Ejecutar 10 veces, quedarse con la mejor
    random_state=42
)
```

## 4. Implementación en Python

### Código Básico

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generar datos de ejemplo
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
    np.random.randn(100, 2) + [10, 0]
])

# IMPORTANTE: Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear y entrenar K-Means
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fit y predecir
labels = kmeans.fit_predict(X_scaled)

# Resultados
print(f"Centroides:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Número de iteraciones: {kmeans.n_iter_}")
print(f"Labels: {labels[:10]}...")
```

### Encontrar K Óptimo

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Método del codo + Silhouette
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Método del Codo')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Análisis Silhouette')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Mejor K
best_k = K_range[np.argmax(silhouettes)]
print(f"Mejor K según Silhouette: {best_k}")
```

### Visualización de Clusters

```python
import matplotlib.pyplot as plt

def plot_kmeans(X, labels, centroids, title="K-Means Clustering"):
    plt.figure(figsize=(10, 8))

    # Plot puntos coloreados por cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels,
                         cmap='viridis', alpha=0.6, s=50)

    # Plot centroides
    plt.scatter(centroids[:, 0], centroids[:, 1],
               c='red', marker='X', s=200, edgecolors='black',
               linewidth=2, label='Centroides')

    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Usar
plot_kmeans(X_scaled, labels, kmeans.cluster_centers_)
```

## 5. Hiperparámetros

### Tabla de Hiperparámetros

```
┌─────────────────┬─────────────┬────────────────────────────────┐
│   Parámetro     │   Default   │   Descripción                  │
├─────────────────┼─────────────┼────────────────────────────────┤
│ n_clusters      │      8      │ Número de clusters (K)         │
│                 │             │ DEBES elegirlo tú              │
├─────────────────┼─────────────┼────────────────────────────────┤
│ init            │ 'k-means++' │ Método de inicialización       │
│                 │             │ 'k-means++', 'random', o array │
├─────────────────┼─────────────┼────────────────────────────────┤
│ n_init          │     10      │ Número de inicializaciones     │
│                 │             │ (se queda con la mejor)        │
├─────────────────┼─────────────┼────────────────────────────────┤
│ max_iter        │    300      │ Máximo de iteraciones          │
├─────────────────┼─────────────┼────────────────────────────────┤
│ tol             │   1e-4      │ Tolerancia para convergencia   │
├─────────────────┼─────────────┼────────────────────────────────┤
│ algorithm       │  'lloyd'    │ Algoritmo: 'lloyd' o 'elkan'   │
│                 │             │ elkan más rápido para K bajo   │
├─────────────────┼─────────────┼────────────────────────────────┤
│ random_state    │    None     │ Semilla para reproducibilidad  │
└─────────────────┴─────────────┴────────────────────────────────┘
```

## 6. Mini-Batch K-Means

### Para Datasets Grandes

```
┌────────────────────────────────────────────────────────────────┐
│  MINI-BATCH K-MEANS                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  K-Means estándar usa TODOS los datos en cada iteración        │
│  → Lento para datasets muy grandes                             │
│                                                                │
│  Mini-Batch K-Means usa solo un SUBSET (batch) por iteración   │
│  → Mucho más rápido                                            │
│  → Resultado ligeramente peor pero aceptable                   │
│                                                                │
│  COMPARACIÓN:                                                  │
│                                                                │
│    K-Means:       O(n × k × d × i)                             │
│    Mini-Batch:    O(b × k × d × i)    donde b << n             │
│                                                                │
│    n = datos, k = clusters, d = dimensiones, i = iteraciones   │
│    b = tamaño del batch                                        │
│                                                                │
│  USAR CUANDO:                                                  │
│    • Dataset > 10,000 muestras                                 │
│    • Necesitas resultados rápidos                              │
│    • Puedes tolerar resultado ligeramente subóptimo            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código Mini-Batch

```python
from sklearn.cluster import MiniBatchKMeans

# Para datasets grandes
minibatch_kmeans = MiniBatchKMeans(
    n_clusters=5,
    batch_size=100,      # Muestras por batch
    max_iter=100,
    n_init=3,            # Menos inicializaciones (más rápido)
    random_state=42
)

labels = minibatch_kmeans.fit_predict(X_scaled)
print(f"Inertia: {minibatch_kmeans.inertia_:.2f}")
```

## 7. Limitaciones de K-Means

### Problemas Conocidos

```
┌────────────────────────────────────────────────────────────────┐
│  LIMITACIONES DE K-MEANS                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SOLO CLUSTERS ESFÉRICOS                                    │
│     ─────────────────────────                                  │
│                                                                │
│     Funciona bien:              NO funciona bien:              │
│        ●●●     ○○○                    ●●●●●●●●●●               │
│       ●●●●    ○○○○                   ●●●●●●●●●●●               │
│        ●●●     ○○○                  ○○○○○○○○○○○○               │
│                                     ○○○○○○○○○○○                │
│                                                                │
│  2. SENSIBLE A OUTLIERS                                        │
│     ────────────────────                                       │
│                                                                │
│        ●●●●                          ●●●●                      │
│       ●●★●●                         ●●●●●          ✗ (outlier) │
│        ●●●●                          ●●●●     ★                │
│                                          ↑                     │
│     Centroide correcto             Centroide arrastrado        │
│                                                                │
│  3. REQUIERE ESPECIFICAR K                                     │
│     ──────────────────────                                     │
│     No siempre sabemos cuántos clusters hay                    │
│                                                                │
│  4. CLUSTERS DE DIFERENTE TAMAÑO                               │
│     ─────────────────────────────                              │
│     Tiende a crear clusters de tamaño similar                  │
│                                                                │
│     Datos reales:                 K-Means produce:             │
│        ●●●●●     ○                   ●●●     ●●                │
│       ●●●●●●●                       ●●●●     ●●○               │
│        ●●●●●                         ●●●●                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Cuándo NO Usar K-Means

```
NO usar K-Means cuando:

  ✗ Clusters tienen formas irregulares (usar DBSCAN)

  ✗ Hay muchos outliers (usar DBSCAN o preprocesar)

  ✗ Clusters tienen densidades muy diferentes

  ✗ No tienes idea de cuántos clusters hay (usar DBSCAN o jerárquico)

  ✗ Features categóricas (usar K-Modes o K-Prototypes)
```

## 8. Ejemplo Práctico: Segmentación de Ataques

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Simular datos de conexiones de red (posibles ataques)
np.random.seed(42)

# Tipo 1: Conexiones normales (bajo volumen, duración normal)
normal = np.column_stack([
    np.random.normal(100, 30, 500),     # bytes
    np.random.normal(5, 2, 500),        # paquetes
    np.random.normal(10, 3, 500),       # duración (segundos)
    np.random.normal(1, 0.2, 500),      # conexiones por minuto
])

# Tipo 2: Port Scan (muchas conexiones cortas)
portscan = np.column_stack([
    np.random.normal(50, 10, 200),      # bytes (poco)
    np.random.normal(1, 0.3, 200),      # paquetes (poco)
    np.random.normal(0.1, 0.02, 200),   # duración (muy corta)
    np.random.normal(100, 20, 200),     # conexiones por minuto (muchas!)
])

# Tipo 3: DDoS (alto volumen, corta duración)
ddos = np.column_stack([
    np.random.normal(5000, 1000, 150),  # bytes (mucho)
    np.random.normal(100, 20, 150),     # paquetes (muchos)
    np.random.normal(0.5, 0.1, 150),    # duración (corta)
    np.random.normal(50, 10, 150),      # conexiones por minuto (alto)
])

# Tipo 4: Data Exfiltration (alto volumen saliente, larga duración)
exfil = np.column_stack([
    np.random.normal(10000, 2000, 100), # bytes (muy alto)
    np.random.normal(50, 10, 100),      # paquetes
    np.random.normal(300, 60, 100),     # duración (larga)
    np.random.normal(2, 0.5, 100),      # conexiones por minuto (pocas)
])

# Combinar datos
X = np.vstack([normal, portscan, ddos, exfil])
tipos_reales = (['Normal']*500 + ['PortScan']*200 +
                ['DDoS']*150 + ['Exfiltration']*100)

print(f"Total conexiones: {len(X)}")
print(f"Features: bytes, paquetes, duración, conexiones/min")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar K óptimo
print("\nBuscando número óptimo de clusters...")
silhouettes = []
inertias = []

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    inertias.append(kmeans.inertia_)
    print(f"  K={k}: Silhouette={sil:.3f}, Inertia={kmeans.inertia_:.0f}")

best_k = range(2, 8)[np.argmax(silhouettes)]
print(f"\nMejor K: {best_k}")

# Clustering final
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Análisis de resultados
print("\n" + "=" * 70)
print("ANÁLISIS DE CLUSTERS - DETECCIÓN DE PATRONES DE ATAQUE")
print("=" * 70)

df = pd.DataFrame(X, columns=['bytes', 'paquetes', 'duracion', 'conn_min'])
df['cluster'] = labels
df['tipo_real'] = tipos_reales

for cluster in range(best_k):
    mask = df['cluster'] == cluster
    n_cluster = mask.sum()

    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster} ({n_cluster} conexiones)")
    print(f"{'='*50}")

    # Estadísticas
    print("\nCaracterísticas promedio:")
    print(f"  Bytes/conexión:    {df[mask]['bytes'].mean():,.0f}")
    print(f"  Paquetes/conexión: {df[mask]['paquetes'].mean():.1f}")
    print(f"  Duración (seg):    {df[mask]['duracion'].mean():.1f}")
    print(f"  Conn/minuto:       {df[mask]['conn_min'].mean():.1f}")

    # Composición real
    print("\nComposición real:")
    for tipo in df[mask]['tipo_real'].unique():
        count = (df[mask]['tipo_real'] == tipo).sum()
        pct = count / n_cluster * 100
        print(f"  {tipo}: {count} ({pct:.1f}%)")

    # Interpretación automática
    avg_bytes = df[mask]['bytes'].mean()
    avg_conn = df[mask]['conn_min'].mean()
    avg_dur = df[mask]['duracion'].mean()

    if avg_conn > 50 and avg_dur < 1:
        print("\n⚠️  PATRÓN DETECTADO: Posible PORT SCAN")
        print("   Muchas conexiones muy cortas")
    elif avg_bytes > 3000 and avg_dur < 2:
        print("\n🚨 PATRÓN DETECTADO: Posible DDoS")
        print("   Alto volumen en corto tiempo")
    elif avg_bytes > 5000 and avg_dur > 100:
        print("\n🔴 PATRÓN DETECTADO: Posible DATA EXFILTRATION")
        print("   Alto volumen saliente sostenido")
    else:
        print("\n✅ PATRÓN: Tráfico aparentemente normal")

# Visualización
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                     cmap='tab10', alpha=0.6, s=50)

# Centroides en espacio PCA
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='red', marker='X', s=200, edgecolors='black',
           linewidth=2, label='Centroides')

plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Segmentación de Tráfico de Red (K-Means)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Matriz de confusión simplificada
print("\n" + "=" * 70)
print("MATRIZ: CLUSTER vs TIPO REAL")
print("=" * 70)
confusion = pd.crosstab(df['cluster'], df['tipo_real'])
print(confusion)
```

## 9. Predicción de Nuevos Datos

```python
# Después de entrenar, predecir nuevos datos
nuevas_conexiones = np.array([
    [80, 3, 8, 1.5],      # Parece normal
    [40, 1, 0.08, 120],   # Parece port scan
    [8000, 150, 0.3, 80], # Parece DDoS
    [15000, 40, 400, 1],  # Parece exfiltración
])

# IMPORTANTE: Escalar con el mismo scaler
nuevas_scaled = scaler.transform(nuevas_conexiones)

# Predecir cluster
predicciones = kmeans.predict(nuevas_scaled)

print("Clasificación de nuevas conexiones:")
for i, (conexion, cluster) in enumerate(zip(nuevas_conexiones, predicciones)):
    print(f"\nConexión {i+1}: bytes={conexion[0]:.0f}, "
          f"paquetes={conexion[1]:.0f}, dur={conexion[2]:.1f}s, "
          f"conn/min={conexion[3]:.1f}")
    print(f"  → Cluster {cluster}")
```

## 10. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE K-MEANS                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Muy rápido (O(n × k × d × i))                               │
│  ✓ Escalable a datasets grandes                                │
│  ✓ Simple de implementar y entender                            │
│  ✓ Garantía de convergencia                                    │
│  ✓ Funciona bien con clusters esféricos                        │
│  ✓ Resultados interpretables (centroides)                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE K-MEANS                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Requiere especificar K                                      │
│  ✗ Solo encuentra clusters esféricos                           │
│  ✗ Sensible a outliers                                         │
│  ✗ Sensible a la inicialización                                │
│  ✗ Clusters de tamaño similar                                  │
│  ✗ Solo distancia Euclidiana                                   │
│  ✗ Puede quedar en mínimo local                                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 11. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  K-MEANS - RESUMEN                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ALGORITMO:                                                    │
│    1. Inicializar K centroides (k-means++)                     │
│    2. Asignar puntos al centroide más cercano                  │
│    3. Recalcular centroides como media                         │
│    4. Repetir 2-3 hasta convergencia                           │
│                                                                │
│  HIPERPARÁMETROS:                                              │
│    • n_clusters: número de clusters (K)                        │
│    • init: 'k-means++' (recomendado)                           │
│    • n_init: 10 (múltiples inicializaciones)                   │
│                                                                │
│  ELEGIR K:                                                     │
│    • Método del codo (inertia)                                 │
│    • Silhouette score (recomendado)                            │
│                                                                │
│  PREPROCESAMIENTO:                                             │
│    • StandardScaler OBLIGATORIO                                │
│                                                                │
│  CUÁNDO USAR:                                                  │
│    ✓ Clusters esféricos esperados                              │
│    ✓ Conoces K aproximadamente                                 │
│    ✓ Dataset grande                                            │
│    ✓ Necesitas rapidez                                         │
│                                                                │
│  CUÁNDO NO USAR:                                               │
│    ✗ Formas irregulares → DBSCAN                               │
│    ✗ Muchos outliers → DBSCAN                                  │
│    ✗ No conoces K → Jerárquico, DBSCAN                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** DBSCAN
