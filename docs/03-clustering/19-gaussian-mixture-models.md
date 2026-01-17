# Gaussian Mixture Models (GMM)

## 1. ¿Qué es un GMM?

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  GAUSSIAN MIXTURE MODEL                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Los datos son generados por una MEZCLA de               │
│        distribuciones Gaussianas (normales)                    │
│                                                                │
│  Cada cluster es una Gaussiana con:                            │
│    • μ (mu): Centro (media)                                    │
│    • Σ (sigma): Forma (covarianza)                             │
│    • π (pi): Peso (proporción de datos)                        │
│                                                                │
│           Gaussiana 1          Gaussiana 2                     │
│              ╭───╮               ╭───╮                         │
│            ╱  μ₁  ╲           ╱   μ₂  ╲                        │
│          ╱    │    ╲       ╱     │     ╲                       │
│        ╱      │      ╲   ╱       │       ╲                     │
│      ╱        │        ╲╱        │         ╲                   │
│    ──────────────────────────────────────────                  │
│              π₁=0.4              π₂=0.6                        │
│                                                                │
│  La mezcla de ambas genera los datos observados                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### GMM vs K-Means

```
┌─────────────────────────────────────────────────────────────────┐
│  K-MEANS                          GMM                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HARD ASSIGNMENT                  SOFT ASSIGNMENT               │
│  (asignación dura)                (asignación suave)            │
│                                                                 │
│  Cada punto pertenece a           Cada punto tiene PROBABILIDAD │
│  UN solo cluster                  de pertenecer a cada cluster  │
│                                                                 │
│     ●●●      ○○○                    ●●●      ○○○                │
│    ●●●●     ○○○○                   ●●◐●     ○◐○○                │
│     ●●●      ○○○                    ●●●      ○○○                │
│                                        ↑                        │
│  Punto: 100% cluster 1         Punto: 70% cluster 1             │
│                                       30% cluster 2             │
│                                                                 │
│  Solo clusters esféricos          Clusters elípticos            │
│  (misma varianza)                 (diferente forma)             │
│                                                                 │
│      ○○○                              ○○○                       │
│     ○○○○○                            ○○○○○                      │
│      ○○○                            ○○○○○○○                     │
│                                      ○○○○○                      │
│  Círculos                        Elipses                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Ejemplo Intuitivo

```
PROBLEMA: Clasificar tráfico de red

Tráfico tiene 2 tipos:
  • Normal: bytes ≈ 100, paquetes ≈ 5 (la mayoría)
  • Streaming: bytes ≈ 5000, paquetes ≈ 50 (algunos)

GMM aprende:
  • Gaussiana 1: μ₁=[100, 5], π₁=0.8 (80% del tráfico)
  • Gaussiana 2: μ₂=[5000, 50], π₂=0.2 (20% del tráfico)

Nueva conexión: bytes=500, paquetes=10
  → P(Normal | conexión) = 0.65
  → P(Streaming | conexión) = 0.35
  → Clasificación: Normal (pero con incertidumbre)

Esto es más informativo que K-Means que diría "100% Normal"
```

## 2. Formulación Matemática

### Distribución Gaussiana Multivariada

```
┌────────────────────────────────────────────────────────────────┐
│  GAUSSIANA MULTIVARIADA                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Para un punto x de dimensión d:                               │
│                                                                │
│                        1                    1                  │
│  p(x|μ,Σ) = ─────────────────── exp(- ─ (x-μ)ᵀ Σ⁻¹ (x-μ))     │
│             (2π)^(d/2) |Σ|^(1/2)        2                      │
│                                                                │
│  Donde:                                                        │
│    μ = vector de medias (d dimensiones)                        │
│    Σ = matriz de covarianza (d × d)                            │
│    |Σ| = determinante de Σ                                     │
│                                                                │
│  FORMA DE LA GAUSSIANA:                                        │
│                                                                │
│  Σ diagonal:           Σ completa:                             │
│     ○○○                   ╱○○╲                                 │
│    ○○○○○                 ╱○○○○╲                                │
│     ○○○                 ╱○○○○○○╲                               │
│  Círculo/elipse         Elipse rotada                          │
│  alineada               cualquier orientación                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Mezcla de Gaussianas

```
┌────────────────────────────────────────────────────────────────┐
│  GAUSSIAN MIXTURE MODEL                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Probabilidad de observar x:                                   │
│                                                                │
│           K                                                    │
│  p(x) = Σ  πₖ · N(x | μₖ, Σₖ)                                  │
│          k=1                                                   │
│                                                                │
│  Donde:                                                        │
│    K = número de componentes (clusters)                        │
│    πₖ = peso del componente k (Σπₖ = 1)                        │
│    μₖ = media del componente k                                 │
│    Σₖ = covarianza del componente k                            │
│    N(x|μ,Σ) = distribución Gaussiana                           │
│                                                                │
│  PARÁMETROS A APRENDER:                                        │
│    • K medias (μ₁, μ₂, ..., μₖ)                                │
│    • K matrices de covarianza (Σ₁, Σ₂, ..., Σₖ)                │
│    • K-1 pesos (π₁, π₂, ..., πₖ) (el último es 1 - suma)       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 3. Algoritmo EM (Expectation-Maximization)

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  ALGORITMO EM                                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Problema: No sabemos qué Gaussiana generó cada punto          │
│            (variable latente/oculta)                           │
│                                                                │
│  Solución: Algoritmo EM (iterativo)                            │
│                                                                │
│  E-STEP (Expectation):                                         │
│    Dado los parámetros actuales, calcular la probabilidad      │
│    de que cada punto pertenezca a cada Gaussiana               │
│                                                                │
│    γₖ(xᵢ) = P(componente k | xᵢ)                               │
│           = πₖ · N(xᵢ|μₖ,Σₖ) / Σⱼ πⱼ · N(xᵢ|μⱼ,Σⱼ)            │
│                                                                │
│  M-STEP (Maximization):                                        │
│    Dado las probabilidades, actualizar los parámetros          │
│                                                                │
│    μₖ = Σᵢ γₖ(xᵢ)·xᵢ / Σᵢ γₖ(xᵢ)  (media ponderada)           │
│    Σₖ = ... (covarianza ponderada)                             │
│    πₖ = Σᵢ γₖ(xᵢ) / N  (proporción promedio)                   │
│                                                                │
│  REPETIR hasta convergencia                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización EM

```
ITERACIÓN 0: Inicialización aleatoria
──────────────────────────────────────
         ●  ●   ●
        ●  ●●  ●  ●            ○ ○  ○
         ●  ●●  ●             ○ ○ ○ ○
                              ○ ○  ○
        ★₁              ★₂
     (μ₁ inicial)    (μ₂ inicial)


ITERACIÓN 1: E-Step (calcular responsabilidades)
─────────────────────────────────────────────────
Cada punto tiene probabilidad de pertenecer a cada cluster:

    Punto A: P(cluster 1) = 0.95, P(cluster 2) = 0.05
    Punto B: P(cluster 1) = 0.30, P(cluster 2) = 0.70
    ...


ITERACIÓN 1: M-Step (actualizar parámetros)
────────────────────────────────────────────
    μ₁ = promedio ponderado de puntos con alta P(cluster 1)
    Σ₁ = covarianza ponderada
    π₁ = proporción promedio de pertenencia a cluster 1


ITERACIÓN N: Convergencia
─────────────────────────
         ●  ●   ●
        ●  ★₁  ●  ●            ○ ○  ○
         ●  ●●  ●             ○ ★₂ ○ ○
                              ○ ○  ○

    ★ = centros finales (óptimos)
    Elipses de covarianza definidas por Σ
```

## 4. Implementación en Python

### Código Básico

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100),
    np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 100),
    np.random.multivariate_normal([10, 0], [[2, 0], [0, 0.5]], 100),
])

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GMM
gmm = GaussianMixture(
    n_components=3,           # Número de Gaussianas
    covariance_type='full',   # Tipo de covarianza
    n_init=10,                # Inicializaciones
    random_state=42
)

# Fit
gmm.fit(X_scaled)

# Predicciones HARD (como K-Means)
labels = gmm.predict(X_scaled)

# Predicciones SOFT (probabilidades)
probs = gmm.predict_proba(X_scaled)

print(f"Medias:\n{gmm.means_}")
print(f"\nPesos de componentes: {gmm.weights_}")
print(f"\nLog-likelihood: {gmm.score(X_scaled):.2f}")
print(f"\nConvergencia: {gmm.converged_}")
print(f"Iteraciones: {gmm.n_iter_}")

# Ejemplo de probabilidades soft
print(f"\nPunto 0 probabilidades: {probs[0]}")
print(f"  Cluster asignado: {labels[0]}")
```

### Tipos de Covarianza

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

"""
TIPOS DE COVARIANZA:

  'full': Cada componente tiene su propia matriz de covarianza
          → Elipses de cualquier forma y orientación
          → Más flexible pero más parámetros

  'tied': Todos los componentes comparten la misma covarianza
          → Misma forma, diferente centro
          → Menos parámetros

  'diag': Covarianza diagonal (ejes alineados)
          → Elipses alineadas con los ejes
          → Menos parámetros que 'full'

  'spherical': Covarianza esférica (una sola varianza)
               → Círculos (como K-Means)
               → Menos parámetros
"""

cov_types = ['spherical', 'diag', 'tied', 'full']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, cov_type in zip(axes, cov_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type,
                          random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=20)
    ax.set_title(f'covariance_type={cov_type}')

plt.tight_layout()
plt.show()
```

### Elegir Número de Componentes

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def elegir_n_componentes(X, max_k=10):
    """Usa BIC y AIC para elegir número de componentes"""

    bics = []
    aics = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    # Visualizar
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_range, bics, 'bo-', label='BIC')
    ax.plot(K_range, aics, 'ro-', label='AIC')
    ax.set_xlabel('Número de componentes')
    ax.set_ylabel('Criterio de información')
    ax.set_title('Selección del número de componentes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # Mejor K (menor BIC)
    best_k = K_range[np.argmin(bics)]
    print(f"Mejor K según BIC: {best_k}")

    return best_k

best_k = elegir_n_componentes(X_scaled)
```

### Visualizar Elipses de Covarianza

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

def plot_gmm_ellipses(gmm, X, ax=None):
    """Visualiza GMM con elipses de covarianza"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    labels = gmm.predict(X)

    # Plot puntos
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=30, alpha=0.6)

    # Plot elipses de covarianza
    for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
        # Eigenvalores y eigenvectores
        if gmm.covariance_type == 'full':
            v, w = np.linalg.eigh(cov)
        elif gmm.covariance_type == 'diag':
            v = cov
            w = np.eye(len(cov))
        elif gmm.covariance_type == 'spherical':
            v = [cov, cov]
            w = np.eye(2)
        else:  # tied
            v, w = np.linalg.eigh(gmm.covariances_)

        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% de los puntos
        angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))

        # Dibujar elipse
        ellipse = Ellipse(mean, v[0], v[1], angle=angle,
                         fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(ellipse)

        # Marcar centro
        ax.scatter(mean[0], mean[1], c='red', marker='x', s=200, linewidths=3)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('GMM con elipses de covarianza')
    plt.colorbar(scatter, label='Cluster')

    return ax

# Uso
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
plot_gmm_ellipses(gmm, X_scaled)
plt.show()
```

## 5. Detección de Anomalías con GMM

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  GMM PARA DETECCIÓN DE ANOMALÍAS                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Los puntos normales tienen ALTA probabilidad            │
│        Las anomalías tienen BAJA probabilidad                  │
│                                                                │
│         ╭───────╮                                              │
│       ╱  ●  ●    ╲                                             │
│      │  ●  ●  ●   │    ← Alta probabilidad (normal)            │
│      │   ●  ●  ●  │                                            │
│       ╲  ●  ●    ╱                                             │
│         ╰───────╯                                              │
│                                                                │
│                            ✗  ← Baja probabilidad (anomalía)   │
│                                                                │
│  MÉTODO:                                                       │
│    1. Entrenar GMM con datos normales                          │
│    2. Calcular log-likelihood de nuevos datos                  │
│    3. Si log-likelihood < umbral → ANOMALÍA                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código Detección de Anomalías

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Entrenar GMM con datos normales
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_train_normal)

# Calcular log-likelihood
log_likelihood = gmm.score_samples(X_test)

# Definir umbral (percentil bajo de los datos de entrenamiento)
threshold = np.percentile(gmm.score_samples(X_train_normal), 5)

# Detectar anomalías
anomalies = log_likelihood < threshold

print(f"Umbral: {threshold:.2f}")
print(f"Anomalías detectadas: {anomalies.sum()}")

# Visualizar
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(gmm.score_samples(X_train_normal), bins=50, alpha=0.7, label='Normal')
plt.axvline(threshold, color='r', linestyle='--', label='Umbral')
plt.xlabel('Log-likelihood')
plt.ylabel('Frecuencia')
plt.title('Distribución de Log-likelihood (Train)')
plt.legend()

plt.subplot(1, 2, 2)
colors = ['green' if not a else 'red' for a in anomalies]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Detección de Anomalías')
plt.show()
```

## 6. Ejemplo Práctico: Clasificación de Tráfico de Red

```python
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Simular tráfico de red
np.random.seed(42)

# Tipo 1: Web browsing (la mayoría)
web = np.column_stack([
    np.random.normal(500, 100, 600),    # bytes
    np.random.normal(10, 3, 600),       # paquetes
    np.random.normal(0.5, 0.1, 600),    # duración
])

# Tipo 2: Streaming (menos común)
streaming = np.column_stack([
    np.random.normal(50000, 10000, 200),  # muchos bytes
    np.random.normal(100, 20, 200),       # muchos paquetes
    np.random.normal(300, 60, 200),       # larga duración
])

# Tipo 3: Malicioso (raro)
malicioso = np.column_stack([
    np.random.normal(100, 50, 50),        # pocos bytes
    np.random.normal(5, 2, 50),           # pocos paquetes
    np.random.normal(0.01, 0.005, 50),    # muy corto (scan)
])

# Combinar
X = np.vstack([web, streaming, malicioso])
tipos = ['Web']*600 + ['Streaming']*200 + ['Malicioso']*50

print(f"Total conexiones: {len(X)}")
print(f"Distribución: Web=600, Streaming=200, Malicioso=50")

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, tipos_train, tipos_test = train_test_split(
    X_scaled, tipos, test_size=0.2, random_state=42, stratify=tipos
)

# Elegir número de componentes
print("\nEligiendo número de componentes...")
for k in range(1, 6):
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_train)
    print(f"  K={k}: BIC={gmm.bic(X_train):.0f}, AIC={gmm.aic(X_train):.0f}")

# GMM con 3 componentes
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_train)

# Predicciones
labels_test = gmm.predict(X_test)
probs_test = gmm.predict_proba(X_test)

# Análisis
print("\n" + "=" * 60)
print("ANÁLISIS DE COMPONENTES GMM")
print("=" * 60)

df_test = pd.DataFrame({
    'tipo_real': tipos_test,
    'cluster': labels_test,
    'prob_0': probs_test[:, 0],
    'prob_1': probs_test[:, 1],
    'prob_2': probs_test[:, 2],
    'max_prob': probs_test.max(axis=1)
})

for cluster in range(3):
    mask = df_test['cluster'] == cluster
    print(f"\nCluster {cluster}:")
    print(f"  Peso del componente: {gmm.weights_[cluster]:.3f}")
    print(f"  Muestras en test: {mask.sum()}")
    print(f"  Tipos reales:")
    for tipo, count in df_test[mask]['tipo_real'].value_counts().items():
        print(f"    - {tipo}: {count}")

# Detección de anomalías (tráfico malicioso)
print("\n" + "=" * 60)
print("DETECCIÓN DE ANOMALÍAS")
print("=" * 60)

# Log-likelihood
log_likelihood = gmm.score_samples(X_test)

# Umbral: percentil 10 de los datos de entrenamiento
threshold = np.percentile(gmm.score_samples(X_train), 10)
print(f"Umbral de anomalía: {threshold:.2f}")

# Detectar
anomalias_pred = log_likelihood < threshold

# Evaluar
df_test['anomalia_pred'] = anomalias_pred
df_test['log_likelihood'] = log_likelihood

print("\nResultados de detección:")
print(f"  Total anomalías predichas: {anomalias_pred.sum()}")

# Comparar con tráfico malicioso real
maliciosos_reales = df_test['tipo_real'] == 'Malicioso'
maliciosos_detectados = (anomalias_pred & maliciosos_reales).sum()
total_maliciosos = maliciosos_reales.sum()

print(f"  Maliciosos reales: {total_maliciosos}")
print(f"  Maliciosos detectados: {maliciosos_detectados}")
print(f"  Tasa de detección: {maliciosos_detectados/total_maliciosos*100:.1f}%")

# Falsos positivos
fp = (anomalias_pred & ~maliciosos_reales).sum()
print(f"  Falsos positivos: {fp}")

# Visualización
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Clusters GMM
ax = axes[0]
scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1],
                    c=labels_test, cmap='viridis', alpha=0.6)
ax.set_title('Clusters GMM')
plt.colorbar(scatter, ax=ax, label='Cluster')

# Plot 2: Tipos reales
ax = axes[1]
tipo_colors = {'Web': 'blue', 'Streaming': 'green', 'Malicioso': 'red'}
colors = [tipo_colors[t] for t in tipos_test]
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=colors, alpha=0.6)
ax.set_title('Tipos Reales')

# Plot 3: Anomalías detectadas
ax = axes[2]
colors = ['red' if a else 'green' for a in anomalias_pred]
ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=colors, alpha=0.6)
ax.set_title('Anomalías Detectadas (rojo)')

plt.tight_layout()
plt.show()

# Mostrar ejemplos de incertidumbre
print("\n" + "=" * 60)
print("EJEMPLOS DE ASIGNACIÓN PROBABILÍSTICA")
print("=" * 60)

# Casos con alta incertidumbre
uncertain = df_test[df_test['max_prob'] < 0.7].head(5)
print("\nCasos con alta incertidumbre (prob < 70%):")
for _, row in uncertain.iterrows():
    print(f"  Tipo real: {row['tipo_real']}")
    print(f"  Probabilidades: C0={row['prob_0']:.2f}, "
          f"C1={row['prob_1']:.2f}, C2={row['prob_2']:.2f}")
    print()
```

## 7. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE GMM                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Asignación SOFT (probabilidades, no solo etiquetas)         │
│  ✓ Clusters elípticos (no solo esféricos)                      │
│  ✓ Modelo generativo (puede generar nuevos datos)              │
│  ✓ Framework probabilístico riguroso                           │
│  ✓ BIC/AIC para selección de modelo                            │
│  ✓ Útil para detección de anomalías                            │
│  ✓ Maneja clusters de diferente tamaño/forma                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE GMM                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Requiere especificar número de componentes                  │
│  ✗ Asume datos Gaussianos (puede no ser válido)                │
│  ✗ Sensible a inicialización                                   │
│  ✗ Puede converger a mínimo local                              │
│  ✗ Problemas con covarianza singular (pocos datos)             │
│  ✗ Más lento que K-Means                                       │
│  ✗ Muchos parámetros con covarianza 'full'                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. Cuándo Usar GMM

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Necesitas probabilidades de pertenencia                     │
│  ✓ Clusters tienen diferentes formas/tamaños                   │
│  ✓ Detección de anomalías                                      │
│  ✓ Datos aproximadamente Gaussianos                            │
│  ✓ Modelo generativo (sampling)                                │
│  ✓ Clusters que se solapan                                     │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Clusters no Gaussianos (formas arbitrarias → DBSCAN)        │
│  ✗ Muchos clusters (>20)                                       │
│  ✗ Alta dimensionalidad sin reducción                          │
│  ✗ Pocos datos por cluster (covarianza singular)               │
│  ✗ Solo necesitas etiquetas hard (usar K-Means, más rápido)    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 9. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  GAUSSIAN MIXTURE MODELS - RESUMEN                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Datos generados por mezcla de Gaussianas                    │
│    Cada componente: media (μ), covarianza (Σ), peso (π)        │
│                                                                │
│  ALGORITMO EM:                                                 │
│    E-Step: Calcular probabilidades de pertenencia              │
│    M-Step: Actualizar parámetros (μ, Σ, π)                     │
│    Repetir hasta convergencia                                  │
│                                                                │
│  TIPOS DE COVARIANZA:                                          │
│    'full': Elipses de cualquier forma (más flexible)           │
│    'diag': Elipses alineadas con ejes                          │
│    'spherical': Círculos (como K-Means)                        │
│    'tied': Misma forma para todos los componentes              │
│                                                                │
│  SELECCIÓN DE COMPONENTES:                                     │
│    BIC (Bayesian Information Criterion) - preferido            │
│    AIC (Akaike Information Criterion)                          │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de anomalías (baja probabilidad)                │
│    • Clasificación de tráfico con incertidumbre                │
│    • Modelado de comportamiento normal                         │
│    • Detección de intrusiones                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Con esto completamos los algoritmos de clustering:**
1. Introducción al Clustering
2. K-Means
3. DBSCAN
4. Clustering Jerárquico
5. Gaussian Mixture Models
