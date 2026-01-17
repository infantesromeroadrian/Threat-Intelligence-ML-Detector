# Álgebra Lineal Numérica para Machine Learning

## 1. Introducción: Por Qué Importa

### Las Matemáticas Ocultas del ML

Detrás de casi todo algoritmo de ML hay **operaciones con matrices**:

```
┌─────────────────────────────────────────────────────────────┐
│  MATRICES EN TODAS PARTES                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DATOS:                                                      │
│    X ∈ R^(n×d) = n muestras, d features                    │
│    Los datos SON una matriz                                  │
│                                                              │
│  MODELOS LINEALES:                                           │
│    y = Xθ  →  Multiplicación matriz-vector                 │
│    θ* = (X^TX)^(-1)X^Ty  →  Inversión, productos           │
│                                                              │
│  REDES NEURONALES:                                           │
│    h = σ(Wx + b)  →  Multiplicación en cada capa           │
│    Backward: ∂L/∂W = δ · x^T                               │
│                                                              │
│  NLP (Embeddings):                                           │
│    E ∈ R^(vocab×dim)  →  Matriz de embeddings              │
│    word_vec = E[word_id]  →  Lookup de fila                │
│                                                              │
│  RECOMENDADORES:                                             │
│    R ≈ UV^T  →  Factorización matricial                    │
│                                                              │
│  PCA:                                                        │
│    Proyección sobre eigenvectores de X^TX                   │
└─────────────────────────────────────────────────────────────┘
```

### Objetivos de Este Documento

```
┌─────────────────────────────────────────────────────────────┐
│  QUÉ APRENDEREMOS                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. EIGENDECOMPOSITION                                       │
│     • Valores y vectores propios                            │
│     • PCA desde esta perspectiva                            │
│                                                              │
│  2. SVD (Singular Value Decomposition)                       │
│     • La descomposición más importante en ML                │
│     • Aplicaciones: PCA, compresión, NLP                    │
│                                                              │
│  3. MATRIX FACTORIZATION                                     │
│     • Sistemas de recomendación                              │
│     • NMF, UV decomposition                                  │
│                                                              │
│  4. LOW-RANK APPROXIMATIONS                                  │
│     • Compresión de matrices                                │
│     • Aceleración de modelos                                │
│                                                              │
│  5. CONDICIONAMIENTO NUMÉRICO                                │
│     • Estabilidad de algoritmos                              │
│     • Por qué algunas matrices son "malas"                  │
└─────────────────────────────────────────────────────────────┘
```

## 2. Eigendecomposition: Valores y Vectores Propios

### Definición de Eigenpares

```
┌─────────────────────────────────────────────────────────────┐
│  EIGENVECTORES Y EIGENVALORES                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para una matriz cuadrada A ∈ R^(n×n):                      │
│                                                              │
│    A·v = λ·v                                                │
│                                                              │
│  Donde:                                                      │
│    v = eigenvector (vector propio)                          │
│    λ = eigenvalue (valor propio)                            │
│                                                              │
│  Interpretación geométrica:                                  │
│    A transforma v, pero SOLO lo escala por λ               │
│    v mantiene su DIRECCIÓN después de la transformación    │
│                                                              │
│  Propiedades:                                                │
│    • Una matriz n×n tiene n eigenvalores (contando multip.)│
│    • Eigenvalores pueden ser complejos (para A no simétrica)│
│    • Para A simétrica: eigenvalores reales, eigenvec. ortog.│
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Eigenvectores

```
Matriz A = [[2, 1],    transforma vectores:
            [1, 2]]

ANTES (círculo)                DESPUÉS (elipse)

    y│                              y│
     │      ╱                        │         ╱
     │    ╱   ↑ v₂                   │       ╱   ↑ λ₂·v₂
     │  ●───────●                    │    ●────────────●
     │ ╱        │                    │  ╱              │
  ───●──────────●── x             ───●─────────────────●── x
     │          │                    │                 │
     │          │                    │                 │
     │    ← v₁  │                    │    ← λ₁·v₁     │
     │          │                    │                 │

Los eigenvectores v₁, v₂ solo se escalan, no rotan.
λ₁ = 3, λ₂ = 1 (eigenvalores)

v₁ = [1, 1]/√2  → se estira ×3
v₂ = [-1, 1]/√2 → se mantiene igual (×1)
```

### Eigendecomposition Completa

```
┌─────────────────────────────────────────────────────────────┐
│  EIGENDECOMPOSITION                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para A simétrica (o más general, diagonalizable):          │
│                                                              │
│    A = V Λ V^(-1)                                           │
│                                                              │
│  Donde:                                                      │
│    V = [v₁ | v₂ | ... | vₙ]  matriz de eigenvectores       │
│    Λ = diag(λ₁, λ₂, ..., λₙ)  matriz diagonal eigenvalores │
│                                                              │
│  Para A SIMÉTRICA (caso común en ML):                       │
│    V es ortogonal: V^(-1) = V^T                             │
│    A = V Λ V^T                                              │
│                                                              │
│  Utilidad:                                                   │
│    • Potencias: A^k = V Λ^k V^T                             │
│    • Inversas: A^(-1) = V Λ^(-1) V^T                        │
│    • Funciones: f(A) = V f(Λ) V^T                           │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def eigendecomposition_demo():
    """Demuestra eigendecomposition y sus propiedades."""

    # Matriz simétrica (común en ML: covariance matrices)
    A = np.array([
        [4, 2, 0],
        [2, 5, 3],
        [0, 3, 6]
    ])

    # Calcular eigenpares
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # eigh para simétricas

    print("Matriz A:")
    print(A)
    print(f"\nEigenvalores: {eigenvalues}")
    print(f"\nEigenvectores (columnas de V):")
    print(eigenvectors)

    # Verificar A = V Λ V^T
    Lambda = np.diag(eigenvalues)
    V = eigenvectors
    A_reconstructed = V @ Lambda @ V.T

    print(f"\nReconstrucción A = V Λ V^T:")
    print(A_reconstructed)
    print(f"\nError de reconstrucción: {np.linalg.norm(A - A_reconstructed):.2e}")

    # Verificar ortogonalidad de V
    print(f"\nV^T V (debería ser I):")
    print(np.round(V.T @ V, 10))

    return eigenvalues, eigenvectors

eigenvalues, V = eigendecomposition_demo()
```

### Eigendecomposition de Covarianza = PCA

```
┌─────────────────────────────────────────────────────────────┐
│  PCA DESDE EIGENDECOMPOSITION                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Datos: X ∈ R^(n×d), centrados (media 0 por columna)       │
│                                                              │
│  Matriz de covarianza: C = (1/n) X^T X ∈ R^(d×d)           │
│                                                              │
│  PCA = Eigendecomposition de C:                              │
│    C = V Λ V^T                                              │
│                                                              │
│  Donde:                                                      │
│    • Columnas de V = direcciones principales (PC)           │
│    • λᵢ = varianza explicada por cada PC                   │
│    • λ₁ ≥ λ₂ ≥ ... ≥ λₐ (ordenados)                       │
│                                                              │
│  Proyección a k dimensiones:                                 │
│    X_reducido = X · V[:, :k]                                │
│                                                              │
│  Varianza preservada:                                        │
│    (Σᵢ₌₁ᵏ λᵢ) / (Σᵢ₌₁ᵈ λᵢ) × 100%                        │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def pca_via_eigendecomposition(X: np.ndarray, n_components: int = 2):
    """
    PCA usando eigendecomposition de la matriz de covarianza.

    Args:
        X: Datos (n_samples, n_features)
        n_components: Número de componentes a mantener

    Returns:
        X_reduced: Datos proyectados
        components: Direcciones principales
        explained_variance_ratio: Varianza explicada por componente
    """
    n, d = X.shape

    # 1. Centrar datos
    X_centered = X - X.mean(axis=0)

    # 2. Matriz de covarianza
    C = (1/n) * X_centered.T @ X_centered

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # 4. Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Seleccionar top-k componentes
    components = eigenvectors[:, :n_components]

    # 6. Proyectar datos
    X_reduced = X_centered @ components

    # 7. Calcular varianza explicada
    total_var = eigenvalues.sum()
    explained_variance_ratio = eigenvalues[:n_components] / total_var

    return X_reduced, components, explained_variance_ratio


# Ejemplo
np.random.seed(42)
n, d = 100, 5
X = np.random.randn(n, d)
# Crear correlación entre features
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n)
X[:, 2] = X[:, 0] - 0.3 * np.random.randn(n)

X_pca, pcs, var_ratio = pca_via_eigendecomposition(X, n_components=2)

print("PCA via Eigendecomposition:")
print(f"  Shape original: {X.shape}")
print(f"  Shape reducido: {X_pca.shape}")
print(f"  Varianza explicada PC1: {var_ratio[0]:.2%}")
print(f"  Varianza explicada PC2: {var_ratio[1]:.2%}")
print(f"  Total con 2 PCs: {var_ratio.sum():.2%}")
```

## 3. SVD: La Descomposición Más Importante

### Definición de SVD

```
┌─────────────────────────────────────────────────────────────┐
│  SINGULAR VALUE DECOMPOSITION                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para CUALQUIER matriz A ∈ R^(m×n):                         │
│                                                              │
│    A = U Σ V^T                                              │
│                                                              │
│  Donde:                                                      │
│    U ∈ R^(m×m) = Vectores singulares izquierdos (ortogonal)│
│    Σ ∈ R^(m×n) = Valores singulares (diagonal, ≥ 0)        │
│    V ∈ R^(n×n) = Vectores singulares derechos (ortogonal)  │
│                                                              │
│  Valores singulares:                                         │
│    σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0  (r = rango de A)               │
│                                                              │
│  Forma "thin" (eficiente):                                   │
│    A = U_r Σ_r V_r^T                                        │
│    U_r ∈ R^(m×r), Σ_r ∈ R^(r×r), V_r ∈ R^(n×r)            │
│                                                              │
│  Diferencia con Eigen:                                       │
│    • Eigen: solo matrices cuadradas                         │
│    • SVD: cualquier matriz rectangular                      │
│    • SVD siempre existe y es única (salvo signos)          │
└─────────────────────────────────────────────────────────────┘
```

### Visualización Geométrica de SVD

```
SVD = Rotación + Escalado + Rotación

      V^T            Σ              U
   (rotación)    (escalado)    (rotación)

    ●────●         ●────────●      ●──────●
    │    │    →    │        │   →  │      │
    │    │         │        │      │      │
    ●────●         ●────────●      ●──────●

  Dominio       Ejes principales   Codominio
  (espacio n)                      (espacio m)

Para vector x:
  1. V^T x = rota x a base de vectores singulares derechos
  2. Σ (V^T x) = escala cada coordenada por σᵢ
  3. U Σ V^T x = rota al espacio final

Los σᵢ son las "amplificaciones" en cada dirección principal
```

### Relación SVD-Eigen

```
┌─────────────────────────────────────────────────────────────┐
│  CONEXIÓN ENTRE SVD Y EIGENDECOMPOSITION                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Si A = U Σ V^T, entonces:                                  │
│                                                              │
│    A^T A = (U Σ V^T)^T (U Σ V^T)                           │
│          = V Σ^T U^T U Σ V^T                                │
│          = V Σ² V^T                                         │
│                                                              │
│  → Eigenvalores de A^T A = σᵢ²                             │
│  → Eigenvectores de A^T A = columnas de V                  │
│                                                              │
│  Similarmente:                                               │
│    A A^T = U Σ² U^T                                         │
│                                                              │
│  → Eigenvalores de A A^T = σᵢ²                             │
│  → Eigenvectores de A A^T = columnas de U                  │
│                                                              │
│  IMPLICACIÓN PARA PCA:                                       │
│    PCA via covarianza = PCA via SVD de datos centrados     │
│    Es más estable numéricamente usar SVD directamente       │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def svd_demo():
    """Demuestra SVD y su relación con eigendecomposition."""

    # Matriz rectangular
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ], dtype=float)

    print("Matriz A (4×3):")
    print(A)

    # SVD completa
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"\nU (left singular vectors) shape: {U.shape}")
    print(f"Valores singulares: {s}")
    print(f"V^T (right singular vectors) shape: {Vt.shape}")

    # Reconstruir
    S = np.diag(s)
    A_reconstructed = U @ S @ Vt
    print(f"\nError de reconstrucción: {np.linalg.norm(A - A_reconstructed):.2e}")

    # Verificar relación con A^T A
    AtA = A.T @ A
    eigenvalues_AtA, _ = np.linalg.eigh(AtA)
    eigenvalues_AtA = np.sort(eigenvalues_AtA)[::-1]

    print(f"\nVerificación σ² = eigenvalores de A^T A:")
    print(f"  σ²: {s**2}")
    print(f"  eigenvalores A^T A: {eigenvalues_AtA}")

    return U, s, Vt

U, s, Vt = svd_demo()
```

### PCA via SVD (Método Preferido)

```
┌─────────────────────────────────────────────────────────────┐
│  PCA VIA SVD - MÁS EFICIENTE Y ESTABLE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Datos centrados: X_c = X - mean(X)                         │
│                                                              │
│  SVD de X_c:                                                 │
│    X_c = U Σ V^T                                            │
│                                                              │
│  Entonces:                                                   │
│    • Componentes principales = columnas de V                │
│    • Scores = X_c V = U Σ                                   │
│    • Varianza explicada ∝ σᵢ²                              │
│                                                              │
│  Ventajas sobre eigendecomposition de covarianza:           │
│    1. No calculamos X^T X (puede ser muy grande)           │
│    2. Más estable numéricamente                             │
│    3. Funciona directamente con matrices rectangulares     │
│                                                              │
│  sklearn.decomposition.PCA usa SVD internamente             │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def pca_via_svd(X: np.ndarray, n_components: int = 2):
    """
    PCA usando SVD (método preferido).

    Args:
        X: Datos (n_samples, n_features)
        n_components: Componentes a mantener

    Returns:
        X_transformed: Datos proyectados (scores)
        components: Direcciones principales (loadings)
        explained_variance_ratio: Varianza explicada
    """
    n, d = X.shape

    # 1. Centrar
    X_centered = X - X.mean(axis=0)

    # 2. SVD (thin/reduced)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Componentes = filas de V^T (columnas de V)
    components = Vt[:n_components, :]  # (n_components, n_features)

    # 4. Scores = U @ Σ (primeras k columnas)
    X_transformed = U[:, :n_components] * s[:n_components]

    # 5. Varianza explicada (proporcional a σ²)
    explained_variance = (s ** 2) / (n - 1)
    explained_variance_ratio = explained_variance[:n_components] / explained_variance.sum()

    return X_transformed, components, explained_variance_ratio


# Comparar con sklearn
from sklearn.decomposition import PCA

np.random.seed(42)
X = np.random.randn(100, 10)

# Nuestro PCA via SVD
X_our, comp_our, var_our = pca_via_svd(X, n_components=3)

# sklearn PCA
pca_sklearn = PCA(n_components=3)
X_sklearn = pca_sklearn.fit_transform(X)

print("Comparación con sklearn:")
print(f"  Diferencia en transformación: {np.abs(np.abs(X_our) - np.abs(X_sklearn)).max():.2e}")
print(f"  Varianza explicada (nuestra): {var_our}")
print(f"  Varianza explicada (sklearn): {pca_sklearn.explained_variance_ratio_}")
# Nota: signos pueden diferir (eigenvectores son únicos salvo signo)
```

## 4. Low-Rank Approximations: Comprimiendo Matrices

### Teorema de Eckart-Young-Mirsky

```
┌─────────────────────────────────────────────────────────────┐
│  MEJOR APROXIMACIÓN DE RANGO-K                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema: Aproximar A con matriz de rango ≤ k              │
│                                                              │
│    min ||A - B||_F   sujeto a  rango(B) ≤ k                │
│     B                                                        │
│                                                              │
│  Solución (Eckart-Young): Usar SVD truncada                 │
│                                                              │
│    A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢ^T                                 │
│                                                              │
│        = U_k Σ_k V_k^T                                      │
│                                                              │
│  Error de aproximación:                                      │
│    ||A - A_k||_F = √(σ²ₖ₊₁ + σ²ₖ₊₂ + ... + σ²ᵣ)           │
│                                                              │
│  Interpretación:                                             │
│    Mantener los k valores singulares más grandes            │
│    descarta el "ruido" (singulares pequeños)               │
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Compresión

```
Matriz original A (1000 × 1000)        Comprimida A_k (k = 50)

Almacenamiento:                        Almacenamiento:
  1000 × 1000 = 1,000,000 floats        U_k: 1000 × 50 = 50,000
                                         Σ_k: 50 valores
                                         V_k: 1000 × 50 = 50,000
                                         Total: ~100,000 floats

                                       Compresión: 10× menos espacio

Error de reconstrucción depende de la distribución de σᵢ:

σᵢ│
   │●                           Caída rápida → buena compresión
   │ ●                          (información concentrada en pocos σ)
   │  ●●
   │    ●●●●
   │        ●●●●●●●●●●●●●●●
   └────────────────────────── i
        k
```

```python
import numpy as np

def low_rank_approximation(A: np.ndarray, k: int):
    """
    Aproximación de rango-k usando SVD truncada.

    Args:
        A: Matriz a aproximar
        k: Rango de la aproximación

    Returns:
        A_k: Aproximación de rango k
        compression_ratio: Ratio de compresión
        relative_error: Error relativo de Frobenius
    """
    m, n = A.shape

    # SVD completa
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncar a rango k
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # Reconstruir
    A_k = U_k @ np.diag(s_k) @ Vt_k

    # Métricas
    original_size = m * n
    compressed_size = m * k + k + n * k  # U_k + s_k + V_k
    compression_ratio = original_size / compressed_size

    relative_error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')

    return A_k, compression_ratio, relative_error, s


def analizar_compresion(A: np.ndarray):
    """Analiza diferentes niveles de compresión."""
    print(f"Matriz original: {A.shape}")
    print(f"Rango real: {np.linalg.matrix_rank(A)}")

    _, _, _, s = low_rank_approximation(A, k=1)

    print(f"\nValores singulares (top 10): {s[:10]}")

    print("\nCompresión a diferentes rangos:")
    print("-" * 50)
    print(f"{'k':>5} {'Ratio':>10} {'Error Rel.':>12} {'σ_k':>10}")
    print("-" * 50)

    for k in [1, 5, 10, 20, 50, 100]:
        if k > min(A.shape):
            break
        A_k, ratio, error, _ = low_rank_approximation(A, k)
        print(f"{k:>5} {ratio:>10.2f}x {error:>12.4%} {s[k-1]:>10.4f}")


# Ejemplo: Imagen como matriz
np.random.seed(42)

# Simular imagen con estructura (rango bajo aproximado)
# Imagen real = base de rango bajo + ruido
rank_true = 20
m, n = 256, 256
U_true = np.random.randn(m, rank_true)
V_true = np.random.randn(n, rank_true)
noise = 0.1 * np.random.randn(m, n)
imagen = U_true @ V_true.T + noise

print("=== Compresión de 'imagen' con estructura de rango bajo ===")
analizar_compresion(imagen)
```

### Aplicación: Compresión de Embeddings

```
┌─────────────────────────────────────────────────────────────┐
│  COMPRESIÓN DE WORD EMBEDDINGS                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema típico:                                            │
│    • Vocabulario: 100,000 palabras                          │
│    • Dimensión embedding: 300                               │
│    • Matriz E: 100,000 × 300 = 30 millones de floats       │
│    • Tamaño: ~120 MB (float32)                              │
│                                                              │
│  SVD comprimida (k = 100):                                   │
│    E ≈ U_k @ diag(s_k) @ V_k^T                             │
│    • U_k: 100,000 × 100 = 10M floats                       │
│    • s_k: 100 floats                                        │
│    • V_k: 300 × 100 = 30K floats                           │
│    • Total: ~10M floats ≈ 40 MB                            │
│                                                              │
│  Compresión 3× con pérdida mínima de calidad               │
│                                                              │
│  En inferencia:                                              │
│    embedding(word) = U_k[word_id] @ diag(s_k) @ V_k         │
│    O precalcular: U_k_scaled = U_k @ diag(s_k)             │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def comprimir_embeddings(embeddings: np.ndarray, k: int):
    """
    Comprime matriz de embeddings usando SVD.

    Args:
        embeddings: Matriz (vocab_size, embedding_dim)
        k: Dimensión reducida

    Returns:
        U_scaled: Matriz comprimida para lookup eficiente
        V: Matriz de proyección (para proyectar queries)
    """
    U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)

    # Truncar
    U_k = U[:, :k]
    s_k = s[:k]
    V_k = Vt[:k, :].T  # (embedding_dim, k)

    # Precalcular U @ diag(s) para lookup eficiente
    U_scaled = U_k * s_k  # Broadcasting: (vocab, k) * (k,)

    return U_scaled, V_k


def demo_compresion_embeddings():
    """Demo de compresión de embeddings."""
    np.random.seed(42)

    # Simular embeddings (en realidad serían GloVe, Word2Vec, etc.)
    vocab_size = 50000
    embed_dim = 300
    embeddings = np.random.randn(vocab_size, embed_dim)

    print(f"Embeddings originales: {embeddings.shape}")
    print(f"Tamaño: {embeddings.nbytes / 1e6:.1f} MB")

    # Comprimir a diferentes k
    for k in [50, 100, 150]:
        U_scaled, V = comprimir_embeddings(embeddings, k)

        # Tamaño comprimido
        size_compressed = U_scaled.nbytes + V.nbytes
        ratio = embeddings.nbytes / size_compressed

        # Error de reconstrucción
        E_approx = U_scaled @ V.T
        error = np.linalg.norm(embeddings - E_approx, 'fro') / np.linalg.norm(embeddings, 'fro')

        print(f"\nk = {k}:")
        print(f"  Tamaño: {size_compressed / 1e6:.1f} MB (ratio {ratio:.1f}×)")
        print(f"  Error relativo: {error:.2%}")

        # Verificar similaridad preservada
        # Tomar 2 palabras "similares" (cercanas en el espacio original)
        w1, w2 = 100, 101
        sim_original = np.dot(embeddings[w1], embeddings[w2]) / (
            np.linalg.norm(embeddings[w1]) * np.linalg.norm(embeddings[w2])
        )
        sim_compressed = np.dot(E_approx[w1], E_approx[w2]) / (
            np.linalg.norm(E_approx[w1]) * np.linalg.norm(E_approx[w2])
        )
        print(f"  Similaridad coseno preservada: {sim_original:.4f} → {sim_compressed:.4f}")

demo_compresion_embeddings()
```

## 5. Matrix Factorization para Recomendadores

### El Problema de Recomendación

```
┌─────────────────────────────────────────────────────────────┐
│  COLLABORATIVE FILTERING VIA MATRIX FACTORIZATION          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Datos: Matriz de ratings R ∈ R^(users × items)             │
│                                                              │
│              Item₁  Item₂  Item₃  Item₄  Item₅              │
│    User₁  │   5      ?      3      ?      1                │
│    User₂  │   ?      4      ?      2      ?                │
│    User₃  │   3      ?      5      ?      4                │
│    User₄  │   ?      2      ?      5      ?                │
│                                                              │
│  Problema: Predecir los "?" (ratings faltantes)             │
│                                                              │
│  Idea: R ≈ U · V^T                                          │
│    U ∈ R^(users × k) = embeddings de usuarios               │
│    V ∈ R^(items × k) = embeddings de items                 │
│    k = dimensión latente (típico: 10-200)                  │
│                                                              │
│  Predicción: r̂_{ui} = u_user · v_item                      │
│    El rating es el producto escalar de embeddings           │
└─────────────────────────────────────────────────────────────┘
```

### Factorización Matricial como Minimización

```
┌─────────────────────────────────────────────────────────────┐
│  OBJETIVO DE MATRIX FACTORIZATION                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Minimizar (solo sobre ratings observados):                 │
│                                                              │
│    L = Σ (r_ui - u_user · v_item)² + λ(||U||²_F + ||V||²_F)│
│       (u,i)∈Ω                                               │
│                                                              │
│  Donde:                                                      │
│    Ω = conjunto de (usuario, item) con rating observado    │
│    λ = regularización para evitar overfitting               │
│                                                              │
│  Optimización:                                               │
│    • ALS (Alternating Least Squares): Fija U, resuelve V,  │
│      luego fija V, resuelve U. Repetir.                    │
│    • SGD: Actualiza U, V juntos con gradiente              │
│                                                              │
│  Extensiones:                                                │
│    • Bias: r̂_ui = μ + b_u + b_i + u · v                   │
│    • Temporal: factores varían con el tiempo               │
│    • Implicit feedback: clicks, views en vez de ratings    │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

class MatrixFactorization:
    """
    Factorización matricial para sistemas de recomendación.
    Implementación básica con SGD.
    """

    def __init__(self, n_factors: int = 10, lr: float = 0.01,
                 reg: float = 0.1, n_epochs: int = 100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.U = None
        self.V = None

    def fit(self, ratings: np.ndarray, mask: np.ndarray):
        """
        Entrena el modelo.

        Args:
            ratings: Matriz de ratings (users, items)
            mask: Máscara binaria (1 = observado, 0 = faltante)
        """
        n_users, n_items = ratings.shape

        # Inicializar factores aleatorios
        np.random.seed(42)
        self.U = np.random.randn(n_users, self.n_factors) * 0.1
        self.V = np.random.randn(n_items, self.n_factors) * 0.1

        # Índices de ratings observados
        observed = np.argwhere(mask == 1)

        losses = []
        for epoch in range(self.n_epochs):
            # Shuffle
            np.random.shuffle(observed)

            epoch_loss = 0
            for u, i in observed:
                # Predicción
                pred = self.U[u] @ self.V[i]
                error = ratings[u, i] - pred

                # Gradientes
                grad_u = -2 * error * self.V[i] + 2 * self.reg * self.U[u]
                grad_v = -2 * error * self.U[u] + 2 * self.reg * self.V[i]

                # Actualizar
                self.U[u] -= self.lr * grad_u
                self.V[i] -= self.lr * grad_v

                epoch_loss += error ** 2

            losses.append(epoch_loss / len(observed))

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: MSE = {losses[-1]:.4f}")

        return losses

    def predict(self, user: int, item: int) -> float:
        """Predice rating para (user, item)."""
        return self.U[user] @ self.V[item]

    def predict_all(self) -> np.ndarray:
        """Predice matriz completa."""
        return self.U @ self.V.T

    def recommend(self, user: int, n: int = 5,
                   already_rated: np.ndarray = None) -> np.ndarray:
        """
        Recomienda top-n items para un usuario.
        """
        scores = self.U[user] @ self.V.T

        if already_rated is not None:
            scores[already_rated] = -np.inf

        return np.argsort(scores)[::-1][:n]


def demo_recomendador():
    """Demo de sistema de recomendación."""
    np.random.seed(42)

    # Simular datos: usuarios con preferencias latentes
    n_users, n_items = 100, 50
    n_factors_true = 5

    # Factores latentes verdaderos
    U_true = np.random.randn(n_users, n_factors_true)
    V_true = np.random.randn(n_items, n_factors_true)
    R_true = U_true @ V_true.T + np.random.randn(n_users, n_items) * 0.5

    # Escalar a rango 1-5
    R_true = np.clip(R_true, 1, 5)

    # Simular datos faltantes (solo 20% observado)
    mask = np.random.rand(n_users, n_items) < 0.2
    R_observed = R_true * mask

    print(f"Matriz de ratings: {n_users} usuarios × {n_items} items")
    print(f"Ratings observados: {mask.sum()} de {n_users * n_items} ({mask.mean():.1%})")

    # Entrenar modelo
    mf = MatrixFactorization(n_factors=10, lr=0.01, reg=0.01, n_epochs=100)
    losses = mf.fit(R_observed, mask)

    # Evaluar en datos no observados
    R_pred = mf.predict_all()
    test_mask = ~mask
    mse_test = ((R_pred[test_mask] - R_true[test_mask]) ** 2).mean()
    print(f"\nMSE en ratings no observados: {mse_test:.4f}")

    # Recomendaciones para usuario 0
    user = 0
    items_rated = np.where(mask[user])[0]
    recommendations = mf.recommend(user, n=5, already_rated=items_rated)
    print(f"\nTop 5 recomendaciones para usuario {user}: {recommendations}")

demo_recomendador()
```

### NMF: Non-negative Matrix Factorization

```
┌─────────────────────────────────────────────────────────────┐
│  NMF (Non-negative Matrix Factorization)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Restricción adicional: U ≥ 0, V ≥ 0                       │
│                                                              │
│  A ≈ W · H,  donde W ≥ 0, H ≥ 0                            │
│                                                              │
│  Beneficios:                                                 │
│    • Interpretabilidad: factores son "partes"              │
│    • No hay cancelaciones negativas                        │
│    • Representa A como suma ponderada de componentes       │
│                                                              │
│  Aplicaciones:                                               │
│    • Topic modeling: documentos = mezcla de topics         │
│    • Image analysis: imágenes = combinación de partes     │
│    • Audio: separación de fuentes                          │
│    • Bioinformática: expresión génica                      │
│                                                              │
│  Algoritmo típico: Multiplicative update rules             │
│    W = W * ((A @ H.T) / (W @ H @ H.T + ε))                │
│    H = H * ((W.T @ A) / (W.T @ W @ H + ε))                │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def nmf(A: np.ndarray, k: int, max_iter: int = 200,
        tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-negative Matrix Factorization.
    A ≈ W @ H, con W, H ≥ 0.

    Args:
        A: Matriz no negativa (m, n)
        k: Número de componentes
        max_iter: Iteraciones máximas
        tol: Tolerancia para convergencia

    Returns:
        W: Factor izquierdo (m, k)
        H: Factor derecho (k, n)
    """
    m, n = A.shape
    eps = 1e-10

    # Inicialización aleatoria no negativa
    np.random.seed(42)
    W = np.abs(np.random.randn(m, k))
    H = np.abs(np.random.randn(k, n))

    for i in range(max_iter):
        # Guardar para verificar convergencia
        W_old = W.copy()

        # Update H
        numerator_H = W.T @ A
        denominator_H = W.T @ W @ H + eps
        H = H * numerator_H / denominator_H

        # Update W
        numerator_W = A @ H.T
        denominator_W = W @ H @ H.T + eps
        W = W * numerator_W / denominator_W

        # Verificar convergencia
        change = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + eps)
        if change < tol:
            print(f"Convergencia en iteración {i}")
            break

    return W, H


# Ejemplo: Topic modeling simple
def demo_nmf_topics():
    """Demo de NMF para descubrir topics en documentos."""
    np.random.seed(42)

    # Simular matriz documento-término
    # (en realidad usaríamos TF-IDF)
    n_docs, n_words = 100, 200
    n_topics = 5

    # Generar datos con estructura de topics
    topics = np.random.rand(n_topics, n_words)  # Distribución de palabras por topic
    doc_topics = np.random.rand(n_docs, n_topics)  # Mezcla de topics por doc

    # Matriz documento-término (con ruido)
    A = doc_topics @ topics + 0.01 * np.random.rand(n_docs, n_words)
    A = np.clip(A, 0, None)  # Asegurar no negatividad

    print(f"Matriz documento-término: {A.shape}")

    # Aplicar NMF
    W, H = nmf(A, k=n_topics)

    print(f"\nW (doc × topics): {W.shape}")
    print(f"H (topics × words): {H.shape}")

    # Error de reconstrucción
    A_approx = W @ H
    error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    print(f"Error relativo: {error:.2%}")

    # Mostrar top palabras por topic
    print("\nTop 5 palabras por topic:")
    for t in range(n_topics):
        top_words = np.argsort(H[t])[-5:][::-1]
        print(f"  Topic {t}: palabras {top_words}")

    # Mostrar topic dominante por documento (primeros 5)
    print("\nTopic dominante por documento (primeros 5):")
    for d in range(5):
        dominant_topic = np.argmax(W[d])
        print(f"  Doc {d}: Topic {dominant_topic} (peso {W[d, dominant_topic]:.2f})")

demo_nmf_topics()
```

## 6. Condicionamiento Numérico

### Por Qué Algunas Matrices Son "Malas"

```
┌─────────────────────────────────────────────────────────────┐
│  NÚMERO DE CONDICIÓN                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  El número de condición κ(A) mide sensibilidad a errores:  │
│                                                              │
│    κ(A) = ||A|| · ||A⁻¹|| = σ_max / σ_min                  │
│                                                              │
│  Interpretación:                                             │
│    κ ≈ 1     → Matriz bien condicionada, estable           │
│    κ >> 1   → Matriz mal condicionada, inestable           │
│    κ = ∞    → Matriz singular (no invertible)              │
│                                                              │
│  Efecto en sistemas lineales Ax = b:                        │
│    Error relativo en solución ≤ κ(A) × error relativo en b │
│                                                              │
│  Ejemplo:                                                    │
│    Si κ(A) = 10⁶ y b tiene error de 10⁻¹⁰                  │
│    entonces x puede tener error de 10⁻⁴ (!)                │
│                                                              │
│  Causas comunes de mal condicionamiento:                    │
│    • Features muy correlacionadas (multicolinealidad)      │
│    • Escalas muy diferentes entre features                  │
│    • Matrices casi singulares                               │
└─────────────────────────────────────────────────────────────┘
```

### Visualización de Condicionamiento

```
BIEN CONDICIONADA (κ ≈ 1)        MAL CONDICIONADA (κ >> 1)

Curvas de nivel (Ax = b):        Curvas de nivel:

    y│     ╭───╮                     y│        ╱────────╲
     │    ╱     ╲                     │       ╱          ╲
     │   │   ●   │  ← Círculos        │      ╱    ●       ╲  ← Elipses muy
     │    ╲     ╱     casi            │     ╱              ╲   alargadas
     │     ╰───╯                      │    ╱────────────────╲
     └───────────── x                 └─────────────────────── x

Pequeño cambio en b              Pequeño cambio en b
→ pequeño cambio en x           → GRAN cambio en x

El gradiente "salta" en          El gradiente varía mucho
todas las direcciones            según la dirección
```

```python
import numpy as np

def analizar_condicionamiento(A: np.ndarray, nombre: str = "A"):
    """Analiza el condicionamiento de una matriz."""

    # Valores singulares
    s = np.linalg.svd(A, compute_uv=False)

    # Número de condición
    kappa = s[0] / s[-1] if s[-1] > 1e-15 else np.inf

    print(f"\n=== Análisis de condicionamiento: {nombre} ===")
    print(f"Shape: {A.shape}")
    print(f"Rango: {np.linalg.matrix_rank(A)}")
    print(f"σ_max: {s[0]:.6f}")
    print(f"σ_min: {s[-1]:.6e}")
    print(f"κ(A) = σ_max/σ_min: {kappa:.2e}")

    if kappa < 10:
        print("→ Excelente condicionamiento")
    elif kappa < 1e4:
        print("→ Buen condicionamiento")
    elif kappa < 1e8:
        print("→ Condicionamiento moderado, cuidado con precisión")
    elif kappa < 1e12:
        print("→ Mal condicionamiento, resultados pueden ser inestables")
    else:
        print("→ CRÍTICO: Muy mal condicionada, considerar regularización")

    return kappa


def demo_condicionamiento():
    """Demuestra efecto del condicionamiento en regresión."""
    np.random.seed(42)
    n = 100

    # Caso 1: Features independientes (bien condicionada)
    X_buena = np.random.randn(n, 5)

    # Caso 2: Features correlacionadas (mal condicionada)
    X_mala = np.random.randn(n, 5)
    X_mala[:, 1] = X_mala[:, 0] + 1e-6 * np.random.randn(n)  # Casi colineal
    X_mala[:, 2] = X_mala[:, 0] - 1e-6 * np.random.randn(n)

    # Analizar X^T X (lo que invertimos en OLS)
    kappa_buena = analizar_condicionamiento(X_buena.T @ X_buena, "X^T X (independientes)")
    kappa_mala = analizar_condicionamiento(X_mala.T @ X_mala, "X^T X (colineales)")

    # Demostrar inestabilidad
    print("\n=== Efecto en regresión OLS ===")

    # Target
    y = X_buena[:, 0] + 0.1 * np.random.randn(n)

    # Resolver con perturbación pequeña
    for nombre, X, kappa in [("Bien cond.", X_buena, kappa_buena),
                               ("Mal cond.", X_mala, kappa_mala)]:
        # Solución original
        theta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Solución con y perturbado
        y_perturbed = y + 1e-10 * np.random.randn(n)
        theta_perturbed = np.linalg.lstsq(X, y_perturbed, rcond=None)[0]

        cambio_theta = np.linalg.norm(theta - theta_perturbed) / np.linalg.norm(theta)

        print(f"\n{nombre} (κ = {kappa:.2e}):")
        print(f"  Perturbación en y: 1e-10")
        print(f"  Cambio relativo en θ: {cambio_theta:.2e}")

demo_condicionamiento()
```

### Soluciones al Mal Condicionamiento

```
┌─────────────────────────────────────────────────────────────┐
│  TÉCNICAS PARA MEJORAR CONDICIONAMIENTO                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. REGULARIZACIÓN (Ridge/Tikhonov)                         │
│     (X^T X + λI)⁻¹ en vez de (X^T X)⁻¹                     │
│     λ > 0 aumenta σ_min → reduce κ                         │
│     Es la solución más común en ML                          │
│                                                              │
│  2. NORMALIZACIÓN / ESCALADO                                 │
│     Estandarizar features: (x - μ) / σ                      │
│     Evita diferencias de escala                             │
│                                                              │
│  3. USAR SVD / PSEUDOINVERSA                                 │
│     En vez de invertir: usar np.linalg.lstsq                │
│     Truncar valores singulares pequeños                     │
│                                                              │
│  4. ELIMINAR COLINEALIDAD                                    │
│     Feature selection                                        │
│     PCA antes de regresión                                  │
│                                                              │
│  5. PRECISIÓN AUMENTADA                                      │
│     float64 en vez de float32                               │
│     Algoritmos numéricamente estables                       │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def comparar_soluciones_ols(X: np.ndarray, y: np.ndarray):
    """
    Compara diferentes métodos para resolver OLS
    en matrices mal condicionadas.
    """
    n, d = X.shape

    print("=== Comparación de métodos OLS ===")
    print(f"Datos: {n} muestras, {d} features")

    kappa = np.linalg.cond(X)
    print(f"κ(X) = {kappa:.2e}")

    # Método 1: Normal equations (inestable si mal condicionada)
    try:
        XtX = X.T @ X
        theta_normal = np.linalg.solve(XtX, X.T @ y)
        error_normal = np.linalg.norm(X @ theta_normal - y)
        print(f"\n1. Normal equations: ||Xθ - y|| = {error_normal:.6e}")
    except np.linalg.LinAlgError:
        print("\n1. Normal equations: FALLÓ (singular)")
        theta_normal = None

    # Método 2: lstsq (usa SVD, más estable)
    theta_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    error_lstsq = np.linalg.norm(X @ theta_lstsq - y)
    print(f"2. lstsq (SVD): ||Xθ - y|| = {error_lstsq:.6e}")

    # Método 3: Ridge regularization
    lambdas = [1e-10, 1e-6, 1e-2, 1.0]
    for lam in lambdas:
        XtX_reg = X.T @ X + lam * np.eye(d)
        theta_ridge = np.linalg.solve(XtX_reg, X.T @ y)
        error_ridge = np.linalg.norm(X @ theta_ridge - y)
        kappa_reg = np.linalg.cond(XtX_reg)
        print(f"3. Ridge (λ={lam:.0e}): ||Xθ - y|| = {error_ridge:.6e}, κ = {kappa_reg:.2e}")

    # Método 4: SVD truncada
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Truncar valores singulares pequeños
    threshold = 1e-10 * s[0]
    s_inv = np.where(s > threshold, 1/s, 0)
    theta_svd_trunc = Vt.T @ np.diag(s_inv) @ U.T @ y
    error_svd = np.linalg.norm(X @ theta_svd_trunc - y)
    print(f"4. SVD truncada: ||Xθ - y|| = {error_svd:.6e}")


# Demo con matriz mal condicionada
np.random.seed(42)
n, d = 100, 10

# Crear matriz mal condicionada
X = np.random.randn(n, d)
# Hacer algunas columnas casi colineales
X[:, 1] = X[:, 0] + 1e-8 * np.random.randn(n)
X[:, 3] = 0.5 * X[:, 2] + 0.5 * X[:, 4] + 1e-8 * np.random.randn(n)

y = np.random.randn(n)

comparar_soluciones_ols(X, y)
```

## 7. Resumen: Mapa de Álgebra Lineal en ML

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 MAPA DE ÁLGEBRA LINEAL NUMÉRICA EN ML                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DESCOMPOSICIONES MATRICIALES                                            │
│  ────────────────────────────────                                        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  EIGENDECOMPOSITION              SVD                        │        │
│  │  A = V Λ V^T                     A = U Σ V^T               │        │
│  │                                                              │        │
│  │  • Solo matrices cuadradas       • Cualquier matriz         │        │
│  │  • Simétricas: real, ortogonal   • Siempre existe          │        │
│  │  • PCA (via covarianza)          • PCA (directo)           │        │
│  │  • Spectral clustering           • Compresión              │        │
│  │                                   • LSA/LSI en NLP          │        │
│  │                                   • Recomendadores          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
│  FACTORIZACIONES PARA ML                                                 │
│  ───────────────────────────                                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  LOW-RANK (UV^T)              NMF (WH, W,H ≥ 0)            │        │
│  │                                                              │        │
│  │  • Collaborative filtering    • Topic modeling              │        │
│  │  • Embeddings comprimidos     • Partes interpretables      │        │
│  │  • Predicción de ratings      • Mezclas de componentes     │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
│  ESTABILIDAD NUMÉRICA                                                    │
│  ───────────────────────                                                 │
│                                                                          │
│  κ(A) = σ_max / σ_min  →  Mide sensibilidad a errores                  │
│                                                                          │
│  Soluciones:                                                             │
│    • Regularización (Ridge)                                              │
│    • Normalización de features                                           │
│    • Usar SVD/lstsq en vez de inversión directa                         │
│    • PCA para eliminar colinealidad                                      │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  APLICACIONES PRÁCTICAS                                                  │
│                                                                          │
│  Tarea                         Técnica                                   │
│  ─────────────────────────────────────────                              │
│  Reducción dimensionalidad  →  PCA (via SVD)                            │
│  Sistemas recomendación    →  Matrix Factorization                      │
│  Compresión de modelos      →  Low-rank approximation                   │
│  NLP / Topic modeling       →  NMF, LSA                                 │
│  Embeddings                 →  SVD truncada                              │
│  Resolver Ax = b (ML)       →  Ridge, lstsq (no invertir!)             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conceptos Clave

```
┌─────────────────────────────────────────────────────────────┐
│  RESUMEN DE CONCEPTOS                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  EIGENDECOMPOSITION                                          │
│    A·v = λ·v (eigenvectores no cambian dirección)          │
│    A = V Λ V^T para matrices simétricas                     │
│    Eigenvalores de covarianza = varianzas en PCA           │
│                                                              │
│  SVD (A = U Σ V^T)                                          │
│    Generaliza eigen a matrices rectangulares                │
│    σᵢ² = eigenvalores de A^T A                             │
│    Columnas de V = direcciones principales (PCA)           │
│    Base de casi todo en álgebra lineal numérica            │
│                                                              │
│  LOW-RANK APPROXIMATION                                      │
│    A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢ^T (mejor aproximación rango-k)   │
│    Compresión: mantener σ grandes, descartar pequeños      │
│    Error = √(Σ σᵢ² para i > k)                             │
│                                                              │
│  MATRIX FACTORIZATION                                        │
│    R ≈ U V^T para recomendadores                           │
│    NMF: factores no negativos, interpretables              │
│    Optimizar solo sobre observados                         │
│                                                              │
│  CONDICIONAMIENTO                                            │
│    κ(A) = σ_max / σ_min                                     │
│    κ grande → inestable numéricamente                      │
│    Solución: regularización, normalización, SVD            │
└─────────────────────────────────────────────────────────────┘
```

---

**Conexiones con otros temas:**
- PCA → Eigendecomposition/SVD de covarianza
- Word Embeddings (Word2Vec, GloVe) → SVD implícita de co-ocurrencia
- Recomendadores (Netflix, Spotify) → Matrix Factorization
- Compresión de redes neuronales → Low-rank approximation
- Regularización Ridge → Mejora condicionamiento

**Nota práctica:** En producción, siempre usar las implementaciones optimizadas de numpy/scipy. Evitar invertir matrices directamente; preferir `np.linalg.lstsq` o `np.linalg.solve`.
