# Teoría de la Información para Machine Learning

## 1. Introducción: ¿Qué es la Información?

### La Idea de Shannon

Claude Shannon (1948) formalizó la información como **reducción de incertidumbre**.

```
┌─────────────────────────────────────────────────────────────┐
│  INFORMACIÓN = SORPRESA                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Evento PROBABLE   → Poca información (poca sorpresa)       │
│  Evento IMPROBABLE → Mucha información (mucha sorpresa)     │
│                                                              │
│  Ejemplo:                                                    │
│    "Mañana sale el sol" → Poca info (obvio)                │
│    "Mañana cae meteorito" → Mucha info (inesperado)        │
│                                                              │
│  Fórmula de contenido de información:                        │
│                                                              │
│    I(x) = -log₂ P(x)                                        │
│                                                              │
│  Si P(x) = 1   → I(x) = 0 bits (certeza, sin info)         │
│  Si P(x) = 0.5 → I(x) = 1 bit (máxima info binaria)        │
│  Si P(x) = 0   → I(x) = ∞ (imposible, info infinita)       │
└─────────────────────────────────────────────────────────────┘
```

### ¿Por Qué Importa en ML?

```
┌─────────────────────────────────────────────────────────────┐
│  TEORÍA DE LA INFORMACIÓN EN ML                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. FUNCIONES DE PÉRDIDA                                     │
│     Cross-entropy loss = KL divergence + constante          │
│     → Minimizar loss = Aproximar distribución real          │
│                                                              │
│  2. COMPRESIÓN DE MODELOS                                    │
│     ¿Cuántos bits necesito para codificar los pesos?        │
│     → Límites teóricos de compresión                        │
│                                                              │
│  3. SELECCIÓN DE FEATURES                                    │
│     Mutual Information = dependencia entre X e Y            │
│     → Features con alta MI son más informativas             │
│                                                              │
│  4. REPRESENTACIONES (VAEs, Autoencoders)                   │
│     Information Bottleneck = comprimir conservando lo útil │
│     → Balance entre compresión y predicción                 │
│                                                              │
│  5. GENERACIÓN (GANs, Diffusion)                            │
│     KL divergence entre distribución generada y real        │
│     → Medir calidad de generación                           │
└─────────────────────────────────────────────────────────────┘
```

## 2. Entropía: Cuantificando la Incertidumbre

### Definición de Entropía

```
┌─────────────────────────────────────────────────────────────┐
│  ENTROPÍA DE SHANNON                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para una variable aleatoria discreta X con distribución P: │
│                                                              │
│    H(X) = -Σ P(x) log₂ P(x)                                 │
│            x                                                 │
│                                                              │
│         = E[-log₂ P(X)]                                     │
│         = "Número promedio de bits para codificar X"        │
│                                                              │
│  Propiedades:                                                │
│    • H(X) ≥ 0 siempre                                       │
│    • H(X) = 0 si X es determinista (un solo valor posible) │
│    • H(X) es máxima cuando X es uniforme                    │
│                                                              │
│  Para distribución continua (Entropía diferencial):         │
│    h(X) = -∫ p(x) log p(x) dx                               │
│    (Puede ser negativa!)                                     │
└─────────────────────────────────────────────────────────────┘
```

### Visualización: Entropía de Distribución Binaria

```
Entropía de Bernoulli H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

H(p)│
    │            ╱╲
1.0 │          ╱    ╲         ← Máxima incertidumbre
    │        ╱        ╲          cuando p = 0.5
0.8 │      ╱            ╲
    │    ╱                ╲
0.6 │  ╱                    ╲
    │╱                        ╲
0.4 │                          ╲
    │                            ╲
0.2 │                              ╲
    │                                ╲
0.0 ├────────────────────────────────── p
    0        0.25      0.5       0.75        1

    ↑                              ↑
  p=0 o p=1                    p=0.5
  H=0 (certeza)               H=1 (max incertidumbre)
```

```python
import numpy as np
from scipy import stats

def entropia_bernoulli(p: float) -> float:
    """Calcula entropía de distribución Bernoulli."""
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def entropia_discreta(probs: np.ndarray) -> float:
    """
    Calcula entropía de Shannon para distribución discreta.

    Args:
        probs: Array de probabilidades (debe sumar 1)

    Returns:
        Entropía en bits
    """
    # Evitar log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# Ejemplos
print("Entropía de Bernoulli:")
for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
    print(f"  p = {p}: H = {entropia_bernoulli(p):.4f} bits")

print("\nEntropía de distribuciones:")
# Distribución uniforme (4 clases)
uniforme = np.array([0.25, 0.25, 0.25, 0.25])
print(f"  Uniforme 4 clases: H = {entropia_discreta(uniforme):.4f} bits")

# Distribución sesgada
sesgada = np.array([0.9, 0.05, 0.03, 0.02])
print(f"  Sesgada [0.9, ...]: H = {entropia_discreta(sesgada):.4f} bits")

# Distribución determinista
determinista = np.array([1.0, 0.0, 0.0, 0.0])
print(f"  Determinista: H = {entropia_discreta(determinista):.4f} bits")
```

### Entropía y Árboles de Decisión

```
┌─────────────────────────────────────────────────────────────┐
│  ENTROPÍA EN ÁRBOLES DE DECISIÓN                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Criterio de split: MAXIMIZAR reducción de entropía         │
│                                                              │
│  Ganancia de Información:                                    │
│    IG(Y, X) = H(Y) - H(Y|X)                                 │
│             = Entropía antes - Entropía después del split   │
│                                                              │
│  Ejemplo:                                                    │
│    Dataset: 50% spam, 50% no spam → H = 1 bit               │
│    Después de split por "contiene $$$":                     │
│      - Si contiene: 90% spam → H = 0.47 bits               │
│      - Si no contiene: 20% spam → H = 0.72 bits            │
│    H promedio después = 0.6 bits                            │
│    Ganancia = 1 - 0.6 = 0.4 bits                            │
│                                                              │
│  Split es BUENO si IG es ALTO                               │
└─────────────────────────────────────────────────────────────┘
```

```python
def ganancia_informacion(y: np.ndarray, mascara_split: np.ndarray) -> float:
    """
    Calcula ganancia de información de un split.

    Args:
        y: Labels (0 o 1 para clasificación binaria)
        mascara_split: Boolean array indicando qué ejemplos van a la izquierda

    Returns:
        Information Gain en bits
    """
    # Entropía antes del split
    p_before = y.mean()
    H_before = entropia_bernoulli(p_before)

    # Split
    y_left = y[mascara_split]
    y_right = y[~mascara_split]

    # Entropía después (promedio ponderado)
    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0  # Split trivial

    H_left = entropia_bernoulli(y_left.mean()) if n_left > 0 else 0
    H_right = entropia_bernoulli(y_right.mean()) if n_right > 0 else 0

    H_after = (n_left / n) * H_left + (n_right / n) * H_right

    return H_before - H_after


# Ejemplo
np.random.seed(42)
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 50% cada clase

# Split bueno: separa bien las clases
split_bueno = np.array([True, True, True, True, True, False, False, False, False, False])
ig_bueno = ganancia_informacion(y, split_bueno)
print(f"Split perfecto: IG = {ig_bueno:.4f} bits")

# Split malo: mezcla las clases
split_malo = np.array([True, False, True, False, True, False, True, False, True, False])
ig_malo = ganancia_informacion(y, split_malo)
print(f"Split aleatorio: IG = {ig_malo:.4f} bits")
```

## 3. Entropía Cruzada (Cross-Entropy)

### Definición y Conexión con ML

```
┌─────────────────────────────────────────────────────────────┐
│  CROSS-ENTROPY                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para distribuciones P (real) y Q (predicha):               │
│                                                              │
│    H(P, Q) = -Σ P(x) log Q(x)                               │
│               x                                              │
│                                                              │
│            = E_P[-log Q(X)]                                 │
│            = "Bits para codificar P usando código de Q"    │
│                                                              │
│  Propiedades:                                                │
│    • H(P, Q) ≥ H(P)  (igualdad solo si P = Q)              │
│    • H(P, Q) ≠ H(Q, P)  (no es simétrica!)                 │
│    • Minimizar H(P, Q) respecto a Q → Q se acerca a P      │
│                                                              │
│  EN ML:                                                      │
│    P = distribución real (one-hot en clasificación)        │
│    Q = predicciones del modelo (softmax)                   │
│    Objetivo: min H(P, Q) → modelo aproxima realidad        │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Entropy como Loss Function

```
┌─────────────────────────────────────────────────────────────┐
│  CROSS-ENTROPY LOSS EN CLASIFICACIÓN                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Clasificación binaria:                                      │
│    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]                        │
│                                                              │
│    Donde:                                                    │
│      y = label real (0 o 1)                                 │
│      ŷ = probabilidad predicha                              │
│                                                              │
│  Clasificación multiclase:                                   │
│    L = -Σ yₖ·log(ŷₖ)                                        │
│         k                                                    │
│                                                              │
│    Donde:                                                    │
│      y = one-hot del label real                             │
│      ŷ = softmax de logits                                  │
│                                                              │
│  Equivalencia con Negative Log-Likelihood:                   │
│    Cross-Entropy = NLL para distribución categórica        │
│    Minimizar CE = Maximizar likelihood                      │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def cross_entropy_binaria(y_true: np.ndarray, y_pred: np.ndarray,
                           eps: float = 1e-15) -> float:
    """
    Calcula cross-entropy para clasificación binaria.

    Args:
        y_true: Labels reales (0 o 1)
        y_pred: Probabilidades predichas
        eps: Pequeño valor para estabilidad numérica

    Returns:
        Cross-entropy loss promedio
    """
    # Clip para evitar log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )
    return loss

def cross_entropy_multiclase(y_true: np.ndarray, y_pred: np.ndarray,
                               eps: float = 1e-15) -> float:
    """
    Calcula cross-entropy para clasificación multiclase.

    Args:
        y_true: Labels one-hot (n_samples, n_classes)
        y_pred: Probabilidades (después de softmax)

    Returns:
        Cross-entropy loss promedio
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def softmax(logits: np.ndarray) -> np.ndarray:
    """Calcula softmax con estabilidad numérica."""
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


# Ejemplo binario
y_true = np.array([1, 1, 0, 0, 1])
y_pred_bueno = np.array([0.9, 0.8, 0.1, 0.2, 0.85])
y_pred_malo = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

print("Cross-entropy binaria:")
print(f"  Predicción buena: {cross_entropy_binaria(y_true, y_pred_bueno):.4f}")
print(f"  Predicción mala: {cross_entropy_binaria(y_true, y_pred_malo):.4f}")

# Ejemplo multiclase
y_true_multi = np.array([
    [1, 0, 0],  # Clase 0
    [0, 1, 0],  # Clase 1
    [0, 0, 1],  # Clase 2
])

logits_buenos = np.array([
    [3.0, 0.1, 0.1],  # Muy seguro de clase 0
    [0.1, 2.5, 0.2],  # Seguro de clase 1
    [0.2, 0.1, 2.8],  # Seguro de clase 2
])

logits_malos = np.array([
    [0.3, 0.3, 0.4],  # Inseguro
    [0.4, 0.3, 0.3],  # Incorrecto
    [0.3, 0.4, 0.3],  # Incorrecto
])

print("\nCross-entropy multiclase:")
print(f"  Predicción buena: {cross_entropy_multiclase(y_true_multi, softmax(logits_buenos)):.4f}")
print(f"  Predicción mala: {cross_entropy_multiclase(y_true_multi, softmax(logits_malos)):.4f}")
```

### Relación: H(P,Q) = H(P) + D_KL(P||Q)

```
┌─────────────────────────────────────────────────────────────┐
│  DESCOMPOSICIÓN DE CROSS-ENTROPY                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  H(P, Q) = H(P) + D_KL(P || Q)                              │
│                                                              │
│  Donde:                                                      │
│    H(P) = entropía de P (constante durante entrenamiento)  │
│    D_KL(P||Q) = divergencia KL (mide diferencia P vs Q)    │
│                                                              │
│  Implicación para ML:                                        │
│    Minimizar H(P, Q) ≡ Minimizar D_KL(P || Q)              │
│    (porque H(P) es constante)                               │
│                                                              │
│    → El modelo aprende a aproximar la distribución real    │
│    → Cross-entropy "mide distancia" a la verdad            │
└─────────────────────────────────────────────────────────────┘
```

## 4. Divergencia KL (Kullback-Leibler)

### Definición

```
┌─────────────────────────────────────────────────────────────┐
│  KL DIVERGENCE                                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  La divergencia KL mide cuánto difiere Q de P:              │
│                                                              │
│    D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))                    │
│                   x                                          │
│                                                              │
│                 = E_P[log(P(X)/Q(X))]                       │
│                                                              │
│                 = H(P, Q) - H(P)                            │
│                                                              │
│  Propiedades:                                                │
│    • D_KL(P||Q) ≥ 0  siempre (Desigualdad de Gibbs)        │
│    • D_KL(P||Q) = 0  sii P = Q                              │
│    • D_KL(P||Q) ≠ D_KL(Q||P)  (NO es simétrica!)           │
│    • NO es una distancia (no cumple desigualdad triangular)│
│                                                              │
│  Interpretación:                                             │
│    "Bits extra necesarios para codificar P usando Q"       │
│    "Ineficiencia de usar Q cuando la realidad es P"        │
└─────────────────────────────────────────────────────────────┘
```

### Forward KL vs Reverse KL

```
┌─────────────────────────────────────────────────────────────┐
│  ASIMETRÍA DE KL DIVERGENCE                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  FORWARD KL: D_KL(P || Q) - Minimizar respecto a Q         │
│    "Zero-avoiding": Q evita poner 0 donde P > 0            │
│    Q tiende a CUBRIR todo el soporte de P                  │
│    → Q se expande, puede ser multimodal                    │
│                                                              │
│  REVERSE KL: D_KL(Q || P) - Minimizar respecto a Q         │
│    "Zero-forcing": Q pone 0 donde P ≈ 0                    │
│    Q tiende a CONCENTRARSE en un modo de P                 │
│    → Q colapsa, típicamente unimodal                       │
│                                                              │
│  Visualización con P bimodal:                                │
│                                                              │
│       P│    ╱╲  ╱╲       Forward KL          Reverse KL     │
│        │   ╱  ╲╱  ╲                                         │
│        │  ╱        ╲     Q│  ╱──────╲      Q│    ╱╲        │
│        │ ╱          ╲     │ ╱        ╲      │   ╱  ╲       │
│        └────────────       └──────────       └────────      │
│                           "Cubre ambos"    "Elige uno"     │
│                                                              │
│  En ML:                                                      │
│    • VAEs usan reverse KL → pueden perder modos            │
│    • MLE implica forward KL → model covering               │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calcula D_KL(P || Q) para distribuciones discretas.

    Args:
        p: Distribución P (debe sumar 1)
        q: Distribución Q (debe sumar 1)
        eps: Pequeño valor para estabilidad

    Returns:
        KL divergence en bits (usando log2) o nats (log natural)
    """
    # Evitar log(0) y 0*log(0)
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    kl = np.sum(p * np.log(p / q))
    return kl

def kl_gaussians(mu1: float, sigma1: float,
                  mu2: float, sigma2: float) -> float:
    """
    KL divergence entre dos Gaussianas univariadas.
    D_KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
    """
    kl = (np.log(sigma2 / sigma1) +
          (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) -
          0.5)
    return kl


# Ejemplo: Distribuciones discretas
p = np.array([0.4, 0.4, 0.2])  # Distribución real
q1 = np.array([0.4, 0.4, 0.2])  # Predicción perfecta
q2 = np.array([0.33, 0.33, 0.34])  # Predicción uniforme
q3 = np.array([0.1, 0.1, 0.8])  # Predicción mala

print("KL Divergence D_KL(P || Q):")
print(f"  Q = P (perfecto): {kl_divergence(p, q1):.6f}")
print(f"  Q uniforme: {kl_divergence(p, q2):.4f}")
print(f"  Q muy diferente: {kl_divergence(p, q3):.4f}")

# Mostrar asimetría
print(f"\nAsimetría:")
print(f"  D_KL(P || Q3): {kl_divergence(p, q3):.4f}")
print(f"  D_KL(Q3 || P): {kl_divergence(q3, p):.4f}")

# Ejemplo: Gaussianas
print(f"\nKL entre Gaussianas:")
print(f"  N(0,1) || N(0,1): {kl_gaussians(0, 1, 0, 1):.4f}")
print(f"  N(0,1) || N(1,1): {kl_gaussians(0, 1, 1, 1):.4f}")
print(f"  N(0,1) || N(0,2): {kl_gaussians(0, 1, 0, 2):.4f}")
```

### KL en VAEs (Variational Autoencoders)

```
┌─────────────────────────────────────────────────────────────┐
│  KL EN LA LOSS DE VAE                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Loss de VAE:                                                │
│    L = -E_q[log p(x|z)] + D_KL(q(z|x) || p(z))             │
│         ↑                      ↑                             │
│    Reconstrucción        Regularización KL                  │
│                                                              │
│  El término KL:                                              │
│    • q(z|x) = encoder output (Gaussiana con μ, σ aprendidos)│
│    • p(z) = prior sobre z (típicamente N(0, I))            │
│    • KL "empuja" q hacia el prior                          │
│    • Evita que el encoder aprenda distribuciones degeneradas│
│                                                              │
│  Para Gaussianas (forma cerrada):                            │
│    D_KL(N(μ, σ²) || N(0, 1))                               │
│      = 0.5 * Σ(μ² + σ² - 1 - log(σ²))                     │
│                                                              │
│  Beneficio: El espacio latente es continuo y estructurado  │
└─────────────────────────────────────────────────────────────┘
```

```python
def vae_kl_loss(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Término KL de la loss de VAE.
    D_KL(N(mu, diag(exp(log_var))) || N(0, I))

    Args:
        mu: Media del encoder (batch_size, latent_dim)
        log_var: Log-varianza del encoder

    Returns:
        KL divergence promedio
    """
    # KL para Gaussiana vs N(0,1)
    kl = 0.5 * np.sum(mu**2 + np.exp(log_var) - 1 - log_var, axis=-1)
    return np.mean(kl)


# Ejemplo: Comparar diferentes encodings
np.random.seed(42)

# Caso 1: Encoder bien regularizado (cercano al prior)
mu_bueno = np.random.randn(10, 8) * 0.5
log_var_bueno = np.zeros((10, 8))  # σ = 1

# Caso 2: Encoder degenerado (varianza muy pequeña)
mu_malo = np.random.randn(10, 8) * 0.5
log_var_malo = -5 * np.ones((10, 8))  # σ ≈ 0.08 (muy concentrado)

print("KL Loss en VAE:")
print(f"  Encoder regularizado: {vae_kl_loss(mu_bueno, log_var_bueno):.4f}")
print(f"  Encoder degenerado: {vae_kl_loss(mu_malo, log_var_malo):.4f}")
print("  (Mayor KL indica alejamiento del prior)")
```

## 5. Mutual Information: Dependencia entre Variables

### Definición de Mutual Information

```
┌─────────────────────────────────────────────────────────────┐
│  MUTUAL INFORMATION                                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MI mide cuánta información comparten X e Y:                │
│                                                              │
│    I(X; Y) = Σ Σ P(x,y) log(P(x,y) / (P(x)P(y)))           │
│              x y                                             │
│                                                              │
│            = H(X) - H(X|Y)                                  │
│            = H(Y) - H(Y|X)                                  │
│            = H(X) + H(Y) - H(X,Y)                           │
│            = D_KL(P(X,Y) || P(X)P(Y))                       │
│                                                              │
│  Propiedades:                                                │
│    • I(X; Y) ≥ 0  siempre                                   │
│    • I(X; Y) = 0  sii X e Y son independientes              │
│    • I(X; Y) = I(Y; X)  (simétrica!)                        │
│    • I(X; X) = H(X)  (toda la info de X)                    │
│                                                              │
│  Interpretación:                                             │
│    "Reducción de incertidumbre sobre X al conocer Y"       │
│    "Información que Y proporciona sobre X"                  │
└─────────────────────────────────────────────────────────────┘
```

### Diagrama de Venn de Entropías

```
                    DIAGRAMA DE VENN DE INFORMACIÓN

           ┌─────────────────────────────────────┐
           │              H(X, Y)                 │
           │    Entropía conjunta                │
           │                                      │
           │   ┌───────────────┐───────────────┐ │
           │   │               │               │ │
           │   │     H(X|Y)    │    H(Y|X)    │ │
           │   │  Info única   │   Info única │ │
           │   │    de X       │     de Y      │ │
           │   │               │               │ │
           │   │       ┌───────┤               │ │
           │   │       │ I(X;Y)│               │ │
           │   │       │ Info  │               │ │
           │   │       │mutua  │               │ │
           │   │       └───────┤               │ │
           │   │               │               │ │
           │   └───────────────┴───────────────┘ │
           │        H(X)           H(Y)          │
           └─────────────────────────────────────┘

        H(X, Y) = H(X) + H(Y) - I(X; Y)
        H(X, Y) = H(X|Y) + I(X;Y) + H(Y|X)
```

### MI para Feature Selection

```
┌─────────────────────────────────────────────────────────────┐
│  MUTUAL INFORMATION PARA SELECCIÓN DE FEATURES              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema: Elegir las K features más informativas          │
│                                                              │
│  Método basado en MI:                                        │
│    Para cada feature Xᵢ, calcular I(Xᵢ; Y)                 │
│    Ordenar features por MI descendente                       │
│    Seleccionar las top-K                                     │
│                                                              │
│  Ventajas vs correlación:                                    │
│    • Captura dependencias NO LINEALES                       │
│    • Correlación = 0 no implica MI = 0                      │
│                                                              │
│  Ejemplo:                                                    │
│    X = Uniforme(-1, 1)                                      │
│    Y = X²                                                    │
│    Correlación(X, Y) = 0 (!)                                │
│    I(X; Y) > 0 (hay dependencia perfecta)                  │
│                                                              │
│  Implementación en sklearn:                                  │
│    from sklearn.feature_selection import mutual_info_classif│
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def estimar_mi_discreto(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estima MI para variables discretas usando histogramas.
    """
    # Matriz de frecuencias conjuntas
    xy_counts = {}
    x_counts = {}
    y_counts = {}
    n = len(x)

    for xi, yi in zip(x, y):
        xy_counts[(xi, yi)] = xy_counts.get((xi, yi), 0) + 1
        x_counts[xi] = x_counts.get(xi, 0) + 1
        y_counts[yi] = y_counts.get(yi, 0) + 1

    # Calcular MI
    mi = 0.0
    for (xi, yi), count_xy in xy_counts.items():
        p_xy = count_xy / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        if p_xy > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return mi


# Ejemplo: Comparar correlación vs MI
np.random.seed(42)
n = 1000

# Caso 1: Relación lineal (correlación y MI altas)
x_lineal = np.random.randn(n)
y_lineal = 2 * x_lineal + 0.5 * np.random.randn(n)

# Caso 2: Relación NO lineal (correlación baja, MI alta)
x_cuadrado = np.random.uniform(-2, 2, n)
y_cuadrado = x_cuadrado ** 2 + 0.1 * np.random.randn(n)

# Caso 3: Independientes (correlación y MI bajas)
x_indep = np.random.randn(n)
y_indep = np.random.randn(n)

print("Correlación vs Mutual Information:")
print("\n1. Relación lineal (y = 2x + ruido):")
print(f"   Correlación: {np.corrcoef(x_lineal, y_lineal)[0,1]:.4f}")
mi_lineal = mutual_info_regression(x_lineal.reshape(-1, 1), y_lineal)[0]
print(f"   MI estimada: {mi_lineal:.4f}")

print("\n2. Relación cuadrática (y = x² + ruido):")
print(f"   Correlación: {np.corrcoef(x_cuadrado, y_cuadrado)[0,1]:.4f}")
mi_cuadrado = mutual_info_regression(x_cuadrado.reshape(-1, 1), y_cuadrado)[0]
print(f"   MI estimada: {mi_cuadrado:.4f}")
print("   → Alta MI a pesar de baja correlación!")

print("\n3. Variables independientes:")
print(f"   Correlación: {np.corrcoef(x_indep, y_indep)[0,1]:.4f}")
mi_indep = mutual_info_regression(x_indep.reshape(-1, 1), y_indep)[0]
print(f"   MI estimada: {mi_indep:.4f}")
```

## 6. Information Bottleneck

### El Principio del Cuello de Botella

```
┌─────────────────────────────────────────────────────────────┐
│  INFORMATION BOTTLENECK                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema: Encontrar representación Z de X que:             │
│    1. COMPRIMA X (baja complejidad)                         │
│    2. PRESERVE información sobre Y (buen predictor)         │
│                                                              │
│  Formulación:                                                │
│    max I(Z; Y) - β·I(Z; X)                                  │
│     Z                                                        │
│    ↑           ↑                                             │
│    Preservar   Comprimir                                     │
│    relevancia  representación                                │
│                                                              │
│  β = parámetro de trade-off                                 │
│    β pequeño → Z complejo, muy informativo sobre Y         │
│    β grande  → Z simple, puede perder información          │
│                                                              │
│  Flujo de información:                                       │
│                                                              │
│    X ────→ Z ────→ Y                                        │
│        ↑       ↑                                             │
│    Encoder   Decoder                                         │
│   (comprime) (predice)                                       │
│                                                              │
│  Conexión con Deep Learning:                                 │
│    Las capas ocultas de una red son "cuellos de botella"   │
│    Capas profundas → representaciones más comprimidas      │
└─────────────────────────────────────────────────────────────┘
```

### Visualización: Trade-off Información

```
I(Z; Y)│
       │
       │ ●         ← β muy pequeño (Z ≈ X, nada comprimido)
       │  ╲
       │   ╲       Curva de Pareto
       │    ●      (información preservada vs compresión)
       │     ╲
       │      ╲
       │       ●   ← β óptimo (balance)
       │        ╲
       │         ╲
       │          ●
       │           ╲
       │            ●  ← β grande (Z muy comprimido)
       │
       └──────────────── I(Z; X)
                         (complejidad de Z)

Objetivo: Encontrar β que maximiza predicción con mínima complejidad
```

### Aplicación: Deep Learning como IB

```
┌─────────────────────────────────────────────────────────────┐
│  REDES NEURONALES COMO INFORMATION BOTTLENECK              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input X  →  Capa 1  →  Capa 2  →  ...  →  Output Y        │
│              (Z₁)       (Z₂)                                │
│                                                              │
│  Durante entrenamiento:                                      │
│                                                              │
│  Fase 1 (Fitting):                                           │
│    I(Zₖ; X) aumenta  (capas aprenden features)             │
│    I(Zₖ; Y) aumenta  (capas aprenden a predecir)           │
│                                                              │
│  Fase 2 (Compression):                                       │
│    I(Zₖ; X) DISMINUYE (capas olvidan ruido)                │
│    I(Zₖ; Y) se mantiene (predicción preservada)            │
│                                                              │
│  Hipótesis (Tishby et al.):                                  │
│    La generalización viene de COMPRIMIR el input            │
│    Mantener solo la información relevante para Y            │
│    → Dropout, regularización fuerzan compresión             │
└─────────────────────────────────────────────────────────────┘
```

## 7. Conexiones con Loss Functions

### Resumen de Equivalencias

```
┌─────────────────────────────────────────────────────────────┐
│  LOSS FUNCTIONS DESDE TEORÍA DE LA INFORMACIÓN             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. CROSS-ENTROPY LOSS = - log P(Y|X; θ)                   │
│     → Minimizar = Maximizar log-likelihood                  │
│     → Equivale a minimizar D_KL(P_real || P_modelo)        │
│                                                              │
│  2. MSE = E[(Y - Ŷ)²]                                       │
│     → Asume Y|X ~ N(Ŷ, σ²)                                 │
│     → MSE ∝ -log P(Y|X) para Gaussiano                     │
│                                                              │
│  3. VAE LOSS = Reconstrucción + β·KL                       │
│     → Reconstrucción ≈ -I(Z; X)                            │
│     → KL regulariza hacia prior                             │
│     → Balance información vs simplicidad                    │
│                                                              │
│  4. CONTRASTIVE LOSS (InfoNCE)                              │
│     → Maximiza bound inferior de I(X; Z)                    │
│     → Usado en representaciones autosupervisadas            │
│                                                              │
│  5. GAN LOSS                                                 │
│     → Discriminador minimiza variante de KL                 │
│     → Generador aproxima distribución real                  │
└─────────────────────────────────────────────────────────────┘
```

### InfoNCE: Contrastive Learning

```
┌─────────────────────────────────────────────────────────────┐
│  INFONCE LOSS (Oord et al., 2018)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Objetivo: Aprender representaciones maximizando MI         │
│            entre vistas de un mismo ejemplo                 │
│                                                              │
│  Setup:                                                      │
│    • x⁺ = par positivo (misma imagen, diferente augment)   │
│    • x⁻ = pares negativos (otras imágenes)                 │
│    • f(·) = encoder que produce embeddings                 │
│                                                              │
│  Loss:                                                       │
│    L = -log[exp(f(x)·f(x⁺)/τ) / Σₖexp(f(x)·f(xₖ)/τ)]      │
│                                                              │
│  Interpretación:                                             │
│    • Numerador: similitud con positivo                      │
│    • Denominador: similitud con todos (positivo + negativos)│
│    • τ = temperatura (controla "sharpness")                │
│                                                              │
│  Teorema: InfoNCE es un lower bound de I(X; Z)             │
│    I(X; Z) ≥ log(K) - L_InfoNCE                            │
│    donde K = número de negativos                            │
└─────────────────────────────────────────────────────────────┘
```

```python
import numpy as np

def infonce_loss(anchor: np.ndarray, positive: np.ndarray,
                  negatives: np.ndarray, temperature: float = 0.1) -> float:
    """
    Calcula InfoNCE loss.

    Args:
        anchor: Embedding del anchor (d,)
        positive: Embedding del par positivo (d,)
        negatives: Embeddings de pares negativos (K, d)
        temperature: Temperatura para softmax

    Returns:
        InfoNCE loss (escalar)
    """
    # Similitudes
    sim_positive = np.dot(anchor, positive) / temperature
    sim_negatives = negatives @ anchor / temperature

    # Concatenar positivo con negativos
    all_sims = np.concatenate([[sim_positive], sim_negatives])

    # Softmax normalizado
    exp_sims = np.exp(all_sims - all_sims.max())  # Estabilidad
    prob_positive = exp_sims[0] / exp_sims.sum()

    # Loss = -log de probabilidad del positivo
    return -np.log(prob_positive + 1e-10)


# Ejemplo
np.random.seed(42)
d = 128  # Dimensión del embedding

# Anchor y positivo similares
anchor = np.random.randn(d)
anchor = anchor / np.linalg.norm(anchor)

positive = anchor + 0.1 * np.random.randn(d)
positive = positive / np.linalg.norm(positive)

# Negativos aleatorios
n_negatives = 100
negatives = np.random.randn(n_negatives, d)
negatives = negatives / np.linalg.norm(negatives, axis=1, keepdims=True)

loss = infonce_loss(anchor, positive, negatives, temperature=0.1)
print(f"InfoNCE loss: {loss:.4f}")
print(f"Lower bound de MI: {np.log(n_negatives + 1) - loss:.4f} nats")

# Comparar con positivo menos similar
positive_malo = np.random.randn(d)
positive_malo = positive_malo / np.linalg.norm(positive_malo)

loss_malo = infonce_loss(anchor, positive_malo, negatives, temperature=0.1)
print(f"\nCon positivo aleatorio:")
print(f"InfoNCE loss: {loss_malo:.4f} (mayor = peor)")
```

## 8. Aplicaciones Prácticas

### Compresión de Modelos

```
┌─────────────────────────────────────────────────────────────┐
│  LÍMITES TEÓRICOS DE COMPRESIÓN (Teorema de Shannon)       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para codificar un mensaje con símbolos de distribución P: │
│                                                              │
│    • Mínimo promedio de bits por símbolo = H(P)            │
│    • No se puede comprimir más que la entropía             │
│                                                              │
│  Aplicación a pesos de redes:                                │
│    • Pesos en float32 = 32 bits cada uno                   │
│    • Si pesos tienen entropía efectiva menor → comprimir   │
│    • Cuantización: reducir bits por peso                   │
│    • Pruning: muchos pesos ≈ 0 → baja entropía            │
│                                                              │
│  Ejemplo:                                                    │
│    Red con 1M pesos en float32 = 4 MB                       │
│    Si entropía efectiva = 4 bits/peso → 0.5 MB teórico    │
│    En práctica: cuantización INT8 → 1 MB                   │
└─────────────────────────────────────────────────────────────┘
```

### Detección de Anomalías con Entropía

```python
import numpy as np

def entropia_secuencia(secuencia: list, n_gram: int = 2) -> float:
    """
    Calcula entropía de n-gramas en una secuencia.
    Útil para detectar anomalías en logs, texto, eventos.
    """
    # Extraer n-gramas
    ngrams = []
    for i in range(len(secuencia) - n_gram + 1):
        ngrams.append(tuple(secuencia[i:i+n_gram]))

    # Contar frecuencias
    from collections import Counter
    counts = Counter(ngrams)
    total = sum(counts.values())

    # Calcular entropía
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


# Ejemplo: Detección de comportamiento anómalo
# Secuencia normal (repetitiva, baja entropía)
logs_normales = ["login", "read", "read", "read", "write", "logout"] * 100

# Secuencia anómala (patrones inusuales, alta entropía)
logs_anomalos = ["login", "sudo", "delete_all", "modify_config",
                  "read_secrets", "create_backdoor", "logout"]

print("Entropía de secuencias (2-gramas):")
print(f"  Logs normales: {entropia_secuencia(logs_normales):.4f} bits")
print(f"  Logs anómalos: {entropia_secuencia(logs_anomalos):.4f} bits")
print("\n  Alta entropía puede indicar comportamiento anómalo")

# Para detección de intrusiones:
# 1. Establecer baseline de entropía en tráfico normal
# 2. Alertar cuando entropía se desvía significativamente
```

## 9. Resumen: Mapa de Teoría de la Información

```
┌─────────────────────────────────────────────────────────────────────────┐
│               MAPA DE TEORÍA DE LA INFORMACIÓN EN ML                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                           ┌─────────────┐                               │
│                           │   ENTROPÍA  │                               │
│                           │    H(X)     │                               │
│                           │Incertidumbre│                               │
│                           └──────┬──────┘                               │
│                                  │                                       │
│            ┌─────────────────────┼─────────────────────┐                │
│            │                     │                     │                │
│            ▼                     ▼                     ▼                │
│    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │
│    │ CROSS-ENTROPY │    │ KL DIVERGENCE │    │   MUTUAL      │         │
│    │    H(P,Q)     │    │  D_KL(P||Q)   │    │ INFORMATION   │         │
│    │               │    │               │    │    I(X;Y)     │         │
│    └───────┬───────┘    └───────┬───────┘    └───────┬───────┘         │
│            │                    │                    │                  │
│            ▼                    ▼                    ▼                  │
│    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐         │
│    │ Classification│    │     VAEs      │    │   Feature     │         │
│    │     Loss      │    │  (KL term)    │    │  Selection    │         │
│    │               │    │               │    │               │         │
│    │  NLP, Vision  │    │  Generación   │    │  Contrastive  │         │
│    │   Seq2Seq     │    │  Compresión   │    │   Learning    │         │
│    └───────────────┘    └───────────────┘    └───────────────┘         │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RELACIONES FUNDAMENTALES:                                               │
│                                                                          │
│    H(P, Q) = H(P) + D_KL(P || Q)                                        │
│    I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)                              │
│    D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))                                │
│                                                                          │
│  APLICACIONES CLAVE:                                                     │
│    • Loss = Cross-Entropy → modelo aproxima distribución real           │
│    • VAE = Reconstrucción + KL → compresión con estructura             │
│    • Feature Selection → MI identifica variables informativas          │
│    • Information Bottleneck → generalización via compresión            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conceptos Clave

```
┌─────────────────────────────────────────────────────────────┐
│  RESUMEN DE CONCEPTOS                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ENTROPÍA H(X)                                               │
│    Incertidumbre de X, bits para codificarlo               │
│    Máxima para distribución uniforme                        │
│    Base de árboles de decisión (Information Gain)          │
│                                                              │
│  CROSS-ENTROPY H(P,Q)                                        │
│    Bits para codificar P usando código de Q                │
│    H(P,Q) ≥ H(P), igualdad si P = Q                        │
│    = Loss de clasificación                                  │
│                                                              │
│  KL DIVERGENCE D_KL(P||Q)                                   │
│    "Distancia" de Q a P (asimétrica)                       │
│    = H(P,Q) - H(P)                                          │
│    = Término de regularización en VAEs                      │
│                                                              │
│  MUTUAL INFORMATION I(X;Y)                                   │
│    Información compartida entre X e Y                       │
│    = 0 sii independientes                                   │
│    Captura dependencias no lineales                        │
│    Base de feature selection avanzado                      │
│                                                              │
│  INFORMATION BOTTLENECK                                      │
│    Comprimir X preservando info sobre Y                    │
│    max I(Z;Y) - β·I(Z;X)                                   │
│    Hipótesis de por qué DL generaliza                      │
└─────────────────────────────────────────────────────────────┘
```

---

**Conexiones con otros temas:**
- Cross-Entropy → Loss de clasificación (logística, softmax)
- KL Divergence → VAEs, GANs, Variational Inference
- Mutual Information → Contrastive Learning (SimCLR, CLIP)
- Information Bottleneck → Regularización, Dropout, compresión

**Siguiente:** Álgebra Lineal Numérica (SVD, eigendecomposition, matrix factorization)
