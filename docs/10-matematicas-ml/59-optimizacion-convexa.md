# Optimización Convexa para Machine Learning

## 1. Introducción: ¿Por Qué Importa la Convexidad?

### El Problema Central del ML

Todo algoritmo de Machine Learning busca lo mismo: **minimizar una función de pérdida**.

```
┌─────────────────────────────────────────────────────────────┐
│  OBJETIVO UNIVERSAL DE ML                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    θ* = argmin L(θ; X, y)                                   │
│            θ                                                 │
│                                                              │
│  Donde:                                                      │
│    θ*  = Parámetros óptimos                                 │
│    L   = Función de pérdida (loss)                          │
│    X   = Datos de entrada                                   │
│    y   = Etiquetas/targets                                  │
│                                                              │
│  Ejemplos:                                                   │
│    • Regresión: L = MSE = (1/n)Σ(ŷ - y)²                   │
│    • Clasificación: L = Cross-Entropy                        │
│    • Redes Neuronales: L = cualquier función diferenciable  │
└─────────────────────────────────────────────────────────────┘
```

### ¿Qué Hace Especial a las Funciones Convexas?

```
       FUNCIÓN CONVEXA                    FUNCIÓN NO CONVEXA
           (ideal)                         (problemática)

J(θ)        ╱╲                        J(θ)    ╱╲    ╱╲
    │      ╱  ╲                           │   ╱  ╲  ╱  ╲
    │     ╱    ╲                          │  ╱    ╲╱    ╲
    │    ╱      ╲                         │ ╱            ╲
    │   ╱        ╲                        │╱              ╲
    │  ╱          ╲                       │
    │ ╱            ╲                      │
    └──────●────────── θ                  └───●──────●───── θ
           ↑                                  ↑      ↑
     ÚNICO MÍNIMO                      MÍNIMOS LOCALES
       GLOBAL                          (podemos quedar
                                        atrapados)

Si L(θ) es CONVEXA → Gradient Descent encuentra el óptimo global
Si L(θ) NO es convexa → Podemos quedar en mínimos locales
```

## 2. Definición Formal de Convexidad

### Función Convexa: Definición Geométrica

Una función f: Rⁿ → R es **convexa** si el segmento que une dos puntos cualesquiera en su gráfica está POR ENCIMA de la curva.

```
Matemáticamente, para todo x, y ∈ dominio(f) y todo λ ∈ [0,1]:

    f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)

Interpretación visual:

f(x)│
    │      B ●─────────────● C    ← Segmento SOBRE la curva
    │       ╱             ╱
    │      ╱   ●         ╱        ← Punto de la curva DEBAJO
    │     ╱             ╱
    │    ╱             ╱
    │ A ●─────────────●
    └───────────────────────── x
        x             y

A = f(x), C = f(y)
B = λf(x) + (1-λ)f(y)  ← Interpolación lineal
Punto en curva = f(λx + (1-λ)y)  ← Siempre MENOR que B
```

### Condición de Segundo Orden (Hessiano)

Para funciones dos veces diferenciables, existe una condición más práctica:

```
┌─────────────────────────────────────────────────────────────┐
│  CONDICIÓN DE CONVEXIDAD                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  f es CONVEXA  ⟺  H(x) ≽ 0  (semidefinida positiva)        │
│                                                              │
│  Donde H(x) es la matriz Hessiana:                          │
│                                                              │
│       ┌                                               ┐      │
│       │  ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ   │      │
│       │  ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ   │      │
│  H =  │     ...         ...      ...      ...        │      │
│       │  ∂²f/∂xₙ∂x₁  ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²     │      │
│       └                                               ┘      │
│                                                              │
│  H ≽ 0 significa: todos los eigenvalues λᵢ ≥ 0             │
└─────────────────────────────────────────────────────────────┘
```

### Ejemplo: Verificar Convexidad de MSE

```python
import numpy as np

# MSE para regresión lineal: J(θ) = (1/2n)||Xθ - y||²

# Gradiente:
# ∇J(θ) = (1/n) X^T (Xθ - y)

# Hessiano:
# H = (1/n) X^T X

# X^T X siempre es semidefinida positiva porque:
# Para cualquier vector v:
#   v^T (X^T X) v = (Xv)^T (Xv) = ||Xv||² ≥ 0

def verificar_convexidad_mse(X: np.ndarray) -> bool:
    """Verifica que MSE es convexo verificando que X^T X ≥ 0."""
    n = X.shape[0]
    H = (1/n) * X.T @ X

    # Calcular eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)

    # Todos deben ser ≥ 0 (con tolerancia numérica)
    es_convexo = np.all(eigenvalues >= -1e-10)

    print(f"Eigenvalues del Hessiano: {eigenvalues}")
    print(f"¿Es convexo? {es_convexo}")

    return es_convexo

# Ejemplo
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 muestras, 5 features
verificar_convexidad_mse(X)
# Output: ¿Es convexo? True
```

## 3. El Gradiente: Dirección de Máximo Crecimiento

### Definición del Gradiente

```
┌─────────────────────────────────────────────────────────────┐
│  GRADIENTE DE f: Rⁿ → R                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│         ┌        ┐                                          │
│         │ ∂f/∂x₁ │                                          │
│         │ ∂f/∂x₂ │                                          │
│  ∇f =   │   ...  │   ← Vector de derivadas parciales        │
│         │ ∂f/∂xₙ │                                          │
│         └        ┘                                          │
│                                                              │
│  Propiedades clave:                                          │
│  1. ∇f apunta en la dirección de MÁXIMO CRECIMIENTO         │
│  2. ||∇f|| indica la TASA de crecimiento                    │
│  3. -∇f apunta hacia donde f DECRECE más rápido             │
│  4. ∇f = 0 en puntos críticos (mínimos, máximos, silla)    │
└─────────────────────────────────────────────────────────────┘
```

### Visualización del Gradiente

```
Función: f(x,y) = x² + y²  (paraboloide)

Curvas de nivel (vistas desde arriba):

     y
     │        ╱─────╲
     │     ╱─│       │─╲
     │   ╱   │  f=1  │   ╲
     │  │    │       │    │
─────┼──│────●───────│────│── x
     │  │    ↑       │    │
     │   ╲   ∇f     ╱
     │     ╲─│     │─╱
     │        ╲───╱  f=4
     │

El gradiente ∇f = [2x, 2y]^T:
  • Es PERPENDICULAR a las curvas de nivel
  • Apunta HACIA AFUERA (crecimiento)
  • Para minimizar: moverse en DIRECCIÓN OPUESTA (-∇f)
```

### Cálculo Práctico del Gradiente

```python
import numpy as np

def gradiente_mse(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calcula el gradiente del MSE para regresión lineal.

    J(θ) = (1/2n)||Xθ - y||²
    ∇J(θ) = (1/n) X^T (Xθ - y)
    """
    n = len(y)
    predicciones = X @ theta
    error = predicciones - y
    gradiente = (1/n) * X.T @ error
    return gradiente

# Ejemplo numérico
np.random.seed(42)
n, d = 100, 3
X = np.random.randn(n, d)
theta_real = np.array([2.0, -1.0, 0.5])
y = X @ theta_real + 0.1 * np.random.randn(n)

# Gradiente en punto aleatorio
theta_inicial = np.zeros(d)
grad = gradiente_mse(X, y, theta_inicial)
print(f"Gradiente en θ=0: {grad}")
# El gradiente apunta hacia donde debemos ir (negado)
```

## 4. Condiciones de Optimalidad

### Condiciones de Primer Orden (Necesarias)

```
┌─────────────────────────────────────────────────────────────┐
│  CONDICIÓN NECESARIA DE OPTIMALIDAD                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Si θ* es un MÍNIMO LOCAL de f, entonces:                   │
│                                                              │
│                    ∇f(θ*) = 0                               │
│                                                              │
│  El gradiente se ANULA en el óptimo                         │
│                                                              │
│  PERO: ∇f = 0 NO garantiza mínimo (puede ser máximo/silla) │
└─────────────────────────────────────────────────────────────┘
```

### Condiciones de Segundo Orden (Suficientes)

```
┌─────────────────────────────────────────────────────────────┐
│  CONDICIÓN SUFICIENTE DE OPTIMALIDAD                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  θ* es un MÍNIMO LOCAL ESTRICTO si:                        │
│                                                              │
│    1. ∇f(θ*) = 0           (gradiente cero)                │
│    2. H(θ*) ≻ 0            (Hessiano definido positivo)    │
│                                                              │
│  Para funciones CONVEXAS:                                    │
│    • ∇f(θ*) = 0  →  θ* es MÍNIMO GLOBAL                    │
│    • No hay mínimos locales que no sean globales            │
└─────────────────────────────────────────────────────────────┘
```

### Tipos de Puntos Críticos

```
     MÍNIMO LOCAL                MÁXIMO LOCAL              PUNTO DE SILLA

    f│    ╱╲                    f│                        f│        ╱
     │   ╱  ╲                    │   ╲____╱                │  ╲    ╱
     │  ╱    ╲                   │   ↑                     │   ╲──╱
     │ ╱      ╲                  │  máx                    │    ↑
     │╱        ╲                 │                         │  silla
     └────●───── x               └────●───── x             └────●───── x
          ↑
     ∇f=0, H>0                  ∇f=0, H<0               ∇f=0, H indefinido

En ML buscamos MÍNIMOS: H ≻ 0 (eigenvalues todos positivos)
```

## 5. Gradient Descent y Sus Variantes

### Gradient Descent Clásico (Batch)

```
┌─────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT (Batch/Vanilla)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Repetir hasta convergencia:                                 │
│                                                              │
│      θ(t+1) = θ(t) - α · ∇L(θ(t))                          │
│                                                              │
│  Donde:                                                      │
│    α = learning rate (hiperparámetro crítico)               │
│    ∇L = gradiente calculado sobre TODO el dataset           │
│                                                              │
│  Propiedades:                                                │
│    ✓ Convergencia garantizada para L convexo                │
│    ✓ Gradiente exacto (no ruidoso)                          │
│    ✗ Costoso: O(n) por iteración                            │
│    ✗ Lento para datasets grandes                            │
└─────────────────────────────────────────────────────────────┘
```

```python
def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> tuple[np.ndarray, list[float]]:
    """
    Gradient Descent para regresión lineal.

    Returns:
        theta: Parámetros óptimos
        historia: Lista de valores del loss
    """
    n, d = X.shape
    theta = np.zeros(d)
    historia = []

    for i in range(max_iters):
        # Calcular pérdida actual
        pred = X @ theta
        loss = (1/(2*n)) * np.sum((pred - y)**2)
        historia.append(loss)

        # Calcular gradiente (sobre todo el dataset)
        grad = (1/n) * X.T @ (pred - y)

        # Verificar convergencia
        if np.linalg.norm(grad) < tol:
            print(f"Convergencia en iteración {i}")
            break

        # Actualizar parámetros
        theta = theta - lr * grad

    return theta, historia

# Ejemplo
np.random.seed(42)
n, d = 1000, 5
X = np.random.randn(n, d)
theta_real = np.array([1., 2., -1., 0.5, -0.5])
y = X @ theta_real + 0.1 * np.random.randn(n)

theta_opt, hist = gradient_descent(X, y, lr=0.1)
print(f"θ real: {theta_real}")
print(f"θ encontrado: {theta_opt}")
print(f"Error: {np.linalg.norm(theta_opt - theta_real):.6f}")
```

### Stochastic Gradient Descent (SGD)

```
┌─────────────────────────────────────────────────────────────┐
│  STOCHASTIC GRADIENT DESCENT (SGD)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para cada muestra i (o mini-batch):                        │
│                                                              │
│      θ(t+1) = θ(t) - α · ∇Lᵢ(θ(t))                         │
│                                                              │
│  Donde ∇Lᵢ es el gradiente de UNA SOLA muestra             │
│                                                              │
│  Propiedades:                                                │
│    ✓ Muy rápido: O(1) por iteración                         │
│    ✓ Puede escapar de mínimos locales (ruido ayuda)        │
│    ✓ Actualización online posible                           │
│    ✗ Convergencia ruidosa (no monótona)                     │
│    ✗ Requiere reducir α con el tiempo                       │
└─────────────────────────────────────────────────────────────┘
```

```
Comparación de trayectorias:

      BATCH GD                        SGD

J(θ)│                           J(θ)│    ╱╲
    │   ╲                           │   ╱  ╲   ← Ruidoso
    │    ╲   ← Suave                │  ╱    ╲
    │     ╲                         │ ╱      ╲╱╲
    │      ╲                        │╱          ╲
    │       ╲                       │            ╲
    │        ╲___                   │             ╲__
    └────────────── iter            └──────────────── iter

SGD es más rápido pero oscila cerca del óptimo
```

```python
def sgd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 1
) -> tuple[np.ndarray, list[float]]:
    """
    Stochastic Gradient Descent con mini-batches.
    """
    n, d = X.shape
    theta = np.zeros(d)
    historia = []

    for epoch in range(epochs):
        # Shuffle datos cada época
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            # Mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Gradiente del batch
            pred = X_batch @ theta
            grad = (1/len(y_batch)) * X_batch.T @ (pred - y_batch)

            # Actualizar
            theta = theta - lr * grad

        # Registrar loss al final de cada época
        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        historia.append(loss)

    return theta, historia
```

### Momentum: Acelerando la Convergencia

```
┌─────────────────────────────────────────────────────────────┐
│  SGD CON MOMENTUM                                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  v(t+1) = β · v(t) + ∇L(θ(t))                              │
│  θ(t+1) = θ(t) - α · v(t+1)                                │
│                                                              │
│  Donde:                                                      │
│    β = coeficiente de momentum (típico: 0.9)                │
│    v = "velocidad" acumulada                                │
│                                                              │
│  Analogía física: Una pelota rodando cuesta abajo           │
│    • Acumula velocidad en la dirección consistente          │
│    • Resiste cambios bruscos de dirección                   │
└─────────────────────────────────────────────────────────────┘
```

```
Sin Momentum                     Con Momentum

    ↙ ↘ ↙ ↘ ↙                     ↘
       ↘ ↙ ↘                         ↘
          ↘ ↙                          ↘
             ●                            ●

Oscila mucho en                 Trayectoria más
direcciones laterales           directa al mínimo
```

```python
def sgd_momentum(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    beta: float = 0.9,
    epochs: int = 100,
    batch_size: int = 32
) -> tuple[np.ndarray, list[float]]:
    """SGD con Momentum."""
    n, d = X.shape
    theta = np.zeros(d)
    v = np.zeros(d)  # Velocidad inicial
    historia = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            pred = X_batch @ theta
            grad = (1/len(y_batch)) * X_batch.T @ (pred - y_batch)

            # Actualizar velocidad
            v = beta * v + grad

            # Actualizar parámetros
            theta = theta - lr * v

        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        historia.append(loss)

    return theta, historia
```

### AdaGrad: Learning Rate Adaptativo por Parámetro

```
┌─────────────────────────────────────────────────────────────┐
│  ADAGRAD (Adaptive Gradient)                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  G(t+1) = G(t) + ∇L(θ(t))²   (acumulador de gradientes²)   │
│                                                              │
│  θ(t+1) = θ(t) - α/√(G(t+1) + ε) · ∇L(θ(t))               │
│                                                              │
│  Donde:                                                      │
│    G = acumulador de gradientes al cuadrado                 │
│    ε = término de estabilidad (típico: 1e-8)                │
│                                                              │
│  Efecto:                                                     │
│    • Features frecuentes → learning rate MENOR              │
│    • Features raras → learning rate MAYOR                   │
│    • Ideal para embeddings/NLP con vocabularios grandes     │
│                                                              │
│  Problema: G solo crece → learning rate tiende a 0          │
└─────────────────────────────────────────────────────────────┘
```

```python
def adagrad(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 100,
    batch_size: int = 32,
    eps: float = 1e-8
) -> tuple[np.ndarray, list[float]]:
    """AdaGrad optimizer."""
    n, d = X.shape
    theta = np.zeros(d)
    G = np.zeros(d)  # Acumulador
    historia = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            idx = indices[i:i+batch_size]
            X_batch, y_batch = X[idx], y[idx]

            pred = X_batch @ theta
            grad = (1/len(y_batch)) * X_batch.T @ (pred - y_batch)

            # Acumular gradientes al cuadrado
            G += grad ** 2

            # Learning rate adaptativo por parámetro
            theta = theta - (lr / np.sqrt(G + eps)) * grad

        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        historia.append(loss)

    return theta, historia
```

### RMSprop: Solucionando el Problema de AdaGrad

```
┌─────────────────────────────────────────────────────────────┐
│  RMSprop (Root Mean Square Propagation)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  E[g²](t+1) = ρ·E[g²](t) + (1-ρ)·∇L(θ(t))²                │
│                                                              │
│  θ(t+1) = θ(t) - α/√(E[g²](t+1) + ε) · ∇L(θ(t))           │
│                                                              │
│  Donde:                                                      │
│    ρ = decay rate (típico: 0.9)                             │
│    E[g²] = media móvil exponencial de gradientes²          │
│                                                              │
│  Ventaja sobre AdaGrad:                                      │
│    • La media móvil "olvida" gradientes antiguos           │
│    • El learning rate NO tiende a cero                      │
│    • Mejor para entrenamiento largo                         │
└─────────────────────────────────────────────────────────────┘
```

```python
def rmsprop(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    rho: float = 0.9,
    epochs: int = 100,
    batch_size: int = 32,
    eps: float = 1e-8
) -> tuple[np.ndarray, list[float]]:
    """RMSprop optimizer."""
    n, d = X.shape
    theta = np.zeros(d)
    E_g2 = np.zeros(d)  # Media móvil de g²
    historia = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            idx = indices[i:i+batch_size]
            X_batch, y_batch = X[idx], y[idx]

            pred = X_batch @ theta
            grad = (1/len(y_batch)) * X_batch.T @ (pred - y_batch)

            # Media móvil exponencial
            E_g2 = rho * E_g2 + (1 - rho) * grad ** 2

            # Actualizar
            theta = theta - (lr / np.sqrt(E_g2 + eps)) * grad

        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        historia.append(loss)

    return theta, historia
```

### Adam: El Optimizador Más Popular

```
┌─────────────────────────────────────────────────────────────┐
│  ADAM (Adaptive Moment Estimation)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Combina: Momentum + RMSprop + Corrección de sesgo          │
│                                                              │
│  PASO 1: Actualizar momentos                                 │
│    m(t+1) = β₁·m(t) + (1-β₁)·∇L         (momentum)         │
│    v(t+1) = β₂·v(t) + (1-β₂)·∇L²        (RMSprop)          │
│                                                              │
│  PASO 2: Corregir sesgo (crucial al inicio)                 │
│    m̂ = m(t+1) / (1 - β₁^t)                                 │
│    v̂ = v(t+1) / (1 - β₂^t)                                 │
│                                                              │
│  PASO 3: Actualizar parámetros                              │
│    θ(t+1) = θ(t) - α · m̂ / (√v̂ + ε)                       │
│                                                              │
│  Valores por defecto (paper original):                       │
│    α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8              │
└─────────────────────────────────────────────────────────────┘
```

```python
def adam(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epochs: int = 100,
    batch_size: int = 32,
    eps: float = 1e-8
) -> tuple[np.ndarray, list[float]]:
    """Adam optimizer - el estándar en Deep Learning."""
    n, d = X.shape
    theta = np.zeros(d)
    m = np.zeros(d)  # Primer momento (momentum)
    v = np.zeros(d)  # Segundo momento (RMSprop)
    t = 0            # Paso temporal para corrección de sesgo
    historia = []

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        for i in range(0, n, batch_size):
            t += 1
            idx = indices[i:i+batch_size]
            X_batch, y_batch = X[idx], y[idx]

            pred = X_batch @ theta
            grad = (1/len(y_batch)) * X_batch.T @ (pred - y_batch)

            # Actualizar momentos
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2

            # Corrección de sesgo
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Actualizar parámetros
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        loss = (1/(2*n)) * np.sum((X @ theta - y)**2)
        historia.append(loss)

    return theta, historia
```

### Comparación Visual de Optimizadores

```
      SGD                    Momentum                  Adam

 J(θ)│╲  ╱╲  ╱╲           │╲                       │╲
     │ ╲╱  ╲╱  ╲          │ ╲                      │ ╲
     │         ╲         │  ╲                     │  ╲
     │          ╲        │   ╲                    │   ╲
     │           ╲__     │    ╲___                │    ╲___
     └────────────── t    └────────── t            └────────── t

     Oscilante            Suave, rápido           Muy rápido,
     Lento                                        adaptativo
```

## 6. Optimización con Restricciones: Multiplicadores de Lagrange

### El Problema con Restricciones

```
┌─────────────────────────────────────────────────────────────┐
│  PROBLEMA DE OPTIMIZACIÓN CON RESTRICCIONES                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Minimizar    f(x)                                          │
│  Sujeto a    g(x) = 0    (restricción de igualdad)         │
│              h(x) ≤ 0    (restricción de desigualdad)      │
│                                                              │
│  Ejemplos en ML:                                             │
│    • SVM: maximizar margen sujeto a clasificación correcta  │
│    • L1/L2 regularización: ||θ||₁ ≤ t ó ||θ||₂ ≤ t        │
│    • Normalización: ||x||₂ = 1                              │
└─────────────────────────────────────────────────────────────┘
```

### Método de Lagrange: Restricciones de Igualdad

```
┌─────────────────────────────────────────────────────────────┐
│  MÉTODO DE MULTIPLICADORES DE LAGRANGE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Problema original:                                          │
│    min f(x)  sujeto a  g(x) = 0                             │
│                                                              │
│  Lagrangiano:                                                │
│    L(x, λ) = f(x) + λ · g(x)                                │
│                                                              │
│  Condiciones de optimalidad (KKT para igualdad):            │
│    ∇ₓL = ∇f(x) + λ·∇g(x) = 0                               │
│    g(x) = 0                                                  │
│                                                              │
│  λ = multiplicador de Lagrange                              │
│  Interpretación: "precio" de relajar la restricción        │
└─────────────────────────────────────────────────────────────┘
```

### Ejemplo: SVM Dual

El SVM es el ejemplo clásico de Lagrange en ML:

```
┌─────────────────────────────────────────────────────────────┐
│  SVM PRIMAL                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  min  (1/2)||w||²                                           │
│   w                                                          │
│                                                              │
│  sujeto a:  yᵢ(w·xᵢ + b) ≥ 1  para todo i                  │
│                                                              │
│  Interpretación:                                             │
│    • Minimizar ||w||² → Maximizar margen 2/||w||            │
│    • Restricción → Clasificar correctamente con margen ≥ 1  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  SVM DUAL (vía Lagrange)                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  max  Σᵢαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)                    │
│   α                                                          │
│                                                              │
│  sujeto a:  αᵢ ≥ 0                                          │
│             Σᵢαᵢyᵢ = 0                                      │
│                                                              │
│  αᵢ = multiplicadores de Lagrange                          │
│  Solo αᵢ > 0 para vectores de soporte                      │
└─────────────────────────────────────────────────────────────┘
```

### Visualización Geométrica de Lagrange

```
Problema: min f(x,y) = x + y  sujeto a x² + y² = 1

       y
       │         ╱ f = 1
       │       ╱
   1 ──●─────╱──────── Restricción g = x² + y² - 1 = 0
       │╲  ╱
       │ ●╱  ← Óptimo: ∇f paralelo a ∇g
       │╱ ╲
───────●───╲─────── x
      -1    ╲
             ╲ f = -1

En el óptimo:
  ∇f = (1, 1)           ← Dirección de crecimiento de f
  ∇g = (2x, 2y)         ← Normal a la restricción
  ∇f = -λ∇g             ← Paralelos → condición de Lagrange

Solución: x* = y* = -1/√2, λ* = 1/√2
```

```python
from scipy.optimize import minimize

def optimizar_con_restriccion():
    """Ejemplo de optimización con restricciones usando scipy."""

    # Objetivo: min x + y
    def objetivo(xy):
        return xy[0] + xy[1]

    # Restricción: x² + y² = 1
    restriccion = {
        'type': 'eq',
        'fun': lambda xy: xy[0]**2 + xy[1]**2 - 1
    }

    # Punto inicial
    x0 = np.array([1.0, 0.0])

    # Resolver
    resultado = minimize(
        objetivo,
        x0,
        method='SLSQP',
        constraints=restriccion
    )

    print(f"Óptimo: x = {resultado.x}")
    print(f"Valor objetivo: {resultado.fun:.4f}")
    # Óptimo: x = [-0.707, -0.707] (≈ -1/√2)
    # Valor objetivo: -1.414 (≈ -√2)

    return resultado

optimizar_con_restriccion()
```

### KKT: Condiciones Generales (Igualdad + Desigualdad)

```
┌─────────────────────────────────────────────────────────────┐
│  CONDICIONES KKT (Karush-Kuhn-Tucker)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Para: min f(x)  s.t.  gᵢ(x) = 0, hⱼ(x) ≤ 0               │
│                                                              │
│  1. ESTACIONARIEDAD:                                         │
│     ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0                  │
│                                                              │
│  2. FACTIBILIDAD PRIMAL:                                     │
│     gᵢ(x*) = 0  para todo i                                 │
│     hⱼ(x*) ≤ 0  para todo j                                 │
│                                                              │
│  3. FACTIBILIDAD DUAL:                                       │
│     μⱼ ≥ 0  para todo j                                     │
│                                                              │
│  4. COMPLEMENTARIEDAD:                                       │
│     μⱼ · hⱼ(x*) = 0  para todo j                           │
│     (Si hⱼ < 0 está inactiva → μⱼ = 0)                     │
│     (Si μⱼ > 0 → hⱼ = 0 está activa)                       │
└─────────────────────────────────────────────────────────────┘
```

## 7. Aplicaciones en ML: Ejemplos Concretos

### Regresión Ridge (L2 Regularización)

```
┌─────────────────────────────────────────────────────────────┐
│  RIDGE REGRESSION                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Formulación 1 (penalización):                               │
│    min ||Xθ - y||² + λ||θ||²                                │
│                                                              │
│  Formulación 2 (restricción):                                │
│    min ||Xθ - y||²  sujeto a ||θ||² ≤ t                    │
│                                                              │
│  Ambas son EQUIVALENTES por Lagrange                         │
│  λ y t están relacionados inversamente                       │
│                                                              │
│  Solución analítica (gradiente = 0):                         │
│    θ* = (X^T X + λI)⁻¹ X^T y                                │
└─────────────────────────────────────────────────────────────┘
```

```python
def ridge_regression(X: np.ndarray, y: np.ndarray, lambd: float) -> np.ndarray:
    """Ridge regression con solución analítica."""
    n, d = X.shape
    I = np.eye(d)

    # Solución: (X^T X + λI)^(-1) X^T y
    theta = np.linalg.solve(X.T @ X + lambd * I, X.T @ y)

    return theta

# Comparar con y sin regularización
np.random.seed(42)
n, d = 50, 100  # Más features que muestras (problema mal condicionado)
X = np.random.randn(n, d)
theta_real = np.zeros(d)
theta_real[:5] = np.array([1., -2., 0.5, 1.5, -1.])  # Solo 5 features activas
y = X @ theta_real + 0.1 * np.random.randn(n)

# Sin regularización (OLS) - mal condicionado
try:
    theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
except:
    theta_ols = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]

# Con Ridge
theta_ridge = ridge_regression(X, y, lambd=1.0)

print(f"||θ_OLS||: {np.linalg.norm(theta_ols):.2f}")
print(f"||θ_Ridge||: {np.linalg.norm(theta_ridge):.2f}")
print(f"Error OLS vs real: {np.linalg.norm(theta_ols - theta_real):.2f}")
print(f"Error Ridge vs real: {np.linalg.norm(theta_ridge - theta_real):.2f}")
```

### Lasso (L1 Regularización) - No Diferenciable

```
┌─────────────────────────────────────────────────────────────┐
│  LASSO REGRESSION                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  min ||Xθ - y||² + λ||θ||₁                                  │
│                                                              │
│  Problema: ||θ||₁ = Σ|θᵢ| NO es diferenciable en θᵢ = 0   │
│                                                              │
│  Solución: Proximal Gradient Descent                        │
│                                                              │
│    θ(t+1) = prox_λ(θ(t) - α∇L_smooth(θ(t)))                │
│                                                              │
│  Donde prox_λ es el operador "soft thresholding":          │
│    prox_λ(x)ᵢ = sign(xᵢ) · max(|xᵢ| - λ, 0)               │
│                                                              │
│  Efecto: Lleva θᵢ exactamente a 0 → SPARSITY               │
└─────────────────────────────────────────────────────────────┘
```

```python
def soft_threshold(x: np.ndarray, lambd: float) -> np.ndarray:
    """Operador de soft thresholding (proximal de L1)."""
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

def lasso_proximal_gd(
    X: np.ndarray,
    y: np.ndarray,
    lambd: float,
    lr: float = 0.001,
    max_iters: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """Lasso via Proximal Gradient Descent."""
    n, d = X.shape
    theta = np.zeros(d)

    for i in range(max_iters):
        # Gradiente de la parte smooth (MSE)
        grad = (2/n) * X.T @ (X @ theta - y)

        # Paso de gradiente
        theta_intermedio = theta - lr * grad

        # Paso proximal (soft thresholding)
        theta_nuevo = soft_threshold(theta_intermedio, lr * lambd)

        # Verificar convergencia
        if np.linalg.norm(theta_nuevo - theta) < tol:
            break

        theta = theta_nuevo

    return theta

# Ejemplo: recuperar señal sparse
np.random.seed(42)
n, d = 100, 50
X = np.random.randn(n, d)
theta_real = np.zeros(d)
theta_real[[0, 5, 10, 15, 20]] = [2., -1.5, 1., -0.5, 0.8]  # 5 de 50 no cero
y = X @ theta_real + 0.1 * np.random.randn(n)

theta_lasso = lasso_proximal_gd(X, y, lambd=0.1, lr=0.01)

# Contar ceros
print(f"Coeficientes no-cero reales: {np.sum(theta_real != 0)}")
print(f"Coeficientes no-cero Lasso: {np.sum(np.abs(theta_lasso) > 1e-4)}")
print(f"Lasso promueve SPARSITY")
```

### Logistic Regression con Newton-Raphson

```
┌─────────────────────────────────────────────────────────────┐
│  NEWTON-RAPHSON (Segundo Orden)                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Actualización:                                              │
│    θ(t+1) = θ(t) - H⁻¹ · ∇L(θ(t))                          │
│                                                              │
│  Donde H = Hessiano                                          │
│                                                              │
│  Ventajas:                                                   │
│    • Convergencia CUADRÁTICA (muy rápida cerca del óptimo)  │
│    • No requiere tuning de learning rate                    │
│                                                              │
│  Desventajas:                                                │
│    • O(d³) por inversión del Hessiano                       │
│    • Requiere calcular segundas derivadas                   │
│    • No escalable a deep learning                           │
│                                                              │
│  Usado en: glm, sklearn LogisticRegression (solver='lbfgs')│
└─────────────────────────────────────────────────────────────┘
```

## 8. Convergencia: Análisis Teórico

### Tasas de Convergencia

```
┌─────────────────────────────────────────────────────────────┐
│  TASAS DE CONVERGENCIA                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Gradient Descent (f convexa, L-smooth):                    │
│    • f(θₜ) - f(θ*) ≤ O(1/t)                                │
│    • Convergencia SUBLINEAL                                  │
│                                                              │
│  GD (f fuertemente convexa, L-smooth):                      │
│    • ||θₜ - θ*|| ≤ O((1 - μ/L)^t)                          │
│    • Convergencia LINEAL (exponencial)                      │
│    • μ = parámetro de convexidad fuerte                    │
│                                                              │
│  Newton-Raphson:                                             │
│    • ||θₜ₊₁ - θ*|| ≤ O(||θₜ - θ*||²)                      │
│    • Convergencia CUADRÁTICA (muy rápida)                   │
│                                                              │
│  Nota: L = constante de Lipschitz del gradiente             │
│        Requiere α ≤ 1/L para garantizar convergencia        │
└─────────────────────────────────────────────────────────────┘
```

### Número de Condición y Convergencia

```
El NÚMERO DE CONDICIÓN κ = λ_max/λ_min determina la velocidad:

κ PEQUEÑO (≈ 1)                    κ GRANDE (>> 1)
Curvas de nivel circulares         Curvas de nivel alargadas

    │     ╭───╮                        │     ╱─────────╲
    │    ╱     ╲                       │    ╱           ╲
    │   │   ●   │                      │   │      ●      │
    │    ╲     ╱                       │    ╲           ╱
    │     ╰───╯                        │     ╲─────────╱
    └───────────                       └─────────────────

Convergencia RÁPIDA                Convergencia LENTA
GD va directo al mínimo            GD zigzaguea mucho

Solución: PRECONDICIONAR (ej: BatchNorm en DL)
```

```python
def analizar_condicionamiento(X: np.ndarray):
    """Analiza el condicionamiento de X^T X."""
    H = X.T @ X  # Hessiano de MSE (sin factor 1/n)

    eigenvalues = np.linalg.eigvalsh(H)
    lambda_max = eigenvalues[-1]
    lambda_min = eigenvalues[0]

    if lambda_min > 1e-10:
        kappa = lambda_max / lambda_min
    else:
        kappa = np.inf

    print(f"λ_max: {lambda_max:.4f}")
    print(f"λ_min: {lambda_min:.4f}")
    print(f"Número de condición κ: {kappa:.2f}")

    if kappa < 10:
        print("→ Bien condicionado, GD convergerá rápido")
    elif kappa < 1000:
        print("→ Moderadamente condicionado")
    else:
        print("→ Mal condicionado, considerar regularización")

    return kappa

# Ejemplo bien condicionado
np.random.seed(42)
X_bueno = np.random.randn(100, 5)
print("Dataset bien condicionado:")
analizar_condicionamiento(X_bueno)

print("\nDataset mal condicionado (features correlacionadas):")
X_malo = X_bueno.copy()
X_malo[:, 1] = X_malo[:, 0] + 0.01 * np.random.randn(100)  # Casi idénticas
analizar_condicionamiento(X_malo)
```

## 9. Resumen: Mapa de Optimización en ML

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAPA DE OPTIMIZACIÓN EN ML                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐                                                    │
│  │ ¿Es diferenciable│───NO──→ Proximal Methods, Subgradient            │
│  │   el objetivo?   │         (ej: Lasso)                               │
│  └────────┬─────────┘                                                    │
│           │ SÍ                                                           │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │  ¿Es convexo?   │───NO──→ SGD + Momentum, Adam                      │
│  │                 │         (mínimos locales, escapar con ruido)       │
│  └────────┬─────────┘                                                    │
│           │ SÍ                                                           │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │ ¿Dataset grande?│───SÍ──→ SGD, Mini-batch GD, Adam                  │
│  │   (> 10k)       │         (eficiencia O(batch))                      │
│  └────────┬─────────┘                                                    │
│           │ NO                                                           │
│           ▼                                                              │
│  ┌─────────────────┐                                                    │
│  │ ¿Hay solución   │───SÍ──→ Solución analítica                        │
│  │   cerrada?      │         (ej: OLS, Ridge)                           │
│  └────────┬─────────┘                                                    │
│           │ NO                                                           │
│           ▼                                                              │
│      Newton-Raphson, L-BFGS                                              │
│      (segundo orden, convergencia rápida)                               │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  REGLA PRÁCTICA:                                                         │
│    • Deep Learning: Adam (90% de los casos)                             │
│    • ML Clásico pequeño: L-BFGS o Newton                                │
│    • ML Clásico grande: SGD con momentum                                │
│    • Producción: Adam con learning rate scheduler                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conceptos Clave

```
┌─────────────────────────────────────────────────────────────┐
│  RESUMEN DE CONCEPTOS                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CONVEXIDAD                                                  │
│    • Función convexa → único mínimo global                  │
│    • Verificar: Hessiano ≥ 0 (semidefinido positivo)       │
│    • MSE es convexa, Cross-Entropy es convexa               │
│                                                              │
│  GRADIENTE                                                   │
│    • ∇f = vector de derivadas parciales                     │
│    • Apunta hacia máximo crecimiento                        │
│    • -∇f = dirección de descenso                            │
│                                                              │
│  OPTIMIZADORES                                               │
│    • GD: Básico, usa todo el dataset                        │
│    • SGD: Rápido, ruidoso                                   │
│    • Momentum: Acelera, suaviza                             │
│    • Adam: Adaptativo, el estándar                          │
│                                                              │
│  RESTRICCIONES                                               │
│    • Lagrange: convierte restricciones en penalizaciones    │
│    • KKT: condiciones necesarias y suficientes              │
│    • Ridge/Lasso son optimización restringida               │
│                                                              │
│  CONVERGENCIA                                                │
│    • Depende del número de condición κ = λ_max/λ_min       │
│    • κ grande → lento, necesita precondicionamiento        │
│    • Learning rate crítico: α ≤ 1/L                        │
└─────────────────────────────────────────────────────────────┘
```

---

**Conexiones con otros temas:**
- Gradient Descent → Backpropagation en Redes Neuronales
- Lagrange → SVM (formulación dual)
- Convexidad → Garantías teóricas de convergencia
- Adam → Entrenamiento de Transformers, GPT, BERT

**Siguiente:** Estadística Bayesiana para cuantificar incertidumbre en modelos
