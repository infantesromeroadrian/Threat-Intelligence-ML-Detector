# Support Vector Machines (SVM)

## 1. ¿Qué es SVM?

### Intuición Geométrica

```
┌────────────────────────────────────────────────────────────────┐
│  OBJETIVO DE SVM                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Encontrar el HIPERPLANO que mejor separa las clases           │
│  maximizando el MARGEN entre las clases                        │
│                                                                │
│       ●  ●                                                     │
│     ●   ●  ●                                                   │
│    ●  ●    ●            │                     ○  ○            │
│      ●   ●              │                   ○   ○  ○          │
│    ●   ●                │                  ○  ○    ○          │
│                         │                    ○   ○            │
│                         │                                      │
│      Clase 0        Hiperplano           Clase 1               │
│                      Separador                                 │
│                                                                │
│  ● = Clase 0 (ej: tráfico normal)                             │
│  ○ = Clase 1 (ej: tráfico malicioso)                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### ¿Por qué Maximizar el Margen?

```
Opción A: Margen pequeño          Opción B: Margen GRANDE
                                  (SVM elige esta)

  ●  ●              ○  ○            ●  ●              ○  ○
●   ●  ●│          ○   ○          ●   ●  ●    │    ○   ○
 ●  ●   │●        ○  ○   ○          ●  ●  ║   │    ○  ○   ○
  ●   ● │       ○   ○                ●   ●║   │  ○   ○
 ●   ●  │      ○                    ●   ● ║   │ ○
                                           margen

Margen pequeño:                   Margen grande:
- Sensible a ruido               - Más robusto
- Generaliza MAL                 - Generaliza BIEN
- Overfitting                    - Mejor rendimiento

SVM busca el hiperplano con el MARGEN MÁS GRANDE posible
```

## 2. Conceptos Fundamentales

### Support Vectors

```
┌────────────────────────────────────────────────────────────────┐
│  SUPPORT VECTORS = Puntos MÁS CERCANOS al hiperplano           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│       ●  ●                                                     │
│     ●   ●  ●                                                   │
│    ●  ●    ●                                    ○  ○          │
│      [●]  ←───── Support Vector       ───→ [○] ○   ○  ○       │
│    ●   [●] ←─────────────────────────────→ [○]  ○    ○        │
│                        ║                     ○   ○            │
│                        ║                                      │
│                   Hiperplano                                   │
│                                                                │
│  Solo los SUPPORT VECTORS definen el hiperplano                │
│  Los demás puntos NO influyen en la decisión                   │
│                                                                │
│  Esto hace a SVM eficiente en memoria y robusto               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Formalización Matemática

```
┌────────────────────────────────────────────────────────────────┐
│  HIPERPLANO EN 2D                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Ecuación: w · x + b = 0                                       │
│                                                                │
│  Donde:                                                        │
│    w = vector de pesos (normal al hiperplano)                  │
│    x = vector de features                                      │
│    b = bias (intersección)                                     │
│                                                                │
│  Clasificación:                                                │
│    Si w · x + b > 0  →  Clase +1                              │
│    Si w · x + b < 0  →  Clase -1                              │
│                                                                │
│  El MARGEN se define como: 2 / ||w||                           │
│  Maximizar margen = Minimizar ||w||                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Problema de Optimización

```
┌────────────────────────────────────────────────────────────────┐
│  OPTIMIZACIÓN SVM                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  OBJETIVO: Minimizar  (1/2) ||w||²                             │
│                                                                │
│  SUJETO A: yᵢ(w · xᵢ + b) ≥ 1  para todo i                    │
│                                                                │
│  Interpretación:                                               │
│    • Minimizar ||w||² = Maximizar margen                       │
│    • Restricción: todos los puntos correctamente clasificados │
│      y fuera del margen                                        │
│                                                                │
│  Este es un problema de OPTIMIZACIÓN CUADRÁTICA CONVEXA        │
│  Tiene solución única y global                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 3. Soft Margin: Manejando Datos No Separables

### El Problema

```
Datos LINEALMENTE SEPARABLES:     Datos NO SEPARABLES:

  ●  ●        │      ○  ○            ●  ●     ○  │  ○  ○
●   ●  ●      │    ○   ○  ○        ●   ●  ●  ●  │○   ○  ○
 ●  ●    ●    │   ○  ○    ○         ●  ○   ●   ●│  ○    ○
   ●   ●      │     ○   ○             ●   ●   ○ │  ○   ○
 ●   ●        │    ○                 ●   ●     ○│ ○

Se puede separar                  NO se puede separar
perfectamente                     con una línea recta

¿Solución? Permitir algunos errores (SOFT MARGIN)
```

### Soft Margin SVM

```
┌────────────────────────────────────────────────────────────────┐
│  SOFT MARGIN: Permitir violaciones del margen                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  NUEVO OBJETIVO:                                               │
│                                                                │
│    Minimizar  (1/2) ||w||² + C · Σξᵢ                          │
│                                                                │
│  Donde:                                                        │
│    ξᵢ = "slack variable" (penalización por error)             │
│    C = parámetro de regularización                             │
│                                                                │
│  C controla el trade-off:                                      │
│                                                                │
│    C ALTO (ej: 1000)           C BAJO (ej: 0.01)              │
│    ────────────────            ─────────────────               │
│    • Penaliza mucho errores    • Tolera más errores           │
│    • Margen estrecho           • Margen amplio                │
│    • Riesgo de OVERFITTING     • Riesgo de UNDERFITTING       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización de C

```
C = 0.01 (bajo)              C = 1 (medio)              C = 100 (alto)

●  ●     ○  │  ○  ○        ●  ●        │  ○  ○        ●  ●        │○  ○
●   ●  ● ○  │○   ○  ○     ●   ●  ●     │ ○   ○  ○    ●   ●  ●  ●│ ○   ○  ○
 ●  ○   ●   │  ○    ○      ●  ○   ●    │   ○    ○     ●  ○   ● │    ○    ○
   ●   ●  ○ │  ○   ○         ●   ●  ○  │  ○   ○         ●   ●  │○  ○   ○
 ●   ●    ○ │○                ●   ●    │ ○              ●   ● │   ○

Margen AMPLIO              Margen MEDIO               Margen ESTRECHO
Muchos errores             Balance                    Pocos errores
en training                                           pero overfitting
```

## 4. El Kernel Trick

### Problema: Datos No Linealmente Separables

```
Datos originales (2D):                No hay LÍNEA que los separe

    ●  ●  ●  ●  ●                     Pero... ¿y si proyectamos
  ●              ●                    a un espacio de mayor
 ●    ○  ○  ○    ●                    dimensión?
●    ○      ○    ●
●    ○  ○  ○     ●
 ●              ●
  ●  ●  ●  ●  ●

Los ○ están RODEADOS por ●
Necesitamos una frontera CIRCULAR, no lineal
```

### Proyección a Mayor Dimensión

```
┌────────────────────────────────────────────────────────────────┐
│  KERNEL TRICK                                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Proyectar datos a un espacio de MAYOR dimensión         │
│        donde SÍ sean linealmente separables                    │
│                                                                │
│  ESPACIO ORIGINAL (2D)        ESPACIO TRANSFORMADO (3D)        │
│                                                                │
│       ●  ●  ●                          ●  ●  ●                 │
│     ●  ○  ○  ●                       ●        ●               │
│    ●  ○    ○  ●      ─────→        ●            ●             │
│     ●  ○  ○  ●       φ(x)           ○  ○  ○  ○               │
│       ●  ●  ●                        (más abajo)              │
│                                                                │
│  φ(x) añade dimensión: z = x² + y²                            │
│  Ahora se pueden separar con un PLANO                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Kernels Comunes

```
┌─────────────────┬────────────────────────────────────────────┐
│  Kernel         │  Fórmula                                   │
├─────────────────┼────────────────────────────────────────────┤
│                 │                                            │
│  Linear         │  K(x, x') = x · x'                        │
│                 │  Para datos linealmente separables         │
│                 │                                            │
├─────────────────┼────────────────────────────────────────────┤
│                 │                                            │
│  Polynomial     │  K(x, x') = (γ·x·x' + r)^d               │
│                 │  d = grado del polinomio                   │
│                 │                                            │
├─────────────────┼────────────────────────────────────────────┤
│                 │                                            │
│  RBF            │  K(x, x') = exp(-γ||x - x'||²)            │
│  (Gaussian)     │  El más usado, muy flexible                │
│                 │  γ controla la "anchura" de la campana     │
│                 │                                            │
├─────────────────┼────────────────────────────────────────────┤
│                 │                                            │
│  Sigmoid        │  K(x, x') = tanh(γ·x·x' + r)              │
│                 │  Similar a redes neuronales                │
│                 │                                            │
└─────────────────┴────────────────────────────────────────────┘
```

### Efecto de γ (gamma) en RBF

```
γ PEQUEÑO (0.01)              γ MEDIO (1)              γ GRANDE (100)

Frontera SUAVE              Frontera MEDIA            Frontera COMPLEJA

    ●  ●  ●                     ●  ●  ●                   ●  ●  ●
  ●╱────────╲●                ●╱──────╲●                ●╭──╮╭──╮●
 ●│  ○  ○  ○ │●              ●│ ○  ○  ○│●              ●│○ ││○ │●
●│  ○      ○ │●             ●│ ○    ○ │●             ●╰╯○╰╯╰○╯●
●│  ○  ○  ○  │●             ●│ ○  ○  ○│●             ●╭╮○╭─╮○╭╮●
 ●╲─────────╱●               ●╲──────╱●                ●╰╯─╯ ╰╯●
  ●  ●  ●  ●                   ●  ●  ●                   ●  ●  ●

Generaliza bien              Balance                  OVERFITTING
Puede no capturar                                    (memoriza cada
patrones complejos                                    punto)
```

## 5. Hiperparámetros Clave

### Tabla de Hiperparámetros

```
┌─────────────────┬─────────────────┬────────────────────────────┐
│  Parámetro      │  Default        │  Descripción               │
├─────────────────┼─────────────────┼────────────────────────────┤
│ C               │  1.0            │ Regularización             │
│                 │                 │ Alto: menos regularización │
│                 │                 │ Bajo: más regularización   │
├─────────────────┼─────────────────┼────────────────────────────┤
│ kernel          │  'rbf'          │ Tipo de kernel             │
│                 │                 │ 'linear', 'poly', 'rbf',   │
│                 │                 │ 'sigmoid'                  │
├─────────────────┼─────────────────┼────────────────────────────┤
│ gamma           │  'scale'        │ Coeficiente del kernel     │
│                 │                 │ Alto: más complejo         │
│                 │                 │ Bajo: más suave            │
├─────────────────┼─────────────────┼────────────────────────────┤
│ degree          │  3              │ Grado para kernel 'poly'   │
├─────────────────┼─────────────────┼────────────────────────────┤
│ class_weight    │  None           │ Peso por clase             │
│                 │                 │ 'balanced' para desbalance │
└─────────────────┴─────────────────┴────────────────────────────┘
```

### Guía de Ajuste

```
┌────────────────────────────────────────────────────────────────┐
│  AJUSTE DE HIPERPARÁMETROS SVM                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PASO 1: Elegir kernel                                         │
│  ────────────────────                                          │
│    • Empezar con 'linear' si datos parecen separables          │
│    • 'rbf' es el default y más flexible                        │
│    • 'poly' para relaciones polinómicas conocidas              │
│                                                                │
│  PASO 2: Escalar datos (OBLIGATORIO)                           │
│  ─────────────────────────────────────                         │
│    from sklearn.preprocessing import StandardScaler            │
│    scaler = StandardScaler()                                   │
│    X_scaled = scaler.fit_transform(X)                          │
│                                                                │
│  PASO 3: Grid Search sobre C y gamma                           │
│  ──────────────────────────────────                            │
│    C: [0.01, 0.1, 1, 10, 100]                                  │
│    gamma: ['scale', 'auto', 0.01, 0.1, 1]                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 6. Implementación en Python

### Código Básico

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Datos de ejemplo
np.random.seed(42)
X = np.vstack([
    np.random.randn(100, 2) + [2, 2],   # Clase 0
    np.random.randn(100, 2) + [-2, -2]  # Clase 1
])
y = np.array([0]*100 + [1]*100)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# IMPORTANTE: Escalar datos para SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar SVM
svm = SVC(
    kernel='rbf',     # Kernel gaussiano
    C=1.0,            # Regularización
    gamma='scale',    # Automático basado en varianza
    random_state=42
)

svm.fit(X_train_scaled, y_train)

# Evaluación
print(f"Accuracy Train: {svm.score(X_train_scaled, y_train):.3f}")
print(f"Accuracy Test:  {svm.score(X_test_scaled, y_test):.3f}")
print(f"\nNúmero de Support Vectors: {len(svm.support_vectors_)}")
print(f"Support Vectors por clase: {svm.n_support_}")
```

### Grid Search para Optimización

```python
from sklearn.model_selection import GridSearchCV

# Definir grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

# Grid Search
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1 (CV): {grid_search.best_score_:.3f}")

# Evaluar mejor modelo
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### Visualización de la Frontera de Decisión

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title):
    """Visualiza la frontera de decisión de un SVM"""
    h = 0.02  # paso de la malla

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')

    # Marcar support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='black', linewidths=2,
                label='Support Vectors')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()

# Usar
plot_decision_boundary(svm, X_test_scaled, y_test,
                       f"SVM (kernel=RBF, C={svm.C})")
```

## 7. SVM para Multiclase

### Estrategias

```
┌────────────────────────────────────────────────────────────────┐
│  SVM es BINARIO por naturaleza                                  │
│  Para multiclase: combinar múltiples SVMs binarios             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ONE-VS-REST (OvR)                                             │
│  ─────────────────                                             │
│  Para K clases: K clasificadores                               │
│                                                                │
│    Clase 1 vs (2,3,4)                                          │
│    Clase 2 vs (1,3,4)                                          │
│    Clase 3 vs (1,2,4)                                          │
│    Clase 4 vs (1,2,3)                                          │
│                                                                │
│  Predicción: clase con mayor "confianza"                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ONE-VS-ONE (OvO) - Default en sklearn                         │
│  ────────────────────────────────────                          │
│  Para K clases: K(K-1)/2 clasificadores                        │
│                                                                │
│    Clase 1 vs 2                                                │
│    Clase 1 vs 3                                                │
│    Clase 1 vs 4                                                │
│    Clase 2 vs 3                                                │
│    Clase 2 vs 4                                                │
│    Clase 3 vs 4                                                │
│                                                                │
│  Predicción: votación mayoritaria                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código Multiclase

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset multiclase
iris = load_iris()
X, y = iris.data, iris.target  # 3 clases

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM automáticamente usa OvO para multiclase
svm = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')
svm.fit(X_train_scaled, y_train)

print(f"Accuracy: {svm.score(X_test_scaled, y_test):.3f}")
```

## 8. Ejemplo Práctico: Detección de Phishing

```python
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Simular dataset de URLs
np.random.seed(42)
n_samples = 2000

# Features típicas de detección de phishing
data = {
    'longitud_url': np.random.normal(50, 20, n_samples),
    'num_puntos': np.random.poisson(3, n_samples),
    'num_guiones': np.random.poisson(1, n_samples),
    'tiene_https': np.random.binomial(1, 0.6, n_samples),
    'tiene_ip': np.random.binomial(1, 0.1, n_samples),
    'num_subdominios': np.random.poisson(2, n_samples),
    'longitud_dominio': np.random.normal(15, 5, n_samples),
    'tiene_at': np.random.binomial(1, 0.05, n_samples),
    'tiene_doble_barra': np.random.binomial(1, 0.1, n_samples),
    'edad_dominio_dias': np.random.exponential(500, n_samples),
}

df = pd.DataFrame(data)

# Etiquetas: phishing si cumple varios criterios sospechosos
df['es_phishing'] = (
    (df['longitud_url'] > 75) |
    (df['tiene_ip'] == 1) |
    (df['tiene_at'] == 1) |
    (df['tiene_https'] == 0) & (df['num_subdominios'] > 3) |
    (df['edad_dominio_dias'] < 30)
).astype(int)

# Añadir ruido
ruido = np.random.binomial(1, 0.08, n_samples)
df['es_phishing'] = (df['es_phishing'] + ruido) % 2

print("Distribución de clases:")
print(df['es_phishing'].value_counts())

# Preparar datos
X = df.drop('es_phishing', axis=1)
y = df['es_phishing']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# OBLIGATORIO: Escalar para SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grid Search para encontrar mejores parámetros
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

print("\nOptimizando hiperparámetros...")
grid_search = GridSearchCV(
    SVC(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nMejores parámetros: {grid_search.best_params_}")
print(f"Mejor F1 (CV): {grid_search.best_score_:.4f}")

# Evaluar mejor modelo
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

print("\n" + "=" * 60)
print("RESULTADOS FINALES")
print("=" * 60)
print(f"\nAccuracy: {best_svm.score(X_test_scaled, y_test):.4f}")
print(f"Support Vectors: {len(best_svm.support_vectors_)}")
print(f"SV por clase: {best_svm.n_support_}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legítimo', 'Phishing']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Interpretación para ciberseguridad
print("\n" + "=" * 60)
print("INTERPRETACIÓN PARA CIBERSEGURIDAD")
print("=" * 60)
print(f"""
Matriz de Confusión:
                   Predicho
                 Legítimo  Phishing
Actual Legítimo    {cm[0,0]:4d}      {cm[0,1]:4d}   (Falsos positivos: URLs legítimas bloqueadas)
       Phishing    {cm[1,0]:4d}      {cm[1,1]:4d}   (Falsos negativos: Phishing NO detectado!)

Falsos Negativos ({cm[1,0]}) son CRÍTICOS en seguridad:
  - Cada phishing no detectado es un ataque exitoso
  - Considerar ajustar threshold o class_weight para reducirlos

Trade-off:
  - Más FP = usuarios frustrados por bloqueos incorrectos
  - Más FN = ataques exitosos (PEOR en seguridad)
""")
```

## 9. SVM vs Otros Clasificadores

```
┌─────────────────────┬────────────────────────────────────────┐
│      Aspecto        │  SVM                                   │
├─────────────────────┼────────────────────────────────────────┤
│ Datos pequeños      │  ★★★★★  Excelente                     │
├─────────────────────┼────────────────────────────────────────┤
│ Datos grandes       │  ★★☆☆☆  Lento (O(n²) a O(n³))        │
├─────────────────────┼────────────────────────────────────────┤
│ Alta dimensión      │  ★★★★★  Muy bueno (text, genomics)    │
├─────────────────────┼────────────────────────────────────────┤
│ Interpretabilidad   │  ★★☆☆☆  Difícil (especialmente RBF)   │
├─────────────────────┼────────────────────────────────────────┤
│ Outliers            │  ★★★☆☆  Sensible, pero C ayuda        │
├─────────────────────┼────────────────────────────────────────┤
│ Features categóricas│  ★★☆☆☆  Requiere encoding            │
├─────────────────────┼────────────────────────────────────────┤
│ Probabilidades      │  ★★☆☆☆  No nativo (usa Platt scaling) │
├─────────────────────┼────────────────────────────────────────┤
│ Tuning requerido    │  ★★★★☆  C, gamma son críticos         │
└─────────────────────┴────────────────────────────────────────┘

COMPARACIÓN CON OTROS MODELOS:

           │  Accuracy  │  Velocidad  │ Interpretable │  Alta Dim
───────────┼────────────┼─────────────┼───────────────┼───────────
SVM        │   Alta     │   Media     │     Baja      │   Muy Bueno
───────────┼────────────┼─────────────┼───────────────┼───────────
Logistic   │   Media    │   Alta      │     Alta      │   Bueno
───────────┼────────────┼─────────────┼───────────────┼───────────
Random     │   Alta     │   Media     │     Media     │   Bueno
Forest     │            │             │               │
───────────┼────────────┼─────────────┼───────────────┼───────────
Naive      │   Media    │   Muy Alta  │     Alta      │   Muy Bueno
Bayes      │            │             │               │
```

## 10. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE SVM                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Muy efectivo en espacios de alta dimensionalidad            │
│  ✓ Eficiente en memoria (solo guarda support vectors)          │
│  ✓ Versátil: diferentes kernels para diferentes problemas      │
│  ✓ Funciona bien cuando n_features > n_samples                 │
│  ✓ Robusto contra overfitting (margen + regularización)        │
│  ✓ Garantías teóricas sólidas                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE SVM                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Lento con datasets grandes (>10K muestras)                  │
│  ✗ Requiere ESCALADO de features obligatoriamente              │
│  ✗ No proporciona probabilidades directamente                  │
│  ✗ Difícil de interpretar (caja negra con RBF)                 │
│  ✗ Sensible a la elección de kernel y parámetros               │
│  ✗ No maneja bien datos con mucho ruido                        │
│  ✗ Features categóricas requieren encoding                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 11. Cuándo Usar SVM

```
┌────────────────────────────────────────────────────────────────┐
│  CASOS DE USO IDEALES                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Clasificación de texto (spam, sentimiento)                  │
│  ✓ Reconocimiento de imágenes (antes de Deep Learning)         │
│  ✓ Bioinformática (clasificación de genes)                     │
│  ✓ Detección de malware basada en features                     │
│  ✓ Detección de intrusiones                                    │
│  ✓ Cuando tienes pocas muestras pero muchas features           │
│  ✓ Problemas de clasificación binaria con margen claro         │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│  EVITAR CUANDO                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Dataset muy grande (>100K muestras) - usar SGDClassifier    │
│  ✗ Necesitas probabilidades precisas                           │
│  ✗ Datos muy ruidosos                                          │
│  ✗ Necesitas interpretabilidad del modelo                      │
│  ✗ Datos en tiempo real con latencia crítica                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 12. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  SUPPORT VECTOR MACHINES - RESUMEN                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CONCEPTO:                                                     │
│    Encontrar el hiperplano que maximiza el margen              │
│    entre las clases                                            │
│                                                                │
│  COMPONENTES CLAVE:                                            │
│    • Support Vectors: puntos más cercanos al hiperplano        │
│    • Margen: distancia entre hiperplano y support vectors      │
│    • Kernel Trick: proyectar a mayor dimensión                 │
│                                                                │
│  HIPERPARÁMETROS:                                              │
│    • C: trade-off margen vs errores                            │
│    • kernel: 'linear', 'rbf', 'poly', 'sigmoid'                │
│    • gamma: complejidad del kernel RBF                         │
│                                                                │
│  OBLIGATORIO:                                                  │
│    • Escalar features (StandardScaler)                         │
│    • Grid Search sobre C y gamma                               │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Detección de phishing                                     │
│    • Clasificación de malware                                  │
│    • Análisis de texto (spam, amenazas)                        │
│    • Detección de anomalías                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Naive Bayes
