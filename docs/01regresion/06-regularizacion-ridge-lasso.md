# Regularización: Ridge y Lasso

## 1. El Problema del Overfitting

### ¿Qué es Overfitting?

```
El modelo MEMORIZA los datos de entrenamiento
en lugar de APRENDER patrones generales.

┌────────────────────────────────────────────────────────┐
│  SÍNTOMAS DE OVERFITTING                               │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • Error MUY bajo en training                          │
│  • Error ALTO en test/validación                       │
│  • Coeficientes θ muy GRANDES                          │
│  • Predicciones erráticas en datos nuevos              │
│                                                        │
└────────────────────────────────────────────────────────┘

Visualización:
y │      ╱╲
  │    ╱●  ╲●         Curva pasa por TODOS los puntos
  │  ●╱      ╲●       pero no captura la tendencia real
  │ ╱    ╱╲   ╲
  │╱    ╱  ╲   ╲●
  └─────────────────── x
```

### Causa: Coeficientes Demasiado Grandes

```
Modelo con overfitting típico:

h(x) = 1000 + 5000x - 8000x² + 12000x³ - 3000x⁴

Los coeficientes ENORMES hacen que pequeños cambios
en x produzcan cambios DRÁSTICOS en la predicción.

Ejemplo:
  x = 2.00  →  h(x) = 15,000
  x = 2.01  →  h(x) = 28,000  ← ¡Cambio enorme!

Esto es INESTABLE y no generaliza bien.
```

## 2. Solución: Regularización

### Concepto

**Regularización:** Penalizar coeficientes grandes para mantenerlos pequeños.

```
┌────────────────────────────────────────────────────────┐
│  IDEA DE REGULARIZACIÓN                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Sin regularización:                                   │
│    Minimizar: J(θ) = Error en predicciones            │
│                                                        │
│  Con regularización:                                   │
│    Minimizar: J(θ) = Error + λ·(tamaño de θ)         │
│                       ↑              ↑                 │
│                   Ajuste a      Penalización           │
│                   los datos     por θ grandes          │
│                                                        │
│  λ (lambda) controla el balance:                       │
│    λ pequeño → Más ajuste a datos (posible overfit)   │
│    λ grande  → Más penalización (posible underfit)    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Tipos de Regularización

```
┌─────────────────┬────────────────────┬─────────────────┐
│                 │       RIDGE        │      LASSO      │
│                 │   (L2 Penalty)     │  (L1 Penalty)   │
├─────────────────┼────────────────────┼─────────────────┤
│  Penalización   │   λ·Σθᵢ²          │   λ·Σ|θᵢ|      │
├─────────────────┼────────────────────┼─────────────────┤
│  Efecto en θ    │   Reduce todos     │   Algunos → 0   │
│                 │   (ninguno a 0)    │   (selección)   │
├─────────────────┼────────────────────┼─────────────────┤
│  Uso típico     │   Muchas features  │   Selección de  │
│                 │   correlacionadas  │   features      │
└─────────────────┴────────────────────┴─────────────────┘
```

## 3. Ridge Regression (L2)

### Fórmula

```
┌────────────────────────────────────────────────────────┐
│  RIDGE REGRESSION                                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│              1   m                       n             │
│  J(θ) = ─────── Σ [h_θ(x⁽ⁱ⁾)-y⁽ⁱ⁾]² + λ·Σ θⱼ²        │
│              2m i=1                     j=1            │
│         ↑────────────────────↑     ↑────────↑         │
│              MSE (error)         L2 penalty           │
│                                                        │
│  Nota: θ₀ (bias) NO se regulariza, solo θ₁...θₙ      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Efecto Visual

```
Sin Ridge (λ=0):
  θ₁ = 5000, θ₂ = -8000, θ₃ = 12000
  → Coeficientes grandes, curva errática

Con Ridge (λ=100):
  θ₁ = 50, θ₂ = -30, θ₃ = 20
  → Coeficientes pequeños, curva suave

Visualización:
y │
  │    ╱╲ ╱╲           Sin Ridge (overfitting)
  │   ╱  ╳  ╲
  │  ╱      ╲
  │
  │    ╱‾‾‾‾‾╲         Con Ridge (suave)
  │   ╱       ╲
  │  ╱         ╲
  └─────────────────── x
```

### Implementación Python

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# IMPORTANTE: Escalar features antes de Ridge
pipeline_ridge = Pipeline([
    ('scaler', StandardScaler()),  # Normalizar
    ('ridge', Ridge(alpha=1.0))    # alpha = λ
])

# Entrenar
pipeline_ridge.fit(X_train, y_train)

# Predecir
y_pred = pipeline_ridge.predict(X_test)

# Ver coeficientes
print("Coeficientes:", pipeline_ridge.named_steps['ridge'].coef_)
```

### Efecto del Parámetro α (lambda)

```
α = 0:      Regresión lineal normal (sin regularización)
α = 0.01:   Regularización muy suave
α = 1:      Regularización moderada (default)
α = 100:    Regularización fuerte
α = 10000:  Regularización extrema (underfit)

┌─────────┬───────────────────────────────────────────┐
│    α    │              EFECTO                       │
├─────────┼───────────────────────────────────────────┤
│  Bajo   │  Coef. grandes, posible overfitting      │
│  Medio  │  Balance entre ajuste y generalización   │
│  Alto   │  Coef. ~0, posible underfitting          │
└─────────┴───────────────────────────────────────────┘
```

## 4. Lasso Regression (L1)

### Fórmula

```
┌────────────────────────────────────────────────────────┐
│  LASSO REGRESSION                                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│              1   m                       n             │
│  J(θ) = ─────── Σ [h_θ(x⁽ⁱ⁾)-y⁽ⁱ⁾]² + λ·Σ |θⱼ|       │
│              2m i=1                     j=1            │
│         ↑────────────────────↑     ↑────────↑         │
│              MSE (error)         L1 penalty           │
│                                  (valor absoluto)      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Diferencia Clave: Selección de Features

```
RIDGE (L2):                    LASSO (L1):
  θ₁ = 5.2                       θ₁ = 5.0
  θ₂ = 0.3                       θ₂ = 0.0  ← ¡CERO!
  θ₃ = 2.1                       θ₃ = 2.0
  θ₄ = 0.1                       θ₄ = 0.0  ← ¡CERO!
  θ₅ = 1.8                       θ₅ = 1.5

Ridge: Todos los coef. pequeños pero NO cero
Lasso: Algunos coef. EXACTAMENTE cero
       → Elimina features irrelevantes
       → SELECCIÓN AUTOMÁTICA DE FEATURES
```

### Visualización Geométrica

```
Ridge (L2) - Restricción circular:

θ₂│      ╭───╮
  │    ╱       ╲      El óptimo toca el círculo
  │   │    ●    │     pero NO en los ejes
  │    ╲       ╱
  │      ╰───╯
  └────────────────── θ₁


Lasso (L1) - Restricción diamante:

θ₂│       ╱╲
  │      ╱  ╲         El óptimo toca el diamante
  │     ╱ ●  ╲        frecuentemente en un EJE
  │    ╱      ╲       → θ₁ = 0 o θ₂ = 0
  │   ╲        ╱
  │    ╲      ╱
  └────────────────── θ₁
```

### Implementación Python

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Pipeline con Lasso
pipeline_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

# Entrenar
pipeline_lasso.fit(X_train, y_train)

# Ver coeficientes (algunos serán 0)
coefs = pipeline_lasso.named_steps['lasso'].coef_
feature_names = X_train.columns

print("Coeficientes Lasso:")
for name, coef in zip(feature_names, coefs):
    if coef != 0:
        print(f"  {name}: {coef:.4f}")
    else:
        print(f"  {name}: 0 (eliminada)")
```

## 5. Elastic Net: Lo Mejor de Ambos

### Concepto

**Elastic Net:** Combina penalización L1 y L2.

```
┌────────────────────────────────────────────────────────┐
│  ELASTIC NET                                           │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Penalización = λ₁·Σ|θⱼ| + λ₂·Σθⱼ²                   │
│                  ↑────↑     ↑────↑                     │
│                   Lasso      Ridge                     │
│                   (L1)       (L2)                      │
│                                                        │
│  En sklearn:                                           │
│    l1_ratio = proporción de L1                        │
│    l1_ratio = 0 → Ridge puro                          │
│    l1_ratio = 1 → Lasso puro                          │
│    l1_ratio = 0.5 → Mitad y mitad                     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Implementación

```python
from sklearn.linear_model import ElasticNet

modelo = ElasticNet(
    alpha=1.0,      # Fuerza total de regularización
    l1_ratio=0.5    # Balance L1/L2 (0.5 = igual)
)

modelo.fit(X_train, y_train)
```

## 6. ¿Cuándo Usar Cada Uno?

### Guía de Decisión

```
┌─────────────────────────────────────────────────────────┐
│  ¿QUÉ REGULARIZACIÓN USAR?                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RIDGE (L2):                                            │
│    • Muchas features, todas potencialmente útiles       │
│    • Features correlacionadas entre sí                  │
│    • No necesitas eliminar features                     │
│    • Ejemplo: Datos de sensores redundantes            │
│                                                         │
│  LASSO (L1):                                            │
│    • Sospechas que muchas features son irrelevantes     │
│    • Quieres modelo interpretable (pocas features)      │
│    • Selección automática de features                   │
│    • Ejemplo: Análisis de logs con muchos campos       │
│                                                         │
│  ELASTIC NET:                                           │
│    • Muchas features correlacionadas                    │
│    • Quieres selección pero con estabilidad            │
│    • No estás seguro entre Ridge y Lasso               │
│    • Ejemplo: Datos de red con campos relacionados     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 7. Selección del Hiperparámetro α

### Grid Search con Cross-Validation

```python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Ridge con CV automático
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# RidgeCV prueba todos los alphas y elige el mejor
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Mejor alpha: {ridge_cv.alpha_}")
print(f"Score: {ridge_cv.score(X_test_scaled, y_test):.4f}")

# LassoCV hace lo mismo para Lasso
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X_train_scaled, y_train)

print(f"Mejor alpha Lasso: {lasso_cv.alpha_}")
```

### Visualización del Efecto de α

```
Error │
      │
      │ ╲
      │  ╲    Test error
      │   ╲___________╱
      │              ╱
      │             ╱
      │   _________╱   Train error
      │  ╱
      │ ╱
      └──────────────────────── α (log scale)
         0.01  0.1   1   10  100

        ↑          ↑           ↑
     Overfit   Óptimo     Underfit
```

## 8. Ejemplo Completo: Detección de Anomalías

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Simular datos de red: predecir latencia basado en métricas
np.random.seed(42)
n_samples = 500

# Features (algunas relevantes, otras ruido)
data = {
    'packets_per_sec': np.random.uniform(100, 10000, n_samples),
    'bytes_per_sec': np.random.uniform(1000, 100000, n_samples),
    'connections': np.random.uniform(10, 1000, n_samples),
    'cpu_usage': np.random.uniform(0, 100, n_samples),
    'ruido_1': np.random.randn(n_samples),  # Irrelevante
    'ruido_2': np.random.randn(n_samples),  # Irrelevante
    'ruido_3': np.random.randn(n_samples),  # Irrelevante
}

df = pd.DataFrame(data)

# Target: latencia (depende solo de algunas features)
df['latency'] = (
    0.01 * df['packets_per_sec'] +
    0.001 * df['bytes_per_sec'] +
    0.05 * df['connections'] +
    np.random.randn(n_samples) * 10
)

# Preparar datos
X = df.drop('latency', axis=1)
y = df['latency']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comparar modelos
modelos = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

print("="*60)
print("COMPARACIÓN DE REGULARIZACIÓN")
print("="*60)

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{nombre}:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²:  {r2:.4f}")
    print(f"  Coeficientes:")

    for feat, coef in zip(X.columns, modelo.coef_):
        if abs(coef) > 0.01:
            print(f"    {feat}: {coef:.4f}")
        else:
            print(f"    {feat}: ~0 (eliminada)")
```

### Output Esperado

```
============================================================
COMPARACIÓN DE REGULARIZACIÓN
============================================================

Ridge:
  MSE: 98.45
  R²:  0.9823
  Coeficientes:
    packets_per_sec: 45.2341
    bytes_per_sec: 22.1234
    connections: 18.5678
    cpu_usage: 0.0234
    ruido_1: 0.0156      ← Pequeño pero NO cero
    ruido_2: 0.0089
    ruido_3: 0.0112

Lasso:
  MSE: 99.12
  R²:  0.9818
  Coeficientes:
    packets_per_sec: 45.0000
    bytes_per_sec: 22.0000
    connections: 18.3000
    cpu_usage: ~0 (eliminada)
    ruido_1: ~0 (eliminada)   ← CERO
    ruido_2: ~0 (eliminada)   ← CERO
    ruido_3: ~0 (eliminada)   ← CERO

Lasso ELIMINA las features de ruido automáticamente!
```

## 9. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  REGULARIZACIÓN - RESUMEN                                     │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  OBJETIVO:                                                    │
│    Prevenir overfitting penalizando coeficientes grandes      │
│                                                               │
│  RIDGE (L2):                                                  │
│    J(θ) = MSE + λ·Σθⱼ²                                       │
│    • Reduce todos los coeficientes                            │
│    • Ninguno llega a cero                                     │
│    • Bueno con features correlacionadas                       │
│                                                               │
│  LASSO (L1):                                                  │
│    J(θ) = MSE + λ·Σ|θⱼ|                                      │
│    • Algunos coeficientes → exactamente 0                     │
│    • Selección automática de features                         │
│    • Modelos más interpretables                               │
│                                                               │
│  ELASTIC NET:                                                 │
│    Combina L1 + L2                                            │
│    Balance con l1_ratio                                       │
│                                                               │
│  IMPORTANTE:                                                  │
│    ⚠️ SIEMPRE escalar features antes de regularizar          │
│    ⚠️ Usar CV para elegir α óptimo                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Métricas de evaluación para modelos de regresión (MSE, RMSE, R², MAE)
