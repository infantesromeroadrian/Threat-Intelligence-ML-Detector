# Regresión Polinómica

## 1. Limitaciones de la Regresión Lineal

### El Problema: Datos No Lineales

```
La regresión lineal solo puede ajustar LÍNEAS RECTAS.
¿Qué pasa cuando los datos tienen una tendencia curva?

Coste (€)
    │
    │                           ●
    │                      ●  ●
    │                   ●
    │               ●●
    │            ●●
    │         ●●          Línea recta NO se ajusta bien
    │       ●●    ────────────────────────
    │     ●●
    │   ●●
    │ ●●
    └─────────────────────────────────────── Sistemas afectados

Los datos siguen una curva, no una recta.
Regresión lineal NO es adecuada aquí.
```

### Ejemplo Real: Ley de Rendimientos Decrecientes

```
En ciberseguridad:

  • Primeros sistemas afectados → Coste crece RÁPIDO
    (detección, contención inicial, pánico)

  • Más sistemas → Coste sigue creciendo pero MÁS LENTO
    (procesos ya establecidos, economías de escala)

  • Relación NO ES LINEAL → Es curva (logarítmica/polinómica)
```

## 2. Solución: Regresión Polinómica

### Concepto

**Regresión Polinómica:** Usar potencias de X como features adicionales.

```
┌────────────────────────────────────────────────────────┐
│  REGRESIÓN LINEAL:                                     │
│    h(x) = θ₀ + θ₁·x                                   │
│                                                        │
│  REGRESIÓN POLINÓMICA (grado 2):                       │
│    h(x) = θ₀ + θ₁·x + θ₂·x²                           │
│                                                        │
│  REGRESIÓN POLINÓMICA (grado 3):                       │
│    h(x) = θ₀ + θ₁·x + θ₂·x² + θ₃·x³                   │
│                                                        │
│  REGRESIÓN POLINÓMICA (grado n):                       │
│    h(x) = θ₀ + θ₁·x + θ₂·x² + ... + θₙ·xⁿ            │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Visualización por Grado

```
y
│
│       ●                    Datos con tendencia curva
│      ●●●
│     ●   ●
│    ●     ●●
│   ●        ●●
│  ●           ●●
│ ●              ●●
│●                 ●●
└─────────────────────────── x

Grado 1 (lineal):     ─────────────  (no se ajusta)
Grado 2 (cuadrático): ╱‾‾‾‾‾‾‾‾‾╲    (mejor ajuste)
Grado 3 (cúbico):     Curva más flexible
Grado 10:             Puede pasar por TODOS los puntos (overfitting!)
```

## 3. Transformación de Features

### De Lineal a Polinómico

```
DATOS ORIGINALES:
┌─────────┬────────┐
│    X    │    y   │
├─────────┼────────┤
│    2    │   10   │
│    3    │   18   │
│    5    │   40   │
│    7    │   75   │
└─────────┴────────┘

DATOS TRANSFORMADOS (grado 2):
┌─────────┬─────────┬────────┐
│    X    │   X²    │    y   │
├─────────┼─────────┼────────┤
│    2    │    4    │   10   │
│    3    │    9    │   18   │
│    5    │   25    │   40   │
│    7    │   49    │   75   │
└─────────┴─────────┴────────┘

Ahora aplicamos regresión lineal MULTIVARIABLE:
  h(x) = θ₀ + θ₁·X + θ₂·X²

El modelo es LINEAL en los parámetros θ,
pero POLINÓMICO en la feature original X.
```

### El Truco Matemático

```
┌────────────────────────────────────────────────────────┐
│  INSIGHT CLAVE                                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Regresión polinómica es simplemente                   │
│  regresión lineal con features transformadas.          │
│                                                        │
│  Original: X = [x]                                     │
│  Transformado: X' = [x, x², x³, ...]                  │
│                                                        │
│  Usamos el mismo algoritmo (gradient descent, etc.)   │
│  Solo cambiamos las features de entrada.              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 4. Implementación en Python

### Código Básico

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Datos de ejemplo
X = np.array([[2], [3], [5], [7], [10], [15]])
y = np.array([10, 18, 40, 75, 150, 300])

# Crear pipeline: PolynomialFeatures + LinearRegression
grado = 2

modelo_poly = Pipeline([
    ('poly_features', PolynomialFeatures(degree=grado)),
    ('linear_reg', LinearRegression())
])

# Entrenar
modelo_poly.fit(X, y)

# Predecir
X_nuevo = np.array([[8]])
prediccion = modelo_poly.predict(X_nuevo)
print(f"Predicción para X=8: {prediccion[0]:.2f}")
```

### Comparación Visual

```python
import matplotlib.pyplot as plt

# Generar puntos para graficar
X_plot = np.linspace(0, 20, 100).reshape(-1, 1)

# Entrenar modelos de diferentes grados
grados = [1, 2, 3, 5]

plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='black', label='Datos reales')

for grado in grados:
    modelo = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('reg', LinearRegression())
    ])
    modelo.fit(X, y)
    y_plot = modelo.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f'Grado {grado}')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Comparación de Grados Polinómicos')
plt.show()
```

## 5. El Problema del Grado: Underfitting vs Overfitting

### Visualización

```
UNDERFITTING (Grado muy bajo):
y │
  │     ●   ●
  │   ●       ●        La línea NO captura
  │ ●    ─────────     la tendencia real
  │●
  └─────────────── x
  Error alto en train Y test


BUEN AJUSTE (Grado correcto):
y │
  │     ●   ●
  │   ●   ╱‾‾‾╲●       La curva captura
  │ ● ╱         ╲      la tendencia
  │●╱
  └─────────────── x
  Error bajo en train Y test


OVERFITTING (Grado muy alto):
y │        ╱╲
  │     ●╱  ╲●
  │   ●╱     ╲         La curva pasa por
  │ ●╱   ╱╲   ╲●       TODOS los puntos
  │●    ╱  ╲           pero es irreal
  └─────────────── x
  Error MUY bajo en train, ALTO en test
```

### Tabla Comparativa

```
┌─────────────┬────────────────────┬────────────────────┐
│   GRADO     │  ERROR EN TRAIN    │  ERROR EN TEST     │
├─────────────┼────────────────────┼────────────────────┤
│   1 (bajo)  │      Alto          │      Alto          │
│             │   UNDERFITTING     │   UNDERFITTING     │
├─────────────┼────────────────────┼────────────────────┤
│  2-3 (ok)   │      Bajo          │      Bajo          │
│             │   BUEN AJUSTE      │   BUEN AJUSTE      │
├─────────────┼────────────────────┼────────────────────┤
│  10+ (alto) │    MUY Bajo        │     MUY Alto       │
│             │   OVERFITTING      │   OVERFITTING      │
└─────────────┴────────────────────┴────────────────────┘

Regla: Si error_train << error_test → OVERFITTING
```

## 6. Cómo Elegir el Grado Correcto

### Validación Cruzada

```
┌────────────────────────────────────────────────────────┐
│  PROCESO PARA ELEGIR GRADO                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  1. Dividir datos en train/validation/test            │
│                                                        │
│  2. Para cada grado candidato (1, 2, 3, 4, 5...):     │
│     a. Entrenar con train                             │
│     b. Evaluar con validation                         │
│     c. Guardar error de validation                    │
│                                                        │
│  3. Elegir el grado con MENOR error en validation     │
│                                                        │
│  4. Evaluar modelo final con test (una sola vez)      │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Código de Selección

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Probar diferentes grados
grados = [1, 2, 3, 4, 5, 6, 7, 8]
resultados = []

for grado in grados:
    modelo = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('reg', LinearRegression())
    ])

    # Cross-validation con 5 folds
    scores = cross_val_score(modelo, X, y, cv=5,
                            scoring='neg_mean_squared_error')
    mse_medio = -scores.mean()
    resultados.append((grado, mse_medio))
    print(f"Grado {grado}: MSE = {mse_medio:.2f}")

# Elegir el mejor
mejor_grado = min(resultados, key=lambda x: x[1])[0]
print(f"\nMejor grado: {mejor_grado}")
```

## 7. Curvas de Aprendizaje

### Diagnóstico Visual

```
Error
  │
  │ ╲
  │  ╲  Error en Training
  │   ╲____________________
  │
  │    ____________________
  │   ╱
  │  ╱   Error en Validation
  │ ╱
  └─────────────────────────── Cantidad de datos

BUEN AJUSTE:
  • Ambas curvas convergen a error bajo
  • Brecha pequeña entre train y validation

OVERFITTING:
  • Error train muy bajo
  • Error validation alto
  • Brecha grande entre ambos

UNDERFITTING:
  • Ambos errores altos
  • Convergen pero a valor alto
```

## 8. Ejemplo Completo: Ciberseguridad

### Predecir Tiempo de Respuesta a Incidentes

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Datos: Severidad (1-10) vs Tiempo de respuesta (horas)
# Relación NO lineal: severidad alta → tiempo crece exponencialmente
np.random.seed(42)
severidad = np.random.uniform(1, 10, 50).reshape(-1, 1)
tiempo = 2 + 0.5*severidad + 0.3*severidad**2 + np.random.randn(50, 1)*2

# Split
X_train, X_test, y_train, y_test = train_test_split(
    severidad, tiempo, test_size=0.2, random_state=42
)

# Comparar grados
print("Comparación de grados polinómicos:")
print("="*50)

for grado in [1, 2, 3, 4]:
    modelo = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('reg', LinearRegression())
    ])
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"Grado {grado}:")
    print(f"  MSE Train: {mse_train:.2f}")
    print(f"  MSE Test:  {mse_test:.2f}")
    print(f"  R² Test:   {r2:.3f}")
    print()
```

### Output Típico

```
Comparación de grados polinómicos:
==================================================
Grado 1:
  MSE Train: 15.23
  MSE Test:  18.45
  R² Test:   0.756

Grado 2:
  MSE Train: 4.12      ← Mejora significativa
  MSE Test:  4.89      ← Mejora significativa
  R² Test:   0.935     ← Excelente

Grado 3:
  MSE Train: 4.08
  MSE Test:  5.12      ← Empieza a subir (overfitting)
  R² Test:   0.932

Grado 4:
  MSE Train: 4.01
  MSE Test:  6.78      ← Overfitting claro
  R² Test:   0.910

Conclusión: Grado 2 es óptimo para estos datos
```

## 9. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  REGRESIÓN POLINÓMICA - RESUMEN                               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  CONCEPTO:                                                    │
│    Añadir potencias de X como features adicionales            │
│    h(x) = θ₀ + θ₁x + θ₂x² + ... + θₙxⁿ                       │
│                                                               │
│  IMPLEMENTACIÓN:                                              │
│    PolynomialFeatures(degree=n) + LinearRegression()          │
│                                                               │
│  ELEGIR GRADO:                                                │
│    • Grado bajo → Underfitting (no captura tendencia)        │
│    • Grado correcto → Buen ajuste train Y test               │
│    • Grado alto → Overfitting (memoriza, no generaliza)      │
│                                                               │
│  MÉTODO DE SELECCIÓN:                                         │
│    Cross-validation: elegir grado con menor error en val     │
│                                                               │
│  CUÁNDO USAR:                                                 │
│    • Datos con tendencia curva evidente                       │
│    • Relación no lineal entre X e y                          │
│    • Cuando regresión lineal da error alto                   │
│                                                               │
│  PRECAUCIÓN:                                                  │
│    Más grado ≠ mejor modelo                                   │
│    Siempre validar con datos no vistos                       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Regularización (Ridge y Lasso) para evitar overfitting
