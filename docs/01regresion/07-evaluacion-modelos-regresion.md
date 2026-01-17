# Métricas de Evaluación para Regresión

## 1. ¿Por Qué Necesitamos Métricas?

### El Problema

```
Modelo A predice: [100, 200, 300]
Modelo B predice: [105, 195, 310]
Valores reales:   [100, 200, 300]

¿Cuál modelo es mejor?
  → Modelo A parece perfecto
  → Pero ¿cómo CUANTIFICAMOS la diferencia?

Necesitamos métricas numéricas para:
  • Comparar modelos objetivamente
  • Detectar overfitting
  • Comunicar rendimiento
  • Elegir el mejor modelo
```

### Métricas Principales

```
┌────────────────────────────────────────────────────────┐
│  MÉTRICAS DE REGRESIÓN                                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  • MAE  - Mean Absolute Error (Error Absoluto Medio)  │
│  • MSE  - Mean Squared Error (Error Cuadrático Medio) │
│  • RMSE - Root MSE (Raíz del MSE)                     │
│  • R²   - Coeficiente de Determinación                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 2. MAE (Mean Absolute Error)

### Fórmula

```
         1   m
MAE = ─────── Σ |ŷ⁽ⁱ⁾ - y⁽ⁱ⁾|
         m   i=1

Donde:
  m = número de ejemplos
  ŷ⁽ⁱ⁾ = predicción para ejemplo i
  y⁽ⁱ⁾ = valor real del ejemplo i
  |...| = valor absoluto
```

### Ejemplo Paso a Paso

```
Predicciones: ŷ = [100, 200, 350, 400]
Valores reales: y = [110, 190, 300, 420]

Errores absolutos:
  |100 - 110| = 10
  |200 - 190| = 10
  |350 - 300| = 50
  |400 - 420| = 20

MAE = (10 + 10 + 50 + 20) / 4 = 90 / 4 = 22.5

Interpretación:
  "En promedio, las predicciones se desvían 22.5 unidades
   del valor real"
```

### Características

```
┌────────────────────────────────────────────────────────┐
│  MAE - CARACTERÍSTICAS                                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ Mismas unidades que la variable objetivo           │
│  ✓ Fácil de interpretar                               │
│  ✓ Robusto a outliers (errores grandes no dominan)    │
│                                                        │
│  ✗ No penaliza errores grandes especialmente          │
│  ✗ No diferenciable en 0 (problema para optimización) │
│                                                        │
│  Rango: [0, ∞)                                        │
│  Mejor: Más cercano a 0                               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 3. MSE (Mean Squared Error)

### Fórmula

```
         1   m
MSE = ─────── Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
         m   i=1

Los errores se ELEVAN AL CUADRADO antes de promediar.
```

### Ejemplo Paso a Paso

```
Predicciones: ŷ = [100, 200, 350, 400]
Valores reales: y = [110, 190, 300, 420]

Errores al cuadrado:
  (100 - 110)² = (-10)² = 100
  (200 - 190)² = (10)²  = 100
  (350 - 300)² = (50)²  = 2500  ← ¡PENALIZADO!
  (400 - 420)² = (-20)² = 400

MSE = (100 + 100 + 2500 + 400) / 4 = 3100 / 4 = 775

Comparación con MAE:
  MAE = 22.5 (promedio de errores)
  MSE = 775  (dominado por el error de 50)
```

### Características

```
┌────────────────────────────────────────────────────────┐
│  MSE - CARACTERÍSTICAS                                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ Penaliza errores grandes (cuadrático)              │
│  ✓ Diferenciable en todo punto (bueno para GD)        │
│  ✓ Función de coste estándar en ML                    │
│                                                        │
│  ✗ Unidades al cuadrado (€², metros², etc.)           │
│  ✗ Difícil de interpretar directamente                │
│  ✗ Sensible a outliers                                │
│                                                        │
│  Rango: [0, ∞)                                        │
│  Mejor: Más cercano a 0                               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Visualización: Por qué Penaliza Errores Grandes

```
Error │
  ²   │
      │                    ●  Error = 50 → 2500
      │
      │
1000  │
      │
 500  │           ●  Error = 20 → 400
      │
 100  │  ●  ●        Error = 10 → 100
      │
      └────────────────────────────────── Error
         10    20    30    40    50

Un error de 50 contribuye 25 veces más que uno de 10
(no 5 veces como sería lineal)
```

## 4. RMSE (Root Mean Squared Error)

### Fórmula

```
              ┌─────────────────────────┐
RMSE = √MSE = │  1   m                  │
              │ ─── Σ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²   │
              │  m  i=1                 │
              └─────────────────────────┘

Simplemente la raíz cuadrada del MSE.
```

### Ejemplo

```
MSE = 775

RMSE = √775 = 27.84

Interpretación:
  "En promedio, las predicciones se desvían ~28 unidades
   del valor real, penalizando más los errores grandes"

Comparación:
  MAE  = 22.5  (promedio simple de errores)
  RMSE = 27.84 (promedio que penaliza errores grandes)

RMSE > MAE siempre (excepto si todos los errores son iguales)
```

### Características

```
┌────────────────────────────────────────────────────────┐
│  RMSE - CARACTERÍSTICAS                                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ Mismas unidades que la variable objetivo           │
│  ✓ Fácil de interpretar (como MAE)                    │
│  ✓ Penaliza errores grandes (como MSE)                │
│  ✓ Métrica muy usada en competiciones y papers        │
│                                                        │
│  ✗ Sensible a outliers (hereda de MSE)                │
│                                                        │
│  Rango: [0, ∞)                                        │
│  Mejor: Más cercano a 0                               │
│                                                        │
│  RMSE ≥ MAE siempre                                   │
│  Si RMSE >> MAE → Hay errores muy grandes (outliers)  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 5. R² (Coeficiente de Determinación)

### Concepto

**R²:** Proporción de la varianza explicada por el modelo.

```
         Varianza explicada por el modelo
R² = ─────────────────────────────────────────
            Varianza total de los datos

"¿Cuánto de la variabilidad de y es capturada
 por las predicciones?"
```

### Fórmula

```
              Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²       SS_res
R² = 1 - ──────────────────── = 1 - ────────
              Σ(y⁽ⁱ⁾ - ȳ)²          SS_tot

Donde:
  SS_res = Suma de cuadrados residuales (error del modelo)
  SS_tot = Suma de cuadrados totales (varianza de y)
  ȳ = media de y
```

### Ejemplo Paso a Paso

```
Valores reales: y = [100, 200, 300, 400]
Media: ȳ = 250

Predicciones: ŷ = [110, 190, 310, 390]

SS_tot (varianza total):
  (100-250)² + (200-250)² + (300-250)² + (400-250)²
  = 22500 + 2500 + 2500 + 22500
  = 50000

SS_res (error del modelo):
  (110-100)² + (190-200)² + (310-300)² + (390-400)²
  = 100 + 100 + 100 + 100
  = 400

R² = 1 - (400 / 50000) = 1 - 0.008 = 0.992

Interpretación:
  "El modelo explica el 99.2% de la variabilidad de y"
```

### Interpretación de Valores

```
┌─────────────┬─────────────────────────────────────────┐
│     R²      │           INTERPRETACIÓN                │
├─────────────┼─────────────────────────────────────────┤
│    1.0      │  Predicciones perfectas                 │
│   0.9+      │  Excelente ajuste                       │
│  0.7-0.9    │  Buen ajuste                            │
│  0.5-0.7    │  Ajuste moderado                        │
│  0.3-0.5    │  Ajuste débil                           │
│   <0.3      │  Modelo pobre                           │
│    0.0      │  Igual que predecir la media            │
│   <0.0      │  Peor que predecir la media (muy malo) │
└─────────────┴─────────────────────────────────────────┘
```

### Visualización

```
R² alto (0.95):
y │       ●
  │      ●     Puntos MUY cerca de la línea
  │     ●
  │    ●────────────
  │   ●
  │  ●
  └─────────────────── x


R² bajo (0.30):
y │  ●     ●
  │    ●      ●    Puntos DISPERSOS alrededor
  │  ●   ─────────────  de la línea
  │     ●    ●
  │  ●    ●
  └─────────────────── x
```

### Características

```
┌────────────────────────────────────────────────────────┐
│  R² - CARACTERÍSTICAS                                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ✓ Adimensional (no tiene unidades)                   │
│  ✓ Fácil de interpretar (% de varianza explicada)     │
│  ✓ Comparable entre datasets diferentes               │
│  ✓ Estándar en la industria                           │
│                                                        │
│  ✗ Puede ser negativo (modelo muy malo)               │
│  ✗ Aumenta al añadir features (incluso inútiles)      │
│  ✗ No indica si las predicciones son buenas en        │
│    términos absolutos                                  │
│                                                        │
│  Rango: (-∞, 1]                                       │
│  Mejor: Más cercano a 1                               │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 6. Comparación de Métricas

### Tabla Resumen

```
┌──────────┬────────────┬─────────────┬─────────────────────┐
│ MÉTRICA  │  UNIDADES  │   RANGO     │   USAR CUANDO       │
├──────────┼────────────┼─────────────┼─────────────────────┤
│   MAE    │  Original  │   [0, ∞)    │ Errores similares   │
│          │            │             │ importan igual      │
├──────────┼────────────┼─────────────┼─────────────────────┤
│   MSE    │  Cuadrado  │   [0, ∞)    │ Penalizar errores   │
│          │            │             │ grandes, optimizar  │
├──────────┼────────────┼─────────────┼─────────────────────┤
│   RMSE   │  Original  │   [0, ∞)    │ Interpretar MSE,    │
│          │            │             │ comparar modelos    │
├──────────┼────────────┼─────────────┼─────────────────────┤
│    R²    │ Adimensional│  (-∞, 1]   │ Proporción de       │
│          │            │             │ varianza explicada  │
└──────────┴────────────┴─────────────┴─────────────────────┘
```

### ¿Cuándo Usar Cada Una?

```
┌────────────────────────────────────────────────────────┐
│  GUÍA DE SELECCIÓN DE MÉTRICAS                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Predecir COSTE de incidente:                          │
│    → RMSE o MAE (en €)                                │
│    → "Error promedio de X €"                          │
│                                                        │
│  Comparar modelos en DIFERENTES datasets:              │
│    → R² (adimensional, comparable)                    │
│                                                        │
│  Datos con OUTLIERS:                                   │
│    → MAE (menos sensible)                             │
│                                                        │
│  OPTIMIZACIÓN (training):                              │
│    → MSE (diferenciable, estándar)                    │
│                                                        │
│  Reportar a STAKEHOLDERS:                              │
│    → RMSE + R² (interpretables)                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 7. Implementación en Python

### Código Completo

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# Valores reales y predicciones
y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 190, 310, 380, 520])

# Calcular métricas
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)  # o mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

# Mostrar resultados
print("="*50)
print("MÉTRICAS DE EVALUACIÓN")
print("="*50)
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print("="*50)
```

### Función de Evaluación Completa

```python
def evaluar_regresion(y_true, y_pred, nombre_modelo="Modelo"):
    """
    Evalúa un modelo de regresión con todas las métricas.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"EVALUACIÓN: {nombre_modelo}")
    print(f"{'='*50}")
    print(f"  MAE:  {mae:>10.2f}")
    print(f"  MSE:  {mse:>10.2f}")
    print(f"  RMSE: {rmse:>10.2f}")
    print(f"  R²:   {r2:>10.4f}")

    # Diagnóstico
    if r2 > 0.9:
        print("  → Excelente ajuste")
    elif r2 > 0.7:
        print("  → Buen ajuste")
    elif r2 > 0.5:
        print("  → Ajuste moderado")
    else:
        print("  → Ajuste pobre, considerar otro modelo")

    if rmse > mae * 1.5:
        print("  ⚠️ RMSE >> MAE: Posibles outliers o errores grandes")

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
```

## 8. Ejemplo Completo: Ciberseguridad

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simular datos: Predecir coste de incidente
np.random.seed(42)
n = 200

# Features
sistemas_afectados = np.random.uniform(10, 2000, n)
tiempo_deteccion = np.random.uniform(1, 72, n)  # horas
severidad = np.random.uniform(1, 10, n)

# Target: coste (relación no lineal)
coste = (
    5000 +  # Coste base
    15 * sistemas_afectados +
    500 * np.sqrt(sistemas_afectados) +  # No lineal
    1000 * severidad +
    50 * tiempo_deteccion +
    np.random.randn(n) * 5000  # Ruido
)

# DataFrame
df = pd.DataFrame({
    'sistemas': sistemas_afectados,
    'tiempo_deteccion': tiempo_deteccion,
    'severidad': severidad,
    'coste': coste
})

# Split
X = df[['sistemas', 'tiempo_deteccion', 'severidad']]
y = df['coste']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Comparar modelos
print("="*60)
print("COMPARACIÓN DE MODELOS - PREDICCIÓN DE COSTE DE INCIDENTE")
print("="*60)

modelos = {
    'Lineal': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ]),
    'Polinómico (grado 2)': Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=1.0))
    ])
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{nombre}:")
    print(f"  MAE:  {mae:,.0f} €")
    print(f"  RMSE: {rmse:,.0f} €")
    print(f"  R²:   {r2:.4f}")
```

### Output Esperado

```
============================================================
COMPARACIÓN DE MODELOS - PREDICCIÓN DE COSTE DE INCIDENTE
============================================================

Lineal:
  MAE:  4,523 €
  RMSE: 5,678 €
  R²:   0.9234

Polinómico (grado 2):
  MAE:  3,891 €      ← Mejor MAE
  RMSE: 4,923 €      ← Mejor RMSE
  R²:   0.9567       ← Mejor R²

Ridge:
  MAE:  4,512 €
  RMSE: 5,665 €
  R²:   0.9238

Conclusión: El modelo polinómico captura mejor la
relación no lineal entre sistemas afectados y coste.
```

## 9. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  MÉTRICAS DE REGRESIÓN - RESUMEN                              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  MAE (Mean Absolute Error):                                   │
│    • Promedio de |error|                                      │
│    • Unidades originales, robusto a outliers                  │
│                                                               │
│  MSE (Mean Squared Error):                                    │
│    • Promedio de error²                                       │
│    • Penaliza errores grandes, usado en optimización          │
│                                                               │
│  RMSE (Root MSE):                                             │
│    • √MSE                                                     │
│    • Unidades originales, penaliza errores grandes            │
│    • Métrica más reportada                                    │
│                                                               │
│  R² (Coeficiente de Determinación):                          │
│    • % de varianza explicada                                  │
│    • Adimensional, rango (-∞, 1]                             │
│    • R² > 0.9 = excelente, R² < 0.5 = pobre                  │
│                                                               │
│  REGLAS PRÁCTICAS:                                            │
│    • Reportar: RMSE + R² (interpretables)                     │
│    • Optimizar: MSE (diferenciable)                           │
│    • Outliers: preferir MAE                                   │
│    • Si RMSE >> MAE: revisar outliers                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Clasificación - Regresión Logística
