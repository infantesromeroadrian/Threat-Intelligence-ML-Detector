# Función de Coste y Proceso de Entrenamiento

## 1. El Problema: Encontrar los Parámetros Óptimos

### Situación Inicial

Tenemos datos de incidentes de seguridad:

```
Coste (€)
    │
25k │                    ●
    │
20k │           ●
    │      ●
15k │  ●              ●
    │                      ← DATOS DE ENTRENAMIENTO
10k │     ●                  (X₁, y) etiquetados
    │ ●
 5k │
    │
    └─────────────────────────────── Sistemas afectados (X₁)
    0   500  1000  1500  2000
```

### La Pregunta Clave

**¿Cómo encontramos θ₀ y θ₁ para que h_θ(x) = θ₀ + θ₁·x₁ se ajuste MEJOR a estos datos?**

```
Función hipótesis:
h_θ(x) = θ₀ + θ₁·x₁

Parámetros desconocidos:
  θ₀ = ???  →  ¿Qué valor ponemos?
  θ₁ = ???  →  ¿Qué valor ponemos?
```

### Posibles Funciones Hipótesis

Diferentes valores de θ₀ y θ₁ generan diferentes rectas:

```
Coste (€)
    │
25k │     h₃(x)  ●
    │       /
20k │      /  ●              h₁(x): θ₀=0,  θ₁=10  ✓ ÓPTIMA
    │     /  /               h₂(x): θ₀=0,  θ₁=5   ✗ Mala
15k │ ●  /  / h₁(x)          h₃(x): θ₀=0,  θ₁=15  ✗ Mala
    │   /  /
10k │  /● /
    │ / / h₂(x)
 5k │/ /  /
    │  /  
    └─────────────────────── Sistemas afectados
    0  500  1000  1500

¿Cuál es la MEJOR recta?
→ La que pasa más cerca de TODOS los puntos
```

## 2. Función Hipótesis Óptima vs Subóptima

### Caso 1: Función Bien Ajustada (ÓPTIMA)

```
Coste (€)
    │
    │              ●  ← Predicción = 20k (cercana al real)
20k │             /
    │            /●
    │           /     Nueva predicción: X₁ = 1400
15k │          /↑     h_θ(1400) = 0 + 10·1400 = 14,000€
    │      ●  /       Valor esperado ≈ 14,000€ ✓ CORRECTO
    │       /
10k │   ● /
    │    /
    │   /
    └──────────────────── Sistemas afectados
    
Parámetros: θ₀ = 0, θ₁ = 10
Esta función SE AJUSTA BIEN a la tendencia
```

### Caso 2: Función Mal Ajustada (SUBÓPTIMA)

```
Coste (€)
    │                    
25k │              ●
    │
20k │         ●  
    │      ●           Nueva predicción: X₁ = 1400
15k │  ●              h_θ(1400) = 0 + 5·1400 = 7,000€
    │                  Valor esperado ≈ 14,000€ ✗ ERROR GRANDE
10k │   ●    ┌──── Predicción = 7k (MUY LEJOS del real)
    │       /
 5k │  ●   /  
    │     /
    │ ●  /
    └────/──────────────── Sistemas afectados
        /
       / Recta mal ajustada (pendiente muy baja)

Parámetros: θ₀ = 0, θ₁ = 5
Esta función NO se ajusta bien a los datos
```

## 3. Concepto de Error

### Definición de Error

**Error = Diferencia entre predicción y valor real**

```
Para un ejemplo de entrenamiento (X^(i), y^(i)):

Error = h_θ(X^(i)) - y^(i)
         ↑           ↑
    Predicción   Valor real
```

### Visualización del Error

```
Coste (€)
    │
    │                    ●  ← y_real = 20,000€
20k │                   │
    │                   │ ERROR
    │                   │ = 20k - 7k
15k │                   │ = 13,000€
    │                   │
    │                   │
10k │                   ▼
    │                   ●  ← h_θ(x) = 7,000€ (predicción)
    │                  /
 5k │                 /
    │                /
    └───────────────/────── Sistemas afectados
                   X₁ = 1400

Distancia vertical = ERROR para este ejemplo
```

### Múltiples Errores

```
Coste (€)
    │
    │     │         │  ●  Cada línea vertical
25k │     │    ●    │     es el ERROR para
    │     │   │     │     un ejemplo de
20k │  ●  │   │     │     entrenamiento
    │  │  │   │     │
15k │  │  ●   │     ●     Queremos minimizar
    │  │ /│   │    /│     TODOS estos errores
10k │  ●/ │   ●   / │     simultáneamente
    │  /  │  /│  /  │
 5k │ /   │ / │ /   │
    │/    ●/  │/    │
    └─────/───/─────/──── Sistemas afectados
         /   /     /
    Recta mal ajustada → ERRORES GRANDES
```

## 4. La Función de Coste (Cost Function)

### Definición

La **función de coste J(θ₀, θ₁)** calcula el **error total** del modelo respecto a los datos de entrenamiento.

```
J(θ₀, θ₁) = Mide qué tan MAL está ajustado el modelo

• Entrada: Parámetros θ₀, θ₁
• Salida: Un número que representa el error total
• Objetivo: MINIMIZAR este número
```

### Fórmula (Error Cuadrático Medio - MSE)

```
         1   m
J(θ) = ─── · Σ [h_θ(X^(i)) - y^(i)]²
        2m  i=1

Donde:
  m  = número de ejemplos de entrenamiento
  X^(i) = features del ejemplo i
  y^(i) = valor real del ejemplo i
  h_θ(X^(i)) = predicción para el ejemplo i
```

### Desglose de la Fórmula

```
┌───────────────────────────────────────────────────────┐
│  J(θ) = (1/2m) · Σ [h_θ(X^(i)) - y^(i)]²             │
│          ↑       ↑   ↑─────────────────↑              │
│          │       │          └─ Error individual       │
│          │       └─ Suma sobre TODOS los ejemplos     │
│          └─ Factor de normalización                   │
└───────────────────────────────────────────────────────┘

Paso a paso:
1. Para cada ejemplo: calcular error = predicción - real
2. Elevar al cuadrado (para penalizar errores grandes)
3. Sumar todos los errores cuadráticos
4. Dividir por 2m (promedio normalizado)
```

### ¿Por qué Elevar al Cuadrado?

```
Razones para usar (error)²:

1. SIEMPRE POSITIVO:
   Error = -10  →  Error² = 100
   Error = +10  →  Error² = 100
   (No se cancelan errores positivos y negativos)

2. PENALIZA ERRORES GRANDES:
   Error = 2    →  Error² = 4
   Error = 10   →  Error² = 100  (25 veces peor, no 5)

3. DIFERENCIABLE:
   Permite usar cálculo para encontrar el mínimo
```

### Ejemplo Numérico Paso a Paso

```python
# Datos de entrenamiento (m = 4 ejemplos)
X = [1000, 1500, 800, 1800]  # Sistemas afectados
y = [15000, 20000, 10000, 25000]  # Coste real

# CASO A: Parámetros aleatorios MALOS
# h_θ(x) = 2000 + 5·x  (θ₀=2000, θ₁=5)

┌──────┬─────────┬──────────┬─────────────┬──────────┬──────────────┐
│  i   │  x^(i)  │  y^(i)   │  h_θ(x^(i)) │  Error   │   Error²     │
├──────┼─────────┼──────────┼─────────────┼──────────┼──────────────┤
│  1   │  1000   │  15,000  │   7,000     │  -8,000  │  64,000,000  │
│  2   │  1500   │  20,000  │   9,500     │ -10,500  │ 110,250,000  │
│  3   │   800   │  10,000  │   6,000     │  -4,000  │  16,000,000  │
│  4   │  1800   │  25,000  │  11,000     │ -14,000  │ 196,000,000  │
└──────┴─────────┴──────────┴─────────────┴──────────┴──────────────┘
                                                    Σ = 386,250,000

Cálculo de predicciones:
  h_θ(1000) = 2000 + 5·1000 = 7,000
  h_θ(1500) = 2000 + 5·1500 = 9,500
  h_θ(800)  = 2000 + 5·800  = 6,000
  h_θ(1800) = 2000 + 5·1800 = 11,000

J(θ₀=2000, θ₁=5) = 386,250,000 / (2·4) = 48,281,250  →  ERROR ALTO ✗


# CASO B: Parámetros ÓPTIMOS
# h_θ(x) = 0 + 12.5·x  (θ₀=0, θ₁=12.5)

┌──────┬─────────┬──────────┬─────────────┬──────────┬──────────┐
│  i   │  x^(i)  │  y^(i)   │  h_θ(x^(i)) │  Error   │  Error²  │
├──────┼─────────┼──────────┼─────────────┼──────────┼──────────┤
│  1   │  1000   │  15,000  │  12,500     │  -2,500  │ 6,250,000│
│  2   │  1500   │  20,000  │  18,750     │  -1,250  │ 1,562,500│
│  3   │   800   │  10,000  │  10,000     │       0  │         0│
│  4   │  1800   │  25,000  │  22,500     │  -2,500  │ 6,250,000│
└──────┴─────────┴──────────┴─────────────┴──────────┴──────────┘
                                                   Σ = 14,062,500

J(θ₀=0, θ₁=12.5) = 14,062,500 / (2·4) = 1,757,812  →  ERROR MENOR ✓


# Conclusión: Parámetros óptimos reducen J drásticamente
```

## 5. El Proceso de Optimización

### Objetivo

```
MINIMIZAR J(θ₀, θ₁)

Encontrar los valores de θ₀ y θ₁ que hagan que
J(θ₀, θ₁) sea lo más pequeño posible
```

### Visualización del Espacio de Parámetros

```
J(θ₁)  (simplificado: solo variamos θ₁, θ₀ fijo)
    │
    │      ╱╲
    │     ╱  ╲
15M │    ╱    ╲         Queremos llegar
    │   ╱      ╲        al MÍNIMO
10M │  ╱        ╲
    │ ╱          ╲
 5M │╱            ╲
    │              ╲___
    │                  ╲___
  0 │─────────────────────╲___________
    0   5   10   15   20   θ₁
              ↑
         θ₁ óptimo ≈ 10
         (mínimo de J)
```

### Función de Optimización

La **función de optimización** ajusta iterativamente los parámetros:

```
┌─────────────────────────────────────┐
│  ALGORITMO DE OPTIMIZACIÓN          │
│  (ej: Gradient Descent)             │
├─────────────────────────────────────┤
│                                     │
│  Repetir hasta convergencia:        │
│                                     │
│  1. Calcular J(θ₀, θ₁)              │
│  2. Calcular cómo cambiar θ₀, θ₁    │
│     para reducir J                  │
│  3. Actualizar θ₀ y θ₁              │
│  4. Repetir                         │
│                                     │
└─────────────────────────────────────┘
```

### Proceso Iterativo

```
Iteración 0:  θ₁ = 3   →  J = 20,000,000  (error alto)
              │
              ▼ (ajustar θ₁)
Iteración 1:  θ₁ = 5   →  J = 14,583,333  (mejor)
              │
              ▼ (ajustar θ₁)
Iteración 2:  θ₁ = 7   →  J = 5,833,333   (mejor)
              │
              ▼ (ajustar θ₁)
Iteración 3:  θ₁ = 9   →  J = 583,333     (casi óptimo)
              │
              ▼ (ajustar θ₁)
Iteración 4:  θ₁ = 10  →  J = 0           (ÓPTIMO ✓)
              │
              ▼
         CONVERGENCIA (no mejora más)
```

## 6. Algoritmo Completo de Entrenamiento

### Diagrama de Flujo

```
┌──────────────────────────────┐
│  1. INICIALIZAR PARÁMETROS   │
│     θ₀ = random, θ₁ = random │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  2. CONSTRUIR h_θ(x)         │
│     h_θ(x) = θ₀ + θ₁·x       │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  3. CALCULAR FUNCIÓN COSTE   │
│     J(θ) = (1/2m)·Σ[error²]  │
│     Mide el error total      │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  4. OPTIMIZAR (Minimizar J)  │
│     Ajustar θ₀ y θ₁ para     │
│     reducir el error         │
└────────────┬─────────────────┘
             │
             ▼
         ┌───┴────┐
         │ ¿J ya  │  NO ──┐
         │ mínimo?│       │
         └───┬────┘       │
             │ SÍ         │
             ▼            │
┌──────────────────────┐  │
│  5. MODELO ENTRENADO │  │
│     θ₀ = óptimo      │  │
│     θ₁ = óptimo      │  │
└──────────────────────┘  │
             │            │
             ▼            │
┌──────────────────────┐  │
│  6. HACER            │  │
│     PREDICCIONES     │  │
└──────────────────────┘  │
                          │
      ┌───────────────────┘
      │ (volver al paso 4)
      ▼
```

### Pseudocódigo

```python
# ENTRENAMIENTO DE REGRESIÓN LINEAL

# Paso 1: Inicializar parámetros
θ₀ = random_value()
θ₁ = random_value()

# Paso 2: Bucle de optimización
while not converged:
    
    # Calcular función de coste
    J = 0
    for each (X^(i), y^(i)) in training_data:
        prediction = θ₀ + θ₁ · X^(i)
        error = prediction - y^(i)
        J += error²
    J = J / (2 * m)
    
    # Optimizar (actualizar parámetros)
    θ₀, θ₁ = optimize(θ₀, θ₁, J)
    
    # Verificar convergencia
    if J no mejora significativamente:
        converged = True

# Paso 3: Modelo entrenado
model = h_θ(x) where θ₀, θ₁ son óptimos
```

## 7. Por qué Necesitamos Datos Etiquetados

### Aprendizaje Supervisado

```
Para calcular el ERROR necesitamos conocer el VALOR REAL:

Error = h_θ(X^(i)) - y^(i)
        ↑            ↑
   Predicción    VALOR REAL
                (etiqueta)

Sin etiquetas → No podemos calcular J(θ)
              → No podemos entrenar
```

### Comparación

```
┌────────────────────────────────────────────────────┐
│  CON ETIQUETAS (Supervisado)                       │
├────────────────────────────────────────────────────┤
│  X₁        y_real     →  Podemos calcular error    │
│  1000      10,000€                                 │
│  1500      15,000€                                 │
│  ↑         ↑                                       │
│  features  labels                                  │
└────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────┐
│  SIN ETIQUETAS (No supervisado)                    │
├────────────────────────────────────────────────────┤
│  X₁         ???      →  NO podemos calcular error  │
│  1000       ???                                    │
│  1500       ???                                    │
│  ↑                                                 │
│  features (sin labels)                             │
└────────────────────────────────────────────────────┘
```

## 8. Resumen

```
┌─────────────────────────────────────────────────────────┐
│  PROCESO DE ENTRENAMIENTO - REGRESIÓN LINEAL           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. FUNCIÓN HIPÓTESIS:                                  │
│     h_θ(x) = θ₀ + θ₁·x₁ + ... + θₙ·xₙ                  │
│                                                         │
│  2. FUNCIÓN DE COSTE (MSE):                             │
│     J(θ) = (1/2m) · Σ[h_θ(X^(i)) - y^(i)]²            │
│     • Mide el error total del modelo                    │
│     • Siempre ≥ 0                                       │
│     • Menor valor = mejor ajuste                        │
│                                                         │
│  3. OBJETIVO:                                           │
│     Minimizar J(θ) ajustando θ₀, θ₁, ..., θₙ           │
│                                                         │
│  4. OPTIMIZACIÓN:                                       │
│     Algoritmo iterativo (ej: Gradient Descent)          │
│     que modifica los parámetros para reducir J          │
│                                                         │
│  5. CONVERGENCIA:                                       │
│     Cuando J no puede disminuir más                     │
│     → Modelo entrenado ✓                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Conceptos clave:**
- **Error:** Diferencia entre predicción y valor real
- **Función de coste J(θ):** Suma de todos los errores al cuadrado
- **Optimización:** Proceso de ajustar θ para minimizar J
- **Entrenamiento:** Todo el proceso iterativo hasta convergencia
- **Supervisado:** Requiere datos etiquetados para calcular errores

---

**Siguiente:** Gradient Descent - El algoritmo de optimización en detalle