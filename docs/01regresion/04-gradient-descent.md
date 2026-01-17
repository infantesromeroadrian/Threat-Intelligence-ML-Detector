# Gradient Descent: Algoritmo de Optimización

## 1. Introducción a la Optimización

### ¿Qué es una Función de Optimización?

Al igual que las funciones de coste, podemos usar **diferentes algoritmos de optimización** con el mismo algoritmo de ML.

```
┌────────────────────────────────────────────┐
│  ALGORITMOS DE OPTIMIZACIÓN                │
├────────────────────────────────────────────┤
│  • Gradient Descent ← El más común         │
│  • Stochastic Gradient Descent (SGD)       │
│  • Mini-batch Gradient Descent             │
│  • Adam                                    │
│  • RMSprop                                 │
│  • Adagrad                                 │
└────────────────────────────────────────────┘

Para Regresión Lineal usaremos: Gradient Descent
```

### Relación con la Función de Coste

La optimización está **íntimamente ligada** a la función de coste:

```
┌─────────────────────────────────────────────────┐
│  FUNCIÓN DE COSTE (MSE)                         │
│         1    m                                  │
│  J(θ) = ──── · Σ [h_θ(x^(i)) - y^(i)]²        │
│         2m   i=1                                │
├─────────────────────────────────────────────────┤
│  OBJETIVO DE LA OPTIMIZACIÓN:                   │
│  Minimizar J(θ) ajustando θ₀, θ₁, ..., θₙ      │
└─────────────────────────────────────────────────┘
```

## 2. Función Convexa y Mínimo Global

### Representación Gráfica de J(θ)

Simplificamos: **h_θ(x) = θ₁·x** (sin θ₀ para visualizar en 2D)

```
J(θ₁)
    │
    │      ╱╲
    │     ╱  ╲
15M │    ╱    ╲           Función CONVEXA
    │   ╱      ╲          Característica: tiene un
10M │  ╱        ╲         único MÍNIMO GLOBAL
    │ ╱          ╲
 5M │╱            ╲
    │              ╲___
    │                  ╲____
  0 │──────────────────────╲___________ θ₁
    0   5   10   15   20   25
                ↑
           MÍNIMO GLOBAL
           (θ₁ óptimo)
```

### ¿Qué es el Mínimo Global?

```
Mínimo Global = Valor de θ₁ donde J(θ₁) es MÍNIMO

Ejemplo:
  Si θ₁ = 10  →  J(10) = 100     ✗ (malo)
  Si θ₁ = 20  →  J(20) = 0       ✓ (óptimo!)
  Si θ₁ = 25  →  J(25) = 500     ✗ (malo)

Objetivo: Encontrar θ₁ = 20 de forma AUTOMÁTICA
```

## 3. Ejemplo Simplificado: J(x) = (x - 3)²

### Por qué Usar una Función Simplificada

Para entender Gradient Descent, usamos una función **más simple** que MSE:

```
J(x) = (x - 3)²

Similitud con MSE:
  MSE:  [h_θ(x^(i)) - y^(i)]²  ← Diferencia al cuadrado
  J(x): (x - 3)²               ← Diferencia al cuadrado

En este ejemplo:
  x  ≡  θ (parámetro a optimizar)
```

### Gráfica de J(x)

```
J(x)
    │
 50 │     ╱           ╲
    │    ╱             ╲
 40 │   ╱               ╲
    │  ╱                 ╲
 30 │ ╱                   ╲
    │╱                     ╲
 20 │                       ╲
    │╱                       ╲
 10 │                         ╲
    │                          ╲
  0 │───────────┼───────────────── x
    0   1   2   3   4   5   6
                ↑
           x = 3 (mínimo)

Cálculo mental del mínimo:
  J(3) = (3 - 3)² = 0² = 0  ← Valor mínimo
```

## 4. Inicialización del Algoritmo

### Paso 1: Inicializar x Aleatoriamente

```
Inicio: x = 10 (aleatorio)

J(10) = (10 - 3)² = 7² = 49  ← ERROR ALTO

Posición en gráfica:
J(x)
    │
 50 │ ●  ← x=10, J(10)=49 (muy lejos del mínimo)
    │  ╲
 40 │   ╲
    │    ╲
 30 │     ╲
    │      ╲
  0 │───────────┼───────────── x
    0   5      10
            ↑
         Inicio aquí
```

## 5. Concepto de Gradiente (Derivada)

### ¿Qué es el Gradiente?

En funciones de una variable, **gradiente = derivada = pendiente**

```
J(x) = (x - 3)²

Derivada:
  dJ/dx = 2·(x - 3)

Interpretación:
  La derivada nos dice en qué DIRECCIÓN y
  con qué INTENSIDAD crece la función
```

### Cálculo de la Derivada en Diferentes Puntos

```
┌──────┬────────────────┬───────────┬────────────────┐
│  x   │  Derivada      │  Valor    │  Pendiente     │
├──────┼────────────────┼───────────┼────────────────┤
│  10  │  2·(10-3)      │    14     │  Muy alta ↗    │
│   5  │  2·(5-3)       │     4     │  Moderada ↗    │
│   4  │  2·(4-3)       │     2     │  Baja ↗        │
│   3  │  2·(3-3)       │     0     │  PLANA → (min) │
│   2  │  2·(2-3)       │    -2     │  Baja ↘        │
│   1  │  2·(1-3)       │    -4     │  Moderada ↘    │
└──────┴────────────────┴───────────┴────────────────┘

Observación clave:
  • Lejos del mínimo → Derivada GRANDE
  • Cerca del mínimo → Derivada PEQUEÑA
  • En el mínimo    → Derivada = 0
```

### Visualización de Pendientes

```
J(x)
    │
    │    ╱│╲               Pendiente en x=10
 50 │   ╱ │ ╲──────────   dJ/dx = 14 (alta)
    │  ╱  │   ╲
 40 │ ╱   │    ╲
    │╱    │     ╲
 30 │     │      ╲─────   Pendiente en x=5
    │     │       ╲       dJ/dx = 4 (menor)
 20 │     │        ╲
    │     │         ╲
 10 │     │          ╲
    │     │           ╲
  0 │─────┼────────────╲── x
    0     3     7      10
          ↑
    Pendiente = 0
    (línea horizontal)
```

## 6. Fórmula de Gradient Descent

### Regla de Actualización (sin learning rate)

Idea inicial:
```
x_nuevo = x_actual - dJ/dx

Razonamiento:
  • Si dJ/dx > 0 (pendiente positiva) → restamos → x disminuye
  • Si dJ/dx < 0 (pendiente negativa) → restamos negativo → x aumenta
  • Siempre nos movemos HACIA EL MÍNIMO
```

### Fórmula Completa con Learning Rate

```
┌──────────────────────────────────────────────┐
│  GRADIENT DESCENT                            │
├──────────────────────────────────────────────┤
│  Repetir hasta convergencia:                 │
│                                              │
│    x := x - α · dJ/dx                        │
│         ↑   ↑    ↑                           │
│         │   │    └─ Gradiente (derivada)     │
│         │   └────── Learning rate            │
│         └────────── Valor actual             │
└──────────────────────────────────────────────┘

Donde:
  α (alpha) = Learning rate (típicamente 0.01, 0.1, 0.5)
              Controla el TAMAÑO DEL PASO
```

### ¿Qué es el Learning Rate?

```
Learning Rate (α) = Velocidad de aprendizaje

┌────────────────────────────────────────────────┐
│  α MUY PEQUEÑO (0.001)                         │
│    • Pasos muy pequeños                        │
│    • Convergencia lenta pero segura            │
│    • Muchas iteraciones                        │
├────────────────────────────────────────────────┤
│  α MODERADO (0.1)                              │
│    • Balance entre velocidad y estabilidad     │
│    • Recomendado para empezar                  │
├────────────────────────────────────────────────┤
│  α MUY GRANDE (0.9)                            │
│    • Pasos muy grandes                         │
│    • Puede PASARSE del mínimo                  │
│    • Riesgo de divergencia                     │
└────────────────────────────────────────────────┘
```

## 7. Ejemplo Paso a Paso

### Aplicando Gradient Descent a J(x) = (x - 3)²

**Setup:**
```python
J(x) = (x - 3)²
dJ/dx = 2·(x - 3)
α = 0.5  # Learning rate
x_inicial = 10
```

### Iteración 1

```
x_actual = 10

dJ/dx = 2·(10 - 3) = 2·7 = 14

x_nuevo = x_actual - α · dJ/dx
        = 10 - 0.5 · 14
        = 10 - 7
        = 3  ✓

¡EN UNA SOLA ITERACIÓN LLEGAMOS AL MÍNIMO!
```

### Iteración 2 (verificación)

```
x_actual = 3

dJ/dx = 2·(3 - 3) = 2·0 = 0

x_nuevo = 3 - 0.5 · 0
        = 3 - 0
        = 3  ✓

La derivada es 0 → x no cambia → CONVERGENCIA
```

### Visualización del Proceso

```
J(x)
    │
 50 │ ●────────────┐  Inicio: x=10
    │  ╲           │
 40 │   ╲          │
    │    ╲         │  UN GRAN SALTO
 30 │     ╲        │  (α=0.5, dJ/dx=14)
    │      ╲       │
 20 │       ╲      │
    │        ╲     │
 10 │         ╲    │
    │          ╲   ↓
  0 │───────────●───── x  Llegada: x=3 (mínimo)
    0   5      10
```

## 8. Aplicación a Regresión Lineal

### Fórmula MSE Original

```
         1    m
J(θ) = ──── · Σ [h_θ(x^(i)) - y^(i)]²
        2m   i=1

Donde:
  h_θ(x) = θ₀ + θ₁·x₁ + ... + θₙ·xₙ
```

### Gradient Descent para θ₀ y θ₁

```
┌──────────────────────────────────────────────────┐
│  GRADIENT DESCENT - REGRESIÓN LINEAL            │
├──────────────────────────────────────────────────┤
│  Repetir hasta convergencia:                     │
│                                                  │
│    θ₀ := θ₀ - α · ∂J(θ)/∂θ₀                     │
│                                                  │
│    θ₁ := θ₁ - α · ∂J(θ)/∂θ₁                     │
│                                                  │
│  Actualizar SIMULTÁNEAMENTE todos los θᵢ         │
└──────────────────────────────────────────────────┘
```

### Derivadas Parciales del MSE

```
∂J(θ)/∂θ₀ = (1/m) · Σ [h_θ(x^(i)) - y^(i)]

∂J(θ)/∂θ₁ = (1/m) · Σ [h_θ(x^(i)) - y^(i)] · x₁^(i)

Nota: El factor 2 del 2m se cancela con el 2 de la derivada
```

### Algoritmo Completo

```python
# Pseudocódigo Gradient Descent para Regresión Lineal

# 1. Inicializar parámetros
θ₀ = random()
θ₁ = random()
α = 0.01  # Learning rate
max_iter = 1000
tol = 1e-6  # Tolerancia para convergencia

# 2. Bucle de optimización
for i in range(max_iter):
    # Calcular predicciones
    h = θ₀ + θ₁ * X
    
    # Calcular errores
    errors = h - y
    
    # Calcular gradientes
    grad_θ₀ = (1/m) * sum(errors)
    grad_θ₁ = (1/m) * sum(errors * X)
    
    # Actualizar parámetros SIMULTÁNEAMENTE
    θ₀_nuevo = θ₀ - α * grad_θ₀
    θ₁_nuevo = θ₁ - α * grad_θ₁
    
    # Verificar convergencia
    if abs(θ₀_nuevo - θ₀) < tol and abs(θ₁_nuevo - θ₁) < tol:
        print(f"Convergencia en iteración {i}")
        break
    
    θ₀ = θ₀_nuevo
    θ₁ = θ₁_nuevo

# 3. Parámetros óptimos encontrados
print(f"θ₀ óptimo: {θ₀}")
print(f"θ₁ óptimo: {θ₁}")
```

## 9. Proceso Iterativo Completo

### Ejemplo Numérico Simplificado

```
Datos: [(1, 3), (2, 5), (3, 7)]  # (x, y)
Modelo: h_θ(x) = θ₀ + θ₁·x
Objetivo: y = 1 + 2x (θ₀=1, θ₁=2)

Inicio: θ₀=0, θ₁=0, α=0.1

┌──────┬────────┬────────┬───────────┬───────────┬──────────┐
│ Iter │   θ₀   │   θ₁   │   J(θ)    │ ∂J/∂θ₀   │ ∂J/∂θ₁   │
├──────┼────────┼────────┼───────────┼───────────┼──────────┤
│  0   │  0.00  │  0.00  │  28.00    │  -5.00    │ -10.00   │
│  1   │  0.50  │  1.00  │  10.50    │  -3.00    │  -6.00   │
│  2   │  0.80  │  1.60  │   3.94    │  -1.80    │  -3.60   │
│  3   │  0.98  │  1.96  │   1.48    │  -1.08    │  -2.16   │
│  4   │  1.09  │  2.18  │   0.55    │  -0.65    │  -1.30   │
│  5   │  1.15  │  2.31  │   0.21    │  -0.39    │  -0.78   │
│ ...  │  ...   │  ...   │   ...     │   ...     │   ...    │
│ 50   │  1.00  │  2.00  │   0.00    │   0.00    │   0.00   │
└──────┴────────┴────────┴───────────┴───────────┴──────────┘

Convergencia: θ₀ ≈ 1.00, θ₁ ≈ 2.00 ✓
```

## 10. Condiciones de Parada (Convergencia)

### ¿Cuándo Detenemos el Algoritmo?

```
┌────────────────────────────────────────────────┐
│  CRITERIOS DE CONVERGENCIA                     │
├────────────────────────────────────────────────┤
│  1. Cambio en parámetros < tolerancia          │
│     |θ_nuevo - θ_actual| < 1e-6               │
│                                                │
│  2. Cambio en J(θ) < tolerancia                │
│     |J_nuevo - J_actual| < 1e-6               │
│                                                │
│  3. Gradiente ≈ 0                              │
│     |∂J/∂θ| < 1e-6                            │
│                                                │
│  4. Máximo de iteraciones alcanzado            │
│     iter >= max_iter (ej: 1000)               │
└────────────────────────────────────────────────┘
```

## 11. Problemas Comunes

### Learning Rate Muy Grande

```
α = 0.99 (demasiado grande)

J(θ)
    │     ╱╲
    │    ╱  ╲
    │●──╱────╲──●     Se pasa el mínimo
    │  ╱      ╲       Oscila sin converger
    │ ╱        ╲●
    │╱          ╲
    └─────┼──────── θ
          ↑
      Se pierde
```

### Learning Rate Muy Pequeño

```
α = 0.0001 (demasiado pequeño)

J(θ)
    │
    │●   Pasos microscópicos
    │ ●  Convergencia muy lenta
    │  ● Miles de iteraciones
    │   ●
    │    ●
    └─────┼──────── θ
```

### Solución: Learning Rate Adaptativo

```
Estrategias:
  • Empezar con α = 0.1
  • Reducir α si J(θ) aumenta
  • Probar varios valores: [0.001, 0.01, 0.1, 1.0]
  • Usar algoritmos adaptativos (Adam, RMSprop)
```

## 12. Resumen

```
┌──────────────────────────────────────────────────────┐
│  GRADIENT DESCENT                                    │
├──────────────────────────────────────────────────────┤
│  CONCEPTO:                                           │
│  Algoritmo iterativo para minimizar J(θ)             │
│                                                      │
│  FÓRMULA:                                            │
│    θ := θ - α · ∂J(θ)/∂θ                            │
│                                                      │
│  COMPONENTES:                                        │
│  • Gradiente (∂J/∂θ): Dirección de máximo aumento   │
│  • Learning rate (α): Tamaño del paso               │
│  • Negativo (-): Ir en dirección OPUESTA (descenso) │
│                                                      │
│  PROCESO:                                            │
│  1. Inicializar θ aleatoriamente                     │
│  2. Calcular gradiente ∂J/∂θ                        │
│  3. Actualizar θ := θ - α·gradiente                  │
│  4. Repetir hasta convergencia                       │
│                                                      │
│  CONVERGENCIA:                                       │
│  Cuando θ no cambia significativamente               │
│  (gradiente ≈ 0 en el mínimo)                        │
└──────────────────────────────────────────────────────┘
```

**Conceptos clave:**
- **Gradiente**: Dirección de máximo crecimiento (derivada)
- **Descenso**: Ir en dirección opuesta al gradiente
- **Learning rate**: Controla velocidad de convergencia
- **Iterativo**: Mejora gradual hasta encontrar el mínimo
- **Convergencia**: Cuando los parámetros dejan de cambiar

---

**Siguiente:** Implementación práctica en Python con scikit-learn y desde cero
