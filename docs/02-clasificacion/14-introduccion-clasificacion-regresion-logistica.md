# Introducción a Clasificación: Regresión Logística

## 1. De Regresión a Clasificación

### Recordatorio: Regresión vs Clasificación

```
                    APRENDIZAJE SUPERVISADO
                            │
            ┌───────────────┴───────────────┐
            │                               │
       REGRESIÓN                      CLASIFICACIÓN
            │                               │
    Predecir valores               Predecir categorías
      CONTINUOS                       DISCRETAS
            │                               │
    ┌───────┴───────┐             ┌────────┴────────┐
    │               │             │                 │
  Precio        Temperatura     Spam/Ham        Fraude/Legítimo
  Coste         Edad            Phishing/Safe   Malware/Benigno
  Salario       Tiempo          Gato/Perro      Ataque/Normal
```

### Ejemplos en Ciberseguridad

| Tipo | Problema | Output |
|------|----------|--------|
| **Regresión** | Predecir coste de un incidente | 15,000€, 23,456€, etc. |
| **Clasificación** | ¿Es este email SPAM? | Sí / No |
| **Clasificación** | ¿Es este sitio phishing? | Phishing / Legítimo |
| **Clasificación** | ¿Qué familia de malware es? | Trojan / Worm / Ransomware |

## 2. El Problema de Clasificación Binaria

### Definición

**Clasificación binaria:** Predecir una de **dos clases posibles**.

```
Ejemplos de clasificación binaria:

┌─────────────────────┬────────────────────┐
│     CLASE 0         │      CLASE 1       │
│   (Negativa)        │    (Positiva)      │
├─────────────────────┼────────────────────┤
│      Ham            │       Spam         │
│    Legítimo         │     Phishing       │
│     Normal          │      Ataque        │
│     Benigno         │      Malware       │
│   No Fraude         │      Fraude        │
└─────────────────────┴────────────────────┘

Output: y ∈ {0, 1}
```

### ¿Por qué NO usar Regresión Lineal?

Intentar usar regresión lineal para clasificación tiene problemas:

```
Predicción con Regresión Lineal:

y (predicción)
    │
2.0 │           ●           ← Predicción > 1 ¿Qué significa?
    │          /
1.5 │         /
    │        /
1.0 │───────/───────────    ← Umbral de decisión
    │      /
0.5 │     /
    │    /●
0.0 │───●───────────────    ← Umbral inferior
    │  /
-0.5│ /                     ← Predicción < 0 ¿Qué significa?
    └─────────────────────── x (features)

Problemas:
  • Predicciones fuera del rango [0, 1]
  • No tiene interpretación probabilística
  • Sensible a outliers que "estiran" la línea
```

## 3. Regresión Logística: La Solución

### ¿Qué es la Regresión Logística?

A pesar de su nombre, **Regresión Logística es un algoritmo de CLASIFICACIÓN**.

```
┌────────────────────────────────────────────────────────┐
│  REGRESIÓN LOGÍSTICA                                   │
├────────────────────────────────────────────────────────┤
│  • Tipo: Clasificación (binaria o multiclase)          │
│  • Output: Probabilidad entre 0 y 1                    │
│  • Decisión: Si P > 0.5 → Clase 1, sino → Clase 0     │
│  • Función clave: Sigmoid (logística)                  │
└────────────────────────────────────────────────────────┘

¿Por qué se llama "Regresión"?
  → Porque internamente calcula una regresión lineal
  → Luego transforma el resultado con la función sigmoid
```

### Comparación con Regresión Lineal

```
REGRESIÓN LINEAL:
  h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ

  Output: Cualquier número real (-∞, +∞)


REGRESIÓN LOGÍSTICA:
  z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ    ← Parte lineal
  h(x) = sigmoid(z) = 1 / (1 + e^(-z))  ← Transformación

  Output: Probabilidad entre 0 y 1
```

## 4. La Función Sigmoid (Logística)

### Definición Matemática

```
                    1
σ(z) = ─────────────────
        1 + e^(-z)

Donde:
  z = θ₀ + θ₁x₁ + ... + θₙxₙ  (combinación lineal)
  e = número de Euler (≈ 2.718)
```

### Propiedades Clave

```
┌────────────────────────────────────────────────────────┐
│  PROPIEDADES DE SIGMOID                                │
├────────────────────────────────────────────────────────┤
│  • Rango: (0, 1) - siempre entre 0 y 1                │
│  • σ(0) = 0.5 - punto medio                           │
│  • Si z → +∞, σ(z) → 1                                │
│  • Si z → -∞, σ(z) → 0                                │
│  • Simétrica: σ(-z) = 1 - σ(z)                        │
│  • Diferenciable en todo punto                         │
└────────────────────────────────────────────────────────┘
```

### Visualización

```
σ(z)
    │
1.0 │                          ●────────────
    │                        ●
    │                      ●
0.8 │                    ●
    │                  ●
    │                ●
0.6 │              ●
    │            ●
0.5 │──────────●────────────────────────────  ← Umbral
    │        ●
0.4 │      ●
    │    ●
0.2 │  ●
    │●
0.0 │────────────●
    └───────────────────────────────────────── z
      -6  -4  -2   0   2   4   6

Interpretación:
  z < 0  →  σ(z) < 0.5  →  Clase 0 (más probable)
  z > 0  →  σ(z) > 0.5  →  Clase 1 (más probable)
  z = 0  →  σ(z) = 0.5  →  Incertidumbre total
```

### Tabla de Valores

```
┌─────────┬──────────┬─────────────────────────┐
│    z    │  σ(z)    │     Interpretación      │
├─────────┼──────────┼─────────────────────────┤
│   -6    │  0.002   │  99.8% clase 0          │
│   -4    │  0.018   │  98.2% clase 0          │
│   -2    │  0.119   │  88.1% clase 0          │
│   -1    │  0.269   │  73.1% clase 0          │
│    0    │  0.500   │  50/50 (incierto)       │
│    1    │  0.731   │  73.1% clase 1          │
│    2    │  0.881   │  88.1% clase 1          │
│    4    │  0.982   │  98.2% clase 1          │
│    6    │  0.998   │  99.8% clase 1          │
└─────────┴──────────┴─────────────────────────┘
```

## 5. Función Hipótesis de Regresión Logística

### Fórmula Completa

```
                           1
h_θ(x) = ──────────────────────────────────────
          1 + e^(-(θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ))


Forma compacta:
                    1
h_θ(x) = ─────────────────
          1 + e^(-θᵀx)

Donde θᵀx = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

### Interpretación Probabilística

```
h_θ(x) = P(y = 1 | x; θ)

"La probabilidad de que y sea 1, dado x, con parámetros θ"

Ejemplo: Detección de SPAM
  h_θ(email) = 0.85

  Interpretación:
    • 85% de probabilidad de ser SPAM
    • 15% de probabilidad de ser HAM (1 - 0.85)
```

### Regla de Decisión

```
┌────────────────────────────────────────────────────────┐
│  REGLA DE DECISIÓN                                     │
├────────────────────────────────────────────────────────┤
│                                                        │
│   Si h_θ(x) ≥ 0.5  →  Predecir y = 1 (SPAM)          │
│   Si h_θ(x) < 0.5  →  Predecir y = 0 (HAM)           │
│                                                        │
│   El umbral 0.5 se puede ajustar según el caso        │
└────────────────────────────────────────────────────────┘

Ejemplo con diferentes umbrales:

  Umbral = 0.5 (estándar):
    h(x) = 0.6  →  SPAM
    h(x) = 0.4  →  HAM

  Umbral = 0.8 (más conservador):
    h(x) = 0.6  →  HAM (no suficiente evidencia)
    h(x) = 0.85 →  SPAM
```

## 6. Frontera de Decisión (Decision Boundary)

### Concepto

La **frontera de decisión** es la línea/superficie donde h_θ(x) = 0.5

```
Feature x₂
    │
    │   ●  ●                    ● = Clase 1 (SPAM)
    │  ● ●  ●                   ○ = Clase 0 (HAM)
    │    ●   ●
    │  ●  ╲   ●
    │      ╲
    │   ○   ╲  ●
    │  ○ ○   ╲
    │    ○    ╲
    │  ○  ○    ╲
    │    ○      ╲
    │  ○   ○     ╲
    └─────────────────── Feature x₁

    La línea diagonal ╲ es la frontera de decisión
    A un lado: Clase 0 (HAM)
    Al otro lado: Clase 1 (SPAM)
```

### Matemáticamente

```
Frontera de decisión cuando:

θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = 0

Ejemplo con 2 features:
  θ₀ = -3, θ₁ = 1, θ₂ = 1

  Frontera: -3 + x₁ + x₂ = 0
            x₂ = 3 - x₁

  Es una línea recta en el espacio (x₁, x₂)
```

## 7. Ejemplo Práctico: Detección de SPAM

### El Problema

```
Clasificar emails como SPAM o HAM basándose en su contenido.

Datos de entrenamiento:
┌─────────────────────────────────────────┬─────────┐
│              Email (texto)              │  Label  │
├─────────────────────────────────────────┼─────────┤
│ "Hi mom, I'll be home for dinner"       │   HAM   │
│ "Meeting at 3pm tomorrow"               │   HAM   │
│ "FREE iPhone! Click NOW to WIN!"        │  SPAM   │
│ "You've WON $1000! Claim here"          │  SPAM   │
│ "Can you pick up groceries?"            │   HAM   │
│ "URGENT: Your account suspended"        │  SPAM   │
└─────────────────────────────────────────┴─────────┘
```

### Pipeline de Clasificación

```
┌─────────────────────┐
│   Email (texto)     │
│ "FREE iPhone WIN!"  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  1. PREPROCESAMIENTO                    │
│     • Lowercase                         │
│     • Eliminar puntuación               │
│     • Tokenización                      │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  2. VECTORIZACIÓN (TF-IDF)              │
│     Texto → Vector numérico             │
│     [0.0, 0.3, 0.8, 0.0, 0.5, ...]     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  3. REGRESIÓN LOGÍSTICA                 │
│     z = θ₀ + θ₁x₁ + θ₂x₂ + ...         │
│     h(x) = sigmoid(z)                   │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  4. DECISIÓN                            │
│     h(x) = 0.92                         │
│     0.92 > 0.5 → SPAM                   │
└─────────────────────────────────────────┘
```

### Palabras Importantes (Coeficientes θ)

```
Después del entrenamiento, los coeficientes θ revelan
qué palabras son indicadores de SPAM:

┌────────────────────┬──────────┬─────────────────────┐
│      Palabra       │    θ     │   Interpretación    │
├────────────────────┼──────────┼─────────────────────┤
│       free         │  +2.81   │  Fuerte → SPAM      │
│       win          │  +2.22   │  Fuerte → SPAM      │
│       claim        │  +3.39   │  Muy fuerte → SPAM  │
│       urgent       │  +2.12   │  Fuerte → SPAM      │
│       txt          │  +4.38   │  Muy fuerte → SPAM  │
├────────────────────┼──────────┼─────────────────────┤
│       home         │  -1.35   │  Indica HAM         │
│       sorry        │  -1.23   │  Indica HAM         │
│       ok           │  -1.79   │  Fuerte → HAM       │
│       later        │  -1.16   │  Indica HAM         │
└────────────────────┴──────────┴─────────────────────┘

θ positivo grande → Aumenta probabilidad de SPAM
θ negativo grande → Aumenta probabilidad de HAM
```

## 8. Función de Coste para Regresión Logística

### ¿Por qué NO usar MSE?

```
MSE no funciona bien para clasificación:

J(θ) = (1/m) · Σ[h_θ(x^(i)) - y^(i)]²

Problema: Con sigmoid, esta función NO es convexa
          → Múltiples mínimos locales
          → Gradient descent puede quedarse atascado
```

### Binary Cross-Entropy (Log Loss)

```
┌────────────────────────────────────────────────────────┐
│  FUNCIÓN DE COSTE - REGRESIÓN LOGÍSTICA                │
├────────────────────────────────────────────────────────┤
│                                                        │
│  J(θ) = -(1/m) · Σ [y·log(h) + (1-y)·log(1-h)]       │
│                                                        │
│  Donde:                                                │
│    h = h_θ(x^(i)) = sigmoid(θᵀx^(i))                  │
│    y = y^(i) ∈ {0, 1}                                 │
│                                                        │
└────────────────────────────────────────────────────────┘

Desglose por casos:

  Si y = 1 (SPAM real):
    Coste = -log(h)
    • Si h → 1: coste → 0 (buena predicción)
    • Si h → 0: coste → ∞ (mala predicción, penalización alta)

  Si y = 0 (HAM real):
    Coste = -log(1 - h)
    • Si h → 0: coste → 0 (buena predicción)
    • Si h → 1: coste → ∞ (mala predicción, penalización alta)
```

### Visualización del Coste

```
Coste
    │
  ∞ │●
    │ ╲
    │  ╲   Caso y=1
    │   ╲  Coste = -log(h)
 5  │    ╲
    │     ╲
 3  │      ╲
    │       ╲
 1  │        ╲
    │         ╲____
 0  │──────────────●──── h (predicción)
    0   0.2  0.4  0.6  0.8   1.0

    Si y=1 y predices h=0.1 → Coste ALTO
    Si y=1 y predices h=0.9 → Coste BAJO
```

## 9. Resumen

```
┌───────────────────────────────────────────────────────────────┐
│  REGRESIÓN LOGÍSTICA - RESUMEN                                │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  TIPO: Clasificación (a pesar del nombre)                     │
│                                                               │
│  FUNCIÓN HIPÓTESIS:                                           │
│            1                                                  │
│  h_θ(x) = ─────────────  donde z = θᵀx                       │
│           1 + e^(-z)                                          │
│                                                               │
│  OUTPUT: Probabilidad P(y=1|x) ∈ (0, 1)                      │
│                                                               │
│  DECISIÓN:                                                    │
│    h_θ(x) ≥ 0.5 → Clase 1                                    │
│    h_θ(x) < 0.5 → Clase 0                                    │
│                                                               │
│  FUNCIÓN DE COSTE: Binary Cross-Entropy                       │
│    J(θ) = -(1/m)·Σ[y·log(h) + (1-y)·log(1-h)]               │
│                                                               │
│  FRONTERA DE DECISIÓN: θᵀx = 0                               │
│                                                               │
│  APLICACIONES EN CIBERSEGURIDAD:                              │
│    • Detección de SPAM                                        │
│    • Detección de Phishing                                    │
│    • Clasificación de tráfico (normal/ataque)                 │
│    • Detección de fraude                                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Conceptos clave:**
- **Sigmoid:** Transforma cualquier valor a probabilidad (0, 1)
- **Umbral:** Punto de corte para decidir la clase (default 0.5)
- **Log Loss:** Función de coste que penaliza predicciones incorrectas
- **Coeficientes θ:** Indican importancia y dirección de cada feature

---

**Siguiente:** Vectorización de texto con TF-IDF para clasificación
