# Introducción a las Redes Neuronales

## 1. ¿Qué es una Red Neuronal?

### Inspiración Biológica

```
┌────────────────────────────────────────────────────────────────┐
│  NEURONA BIOLÓGICA                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                    Dendritas                                   │
│                   (entradas)                                   │
│                      │ │ │                                     │
│                      ▼ ▼ ▼                                     │
│                   ┌───────┐                                    │
│                   │       │                                    │
│                   │ Soma  │ ← Cuerpo celular (procesamiento)   │
│                   │       │                                    │
│                   └───┬───┘                                    │
│                       │                                        │
│                       ▼                                        │
│                     Axón                                       │
│                   (salida)                                     │
│                       │                                        │
│                       ▼                                        │
│              Sinapsis (conexión                                │
│              a otras neuronas)                                 │
│                                                                │
│  La neurona DISPARA si la suma de entradas supera un umbral   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Neurona Artificial (Perceptrón)

```
┌────────────────────────────────────────────────────────────────┐
│  NEURONA ARTIFICIAL                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Entradas         Pesos        Suma         Activación        │
│                                                                │
│     x₁ ───────── w₁ ─────┐                                     │
│                          │                                     │
│     x₂ ───────── w₂ ─────┼───→  Σ  ───→  f(z)  ───→  salida   │
│                          │      ↑                              │
│     x₃ ───────── w₃ ─────┘      │                              │
│                                 │                              │
│     1  ───────── b ─────────────┘  (bias)                      │
│                                                                │
│                                                                │
│   z = w₁x₁ + w₂x₂ + w₃x₃ + b   (suma ponderada)               │
│   salida = f(z)                 (función de activación)        │
│                                                                │
│   Los PESOS (w) y BIAS (b) son los parámetros a aprender       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### De Neurona a Red

```
┌────────────────────────────────────────────────────────────────┐
│  RED NEURONAL = Muchas neuronas conectadas en capas            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   CAPA DE        CAPA(S)           CAPA DE                     │
│   ENTRADA        OCULTA(S)         SALIDA                      │
│                                                                │
│     x₁  ●───────●───────●                                      │
│              ╲  │  ╱    │╲                                     │
│     x₂  ●─────╲─●─╱─────●──╲                                   │
│              ╱  │  ╲    │   ╲                                  │
│     x₃  ●───────●───────●────●───→  ŷ                          │
│              ╲  │  ╱    │   ╱                                  │
│     x₄  ●─────╲─●─╱─────●──╱                                   │
│              ╱  │  ╲    │╱                                     │
│     x₅  ●───────●───────●                                      │
│                                                                │
│   Features    Representaciones    Predicción                   │
│   originales  aprendidas                                       │
│                                                                │
│   Cada línea tiene un PESO                                     │
│   Cada neurona tiene un BIAS y una ACTIVACIÓN                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. Funciones de Activación

### ¿Por qué necesitamos activaciones?

```
┌────────────────────────────────────────────────────────────────┐
│  SIN ACTIVACIÓN = Solo transformaciones lineales               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Capa 1: z₁ = W₁x + b₁                                        │
│  Capa 2: z₂ = W₂z₁ + b₂ = W₂(W₁x + b₁) + b₂                  │
│              = (W₂W₁)x + (W₂b₁ + b₂)                          │
│              = W'x + b'  ← ¡Es solo otra transformación lineal!│
│                                                                │
│  No importa cuántas capas pongas, sin activación no lineal     │
│  la red solo puede aprender funciones LINEALES                 │
│                                                                │
│  CON ACTIVACIÓN no lineal:                                     │
│    Puede aprender CUALQUIER función (aproximador universal)    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Activaciones Comunes

```
┌────────────────────────────────────────────────────────────────┐
│  FUNCIONES DE ACTIVACIÓN                                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SIGMOID                                                    │
│     ─────────                                                  │
│     σ(z) = 1 / (1 + e⁻ᶻ)                                       │
│     Rango: (0, 1)                                              │
│                                                                │
│         1 │        ___________                                 │
│           │      ╱                                             │
│       0.5 │    ╱                                               │
│           │  ╱                                                 │
│         0 │╱___________                                        │
│           └─────────────────                                   │
│                  0                                             │
│                                                                │
│     Uso: Capa de salida para probabilidades (clasificación)    │
│     Problema: Vanishing gradient en valores extremos           │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  2. TANH                                                       │
│     ────                                                       │
│     tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)                         │
│     Rango: (-1, 1)                                             │
│                                                                │
│         1 │        ___________                                 │
│           │      ╱                                             │
│         0 │────╱────────────                                   │
│           │  ╱                                                 │
│        -1 │╱___________                                        │
│           └─────────────────                                   │
│                                                                │
│     Uso: Capas ocultas (centrado en 0)                         │
│     Problema: Vanishing gradient también                       │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  3. ReLU (Rectified Linear Unit) ★ MÁS USADA                   │
│     ────────────────────────────                               │
│     ReLU(z) = max(0, z)                                        │
│     Rango: [0, ∞)                                              │
│                                                                │
│           │              ╱                                     │
│           │            ╱                                       │
│           │          ╱                                         │
│         0 │________╱                                           │
│           └─────────────────                                   │
│                  0                                             │
│                                                                │
│     Ventajas: Rápida, no vanishing gradient (para z > 0)       │
│     Problema: "Dying ReLU" (neuronas muertas si z < 0)         │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  4. Leaky ReLU                                                 │
│     ──────────                                                 │
│     f(z) = z if z > 0, else α·z (α ≈ 0.01)                    │
│                                                                │
│           │              ╱                                     │
│           │            ╱                                       │
│         0 │__________╱                                         │
│           │        ╱  (pendiente pequeña)                      │
│           └─────────────────                                   │
│                                                                │
│     Soluciona el problema de "dying ReLU"                      │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  5. SOFTMAX (para clasificación multiclase)                    │
│     ───────                                                    │
│     softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ                                │
│                                                                │
│     Convierte vector de scores en PROBABILIDADES               │
│     Suma de todas las salidas = 1                              │
│                                                                │
│     Entrada: [2.0, 1.0, 0.1]                                   │
│     Salida:  [0.7, 0.2, 0.1] → probabilidades                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Guía de Uso

```
┌─────────────────────┬────────────────────────────────────────┐
│  Tipo de capa       │  Activación recomendada                │
├─────────────────────┼────────────────────────────────────────┤
│ Capas ocultas       │  ReLU (o Leaky ReLU, ELU, GELU)       │
├─────────────────────┼────────────────────────────────────────┤
│ Salida binaria      │  Sigmoid (probabilidad 0-1)            │
├─────────────────────┼────────────────────────────────────────┤
│ Salida multiclase   │  Softmax (probabilidades que suman 1)  │
├─────────────────────┼────────────────────────────────────────┤
│ Salida regresión    │  Ninguna (lineal)                      │
└─────────────────────┴────────────────────────────────────────┘
```

## 3. Forward Propagation

### Flujo de Datos

```
┌────────────────────────────────────────────────────────────────┐
│  FORWARD PROPAGATION = Pasar datos de entrada a salida         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Ejemplo: Red con 1 capa oculta                                │
│                                                                │
│  ENTRADA (x)        CAPA OCULTA (h)       SALIDA (ŷ)          │
│  [x₁, x₂, x₃]                                                  │
│       │                                                        │
│       ▼                                                        │
│   ┌───────┐                                                    │
│   │ z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾    │  Transformación lineal           │
│   │ h = ReLU(z⁽¹⁾)         │  Activación                      │
│   └───────┘                                                    │
│       │                                                        │
│       ▼                                                        │
│   ┌───────┐                                                    │
│   │ z⁽²⁾ = W⁽²⁾h + b⁽²⁾    │  Transformación lineal           │
│   │ ŷ = σ(z⁽²⁾)            │  Activación (sigmoid para binario)│
│   └───────┘                                                    │
│       │                                                        │
│       ▼                                                        │
│   PREDICCIÓN                                                   │
│                                                                │
│  Cada capa transforma la representación de los datos           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ejemplo Numérico

```
ENTRADA: x = [1.0, 2.0]

CAPA OCULTA (2 neuronas):
  W⁽¹⁾ = [[0.5, -0.3],     b⁽¹⁾ = [0.1, -0.1]
          [0.2,  0.8]]

  z⁽¹⁾ = W⁽¹⁾ · x + b⁽¹⁾
       = [[0.5×1 + (-0.3)×2], + [0.1]  = [-0.1 + 0.1] = [0.0]
          [0.2×1 +   0.8×2]]   [-0.1]    [1.8 - 0.1]    [1.7]

  h = ReLU(z⁽¹⁾) = [max(0, 0.0), max(0, 1.7)] = [0.0, 1.7]

CAPA SALIDA (1 neurona):
  W⁽²⁾ = [0.4, 0.6]     b⁽²⁾ = 0.2

  z⁽²⁾ = W⁽²⁾ · h + b⁽²⁾
       = 0.4×0.0 + 0.6×1.7 + 0.2 = 1.22

  ŷ = sigmoid(1.22) = 1/(1 + e⁻¹·²²) ≈ 0.77

PREDICCIÓN: 0.77 (77% probabilidad de clase positiva)
```

## 4. Función de Pérdida (Loss)

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  FUNCIÓN DE PÉRDIDA = Mide qué tan MAL predice el modelo       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Objetivo del entrenamiento: MINIMIZAR la pérdida              │
│                                                                │
│  Pérdida                                                       │
│     │                                                          │
│     │\                                                         │
│     │ \                                                        │
│     │  \                                                       │
│     │   \      ___                                             │
│     │    \___/    \___                                         │
│     │                  \_____  ← Mínimo (queremos llegar aquí) │
│     └───────────────────────────── Parámetros                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Funciones de Pérdida Comunes

```
┌────────────────────────────────────────────────────────────────┐
│  FUNCIONES DE PÉRDIDA                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. BINARY CROSS-ENTROPY (Clasificación binaria)               │
│     ────────────────────────────────────────                   │
│     L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]                          │
│                                                                │
│     y=1, ŷ=0.9: L = -log(0.9) = 0.105  ← Buena predicción     │
│     y=1, ŷ=0.1: L = -log(0.1) = 2.303  ← Mala predicción      │
│                                                                │
│  2. CATEGORICAL CROSS-ENTROPY (Clasificación multiclase)       │
│     ────────────────────────────────────────────────           │
│     L = -Σᵢ yᵢ·log(ŷᵢ)                                        │
│                                                                │
│     y = [0, 1, 0] (clase 2)                                    │
│     ŷ = [0.1, 0.8, 0.1]                                        │
│     L = -log(0.8) = 0.223                                      │
│                                                                │
│  3. MEAN SQUARED ERROR (Regresión)                             │
│     ─────────────────────────────                              │
│     L = (1/n) Σᵢ (yᵢ - ŷᵢ)²                                   │
│                                                                │
│     Penaliza más los errores grandes                           │
│                                                                │
│  4. MEAN ABSOLUTE ERROR (Regresión, robusta)                   │
│     ────────────────────────────────────────                   │
│     L = (1/n) Σᵢ |yᵢ - ŷᵢ|                                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 5. Backpropagation

### Concepto: Regla de la Cadena

```
┌────────────────────────────────────────────────────────────────┐
│  BACKPROPAGATION = Calcular gradientes hacia atrás             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Problema: ¿Cómo saber cuánto cambiar cada peso para           │
│            reducir la pérdida?                                 │
│                                                                │
│  Solución: Regla de la cadena (calculus)                       │
│                                                                │
│            ∂L     ∂L    ∂ŷ    ∂z⁽²⁾   ∂h    ∂z⁽¹⁾              │
│           ──── = ─── · ─── · ───── · ─── · ─────               │
│           ∂W⁽¹⁾  ∂ŷ   ∂z⁽²⁾   ∂h    ∂z⁽¹⁾  ∂W⁽¹⁾              │
│                                                                │
│  Propagamos el error HACIA ATRÁS desde la salida               │
│                                                                │
│   FORWARD:   x → h → ŷ → L                                     │
│   BACKWARD:  x ← h ← ŷ ← L  (gradientes)                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización

```
FORWARD PASS:
─────────────
  x ─────→ z⁽¹⁾ ─────→ h ─────→ z⁽²⁾ ─────→ ŷ ─────→ L
      W⁽¹⁾       ReLU       W⁽²⁾      sigmoid    loss


BACKWARD PASS (Gradientes):
────────────────────────────
  ∂L/∂W⁽¹⁾ ←── ∂L/∂z⁽¹⁾ ←── ∂L/∂h ←── ∂L/∂z⁽²⁾ ←── ∂L/∂ŷ ←── L
                                                        │
                                                        │
  Cada flecha aplica la regla de la cadena              │
  multiplicando por la derivada local                   │
                                                        ▼
                                              ∂L/∂ŷ = (ŷ - y)
```

### Algoritmo

```
┌────────────────────────────────────────────────────────────────┐
│  ALGORITMO DE BACKPROPAGATION                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. FORWARD PASS                                               │
│     Calcular todas las activaciones y guardarlas               │
│                                                                │
│  2. CALCULAR PÉRDIDA                                           │
│     L = loss(ŷ, y)                                             │
│                                                                │
│  3. BACKWARD PASS                                              │
│     Para cada capa, de atrás hacia adelante:                   │
│       a. Calcular ∂L/∂z (gradiente de la pérdida vs pre-activación)
│       b. Calcular ∂L/∂W = ∂L/∂z · activación_anterior          │
│       c. Calcular ∂L/∂b = ∂L/∂z                                │
│       d. Propagar ∂L/∂activación a la capa anterior            │
│                                                                │
│  4. ACTUALIZAR PESOS                                           │
│     W = W - α · ∂L/∂W                                          │
│     b = b - α · ∂L/∂b                                          │
│                                                                │
│     α = learning rate                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 6. Gradient Descent y Optimizadores

### Gradient Descent Básico

```
┌────────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Actualización de parámetros:                                  │
│                                                                │
│     θ = θ - α · ∇L(θ)                                          │
│                                                                │
│  Donde:                                                        │
│    θ = parámetros (W, b)                                       │
│    α = learning rate                                           │
│    ∇L = gradiente de la pérdida                                │
│                                                                │
│  Loss                                                          │
│    │                                                           │
│    │ ●  ← Posición inicial                                     │
│    │  ╲                                                        │
│    │   ╲  ← Paso de gradiente                                  │
│    │    ●                                                      │
│    │     ╲                                                     │
│    │      ●                                                    │
│    │       ╲__                                                 │
│    │          ╲__●  ← Mínimo                                   │
│    └─────────────────── θ                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Variantes de Gradient Descent

```
┌────────────────────────────────────────────────────────────────┐
│  VARIANTES DE GRADIENT DESCENT                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. BATCH Gradient Descent                                     │
│     ─────────────────────                                      │
│     Usa TODOS los datos en cada paso                           │
│     + Gradiente preciso                                        │
│     - Muy lento para datasets grandes                          │
│     - Puede quedar en mínimos locales                          │
│                                                                │
│  2. STOCHASTIC Gradient Descent (SGD)                          │
│     ─────────────────────────────────                          │
│     Usa UNA muestra en cada paso                               │
│     + Muy rápido                                               │
│     + Puede escapar mínimos locales                            │
│     - Gradiente ruidoso                                        │
│                                                                │
│  3. MINI-BATCH Gradient Descent ★ MÁS USADO                    │
│     ───────────────────────────                                │
│     Usa un BATCH (ej: 32, 64, 128 muestras)                    │
│     + Balance entre precisión y velocidad                      │
│     + Aprovecha GPUs                                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Optimizadores Modernos

```
┌────────────────────────────────────────────────────────────────┐
│  OPTIMIZADORES                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. SGD con Momentum                                           │
│     ──────────────────                                         │
│     Añade "inercia" para acelerar y suavizar                   │
│     v = β·v + ∇L                                               │
│     θ = θ - α·v                                                │
│                                                                │
│  2. RMSprop                                                    │
│     ───────                                                    │
│     Adapta learning rate por parámetro                         │
│     Divide por raíz de gradientes pasados                      │
│                                                                │
│  3. ADAM (Adaptive Moment Estimation) ★ MÁS USADO              │
│     ─────────────────────────────────                          │
│     Combina Momentum + RMSprop                                 │
│     • Mantiene media móvil del gradiente (momentum)            │
│     • Mantiene media móvil del gradiente² (adaptivo)           │
│     • Corrección de bias inicial                               │
│                                                                │
│     Parámetros típicos:                                        │
│       α = 0.001 (learning rate)                                │
│       β₁ = 0.9 (momentum)                                      │
│       β₂ = 0.999 (RMSprop)                                     │
│                                                                │
│  4. AdamW                                                      │
│     ─────                                                      │
│     Adam con weight decay correctamente implementado           │
│     Mejor para generalización                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 7. Regularización

### Técnicas de Regularización

```
┌────────────────────────────────────────────────────────────────┐
│  REGULARIZACIÓN = Prevenir overfitting                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. L2 REGULARIZATION (Weight Decay)                           │
│     ────────────────────────────────                           │
│     L_total = L_data + λ·Σw²                                   │
│                                                                │
│     Penaliza pesos grandes → pesos más pequeños                │
│     → modelo más simple → menos overfitting                    │
│                                                                │
│  2. DROPOUT                                                    │
│     ───────                                                    │
│     Durante entrenamiento: "apagar" neuronas aleatoriamente    │
│                                                                │
│     Sin dropout:        Con dropout (p=0.5):                   │
│        ●───●───●           ●───○───●                           │
│        ●───●───●           ○───●───●                           │
│        ●───●───●           ●───●───○                           │
│                                                                │
│     Fuerza redundancia → más robusto                           │
│     En test: usar todas las neuronas (escalar por 1-p)         │
│                                                                │
│  3. BATCH NORMALIZATION                                        │
│     ─────────────────────                                      │
│     Normalizar activaciones en cada capa                       │
│     + Permite learning rates más altos                         │
│     + Reduce dependencia de inicialización                     │
│     + Efecto regularizador                                     │
│                                                                │
│  4. EARLY STOPPING                                             │
│     ──────────────                                             │
│     Parar cuando validation loss deja de mejorar               │
│                                                                │
│     Loss │                                                     │
│          │╲  train                                             │
│          │ ╲____                                               │
│          │      ╲____                                          │
│          │           ╲____                                     │
│          │     ___________╱  validation                        │
│          │____╱   ↑                                            │
│          └────────┴───────── epochs                            │
│                   PARAR AQUÍ                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. Implementación en Python

### Con PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Definir red neuronal
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Datos de ejemplo
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000, 1)).float()

# DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Crear modelo
model = SimpleNN(input_size=10, hidden_size=64, output_size=1)

# Loss y optimizador
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()  # Limpiar gradientes
        loss.backward()        # Calcular gradientes
        optimizer.step()       # Actualizar pesos

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
```

### Con Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Datos de ejemplo
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, (1000, 1))

# Definir modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Entrenar
history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluar
loss, accuracy = model.evaluate(X, y)
print(f'Accuracy: {accuracy:.4f}')
```

## 9. Hiperparámetros Importantes

```
┌────────────────────┬──────────────────────────────────────────┐
│  Hiperparámetro    │  Guía                                    │
├────────────────────┼──────────────────────────────────────────┤
│ Learning rate (α)  │  Empezar: 0.001 (con Adam)               │
│                    │  Muy alto: no converge                   │
│                    │  Muy bajo: muy lento                     │
├────────────────────┼──────────────────────────────────────────┤
│ Batch size         │  32, 64, 128 (potencias de 2)            │
│                    │  Mayor: más estable, necesita más RAM    │
│                    │  Menor: más ruido, puede generalizar mejor│
├────────────────────┼──────────────────────────────────────────┤
│ Neuronas/capa      │  Potencias de 2: 32, 64, 128, 256...     │
│                    │  Más neuronas = más capacidad            │
├────────────────────┼──────────────────────────────────────────┤
│ Número de capas    │  1-3 para problemas simples              │
│                    │  Más capas = más abstracción             │
├────────────────────┼──────────────────────────────────────────┤
│ Dropout rate       │  0.2-0.5                                 │
│                    │  Mayor si overfitting                    │
├────────────────────┼──────────────────────────────────────────┤
│ Epochs             │  Usar early stopping                     │
│                    │  Monitorear validation loss              │
└────────────────────┴──────────────────────────────────────────┘
```

## 10. Aplicaciones en Ciberseguridad

```
┌────────────────────────────────────────────────────────────────┐
│  REDES NEURONALES EN CIBERSEGURIDAD                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DETECCIÓN DE MALWARE                                       │
│     • Clasificación de ejecutables por comportamiento          │
│     • Análisis de APIs calls                                   │
│     • Detección de familias de malware                         │
│                                                                │
│  2. DETECCIÓN DE INTRUSIONES (IDS)                             │
│     • Clasificar tráfico de red                                │
│     • Identificar patrones de ataque                           │
│     • Detección de anomalías                                   │
│                                                                │
│  3. PHISHING DETECTION                                         │
│     • Análisis de URLs                                         │
│     • Clasificación de emails                                  │
│     • Detección de sitios fraudulentos                         │
│                                                                │
│  4. ANÁLISIS DE LOGS                                           │
│     • Clasificación de eventos                                 │
│     • Detección de comportamiento anómalo                      │
│     • Correlación de alertas                                   │
│                                                                │
│  5. ANÁLISIS DE CÓDIGO                                         │
│     • Detección de vulnerabilidades                            │
│     • Identificación de código malicioso                       │
│     • Análisis de binarios                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 11. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  REDES NEURONALES - RESUMEN                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  COMPONENTES:                                                  │
│    • Neuronas: suma ponderada + activación                     │
│    • Capas: entrada, ocultas, salida                           │
│    • Pesos y biases: parámetros a aprender                     │
│                                                                │
│  ACTIVACIONES:                                                 │
│    • ReLU: capas ocultas (más común)                           │
│    • Sigmoid: salida binaria                                   │
│    • Softmax: salida multiclase                                │
│                                                                │
│  ENTRENAMIENTO:                                                │
│    1. Forward pass: calcular predicción                        │
│    2. Loss: medir error                                        │
│    3. Backprop: calcular gradientes                            │
│    4. Update: ajustar pesos (optimizer)                        │
│                                                                │
│  OPTIMIZADORES:                                                │
│    • Adam: el más usado                                        │
│    • SGD con momentum: clásico                                 │
│                                                                │
│  REGULARIZACIÓN:                                               │
│    • Dropout: apagar neuronas aleatoriamente                   │
│    • L2/Weight decay: penalizar pesos grandes                  │
│    • Early stopping: parar cuando val_loss sube                │
│                                                                │
│  FRAMEWORKS:                                                   │
│    • PyTorch: más flexible, investigación                      │
│    • TensorFlow/Keras: más alto nivel, producción              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Redes Feedforward (MLP)
