# Redes Neuronales Convolucionales (CNN)

## 1. ¿Qué es una CNN?

### Motivación: El Problema de las Imágenes

```
┌────────────────────────────────────────────────────────────────┐
│  PROBLEMA CON MLP PARA IMÁGENES                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Imagen 224×224×3 (RGB) = 150,528 pixels                       │
│                                                                │
│  MLP con 1 capa oculta de 1000 neuronas:                       │
│    Parámetros = 150,528 × 1000 = 150 MILLONES                  │
│                                                                │
│  PROBLEMAS:                                                    │
│    ✗ Demasiados parámetros → overfitting                       │
│    ✗ No captura estructura espacial                            │
│    ✗ No invariante a traslación                                │
│    ✗ Pierde relación entre pixels vecinos                      │
│                                                                │
│  SOLUCIÓN: CNN                                                 │
│    ✓ Comparte pesos (menos parámetros)                         │
│    ✓ Captura patrones locales                                  │
│    ✓ Invariante a traslación                                   │
│    ✓ Preserva estructura espacial                              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Concepto de Convolución

```
┌────────────────────────────────────────────────────────────────┐
│  OPERACIÓN DE CONVOLUCIÓN                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Un FILTRO (kernel) se desliza sobre la imagen                 │
│  calculando productos punto en cada posición                   │
│                                                                │
│  IMAGEN DE ENTRADA          FILTRO 3×3        SALIDA           │
│  (5×5)                      (kernel)                           │
│                                                                │
│  ┌─────────────────┐        ┌─────────┐                        │
│  │ 1  0  1  0  1   │        │ 1  0  1 │                        │
│  │ 0  1  0  1  0   │   *    │ 0  1  0 │   =    ┌───────────┐  │
│  │ 1  0 [1  0  1]  │        │ 1  0  1 │        │     4     │  │
│  │ 0  1 [0  1  0]  │        └─────────┘        │   ...     │  │
│  │ 1  0 [1  0  1]  │                           └───────────┘  │
│  └─────────────────┘                             (3×3)         │
│        ↑                                                       │
│    región actual                                               │
│                                                                │
│  Cálculo: 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 + 0×0 + 1×1 = 4
│                                                                │
│  El filtro detecta un PATRÓN específico                        │
│  Diferentes filtros detectan diferentes patrones:              │
│    • Bordes horizontales                                       │
│    • Bordes verticales                                         │
│    • Esquinas                                                  │
│    • Texturas                                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Visualización de Filtros

```
FILTROS COMUNES:

BORDE HORIZONTAL:           BORDE VERTICAL:           BORDE DIAGONAL:
┌─────────────┐            ┌─────────────┐           ┌─────────────┐
│ -1  -1  -1  │            │ -1   0   1  │           │  0  -1  -1  │
│  0   0   0  │            │ -1   0   1  │           │  1   0  -1  │
│  1   1   1  │            │ -1   0   1  │           │  1   1   0  │
└─────────────┘            └─────────────┘           └─────────────┘


DETECTOR DE ESQUINAS:       GAUSSIAN BLUR:            SHARPEN:
┌─────────────┐            ┌─────────────┐           ┌─────────────┐
│  1   0  -1  │            │ 1/16  1/8  1/16│        │  0  -1   0  │
│  0   0   0  │            │ 1/8   1/4  1/8 │        │ -1   5  -1  │
│ -1   0   1  │            │ 1/16  1/8  1/16│        │  0  -1   0  │
└─────────────┘            └─────────────┘           └─────────────┘

En CNN, los filtros se APRENDEN automáticamente
(no se definen manualmente)
```

## 2. Arquitectura de una CNN

### Componentes Principales

```
┌────────────────────────────────────────────────────────────────┐
│  ARQUITECTURA CNN                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IMAGEN     CONV      POOL     CONV      POOL    FLATTEN  FC  │
│   ↓          ↓         ↓        ↓         ↓         ↓      ↓  │
│  ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐   ┌───┐  │
│  │   │    │▓▓▓│    │▓▓ │    │▓▓▓│    │▓▓ │    │▓  │   │▓▓▓│  │
│  │   │ →  │▓▓▓│ →  │▓▓ │ →  │▓▓▓│ →  │▓▓ │ →  │▓  │ → │▓▓▓│ → ŷ
│  │   │    │▓▓▓│    │   │    │▓▓▓│    │   │    │▓  │   │▓▓▓│  │
│  └───┘    └───┘    └───┘    └───┘    └───┘    └───┘   └───┘  │
│                                                                │
│  224×224   112×112  56×56   28×28    14×14    784    Clases   │
│   ×3        ×32      ×32     ×64      ×64                      │
│                                                                │
│  Extracción de features              Clasificación            │
│  (convoluciones)                     (capas densas)            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Capa Convolucional

```
┌────────────────────────────────────────────────────────────────┐
│  CAPA CONVOLUCIONAL                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  PARÁMETROS:                                                   │
│    • kernel_size: tamaño del filtro (ej: 3×3, 5×5)             │
│    • num_filters: número de filtros (ej: 32, 64, 128)          │
│    • stride: paso del filtro (ej: 1, 2)                        │
│    • padding: relleno de bordes ('same' o 'valid')             │
│                                                                │
│  ENTRADA: H × W × C_in (altura × ancho × canales entrada)      │
│  SALIDA:  H' × W' × C_out (nuevas dimensiones × filtros)       │
│                                                                │
│  Ejemplo: Imagen RGB 32×32×3, kernel 3×3, 64 filtros, stride 1 │
│                                                                │
│    Entrada:  32 × 32 × 3                                       │
│    Cada filtro: 3 × 3 × 3 = 27 pesos + 1 bias = 28             │
│    64 filtros: 64 × 28 = 1,792 parámetros                      │
│    Salida (padding='same'): 32 × 32 × 64                       │
│                                                                │
│  vs MLP: 32×32×3 × 64 = 196,608 parámetros (109× más!)        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Pooling

```
┌────────────────────────────────────────────────────────────────┐
│  POOLING = Reducción de dimensionalidad                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  MAX POOLING (más común):                                      │
│  Toma el valor MÁXIMO en cada ventana                          │
│                                                                │
│  ┌─────────────────┐         ┌─────────┐                       │
│  │ 1   3 │ 2   4   │         │         │                       │
│  │ 5   6 │ 7   8   │   →     │ 6    8  │                       │
│  ├───────┼─────────┤         │         │                       │
│  │ 9   2 │ 1   0   │         │ 9    5  │                       │
│  │ 3   1 │ 5   2   │         │         │                       │
│  └─────────────────┘         └─────────┘                       │
│      4×4                        2×2                            │
│                                                                │
│  Pool size: 2×2, Stride: 2                                     │
│  Reduce dimensiones a la mitad                                 │
│                                                                │
│  AVERAGE POOLING:                                              │
│  Toma el promedio de cada ventana                              │
│                                                                │
│  GLOBAL AVERAGE POOLING:                                       │
│  Promedia todo el feature map → 1 valor por canal              │
│  Útil antes de la capa final                                   │
│                                                                │
│  BENEFICIOS DEL POOLING:                                       │
│    ✓ Reduce parámetros                                         │
│    ✓ Reduce cómputo                                            │
│    ✓ Invariancia a pequeñas traslaciones                       │
│    ✓ Evita overfitting                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Stride y Padding

```
┌────────────────────────────────────────────────────────────────┐
│  STRIDE Y PADDING                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  STRIDE: Paso del filtro                                       │
│                                                                │
│  Stride = 1:              Stride = 2:                          │
│  ┌─┬─┬─┬─┐                ┌─┬─┬─┬─┐                            │
│  │█│█│▒│▒│ → █ luego ▒    │█│█│▒│▒│ → █ luego ▒                │
│  │█│█│▒│▒│                │█│█│▒│▒│   (salta 2)                │
│  │ │ │ │ │                │ │ │ │ │                            │
│  │ │ │ │ │                │ │ │ │ │                            │
│  └─┴─┴─┴─┘                └─┴─┴─┴─┘                            │
│  Salida: casi mismo tamaño Salida: mitad del tamaño            │
│                                                                │
│  PADDING: Añadir bordes                                        │
│                                                                │
│  'valid' (sin padding):   'same' (con padding):                │
│  ┌─────┐                  ┌───────────┐                        │
│  │ IMG │  →  Más pequeño  │ 0 0 0 0 0 │  →  Mismo tamaño       │
│  └─────┘                  │ 0│IMG│ 0  │                        │
│                           │ 0 0 0 0 0 │                        │
│                           └───────────┘                        │
│                                                                │
│  FÓRMULA DE TAMAÑO DE SALIDA:                                  │
│    O = (I - K + 2P) / S + 1                                    │
│                                                                │
│    I = tamaño entrada                                          │
│    K = tamaño kernel                                           │
│    P = padding                                                 │
│    S = stride                                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 3. Arquitecturas Famosas

### LeNet-5 (1998)

```
LA PRIMERA CNN EXITOSA (reconocimiento de dígitos)

Input(32×32×1) → Conv(5×5, 6) → Pool(2×2) →
Conv(5×5, 16) → Pool(2×2) → FC(120) → FC(84) → Output(10)

Parámetros: ~60,000
```

### VGG-16 (2014)

```
FILOSOFÍA: Filtros pequeños (3×3), muchas capas

Input(224×224×3) →
[Conv3-64]×2 → Pool →
[Conv3-128]×2 → Pool →
[Conv3-256]×3 → Pool →
[Conv3-512]×3 → Pool →
[Conv3-512]×3 → Pool →
FC(4096) → FC(4096) → FC(1000)

Parámetros: ~138 millones
```

### ResNet (2015)

```
INNOVACIÓN: Conexiones residuales (skip connections)

         ┌──────────────────────┐
         │                      │
    x ───┴─→ Conv → ReLU → Conv ─⊕─→ ReLU → salida
                                ↑
                                │
                    identity (x)

"Si una capa no ayuda, al menos pasa la entrada"
Permite redes muy profundas (50, 101, 152 capas)
```

### Resumen de Arquitecturas

```
┌──────────────┬─────────────┬────────────────┬───────────────┐
│ Arquitectura │    Año      │   Parámetros   │  Top-5 Error  │
├──────────────┼─────────────┼────────────────┼───────────────┤
│ LeNet-5      │    1998     │     60K        │      -        │
│ AlexNet      │    2012     │     60M        │    16.4%      │
│ VGG-16       │    2014     │    138M        │     7.3%      │
│ GoogLeNet    │    2014     │      7M        │     6.7%      │
│ ResNet-50    │    2015     │     25M        │     3.6%      │
│ ResNet-152   │    2015     │     60M        │     3.0%      │
│ EfficientNet │    2019     │     5-66M      │     2.0%      │
└──────────────┴─────────────┴────────────────┴───────────────┘
```

## 4. Implementación

### PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bloque convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Capa de clasificación
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Bloque 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Bloque 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Bloque 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Clasificación
        x = self.dropout(x)
        x = self.fc(x)

        return x

# Crear modelo
model = SimpleCNN(num_classes=10)
print(model)

# Verificar tamaño de salida
x = torch.randn(1, 3, 32, 32)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### Keras/TensorFlow

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization,
                                      GlobalAveragePooling2D)

model = Sequential([
    # Bloque 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Bloque 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Bloque 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Clasificación
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()
```

## 5. Transfer Learning

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  TRANSFER LEARNING                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Usar un modelo pre-entrenado en un dataset grande       │
│        (ImageNet: 1.2M imágenes, 1000 clases)                  │
│        y adaptarlo a tu problema específico                    │
│                                                                │
│  MODELO PRE-ENTRENADO (ej: ResNet-50 en ImageNet)              │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Conv layers │ Conv layers │ Conv layers │ FC(1000) │        │
│  │  (básicos)  │ (medios)    │ (complejos) │ ImageNet │        │
│  └─────────────┴─────────────┴─────────────┴──────────┘        │
│       │               │             │            ✗             │
│       │    CONGELAR   │   FINE-TUNE │      REEMPLAZAR          │
│       ▼               ▼             ▼            ▼             │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Conv layers │ Conv layers │ Conv layers │ FC(N)   │         │
│  │ (no cambian)│(poco cambio)│ (se ajustan)│ TU TAREA│         │
│  └─────────────────────────────────────────────────┘           │
│                                                                │
│  BENEFICIOS:                                                   │
│    ✓ Funciona con pocos datos                                  │
│    ✓ Entrenamiento rápido                                      │
│    ✓ Mejor generalización                                      │
│    ✓ Features ya aprendidas (bordes, texturas, etc.)           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Estrategias de Transfer Learning

```
┌────────────────────────────────────────────────────────────────┐
│  ESTRATEGIAS                                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. FEATURE EXTRACTION (pocos datos)                           │
│     ─────────────────────────────────                          │
│     • Congelar TODAS las capas convolucionales                 │
│     • Solo entrenar la capa de clasificación                   │
│     • Rápido, evita overfitting                                │
│                                                                │
│  2. FINE-TUNING (más datos)                                    │
│     ────────────────────────                                   │
│     • Descongelar últimas capas conv                           │
│     • Entrenar con learning rate BAJO                          │
│     • Mejor rendimiento, más riesgo overfitting                │
│                                                                │
│  3. FINE-TUNING PROGRESIVO                                     │
│     ─────────────────────────                                  │
│     • Primero: feature extraction                              │
│     • Luego: descongelar capas gradualmente                    │
│     • Más estable                                              │
│                                                                │
│  CUÁNDO USAR CADA UNO:                                         │
│                                                                │
│    Datos      Similitud con ImageNet    Estrategia             │
│    ─────      ─────────────────────    ──────────             │
│    Pocos      Similar                   Feature extraction     │
│    Pocos      Diferente                 Feature extraction*    │
│    Muchos     Similar                   Fine-tuning            │
│    Muchos     Diferente                 Fine-tuning (más capas)│
│                                                                │
│    *Quizás necesites capas intermedias, no las últimas         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código Transfer Learning

```python
import torch
import torch.nn as nn
from torchvision import models

# Cargar modelo pre-entrenado
model = models.resnet50(pretrained=True)

# Ver arquitectura
print(model)

# OPCIÓN 1: Feature Extraction (congelar todo)
for param in model.parameters():
    param.requires_grad = False

# Reemplazar capa final
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)  # 2 clases: malware vs benigno
)

# Solo entrenar la nueva capa
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# OPCIÓN 2: Fine-Tuning (descongelar últimas capas)
# Primero congelar todo
for param in model.parameters():
    param.requires_grad = False

# Descongelar layer4 (últimas conv)
for param in model.layer4.parameters():
    param.requires_grad = True

# Reemplazar FC
model.fc = nn.Linear(num_features, 2)

# Optimizer con diferentes learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Bajo para fine-tuning
    {'params': model.fc.parameters(), 'lr': 1e-3}       # Normal para nueva capa
])
```

## 6. Clasificación de Malware con CNN

### Concepto: Malware como Imagen

```
┌────────────────────────────────────────────────────────────────┐
│  MALWARE VISUALIZATION                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IDEA: Convertir bytes del ejecutable en imagen                │
│                                                                │
│  ARCHIVO PE (ejecutable):                                      │
│  ┌─────────────────────────────────────────┐                   │
│  │ 4D 5A 90 00 03 00 00 00 04 00 00 00 ... │                   │
│  │ FF FF 00 00 B8 00 00 00 00 00 00 00 ... │                   │
│  │ ... (bytes del ejecutable)              │                   │
│  └─────────────────────────────────────────┘                   │
│                         ↓                                      │
│  CONVERTIR A IMAGEN:                                           │
│  - Cada byte (0-255) → pixel en escala de grises               │
│  - Reorganizar en matriz 2D (ej: 256×256)                      │
│                         ↓                                      │
│  IMAGEN RESULTANTE:                                            │
│  ┌─────────────────┐                                           │
│  │ ▓▓░░▒▒░░▓▓▒▒    │  Malware de la misma familia              │
│  │ ░░▓▓▒▒▓▓░░▒▒    │  tiene patrones SIMILARES                 │
│  │ ▒▒░░▓▓░░▒▒▓▓    │                                           │
│  └─────────────────┘                                           │
│                                                                │
│  ¿POR QUÉ FUNCIONA?                                            │
│  - Estructura PE tiene secciones distintivas                   │
│  - Código malicioso tiene patrones repetitivos                 │
│  - Packing/ofuscación crea texturas características            │
│  - Familias de malware comparten código → patrones similares   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Implementación Completa

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

# ============================================================
# CONVERTIR BINARIO A IMAGEN
# ============================================================
def binary_to_image(filepath, image_size=256):
    """Convierte archivo binario a imagen en escala de grises"""
    with open(filepath, 'rb') as f:
        content = f.read()

    # Convertir bytes a array numpy
    byte_array = np.frombuffer(content, dtype=np.uint8)

    # Calcular dimensiones
    total_pixels = image_size * image_size

    if len(byte_array) < total_pixels:
        # Padding con ceros si es muy pequeño
        byte_array = np.pad(byte_array, (0, total_pixels - len(byte_array)))
    else:
        # Truncar si es muy grande
        byte_array = byte_array[:total_pixels]

    # Reshape a imagen 2D
    image = byte_array.reshape((image_size, image_size))

    return image

# ============================================================
# DATASET
# ============================================================
class MalwareImageDataset(Dataset):
    def __init__(self, file_paths, labels, image_size=256, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Convertir binario a imagen
        image = binary_to_image(self.file_paths[idx], self.image_size)

        # Normalizar a [0, 1]
        image = image.astype(np.float32) / 255.0

        # Añadir canal (grayscale → 1 canal)
        image = np.expand_dims(image, axis=0)

        # Convertir a tensor
        image = torch.FloatTensor(image)

        if self.transform:
            image = self.transform(image)

        label = torch.LongTensor([self.labels[idx]])[0]

        return image, label

# ============================================================
# MODELO CNN PARA MALWARE
# ============================================================
class MalwareCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MalwareCNN, self).__init__()

        # Encoder (extracción de features)
        self.features = nn.Sequential(
            # Bloque 1: 256 → 128
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Bloque 2: 128 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Bloque 3: 64 → 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Bloque 4: 32 → 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ============================================================
# ENTRENAMIENTO
# ============================================================
def train_malware_classifier():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando: {device}")

    # Crear modelo
    model = MalwareCNN(num_classes=2).to(device)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Simular datos (en producción, cargar archivos reales)
    # file_paths = [...lista de paths a ejecutables...]
    # labels = [...0 para benigno, 1 para malware...]

    # Para demostración, crear datos sintéticos
    num_samples = 1000
    X = torch.randn(num_samples, 1, 256, 256)
    y = torch.randint(0, 2, (num_samples,))

    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1:2d}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

        scheduler.step(val_loss)

    return model

# Ejecutar
# model = train_malware_classifier()
```

## 7. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE CNN                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Excelente para datos con estructura espacial (imágenes)     │
│  ✓ Menos parámetros que MLP para imágenes                      │
│  ✓ Invariancia a traslación                                    │
│  ✓ Captura patrones locales y jerárquicos                      │
│  ✓ Transfer learning muy efectivo                              │
│  ✓ Estado del arte en visión por computador                    │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE CNN                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ Requiere mucho cómputo (GPU recomendada)                    │
│  ✗ Necesita muchos datos sin transfer learning                 │
│  ✗ No maneja bien datos de longitud variable                   │
│  ✗ No captura dependencias de largo alcance                    │
│  ✗ Sensible a rotaciones (sin data augmentation)               │
│  ✗ "Caja negra" - difícil de interpretar                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  CNN - RESUMEN                                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  COMPONENTES:                                                  │
│    • Conv2D: extrae features locales con filtros               │
│    • MaxPool: reduce dimensiones, invariancia                  │
│    • BatchNorm: estabiliza entrenamiento                       │
│    • Dropout: regularización                                   │
│    • Dense/FC: clasificación final                             │
│                                                                │
│  ARQUITECTURA TÍPICA:                                          │
│    [Conv → BN → ReLU → Pool] × N → GAP → FC → Softmax          │
│                                                                │
│  TRANSFER LEARNING:                                            │
│    • Usar modelos pre-entrenados (ResNet, VGG, etc.)           │
│    • Feature extraction para pocos datos                       │
│    • Fine-tuning para más datos                                │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Clasificación de malware (binario → imagen)               │
│    • Análisis de capturas de pantalla (phishing)               │
│    • Detección visual de anomalías                             │
│    • Análisis de tráfico (como imagen temporal)                │
│                                                                │
│  HIPERPARÁMETROS:                                              │
│    • Filtros: 32 → 64 → 128 → 256 (incrementar)               │
│    • Kernel: 3×3 (más común)                                   │
│    • Pooling: 2×2 con stride 2                                 │
│    • Dropout: 0.25-0.5                                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Redes Recurrentes (RNN)
