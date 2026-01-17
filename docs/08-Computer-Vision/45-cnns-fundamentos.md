# Redes Neuronales Convolucionales (CNNs) - Fundamentos

## Introducción

Las CNNs son arquitecturas de deep learning diseñadas específicamente para procesar datos con estructura de grilla, como imágenes. Inspiradas en el córtex visual, aprenden jerárquicamente: capas iniciales detectan bordes, capas medias detectan texturas y patrones, capas finales detectan objetos completos.

```
Jerarquía de Características:
┌─────────────────────────────────────────────────────────┐
│  Capa 1        Capa 2         Capa 3         Capa N    │
│  ┌───┐         ┌───┐          ┌───┐          ┌───┐     │
│  │ / │   →     │░░░│    →     │👁️ │    →     │🐱│     │
│  │ \ │         │▓▓▓│          │👃│          │🐕│     │
│  └───┘         └───┘          └───┘          └───┘     │
│  Bordes       Texturas      Partes        Objetos     │
│              Patrones      Features      Completos    │
└─────────────────────────────────────────────────────────┘
```

## Operación de Convolución

### Convolución 2D

```
Input (5×5)              Kernel (3×3)            Output (3×3)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┐         ┌───┬───┬───┐
│ 1 │ 0 │ 1 │ 0 │ 1 │    │ 1 │ 0 │ 1 │         │   │   │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤    =    ├───┼───┼───┤
│ 0 │ 1 │ 0 │ 1 │ 0 │  * │ 0 │ 1 │ 0 │         │   │   │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┤         ├───┼───┼───┤
│ 1 │ 0 │ 1 │ 0 │ 1 │    │ 1 │ 0 │ 1 │         │   │   │   │
├───┼───┼───┼───┼───┤    └───┴───┴───┘         └───┴───┴───┘
│ 0 │ 1 │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤    Stride = 1
│ 1 │ 0 │ 1 │ 0 │ 1 │    Padding = 0 (valid)
└───┴───┴───┴───┴───┘

Cálculo de un elemento:
(1×1 + 0×0 + 1×1) +
(0×0 + 1×1 + 0×0) +      = 1+1+1+1+1 = 5
(1×1 + 0×0 + 1×1)
```

### Implementación Manual

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def conv2d_manual(
    input_image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    Implementación manual de convolución 2D.
    Para entender qué hace la operación internamente.
    """
    # Añadir padding
    if padding > 0:
        input_image = np.pad(
            input_image,
            ((padding, padding), (padding, padding)),
            mode='constant'
        )

    h_in, w_in = input_image.shape
    h_k, w_k = kernel.shape

    # Calcular dimensiones de salida
    h_out = (h_in - h_k) // stride + 1
    w_out = (w_in - w_k) // stride + 1

    output = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            # Extraer región
            region = input_image[
                i*stride : i*stride + h_k,
                j*stride : j*stride + w_k
            ]
            # Producto elemento a elemento y suma
            output[i, j] = np.sum(region * kernel)

    return output


# Ejemplo
image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.float32)

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

result = conv2d_manual(image, edge_kernel, stride=1, padding=0)
print(f"Output shape: {result.shape}")  # (2, 2)
```

### Parámetros Clave

```python
class ConvolutionParameters:
    """
    Parámetros que controlan la convolución.
    """

    @staticmethod
    def output_size(
        input_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1
    ) -> int:
        """
        Calcula tamaño de salida.

        Fórmula:
        output = floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
        """
        effective_kernel = dilation * (kernel_size - 1) + 1
        return (input_size + 2 * padding - effective_kernel) // stride + 1

    @staticmethod
    def same_padding(kernel_size: int, dilation: int = 1) -> int:
        """Calcula padding para mantener dimensiones (same padding)."""
        effective_kernel = dilation * (kernel_size - 1) + 1
        return (effective_kernel - 1) // 2


# Ejemplos de configuraciones comunes
configs = {
    "reduce_by_2": {"kernel": 3, "stride": 2, "padding": 1},
    "same_size": {"kernel": 3, "stride": 1, "padding": 1},
    "aggressive_reduce": {"kernel": 7, "stride": 2, "padding": 3},
}

calc = ConvolutionParameters()
for name, cfg in configs.items():
    out = calc.output_size(224, cfg["kernel"], cfg["stride"], cfg["padding"])
    print(f"{name}: 224 → {out}")
```

## Capas de una CNN

### Capa Convolucional

```python
class ConvLayer(nn.Module):
    """
    Capa convolucional con batch norm y activación.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()

        # Convolución
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn  # Si hay BN, no necesitamos bias
        )

        # Batch Normalization
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Activación
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),  # Swish
            "none": nn.Identity()
        }
        self.activation = activations.get(activation, nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Conv → BN → Activation

        Args:
            x: (batch, channels, height, width)

        Returns:
            (batch, out_channels, new_height, new_width)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# Ejemplo
layer = ConvLayer(3, 64, kernel_size=3, stride=1, padding=1)
input_tensor = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
output = layer(input_tensor)
print(f"Input:  {input_tensor.shape}")  # [1, 3, 224, 224]
print(f"Output: {output.shape}")         # [1, 64, 224, 224]
```

### Capa de Pooling

```python
class PoolingLayers:
    """
    Operaciones de pooling para reducir dimensiones espaciales.
    """

    @staticmethod
    def max_pool_example():
        """
        Max Pooling: Toma el máximo de cada región.
        Ventaja: Preserva features más prominentes.
        """
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Ejemplo visual
        x = torch.tensor([[
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        ]], dtype=torch.float32)

        output = pool(x)
        # [[[ 6,  8],
        #   [14, 16]]]
        return output

    @staticmethod
    def avg_pool_example():
        """
        Average Pooling: Promedio de cada región.
        Ventaja: Más suave, menos agresivo.
        """
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        return pool

    @staticmethod
    def global_avg_pool():
        """
        Global Average Pooling: Reduce a 1×1.
        Usado al final de CNNs modernas en lugar de Flatten+Dense.
        """
        pool = nn.AdaptiveAvgPool2d(1)

        x = torch.randn(1, 512, 7, 7)  # Feature map típico
        output = pool(x)  # [1, 512, 1, 1]
        output = output.view(output.size(0), -1)  # [1, 512]
        return output


# Comparación visual
"""
Max Pooling 2×2:
┌───┬───┬───┬───┐     ┌───┬───┐
│ 1 │ 2 │ 3 │ 4 │     │ 6 │ 8 │
├───┼───┼───┼───┤  →  ├───┼───┤
│ 5 │ 6 │ 7 │ 8 │     │14 │16 │
├───┼───┼───┼───┤     └───┴───┘
│ 9 │10 │11 │12 │
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘

Average Pooling 2×2:
┌───┬───┬───┬───┐     ┌─────┬─────┐
│ 1 │ 2 │ 3 │ 4 │     │ 3.5 │ 5.5 │
├───┼───┼───┼───┤  →  ├─────┼─────┤
│ 5 │ 6 │ 7 │ 8 │     │11.5 │13.5 │
├───┼───┼───┼───┤     └─────┴─────┘
│ 9 │10 │11 │12 │
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘
"""
```

### Funciones de Activación

```python
class Activations:
    """Funciones de activación para CNNs."""

    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """
        ReLU: max(0, x)
        - Simple y eficiente
        - Problema: "dying ReLU" (neuronas que nunca se activan)
        """
        return F.relu(x)

    @staticmethod
    def leaky_relu(x: torch.Tensor, slope: float = 0.01) -> torch.Tensor:
        """
        Leaky ReLU: x if x > 0 else slope * x
        - Evita dying ReLU
        - Pendiente negativa configurable
        """
        return F.leaky_relu(x, slope)

    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        """
        GELU: Gaussian Error Linear Unit
        - Suave, diferenciable en todo punto
        - Usado en Transformers (BERT, GPT)
        """
        return F.gelu(x)

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        """
        Swish/SiLU: x * sigmoid(x)
        - Auto-gated
        - Mejor que ReLU en redes profundas
        """
        return F.silu(x)


# Visualización de activaciones
"""
        ReLU                    Leaky ReLU
    y │      /              y │      /
      │     /                 │     /
      │    /                  │    /
──────┼───/────── x       ────┼───/────── x
      │  /                   /│  /
      │ /                   / │ /
      │/                   /  │/

        GELU                    Swish
    y │      /              y │      /
      │    _/                 │    _/
      │  _/                   │  _/
──────┼_/──────── x       ────┼_/──────── x
      │                     _/│
      │                    /  │
"""
```

## Arquitectura Básica de CNN

### CNN Simple para Clasificación

```python
class SimpleCNN(nn.Module):
    """
    CNN básica para clasificación de imágenes.
    Arquitectura: Conv blocks → Global Pool → FC
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3
    ):
        super().__init__()

        # Bloque 1: 3 → 32 canales
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 → 112
        )

        # Bloque 2: 32 → 64 canales
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 → 56
        )

        # Bloque 3: 64 → 128 canales
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 → 28
        )

        # Bloque 4: 128 → 256 canales
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # → 1×1
        )

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Extrae feature maps intermedios para visualización."""
        features = {}
        features['input'] = x

        x = self.block1(x)
        features['block1'] = x

        x = self.block2(x)
        features['block2'] = x

        x = self.block3(x)
        features['block3'] = x

        x = self.block4(x)
        features['block4'] = x

        return features


# Resumen de arquitectura
model = SimpleCNN(num_classes=10)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass de prueba
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 10]
```

### Flujo de Dimensiones

```
Input:  [batch, 3, 224, 224]
           │
           ▼
Block 1:  [batch, 32, 112, 112]  ← MaxPool reduce 2×
           │
           ▼
Block 2:  [batch, 64, 56, 56]    ← MaxPool reduce 2×
           │
           ▼
Block 3:  [batch, 128, 28, 28]   ← MaxPool reduce 2×
           │
           ▼
Block 4:  [batch, 256, 1, 1]     ← Global Avg Pool
           │
           ▼
Flatten:  [batch, 256]
           │
           ▼
FC:       [batch, num_classes]
```

## Batch Normalization

### Concepto y Beneficios

```python
class BatchNormExplained:
    """
    Batch Normalization normaliza activaciones por batch.

    Beneficios:
    - Acelera entrenamiento
    - Permite learning rates más altos
    - Reduce sensibilidad a inicialización
    - Tiene efecto regularizador
    """

    @staticmethod
    def manual_batch_norm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """
        Implementación manual para entender BN.

        Para cada canal:
        1. Calcular media y varianza del batch
        2. Normalizar: (x - mean) / sqrt(var + eps)
        3. Escalar y trasladar: gamma * x_norm + beta
        """
        # x shape: [batch, channels, height, width]
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + eps)

        # gamma y beta tienen shape [channels]
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)

        return gamma * x_norm + beta


# Diferencia entre training y eval
"""
Training:
- Usa estadísticas del batch actual
- Actualiza running_mean y running_var

Eval (model.eval()):
- Usa running_mean y running_var acumulados
- Comportamiento determinístico
"""

# Ejemplo de uso correcto
model = SimpleCNN(num_classes=10)

# Entrenamiento
model.train()
for images, labels in train_loader:
    outputs = model(images)
    # BN usa estadísticas del batch

# Evaluación
model.eval()
with torch.no_grad():
    outputs = model(test_images)
    # BN usa running statistics
```

## Dropout en CNNs

### Regularización Espacial

```python
class DropoutTypes:
    """Tipos de dropout para CNNs."""

    @staticmethod
    def standard_dropout():
        """
        Dropout estándar: Desactiva neuronas aleatorias.
        Típicamente usado después de FC layers.
        """
        return nn.Dropout(p=0.5)

    @staticmethod
    def dropout2d():
        """
        Dropout2D: Desactiva canales completos.
        Más apropiado para feature maps convolucionales.
        """
        return nn.Dropout2d(p=0.2)

    @staticmethod
    def dropblock():
        """
        DropBlock: Desactiva regiones contiguas.
        Más efectivo que dropout estándar para CNNs.
        """
        # Implementación simplificada
        class DropBlock2D(nn.Module):
            def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
                super().__init__()
                self.drop_prob = drop_prob
                self.block_size = block_size

            def forward(self, x):
                if not self.training or self.drop_prob == 0:
                    return x

                # Calcular gamma para tener drop_prob efectivo
                gamma = self.drop_prob / (self.block_size ** 2)

                mask = (torch.rand_like(x[:, :1, :, :]) < gamma).float()

                # Expandir máscara con max pooling
                mask = F.max_pool2d(
                    mask,
                    kernel_size=self.block_size,
                    stride=1,
                    padding=self.block_size // 2
                )

                mask = 1 - mask
                mask = mask.expand_as(x)

                # Normalizar
                return x * mask * mask.numel() / mask.sum()

        return DropBlock2D()


# Visualización
"""
Standard Dropout:        Dropout2D:           DropBlock:
┌─────────────────┐      ┌─────────────────┐   ┌─────────────────┐
│ x x . x . x x x │      │ x x x x x x x x │   │ x x . . . x x x │
│ x . x x x . x x │      │ . . . . . . . . │   │ x x . . . x x x │
│ . x x . x x x . │  vs  │ x x x x x x x x │   │ x x . . . x x x │
│ x x . x x x . x │      │ x x x x x x x x │   │ x x x x x x x x │
└─────────────────┘      └─────────────────┘   └─────────────────┘
  Pixels aleatorios       Canales completos     Regiones contiguas
```

## Receptive Field

### Concepto

```python
class ReceptiveField:
    """
    Receptive Field: Región de la imagen de entrada
    que influye en un píxel del feature map.

    Importante para entender qué "ve" cada neurona.
    """

    @staticmethod
    def calculate_rf(
        layers: list,
        input_size: int = 224
    ) -> list:
        """
        Calcula receptive field acumulativo.

        layers: Lista de dicts con 'kernel', 'stride', 'padding'
        """
        rf = 1
        stride_product = 1

        rf_history = [rf]

        for layer in layers:
            k = layer['kernel']
            s = layer['stride']

            # RF crece con cada capa
            rf = rf + (k - 1) * stride_product
            stride_product *= s

            rf_history.append(rf)

        return rf_history


# Ejemplo: VGG-like network
vgg_layers = [
    {'kernel': 3, 'stride': 1},   # Conv
    {'kernel': 3, 'stride': 1},   # Conv
    {'kernel': 2, 'stride': 2},   # Pool
    {'kernel': 3, 'stride': 1},   # Conv
    {'kernel': 3, 'stride': 1},   # Conv
    {'kernel': 2, 'stride': 2},   # Pool
]

rf_calc = ReceptiveField()
rf_history = rf_calc.calculate_rf(vgg_layers)
print("Receptive field por capa:", rf_history)
# [1, 3, 5, 6, 10, 14, 16]

"""
Visualización:
Capa 1 (3×3 conv): RF = 3×3
                   ┌───┬───┬───┐
                   │ ● │ ● │ ● │
                   ├───┼───┼───┤
                   │ ● │ ★ │ ● │  ← Un píxel del output
                   ├───┼───┼───┤     ve estos 9 píxeles
                   │ ● │ ● │ ● │
                   └───┴───┴───┘

Después de pool: RF crece
                   ┌───┬───┬───┬───┬───┬───┐
                   │ ● │ ● │ ● │ ● │ ● │ ● │
                   ├───┼───┼───┼───┼───┼───┤
                   │ ● │ ● │ ● │ ● │ ● │ ● │
                   ├───┼───┼───┼───┼───┼───┤
                   │ ● │ ● │ ★ │ ● │ ● │ ● │
                   ...
"""
```

## Entrenamiento de CNNs

### Loop de Entrenamiento Completo

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class CNNTrainer:
    """Trainer para CNNs con mejores prácticas."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss y optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=100,
            steps_per_epoch=len(train_loader)
        )

    def train_epoch(self) -> Tuple[float, float]:
        """Entrena una época."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()

            # Gradient clipping (evita explosión de gradientes)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Métricas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        return running_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Evalúa en validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / len(self.val_loader), correct / total

    def fit(self, epochs: int, patience: int = 10):
        """Entrenamiento completo con early stopping."""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Cargar mejor modelo
        self.model.load_state_dict(torch.load("best_model.pth"))
        return best_val_acc
```

## Visualización de Features

### Visualizar Filtros y Activaciones

```python
import matplotlib.pyplot as plt


class CNNVisualizer:
    """Herramientas de visualización para CNNs."""

    @staticmethod
    def visualize_filters(model: nn.Module, layer_name: str = "block1"):
        """Visualiza filtros de primera capa convolucional."""
        # Obtener pesos de primera conv
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights = module.weight.data.cpu()
                break

        # Normalizar para visualización
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # Plot
        n_filters = min(weights.shape[0], 64)
        n_cols = 8
        n_rows = n_filters // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < n_filters:
                # Para RGB, mostrar como imagen color
                if weights.shape[1] == 3:
                    ax.imshow(weights[i].permute(1, 2, 0))
                else:
                    ax.imshow(weights[i, 0], cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        return fig

    @staticmethod
    def visualize_feature_maps(
        model: nn.Module,
        image: torch.Tensor,
        layer_idx: int = 0
    ):
        """Visualiza feature maps de una capa."""
        # Hook para capturar activaciones
        activations = []

        def hook(module, input, output):
            activations.append(output.detach())

        # Registrar hook en capa específica
        layer_count = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if layer_count == layer_idx:
                    module.register_forward_hook(hook)
                    break
                layer_count += 1

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(image.unsqueeze(0))

        # Visualizar
        if activations:
            act = activations[0][0]  # Primer batch
            n_channels = min(act.shape[0], 64)
            n_cols = 8
            n_rows = n_channels // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < n_channels:
                    ax.imshow(act[i].cpu(), cmap='viridis')
                ax.axis('off')

            plt.tight_layout()
            return fig

        return None
```

## Aplicaciones en Ciberseguridad

### Clasificador de Capturas Maliciosas

```python
class MaliciousScreenshotClassifier(nn.Module):
    """
    CNN para detectar capturas de pantalla maliciosas.
    Ej: Phishing, fake login, credential harvesting.
    """

    def __init__(self, num_classes: int = 4):
        """
        Classes:
        0: Legítimo
        1: Phishing
        2: Fake login
        3: Malware warning falso
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def predict_with_confidence(
        self,
        image: torch.Tensor
    ) -> Tuple[int, float, dict]:
        """Predicción con scores de confianza."""
        self.eval()
        with torch.no_grad():
            logits = self(image.unsqueeze(0))
            probs = F.softmax(logits, dim=1)[0]

            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

            class_names = ["Legítimo", "Phishing", "Fake Login", "Malware Falso"]
            all_probs = {
                name: probs[i].item()
                for i, name in enumerate(class_names)
            }

        return pred_class, confidence, all_probs
```

## Resumen

| Componente | Función | Parámetros Típicos |
|------------|---------|-------------------|
| Conv2d | Extrae features locales | kernel=3×3, stride=1 |
| BatchNorm | Normaliza activaciones | momentum=0.1 |
| ReLU | No-linealidad | inplace=True |
| MaxPool | Reduce dimensiones | kernel=2×2, stride=2 |
| Dropout | Regularización | p=0.5 (FC), p=0.2 (conv) |
| GlobalAvgPool | Colapsa espacial | output=1×1 |

### Checklist CNN

```
□ Input normalizado (ImageNet stats si transfer learning)
□ Batch Normalization después de Conv, antes de ReLU
□ Dropout2D para capas conv, Dropout estándar para FC
□ Learning rate scheduler (OneCycleLR, CosineAnnealing)
□ Data augmentation en training, NO en val/test
□ model.train() / model.eval() correctamente usado
□ Gradient clipping si hay inestabilidad
```

## Referencias

- Deep Learning (Goodfellow, Bengio, Courville) - Chapter 9
- CS231n: Convolutional Neural Networks for Visual Recognition
- PyTorch Documentation: https://pytorch.org/docs/
