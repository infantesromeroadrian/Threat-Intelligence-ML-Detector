# Arquitecturas CNN Clásicas

## Evolución de las Arquitecturas

```
Timeline de Arquitecturas CNN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1998     2012      2014      2015       2016       2017       2019
LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → EfficientNet
                   │       v1         │                      │
                   │                  │                      │
               "Profundidad"    "Skip Connections"    "Compound Scaling"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ImageNet Top-5 Error:
LeNet (n/a) → AlexNet (16.4%) → VGG (7.3%) → ResNet (3.6%) → EfficientNet (2.9%)
```

## LeNet-5 (1998)

### Arquitectura Original

```
LeNet-5 (Yann LeCun)
Primera CNN exitosa para reconocimiento de dígitos (MNIST)

Input: 32×32×1 (grayscale)
┌──────────────────────────────────────────────────────────────┐
│ Conv 5×5   → 6 filtros   → 28×28×6   → AvgPool 2×2 → 14×14×6 │
│ Conv 5×5   → 16 filtros  → 10×10×16  → AvgPool 2×2 → 5×5×16  │
│ Flatten    → 400                                              │
│ FC 400→120 → FC 120→84  → FC 84→10                           │
└──────────────────────────────────────────────────────────────┘
Output: 10 clases (dígitos 0-9)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5: Arquitectura pionera de CNNs.
    Características:
    - Convoluciones pequeñas (5×5)
    - Average pooling (no max)
    - Activación Tanh (no ReLU)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # Classifier
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 32×32×1 → 28×28×6
        x = torch.tanh(self.conv1(x))
        # 28×28×6 → 14×14×6
        x = F.avg_pool2d(x, 2)

        # 14×14×6 → 10×10×16
        x = torch.tanh(self.conv2(x))
        # 10×10×16 → 5×5×16
        x = F.avg_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x


# Parámetros: ~60K (muy pequeño para estándares modernos)
model = LeNet5()
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
```

## AlexNet (2012)

### Arquitectura

```
AlexNet (Krizhevsky, Sutskever, Hinton)
Ganador ImageNet 2012 - Inició la era del Deep Learning

Input: 227×227×3 (RGB)
┌─────────────────────────────────────────────────────────────────────┐
│ Conv 11×11, s=4 → 96 filtros  → 55×55×96   → MaxPool 3×3, s=2      │
│ Conv 5×5, p=2   → 256 filtros → 27×27×256  → MaxPool 3×3, s=2      │
│ Conv 3×3, p=1   → 384 filtros → 13×13×384                          │
│ Conv 3×3, p=1   → 384 filtros → 13×13×384                          │
│ Conv 3×3, p=1   → 256 filtros → 13×13×256  → MaxPool 3×3, s=2      │
│ Flatten → 9216 → FC 4096 → FC 4096 → FC 1000                       │
└─────────────────────────────────────────────────────────────────────┘

Innovaciones:
- ReLU en lugar de Tanh (más rápido)
- Dropout para regularización
- Data augmentation
- GPU training (2 GPUs)
- Local Response Normalization (LRN) - obsoleto
```

```python
class AlexNet(nn.Module):
    """
    AlexNet: Primera CNN profunda exitosa.

    Innovaciones clave:
    - ReLU activation (5× más rápido que tanh)
    - Dropout regularization (0.5)
    - Data augmentation extensivo
    - Entrenamiento en GPU
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 227×227×3 → 55×55×96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 27×27×96 → 27×27×256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 13×13×256 → 13×13×384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 13×13×384 → 13×13×384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 13×13×384 → 13×13×256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# Parámetros: ~60M (principalmente en FC layers)
model = AlexNet()
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
```

## VGG (2014)

### Filosofía: Simplicidad y Profundidad

```
VGG (Visual Geometry Group, Oxford)
"Muy profundo con filtros muy pequeños"

Insight clave:
2 conv 3×3 = receptive field de 5×5, pero menos parámetros
3 conv 3×3 = receptive field de 7×7, pero menos parámetros

Comparación:
1 conv 7×7: 7×7×C×C = 49C² parámetros
3 conv 3×3: 3×(3×3×C×C) = 27C² parámetros  ← 45% menos!
```

```
VGG-16 Architecture:
┌────────────────────────────────────────────────────────────────────┐
│ Block 1: 2× Conv3-64   → MaxPool  → 224×224×3 → 112×112×64        │
│ Block 2: 2× Conv3-128  → MaxPool  → 112×112×64 → 56×56×128        │
│ Block 3: 3× Conv3-256  → MaxPool  → 56×56×128 → 28×28×256         │
│ Block 4: 3× Conv3-512  → MaxPool  → 28×28×256 → 14×14×512         │
│ Block 5: 3× Conv3-512  → MaxPool  → 14×14×512 → 7×7×512           │
│ Flatten → FC 4096 → FC 4096 → FC 1000                              │
└────────────────────────────────────────────────────────────────────┘
Total: 16 capas con pesos (13 conv + 3 FC)
```

```python
class VGGBlock(nn.Module):
    """Bloque VGG: N convoluciones + MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int
    ):
        super().__init__()

        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))  # Añadido (no en original)
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG16(nn.Module):
    """
    VGG-16: 16 capas con pesos.

    Configuración: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                    512, 512, 512, 'M', 512, 512, 512, 'M']
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            VGGBlock(3, 64, num_convs=2),      # 224 → 112
            VGGBlock(64, 128, num_convs=2),    # 112 → 56
            VGGBlock(128, 256, num_convs=3),   # 56 → 28
            VGGBlock(256, 512, num_convs=3),   # 28 → 14
            VGGBlock(512, 512, num_convs=3),   # 14 → 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# VGG-16: ~138M parámetros
# VGG-19: ~144M parámetros
model = VGG16()
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
```

## Inception / GoogLeNet (2014)

### Módulo Inception

```
Inception Module: "Let the network decide"

En lugar de elegir kernel 1×1, 3×3 o 5×5,
usar TODOS en paralelo y concatenar.

                    Input
                      │
    ┌────────┬────────┼────────┬────────┐
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
  Conv     Conv     Conv     Conv    MaxPool
  1×1      1×1      1×1      1×1      3×3
    │        │        │        │        │
    │        ▼        ▼        │        ▼
    │      Conv     Conv      │      Conv
    │      3×3      5×5       │      1×1
    │        │        │        │        │
    └────────┴────────┴────────┴────────┘
                      │
                      ▼
              Filter Concatenation

Las conv 1×1 antes de 3×3/5×5 reducen dimensionalidad
(bottleneck) y reducen parámetros dramáticamente.
```

```python
class InceptionModule(nn.Module):
    """
    Módulo Inception con bottleneck.

    Procesa input con múltiples tamaños de kernel en paralelo
    y concatena los resultados.
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int
    ):
        super().__init__()

        # Branch 1: 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1×1 reduce → 3×3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1×1 reduce → 5×5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Branch 4: MaxPool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenar en dimensión de canales
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


# Ejemplo de uso
inception = InceptionModule(
    in_channels=192,
    ch1x1=64,
    ch3x3_reduce=96, ch3x3=128,
    ch5x5_reduce=16, ch5x5=32,
    pool_proj=32
)

x = torch.randn(1, 192, 28, 28)
out = inception(x)
print(f"Output channels: {out.shape[1]}")  # 64+128+32+32 = 256
```

## ResNet (2015)

### Conexiones Residuales

```
Problema de Degradación:
Redes muy profundas son PEORES que redes menos profundas.
No es overfitting (train error también sube).

Solución: Skip Connections (Residual Learning)

                    ┌───────────────────┐
                    │                   │
        x ─────────►│     Identity      │───────►
                    │                   │        │
                    └───────────────────┘        │
                                                 │ +
                    ┌───────────────────┐        │
                    │                   │        │
        x ─────────►│   F(x) = Conv     │────────┘───► y = F(x) + x
                    │   blocks          │
                    └───────────────────┘

Intuición:
- Si la capa óptima es identidad, es más fácil aprender F(x)=0
- Gradientes fluyen directamente por skip connection
- Permite entrenar redes de 100+ capas
```

```python
class BasicBlock(nn.Module):
    """
    Bloque básico de ResNet (ResNet-18/34).
    Dos convs 3×3 con skip connection.
    """
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Ajustar dimensiones si es necesario
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bloque Bottleneck de ResNet (ResNet-50/101/152).
    1×1 reduce → 3×3 → 1×1 expand

    Reduce parámetros usando bottleneck design:
    256 → 64 → 64 → 256 en lugar de 256 → 256 → 256
    """
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super().__init__()

        # 1×1 reduce
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3×3
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 expand
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet genérico.

    Configuraciones:
    - ResNet-18:  [2, 2, 2, 2], BasicBlock
    - ResNet-34:  [3, 4, 6, 3], BasicBlock
    - ResNet-50:  [3, 4, 6, 3], Bottleneck
    - ResNet-101: [3, 4, 23, 3], Bottleneck
    - ResNet-152: [3, 8, 36, 3], Bottleneck
    """

    def __init__(
        self,
        block: type,
        layers: list,
        num_classes: int = 1000
    ):
        super().__init__()

        self.in_channels = 64

        # Stem: 7×7 conv + maxpool
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Inicialización
        self._initialize_weights()

    def _make_layer(
        self,
        block: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        # Downsample si cambian dimensiones
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# Comparación de parámetros
models = {
    "ResNet-18": resnet18(),
    "ResNet-50": resnet50(),
    "ResNet-101": resnet101()
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parámetros")
```

## DenseNet (2017)

### Conexiones Densas

```
DenseNet: "Dense Connections"

En lugar de x + F(x), concatena TODAS las features anteriores.

                    Dense Block
    ┌─────────────────────────────────────────┐
    │                                         │
x₀ ─┼──┬──────┬──────┬──────────────────────►│
    │  │      │      │                        │
    │  ▼      │      │                        │
    │ H₁ ────►├──────┼──────────────────────►│
    │         │      │                        │
    │         ▼      │                        │
    │        H₂ ────►├──────────────────────►│
    │                │                        │
    │                ▼                        │
    │               H₃ ────────────────────►│
    │                                         │
    └─────────────────────────────────────────┘
                     │
                     ▼
           [x₀, x₁, x₂, x₃] concatenados

Ventajas:
- Reutilización de features
- Gradientes más fuertes
- Menos parámetros que ResNet
- Feature propagation eficiente
```

```python
class DenseLayer(nn.Module):
    """Una capa dentro de un Dense Block."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4
    ):
        super().__init__()

        # Bottleneck: BN-ReLU-Conv1×1-BN-ReLU-Conv3×3
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    """Bloque denso con múltiples capas."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(
                in_channels + i * growth_rate,
                growth_rate
            ))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    """Capa de transición entre Dense Blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseNet(nn.Module):
    """
    DenseNet con Dense Blocks y Transition layers.

    Configuraciones:
    - DenseNet-121: [6, 12, 24, 16], growth_rate=32
    - DenseNet-169: [6, 12, 32, 32], growth_rate=32
    - DenseNet-201: [6, 12, 48, 32], growth_rate=32
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        num_classes: int = 1000
    ):
        super().__init__()

        # Stem
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Final BN
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Classifier
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes: int = 1000) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64, num_classes)
```

## EfficientNet (2019)

### Compound Scaling

```
EfficientNet: "Scaling Networks Efficiently"

Problema: ¿Cómo escalar una red (más profunda, más ancha, más resolución)?

Métodos tradicionales:
- Width scaling: Más canales (ResNet-Wide)
- Depth scaling: Más capas (ResNet-18 → ResNet-152)
- Resolution scaling: Mayor resolución de entrada

EfficientNet: Escalar las TRES dimensiones de forma balanceada

Compound Scaling:
depth = α^φ
width = β^φ
resolution = γ^φ

donde α × β² × γ² ≈ 2 (para mantener FLOPS ≈ 2^φ)

EfficientNet-B0 (baseline) → B1 → B2 → ... → B7
```

```python
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv (MBConv).
    Usado en EfficientNet y MobileNetV2.

    Estructura: Expand → Depthwise → SE → Project
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25
    ):
        super().__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expand (si expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)  # Swish
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size,
                stride=stride, padding=kernel_size // 2,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        layers.append(SqueezeExcitation(hidden_dim, se_channels))

        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


# Para usar EfficientNet en práctica, mejor usar torchvision
from torchvision.models import efficientnet_b0, efficientnet_b4

model_b0 = efficientnet_b0(weights='IMAGENET1K_V1')
model_b4 = efficientnet_b4(weights='IMAGENET1K_V1')

print(f"EfficientNet-B0: {sum(p.numel() for p in model_b0.parameters()):,}")
print(f"EfficientNet-B4: {sum(p.numel() for p in model_b4.parameters()):,}")
```

## Comparativa de Arquitecturas

```
┌─────────────────┬───────────┬───────────┬──────────────┬───────────┐
│ Arquitectura    │ Params    │ Top-1 Acc │ Input Size   │ Año       │
├─────────────────┼───────────┼───────────┼──────────────┼───────────┤
│ LeNet-5         │ 60K       │ N/A       │ 32×32        │ 1998      │
│ AlexNet         │ 60M       │ 56.5%     │ 227×227      │ 2012      │
│ VGG-16          │ 138M      │ 71.6%     │ 224×224      │ 2014      │
│ GoogLeNet       │ 6.8M      │ 74.8%     │ 224×224      │ 2014      │
│ ResNet-50       │ 25M       │ 76.1%     │ 224×224      │ 2015      │
│ ResNet-152      │ 60M       │ 77.8%     │ 224×224      │ 2015      │
│ DenseNet-121    │ 8M        │ 74.4%     │ 224×224      │ 2017      │
│ EfficientNet-B0 │ 5.3M      │ 77.1%     │ 224×224      │ 2019      │
│ EfficientNet-B7 │ 66M       │ 84.3%     │ 600×600      │ 2019      │
└─────────────────┴───────────┴───────────┴──────────────┴───────────┘
```

## Uso con torchvision

```python
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    VGG16_Weights
)


def load_pretrained_models():
    """Carga modelos preentrenados de torchvision."""

    # Método moderno (recomendado)
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    return {
        'resnet50': resnet,
        'efficientnet_b0': efficientnet,
        'vgg16': vgg
    }


def modify_for_custom_classes(
    model: nn.Module,
    model_name: str,
    num_classes: int
) -> nn.Module:
    """Modifica clasificador final para número custom de clases."""

    if 'resnet' in model_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif 'efficientnet' in model_name:
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif 'densenet' in model_name:
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )

    return model


# Ejemplo
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model = modify_for_custom_classes(model, 'resnet50', num_classes=10)
```

## Aplicaciones en Ciberseguridad

### Detector de Capturas Maliciosas

```python
class SecurityScreenshotClassifier:
    """
    Clasificador basado en arquitecturas clásicas
    para detectar capturas de pantalla maliciosas.
    """

    def __init__(self, num_classes: int = 5):
        """
        Classes:
        0: Legítimo
        1: Phishing
        2: Fake login
        3: Scareware
        4: Ransomware screen
        """
        # Usar EfficientNet por balance eficiencia/accuracy
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

        self.class_names = [
            "Legítimo", "Phishing", "Fake Login",
            "Scareware", "Ransomware"
        ]

    def predict(self, image: torch.Tensor) -> dict:
        """Predicción con explicación."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0))
            probs = F.softmax(logits, dim=1)[0]

            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

            return {
                "prediction": self.class_names[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    name: probs[i].item()
                    for i, name in enumerate(self.class_names)
                },
                "alert": confidence > 0.8 and pred_idx != 0
            }
```

## Resumen

| Arquitectura | Innovación Clave | Cuándo Usar |
|--------------|-----------------|-------------|
| VGG | Profundidad + 3×3 | Baseline simple |
| Inception | Multi-scale parallel | Objetos multi-escala |
| ResNet | Skip connections | Redes muy profundas |
| DenseNet | Dense connections | Datasets pequeños |
| EfficientNet | Compound scaling | Producción (balance) |

### Guía de Selección

```
¿Necesitas velocidad?
├── Sí → EfficientNet-B0 o MobileNet
└── No → ¿Accuracy máximo?
         ├── Sí → EfficientNet-B7 o ResNet-152
         └── No → ResNet-50 (balance)

¿Dataset pequeño?
├── Sí → DenseNet (mejor feature reuse)
└── No → ResNet o EfficientNet

¿Transfer learning?
└── Siempre empezar con modelo preentrenado en ImageNet
```

## Referencias

- ImageNet Classification with Deep CNNs (AlexNet)
- Very Deep CNNs for Large-Scale Image Recognition (VGG)
- Going Deeper with Convolutions (Inception)
- Deep Residual Learning (ResNet)
- Densely Connected CNNs (DenseNet)
- EfficientNet: Rethinking Model Scaling
