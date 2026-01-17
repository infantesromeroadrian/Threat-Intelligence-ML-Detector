# Segmentación de Imágenes

## Introducción

La segmentación asigna una etiqueta a CADA píxel de la imagen. Es más precisa que la detección (bounding boxes) porque delinea exactamente la forma de los objetos.

```
Tipos de Segmentación:

┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Segmentación Semántica:        Segmentación de Instancias:      │
│  ┌─────────────────────┐        ┌─────────────────────┐          │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │        │ ▓▓▓▓▓▓▓▓  ░░░░░░░░ │          │
│  │ ▓▓▓Persona▓▓▓▓▓▓▓▓ │        │ ▓Persona1▓ ░Persona2░│          │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │        │ ▓▓▓▓▓▓▓▓  ░░░░░░░░ │          │
│  └─────────────────────┘        └─────────────────────┘          │
│  Todos los píxeles de           Diferencia entre                  │
│  "persona" = mismo color        instancias individuales          │
│                                                                   │
│  Segmentación Panóptica:                                         │
│  ┌─────────────────────┐                                         │
│  │ ▓▓▓▓▓▓▓▓  ░░░░░░░░ │ Combina semántica + instancias          │
│  │ ▓Persona1▓ ░Persona2░│ + stuff (cielo, carretera)             │
│  │ ████Coche████████████│                                         │
│  └─────────────────────┘                                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Conceptos Fundamentales

### Representación de Máscaras

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict


class SegmentationMask:
    """
    Representaciones de máscaras de segmentación.
    """

    @staticmethod
    def create_binary_mask(
        image_size: Tuple[int, int],
        polygon: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Crea máscara binaria desde polígono."""
        import cv2

        mask = np.zeros(image_size, dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def mask_to_rle(mask: np.ndarray) -> Dict:
        """
        Convierte máscara a Run-Length Encoding (RLE).
        Formato comprimido usado en COCO.

        Ejemplo:
        Máscara:  [0,0,1,1,1,0,0,1,1]
        RLE:      [2, 3, 2, 2]  (2 ceros, 3 unos, 2 ceros, 2 unos)
        """
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {
            'counts': runs.tolist(),
            'size': list(mask.shape)
        }

    @staticmethod
    def rle_to_mask(rle: Dict) -> np.ndarray:
        """Decodifica RLE a máscara."""
        shape = rle['size']
        counts = rle['counts']

        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        position = 0
        for i, count in enumerate(counts):
            if i % 2 == 1:  # Unos
                mask[position:position + count] = 1
            position += count

        return mask.reshape(shape)

    @staticmethod
    def one_hot_encode(
        mask: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Convierte máscara de clases a one-hot.

        Input: (H, W) con valores 0 a num_classes-1
        Output: (num_classes, H, W)
        """
        one_hot = np.zeros((num_classes, *mask.shape), dtype=np.float32)
        for c in range(num_classes):
            one_hot[c] = (mask == c).astype(np.float32)
        return one_hot

    @staticmethod
    def class_mask_to_rgb(
        mask: np.ndarray,
        colormap: Dict[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        """Convierte máscara de clases a imagen RGB para visualización."""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in colormap.items():
            rgb[mask == class_id] = color

        return rgb
```

## Arquitectura Encoder-Decoder

### Concepto

```
Encoder-Decoder para Segmentación:

         Encoder (Downsampling)          Decoder (Upsampling)
┌────────────────────────────────┐  ┌────────────────────────────────┐
│                                │  │                                │
│  Input: 256×256×3              │  │                                │
│          │                     │  │                                │
│          ▼                     │  │                                │
│  ┌───────────────┐            │  │            ┌───────────────┐  │
│  │  128×128×64   │────────────┼──┼────────────│  128×128×64   │  │
│  └───────┬───────┘            │  │            └───────▲───────┘  │
│          │                     │  │                    │          │
│          ▼                     │  │                    │          │
│  ┌───────────────┐            │  │            ┌───────────────┐  │
│  │   64×64×128   │────────────┼──┼────────────│   64×64×128   │  │
│  └───────┬───────┘            │  │            └───────▲───────┘  │
│          │                     │  │                    │          │
│          ▼                     │  │                    │          │
│  ┌───────────────┐            │  │            ┌───────────────┐  │
│  │   32×32×256   │────────────┼──┼────────────│   32×32×256   │  │
│  └───────┬───────┘            │  │            └───────▲───────┘  │
│          │                     │  │                    │          │
│          └─────────────────────┼──┼────────────────────┘          │
│                    Bottleneck  │  │                               │
│                    16×16×512   │  │                               │
└────────────────────────────────┘  └────────────────────────────────┘

                    Skip Connections ─────────────────
                    (preservan detalles espaciales)
```

### FCN (Fully Convolutional Network)

```python
class FCN(nn.Module):
    """
    Fully Convolutional Network (2015).
    Primera arquitectura de segmentación end-to-end.

    Reemplaza FC layers por convoluciones 1×1.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "vgg16"
    ):
        super().__init__()

        # Usar VGG como encoder
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Features del encoder
        self.features = vgg.features

        # Reemplazar FC por conv 1×1
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )

        # Upsampling (bilinear o transposed conv)
        self.upsample = nn.Upsample(
            scale_factor=32,
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.features(x)

        # Clasificador
        x = self.classifier(x)

        # Upsample a tamaño original
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x
```

## U-Net

### Arquitectura

```
U-Net (2015): Arquitectura simétrica con skip connections.
Muy exitosa en segmentación médica.

                    Encoder                         Decoder
                    ┌─────┐                         ┌─────┐
Input 572×572×1 ───►│Conv │──► 570×570×64 ─────────►│Conv │──► 388×388×64 ──► Output
                    │Block│                   │     │Block│       │
                    └──┬──┘                   │     └──▲──┘       │
                       │ Pool                 │        │ UpConv   │
                       ▼                      │        │          │
                    ┌─────┐                   │     ┌─────┐       │
            284×284×64 ───►│Conv │──► 282×282×128 ──┼───►│Conv │──►│
                    │Block│                   │     │Block│       │
                    └──┬──┘                   │     └──▲──┘       │
                       │ Pool                 │        │ UpConv   │
                       ▼                      │        │          │
                    ┌─────┐                   │     ┌─────┐       │
                    │Conv │──► Bottleneck ────┘     │Conv │       │
                    │Block│     28×28×1024          │Block│       │
                    └─────┘                         └─────┘       │
                                                                  │
                                              Output: num_classes ◄┘
```

```python
class DoubleConv(nn.Module):
    """Bloque de doble convolución (Conv-BN-ReLU) × 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling: UpConv + Concatenate + DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        x1: Features del decoder (upsampled)
        x2: Skip connection del encoder
        """
        x1 = self.up(x1)

        # Padding si hay diferencia de tamaño
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])

        # Concatenar skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: Encoder-decoder simétrico con skip connections.

    Ideal para:
    - Segmentación médica
    - Datasets pequeños
    - Preservar detalles finos
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        features: List[int] = [64, 128, 256, 512, 1024],
        bilinear: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])

        # Decoder
        factor = 2 if bilinear else 1
        self.up1 = Up(features[4], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)

        # Output
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder con skip connections
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024 (bottleneck)

        # Decoder con skip connections
        x = self.up1(x5, x4)  # 512
        x = self.up2(x, x3)   # 256
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 64

        # Output
        logits = self.outc(x)
        return logits


# Ejemplo
model = UNet(in_channels=3, num_classes=21)  # 21 clases Pascal VOC
x = torch.randn(1, 3, 256, 256)
output = model(x)
print(f"Output shape: {output.shape}")  # [1, 21, 256, 256]
```

## DeepLab

### Atrous/Dilated Convolution

```
Atrous Convolution: Aumenta receptive field sin reducir resolución.

Standard Conv 3×3:          Dilated Conv 3×3 (rate=2):
┌───┬───┬───┐               ┌───┬───┬───┬───┬───┐
│ × │ × │ × │               │ × │   │ × │   │ × │
├───┼───┼───┤               ├───┼───┼───┼───┼───┤
│ × │ × │ × │               │   │   │   │   │   │
├───┼───┼───┤               ├───┼───┼───┼───┼───┤
│ × │ × │ × │               │ × │   │ × │   │ × │
└───┴───┴───┘               ├───┼───┼───┼───┼───┤
                            │   │   │   │   │   │
RF = 3×3                    ├───┼───┼───┼───┼───┤
                            │ × │   │ × │   │ × │
                            └───┴───┴───┴───┴───┘

                            RF = 5×5 (mismos parámetros!)
```

```python
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captura contexto multi-escala con diferentes dilation rates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: List[int] = [6, 12, 18]
    ):
        super().__init__()

        # 1×1 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convs con diferentes rates
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=3, padding=rate, dilation=rate, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global Average Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Projection
        num_branches = 2 + len(rates)  # 1×1 + atrous + global
        self.project = nn.Sequential(
            nn.Conv2d(num_branches * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]

        # Branch 1: 1×1 conv
        out1 = self.conv1(x)

        # Branch 2-4: Atrous convs
        atrous_outs = [conv(x) for conv in self.atrous_convs]

        # Branch 5: Global pooling
        global_out = self.global_pool(x)
        global_out = F.interpolate(
            global_out, size=size, mode='bilinear', align_corners=True
        )

        # Concatenar y proyectar
        out = torch.cat([out1, *atrous_outs, global_out], dim=1)
        out = self.project(out)

        return out


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+: ASPP + encoder-decoder con skip connections.
    State-of-the-art en segmentación semántica.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50"
    ):
        super().__init__()

        # Backbone (modificado para output stride 16)
        from torchvision.models import resnet50, ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # ASPP
        self.aspp = ASPP(2048, 256)

        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.layer0(x)
        low_level = self.layer1(x)  # Skip connection
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Upsample y concatenar con low-level features
        x = F.interpolate(
            x, size=low_level.shape[2:],
            mode='bilinear', align_corners=True
        )
        low_level = self.low_level_conv(low_level)
        x = torch.cat([x, low_level], dim=1)

        # Decoder
        x = self.decoder(x)

        # Upsample a tamaño original
        x = F.interpolate(
            x, size=input_size,
            mode='bilinear', align_corners=True
        )

        return x
```

## Segmentación de Instancias

### Mask R-CNN

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class InstanceSegmenter:
    """
    Mask R-CNN para segmentación de instancias.

    Extiende Faster R-CNN añadiendo una rama de máscara.
    """

    def __init__(self, num_classes: int, device: str = "cuda"):
        self.device = device
        self.num_classes = num_classes

        # Modelo preentrenado
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        # Modificar cabeza de clasificación
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # Modificar cabeza de máscara
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        self.model = self.model.to(device)

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> Dict:
        """
        Predicción con máscaras.

        Returns:
            boxes: (N, 4) bounding boxes
            labels: (N,) clases
            scores: (N,) confianzas
            masks: (N, H, W) máscaras binarias
        """
        self.model.eval()
        image = image.to(self.device)

        outputs = self.model([image])[0]

        # Filtrar por score
        mask = outputs['scores'] > score_threshold

        # Binarizar máscaras
        masks = outputs['masks'][mask] > mask_threshold
        masks = masks.squeeze(1)  # Remove channel dim

        return {
            "boxes": outputs['boxes'][mask].cpu(),
            "labels": outputs['labels'][mask].cpu(),
            "scores": outputs['scores'][mask].cpu(),
            "masks": masks.cpu()
        }

    def visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict,
        class_names: List[str] = None
    ) -> np.ndarray:
        """Visualiza predicciones sobre imagen."""
        import cv2

        output = image.copy()

        # Colores por instancia
        colors = [
            tuple(np.random.randint(0, 255, 3).tolist())
            for _ in range(len(predictions['masks']))
        ]

        for i, (box, label, score, mask) in enumerate(zip(
            predictions['boxes'],
            predictions['labels'],
            predictions['scores'],
            predictions['masks']
        )):
            color = colors[i]

            # Dibujar máscara semitransparente
            mask_np = mask.numpy().astype(np.uint8)
            colored_mask = np.zeros_like(output)
            colored_mask[mask_np == 1] = color
            output = cv2.addWeighted(output, 1, colored_mask, 0.5, 0)

            # Dibujar bounding box
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Etiqueta
            label_text = f"{class_names[label] if class_names else label}: {score:.2f}"
            cv2.putText(
                output, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return output
```

## Loss Functions

### Segmentation Losses

```python
class SegmentationLoss:
    """Funciones de pérdida para segmentación."""

    @staticmethod
    def cross_entropy_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Cross Entropy estándar para segmentación.

        pred: (B, C, H, W) logits
        target: (B, H, W) índices de clase
        """
        return F.cross_entropy(pred, target, weight=weight)

    @staticmethod
    def dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Dice Loss: Optimiza directamente el Dice coefficient.
        Mejor para clases desbalanceadas.

        Dice = 2 * |A ∩ B| / (|A| + |B|)

        pred: (B, C, H, W) probabilidades (después de softmax)
        target: (B, H, W) índices de clase
        """
        num_classes = pred.shape[1]

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Calcular Dice por clase
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice.mean()

    @staticmethod
    def focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Focal Loss para segmentación.
        Reduce peso de píxeles fáciles de clasificar.
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * ((1 - pt) ** gamma) * ce_loss
        return focal_loss.mean()

    @staticmethod
    def boundary_loss(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Boundary Loss: Penaliza más los errores en bordes.
        Mejora delineación de objetos.
        """
        # Detectar bordes con Sobel
        sobel_x = torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)

        sobel_y = torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)

        target_float = target.float().unsqueeze(1)
        edge_x = F.conv2d(target_float, sobel_x, padding=1)
        edge_y = F.conv2d(target_float, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edges = (edges > 0).float()

        # Ponderar CE por bordes
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * (1 + 5 * edges.squeeze(1))

        return weighted_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """Combinación de múltiples losses."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.loss_fn = SegmentationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Softmax para Dice
        pred_soft = F.softmax(pred, dim=1)

        ce_loss = self.loss_fn.cross_entropy_loss(pred, target)
        dice_loss = self.loss_fn.dice_loss(pred_soft, target)
        boundary_loss = self.loss_fn.boundary_loss(pred, target)

        total_loss = (
            self.ce_weight * ce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss
        )

        return {
            "total": total_loss,
            "ce": ce_loss,
            "dice": dice_loss,
            "boundary": boundary_loss
        }
```

## Métricas de Evaluación

```python
class SegmentationMetrics:
    """Métricas para evaluación de segmentación."""

    @staticmethod
    def pixel_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Pixel Accuracy: % de píxeles correctamente clasificados.
        Simple pero engañosa con clases desbalanceadas.
        """
        pred_classes = pred.argmax(dim=1)
        correct = (pred_classes == target).float().sum()
        total = target.numel()
        return (correct / total).item()

    @staticmethod
    def mean_pixel_accuracy(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int
    ) -> float:
        """Mean Pixel Accuracy: Promedio de accuracy por clase."""
        pred_classes = pred.argmax(dim=1)
        accuracies = []

        for c in range(num_classes):
            mask = target == c
            if mask.sum() > 0:
                correct = ((pred_classes == c) & mask).float().sum()
                accuracies.append((correct / mask.float().sum()).item())

        return np.mean(accuracies) if accuracies else 0.0

    @staticmethod
    def iou_per_class(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int
    ) -> Dict[int, float]:
        """
        IoU por clase.

        IoU = TP / (TP + FP + FN)
        """
        pred_classes = pred.argmax(dim=1)
        ious = {}

        for c in range(num_classes):
            pred_mask = pred_classes == c
            target_mask = target == c

            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()

            if union > 0:
                ious[c] = (intersection / union).item()

        return ious

    @staticmethod
    def mean_iou(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int = None
    ) -> float:
        """
        Mean IoU (mIoU): Principal métrica para segmentación.
        Promedio de IoU sobre todas las clases.
        """
        ious = SegmentationMetrics.iou_per_class(pred, target, num_classes)

        if ignore_index is not None and ignore_index in ious:
            del ious[ignore_index]

        return np.mean(list(ious.values())) if ious else 0.0

    @staticmethod
    def dice_coefficient(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int
    ) -> float:
        """
        Dice Coefficient (F1 Score para segmentación).

        Dice = 2 * |A ∩ B| / (|A| + |B|)
        """
        pred_classes = pred.argmax(dim=1)
        dices = []

        for c in range(num_classes):
            pred_mask = (pred_classes == c).float()
            target_mask = (target == c).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()

            if union > 0:
                dice = (2.0 * intersection / union).item()
                dices.append(dice)

        return np.mean(dices) if dices else 0.0
```

## Aplicaciones en Ciberseguridad

### Segmentación de Screenshots

```python
class ScreenshotSegmenter:
    """
    Segmenta elementos en screenshots para análisis de seguridad.

    Clases:
    0: Background
    1: Input field
    2: Button
    3: Logo
    4: Text
    5: Image
    6: Form
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.num_classes = 7
        self.class_names = [
            "background", "input_field", "button",
            "logo", "text", "image", "form"
        ]

        # U-Net para segmentación precisa
        self.model = UNet(
            in_channels=3,
            num_classes=self.num_classes
        ).to(device)

    @torch.no_grad()
    def segment(self, image: torch.Tensor) -> Dict:
        """Segmenta screenshot."""
        self.model.eval()

        image = image.to(self.device)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        logits = self.model(image)
        probs = F.softmax(logits, dim=1)
        pred_mask = logits.argmax(dim=1)

        return {
            "mask": pred_mask.cpu(),
            "probabilities": probs.cpu(),
            "elements": self._extract_elements(pred_mask[0].cpu())
        }

    def _extract_elements(self, mask: torch.Tensor) -> List[Dict]:
        """Extrae elementos individuales de la máscara."""
        import cv2

        elements = []
        mask_np = mask.numpy().astype(np.uint8)

        for cls in range(1, self.num_classes):  # Skip background
            class_mask = (mask_np == cls).astype(np.uint8)

            # Encontrar componentes conectados
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )

            for i in range(1, num_labels):  # Skip background component
                x, y, w, h, area = stats[i]

                if area > 100:  # Filtrar ruido
                    elements.append({
                        "class": self.class_names[cls],
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "centroid": centroids[i].tolist()
                    })

        return elements

    def analyze_phishing_indicators(self, segmentation: Dict) -> Dict:
        """Analiza indicadores de phishing basado en segmentación."""
        elements = segmentation["elements"]

        analysis = {
            "risk_score": 0.0,
            "indicators": [],
            "statistics": {}
        }

        # Contar elementos por tipo
        type_counts = {}
        for elem in elements:
            type_counts[elem["class"]] = type_counts.get(elem["class"], 0) + 1

        analysis["statistics"] = type_counts

        # Indicadores de riesgo
        input_fields = [e for e in elements if e["class"] == "input_field"]
        if len(input_fields) > 3:
            analysis["indicators"].append("Múltiples campos de entrada")
            analysis["risk_score"] += 0.2

        # Input fields muy grandes (sospechoso)
        for field in input_fields:
            x1, y1, x2, y2 = field["bbox"]
            if (x2 - x1) > 400 or (y2 - y1) > 100:
                analysis["indicators"].append("Campo de entrada inusualmente grande")
                analysis["risk_score"] += 0.1
                break

        # Muchos botones
        buttons = [e for e in elements if e["class"] == "button"]
        if len(buttons) > 5:
            analysis["indicators"].append("Exceso de botones")
            analysis["risk_score"] += 0.1

        # Logo pequeño o ausente
        logos = [e for e in elements if e["class"] == "logo"]
        if not logos:
            analysis["indicators"].append("Sin logo visible")
            analysis["risk_score"] += 0.15
        elif all(l["area"] < 5000 for l in logos):
            analysis["indicators"].append("Logo muy pequeño")
            analysis["risk_score"] += 0.1

        analysis["risk_score"] = min(analysis["risk_score"], 1.0)
        analysis["risk_level"] = (
            "HIGH" if analysis["risk_score"] > 0.5 else
            "MEDIUM" if analysis["risk_score"] > 0.25 else
            "LOW"
        )

        return analysis


class DocumentSegmenter:
    """
    Segmenta documentos escaneados para extracción de información.
    Útil para OCR y análisis forense.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Clases para documentos
        self.class_names = [
            "background", "text_block", "table",
            "figure", "header", "footer", "signature"
        ]
        self.num_classes = len(self.class_names)

        self.model = DeepLabV3Plus(
            num_classes=self.num_classes
        ).to(device)

    @torch.no_grad()
    def segment_document(self, image: torch.Tensor) -> Dict:
        """Segmenta documento."""
        self.model.eval()

        image = image.to(self.device).unsqueeze(0)
        logits = self.model(image)
        pred_mask = logits.argmax(dim=1)[0]

        return {
            "mask": pred_mask.cpu(),
            "regions": self._extract_regions(pred_mask.cpu())
        }

    def _extract_regions(self, mask: torch.Tensor) -> List[Dict]:
        """Extrae regiones del documento."""
        import cv2

        regions = []
        mask_np = mask.numpy().astype(np.uint8)

        for cls in range(1, self.num_classes):
            class_mask = (mask_np == cls).astype(np.uint8)
            contours, _ = cv2.findContours(
                class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 500:  # Filtrar regiones pequeñas
                    regions.append({
                        "type": self.class_names[cls],
                        "bbox": [x, y, x + w, y + h],
                        "priority": self._get_extraction_priority(cls)
                    })

        # Ordenar por prioridad (para OCR)
        regions.sort(key=lambda r: (r["priority"], r["bbox"][1]))
        return regions

    def _get_extraction_priority(self, cls: int) -> int:
        """Prioridad para extracción de texto."""
        priorities = {
            1: 1,  # text_block - alta prioridad
            4: 2,  # header
            5: 3,  # footer
            2: 4,  # table
            6: 5,  # signature
            3: 6,  # figure - baja prioridad
        }
        return priorities.get(cls, 10)
```

## Resumen

| Arquitectura | Tipo | Uso Típico |
|--------------|------|------------|
| FCN | Semántica | Baseline |
| U-Net | Semántica | Imágenes médicas, datasets pequeños |
| DeepLabV3+ | Semántica | State-of-the-art general |
| Mask R-CNN | Instancias | Detección + segmentación |
| Panoptic FPN | Panóptica | Escenas complejas |

### Checklist Segmentación

```
□ Máscaras anotadas correctamente (sin huecos, bordes limpios)
□ Class weights para clases desbalanceadas
□ Data augmentation que preserva máscaras
□ Loss apropiado (Dice para desbalance, CE + Dice combinado)
□ mIoU como métrica principal
□ Visualizar predicciones durante entrenamiento
□ Post-procesamiento (CRF, morfología) si es necesario
```

## Referencias

- Fully Convolutional Networks for Semantic Segmentation
- U-Net: Convolutional Networks for Biomedical Image Segmentation
- DeepLab: Semantic Image Segmentation with Deep CNNs and Fully Connected CRFs
- Mask R-CNN
- Panoptic Segmentation
