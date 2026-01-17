# Detección de Objetos

## Introducción

La detección de objetos responde DOS preguntas: ¿QUÉ hay en la imagen? y ¿DÓNDE está? A diferencia de clasificación (una etiqueta por imagen), la detección localiza múltiples objetos con bounding boxes.

```
Clasificación vs Detección vs Segmentación:

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│                 │  │   ┌───────┐     │  │   ▓▓▓▓▓▓▓       │
│    🐱           │  │   │  🐱  │     │  │   ▓▓🐱▓▓▓       │
│                 │  │   └───────┘     │  │   ▓▓▓▓▓▓▓       │
│                 │  │       ┌────┐    │  │         ░░░░░   │
│        🐕       │  │       │ 🐕│    │  │         ░🐕░░   │
│                 │  │       └────┘    │  │         ░░░░░   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
  Clasificación:       Detección:          Segmentación:
  "gato, perro"        Boxes + labels      Máscaras píxel
```

## Conceptos Fundamentales

### Bounding Box

```python
from dataclasses import dataclass
from typing import Tuple, List
import torch


@dataclass
class BoundingBox:
    """
    Representación de bounding box.

    Formatos comunes:
    - (x1, y1, x2, y2): Esquinas (PASCAL VOC, COCO)
    - (x_center, y_center, width, height): Centro (YOLO)
    - Normalizadas vs absolutas
    """
    x1: float  # Top-left X
    y1: float  # Top-left Y
    x2: float  # Bottom-right X
    y2: float  # Bottom-right Y
    label: int = 0
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_yolo_format(
        self,
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """Convierte a formato YOLO normalizado (cx, cy, w, h)."""
        cx = (self.x1 + self.x2) / 2 / img_width
        cy = (self.y1 + self.y2) / 2 / img_height
        w = self.width / img_width
        h = self.height / img_height
        return (cx, cy, w, h)

    @classmethod
    def from_yolo_format(
        cls,
        cx: float, cy: float, w: float, h: float,
        img_width: int, img_height: int,
        label: int = 0
    ) -> 'BoundingBox':
        """Convierte de formato YOLO a esquinas."""
        x1 = (cx - w/2) * img_width
        y1 = (cy - h/2) * img_height
        x2 = (cx + w/2) * img_width
        y2 = (cy + h/2) * img_height
        return cls(x1, y1, x2, y2, label)
```

### IoU (Intersection over Union)

```python
def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calcula IoU entre dos bounding boxes.

    IoU = Área de Intersección / Área de Unión

    ┌─────────────┐
    │    box1     │
    │   ┌─────┼───┼───┐
    │   │ INT │   │   │
    └───┼─────┘   │   │
        │   box2      │
        └─────────────┘

    IoU = 0: Sin overlap
    IoU = 1: Boxes idénticos
    IoU típico para match: > 0.5
    """
    # Coordenadas de intersección
    x1_int = max(box1.x1, box2.x1)
    y1_int = max(box1.y1, box2.y1)
    x2_int = min(box1.x2, box2.x2)
    y2_int = min(box1.y2, box2.y2)

    # Área de intersección
    if x2_int < x1_int or y2_int < y1_int:
        return 0.0

    intersection = (x2_int - x1_int) * (y2_int - y1_int)

    # Área de unión
    union = box1.area + box2.area - intersection

    return intersection / union if union > 0 else 0.0


def batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU vectorizado para tensores.

    Args:
        boxes1: (N, 4) en formato (x1, y1, x2, y2)
        boxes2: (M, 4) en formato (x1, y1, x2, y2)

    Returns:
        (N, M) matriz de IoU
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersección
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / union

    return iou
```

### Non-Maximum Suppression (NMS)

```python
def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Non-Maximum Suppression.
    Elimina detecciones redundantes manteniendo las de mayor confianza.

    Proceso:
    1. Ordenar boxes por score (descendente)
    2. Tomar box con mayor score
    3. Eliminar boxes con IoU > threshold
    4. Repetir hasta procesar todos

    ┌───────────────────────────────────────────┐
    │  Antes de NMS:        Después de NMS:     │
    │  ┌─────────┐          ┌─────────┐         │
    │  │┌───────┐│          │         │         │
    │  ││  🐱   ││    →     │   🐱    │         │
    │  │└───────┘│          │         │         │
    │  └─────────┘          └─────────┘         │
    │  (múltiples boxes)    (1 box mejor)       │
    └───────────────────────────────────────────┘
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    # Ordenar por score
    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        # Mantener el de mayor score
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Calcular IoU con resto
        current_box = boxes[i].unsqueeze(0)
        remaining_boxes = boxes[order[1:]]

        ious = batch_iou(current_box, remaining_boxes).squeeze(0)

        # Mantener boxes con IoU < threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS: En lugar de eliminar, reduce scores.
    Mejor para objetos que se solapan.

    score_i = score_i * exp(-(iou²/sigma))
    """
    boxes = boxes.clone()
    scores = scores.clone()

    keep = []
    while scores.max() > score_threshold:
        max_idx = scores.argmax()
        keep.append(max_idx.item())

        if len(keep) == len(boxes):
            break

        # Calcular IoU
        current_box = boxes[max_idx].unsqueeze(0)
        ious = batch_iou(current_box, boxes).squeeze(0)

        # Soft decay
        decay = torch.exp(-(ious ** 2) / sigma)
        scores = scores * decay
        scores[max_idx] = 0  # Eliminar el procesado

    return boxes[keep], scores[keep]
```

## Arquitecturas de Detección

### Two-Stage Detectors

```
Two-Stage: Region Proposal + Classification

Fase 1: RPN (Region Proposal Network)
┌───────────────┐
│   Imagen      │
└───────┬───────┘
        ▼
┌───────────────┐
│   Backbone    │ (ResNet, VGG)
│   (features)  │
└───────┬───────┘
        ▼
┌───────────────┐
│     RPN       │ → ~2000 proposals con "objectness" score
└───────┬───────┘
        ▼
Fase 2: Clasificación + Regresión
┌───────────────┐
│  ROI Pooling  │ → Features de tamaño fijo por proposal
└───────┬───────┘
        ▼
┌───────────────┐
│   FC Heads    │ → Clase + Bounding box refinado
└───────────────┘

Ejemplos: R-CNN, Fast R-CNN, Faster R-CNN
```

### Faster R-CNN

```python
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector:
    """
    Faster R-CNN con backbone ResNet50-FPN.

    Componentes:
    1. Backbone: Extrae features multi-escala (FPN)
    2. RPN: Propone regiones candidatas
    3. ROI Head: Clasifica y refina boxes
    """

    def __init__(self, num_classes: int, device: str = "cuda"):
        self.device = device

        # Modelo preentrenado en COCO
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Modificar cabeza de clasificación
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        self.model = self.model.to(device)

    def train_step(
        self,
        images: List[torch.Tensor],
        targets: List[dict]
    ) -> dict:
        """
        Paso de entrenamiento.

        targets debe ser lista de dicts con:
        - boxes: (N, 4) tensor en formato (x1, y1, x2, y2)
        - labels: (N,) tensor de clases
        """
        self.model.train()

        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        return {
            "total_loss": losses,
            "loss_classifier": loss_dict.get("loss_classifier", 0),
            "loss_box_reg": loss_dict.get("loss_box_reg", 0),
            "loss_objectness": loss_dict.get("loss_objectness", 0),
            "loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", 0),
        }

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        score_threshold: float = 0.5
    ) -> dict:
        """Predicción en una imagen."""
        self.model.eval()

        image = image.to(self.device)
        outputs = self.model([image])[0]

        # Filtrar por score
        mask = outputs['scores'] > score_threshold

        return {
            "boxes": outputs['boxes'][mask].cpu(),
            "labels": outputs['labels'][mask].cpu(),
            "scores": outputs['scores'][mask].cpu()
        }


# Ejemplo de uso
detector = FasterRCNNDetector(num_classes=91)  # COCO tiene 91 clases

# Formato de targets para entrenamiento
targets = [{
    "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
    "labels": torch.tensor([1, 2])
}]
```

### One-Stage Detectors (YOLO)

```
One-Stage: Detección directa sin proposals

┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Imagen → Backbone → Feature Map → Predictions        │
│                           │                             │
│                           ▼                             │
│                    ┌─────────────┐                      │
│                    │ S × S Grid  │                      │
│                    └─────────────┘                      │
│                           │                             │
│              Para cada celda:                           │
│              - B bounding boxes                         │
│              - Confianza (objectness)                   │
│              - C probabilidades de clase               │
│                                                         │
└─────────────────────────────────────────────────────────┘

YOLO Grid:
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ 🐱│   │   │   │   │  ← Celda responsable
├───┼───┼───┼───┼───┼───┼───┤     del objeto cuyo
│   │   │   │   │   │   │   │     centro cae aquí
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │ 🐕│   │   │
└───┴───┴───┴───┴───┴───┴───┘
```

```python
class YOLOv1Head(nn.Module):
    """
    Cabeza de detección estilo YOLO v1.
    Simplificado para entender el concepto.
    """

    def __init__(
        self,
        num_classes: int,
        num_boxes: int = 2,
        grid_size: int = 7
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.grid_size = grid_size

        # Output: S×S×(B*5 + C)
        # 5 = (x, y, w, h, confidence)
        # C = num_classes
        self.output_size = num_boxes * 5 + num_classes

        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * self.output_size)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, 1024, S, S) del backbone

        Returns:
            (batch, S, S, B*5 + C) predicciones
        """
        batch_size = features.size(0)
        output = self.fc(features)
        output = output.view(
            batch_size, self.grid_size, self.grid_size, self.output_size
        )
        return output

    def decode_predictions(
        self,
        predictions: torch.Tensor,
        conf_threshold: float = 0.5
    ) -> List[dict]:
        """Decodifica predicciones a bounding boxes."""
        batch_size = predictions.size(0)
        results = []

        for b in range(batch_size):
            boxes = []
            scores = []
            labels = []

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_pred = predictions[b, i, j]

                    # Extraer probabilidades de clase
                    class_probs = torch.softmax(cell_pred[self.num_boxes * 5:], dim=0)

                    # Para cada box predicho
                    for box_idx in range(self.num_boxes):
                        offset = box_idx * 5
                        x = (cell_pred[offset] + j) / self.grid_size
                        y = (cell_pred[offset + 1] + i) / self.grid_size
                        w = cell_pred[offset + 2] ** 2
                        h = cell_pred[offset + 3] ** 2
                        conf = torch.sigmoid(cell_pred[offset + 4])

                        # Confianza total = objectness × class_prob
                        class_conf = conf * class_probs

                        max_conf, max_class = class_conf.max(0)

                        if max_conf > conf_threshold:
                            # Convertir a (x1, y1, x2, y2)
                            x1 = x - w/2
                            y1 = y - h/2
                            x2 = x + w/2
                            y2 = y + h/2

                            boxes.append([x1, y1, x2, y2])
                            scores.append(max_conf.item())
                            labels.append(max_class.item())

            results.append({
                "boxes": torch.tensor(boxes) if boxes else torch.zeros((0, 4)),
                "scores": torch.tensor(scores) if scores else torch.zeros(0),
                "labels": torch.tensor(labels) if labels else torch.zeros(0, dtype=torch.long)
            })

        return results
```

### YOLO Moderno (Ultralytics)

```python
# YOLO moderno con ultralytics (YOLOv8, YOLO11)
# pip install ultralytics

from ultralytics import YOLO


class ModernYOLODetector:
    """
    Wrapper para YOLO moderno (v8, v11).
    Ultralytics proporciona API simple.
    """

    def __init__(
        self,
        model_size: str = "n",  # n, s, m, l, x
        task: str = "detect"
    ):
        """
        model_size:
        - n: Nano (más rápido, menos preciso)
        - s: Small
        - m: Medium
        - l: Large
        - x: XLarge (más preciso, más lento)
        """
        self.model = YOLO(f"yolov8{model_size}.pt")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16
    ):
        """Entrena modelo en dataset custom."""
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0  # GPU
        )

    def predict(
        self,
        source: str,  # Path a imagen/video/directorio
        conf: float = 0.5,
        save: bool = True
    ):
        """Predicción."""
        return self.model.predict(
            source=source,
            conf=conf,
            save=save
        )

    def export(self, format: str = "onnx"):
        """Exporta modelo a diferentes formatos."""
        return self.model.export(format=format)


# Ejemplo de dataset YAML para YOLO
"""
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: truck
"""

# Uso
detector = ModernYOLODetector(model_size="s")

# Predicción rápida
results = detector.predict("imagen.jpg", conf=0.5)

for result in results:
    boxes = result.boxes.xyxy  # Bounding boxes
    scores = result.boxes.conf  # Confianzas
    classes = result.boxes.cls  # Clases
```

## SSD (Single Shot Detector)

```python
class SSDHead(nn.Module):
    """
    Cabeza SSD: Detecta en múltiples escalas.

    Feature Pyramid:
    ┌─────────────────────────────────────────────┐
    │  38×38 (objetos pequeños)                   │
    │  19×19                                      │
    │  10×10                                      │
    │   5×5                                       │
    │   3×3                                       │
    │   1×1 (objetos grandes)                     │
    └─────────────────────────────────────────────┘

    Cada escala tiene diferentes anchor boxes.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: List[int] = [512, 1024, 512, 256, 256, 256],
        num_anchors: List[int] = [4, 6, 6, 6, 4, 4]
    ):
        super().__init__()

        self.num_classes = num_classes

        # Cabezas de clasificación
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(in_ch, n_anc * num_classes, kernel_size=3, padding=1)
            for in_ch, n_anc in zip(in_channels, num_anchors)
        ])

        # Cabezas de regresión (4 coords por anchor)
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(in_ch, n_anc * 4, kernel_size=3, padding=1)
            for in_ch, n_anc in zip(in_channels, num_anchors)
        ])

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Lista de feature maps de diferentes escalas

        Returns:
            cls_preds: (batch, total_anchors, num_classes)
            reg_preds: (batch, total_anchors, 4)
        """
        cls_preds = []
        reg_preds = []

        for feat, cls_head, reg_head in zip(features, self.cls_heads, self.reg_heads):
            batch_size = feat.size(0)

            # Clasificación
            cls = cls_head(feat)
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(batch_size, -1, self.num_classes)
            cls_preds.append(cls)

            # Regresión
            reg = reg_head(feat)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            reg = reg.view(batch_size, -1, 4)
            reg_preds.append(reg)

        return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)
```

## Anchor Boxes

### Concepto

```
Anchor Boxes: Templates de diferentes tamaños y aspect ratios.
La red predice offsets respecto a estos anchors.

┌────────────────────────────────────────────────────┐
│                                                    │
│  Anchor 1:1      Anchor 2:1      Anchor 1:2       │
│  ┌────────┐      ┌──────────┐    ┌─────┐          │
│  │        │      │          │    │     │          │
│  │        │      └──────────┘    │     │          │
│  │        │                      │     │          │
│  └────────┘                      │     │          │
│                                  └─────┘          │
│                                                    │
│  La red predice:                                   │
│  Δx, Δy: offset del centro                        │
│  Δw, Δh: escala respecto al anchor                │
│                                                    │
└────────────────────────────────────────────────────┘
```

```python
class AnchorGenerator:
    """Genera anchor boxes para detección."""

    def __init__(
        self,
        sizes: List[float] = [32, 64, 128, 256, 512],
        aspect_ratios: List[float] = [0.5, 1.0, 2.0]
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def generate_anchors(
        self,
        feature_map_size: int,
        image_size: int
    ) -> torch.Tensor:
        """
        Genera anchors para un feature map.

        Returns:
            (H*W*num_anchors, 4) en formato (x1, y1, x2, y2)
        """
        stride = image_size / feature_map_size
        anchors = []

        for y in range(feature_map_size):
            for x in range(feature_map_size):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride

                for size in self.sizes:
                    for ratio in self.aspect_ratios:
                        w = size * (ratio ** 0.5)
                        h = size / (ratio ** 0.5)

                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2

                        anchors.append([x1, y1, x2, y2])

        return torch.tensor(anchors)

    def encode_boxes(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Codifica ground truth boxes como offsets de anchors.

        Encoding:
        tx = (gx - ax) / aw
        ty = (gy - ay) / ah
        tw = log(gw / aw)
        th = log(gh / ah)
        """
        # Convertir a (cx, cy, w, h)
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

        anc_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anc_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anc_w = anchors[:, 2] - anchors[:, 0]
        anc_h = anchors[:, 3] - anchors[:, 1]

        tx = (gt_cx - anc_cx) / anc_w
        ty = (gt_cy - anc_cy) / anc_h
        tw = torch.log(gt_w / anc_w)
        th = torch.log(gt_h / anc_h)

        return torch.stack([tx, ty, tw, th], dim=1)

    def decode_boxes(
        self,
        pred_offsets: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """Decodifica predicciones a bounding boxes."""
        anc_cx = (anchors[:, 0] + anchors[:, 2]) / 2
        anc_cy = (anchors[:, 1] + anchors[:, 3]) / 2
        anc_w = anchors[:, 2] - anchors[:, 0]
        anc_h = anchors[:, 3] - anchors[:, 1]

        pred_cx = pred_offsets[:, 0] * anc_w + anc_cx
        pred_cy = pred_offsets[:, 1] * anc_h + anc_cy
        pred_w = torch.exp(pred_offsets[:, 2]) * anc_w
        pred_h = torch.exp(pred_offsets[:, 3]) * anc_h

        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2

        return torch.stack([x1, y1, x2, y2], dim=1)
```

## Loss Functions

### Detection Loss

```python
class DetectionLoss(nn.Module):
    """
    Loss para detección de objetos.

    Componentes:
    1. Classification Loss: Cross-entropy o Focal Loss
    2. Localization Loss: Smooth L1 o IoU Loss
    """

    def __init__(
        self,
        num_classes: int,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal Loss: Reduce peso de ejemplos fáciles.
        Importante para class imbalance en detección
        (muchos más backgrounds que objetos).

        FL(p) = -α(1-p)^γ * log(p)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_weight = self.alpha * ((1 - pt) ** self.gamma)
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()

    def smooth_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Smooth L1: Robusto a outliers.

        L = 0.5 * x² / β    si |x| < β
            |x| - 0.5 * β   otherwise
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
        return loss.mean()

    def giou_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        GIoU Loss: Generaliza IoU para boxes sin overlap.

        GIoU = IoU - (C - Union) / C
        donde C es el box más pequeño que contiene ambos.
        """
        # IoU normal
        iou = batch_iou(pred_boxes, target_boxes).diagonal()

        # Box envolvente
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        # Áreas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

        union = pred_area + target_area - iou * (pred_area + target_area) / (1 + iou)

        giou = iou - (enclose_area - union) / enclose_area

        return (1 - giou).mean()

    def forward(
        self,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor,
        box_pred: torch.Tensor,
        box_target: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> dict:
        """
        Calcula loss total.

        Args:
            cls_pred: (N, num_classes) predicciones de clase
            cls_target: (N,) ground truth classes
            box_pred: (N, 4) predicciones de boxes
            box_target: (N, 4) ground truth boxes
            pos_mask: (N,) máscara de anchors positivos
        """
        # Classification loss (todos los anchors)
        if self.use_focal_loss:
            cls_loss = self.focal_loss(cls_pred, cls_target)
        else:
            cls_loss = F.cross_entropy(cls_pred, cls_target)

        # Localization loss (solo positivos)
        if pos_mask.sum() > 0:
            box_loss = self.smooth_l1_loss(
                box_pred[pos_mask],
                box_target[pos_mask]
            )
        else:
            box_loss = torch.tensor(0.0)

        return {
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "total_loss": cls_loss + box_loss
        }
```

## Métricas de Evaluación

### Mean Average Precision (mAP)

```python
def calculate_ap(
    pred_boxes: List[torch.Tensor],
    pred_scores: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    iou_threshold: float = 0.5
) -> float:
    """
    Calcula Average Precision para una clase.

    Proceso:
    1. Ordenar predicciones por score
    2. Para cada predicción, determinar si es TP o FP
    3. Calcular precision y recall acumulados
    4. Interpolar curva PR y calcular área
    """
    all_preds = []
    n_gt = 0

    # Agrupar predicciones con imagen index
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            all_preds.append({
                'img_idx': img_idx,
                'box': box,
                'score': score.item()
            })
        n_gt += len(gt_boxes[img_idx])

    # Ordenar por score
    all_preds.sort(key=lambda x: x['score'], reverse=True)

    # Tracking de GT matcheados
    gt_matched = [torch.zeros(len(gt)) for gt in gt_boxes]

    tp = []
    fp = []

    for pred in all_preds:
        img_idx = pred['img_idx']
        pred_box = pred['box']

        # Buscar mejor match con GT
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes[img_idx]):
            if gt_matched[img_idx][gt_idx]:
                continue

            iou = calculate_iou(
                BoundingBox(*pred_box.tolist()),
                BoundingBox(*gt_box.tolist())
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Determinar TP o FP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[img_idx][best_gt_idx] = 1
        else:
            tp.append(0)
            fp.append(1)

    # Calcular precision y recall acumulados
    tp_cumsum = torch.tensor(tp).cumsum(0)
    fp_cumsum = torch.tensor(fp).cumsum(0)

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Interpolar y calcular AP (11-point interpolation)
    ap = 0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11

    return ap


def calculate_map(
    predictions: List[dict],
    ground_truths: List[dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.5]
) -> dict:
    """
    Calcula mAP sobre todas las clases.

    predictions y ground_truths: Lista por imagen de dicts con:
    - boxes: (N, 4)
    - labels: (N,)
    - scores: (N,) - solo para predictions
    """
    aps = {}

    for cls in range(num_classes):
        cls_preds_boxes = []
        cls_preds_scores = []
        cls_gt_boxes = []

        for pred, gt in zip(predictions, ground_truths):
            # Filtrar por clase
            pred_mask = pred['labels'] == cls
            gt_mask = gt['labels'] == cls

            cls_preds_boxes.append(pred['boxes'][pred_mask])
            cls_preds_scores.append(pred['scores'][pred_mask])
            cls_gt_boxes.append(gt['boxes'][gt_mask])

        # AP por threshold
        for iou_thresh in iou_thresholds:
            ap = calculate_ap(
                cls_preds_boxes, cls_preds_scores,
                cls_gt_boxes, iou_thresh
            )
            aps[f"AP_{cls}_IoU{iou_thresh}"] = ap

    # mAP
    mAP = sum(v for k, v in aps.items()) / len(aps) if aps else 0

    return {"mAP": mAP, **aps}
```

## Aplicaciones en Ciberseguridad

### Detector de Elementos en Screenshots

```python
class ScreenshotElementDetector:
    """
    Detecta elementos en screenshots para análisis de seguridad.

    Elementos a detectar:
    - Campos de input (posibles credenciales)
    - Botones de submit
    - Logos (para phishing)
    - Pop-ups sospechosos
    - Mensajes de error falsos
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.class_names = [
            "input_field", "password_field", "submit_button",
            "logo", "popup", "fake_warning", "captcha"
        ]

        # Faster R-CNN para detección precisa
        self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(self.class_names) + 1)
        self.model = self.model.to(device)

    def analyze_screenshot(self, image: torch.Tensor) -> dict:
        """
        Analiza screenshot y detecta elementos sospechosos.
        """
        self.model.eval()

        with torch.no_grad():
            detections = self.model([image.to(self.device)])[0]

        # Analizar detecciones
        analysis = {
            "elements": [],
            "risk_factors": [],
            "risk_score": 0.0
        }

        for box, label, score in zip(
            detections['boxes'],
            detections['labels'],
            detections['scores']
        ):
            if score < 0.5:
                continue

            element = {
                "type": self.class_names[label - 1],  # -1 porque 0 es background
                "box": box.cpu().tolist(),
                "confidence": score.item()
            }
            analysis["elements"].append(element)

            # Factores de riesgo
            if element["type"] == "password_field":
                analysis["risk_factors"].append("Campo de contraseña detectado")
                analysis["risk_score"] += 0.2

            if element["type"] == "fake_warning":
                analysis["risk_factors"].append("Advertencia potencialmente falsa")
                analysis["risk_score"] += 0.4

            if element["type"] == "popup":
                analysis["risk_factors"].append("Pop-up detectado")
                analysis["risk_score"] += 0.1

        # Patrones sospechosos
        password_fields = sum(1 for e in analysis["elements"] if e["type"] == "password_field")
        if password_fields > 1:
            analysis["risk_factors"].append(f"Múltiples campos de contraseña ({password_fields})")
            analysis["risk_score"] += 0.3

        analysis["risk_score"] = min(analysis["risk_score"], 1.0)
        analysis["risk_level"] = (
            "HIGH" if analysis["risk_score"] > 0.7 else
            "MEDIUM" if analysis["risk_score"] > 0.4 else
            "LOW"
        )

        return analysis


class CCTVObjectDetector:
    """
    Detector de objetos para análisis de CCTV.
    Detecta personas, vehículos, objetos abandonados.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # YOLO para tiempo real
        self.model = YOLO("yolov8m.pt")

        # Clases de interés para seguridad
        self.security_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            24: "backpack",
            26: "handbag",
            28: "suitcase"
        }

    def analyze_frame(
        self,
        frame: torch.Tensor,
        zones: List[dict] = None
    ) -> dict:
        """
        Analiza frame de CCTV.

        zones: Lista de zonas restringidas con formato:
        {"name": "Entrada", "box": [x1, y1, x2, y2]}
        """
        results = self.model(frame, classes=list(self.security_classes.keys()))

        analysis = {
            "detections": [],
            "alerts": [],
            "statistics": {
                "persons": 0,
                "vehicles": 0,
                "objects": 0
            }
        }

        for result in results:
            for box, cls, conf in zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            ):
                cls_id = int(cls)
                if cls_id not in self.security_classes:
                    continue

                detection = {
                    "class": self.security_classes[cls_id],
                    "box": box.cpu().tolist(),
                    "confidence": conf.item()
                }
                analysis["detections"].append(detection)

                # Estadísticas
                if cls_id == 0:
                    analysis["statistics"]["persons"] += 1
                elif cls_id in [2, 3, 5, 7]:
                    analysis["statistics"]["vehicles"] += 1
                else:
                    analysis["statistics"]["objects"] += 1

                # Verificar zonas restringidas
                if zones:
                    for zone in zones:
                        if self._box_in_zone(box, zone["box"]):
                            analysis["alerts"].append({
                                "type": "zone_intrusion",
                                "zone": zone["name"],
                                "object": detection
                            })

        return analysis

    def _box_in_zone(self, box: torch.Tensor, zone: List[float]) -> bool:
        """Verifica si centro del box está en la zona."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (zone[0] <= cx <= zone[2]) and (zone[1] <= cy <= zone[3])
```

## Resumen

| Detector | Velocidad | Accuracy | Uso |
|----------|-----------|----------|-----|
| Faster R-CNN | Lento | Alto | Precisión crítica |
| SSD | Medio | Medio | Balance |
| YOLO | Rápido | Alto | Tiempo real |
| RetinaNet | Medio | Alto | Class imbalance |

### Checklist Detección

```
□ Dataset anotado en formato correcto (COCO, YOLO, VOC)
□ Anchors ajustados al dataset (k-means clustering)
□ Data augmentation preserva boxes (albumentations)
□ Loss balanceada (focal loss si hay class imbalance)
□ NMS threshold ajustado (0.5 típico)
□ Confidence threshold para deployment
□ mAP evaluado en múltiples IoU thresholds
```

## Referencias

- Rich feature hierarchies for accurate object detection (R-CNN)
- Faster R-CNN: Towards Real-Time Object Detection
- You Only Look Once: Unified, Real-Time Object Detection
- SSD: Single Shot MultiBox Detector
- Focal Loss for Dense Object Detection (RetinaNet)
