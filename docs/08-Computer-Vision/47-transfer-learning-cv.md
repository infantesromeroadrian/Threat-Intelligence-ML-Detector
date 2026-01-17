# Transfer Learning en Computer Vision

## Concepto

Transfer Learning reutiliza conocimiento aprendido en una tarea (source) para mejorar el rendimiento en otra tarea (target). En CV, típicamente usamos modelos preentrenados en ImageNet (14M imágenes, 1000 clases) como punto de partida.

```
Transfer Learning:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Source Task (ImageNet)              Target Task (Tu problema)     │
│   ┌─────────────────────┐             ┌─────────────────────┐      │
│   │ 14M imágenes        │             │ 1K-10K imágenes     │      │
│   │ 1000 clases         │   ──────►   │ 2-100 clases        │      │
│   │ Objetos genéricos   │  Transfer   │ Dominio específico  │      │
│   └─────────────────────┘             └─────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

¿Por qué funciona?
- Capas iniciales: detectan bordes, texturas (universales)
- Capas medias: detectan patrones complejos (transferibles)
- Capas finales: específicas de la tarea (reemplazar)
```

## Estrategias de Transfer Learning

### 1. Feature Extraction

```
Feature Extraction:
Congelar backbone, entrenar solo clasificador.

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Backbone (Frozen)              Classifier (Trainable)    │
│   ┌─────────────────────┐       ┌─────────────────────┐    │
│   │ ❄️ Conv1            │       │ 🔥 FC 1024          │    │
│   │ ❄️ Conv2            │  ──►  │ 🔥 FC 512           │    │
│   │ ❄️ Conv3            │       │ 🔥 FC num_classes   │    │
│   │ ❄️ ...              │       └─────────────────────┘    │
│   └─────────────────────┘                                   │
│                                                             │
│   ❄️ = Pesos congelados (no se actualizan)                 │
│   🔥 = Pesos entrenables                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Cuándo usar:
- Dataset pequeño (<1K imágenes)
- Dominio similar a ImageNet
- Recursos computacionales limitados
```

### 2. Fine-tuning

```
Fine-tuning:
Descongelar parte del backbone y entrenar con LR bajo.

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Backbone                       Classifier                 │
│   ┌─────────────────────┐       ┌─────────────────────┐    │
│   │ ❄️ Conv1 (frozen)   │       │ 🔥 FC 1024          │    │
│   │ ❄️ Conv2 (frozen)   │  ──►  │ 🔥 FC 512           │    │
│   │ 🔥 Conv3 (lr=1e-5)  │       │ 🔥 FC num_classes   │    │
│   │ 🔥 Conv4 (lr=1e-4)  │       │    (lr=1e-3)        │    │
│   └─────────────────────┘       └─────────────────────┘    │
│                                                             │
│   Learning rates discriminativos:                          │
│   - Capas bajas: lr muy pequeño (features genéricos)       │
│   - Capas altas: lr más alto (adaptar a dominio)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Cuándo usar:
- Dataset medio (1K-100K imágenes)
- Dominio diferente a ImageNet
- Necesitas accuracy máximo
```

## Implementación con PyTorch

### Feature Extraction

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from typing import Tuple, List


class FeatureExtractor:
    """
    Transfer learning como feature extraction.
    Congela backbone, entrena solo clasificador.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        device: str = "cuda"
    ):
        self.device = device
        self.model = self._create_model(model_name, num_classes)
        self.model = self.model.to(device)

    def _create_model(
        self,
        model_name: str,
        num_classes: int
    ) -> nn.Module:
        """Crea modelo con backbone congelado."""

        if model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Congelar backbone
            for param in model.parameters():
                param.requires_grad = False
            # Nuevo clasificador (trainable por defecto)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )

        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )

        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        return model

    def get_trainable_params(self) -> int:
        """Cuenta parámetros entrenables."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Cuenta parámetros totales."""
        return sum(p.numel() for p in self.model.parameters())

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-3
    ) -> dict:
        """Entrena solo el clasificador."""

        # Solo parámetros entrenables
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3
        )

        history = {'train_loss': [], 'val_acc': []}
        best_acc = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            val_acc = self._evaluate(val_loader)
            scheduler.step(val_acc)

            history['train_loss'].append(running_loss / len(train_loader))
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        """Evalúa accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return correct / total


# Ejemplo de uso
extractor = FeatureExtractor("resnet50", num_classes=10)
print(f"Parámetros totales: {extractor.get_total_params():,}")
print(f"Parámetros entrenables: {extractor.get_trainable_params():,}")
# ResNet50: ~25M totales, ~20K entrenables (solo FC)
```

### Fine-tuning

```python
class FineTuner:
    """
    Fine-tuning: Descongelar capas superiores del backbone.
    Usa learning rates discriminativos.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        unfreeze_layers: int = 2,
        device: str = "cuda"
    ):
        self.device = device
        self.model_name = model_name
        self.model = self._create_model(model_name, num_classes, unfreeze_layers)
        self.model = self.model.to(device)

    def _create_model(
        self,
        model_name: str,
        num_classes: int,
        unfreeze_layers: int
    ) -> nn.Module:
        """Crea modelo con capas parcialmente congeladas."""

        if model_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Congelar todo primero
            for param in model.parameters():
                param.requires_grad = False

            # Descongelar últimas N capas
            layers = [model.layer4, model.layer3, model.layer2, model.layer1]
            for i, layer in enumerate(layers[:unfreeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = True

            # Nuevo clasificador
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

            for param in model.parameters():
                param.requires_grad = False

            # EfficientNet tiene features.X donde X es índice de bloque
            # Descongelar últimos bloques
            total_blocks = len(model.features)
            for i in range(total_blocks - unfreeze_layers, total_blocks):
                for param in model.features[i].parameters():
                    param.requires_grad = True

            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )

        return model

    def get_parameter_groups(self) -> List[dict]:
        """
        Crea grupos de parámetros con diferentes learning rates.
        Capas más profundas = LR más bajo.
        """
        if self.model_name == "resnet50":
            return [
                # Backbone congelado (no incluir)
                {'params': self.model.layer3.parameters(), 'lr': 1e-5},
                {'params': self.model.layer4.parameters(), 'lr': 1e-4},
                {'params': self.model.fc.parameters(), 'lr': 1e-3},
            ]
        else:
            # Para EfficientNet
            backbone_params = []
            classifier_params = list(self.model.classifier.parameters())

            for name, param in self.model.features.named_parameters():
                if param.requires_grad:
                    backbone_params.append(param)

            return [
                {'params': backbone_params, 'lr': 1e-5},
                {'params': classifier_params, 'lr': 1e-3},
            ]

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20
    ) -> dict:
        """Entrena con learning rates discriminativos."""

        param_groups = self.get_parameter_groups()
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Warmup + Cosine decay
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g['lr'] for g in param_groups],
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

        history = {'train_loss': [], 'val_acc': []}
        best_acc = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            # Validation
            val_acc = self._evaluate(val_loader)

            history['train_loss'].append(running_loss / len(train_loader))
            history['val_acc'].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_finetuned.pth')

            current_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} "
                  f"- Val Acc: {val_acc:.4f} - LRs: {current_lrs}")

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return correct / total


# Comparación
print("=== Feature Extraction ===")
fe = FeatureExtractor("resnet50", num_classes=10)
print(f"Trainable: {fe.get_trainable_params():,}")

print("\n=== Fine-tuning (2 layers) ===")
ft = FineTuner("resnet50", num_classes=10, unfreeze_layers=2)
trainable = sum(p.numel() for p in ft.model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,}")
```

## Gradual Unfreezing

### Estrategia Progresiva

```python
class GradualUnfreezer:
    """
    Gradual Unfreezing: Descongelar capas progresivamente.

    Epoch 1-5: Solo clasificador
    Epoch 6-10: + última capa conv
    Epoch 11-15: + penúltima capa conv
    ...

    Permite que capas superiores se adapten antes de
    perturbar features de capas inferiores.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 10,
        device: str = "cuda"
    ):
        self.device = device
        self.model = self._create_model(model_name, num_classes)
        self.model = self.model.to(device)

        # Guardar referencia a capas para descongelar
        if model_name == "resnet50":
            self.layer_groups = [
                self.model.fc,        # Clasificador
                self.model.layer4,    # Última capa residual
                self.model.layer3,    # Penúltima
                self.model.layer2,    # Etc.
                self.model.layer1,
            ]
        self.unfrozen_groups = 1  # Empezar solo con clasificador

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Congelar TODO
        for param in model.parameters():
            param.requires_grad = False

        # Solo clasificador trainable inicialmente
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

        return model

    def unfreeze_next_group(self):
        """Descongela el siguiente grupo de capas."""
        if self.unfrozen_groups < len(self.layer_groups):
            layer = self.layer_groups[self.unfrozen_groups]
            for param in layer.parameters():
                param.requires_grad = True
            self.unfrozen_groups += 1
            return True
        return False

    def get_optimizer(self, base_lr: float = 1e-3) -> optim.Optimizer:
        """Crea optimizador con LR discriminativo."""
        param_groups = []

        for i, layer in enumerate(self.layer_groups[:self.unfrozen_groups]):
            # LR decrece para capas más profundas
            lr = base_lr * (0.1 ** i)
            params = [p for p in layer.parameters() if p.requires_grad]
            if params:
                param_groups.append({'params': params, 'lr': lr})

        return optim.Adam(param_groups)

    def train_with_gradual_unfreezing(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs_per_stage: int = 5,
        num_stages: int = 4
    ) -> dict:
        """Entrena con descongelamiento gradual."""

        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'val_acc': [], 'unfrozen': []}

        for stage in range(num_stages):
            print(f"\n=== Stage {stage + 1}: {self.unfrozen_groups} grupos descongelados ===")

            optimizer = self.get_optimizer()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs_per_stage
            )

            for epoch in range(epochs_per_stage):
                self.model.train()
                running_loss = 0

                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                scheduler.step()

                val_acc = self._evaluate(val_loader)
                history['train_loss'].append(running_loss / len(train_loader))
                history['val_acc'].append(val_acc)
                history['unfrozen'].append(self.unfrozen_groups)

                print(f"  Epoch {epoch+1}/{epochs_per_stage} - "
                      f"Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")

            # Descongelar siguiente grupo
            self.unfreeze_next_group()

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return correct / total
```

## Data Augmentation para Transfer Learning

### Augmentation Específico

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TransferLearningAugmentation:
    """
    Data augmentation optimizado para transfer learning.

    Consideraciones:
    - ImageNet usa normalización específica
    - Augmentation agresivo puede dañar features preentrenados
    - Test Time Augmentation (TTA) mejora resultados
    """

    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @classmethod
    def get_train_transform(
        cls,
        image_size: int = 224,
        augment_strength: str = "medium"
    ) -> A.Compose:
        """
        Transformaciones de entrenamiento.

        augment_strength:
        - "light": Para datasets muy pequeños o dominios muy similares a ImageNet
        - "medium": Balance estándar
        - "heavy": Para datasets grandes o dominios muy diferentes
        """

        if augment_strength == "light":
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
                ToTensorV2()
            ])

        elif augment_strength == "medium":
            return A.Compose([
                A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20),
                ], p=0.3),
                A.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
                ToTensorV2()
            ])

        else:  # heavy
            return A.Compose([
                A.RandomResizedCrop(image_size, image_size, scale=(0.6, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7
                ),
                A.OneOf([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30),
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8, max_height=16, max_width=16,
                    fill_value=0, p=0.3
                ),
                A.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
                ToTensorV2()
            ])

    @classmethod
    def get_val_transform(cls, image_size: int = 224) -> A.Compose:
        """Transformaciones de validación (determinísticas)."""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ToTensorV2()
        ])

    @classmethod
    def get_tta_transforms(cls, image_size: int = 224) -> List[A.Compose]:
        """
        Test Time Augmentation transforms.
        Aplica múltiples transformaciones y promedia predicciones.
        """
        base_normalize = [
            A.Normalize(mean=cls.IMAGENET_MEAN, std=cls.IMAGENET_STD),
            ToTensorV2()
        ]

        return [
            # Original
            A.Compose([A.Resize(image_size, image_size)] + base_normalize),
            # Horizontal flip
            A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=1.0)
            ] + base_normalize),
            # Slight rotation left
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=(-10, -10), p=1.0)
            ] + base_normalize),
            # Slight rotation right
            A.Compose([
                A.Resize(image_size, image_size),
                A.Rotate(limit=(10, 10), p=1.0)
            ] + base_normalize),
            # Center crop
            A.Compose([
                A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
                A.CenterCrop(image_size, image_size)
            ] + base_normalize),
        ]


class TTAPredictor:
    """Predictor con Test Time Augmentation."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.tta_transforms = TransferLearningAugmentation.get_tta_transforms()

    @torch.no_grad()
    def predict_with_tta(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Predicción con TTA.
        Promedia probabilidades de múltiples augmentaciones.
        """
        self.model.eval()
        all_probs = []

        # image es numpy array (H, W, C)
        for transform in self.tta_transforms:
            augmented = transform(image=image)['image']
            augmented = augmented.unsqueeze(0).to(self.device)

            logits = self.model(augmented)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

        # Promediar probabilidades
        avg_probs = torch.stack(all_probs).mean(dim=0)[0]

        pred_class = avg_probs.argmax().item()
        confidence = avg_probs[pred_class].item()

        return pred_class, confidence
```

## Selección de Modelo Preentrenado

### Guía de Selección

```python
class ModelSelector:
    """
    Guía para seleccionar modelo preentrenado según el caso de uso.
    """

    MODELS = {
        "resnet18": {
            "params": "11M",
            "accuracy": "69.8%",
            "speed": "fast",
            "memory": "low",
            "use_case": "Recursos limitados, móvil, edge"
        },
        "resnet50": {
            "params": "25M",
            "accuracy": "76.1%",
            "speed": "medium",
            "memory": "medium",
            "use_case": "Balance general, producción estándar"
        },
        "resnet152": {
            "params": "60M",
            "accuracy": "78.3%",
            "speed": "slow",
            "memory": "high",
            "use_case": "Accuracy máximo, sin restricciones"
        },
        "efficientnet_b0": {
            "params": "5.3M",
            "accuracy": "77.1%",
            "speed": "fast",
            "memory": "low",
            "use_case": "Mejor eficiencia, móvil, edge"
        },
        "efficientnet_b4": {
            "params": "19M",
            "accuracy": "82.9%",
            "speed": "medium",
            "memory": "medium",
            "use_case": "Alto accuracy con eficiencia"
        },
        "vit_b_16": {
            "params": "86M",
            "accuracy": "81.1%",
            "speed": "slow",
            "memory": "high",
            "use_case": "Dominios muy diferentes a ImageNet"
        },
    }

    @classmethod
    def recommend(
        cls,
        dataset_size: int,
        domain_similarity: str,
        resource_constraint: str
    ) -> str:
        """
        Recomienda modelo basado en restricciones.

        Args:
            dataset_size: Número de imágenes de entrenamiento
            domain_similarity: "high", "medium", "low" respecto a ImageNet
            resource_constraint: "none", "medium", "strict"
        """

        # Dataset muy pequeño → menos parámetros para evitar overfit
        if dataset_size < 1000:
            if resource_constraint == "strict":
                return "efficientnet_b0"
            return "resnet18"

        # Dataset pequeño-medio
        elif dataset_size < 10000:
            if resource_constraint == "strict":
                return "efficientnet_b0"
            elif domain_similarity == "low":
                return "efficientnet_b4"  # Mejor generalización
            return "resnet50"

        # Dataset grande
        else:
            if resource_constraint == "strict":
                return "efficientnet_b0"
            elif resource_constraint == "medium":
                return "efficientnet_b4"
            elif domain_similarity == "low":
                return "vit_b_16"  # Transformers mejor para dominios diferentes
            return "resnet50"

    @classmethod
    def print_comparison(cls):
        """Imprime tabla comparativa."""
        print("\n" + "="*80)
        print("COMPARATIVA DE MODELOS PREENTRENADOS")
        print("="*80)

        for name, info in cls.MODELS.items():
            print(f"\n{name.upper()}")
            for key, value in info.items():
                print(f"  {key}: {value}")


ModelSelector.print_comparison()
```

## Domain Adaptation

### Cuando el Dominio es Muy Diferente

```python
class DomainAdaptation:
    """
    Técnicas cuando el dominio target es muy diferente de ImageNet.

    Ejemplos:
    - Imágenes médicas (rayos X, MRI)
    - Imágenes satelitales
    - Imágenes industriales (defectos)
    - Screenshots/UI
    """

    @staticmethod
    def domain_specific_normalization(domain: str) -> Tuple[list, list]:
        """
        Normalización específica por dominio.
        En lugar de ImageNet stats, usar stats del dominio.
        """
        # Valores ejemplo - calcular con tu dataset
        domain_stats = {
            "medical_xray": {
                "mean": [0.5025],  # Grayscale
                "std": [0.2481]
            },
            "satellite": {
                "mean": [0.3847, 0.3847, 0.3096],  # RGB pero diferente
                "std": [0.1753, 0.1612, 0.1503]
            },
            "industrial": {
                "mean": [0.4510, 0.4510, 0.4510],
                "std": [0.2260, 0.2260, 0.2260]
            },
            "screenshots": {
                "mean": [0.9, 0.9, 0.9],  # Fondos claros
                "std": [0.15, 0.15, 0.15]
            }
        }

        if domain in domain_stats:
            return domain_stats[domain]["mean"], domain_stats[domain]["std"]

        # Default: calcular del dataset
        return None, None

    @staticmethod
    def calculate_dataset_stats(
        dataset: torch.utils.data.Dataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula mean y std del dataset."""

        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        mean = torch.zeros(3)
        std = torch.zeros(3)
        n_samples = 0

        for images, _ in loader:
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            n_samples += batch_size

        mean /= n_samples
        std /= n_samples

        return mean, std


class MedicalImageTransfer(nn.Module):
    """
    Transfer learning especializado para imágenes médicas.

    Modificaciones:
    - Primer conv adaptado a 1 canal (grayscale)
    - Normalización específica
    - Arquitectura más conservadora
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: str = "resnet50"
    ):
        super().__init__()

        # Cargar backbone
        if pretrained_backbone == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Modificar primera conv para 1 canal
            # Promediamos pesos RGB para grayscale
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Inicializar con promedio de pesos RGB
            with torch.no_grad():
                backbone.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

            # Congelar capas iniciales
            for param in backbone.conv1.parameters():
                param.requires_grad = False
            for param in backbone.bn1.parameters():
                param.requires_grad = False
            for param in backbone.layer1.parameters():
                param.requires_grad = False
            for param in backbone.layer2.parameters():
                param.requires_grad = False

            # Clasificador
            backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(backbone.fc.in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

            self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Ejemplo para rayos X
model = MedicalImageTransfer(num_classes=2)  # Normal vs Anomalía
print(f"Modelo médico creado para imágenes grayscale")
```

## Aplicaciones en Ciberseguridad

### Detector de Phishing Visual

```python
class PhishingScreenshotDetector:
    """
    Detector de páginas de phishing usando transfer learning.
    Analiza screenshots de páginas web.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Usar EfficientNet por eficiencia
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Congelar backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Descongelar últimos bloques
        for param in self.model.features[-3:].parameters():
            param.requires_grad = True

        # Clasificador para phishing
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier[1].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # Legítimo, Phishing, Sospechoso
        )

        self.model = self.model.to(device)

        self.class_names = ["Legítimo", "Phishing", "Sospechoso"]

        # Transform para screenshots
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @torch.no_grad()
    def analyze_screenshot(self, image_path: str) -> dict:
        """Analiza screenshot y devuelve predicción."""
        import cv2

        # Cargar imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transformar
        transformed = self.transform(image=image)['image']
        transformed = transformed.unsqueeze(0).to(self.device)

        # Predicción
        self.model.eval()
        logits = self.model(transformed)
        probs = torch.softmax(logits, dim=1)[0]

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        return {
            "prediction": self.class_names[pred_idx],
            "confidence": confidence,
            "probabilities": {
                name: probs[i].item()
                for i, name in enumerate(self.class_names)
            },
            "risk_level": "HIGH" if pred_idx == 1 and confidence > 0.8 else
                          "MEDIUM" if pred_idx in [1, 2] else "LOW",
            "recommendation": self._get_recommendation(pred_idx, confidence)
        }

    def _get_recommendation(self, pred_idx: int, confidence: float) -> str:
        if pred_idx == 1 and confidence > 0.8:
            return "⚠️ ALTO RIESGO: No introducir credenciales. Verificar URL manualmente."
        elif pred_idx == 1:
            return "⚠️ Posible phishing. Verificar legitimidad antes de proceder."
        elif pred_idx == 2:
            return "Página sospechosa. Proceder con precaución."
        return "Página parece legítima."


class MalwareVisualDetector:
    """
    Detector visual de malware basado en visualización de binarios.
    Convierte binarios a imágenes y usa CNN para clasificar.
    """

    def __init__(self, num_classes: int = 5, device: str = "cuda"):
        """
        Classes ejemplo:
        0: Benigno
        1: Trojan
        2: Ransomware
        3: Worm
        4: Adware
        """
        self.device = device
        self.num_classes = num_classes

        # ResNet para análisis de patrones
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Modificar primera conv para 1 canal (grayscale del binario)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Congelar la mayoría
        for name, param in self.model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False

        # Clasificador
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        self.model = self.model.to(device)

    @staticmethod
    def binary_to_image(
        binary_path: str,
        image_size: int = 256
    ) -> torch.Tensor:
        """
        Convierte archivo binario a imagen grayscale.
        Técnica usada en malware visualization.
        """
        import numpy as np

        # Leer bytes
        with open(binary_path, 'rb') as f:
            binary_data = f.read()

        # Convertir a array de bytes
        byte_array = np.frombuffer(binary_data, dtype=np.uint8)

        # Calcular dimensiones para imagen cuadrada
        total_pixels = image_size * image_size

        if len(byte_array) < total_pixels:
            # Padding si es muy pequeño
            byte_array = np.pad(byte_array, (0, total_pixels - len(byte_array)))
        else:
            # Truncar si es muy grande
            byte_array = byte_array[:total_pixels]

        # Reshape a imagen
        image = byte_array.reshape(image_size, image_size)

        # Normalizar y convertir a tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dim

        return image

    @torch.no_grad()
    def analyze_binary(self, binary_path: str) -> dict:
        """Analiza binario y predice tipo de malware."""
        self.model.eval()

        # Convertir a imagen
        image = self.binary_to_image(binary_path)
        image = image.unsqueeze(0).to(self.device)  # Add batch dim

        # Predicción
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)[0]

        class_names = ["Benigno", "Trojan", "Ransomware", "Worm", "Adware"]
        pred_idx = probs.argmax().item()

        return {
            "prediction": class_names[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probabilities": {
                name: probs[i].item() for i, name in enumerate(class_names)
            },
            "is_malicious": pred_idx != 0,
            "threat_level": "CRITICAL" if pred_idx in [1, 2] else
                           "HIGH" if pred_idx == 3 else
                           "MEDIUM" if pred_idx == 4 else "SAFE"
        }
```

## Resumen

| Estrategia | Dataset Size | Dominio | Recursos |
|------------|--------------|---------|----------|
| Feature Extraction | < 1K | Similar | Limitados |
| Fine-tuning (top layers) | 1K - 10K | Similar | Medios |
| Fine-tuning (gradual) | 10K - 100K | Diferente | Amplios |
| From scratch | > 100K | Muy diferente | Muy amplios |

### Checklist Transfer Learning

```
□ Normalización ImageNet (mean, std)
□ Data augmentation apropiado al dominio
□ Congelar capas según tamaño de dataset
□ Learning rates discriminativos (menores para capas bajas)
□ Label smoothing para evitar overconfidence
□ Early stopping + model checkpointing
□ Evaluar con TTA para mejorar accuracy final
```

## Referencias

- How transferable are features in deep neural networks? (Yosinski et al.)
- ULMFiT: Universal Language Model Fine-tuning
- Rethinking ImageNet Pre-training (He et al.)
- torchvision models: https://pytorch.org/vision/stable/models.html
