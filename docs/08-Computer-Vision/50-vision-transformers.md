# Vision Transformers (ViT)

## Introducción

Los Vision Transformers aplican la arquitectura Transformer (originalmente para NLP) directamente a imágenes. En lugar de procesar secuencias de tokens de texto, procesan secuencias de "patches" de imagen.

```
Transformer vs CNN:

CNN:                              Transformer:
┌─────────────────────────┐       ┌─────────────────────────┐
│ Inductive bias local:   │       │ Sin inductive bias:     │
│ - Localidad             │       │ - Aprende todo          │
│ - Invarianza traslación │       │ - Necesita más datos    │
│ - Jerárquico            │       │ - Atención global       │
└─────────────────────────┘       └─────────────────────────┘

CNN: Bueno con pocos datos, estructura fija
ViT: Mejor con muchos datos, más flexible
```

## Arquitectura ViT

### Proceso de Patchify

```
Imagen a Patches:

Imagen 224×224×3
┌─────────────────────────────────────┐
│ ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
│ │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ │   Patches 16×16
│ ├───┼───┼───┼───┼───┼───┼───┼───┤ │   → 14×14 = 196 patches
│ │ 9 │10 │...│...│...│...│...│16 │ │
│ ├───┼───┼───┼───┼───┼───┼───┼───┤ │   Cada patch: 16×16×3 = 768 valores
│ │...│...│...│...│...│...│...│...│ │
│ ...                               │   Flatten → 196 tokens de dim 768
│ │...│...│...│...│...│...│..│196│ │
│ └───┴───┴───┴───┴───┴───┴───┴───┘ │
└─────────────────────────────────────┘

        ↓ Linear Projection

Secuencia: [CLS, P1, P2, ..., P196] + Positional Embeddings
           ↓
    Transformer Encoder (12 capas)
           ↓
    [CLS] → MLP Head → Clasificación
```

### Implementación

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Divide imagen en patches y proyecta a embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Proyección lineal de patches (equivalente a conv con kernel=stride=patch_size)
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)

        Returns:
            (batch, num_patches, embed_dim)
        """
        # (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)

        Returns:
            (batch, seq_len, embed_dim)
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """MLP block en Transformer."""

    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Bloque Transformer: Attention + MLP con residuals."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT).

    Configuraciones estándar:
    - ViT-B/16: 12 layers, 768 dim, 12 heads, patch=16
    - ViT-L/16: 24 layers, 1024 dim, 16 heads, patch=16
    - ViT-H/14: 32 layers, 1280 dim, 16 heads, patch=14
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Classification: usar CLS token
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Extrae attention maps para visualización."""
        B = x.shape[0]
        attention_maps = []

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            # Extraer attention antes de aplicar
            with torch.no_grad():
                qkv = block.attn.qkv(block.norm1(x))
                qkv = qkv.reshape(B, -1, 3, block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attention_maps.append(attn.cpu())

            x = block(x)

        return attention_maps


# Crear ViT-B/16
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

params = sum(p.numel() for p in model.parameters())
print(f"ViT-B/16 parámetros: {params:,}")  # ~86M
```

## Variantes de ViT

### DeiT (Data-efficient Image Transformers)

```python
class DeiT(VisionTransformer):
    """
    DeiT: ViT entrenado eficientemente con distillation.

    Añade:
    - Distillation token (aprende de CNN teacher)
    - Data augmentation fuerte (RandAugment, Mixup)
    - Regularización (Stochastic Depth, dropout)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_distillation: bool = True
    ):
        super().__init__(
            image_size, patch_size, 3, num_classes,
            embed_dim, depth, num_heads, mlp_ratio, dropout
        )

        self.use_distillation = use_distillation

        if use_distillation:
            # Distillation token
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Actualizar positional embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim)
            )
            # Distillation head
            self.head_dist = nn.Linear(embed_dim, num_classes)

            nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.use_distillation:
            cls_output = x[:, 0]
            dist_output = x[:, 1]

            if self.training:
                # Durante entrenamiento, retornar ambas salidas
                return self.head(cls_output), self.head_dist(dist_output)
            else:
                # Durante inferencia, promediar
                return (self.head(cls_output) + self.head_dist(dist_output)) / 2
        else:
            return self.head(x[:, 0])
```

### Swin Transformer

```python
class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Attention.
    Computa attention solo dentro de ventanas locales.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.

    Alterna entre:
    - Window Multi-Head Self-Attention (W-MSA)
    - Shifted Window Multi-Head Self-Attention (SW-MSA)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad for window partition
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Shift if needed
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Window partition
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        # Window attention
        x = self.attn(x)

        # Reverse window partition
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Divide feature map en ventanas no-solapadas."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruye feature map desde ventanas."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
```

## CLIP (Contrastive Language-Image Pre-training)

### Concepto

```
CLIP: Aprende representaciones imagen-texto alineadas.

                   Image Encoder              Text Encoder
                   ┌───────────┐              ┌───────────┐
 Imagen ──────────►│    ViT    │──►I₁         │Transformer│◄────── "a dog"
                   └───────────┘              └───────────┘
                                      ↘                ↙
                                        Contrastive
                                          Loss
                                      ↗                ↘
 Imagen ──────────►│    ViT    │──►I₂         │Transformer│◄────── "a cat"
                   └───────────┘              └───────────┘

Objetivo: Maximizar similitud (Iᵢ, Tᵢ) para pares correctos.
          Minimizar similitud para pares incorrectos.

Zero-shot classification:
1. Crear prompts: "a photo of a {class}"
2. Encodear imagen y todos los prompts
3. Clasificar según máxima similitud
```

```python
import clip
from PIL import Image


class CLIPClassifier:
    """
    Clasificador zero-shot usando CLIP.
    No necesita entrenamiento específico.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    @torch.no_grad()
    def classify(
        self,
        image: Image.Image,
        class_names: list,
        prompt_template: str = "a photo of a {}"
    ) -> dict:
        """
        Clasificación zero-shot.

        Args:
            image: PIL Image
            class_names: Lista de clases posibles
            prompt_template: Template para generar prompts
        """
        # Preprocesar imagen
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Crear prompts para cada clase
        prompts = [prompt_template.format(c) for c in class_names]
        text_inputs = clip.tokenize(prompts).to(self.device)

        # Encodear
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_inputs)

        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calcular similitud
        similarity = (image_features @ text_features.T).squeeze(0)
        probs = similarity.softmax(dim=-1)

        # Resultados
        pred_idx = probs.argmax().item()

        return {
            "prediction": class_names[pred_idx],
            "confidence": probs[pred_idx].item(),
            "all_probabilities": {
                name: probs[i].item()
                for i, name in enumerate(class_names)
            }
        }

    @torch.no_grad()
    def compute_image_embeddings(self, images: list) -> torch.Tensor:
        """Computa embeddings para lista de imágenes."""
        inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(inputs)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def compute_text_embeddings(self, texts: list) -> torch.Tensor:
        """Computa embeddings para lista de textos."""
        inputs = clip.tokenize(texts).to(self.device)
        features = self.model.encode_text(inputs)
        return features / features.norm(dim=-1, keepdim=True)


# Ejemplo: Detección de contenido malicioso
classifier = CLIPClassifier()

# Clases para detectar
security_classes = [
    "legitimate website screenshot",
    "phishing page screenshot",
    "malware warning popup",
    "fake login page",
    "normal desktop screenshot"
]

# Clasificar
result = classifier.classify(
    image=Image.open("screenshot.png"),
    class_names=security_classes
)

print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## DINO (Self-Distillation with No Labels)

```python
class DINO:
    """
    DINO: Self-supervised ViT.
    Aprende features sin etiquetas usando self-distillation.

    Útil para:
    - Transfer learning con pocos datos
    - Segmentación sin supervisión
    - Feature extraction general
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        # Cargar modelo preentrenado
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.model = self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extrae features de imagen."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Extraer features del CLS token
        features = self.model(image)
        return features

    @torch.no_grad()
    def get_attention_maps(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extrae attention maps del último bloque.
        DINO attention localiza objetos sin supervisión.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Hook para capturar attention
        attentions = []
        def hook_fn(module, input, output):
            # output es (attn_output, attn_weights)
            attentions.append(output[1])

        # Registrar hook en último bloque
        handle = self.model.blocks[-1].attn.register_forward_hook(hook_fn)

        # Forward
        _ = self.model(image)

        handle.remove()

        if attentions:
            return attentions[0]
        return None

    def visualize_attention(
        self,
        image: torch.Tensor,
        patch_size: int = 8
    ) -> torch.Tensor:
        """
        Visualiza attention del CLS token sobre patches.
        Muestra qué partes de la imagen son importantes.
        """
        attn = self.get_attention_maps(image)
        if attn is None:
            return None

        # Attention del CLS token hacia todos los patches
        # attn shape: (batch, heads, num_tokens, num_tokens)
        cls_attn = attn[0, :, 0, 1:]  # (heads, num_patches)

        # Promediar sobre heads
        cls_attn = cls_attn.mean(dim=0)

        # Reshape a grid
        H = W = int(cls_attn.shape[0] ** 0.5)
        cls_attn = cls_attn.reshape(H, W)

        # Upsample a tamaño original
        cls_attn = F.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),
            scale_factor=patch_size,
            mode='bilinear'
        ).squeeze()

        return cls_attn.cpu()
```

## Transfer Learning con ViT

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTTransferLearning:
    """
    Transfer learning con Vision Transformer preentrenado.
    """

    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        self.device = device

        # Cargar ViT preentrenado
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Congelar backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Reemplazar cabeza de clasificación
        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads[0].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.model = self.model.to(device)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def unfreeze_last_n_blocks(self, n: int):
        """Descongela últimos n bloques para fine-tuning."""
        total_blocks = len(self.model.encoder.layers)

        for i, block in enumerate(self.model.encoder.layers):
            if i >= total_blocks - n:
                for param in block.parameters():
                    param.requires_grad = True

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Paso de entrenamiento."""
        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predicción."""
        self.model.eval()
        images = images.to(self.device)
        outputs = self.model(images)
        return outputs.softmax(dim=-1).cpu()


# Ejemplo
vit_classifier = ViTTransferLearning(num_classes=10, freeze_backbone=True)
print(f"Parámetros entrenables: {vit_classifier.get_trainable_params():,}")

# Fine-tuning gradual
vit_classifier.unfreeze_last_n_blocks(2)
print(f"Después de unfreeze: {vit_classifier.get_trainable_params():,}")
```

## Aplicaciones en Ciberseguridad

### Clasificador Multi-modal con CLIP

```python
class SecurityCLIPAnalyzer:
    """
    Análisis de seguridad multi-modal con CLIP.
    Combina análisis visual y textual.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

        # Prompts de seguridad
        self.security_prompts = {
            "phishing": [
                "a screenshot of a phishing website",
                "a fake login page trying to steal credentials",
                "a deceptive website imitating a legitimate brand"
            ],
            "legitimate": [
                "a screenshot of a legitimate website",
                "an authentic login page",
                "a genuine corporate website"
            ],
            "malware_warning": [
                "a fake virus warning popup",
                "a scareware alert message",
                "a tech support scam popup"
            ],
            "ransomware": [
                "a ransomware encryption notice",
                "a file encryption warning screen",
                "a ransom payment demand"
            ]
        }

    @torch.no_grad()
    def analyze_screenshot(self, image: Image.Image) -> dict:
        """Analiza screenshot para amenazas de seguridad."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        results = {}

        for category, prompts in self.security_prompts.items():
            text_inputs = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Similitud promedio para la categoría
            similarity = (image_features @ text_features.T).squeeze(0)
            results[category] = similarity.mean().item()

        # Determinar amenaza principal
        threat_scores = {k: v for k, v in results.items() if k != "legitimate"}
        max_threat = max(threat_scores, key=threat_scores.get)
        max_threat_score = threat_scores[max_threat]
        legitimate_score = results["legitimate"]

        return {
            "scores": results,
            "primary_threat": max_threat if max_threat_score > legitimate_score else None,
            "threat_confidence": max_threat_score,
            "legitimate_confidence": legitimate_score,
            "risk_level": self._calculate_risk(max_threat_score, legitimate_score),
            "recommendation": self._get_recommendation(max_threat, max_threat_score, legitimate_score)
        }

    def _calculate_risk(self, threat: float, legitimate: float) -> str:
        if threat > legitimate + 0.1:
            return "HIGH"
        elif threat > legitimate:
            return "MEDIUM"
        return "LOW"

    def _get_recommendation(self, threat: str, threat_score: float, legit_score: float) -> str:
        if threat_score > legit_score + 0.1:
            recommendations = {
                "phishing": "⚠️ NO introducir credenciales. Verificar URL.",
                "malware_warning": "⚠️ Popup sospechoso. Cerrar navegador.",
                "ransomware": "⚠️ DESCONECTAR de red. Contactar IT."
            }
            return recommendations.get(threat, "Proceder con precaución.")
        return "Página parece legítima."


class ViTMalwareVisualizer:
    """
    Visualización de malware usando ViT attention.
    Identifica regiones importantes en visualizaciones de binarios.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.dino = DINO(device)

    def analyze_binary_image(self, binary_image: torch.Tensor) -> dict:
        """
        Analiza visualización de binario.
        Identifica regiones sospechosas basado en attention.
        """
        # Normalizar imagen
        if binary_image.dim() == 2:
            binary_image = binary_image.unsqueeze(0).repeat(3, 1, 1)

        # Obtener attention map
        attn_map = self.dino.visualize_attention(binary_image)

        # Extraer features
        features = self.dino.extract_features(binary_image)

        # Identificar regiones de alta atención
        threshold = attn_map.mean() + attn_map.std()
        high_attention_mask = attn_map > threshold

        # Calcular estadísticas
        high_attention_ratio = high_attention_mask.float().mean().item()

        return {
            "attention_map": attn_map,
            "features": features.cpu(),
            "high_attention_regions": high_attention_mask,
            "high_attention_ratio": high_attention_ratio,
            "anomaly_score": self._calculate_anomaly_score(features, high_attention_ratio)
        }

    def _calculate_anomaly_score(self, features: torch.Tensor, attn_ratio: float) -> float:
        """Calcula score de anomalía basado en features y attention."""
        # Heurística simple: malware tiende a tener patrones más concentrados
        feature_norm = features.norm().item()
        return min(1.0, attn_ratio * 2 + feature_norm / 100)
```

## Resumen

| Modelo | Parámetros | Características |
|--------|------------|-----------------|
| ViT-B/16 | 86M | Balance, general |
| ViT-L/16 | 307M | Alta precisión |
| DeiT | 86M | Eficiente, distillation |
| Swin | Variable | Jerárquico, eficiente |
| CLIP | 400M+ | Multi-modal, zero-shot |
| DINO | Variable | Self-supervised |

### Checklist ViT

```
□ Datos suficientes (>10K para fine-tuning, >100K para pretrain)
□ Patch size apropiado (16 para general, 14/8 para detalle)
□ Augmentation fuerte (RandAugment, Mixup, CutMix)
□ Learning rate warmup (critical para Transformers)
□ Weight decay alto (0.3) para regularización
□ Gradient clipping para estabilidad
□ Label smoothing para mejor calibración
```

## Referencias

- An Image is Worth 16x16 Words: Transformers for Image Recognition
- DeiT: Training data-efficient image transformers
- Swin Transformer: Hierarchical Vision Transformer
- Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- Emerging Properties in Self-Supervised Vision Transformers (DINO)
