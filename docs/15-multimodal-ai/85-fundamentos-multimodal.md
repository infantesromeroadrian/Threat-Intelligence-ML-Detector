# Fundamentos de Multimodal AI

## Introduccion

La Inteligencia Artificial Multimodal procesa y relaciona informacion de multiples modalidades: texto, imagenes, audio, video, datos sensoriales, etc. A diferencia de los modelos unimodales que operan en un solo dominio, los sistemas multimodales capturan relaciones semanticas entre diferentes tipos de datos.

```
Evolucion de los Modelos:

Unimodal:                    Multimodal:
┌─────────────┐              ┌─────────────────────────────┐
│   Imagen    │              │    Imagen ←──┐              │
│      ↓      │              │              │              │
│    CNN      │              │         Fusion ──→ Output   │
│      ↓      │              │              │              │
│  "perro"    │              │    Texto  ←──┘              │
└─────────────┘              └─────────────────────────────┘

Un modelo por modalidad        Un modelo que entiende
                               relaciones cross-modal
```

## Por Que Multimodal?

### Limitaciones de Modelos Unimodales

```
Escenario: Analisis de contenido en redes sociales

Unimodal (solo texto):
┌──────────────────────────────────────────────────────────┐
│ Post: "Este producto es increible! 🔥"                   │
│                                                          │
│ Analisis texto: POSITIVO ✓                               │
│                                                          │
│ Pero... la imagen muestra producto defectuoso            │
│ Resultado REAL: NEGATIVO (sarcasmo)                      │
└──────────────────────────────────────────────────────────┘

Multimodal (texto + imagen):
┌──────────────────────────────────────────────────────────┐
│ Texto: "Este producto es increible! 🔥"                  │
│ Imagen: [producto roto/defectuoso]                       │
│                                                          │
│ Cross-modal: Detecta incongruencia texto-imagen          │
│ Resultado: SARCASMO/NEGATIVO ✓                           │
└──────────────────────────────────────────────────────────┘
```

### Aplicaciones Clave

```
Dominios de Aplicacion Multimodal:

┌─────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL AI                            │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Healthcare    │    Security     │      Content Understanding  │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ - Radiologia    │ - Deepfake det  │ - Image captioning          │
│ - Histopatologia│ - Fraud detect  │ - VQA (Visual Q&A)          │
│ - Telemedicina  │ - Surveillance  │ - Video understanding       │
│ - Drug discovery│ - Biometrics    │ - Document analysis         │
├─────────────────┼─────────────────┼─────────────────────────────┤
│   Robotics      │   E-commerce    │      Accessibility          │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ - Navigation    │ - Visual search │ - Alt-text generation       │
│ - Manipulation  │ - Product match │ - Audio description         │
│ - Human-robot   │ - Review analys │ - Sign language             │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

## Arquitectura General Multimodal

### Componentes Fundamentales

```
Arquitectura Multimodal Generica:

         Modalidad 1              Modalidad 2              Modalidad N
         (Imagen)                 (Texto)                  (Audio)
             │                        │                        │
             ▼                        ▼                        ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Encoder 1     │     │   Encoder 2     │     │   Encoder N     │
    │   (ViT/CNN)     │     │   (BERT/GPT)    │     │   (Wav2Vec)     │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Projection     │     │  Projection     │     │  Projection     │
    │  to shared dim  │     │  to shared dim  │     │  to shared dim  │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   FUSION MODULE     │
                          │  (Attention/Concat) │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Task-Specific     │
                          │   Head(s)           │
                          └─────────────────────┘
```

### Implementacion Base

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """Configuracion para una modalidad."""
    name: str
    input_dim: int
    hidden_dim: int
    output_dim: int  # Dimension compartida


class ModalityEncoder(nn.Module, ABC):
    """
    Clase base abstracta para encoders de modalidad.
    Cada modalidad implementa su propio encoder.
    """

    def __init__(self, config: ModalityConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.hidden_dim, config.output_dim)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input to hidden representation."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project to shared space."""
        hidden = self.encode(x)
        return self.projection(hidden)


class ImageEncoder(ModalityEncoder):
    """
    Encoder para modalidad de imagen usando CNN simple.
    En produccion: usar ViT, ResNet, etc.
    """

    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        # CNN simple para ejemplo
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(256, config.hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) imagen
        Returns:
            (batch, hidden_dim) features
        """
        features = self.conv_layers(x)
        features = features.flatten(1)
        return self.fc(features)


class TextEncoder(ModalityEncoder):
    """
    Encoder para modalidad de texto.
    En produccion: usar BERT, RoBERTa, etc.
    """

    def __init__(
        self,
        config: ModalityConfig,
        vocab_size: int = 30000,
        max_seq_len: int = 512
    ):
        super().__init__(config)

        self.embedding = nn.Embedding(vocab_size, config.input_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, config.input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=8,
            dim_feedforward=config.input_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.fc = nn.Linear(config.input_dim, config.hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids
        Returns:
            (batch, hidden_dim) features
        """
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embeddings = self.embedding(x) + self.pos_embedding(positions)

        # Transformer
        encoded = self.transformer(embeddings)

        # Pool: usar [CLS] token o mean pooling
        pooled = encoded.mean(dim=1)  # Mean pooling

        return self.fc(pooled)


class AudioEncoder(ModalityEncoder):
    """
    Encoder para modalidad de audio.
    En produccion: usar Wav2Vec2, HuBERT, etc.
    """

    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        # Conv1D para procesar espectrograma/waveform
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(512, config.hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features, time) espectrograma
        Returns:
            (batch, hidden_dim) features
        """
        features = self.conv_layers(x)
        features = features.flatten(1)
        return self.fc(features)
```

## Estrategias de Fusion

La fusion multimodal es el proceso de combinar representaciones de diferentes modalidades. Existen tres estrategias principales: Early Fusion, Late Fusion, y Hybrid Fusion.

### Early Fusion (Fusion Temprana)

```
Early Fusion:

    Imagen          Texto           Audio
       │               │               │
       ▼               ▼               ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ Flatten │    │ Flatten │    │ Flatten │
  └────┬────┘    └────┬────┘    └────┬────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Concatenate    │  ← Fusion ANTES
              │  Raw Features   │    de encoding
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Joint Encoder  │
              │  (Transformer)  │
              └────────┬────────┘
                       │
                       ▼
                   Output

Ventajas:
+ Captura interacciones de bajo nivel
+ Un solo modelo aprende todo

Desventajas:
- Requiere datos alineados
- Dimensionalidad muy alta
- Dificil de pre-entrenar por modalidad
```

```python
class EarlyFusion(nn.Module):
    """
    Early Fusion: concatena features crudas antes del encoding.

    Util cuando las modalidades estan bien alineadas temporalmente
    y queremos capturar interacciones de bajo nivel.
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 10
    ):
        super().__init__()

        total_dim = image_dim + text_dim + audio_dim

        # Proyeccion inicial para reducir dimensionalidad
        self.input_projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Encoder conjunto
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.joint_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch, image_dim)
            text_features: (batch, text_dim)
            audio_features: (batch, audio_dim)

        Returns:
            logits: (batch, num_classes)
        """
        # Concatenar todas las features
        fused = torch.cat([image_features, text_features, audio_features], dim=-1)

        # Proyectar
        fused = self.input_projection(fused)

        # Anadir dimension de secuencia para transformer
        fused = fused.unsqueeze(1)  # (batch, 1, hidden*2)

        # Encoding conjunto
        encoded = self.joint_encoder(fused)

        # Pooling y clasificacion
        pooled = encoded.squeeze(1)
        return self.classifier(pooled)


# Ejemplo de uso
def demonstrate_early_fusion():
    """Demuestra Early Fusion con datos sinteticos."""
    batch_size = 4

    model = EarlyFusion(
        image_dim=768,
        text_dim=768,
        audio_dim=768,
        hidden_dim=512,
        num_classes=10
    )

    # Features simuladas (normalmente vienen de encoders pre-entrenados)
    image_feats = torch.randn(batch_size, 768)
    text_feats = torch.randn(batch_size, 768)
    audio_feats = torch.randn(batch_size, 768)

    logits = model(image_feats, text_feats, audio_feats)
    print(f"Output shape: {logits.shape}")  # (4, 10)

    return logits
```

### Late Fusion (Fusion Tardia)

```
Late Fusion:

    Imagen              Texto              Audio
       │                   │                  │
       ▼                   ▼                  ▼
  ┌──────────┐       ┌──────────┐       ┌──────────┐
  │  Image   │       │   Text   │       │  Audio   │
  │  Encoder │       │  Encoder │       │  Encoder │
  └────┬─────┘       └────┬─────┘       └────┬─────┘
       │                   │                  │
       ▼                   ▼                  ▼
  ┌──────────┐       ┌──────────┐       ┌──────────┐
  │  Image   │       │   Text   │       │  Audio   │
  │   Head   │       │   Head   │       │   Head   │
  └────┬─────┘       └────┬─────┘       └────┬─────┘
       │                   │                  │
       ▼                   ▼                  ▼
   pred_img            pred_txt           pred_aud
       │                   │                  │
       └───────────────────┼──────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Combine       │  ← Fusion
                  │   Predictions   │    al final
                  │ (avg/vote/MLP)  │
                  └────────┬────────┘
                           │
                           ▼
                      Final Pred

Ventajas:
+ Encoders independientes (facil pre-entrenar)
+ Robusto a modalidades faltantes
+ Interpretable por modalidad

Desventajas:
- No captura interacciones cross-modal
- Pierde informacion complementaria
```

```python
class LateFusion(nn.Module):
    """
    Late Fusion: cada modalidad tiene su propio pipeline completo,
    la fusion ocurre al nivel de predicciones/decisiones.

    Ideal cuando:
    - Modalidades pueden faltar en algunos ejemplos
    - Queremos interpretabilidad por modalidad
    - Tenemos encoders pre-entrenados buenos
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 10,
        fusion_method: str = "attention"  # "average", "mlp", "attention"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Heads independientes por modalidad
        self.image_head = self._make_head(image_dim, hidden_dim, num_classes)
        self.text_head = self._make_head(text_dim, hidden_dim, num_classes)
        self.audio_head = self._make_head(audio_dim, hidden_dim, num_classes)

        # Fusion layer (si no es average)
        if fusion_method == "mlp":
            self.fusion = nn.Sequential(
                nn.Linear(num_classes * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        elif fusion_method == "attention":
            self.modality_attention = nn.Sequential(
                nn.Linear(num_classes * 3, 3),
                nn.Softmax(dim=-1)
            )

    def _make_head(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> nn.Module:
        """Crea un head de clasificacion."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con soporte para modalidades faltantes.

        Returns:
            Dict con 'logits' fusionados y predicciones individuales
        """
        predictions = []
        outputs = {}

        # Prediccion por modalidad (si esta disponible)
        if image_features is not None:
            img_logits = self.image_head(image_features)
            predictions.append(img_logits)
            outputs['image_logits'] = img_logits

        if text_features is not None:
            txt_logits = self.text_head(text_features)
            predictions.append(txt_logits)
            outputs['text_logits'] = txt_logits

        if audio_features is not None:
            aud_logits = self.audio_head(audio_features)
            predictions.append(aud_logits)
            outputs['audio_logits'] = aud_logits

        # Fusion
        if len(predictions) == 0:
            raise ValueError("Al menos una modalidad debe estar presente")

        if len(predictions) == 1:
            outputs['logits'] = predictions[0]
        else:
            stacked = torch.stack(predictions, dim=1)  # (batch, num_modalities, classes)

            if self.fusion_method == "average":
                outputs['logits'] = stacked.mean(dim=1)

            elif self.fusion_method == "mlp":
                # Asumiendo todas las modalidades presentes
                concat = torch.cat(predictions, dim=-1)
                outputs['logits'] = self.fusion(concat)

            elif self.fusion_method == "attention":
                # Attencion sobre modalidades
                concat = torch.cat(predictions, dim=-1)
                weights = self.modality_attention(concat)  # (batch, 3)
                outputs['modality_weights'] = weights

                # Weighted sum
                weighted = stacked * weights.unsqueeze(-1)
                outputs['logits'] = weighted.sum(dim=1)

        return outputs


def demonstrate_late_fusion():
    """Demuestra Late Fusion con modalidad faltante."""
    batch_size = 4

    model = LateFusion(
        image_dim=768,
        text_dim=768,
        audio_dim=768,
        hidden_dim=256,
        num_classes=10,
        fusion_method="attention"
    )

    # Escenario 1: Todas las modalidades
    image_feats = torch.randn(batch_size, 768)
    text_feats = torch.randn(batch_size, 768)
    audio_feats = torch.randn(batch_size, 768)

    outputs = model(image_feats, text_feats, audio_feats)
    print(f"Con todas las modalidades:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Modality weights: {outputs['modality_weights'][0]}")

    # Escenario 2: Sin audio (modalidad faltante)
    outputs_no_audio = model(image_feats, text_feats, None)
    print(f"\nSin audio:")
    print(f"  Logits shape: {outputs_no_audio['logits'].shape}")
```

### Hybrid Fusion (Fusion Hibrida)

```
Hybrid Fusion (Cross-Modal Attention):

    Imagen                  Texto                  Audio
       │                       │                      │
       ▼                       ▼                      ▼
  ┌──────────┐           ┌──────────┐           ┌──────────┐
  │  Image   │           │   Text   │           │  Audio   │
  │  Encoder │           │  Encoder │           │  Encoder │
  └────┬─────┘           └────┬─────┘           └────┬─────┘
       │                      │                      │
       ▼                      ▼                      ▼
  ┌─────────────────────────────────────────────────────────┐
  │                Cross-Modal Attention Layers              │
  │  ┌─────────────────────────────────────────────────────┐│
  │  │ Img attends to Txt  │  Txt attends to Img          ││
  │  │ Img attends to Aud  │  Txt attends to Aud          ││
  │  │ Aud attends to Img  │  Aud attends to Txt          ││
  │  └─────────────────────────────────────────────────────┘│
  └────┬────────────────────────┬───────────────────┬───────┘
       │                        │                   │
       ▼                        ▼                   ▼
  Enhanced Img           Enhanced Txt          Enhanced Aud
       │                        │                   │
       └────────────────────────┼───────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Final Fusion  │
                       │   (Concat/Pool) │
                       └────────┬────────┘
                                │
                                ▼
                            Output

Ventajas:
+ Lo mejor de ambos mundos
+ Captura interacciones cross-modal
+ Mantiene representaciones por modalidad

Desventajas:
- Mas complejo de entrenar
- Mayor costo computacional
```

```python
class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention: permite que una modalidad
    atienda a otra para enriquecer su representacion.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, dim) - modalidad que atiende
            key_value: (batch, seq_kv, dim) - modalidad atendida
            key_padding_mask: (batch, seq_kv) mascara de padding

        Returns:
            (batch, seq_q, dim) - query enriquecida
        """
        # Cross-attention
        attended, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + attended)

        # FFN
        query = self.norm2(query + self.ffn(query))

        return query


class HybridFusion(nn.Module):
    """
    Hybrid Fusion con Cross-Modal Attention bidireccional.

    Cada modalidad puede atender a las otras, capturando
    relaciones semanticas entre dominios.
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        audio_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        # Proyeccion a dimension comun
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Capas de cross-modal attention
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                # Image attends to Text and Audio
                'img_to_txt': CrossModalAttention(hidden_dim, num_heads, dropout),
                'img_to_aud': CrossModalAttention(hidden_dim, num_heads, dropout),
                # Text attends to Image and Audio
                'txt_to_img': CrossModalAttention(hidden_dim, num_heads, dropout),
                'txt_to_aud': CrossModalAttention(hidden_dim, num_heads, dropout),
                # Audio attends to Image and Text
                'aud_to_img': CrossModalAttention(hidden_dim, num_heads, dropout),
                'aud_to_txt': CrossModalAttention(hidden_dim, num_heads, dropout),
            })
            for _ in range(num_layers)
        ])

        # Self-attention para combinar
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_features: (batch, seq_img, image_dim) o (batch, image_dim)
            text_features: (batch, seq_txt, text_dim) o (batch, text_dim)
            audio_features: (batch, seq_aud, audio_dim) o (batch, audio_dim)

        Returns:
            Dict con logits y representaciones intermedias
        """
        # Asegurar 3D tensors
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)

        # Proyectar a espacio comun
        img = self.image_proj(image_features)
        txt = self.text_proj(text_features)
        aud = self.audio_proj(audio_features)

        # Cross-modal attention layers
        for layer in self.cross_attention_layers:
            # Image enrichment
            img_from_txt = layer['img_to_txt'](img, txt)
            img_from_aud = layer['img_to_aud'](img, aud)
            img = img + img_from_txt + img_from_aud

            # Text enrichment
            txt_from_img = layer['txt_to_img'](txt, img)
            txt_from_aud = layer['txt_to_aud'](txt, aud)
            txt = txt + txt_from_img + txt_from_aud

            # Audio enrichment
            aud_from_img = layer['aud_to_img'](aud, img)
            aud_from_txt = layer['aud_to_txt'](aud, txt)
            aud = aud + aud_from_img + aud_from_txt

        # Pooling por modalidad
        img_pooled = img.mean(dim=1)
        txt_pooled = txt.mean(dim=1)
        aud_pooled = aud.mean(dim=1)

        # Concatenar y clasificar
        fused = torch.cat([img_pooled, txt_pooled, aud_pooled], dim=-1)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'image_representation': img_pooled,
            'text_representation': txt_pooled,
            'audio_representation': aud_pooled
        }
```

## Contrastive Learning para Multimodal

El aprendizaje contrastivo es fundamental en multimodal AI. La idea: acercar representaciones de pares relacionados (imagen y su descripcion) y alejar representaciones de pares no relacionados.

```
Contrastive Learning (InfoNCE/CLIP-style):

Batch de N pares (imagen, texto):

         I1    I2    I3    I4          T1    T2    T3    T4
          │     │     │     │           │     │     │     │
          ▼     ▼     ▼     ▼           ▼     ▼     ▼     ▼
     ┌────────────────────────┐    ┌────────────────────────┐
     │     Image Encoder      │    │     Text Encoder       │
     └────────────────────────┘    └────────────────────────┘
          │     │     │     │           │     │     │     │
          ▼     ▼     ▼     ▼           ▼     ▼     ▼     ▼
         e_i1  e_i2  e_i3  e_i4       e_t1  e_t2  e_t3  e_t4

Similarity Matrix:
                    e_t1   e_t2   e_t3   e_t4
              ┌──────┬──────┬──────┬──────┐
         e_i1 │ 0.9✓ │ 0.1  │ 0.2  │ 0.1  │  ← maximizar diagonal
              ├──────┼──────┼──────┼──────┤
         e_i2 │ 0.2  │ 0.8✓ │ 0.1  │ 0.3  │    (pares positivos)
              ├──────┼──────┼──────┼──────┤
         e_i3 │ 0.1  │ 0.2  │ 0.9✓ │ 0.1  │  ← minimizar off-diagonal
              ├──────┼──────┼──────┼──────┤    (pares negativos)
         e_i4 │ 0.3  │ 0.1  │ 0.2  │ 0.7✓ │
              └──────┴──────┴──────┴──────┘

Loss = -log(exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ))
```

```python
class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss para aprendizaje contrastivo multimodal.

    Usado en CLIP, ALIGN, y otros modelos vision-language.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learn_temperature: bool = True
    ):
        super().__init__()

        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer('temperature', torch.tensor([temperature]))

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula contrastive loss bidireccional.

        Args:
            image_embeddings: (batch, dim) embeddings normalizados
            text_embeddings: (batch, dim) embeddings normalizados

        Returns:
            Dict con loss total y metricas
        """
        # Normalizar embeddings (si no estan normalizados)
        image_embeddings = nn.functional.normalize(image_embeddings, dim=-1)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=-1)

        # Calcular matriz de similitud
        # (batch, dim) @ (dim, batch) -> (batch, batch)
        logits = image_embeddings @ text_embeddings.T / self.temperature

        # Labels: diagonal (i-esimo imagen con i-esimo texto)
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Loss bidireccional
        # Image-to-Text: para cada imagen, encuentra su texto
        loss_i2t = nn.functional.cross_entropy(logits, labels)

        # Text-to-Image: para cada texto, encuentra su imagen
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)

        # Loss total
        total_loss = (loss_i2t + loss_t2i) / 2

        # Metricas
        with torch.no_grad():
            # Accuracy image-to-text
            i2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            # Accuracy text-to-image
            t2i_acc = (logits.T.argmax(dim=1) == labels).float().mean()

        return {
            'loss': total_loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i,
            'accuracy_i2t': i2t_acc,
            'accuracy_t2i': t2i_acc,
            'temperature': self.temperature.item()
        }


class ContrastiveMultimodalModel(nn.Module):
    """
    Modelo multimodal entrenado con contrastive learning.
    Arquitectura similar a CLIP.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Proyecciones a espacio compartido
        self.image_projection = nn.Linear(768, embed_dim)  # Asumiendo 768-dim encoders
        self.text_projection = nn.Linear(768, embed_dim)

        self.contrastive_loss = ContrastiveLoss(temperature)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to shared embedding space."""
        features = self.image_encoder(image)
        projected = self.image_projection(features)
        return nn.functional.normalize(projected, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to shared embedding space."""
        features = self.text_encoder(text)
        projected = self.text_projection(features)
        return nn.functional.normalize(projected, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        texts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass para entrenamiento.

        Args:
            images: batch de imagenes
            texts: batch de textos (alineados con imagenes)

        Returns:
            Dict con embeddings y loss
        """
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(texts)

        loss_dict = self.contrastive_loss(image_embeddings, text_embeddings)

        return {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            **loss_dict
        }

    def compute_similarity(
        self,
        images: torch.Tensor,
        texts: torch.Tensor
    ) -> torch.Tensor:
        """
        Computa matriz de similitud imagen-texto.

        Util para retrieval y zero-shot classification.
        """
        with torch.no_grad():
            image_embeddings = self.encode_image(images)
            text_embeddings = self.encode_text(texts)

            return image_embeddings @ text_embeddings.T
```

## Alineamiento Cross-Modal

```
Cross-Modal Alignment:

El reto principal de multimodal AI es alinear representaciones
de diferentes modalidades en un espacio semantico comun.

Sin Alineamiento:
┌────────────────────────────────────────────────────────────┐
│                    Embedding Space                          │
│                                                            │
│     🖼️ imagen1        ● texto1                             │
│           🖼️ imagen2                                        │
│                            ● texto2                         │
│   🖼️ imagen3                    ● texto3                    │
│                                                            │
│  Representaciones de mismos conceptos estan dispersas      │
└────────────────────────────────────────────────────────────┘

Con Alineamiento (Contrastive):
┌────────────────────────────────────────────────────────────┐
│                    Shared Embedding Space                   │
│                                                            │
│     🖼️● (imagen1, texto1) - "perro"                        │
│                                                            │
│              🖼️● (imagen2, texto2) - "gato"                │
│                                                            │
│   🖼️● (imagen3, texto3) - "coche"                          │
│                                                            │
│  Conceptos similares agrupados, diferentes separados       │
└────────────────────────────────────────────────────────────┘
```

## Aplicaciones Practicas en Seguridad

### Deteccion de Phishing Multimodal

```python
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn


@dataclass
class PhishingDetectionResult:
    """Resultado de deteccion de phishing multimodal."""
    is_phishing: bool
    confidence: float
    visual_score: float
    text_score: float
    url_score: float
    explanation: str


class MultimodalPhishingDetector(nn.Module):
    """
    Detector de phishing que combina:
    - Analisis visual de screenshots de paginas
    - Analisis de texto del contenido
    - Analisis de patrones URL

    Los phishers suelen copiar visualmente pero tienen
    inconsistencias en texto y URLs.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        url_encoder: nn.Module,
        hidden_dim: int = 512
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.url_encoder = url_encoder

        # Proyecciones
        self.image_proj = nn.Linear(768, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)
        self.url_proj = nn.Linear(256, hidden_dim)

        # Cross-modal attention para detectar inconsistencias
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads=8)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # [legit, phishing]
        )

        # Heads para scores individuales (interpretabilidad)
        self.visual_head = nn.Linear(hidden_dim, 1)
        self.text_head = nn.Linear(hidden_dim, 1)
        self.url_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        screenshot: torch.Tensor,
        page_text: torch.Tensor,
        url_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            screenshot: (batch, 3, H, W) screenshot de la pagina
            page_text: (batch, seq_len) tokens del texto
            url_features: (batch, url_feat_dim) features de URL

        Returns:
            Dict con logits y scores por modalidad
        """
        # Encode cada modalidad
        img_feats = self.image_proj(self.image_encoder(screenshot))
        txt_feats = self.text_proj(self.text_encoder(page_text))
        url_feats = self.url_proj(self.url_encoder(url_features))

        # Asegurar dimensiones para attention
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(1)
        if url_feats.dim() == 2:
            url_feats = url_feats.unsqueeze(1)

        # Cross-modal attention: detectar inconsistencias
        # Imagen atiende a texto (detecta si visual no coincide con contenido)
        img_attended = self.cross_attention(img_feats, txt_feats)

        # Pool
        img_pooled = img_attended.mean(dim=1)
        txt_pooled = txt_feats.mean(dim=1)
        url_pooled = url_feats.mean(dim=1)

        # Scores individuales
        visual_score = torch.sigmoid(self.visual_head(img_pooled))
        text_score = torch.sigmoid(self.text_head(txt_pooled))
        url_score = torch.sigmoid(self.url_head(url_pooled))

        # Clasificacion final
        fused = torch.cat([img_pooled, txt_pooled, url_pooled], dim=-1)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'visual_score': visual_score,
            'text_score': text_score,
            'url_score': url_score
        }

    def detect(
        self,
        screenshot: torch.Tensor,
        page_text: torch.Tensor,
        url_features: torch.Tensor
    ) -> PhishingDetectionResult:
        """Interfaz de alto nivel para deteccion."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(screenshot, page_text, url_features)

            probs = torch.softmax(outputs['logits'], dim=-1)
            is_phishing = probs[0, 1] > 0.5
            confidence = probs[0, 1].item() if is_phishing else probs[0, 0].item()

            # Generar explicacion
            visual = outputs['visual_score'].item()
            text = outputs['text_score'].item()
            url = outputs['url_score'].item()

            explanations = []
            if visual > 0.6:
                explanations.append("Visual similar a sitio legitimo")
            if text > 0.6:
                explanations.append("Texto sospechoso detectado")
            if url > 0.6:
                explanations.append("URL con patrones de phishing")

            return PhishingDetectionResult(
                is_phishing=is_phishing.item(),
                confidence=confidence,
                visual_score=visual,
                text_score=text,
                url_score=url,
                explanation="; ".join(explanations) if explanations else "No hay senales claras"
            )
```

### Deteccion de Deepfakes Multimodal

```python
class MultimodalDeepfakeDetector(nn.Module):
    """
    Detector de deepfakes que analiza:
    - Inconsistencias visuales (artefactos faciales)
    - Sincronizacion audio-visual (lip-sync)
    - Patrones temporales anomalos

    Los deepfakes suelen fallar en la coherencia multimodal.
    """

    def __init__(
        self,
        video_encoder: nn.Module,
        audio_encoder: nn.Module,
        hidden_dim: int = 512,
        num_frames: int = 16
    ):
        super().__init__()

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.num_frames = num_frames

        # Proyecciones
        self.video_proj = nn.Linear(768, hidden_dim)
        self.audio_proj = nn.Linear(768, hidden_dim)

        # Temporal modeling
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=4
        )

        # Audio-Visual sync module
        self.av_sync = AudioVisualSyncModule(hidden_dim)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # [real, fake]
        )

    def forward(
        self,
        video_frames: torch.Tensor,
        audio_mel: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video_frames: (batch, num_frames, 3, H, W)
            audio_mel: (batch, mel_bins, time)

        Returns:
            Dict con prediccion y scores de sincronizacion
        """
        batch_size = video_frames.shape[0]

        # Encode video frames
        frames_flat = video_frames.view(-1, *video_frames.shape[2:])
        video_feats = self.video_encoder(frames_flat)
        video_feats = video_feats.view(batch_size, self.num_frames, -1)
        video_feats = self.video_proj(video_feats)

        # Encode audio
        audio_feats = self.audio_encoder(audio_mel)
        audio_feats = self.audio_proj(audio_feats)

        # Temporal analysis
        temporal_feats = self.temporal_transformer(video_feats)
        temporal_pooled = temporal_feats.mean(dim=1)

        # Audio-Visual sync analysis
        sync_score, av_feats = self.av_sync(video_feats, audio_feats)

        # Visual artifacts (usar ultimo frame encoding)
        visual_feats = video_feats.mean(dim=1)

        # Fusion y clasificacion
        fused = torch.cat([temporal_pooled, av_feats, visual_feats], dim=-1)
        logits = self.classifier(fused)

        return {
            'logits': logits,
            'sync_score': sync_score,
            'temporal_features': temporal_pooled,
            'av_features': av_feats
        }


class AudioVisualSyncModule(nn.Module):
    """
    Modulo para detectar desincronizacion audio-visual.
    Los deepfakes suelen tener lip-sync imperfecto.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        self.cross_attention = CrossModalAttention(hidden_dim, num_heads=8)

        self.sync_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_feats: (batch, num_frames, dim)
            audio_feats: (batch, dim) o (batch, time, dim)

        Returns:
            sync_score: (batch, 1) - 1.0 = sincronizado, 0.0 = desincronizado
            fused_feats: (batch, dim) - features fusionadas
        """
        if audio_feats.dim() == 2:
            audio_feats = audio_feats.unsqueeze(1)

        # Video atiende a audio
        v_attended = self.cross_attention(video_feats, audio_feats)

        # Audio atiende a video
        a_attended = self.cross_attention(audio_feats, video_feats)

        # Pool
        v_pooled = v_attended.mean(dim=1)
        a_pooled = a_attended.mean(dim=1)

        # Sync score
        concat = torch.cat([v_pooled, a_pooled], dim=-1)
        sync_score = self.sync_head(concat)

        return sync_score, v_pooled + a_pooled
```

## Metricas y Evaluacion Multimodal

```python
from typing import Dict, List
import torch
import numpy as np
from sklearn.metrics import ndcg_score


def compute_retrieval_metrics(
    similarities: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Computa metricas de retrieval para evaluacion multimodal.

    Args:
        similarities: (N, N) matriz de similitud
        k_values: valores de k para Recall@k

    Returns:
        Dict con Recall@k y otras metricas
    """
    n = similarities.shape[0]

    # Ground truth: diagonal (i-th query matches i-th item)
    gt_indices = torch.arange(n)

    # Rankings
    rankings = torch.argsort(similarities, dim=1, descending=True)

    metrics = {}

    for k in k_values:
        # Recall@k: proporcion de queries donde el item correcto esta en top-k
        top_k = rankings[:, :k]
        hits = (top_k == gt_indices.unsqueeze(1)).any(dim=1)
        metrics[f'recall@{k}'] = hits.float().mean().item()

    # Mean Rank
    correct_positions = (rankings == gt_indices.unsqueeze(1)).float().argmax(dim=1)
    metrics['mean_rank'] = (correct_positions.float().mean() + 1).item()

    # Median Rank
    metrics['median_rank'] = (correct_positions.float().median() + 1).item()

    return metrics


def evaluate_multimodal_model(
    model: nn.Module,
    image_loader,
    text_loader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluacion completa de modelo multimodal.

    Returns:
        Dict con metricas image-to-text y text-to-image
    """
    model.eval()

    all_image_embeddings = []
    all_text_embeddings = []

    with torch.no_grad():
        # Encode todas las imagenes
        for images, _ in image_loader:
            images = images.to(device)
            img_emb = model.encode_image(images)
            all_image_embeddings.append(img_emb.cpu())

        # Encode todos los textos
        for texts, _ in text_loader:
            texts = texts.to(device)
            txt_emb = model.encode_text(texts)
            all_text_embeddings.append(txt_emb.cpu())

    # Concatenar
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)

    # Similitudes
    similarities = image_embeddings @ text_embeddings.T

    # Metricas Image-to-Text
    i2t_metrics = compute_retrieval_metrics(similarities)
    i2t_metrics = {f'i2t_{k}': v for k, v in i2t_metrics.items()}

    # Metricas Text-to-Image
    t2i_metrics = compute_retrieval_metrics(similarities.T)
    t2i_metrics = {f't2i_{k}': v for k, v in t2i_metrics.items()}

    return {**i2t_metrics, **t2i_metrics}
```

## Consideraciones de Entrenamiento

```
Retos del Entrenamiento Multimodal:

1. DATA IMBALANCE
   ┌────────────────────────────────────────────────────────┐
   │ Modalidad A: 1M ejemplos                               │
   │ Modalidad B: 100K ejemplos                             │
   │ Modalidad C: 10K ejemplos                              │
   │                                                        │
   │ Solucion: Sampling strategies, loss weighting          │
   └────────────────────────────────────────────────────────┘

2. MISSING MODALITIES
   ┌────────────────────────────────────────────────────────┐
   │ En inferencia, puede faltar una modalidad              │
   │                                                        │
   │ Solucion: Dropout de modalidades durante entrenamiento │
   │           Late fusion permite manejo natural           │
   └────────────────────────────────────────────────────────┘

3. ALIGNMENT QUALITY
   ┌────────────────────────────────────────────────────────┐
   │ Los datos multimodales pueden estar mal alineados      │
   │ Ej: imagen-caption con caption incorrecto              │
   │                                                        │
   │ Solucion: Filtrado de datos, self-cleaning             │
   └────────────────────────────────────────────────────────┘

4. COMPUTATIONAL COST
   ┌────────────────────────────────────────────────────────┐
   │ Multiples encoders = muchos parametros                 │
   │ Cross-attention = O(n^2) por par de modalidades        │
   │                                                        │
   │ Solucion: Gradient checkpointing, mixed precision      │
   │           Frozen encoders cuando sea posible           │
   └────────────────────────────────────────────────────────┘
```

```python
class MultimodalTrainer:
    """
    Trainer con estrategias para retos multimodales.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_dropout: float = 0.2,
        loss_weights: Dict[str, float] = None
    ):
        self.model = model
        self.modality_dropout = modality_dropout
        self.loss_weights = loss_weights or {}

    def apply_modality_dropout(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Aplica dropout de modalidades durante entrenamiento.
        Ayuda al modelo a ser robusto a modalidades faltantes.
        """
        if not training:
            return batch

        new_batch = {}
        for key, value in batch.items():
            if torch.rand(1).item() < self.modality_dropout:
                # Dropout: reemplazar con zeros
                new_batch[key] = torch.zeros_like(value)
            else:
                new_batch[key] = value

        # Asegurar al menos una modalidad
        if all(v.sum() == 0 for v in new_batch.values()):
            # Restaurar una modalidad aleatoria
            key = list(batch.keys())[torch.randint(len(batch), (1,)).item()]
            new_batch[key] = batch[key]

        return new_batch

    def compute_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combina losses con pesos configurados."""
        total_loss = 0
        for name, loss in losses.items():
            weight = self.loss_weights.get(name, 1.0)
            total_loss = total_loss + weight * loss
        return total_loss
```

## Resumen

```
FUNDAMENTOS MULTIMODAL AI:

┌─────────────────────────────────────────────────────────────────┐
│                     KEY TAKEAWAYS                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FUSION STRATEGIES                                           │
│     ├── Early: captura interacciones bajo nivel                │
│     ├── Late: robusto, interpretable                           │
│     └── Hybrid: lo mejor de ambos (cross-attention)            │
│                                                                 │
│  2. CONTRASTIVE LEARNING                                        │
│     ├── Alinea modalidades en espacio comun                    │
│     ├── InfoNCE loss (CLIP-style)                              │
│     └── Habilita zero-shot transfer                            │
│                                                                 │
│  3. CROSS-MODAL ATTENTION                                       │
│     ├── Una modalidad atiende a otra                           │
│     ├── Captura relaciones semanticas                          │
│     └── Detecta inconsistencias (seguridad)                    │
│                                                                 │
│  4. APLICACIONES SEGURIDAD                                      │
│     ├── Phishing: inconsistencia visual-texto-URL              │
│     ├── Deepfakes: desincronizacion audio-visual               │
│     └── Fraud: patrones multimodales anomalos                  │
│                                                                 │
│  5. RETOS                                                       │
│     ├── Data imbalance entre modalidades                       │
│     ├── Modalidades faltantes en inferencia                    │
│     ├── Calidad de alineamiento                                │
│     └── Costo computacional                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Referencias

1. "Multimodal Machine Learning: A Survey and Taxonomy" - Baltrušaitis et al., 2019
2. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP) - Radford et al., 2021
3. "Attention Bottlenecks for Multimodal Fusion" - Nagrani et al., 2021
4. "Multimodal Learning with Transformers: A Survey" - Xu et al., 2023
5. "FLAVA: A Foundational Language And Vision Alignment Model" - Singh et al., 2022
