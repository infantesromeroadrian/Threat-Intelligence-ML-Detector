# CLIP y SigLIP: Vision-Language Pre-training

## Introduccion a CLIP

CLIP (Contrastive Language-Image Pre-training) de OpenAI revoluciono el campo de vision-language al demostrar que un modelo entrenado con millones de pares imagen-texto de internet puede hacer zero-shot transfer a casi cualquier tarea de clasificacion visual.

```
CLIP: La Gran Idea

Entrenamiento Tradicional:          CLIP:
┌───────────────────────────┐       ┌───────────────────────────────────────┐
│ Dataset: ImageNet (1K)    │       │ Dataset: 400M pares imagen-texto     │
│ Labels: 1000 categorias   │       │ de internet                          │
│                           │       │                                       │
│ Modelo aprende:           │       │ Modelo aprende:                       │
│ "Esta imagen es clase 42" │       │ "Esta imagen va con este texto"      │
│                           │       │                                       │
│ Limitacion:               │       │ Poder:                                │
│ Solo clasifica en 1K      │       │ Puede clasificar en CUALQUIER        │
│ categorias vistas         │       │ categoria descrita en texto          │
└───────────────────────────┘       └───────────────────────────────────────┘

Zero-Shot Classification:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Imagen: [foto de un Shiba Inu]                                            │
│                                                                             │
│  Textos (prompts):                                                          │
│  - "a photo of a golden retriever"  → similitud: 0.15                      │
│  - "a photo of a siamese cat"       → similitud: 0.08                      │
│  - "a photo of a shiba inu"         → similitud: 0.92  ← Winner!           │
│  - "a photo of a german shepherd"   → similitud: 0.12                      │
│                                                                             │
│  Sin entrenamiento adicional, clasifica correctamente!                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Arquitectura CLIP

```
Arquitectura CLIP:

                     CLIP
    ┌────────────────────────────────────────┐
    │                                        │
    │   Image Encoder        Text Encoder    │
    │   (ViT-B/32)          (Transformer)    │
    │        │                    │          │
    │        ▼                    ▼          │
    │   ┌─────────┐         ┌─────────┐      │
    │   │ [CLS]   │         │ [EOS]   │      │
    │   │ token   │         │ token   │      │
    │   └────┬────┘         └────┬────┘      │
    │        │                    │          │
    │        ▼                    ▼          │
    │   ┌─────────┐         ┌─────────┐      │
    │   │ Linear  │         │ Linear  │      │
    │   │ Proj    │         │ Proj    │      │
    │   └────┬────┘         └────┬────┘      │
    │        │                    │          │
    │        ▼                    ▼          │
    │      I_emb               T_emb         │
    │      (512d)              (512d)        │
    │        │                    │          │
    │        └────────┬───────────┘          │
    │                 │                      │
    │                 ▼                      │
    │         Cosine Similarity              │
    │              Matrix                    │
    │                 │                      │
    │                 ▼                      │
    │          Contrastive Loss              │
    │         (InfoNCE / NCE)                │
    │                                        │
    └────────────────────────────────────────┘

Variantes de Image Encoder:
┌─────────────────────────────────────────────────────────────┐
│ Modelo          │ Params │ Image Size │ Embedding Dim      │
├─────────────────┼────────┼────────────┼────────────────────┤
│ ViT-B/32        │ 88M    │ 224×224    │ 512                │
│ ViT-B/16        │ 88M    │ 224×224    │ 512                │
│ ViT-L/14        │ 304M   │ 224×224    │ 768                │
│ ViT-L/14@336px  │ 304M   │ 336×336    │ 768                │
│ ResNet-50       │ 38M    │ 224×224    │ 1024               │
│ ResNet-101      │ 57M    │ 224×224    │ 512                │
└─────────────────┴────────┴────────────┴────────────────────┘
```

### Implementacion del Image Encoder (ViT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CLIPVisionConfig:
    """Configuracion para el Vision Encoder de CLIP."""
    image_size: int = 224
    patch_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.0
    attention_dropout: float = 0.0
    projection_dim: int = 512


class CLIPVisionEmbeddings(nn.Module):
    """
    Embeddings de imagen para CLIP Vision Encoder.
    Convierte imagen en secuencia de patch embeddings.
    """

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config

        # Patch embedding via Conv2d
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )

        # Numero de patches
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Class embedding ([CLS] token)
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))

        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(
            self.num_patches + 1,  # +1 para [CLS]
            config.hidden_size
        )

        # Register buffer para position ids
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches + 1).unsqueeze(0),
            persistent=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, H, W)

        Returns:
            (batch, num_patches+1, hidden_size)
        """
        batch_size = pixel_values.shape[0]

        # Patch embeddings: (batch, hidden, H/patch, W/patch)
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten: (batch, hidden, num_patches) -> (batch, num_patches, hidden)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Prepend [CLS] token
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # Add positional embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-head attention para CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq, hidden)
            attention_mask: (batch, 1, seq, seq) opcional

        Returns:
            (batch, seq, hidden)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape para multi-head: (batch, num_heads, seq, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back: (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(nn.Module):
    """MLP para CLIP Transformer blocks."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Una capa del CLIP Encoder."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention con residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP con residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPVisionEncoder(nn.Module):
    """Vision Encoder completo de CLIP."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)

        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.post_layernorm = nn.LayerNorm(config.hidden_size)

        # Projection to shared embedding space
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, H, W)

        Returns:
            (batch, projection_dim) normalized embeddings
        """
        # Get embeddings
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Get [CLS] token output
        hidden_states = hidden_states[:, 0, :]  # (batch, hidden_size)
        hidden_states = self.post_layernorm(hidden_states)

        # Project to shared space
        image_embeds = self.projection(hidden_states)

        # L2 normalize
        image_embeds = F.normalize(image_embeds, dim=-1)

        return image_embeds
```

### Implementacion del Text Encoder

```python
@dataclass
class CLIPTextConfig:
    """Configuracion para el Text Encoder de CLIP."""
    vocab_size: int = 49408
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 77
    dropout: float = 0.0
    attention_dropout: float = 0.0
    projection_dim: int = 512


class CLIPTextEmbeddings(nn.Module):
    """Embeddings de texto para CLIP."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token ids

        Returns:
            (batch, seq_len, hidden_size)
        """
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


class CLIPTextEncoder(nn.Module):
    """Text Encoder de CLIP con causal attention."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config

        self.embeddings = CLIPTextEmbeddings(config)

        self.layers = nn.ModuleList([
            CLIPEncoderLayer(
                CLIPVisionConfig(
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout
                )
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        # Projection
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Construye mascara causal para text encoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len) padding mask

        Returns:
            (batch, projection_dim) normalized embeddings
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Causal mask
        causal_mask = self.build_causal_mask(seq_len, input_ids.device)

        # Combine with attention mask si se proporciona
        if attention_mask is not None:
            # Expand attention mask
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = (1.0 - extended_mask) * float('-inf')
            causal_mask = causal_mask + extended_mask

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)

        hidden_states = self.final_layer_norm(hidden_states)

        # Pool: usar el token [EOS] (ultimo token valido)
        # En CLIP, esto es el token con mayor id en la secuencia
        pooled_output = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            input_ids.argmax(dim=-1)
        ]

        # Project
        text_embeds = self.projection(pooled_output)

        # L2 normalize
        text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds
```

### Modelo CLIP Completo

```python
class CLIPModel(nn.Module):
    """
    Modelo CLIP completo con vision y text encoders.
    """

    def __init__(
        self,
        vision_config: Optional[CLIPVisionConfig] = None,
        text_config: Optional[CLIPTextConfig] = None,
        logit_scale_init: float = 2.6592  # ln(1/0.07)
    ):
        super().__init__()

        vision_config = vision_config or CLIPVisionConfig()
        text_config = text_config or CLIPTextConfig()

        self.vision_encoder = CLIPVisionEncoder(vision_config)
        self.text_encoder = CLIPTextEncoder(text_config)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.tensor([logit_scale_init]))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.vision_encoder(pixel_values)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass para entrenamiento.

        Returns:
            Dict con embeddings, logits y loss
        """
        # Encode both modalities
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # Compute similarity logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        # Contrastive loss
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'loss': loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i
        }
```

## Uso Practico con OpenCLIP

OpenCLIP es la implementacion open-source de CLIP con modelos pre-entrenados adicionales.

```python
import torch
from PIL import Image
from typing import List, Tuple, Dict
import open_clip


class CLIPWrapper:
    """
    Wrapper para uso practico de CLIP/OpenCLIP.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Arquitectura del modelo (ViT-B-32, ViT-L-14, etc.)
            pretrained: Dataset de pre-training (openai, laion2b, etc.)
            device: Dispositivo de ejecucion
        """
        self.device = device

        # Cargar modelo y tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    @torch.no_grad()
    def encode_images(
        self,
        images: List[Image.Image]
    ) -> torch.Tensor:
        """
        Encode una lista de imagenes a embeddings.

        Args:
            images: Lista de imagenes PIL

        Returns:
            (N, embed_dim) tensor de embeddings normalizados
        """
        # Preprocesar imagenes
        image_tensors = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)

        # Encode
        image_features = self.model.encode_image(image_tensors)

        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Encode una lista de textos a embeddings.

        Args:
            texts: Lista de strings

        Returns:
            (N, embed_dim) tensor de embeddings normalizados
        """
        # Tokenizar
        text_tokens = self.tokenizer(texts).to(self.device)

        # Encode
        text_features = self.model.encode_text(text_tokens)

        # Normalizar
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    @torch.no_grad()
    def zero_shot_classify(
        self,
        images: List[Image.Image],
        class_names: List[str],
        prompt_template: str = "a photo of a {}"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-shot classification usando CLIP.

        Args:
            images: Lista de imagenes
            class_names: Lista de nombres de clases
            prompt_template: Template para generar prompts

        Returns:
            (probs, predicted_classes) tupla de probabilidades y predicciones
        """
        # Generar prompts para cada clase
        prompts = [prompt_template.format(name) for name in class_names]

        # Encode
        image_features = self.encode_images(images)
        text_features = self.encode_texts(prompts)

        # Similitud
        similarity = image_features @ text_features.T  # (num_images, num_classes)

        # Softmax para probabilidades
        probs = similarity.softmax(dim=-1)

        # Predicciones
        predicted_classes = probs.argmax(dim=-1)

        return probs, predicted_classes

    @torch.no_grad()
    def image_to_text_retrieval(
        self,
        image: Image.Image,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Retrieval: encuentra los textos mas similares a una imagen.

        Returns:
            Lista de (texto, score) ordenada por similitud
        """
        image_features = self.encode_images([image])
        text_features = self.encode_texts(candidate_texts)

        # Similitud
        similarities = (image_features @ text_features.T).squeeze(0)

        # Top-k
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = [
            (candidate_texts[idx], similarities[idx].item())
            for idx in top_indices
        ]

        return results

    @torch.no_grad()
    def text_to_image_retrieval(
        self,
        text: str,
        candidate_images: List[Image.Image],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Retrieval: encuentra las imagenes mas similares a un texto.

        Returns:
            Lista de (indice_imagen, score) ordenada por similitud
        """
        text_features = self.encode_texts([text])
        image_features = self.encode_images(candidate_images)

        # Similitud
        similarities = (text_features @ image_features.T).squeeze(0)

        # Top-k
        top_indices = similarities.argsort(descending=True)[:top_k]

        results = [
            (idx.item(), similarities[idx].item())
            for idx in top_indices
        ]

        return results


# Ejemplo de uso
def demo_clip():
    """Demostracion de capacidades de CLIP."""

    # Inicializar
    clip = CLIPWrapper(model_name="ViT-B-32", pretrained="openai")

    # Imagen de ejemplo (usar imagen real)
    # image = Image.open("example.jpg")

    # Zero-shot classification
    class_names = ["cat", "dog", "bird", "fish", "horse"]
    # probs, preds = clip.zero_shot_classify([image], class_names)
    # print(f"Predicted class: {class_names[preds[0]]}")
    # print(f"Probabilities: {dict(zip(class_names, probs[0].tolist()))}")

    # Prompt engineering para mejor performance
    improved_prompts = [
        "a photo of a cat, a type of pet",
        "a photo of a dog, a type of pet",
        "a photo of a bird, a type of animal",
        "a photo of a fish, an aquatic animal",
        "a photo of a horse, a large mammal"
    ]

    # Diferentes templates para ensembling
    templates = [
        "a photo of a {}",
        "a picture of a {}",
        "an image of a {}",
        "a {} in the photo",
        "a high-quality photo of a {}"
    ]

    print("CLIP demo initialized successfully")
    return clip
```

## Zero-Shot Classification en Detalle

```
Zero-Shot Classification Pipeline:

1. PREPARACION DE PROMPTS
   ┌────────────────────────────────────────────────────────────────┐
   │ Clases: ["malware", "benign", "phishing"]                     │
   │                                                                │
   │ Template: "a screenshot of {} software"                       │
   │                                                                │
   │ Prompts generados:                                             │
   │ - "a screenshot of malware software"                          │
   │ - "a screenshot of benign software"                           │
   │ - "a screenshot of phishing software"                         │
   └────────────────────────────────────────────────────────────────┘

2. ENCODING
   ┌────────────────────────────────────────────────────────────────┐
   │ Image: screenshot.png                                          │
   │         │                                                      │
   │         ▼                                                      │
   │   ┌──────────────┐                                            │
   │   │ Vision Enc.  │ ──→ image_emb (512d)                       │
   │   └──────────────┘                                            │
   │                                                                │
   │ Prompts:                                                       │
   │   "malware..." ──→ ┌──────────────┐ ──→ text_emb_1 (512d)     │
   │   "benign..."  ──→ │  Text Enc.   │ ──→ text_emb_2 (512d)     │
   │   "phishing.." ──→ └──────────────┘ ──→ text_emb_3 (512d)     │
   └────────────────────────────────────────────────────────────────┘

3. SIMILITUD Y PREDICCION
   ┌────────────────────────────────────────────────────────────────┐
   │                                                                │
   │ Cosine Similarity:                                            │
   │                                                                │
   │   image_emb · text_emb_1 = 0.82  (malware)                    │
   │   image_emb · text_emb_2 = 0.15  (benign)                     │
   │   image_emb · text_emb_3 = 0.31  (phishing)                   │
   │                                                                │
   │ Softmax(logits/temperature):                                  │
   │   P(malware)  = 0.73                                          │
   │   P(benign)   = 0.08                                          │
   │   P(phishing) = 0.19                                          │
   │                                                                │
   │ Prediccion: MALWARE (confidence: 73%)                         │
   └────────────────────────────────────────────────────────────────┘
```

```python
class ZeroShotClassifier:
    """
    Clasificador zero-shot robusto con prompt ensembling.
    """

    def __init__(
        self,
        clip_wrapper: CLIPWrapper,
        class_names: List[str],
        templates: Optional[List[str]] = None
    ):
        self.clip = clip_wrapper
        self.class_names = class_names

        # Templates por defecto
        self.templates = templates or [
            "a photo of a {}",
            "a picture of a {}",
            "an image showing {}",
            "a {} in this image",
            "this is a photo of {}"
        ]

        # Pre-compute text embeddings con prompt ensembling
        self.text_features = self._compute_text_features()

    def _compute_text_features(self) -> torch.Tensor:
        """
        Pre-computa embeddings de texto con ensembling.
        Promedia embeddings de diferentes templates.
        """
        all_features = []

        for class_name in self.class_names:
            # Generar prompts para esta clase
            prompts = [template.format(class_name) for template in self.templates]

            # Encode todos los prompts
            features = self.clip.encode_texts(prompts)

            # Promediar y normalizar
            mean_features = features.mean(dim=0)
            mean_features = mean_features / mean_features.norm()

            all_features.append(mean_features)

        # Stack: (num_classes, embed_dim)
        return torch.stack(all_features)

    @torch.no_grad()
    def classify(
        self,
        images: List[Image.Image],
        return_all_probs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Clasifica imagenes.

        Args:
            images: Lista de imagenes PIL
            return_all_probs: Si devolver probabilidades para todas las clases

        Returns:
            Dict con predicciones y probabilidades
        """
        # Encode imagenes
        image_features = self.clip.encode_images(images)

        # Similitud con text features pre-computadas
        logits = image_features @ self.text_features.T

        # Probabilidades
        probs = logits.softmax(dim=-1)

        # Predicciones
        predictions = probs.argmax(dim=-1)
        predicted_names = [self.class_names[idx] for idx in predictions]
        confidences = probs.max(dim=-1).values

        result = {
            'predictions': predictions,
            'predicted_names': predicted_names,
            'confidences': confidences
        }

        if return_all_probs:
            result['all_probs'] = probs
            result['class_names'] = self.class_names

        return result


# Aplicacion: Clasificacion de screenshots de seguridad
def security_screenshot_classifier():
    """Clasificador de screenshots para analisis de seguridad."""

    clip = CLIPWrapper()

    # Clases de seguridad
    security_classes = [
        "legitimate login page",
        "phishing login page",
        "malware warning dialog",
        "system error message",
        "captcha verification",
        "two-factor authentication",
        "suspicious download prompt",
        "normal website content"
    ]

    # Templates especificos para seguridad
    security_templates = [
        "a screenshot of {}",
        "a computer screen showing {}",
        "an interface displaying {}",
        "{} on a web browser",
        "this is {}"
    ]

    classifier = ZeroShotClassifier(
        clip_wrapper=clip,
        class_names=security_classes,
        templates=security_templates
    )

    return classifier
```

## SigLIP: Mejoras sobre CLIP

SigLIP (Sigmoid Loss for Language-Image Pre-training) de Google introduce mejoras clave sobre CLIP.

```
CLIP vs SigLIP Loss:

CLIP (Softmax-based InfoNCE):
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ Loss = -log( exp(sim(i,t_i)/τ) / Σ_j exp(sim(i,t_j)/τ) )     │
│                                                                │
│ Problema: Normaliza sobre TODO el batch                        │
│          → Muy dependiente del batch size                      │
│          → Necesita batches grandes (32K+) para buen resultado │
│                                                                │
│ Batch pequeño (256):   │  Batch grande (32K):                 │
│ - Pocos negativos      │  - Muchos negativos                  │
│ - Senales debiles      │  - Senales fuertes                   │
│ - Peor rendimiento     │  - Mejor rendimiento                 │
└────────────────────────────────────────────────────────────────┘

SigLIP (Sigmoid-based):
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│ Loss = -Σ_{i,j} [ y_{ij} log(σ(z_{ij}))                       │
│                  + (1-y_{ij}) log(1-σ(z_{ij})) ]              │
│                                                                │
│ donde:                                                         │
│   z_{ij} = sim(image_i, text_j) * scale + bias                │
│   y_{ij} = 1 si i==j (positivo), 0 si i!=j (negativo)         │
│                                                                │
│ Ventajas:                                                      │
│ + Cada par es clasificacion binaria independiente              │
│ + No depende tanto del batch size                              │
│ + Funciona bien con batches pequenos                           │
│ + Mejor calibracion de probabilidades                          │
└────────────────────────────────────────────────────────────────┘
```

```python
class SigLIPLoss(nn.Module):
    """
    Sigmoid Loss para pre-training vision-language.
    Trata cada par imagen-texto como clasificacion binaria.
    """

    def __init__(
        self,
        init_logit_scale: float = 10.0,
        init_logit_bias: float = -10.0
    ):
        super().__init__()

        # Learnable scale y bias
        self.logit_scale = nn.Parameter(torch.tensor([init_logit_scale]))
        self.logit_bias = nn.Parameter(torch.tensor([init_logit_bias]))

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula SigLIP loss.

        Args:
            image_embeds: (batch, dim) embeddings normalizados
            text_embeds: (batch, dim) embeddings normalizados

        Returns:
            Dict con loss y metricas
        """
        batch_size = image_embeds.shape[0]
        device = image_embeds.device

        # Compute raw similarities
        # (batch, batch)
        similarities = image_embeds @ text_embeds.T

        # Apply learnable scale and bias
        logits = similarities * self.logit_scale + self.logit_bias

        # Create labels: positivos en diagonal, negativos fuera
        # y = 1 para (i,i), y = 0 para (i,j) donde i != j
        labels = torch.eye(batch_size, device=device)

        # Binary cross-entropy con logits
        # Positivos: -log(sigmoid(logit))
        # Negativos: -log(1 - sigmoid(logit))
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

        # Metricas
        with torch.no_grad():
            # Accuracy en diagonal (positivos)
            pos_correct = (torch.sigmoid(logits.diagonal()) > 0.5).float().mean()

            # Accuracy en off-diagonal (negativos)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_logits = logits[mask]
            neg_correct = (torch.sigmoid(neg_logits) < 0.5).float().mean()

        return {
            'loss': loss,
            'positive_accuracy': pos_correct,
            'negative_accuracy': neg_correct,
            'logit_scale': self.logit_scale.item(),
            'logit_bias': self.logit_bias.item()
        }


class SigLIPModel(nn.Module):
    """
    Modelo SigLIP completo.
    Similar a CLIP pero con sigmoid loss.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # SigLIP loss
        self.loss_fn = SigLIPLoss()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode y normaliza imagenes."""
        embeds = self.vision_encoder(images)
        return F.normalize(embeds, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode y normaliza textos."""
        embeds = self.text_encoder(input_ids, attention_mask)
        return F.normalize(embeds, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass para entrenamiento."""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)

        loss_dict = self.loss_fn(image_embeds, text_embeds)

        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            **loss_dict
        }
```

## Aplicaciones en Seguridad

### Image Retrieval para Threat Intelligence

```python
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ThreatImage:
    """Representa una imagen en la base de datos de amenazas."""
    id: str
    path: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class ThreatImageDatabase:
    """
    Base de datos de imagenes de amenazas con retrieval CLIP.

    Casos de uso:
    - Buscar malware con UI similar
    - Encontrar paginas de phishing que imitan marcas
    - Identificar variantes de ransomware por screenshots
    """

    def __init__(
        self,
        clip_wrapper: CLIPWrapper,
        similarity_threshold: float = 0.7
    ):
        self.clip = clip_wrapper
        self.similarity_threshold = similarity_threshold

        self.images: List[ThreatImage] = []
        self.embeddings: Optional[torch.Tensor] = None

    def add_image(
        self,
        image_id: str,
        image: Image.Image,
        metadata: Dict[str, Any]
    ) -> None:
        """Anade una imagen a la base de datos."""
        # Compute embedding
        embedding = self.clip.encode_images([image]).cpu().numpy()[0]

        threat_image = ThreatImage(
            id=image_id,
            path=metadata.get('path', ''),
            embedding=embedding,
            metadata=metadata
        )
        self.images.append(threat_image)

        # Invalidar cache de embeddings
        self.embeddings = None

    def build_index(self) -> None:
        """Construye indice de embeddings para busqueda rapida."""
        if not self.images:
            return

        self.embeddings = torch.tensor(
            np.stack([img.embedding for img in self.images])
        ).to(self.clip.device)

    def search_by_image(
        self,
        query_image: Image.Image,
        top_k: int = 10
    ) -> List[Tuple[ThreatImage, float]]:
        """
        Busca imagenes similares a una query.

        Returns:
            Lista de (ThreatImage, similitud) ordenada
        """
        if self.embeddings is None:
            self.build_index()

        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Encode query
        query_embedding = self.clip.encode_images([query_image])

        # Similitudes
        similarities = (query_embedding @ self.embeddings.T).squeeze(0)

        # Top-k
        top_k = min(top_k, len(self.images))
        top_similarities, top_indices = torch.topk(similarities, top_k)

        results = [
            (self.images[idx], top_similarities[i].item())
            for i, idx in enumerate(top_indices)
            if top_similarities[i].item() >= self.similarity_threshold
        ]

        return results

    def search_by_description(
        self,
        description: str,
        top_k: int = 10
    ) -> List[Tuple[ThreatImage, float]]:
        """
        Busca imagenes por descripcion textual.

        Ejemplo: "phishing page imitating PayPal login"
        """
        if self.embeddings is None:
            self.build_index()

        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        # Encode query text
        query_embedding = self.clip.encode_texts([description])

        # Similitudes
        similarities = (query_embedding @ self.embeddings.T).squeeze(0)

        # Top-k
        top_k = min(top_k, len(self.images))
        top_similarities, top_indices = torch.topk(similarities, top_k)

        results = [
            (self.images[idx], top_similarities[i].item())
            for i, idx in enumerate(top_indices)
        ]

        return results

    def find_similar_threats(
        self,
        query_image: Image.Image,
        threat_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analisis completo de amenazas similares.
        """
        results = self.search_by_image(query_image, top_k=20)

        if not results:
            return {
                'similar_threats_found': False,
                'message': 'No similar threats in database'
            }

        # Filtrar por tipo si se especifica
        if threat_type:
            results = [
                (img, score) for img, score in results
                if img.metadata.get('threat_type') == threat_type
            ]

        # Analisis
        threat_types = {}
        for img, score in results:
            t_type = img.metadata.get('threat_type', 'unknown')
            if t_type not in threat_types:
                threat_types[t_type] = []
            threat_types[t_type].append(score)

        return {
            'similar_threats_found': True,
            'num_matches': len(results),
            'top_match': {
                'id': results[0][0].id,
                'similarity': results[0][1],
                'metadata': results[0][0].metadata
            } if results else None,
            'threat_type_distribution': {
                k: {'count': len(v), 'avg_similarity': np.mean(v)}
                for k, v in threat_types.items()
            },
            'all_matches': [
                {'id': img.id, 'similarity': score, 'type': img.metadata.get('threat_type')}
                for img, score in results[:10]
            ]
        }


# Ejemplo de uso para phishing detection
def demo_phishing_detection():
    """Demo: Detectar phishing usando similarity con paginas conocidas."""

    clip = CLIPWrapper()
    db = ThreatImageDatabase(clip, similarity_threshold=0.6)

    # En produccion: cargar base de datos de phishing conocido
    # db.add_image("paypal_phish_001", phish_screenshot, {
    #     "threat_type": "phishing",
    #     "target_brand": "PayPal",
    #     "first_seen": "2024-01-15"
    # })

    # Busqueda por texto
    results = db.search_by_description(
        "login page that looks like PayPal with suspicious URL"
    )

    # Busqueda por imagen sospechosa
    # suspicious_screenshot = Image.open("suspect.png")
    # analysis = db.find_similar_threats(suspicious_screenshot, threat_type="phishing")

    print("Phishing detection demo initialized")
    return db
```

### Brand Impersonation Detection

```python
class BrandProtectionSystem:
    """
    Sistema de proteccion de marca usando CLIP.
    Detecta impersonacion visual de marcas en:
    - Paginas web
    - Apps moviles
    - Emails
    - Redes sociales
    """

    def __init__(self, clip_wrapper: CLIPWrapper):
        self.clip = clip_wrapper

        # Embeddings de marcas legitimas (pre-computados)
        self.brand_embeddings: Dict[str, torch.Tensor] = {}
        self.brand_descriptions: Dict[str, List[str]] = {}

    def register_brand(
        self,
        brand_name: str,
        legitimate_screenshots: List[Image.Image],
        descriptions: List[str]
    ) -> None:
        """
        Registra una marca con sus assets visuales legitimos.

        Args:
            brand_name: Nombre de la marca
            legitimate_screenshots: Screenshots de paginas/apps legitimas
            descriptions: Descripciones textuales de la marca
        """
        # Embed imagenes
        image_embeds = self.clip.encode_images(legitimate_screenshots)

        # Embed descripciones
        text_embeds = self.clip.encode_texts(descriptions)

        # Combinar (promedio ponderado o concatenar)
        combined = torch.cat([image_embeds, text_embeds], dim=0)
        mean_embed = combined.mean(dim=0, keepdim=True)
        mean_embed = mean_embed / mean_embed.norm(dim=-1, keepdim=True)

        self.brand_embeddings[brand_name] = mean_embed
        self.brand_descriptions[brand_name] = descriptions

    def check_impersonation(
        self,
        screenshot: Image.Image,
        claimed_brand: Optional[str] = None,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verifica si un screenshot podria ser impersonacion de marca.

        Args:
            screenshot: Screenshot a verificar
            claimed_brand: Marca que dice ser (opcional)
            threshold: Umbral de similitud para considerar match

        Returns:
            Analisis de impersonacion
        """
        # Encode screenshot
        screenshot_embed = self.clip.encode_images([screenshot])

        # Comparar con todas las marcas registradas
        similarities = {}
        for brand, brand_embed in self.brand_embeddings.items():
            sim = (screenshot_embed @ brand_embed.T).item()
            similarities[brand] = sim

        # Encontrar mejor match
        best_brand = max(similarities, key=similarities.get)
        best_similarity = similarities[best_brand]

        # Analisis
        is_suspicious = False
        risk_level = "low"
        reason = ""

        if claimed_brand:
            # Verificar si coincide con lo que dice ser
            claimed_similarity = similarities.get(claimed_brand, 0)

            if claimed_similarity < threshold:
                is_suspicious = True
                risk_level = "high"
                reason = f"Claims to be {claimed_brand} but low visual similarity ({claimed_similarity:.2f})"

                if best_similarity > threshold:
                    reason += f". Looks more like {best_brand} ({best_similarity:.2f})"

        else:
            # Sin claim, buscar si imita alguna marca
            if best_similarity > threshold:
                is_suspicious = True
                risk_level = "medium"
                reason = f"Visually similar to {best_brand} ({best_similarity:.2f}). Verify legitimacy."

        return {
            'is_suspicious': is_suspicious,
            'risk_level': risk_level,
            'reason': reason,
            'claimed_brand': claimed_brand,
            'best_match': {
                'brand': best_brand,
                'similarity': best_similarity
            },
            'all_similarities': similarities
        }


def setup_brand_protection():
    """Configura sistema de proteccion de marca."""

    clip = CLIPWrapper()
    system = BrandProtectionSystem(clip)

    # Registrar marcas (en produccion, usar assets reales)
    # system.register_brand(
    #     "PayPal",
    #     legitimate_screenshots=[paypal_login, paypal_home, paypal_app],
    #     descriptions=[
    #         "PayPal official login page with blue header",
    #         "PayPal secure payment interface",
    #         "PayPal mobile application screen"
    #     ]
    # )

    print("Brand protection system initialized")
    return system
```

## Mejores Modelos OpenCLIP

```
Comparacion de Modelos OpenCLIP (2024):

┌──────────────────────┬──────────┬────────────┬────────────────────────┐
│ Modelo               │ ImageNet │ Zero-Shot  │ Notas                  │
│                      │ Acc (%)  │ Retrieval  │                        │
├──────────────────────┼──────────┼────────────┼────────────────────────┤
│ ViT-B/32 (OpenAI)    │ 63.3     │ Baseline   │ Rapido, ligero         │
│ ViT-B/16 (OpenAI)    │ 68.3     │ +5%        │ Mejor que B/32         │
│ ViT-L/14 (OpenAI)    │ 75.5     │ +15%       │ Best OpenAI            │
│ ViT-L/14@336 (OpenAI)│ 76.6     │ +17%       │ Higher res             │
├──────────────────────┼──────────┼────────────┼────────────────────────┤
│ ViT-G/14 (LAION-2B)  │ 80.1     │ +25%       │ Huge, SOTA open        │
│ ViT-bigG/14 (LAION)  │ 80.5     │ +27%       │ Largest open           │
│ EVA-CLIP             │ 82.0     │ +30%       │ Best overall           │
├──────────────────────┼──────────┼────────────┼────────────────────────┤
│ SigLIP-B/16          │ 73.5     │ Similar    │ Mejor con batch peq.   │
│ SigLIP-L/16@384      │ 78.8     │ +20%       │ Excellent balance      │
│ SigLIP-So400m/14@384 │ 83.1     │ Best       │ Google's best          │
└──────────────────────┴──────────┴────────────┴────────────────────────┘

Recomendaciones por caso de uso:

1. Prototipado rapido: ViT-B/32 (rapido, suficiente para demos)
2. Produccion general: ViT-L/14 o SigLIP-L/16 (buen balance)
3. Maxima precision: ViT-G/14 o SigLIP-So400m (pero pesado)
4. Batches pequenos: SigLIP cualquier tamanio
5. Recursos limitados: ViT-B/16 con cuantizacion
```

```python
def load_best_clip_model(
    use_case: str = "production",
    device: str = "cuda"
) -> CLIPWrapper:
    """
    Carga el mejor modelo CLIP segun el caso de uso.

    Args:
        use_case: "prototyping", "production", "accuracy", "limited_resources"
        device: Dispositivo de ejecucion
    """
    model_configs = {
        "prototyping": ("ViT-B-32", "openai"),
        "production": ("ViT-L-14", "openai"),
        "accuracy": ("ViT-bigG-14", "laion2b_s39b_b160k"),
        "limited_resources": ("ViT-B-16", "openai"),
        "small_batch": ("ViT-L-16-SigLIP", "webli")
    }

    if use_case not in model_configs:
        raise ValueError(f"Unknown use case: {use_case}")

    model_name, pretrained = model_configs[use_case]

    return CLIPWrapper(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
```

## Resumen

```
CLIP Y SIGLIP - KEY TAKEAWAYS:

┌─────────────────────────────────────────────────────────────────┐
│                     CONCEPTOS CLAVE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CLIP ARCHITECTURE                                           │
│     ├── Dual encoder: Vision (ViT) + Text (Transformer)       │
│     ├── Contrastive pre-training en 400M pares                 │
│     ├── Proyeccion a espacio compartido                        │
│     └── Zero-shot transfer a cualquier tarea                   │
│                                                                 │
│  2. SIGLIP IMPROVEMENTS                                         │
│     ├── Sigmoid loss vs Softmax                                │
│     ├── Cada par = clasificacion binaria independiente         │
│     ├── Funciona mejor con batches pequenos                    │
│     └── Mejor calibracion de probabilidades                    │
│                                                                 │
│  3. ZERO-SHOT CLASSIFICATION                                    │
│     ├── Generar prompts para cada clase                        │
│     ├── Comparar embedding imagen con texto                    │
│     ├── Prompt engineering mejora resultados                   │
│     └── Template ensembling para robustez                      │
│                                                                 │
│  4. RETRIEVAL APPLICATIONS                                      │
│     ├── Image-to-Text: encontrar descripciones                 │
│     ├── Text-to-Image: busqueda semantica                      │
│     └── Cross-modal search en bases de datos                   │
│                                                                 │
│  5. SECURITY APPLICATIONS                                       │
│     ├── Phishing detection por similitud visual                │
│     ├── Brand impersonation detection                          │
│     ├── Threat intelligence visual                             │
│     └── Malware UI classification                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

BEST PRACTICES:
- Usar prompt ensembling para clasificacion robusta
- Pre-computar embeddings cuando sea posible
- SigLIP para batches pequenos o recursos limitados
- Ajustar threshold segun precision/recall deseado
- Considerar fine-tuning para dominios especificos
```

## Referencias

1. "Learning Transferable Visual Models From Natural Language Supervision" - Radford et al., 2021 (CLIP)
2. "Sigmoid Loss for Language Image Pre-Training" - Zhai et al., 2023 (SigLIP)
3. "OpenCLIP: Open-source implementation of CLIP" - Ilharco et al., 2021
4. "Scaling Up Visual and Vision-Language Representation Learning" - Jia et al., 2021 (ALIGN)
5. "EVA-CLIP: Improved Training Techniques for CLIP" - Fang et al., 2023
