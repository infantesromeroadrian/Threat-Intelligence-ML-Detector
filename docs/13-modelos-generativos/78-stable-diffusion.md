# Stable Diffusion: Arquitectura y Aplicaciones

## Introduccion

**Stable Diffusion** es un modelo de difusion latente (Latent Diffusion Model, LDM) que revoluciono la generacion de imagenes text-to-image. En lugar de operar directamente en el espacio de pixeles (computacionalmente costoso), opera en un **espacio latente comprimido**, reduciendo dramaticamente los requisitos de memoria y tiempo de inferencia.

```
STABLE DIFFUSION: VISION GENERAL
================================

Componentes principales:

┌─────────────────────────────────────────────────────────────────────────────┐
│                         STABLE DIFFUSION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "A photo of a cat"     ┌──────────────┐                                  │
│          │               │   CLIP Text  │                                  │
│          └──────────────▶│   Encoder    │──────────────┐                  │
│                          └──────────────┘              │                  │
│                                                        │ text embeddings   │
│                                                        │  (77, 768)       │
│                                                        ▼                  │
│   ┌───────────┐        ┌───────────────────────────────────┐              │
│   │   Ruido   │        │                                   │              │
│   │   z_T     │───────▶│          U-Net (Denoiser)         │              │
│   │  (4,64,64)│        │     Condicionado con texto        │              │
│   └───────────┘        │                                   │              │
│        │               └─────────────┬─────────────────────┘              │
│        │                             │                                     │
│        │  T iteraciones              │ z_0 (4, 64, 64)                    │
│        │  de denoising               │                                     │
│        └─────────────────────────────┘                                     │
│                                      │                                     │
│                                      ▼                                     │
│                          ┌──────────────────┐                              │
│                          │   VAE Decoder    │                              │
│                          │  (latent→pixel)  │                              │
│                          └────────┬─────────┘                              │
│                                   │                                        │
│                                   ▼                                        │
│                          ┌──────────────────┐                              │
│                          │   Imagen Final   │                              │
│                          │   (3, 512, 512)  │                              │
│                          └──────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


POR QUE LATENT DIFFUSION?
=========================

Diffusion en pixeles:           Latent Diffusion:
────────────────────            ─────────────────

512x512x3 = 786,432 dims        64x64x4 = 16,384 dims
                                      │
   U-Net procesa TODA              U-Net procesa 48x
   la imagen en alta               MENOS dimensiones
   resolucion                           │
        │                               │
        ▼                               ▼
   Muy lento                       Muy rapido
   Mucha memoria                   Poca memoria
   (16-32 GB VRAM)                 (4-8 GB VRAM)


FACTOR DE COMPRESION:
────────────────────

    Imagen: 512 x 512 x 3 = 786,432 valores
    Latent:  64 x  64 x 4 =  16,384 valores
                            ───────
                              48x compresion espacial

    El VAE preserva la informacion semantica importante
    mientras descarta detalles redundantes.
```

---

## Arquitectura del VAE (Autoencoder)

### Encoder-Decoder para Compresion

```
VAE EN STABLE DIFFUSION
=======================

El VAE tiene dos funciones:

1. ENCODER: imagen → latent (para entrenamiento)
2. DECODER: latent → imagen (para generacion)


ARQUITECTURA DEL ENCODER:
─────────────────────────

Input: (B, 3, 512, 512)
          │
          ▼
┌─────────────────────────────────────────┐
│  Conv 3x3, 128 channels                 │
└────────────────┬────────────────────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
     ┌─────────┐   ┌─────────┐
     │ResBlock │   │ResBlock │
     │  128    │   │  128    │
     └────┬────┘   └────┬────┘
          │             │
          └──────┬──────┘
                 │
          ▼ Downsample 2x
                 │
┌─────────────────────────────────────────┐
│  256 channels, 256x256                  │
│  ResBlock → ResBlock → Downsample       │
└────────────────┬────────────────────────┘
                 │
          ▼ Downsample 2x
                 │
┌─────────────────────────────────────────┐
│  512 channels, 128x128                  │
│  ResBlock → ResBlock → Downsample       │
└────────────────┬────────────────────────┘
                 │
          ▼ Downsample 2x
                 │
┌─────────────────────────────────────────┐
│  512 channels, 64x64                    │
│  ResBlock → Attention → ResBlock        │
└────────────────┬────────────────────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
    ┌──────────┐  ┌──────────┐
    │ Conv μ   │  │ Conv σ   │
    │ (4 ch)   │  │ (4 ch)   │
    └────┬─────┘  └────┬─────┘
         │             │
         └──────┬──────┘
                │
                ▼
     z = μ + σ * ε    (reparameterization)
                │
                ▼
    Output: (B, 4, 64, 64)


DECODER (simetrico):
────────────────────

Input: z (B, 4, 64, 64)
          │
          ▼
┌─────────────────────────────────────────┐
│  Conv → ResBlocks → Attention           │
│  512 channels, 64x64                    │
└────────────────┬────────────────────────┘
                 │
          ▼ Upsample 2x
                 │
┌─────────────────────────────────────────┐
│  ResBlocks, 512 ch, 128x128             │
└────────────────┬────────────────────────┘
                 │
          ▼ Upsample 2x  (repeat...)
                 │
                ...
                 │
          ▼
    Output: (B, 3, 512, 512)


NOTA IMPORTANTE:
────────────────
El VAE de SD NO es variacional durante inferencia.
Se usa como autoencoder deterministico:
    - Encoder: imagen → μ (ignoramos σ)
    - Decoder: z → imagen
```

### Implementacion del VAE

```python
"""
VAE para Stable Diffusion.
Implementacion simplificada de la arquitectura.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    """Bloque residual con GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        k = k.view(B, C, H * W)                   # [B, C, HW]
        v = v.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]

        # Attention
        attn = torch.bmm(q, k) * self.scale  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)
        h = torch.bmm(attn, v)  # [B, HW, C]

        # Reshape back
        h = h.transpose(1, 2).view(B, C, H, W)
        h = self.proj(h)

        return x + h


class Downsample(nn.Module):
    """Downsample 2x con strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample 2x con interpolacion + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class VAEEncoder(nn.Module):
    """
    Encoder del VAE de Stable Diffusion.
    Imagen (3, 512, 512) -> Latent (4, 64, 64)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch))
                ch = out_ch

            # Attention en la penultima resolucion
            if i == len(channel_mult) - 1:
                self.down_blocks.append(AttentionBlock(ch))

            # Downsample excepto ultima capa
            if i < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(ch))

        # Mid block
        self.mid_block1 = ResidualBlock(ch, ch)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch)

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, latent_channels * 2, 3, padding=1)  # mu y log_var

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Imagen [B, 3, H, W]

        Returns:
            mu: Media [B, 4, H/8, W/8]
            log_var: Log varianza [B, 4, H/8, W/8]
        """
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        mu, log_var = h.chunk(2, dim=1)

        return mu, log_var


class VAEDecoder(nn.Module):
    """
    Decoder del VAE de Stable Diffusion.
    Latent (4, 64, 64) -> Imagen (3, 512, 512)
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()

        # Channels despues del encoder
        ch = base_channels * channel_mult[-1]

        self.conv_in = nn.Conv2d(latent_channels, ch, 3, padding=1)

        # Mid block
        self.mid_block1 = ResidualBlock(ch, ch)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch)

        # Decoder blocks (reverse order)
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch, out_ch))
                ch = out_ch

            # Attention
            if i == 0:
                self.up_blocks.append(AttentionBlock(ch))

            # Upsample excepto ultima capa
            if i < len(channel_mult) - 1:
                self.up_blocks.append(Upsample(ch))

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent [B, 4, H/8, W/8]

        Returns:
            Imagen reconstruida [B, 3, H, W]
        """
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class StableDiffusionVAE(nn.Module):
    """VAE completo para Stable Diffusion."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,  # Factor de escala estandar de SD
    ):
        super().__init__()

        self.encoder = VAEEncoder(in_channels, latent_channels)
        self.decoder = VAEDecoder(in_channels, latent_channels)
        self.scaling_factor = scaling_factor

    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        """
        Encode imagen a latent.

        Args:
            x: Imagen [B, 3, H, W] en rango [-1, 1]
            sample: Si True, muestrea de la distribucion

        Returns:
            Latent [B, 4, H/8, W/8]
        """
        mu, log_var = self.encoder(x)

        if sample:
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # Escalar latents
        z = z * self.scaling_factor

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent a imagen.

        Args:
            z: Latent [B, 4, H/8, W/8]

        Returns:
            Imagen [B, 3, H, W] en rango [-1, 1]
        """
        # Desescalar latents
        z = z / self.scaling_factor

        x = self.decoder(z)

        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass completo (encode + decode).

        Returns:
            (reconstructed, mu, log_var)
        """
        mu, log_var = self.encoder(x)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

        x_recon = self.decoder(z)

        return x_recon, mu, log_var
```

---

## CLIP Text Encoder

```
CLIP TEXT ENCODER EN STABLE DIFFUSION
=====================================

CLIP (Contrastive Language-Image Pre-training) proporciona
el encoding de texto que condiciona la generacion.


ARQUITECTURA:
─────────────

"A photo of a cat wearing a hat"
         │
         ▼
┌─────────────────────────────────────────┐
│            TOKENIZER                     │
│  BPE (Byte Pair Encoding)               │
│                                          │
│  Tokens: [<start>, "a", "photo", "of",  │
│           "a", "cat", "wearing", "a",   │
│           "hat", <end>, <pad>...]       │
│                                          │
│  Shape: (77,) - secuencia fija          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         TOKEN EMBEDDINGS                 │
│                                          │
│  Embedding table: 49408 x 768           │
│  + Positional embeddings: 77 x 768      │
│                                          │
│  Output: (77, 768)                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      TRANSFORMER ENCODER (x12)          │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │  Multi-Head Self-Attention      │    │
│  │  (causal mask)                  │    │
│  └─────────────────────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │  Layer Norm + Feed Forward      │    │
│  │  768 → 3072 → 768               │    │
│  └─────────────────────────────────┘    │
│                                          │
│  x 12 capas                             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           LAYER NORM                     │
│                                          │
│  Output: (77, 768)                       │
│                                          │
│  Este tensor se usa para condicionar    │
│  la U-Net via cross-attention           │
└─────────────────────────────────────────┘


VERSIONES DE CLIP EN SD:
────────────────────────

┌────────────────┬─────────────┬─────────────┬─────────────┐
│ Version SD     │ CLIP Model  │ Hidden Size │ Capas       │
├────────────────┼─────────────┼─────────────┼─────────────┤
│ SD 1.x         │ ViT-L/14    │ 768         │ 12          │
│ SD 2.x         │ ViT-H/14    │ 1024        │ 24          │
│ SDXL           │ ViT-L + ViT-G│ 768 + 1280  │ 12 + 32    │
└────────────────┴─────────────┴─────────────┴─────────────┘

SDXL usa DOS encoders de texto y concatena los embeddings.
```

### Implementacion del Text Encoder

```python
"""
CLIP Text Encoder para Stable Diffusion.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionTextEncoder:
    """
    Wrapper para el CLIP Text Encoder usado en Stable Diffusion.
    Usa la implementacion de HuggingFace Transformers.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        max_length: int = 77,
    ):
        """
        Args:
            model_name: Nombre del modelo CLIP en HuggingFace
            device: Dispositivo (cuda/cpu)
            max_length: Longitud maxima de secuencia (77 para SD)
        """
        self.device = device
        self.max_length = max_length

        # Cargar tokenizer y modelo
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)

        # Congelar parametros
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(
        self,
        prompts: List[str],
        return_pooled: bool = False,
    ) -> Tensor:
        """
        Encode una lista de prompts a embeddings.

        Args:
            prompts: Lista de strings
            return_pooled: Si retornar tambien el pooled output

        Returns:
            Text embeddings [B, 77, 768]
        """
        # Tokenizar
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        # Forward pass
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Usar el ultimo hidden state
        text_embeddings = outputs.last_hidden_state

        if return_pooled:
            pooled = outputs.pooler_output
            return text_embeddings, pooled

        return text_embeddings

    def get_unconditional_embeddings(self, batch_size: int = 1) -> Tensor:
        """
        Obtiene embeddings para texto vacio (para classifier-free guidance).

        Args:
            batch_size: Numero de embeddings a generar

        Returns:
            Embeddings [B, 77, 768]
        """
        prompts = [""] * batch_size
        return self.encode(prompts)


# Ejemplo de uso
def demo_text_encoder():
    """Demostracion del text encoder."""

    encoder = StableDiffusionTextEncoder(device="cuda")

    prompts = [
        "A beautiful sunset over mountains",
        "A cyberpunk city at night with neon lights",
    ]

    embeddings = encoder.encode(prompts)
    print(f"Shape de embeddings: {embeddings.shape}")  # [2, 77, 768]

    # Para classifier-free guidance
    uncond = encoder.get_unconditional_embeddings(batch_size=2)
    print(f"Shape de uncond: {uncond.shape}")  # [2, 77, 768]

    return embeddings, uncond
```

---

## U-Net Condicionada con Cross-Attention

```
U-NET CONDICIONADA EN STABLE DIFFUSION
======================================

La U-Net de SD es similar a DDPM pero con DOS diferencias clave:

1. Opera en espacio LATENT (4, 64, 64) no pixel
2. Tiene CROSS-ATTENTION con text embeddings


ARQUITECTURA GENERAL:
─────────────────────

                    Text Embeddings (77, 768)
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
              Cross-Attn      Cross-Attn
                    │               │
┌───────────────────┼───────────────┼───────────────────┐
│                   │               │                   │
│   Input z_t       │               │       Output ε    │
│   (4, 64, 64)     │               │    (4, 64, 64)   │
│        │          │               │          ▲        │
│        ▼          ▼               ▼          │        │
│   ┌─────────┐ ┌─────────┐   ┌─────────┐ ┌─────────┐   │
│   │ Down    │ │ Cross   │   │ Cross   │ │  Up     │   │
│   │ Block   │ │ Attn    │   │ Attn    │ │ Block   │   │
│   └────┬────┘ └────┬────┘   └────┬────┘ └────┬────┘   │
│        │           │             │           │        │
│        └───────────┼─────────────┼───────────┘        │
│                    │             │                    │
│                    │   Middle    │                    │
│                    │   Block     │                    │
│                    └──────┬──────┘                    │
│                           │                          │
│                   ┌───────┴───────┐                  │
│                   │  Self-Attn    │                  │
│                   │  Cross-Attn   │                  │
│                   └───────────────┘                  │
│                                                       │
│                  + Time Embedding                     │
│                                                       │
└───────────────────────────────────────────────────────┘


DETALLE DEL CROSS-ATTENTION BLOCK:
──────────────────────────────────

┌─────────────────────────────────────────────────────────┐
│                 TRANSFORMER BLOCK                        │
│                                                          │
│   Input x (features de imagen)                          │
│        │                                                │
│        ▼                                                │
│   ┌──────────────────────────────────────┐              │
│   │         SELF-ATTENTION               │              │
│   │                                      │              │
│   │   Q = x @ W_q                        │              │
│   │   K = x @ W_k                        │              │
│   │   V = x @ W_v                        │              │
│   │                                      │              │
│   │   Attn(Q, K, V) = softmax(QK^T)V     │              │
│   └──────────────────┬───────────────────┘              │
│                      │                                  │
│                      ▼                                  │
│   ┌──────────────────────────────────────┐              │
│   │         CROSS-ATTENTION              │              │
│   │                                      │              │
│   │   Q = x @ W_q          (de imagen)   │              │
│   │   K = text @ W_k       (de texto)    │◄── context   │
│   │   V = text @ W_v       (de texto)    │    (77, 768) │
│   │                                      │              │
│   │   Attn(Q, K, V) = softmax(QK^T)V     │              │
│   └──────────────────┬───────────────────┘              │
│                      │                                  │
│                      ▼                                  │
│   ┌──────────────────────────────────────┐              │
│   │         FEED-FORWARD                 │              │
│   │                                      │              │
│   │   x = GEGLU(Linear(x))               │              │
│   │   x = Linear(x)                      │              │
│   └──────────────────┬───────────────────┘              │
│                      │                                  │
│                      ▼                                  │
│                  Output x                               │
└─────────────────────────────────────────────────────────┘


COMO FUNCIONA EL CONDICIONAMIENTO:
──────────────────────────────────

Cross-Attention permite que cada "parche" de la imagen
atienda a diferentes partes del texto.

Ejemplo: "A cat wearing a red hat"

    Parche de la cabeza → atiende a "hat", "red"
    Parche del cuerpo  → atiende a "cat"
    Parche del fondo   → atiende menos (bajo weight)

Visualization:

    Query (imagen)     Keys (texto)
    ┌───┬───┬───┐      ┌─────────────────────┐
    │ 1 │ 2 │ 3 │      │ A cat wearing a     │
    ├───┼───┼───┤      │ red hat             │
    │ 4 │ 5 │ 6 │      └─────────────────────┘
    ├───┼───┼───┤              │
    │ 7 │ 8 │ 9 │              │
    └───┴───┴───┘              │
          │                    │
          │  Cross-Attention   │
          └────────────────────┘
                    │
                    ▼
         Attention weights:
         Parche 2 → "hat": 0.8
         Parche 5 → "cat": 0.9
         etc.
```

### Implementacion del Transformer Block

```python
"""
Transformer Blocks para la U-Net de Stable Diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class GEGLU(nn.Module):
    """
    GEGLU activation: x * GELU(gate)
    Usado en los feed-forward blocks.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class CrossAttention(nn.Module):
    """
    Cross-Attention layer.
    Puede ser self-attention (context=None) o cross-attention.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Query tensor [B, N, C]
            context: Key/Value tensor [B, M, C'] (si None, self-attention)

        Returns:
            Output [B, N, C]
        """
        h = self.heads

        q = self.to_q(x)

        # Self-attention si no hay context
        context = context if context is not None else x

        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape para multi-head
        B, N, _ = q.shape
        _, M, _ = k.shape

        q = q.view(B, N, h, -1).transpose(1, 2)  # [B, h, N, d]
        k = k.view(B, M, h, -1).transpose(1, 2)  # [B, h, M, d]
        v = v.view(B, M, h, -1).transpose(1, 2)  # [B, h, M, d]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, -1)  # [B, N, h*d]

        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network con GEGLU."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """
    Transformer block usado en la U-Net de Stable Diffusion.

    Consiste en:
    1. Self-Attention
    2. Cross-Attention (condicionado con texto)
    3. Feed-Forward
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        context_dim: int = 768,  # Dimension de CLIP
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Image features [B, N, dim]
            context: Text embeddings [B, seq_len, context_dim]

        Returns:
            Output [B, N, dim]
        """
        # Self-attention
        x = x + self.attn1(self.norm1(x))

        # Cross-attention con texto
        x = x + self.attn2(self.norm2(x), context=context)

        # Feed-forward
        x = x + self.ff(self.norm3(x))

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer espacial para la U-Net.
    Convierte features 2D a secuencia, aplica transformer, y vuelve a 2D.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        dim_head: int = 64,
        depth: int = 1,
        context_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Linear(in_channels, in_channels)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=in_channels,
                num_heads=num_heads,
                dim_head=dim_head,
                context_dim=context_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Features [B, C, H, W]
            context: Text embeddings [B, seq_len, context_dim]

        Returns:
            Output [B, C, H, W]
        """
        B, C, H, W = x.shape

        residual = x

        # Norm
        x = self.norm(x)

        # Reshape a secuencia: [B, C, H, W] -> [B, H*W, C]
        x = x.view(B, C, H * W).transpose(1, 2)

        # Project in
        x = self.proj_in(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Project out
        x = self.proj_out(x)

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)

        return x + residual
```

---

## Classifier-Free Guidance

```
CLASSIFIER-FREE GUIDANCE (CFG)
==============================

CFG es la tecnica que permite controlar que tan "fuerte"
el modelo sigue el prompt de texto.


INTUICION:
──────────

Sin CFG:
    El modelo genera imagenes que "mas o menos" siguen el prompt

Con CFG:
    El modelo genera imagenes que FUERTEMENTE siguen el prompt


MATEMATICA:
───────────

Durante sampling, en lugar de usar directamente ε_θ(z_t, t, c):

    ε_guided = ε_uncond + guidance_scale * (ε_cond - ε_uncond)

Donde:
    - ε_cond: Prediccion condicionada (con texto)
    - ε_uncond: Prediccion no condicionada (texto vacio "")
    - guidance_scale: Tipicamente 7.5 para SD


INTERPRETACION GEOMETRICA:
──────────────────────────

                    ε_cond
                      │
                      │    ε_guided (scale > 1)
                      │   ╱
                      │  ╱
                      │ ╱
                      │╱
    ε_uncond ─────────●───────────────
                     ╱│
                    ╱ │
                   ╱  │
                  ╱   │
                 ╱    │  ε_guided (scale < 0)
                      │


    scale = 0   : Solo ε_uncond (ignora texto)
    scale = 1   : Solo ε_cond (sin guia)
    scale = 7.5 : Guia fuerte hacia texto
    scale > 15  : Puede sobre-saturar


EFECTO EN LA GENERACION:
────────────────────────

Prompt: "A cat sitting on a couch"

┌─────────────────────────────────────────────────────────┐
│                                                         │
│   scale=1        scale=5        scale=7.5      scale=15│
│                                                         │
│   ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐ │
│   │ 🐱? │        │ 🐱  │        │ 🐱  │        │ 🐱  │ │
│   │     │        │ 🛋️  │        │ 🛋️  │        │ 🛋️! │ │
│   │     │        │     │        │ ✓✓✓ │        │ !!!  │ │
│   └─────┘        └─────┘        └─────┘        └─────┘ │
│                                                         │
│   Vago,          Razonable,     Optimo,        Saturado│
│   no sigue       sigue bien     mejor calidad  artefactos│
│   el prompt                                             │
│                                                         │
└─────────────────────────────────────────────────────────┘


IMPLEMENTACION PRACTICA:
────────────────────────

Para cada step de denoising, hay que hacer DOS forward passes:

1. Forward con prompt real:     ε_cond = model(z_t, t, text_emb)
2. Forward con prompt vacio:    ε_uncond = model(z_t, t, empty_emb)
3. Combinar:                    ε = ε_uncond + scale * (ε_cond - ε_uncond)

Esto DUPLICA el costo computacional.

Optimizacion: Batch de 2
    - Concatenar [z_t, z_t] en batch
    - Concatenar [text_emb, empty_emb] en batch
    - Un solo forward pass
    - Separar outputs
```

### Implementacion de CFG

```python
"""
Classifier-Free Guidance para Stable Diffusion.
"""

import torch
from torch import Tensor
from typing import Optional, Callable


def apply_cfg(
    noise_pred_cond: Tensor,
    noise_pred_uncond: Tensor,
    guidance_scale: float = 7.5,
) -> Tensor:
    """
    Aplica Classifier-Free Guidance.

    Args:
        noise_pred_cond: Prediccion con condicionamiento [B, C, H, W]
        noise_pred_uncond: Prediccion sin condicionamiento [B, C, H, W]
        guidance_scale: Factor de guia (tipico: 7.5)

    Returns:
        Prediccion guiada [B, C, H, W]
    """
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


class CFGDenoiser:
    """
    Wrapper para aplicar CFG al denoiser.
    """

    def __init__(
        self,
        model: Callable,
        guidance_scale: float = 7.5,
    ):
        """
        Args:
            model: Modelo de denoising (U-Net)
            guidance_scale: Factor de CFG
        """
        self.model = model
        self.guidance_scale = guidance_scale

    def __call__(
        self,
        z_t: Tensor,
        t: Tensor,
        text_embeddings: Tensor,
        uncond_embeddings: Tensor,
    ) -> Tensor:
        """
        Forward pass con CFG.

        Args:
            z_t: Latent con ruido [B, 4, H, W]
            t: Timestep [B]
            text_embeddings: Embeddings del prompt [B, 77, 768]
            uncond_embeddings: Embeddings vacios [B, 77, 768]

        Returns:
            Noise prediction guiada [B, 4, H, W]
        """
        # Concatenar para un solo forward pass
        z_t_input = torch.cat([z_t, z_t], dim=0)
        t_input = torch.cat([t, t], dim=0)
        embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)

        # Forward pass
        noise_pred = self.model(z_t_input, t_input, embeddings)

        # Separar predicciones
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)

        # Aplicar CFG
        noise_pred_guided = apply_cfg(
            noise_pred_cond,
            noise_pred_uncond,
            self.guidance_scale,
        )

        return noise_pred_guided
```

---

## Pipeline Completo de Generacion

```python
"""
Pipeline completo de Stable Diffusion.
"""

from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class StableDiffusionConfig:
    """Configuracion para Stable Diffusion."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Generacion
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # Scheduler
    scheduler_type: str = "ddim"  # "ddim", "pndm", "euler"


class StableDiffusionPipeline:
    """
    Pipeline de Stable Diffusion para text-to-image.
    """

    def __init__(self, config: StableDiffusionConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        # Cargar componentes
        self._load_components()

    def _load_components(self):
        """Carga todos los componentes del modelo."""

        model_id = self.config.model_id

        print(f"Cargando modelo: {model_id}")

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae"
        ).to(self.device, self.dtype)

        # Text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        ).to(self.device, self.dtype)

        # U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        ).to(self.device, self.dtype)

        # Scheduler
        if self.config.scheduler_type == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
        elif self.config.scheduler_type == "pndm":
            self.scheduler = PNDMScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
        elif self.config.scheduler_type == "euler":
            self.scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )

        # Congelar parametros
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

        print("Modelo cargado correctamente")

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode prompts a embeddings.

        Args:
            prompt: Prompt(s) positivo
            negative_prompt: Prompt(s) negativo (default: "")

        Returns:
            (text_embeddings, uncond_embeddings)
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        # Tokenizar prompt positivo
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)

        # Encode
        text_embeddings = self.text_encoder(text_input_ids)[0]

        # Prompt negativo (para CFG)
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size

        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        uncond_input_ids = uncond_inputs.input_ids.to(self.device)
        uncond_embeddings = self.text_encoder(uncond_input_ids)[0]

        return text_embeddings, uncond_embeddings

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        return_latents: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Genera imagenes a partir de texto.

        Args:
            prompt: Descripcion de la imagen
            negative_prompt: Que evitar en la imagen
            height: Altura (default: 512)
            width: Ancho (default: 512)
            num_inference_steps: Pasos de denoising
            guidance_scale: Fuerza de CFG
            seed: Semilla para reproducibilidad
            return_latents: Si retornar tambien los latents

        Returns:
            Imagenes generadas [B, 3, H, W] en rango [0, 1]
        """
        # Defaults
        height = height or self.config.height
        width = width or self.config.width
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # Seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Encode prompts
        text_embeddings, uncond_embeddings = self.encode_prompt(
            prompt, negative_prompt
        )

        # Concatenar para CFG
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Preparar latents iniciales
        latent_height = height // 8
        latent_width = width // 8

        latents = torch.randn(
            (batch_size, 4, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Escalar latents
        latents = latents * self.scheduler.init_noise_sigma

        # Configurar scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
            # Expandir latents para CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            # Predecir ruido
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents a imagen
        latents_scaled = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents_scaled).sample

        # Denormalizar a [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)

        if return_latents:
            return images, latents

        return images

    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]],
        image: Tensor,
        strength: float = 0.8,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Tensor:
        """
        Image-to-image generation.

        Args:
            prompt: Descripcion del resultado deseado
            image: Imagen de entrada [B, 3, H, W] en [0, 1]
            strength: Fuerza de modificacion (0-1)
            ...

        Returns:
            Imagen modificada [B, 3, H, W]
        """
        batch_size = image.shape[0]

        # Encode image a latent
        image = image.to(self.device, self.dtype)
        image = image * 2 - 1  # Normalizar a [-1, 1]

        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Encode prompts
        text_embeddings, uncond_embeddings = self.encode_prompt(
            prompt, negative_prompt
        )
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Configurar scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Calcular desde que timestep empezar
        init_timestep = int(num_inference_steps * strength)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        # Agregar ruido al latent
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        noise = torch.randn(latents.shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        # Denoising
        for t in tqdm(timesteps, desc="img2img Denoising"):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        return images


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def demo_stable_diffusion():
    """Demo de generacion con Stable Diffusion."""

    config = StableDiffusionConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        device="cuda",
        dtype=torch.float16,
        num_inference_steps=50,
        guidance_scale=7.5,
    )

    pipeline = StableDiffusionPipeline(config)

    # Text-to-image
    prompt = "A beautiful sunset over mountains, digital art, highly detailed"
    negative_prompt = "blurry, bad quality, distorted"

    images = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=42,
    )

    print(f"Generated image shape: {images.shape}")

    # Guardar imagen
    from torchvision.utils import save_image
    save_image(images, "generated_image.png")

    return images


if __name__ == "__main__":
    demo_stable_diffusion()
```

---

## ControlNet

```
CONTROLNET: CONTROL FINO SOBRE LA GENERACION
============================================

ControlNet permite condicionar la generacion con senales
adicionales: poses, bordes, profundidad, segmentacion, etc.


ARQUITECTURA:
─────────────

ControlNet es una COPIA de la U-Net encoder que se entrena
para procesar condiciones adicionales.

                    Condicion (ej: Canny edges)
                              │
                              ▼
                    ┌──────────────────┐
                    │   ControlNet     │
                    │   (copia del     │
                    │    encoder)      │
                    └────────┬─────────┘
                             │
                    Residuals (skip connections)
                             │
┌────────────────────────────┼────────────────────────────┐
│                            │                            │
│   Latent z_t    ┌──────────▼──────────┐    Output ε    │
│        │        │                     │        ▲        │
│        ▼        │                     │        │        │
│   ┌─────────┐   │     ┌─────────┐     │   ┌─────────┐   │
│   │  U-Net  │   │     │  + res  │     │   │  U-Net  │   │
│   │ Encoder │───┼────▶│         │────▶│───│ Decoder │   │
│   │         │   │     │         │     │   │         │   │
│   └─────────┘   │     └─────────┘     │   └─────────┘   │
│                 │                     │                 │
│                 │       U-Net        │                  │
│                 │      (frozen)       │                 │
│                 │                     │                 │
│                 └─────────────────────┘                 │
│                                                         │
└─────────────────────────────────────────────────────────┘


ZERO CONVOLUTION:
─────────────────

ControlNet usa "zero convolutions" para inicializar:

    - Pesos inicializados a CERO
    - Al inicio, ControlNet no afecta la U-Net
    - Gradualmente aprende a inyectar informacion

    y = zero_conv(c)  # Empieza en 0, aprende poco a poco


TIPOS DE CONDICIONES:
─────────────────────

┌─────────────────┬────────────────────────────────────────┐
│ Condicion       │ Uso                                    │
├─────────────────┼────────────────────────────────────────┤
│ Canny Edges     │ Generar siguiendo bordes               │
│ Depth Map       │ Generar con estructura 3D              │
│ OpenPose        │ Generar personas con pose especifica   │
│ Segmentation    │ Generar con layout de regiones         │
│ Scribbles       │ Generar desde bocetos                  │
│ Normal Map      │ Generar con geometria de superficie    │
│ Line Art        │ Generar desde dibujos lineales         │
└─────────────────┴────────────────────────────────────────┘


EJEMPLO VISUAL:
───────────────

Input:                    Output:
┌─────────────┐          ┌─────────────┐
│    Pose     │          │   Persona   │
│             │          │   con esa   │
│    / \      │  ──────▶ │    pose     │
│   /   \     │          │             │
│  /     \    │          │             │
└─────────────┘          └─────────────┘
  (OpenPose)              (Generado)
```

### Uso de ControlNet con Diffusers

```python
"""
Uso de ControlNet con la libreria diffusers.
"""

from typing import Optional, Union, List
import torch
from torch import Tensor
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image


def load_controlnet_pipeline(
    controlnet_type: str = "canny",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionControlNetPipeline:
    """
    Carga un pipeline de SD con ControlNet.

    Args:
        controlnet_type: Tipo de control ("canny", "depth", "openpose", etc.)
        device: Dispositivo
        dtype: Tipo de datos

    Returns:
        Pipeline configurado
    """
    # Mapping de tipos a modelos
    controlnet_models = {
        "canny": "lllyasviel/sd-controlnet-canny",
        "depth": "lllyasviel/sd-controlnet-depth",
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "scribble": "lllyasviel/sd-controlnet-scribble",
        "seg": "lllyasviel/sd-controlnet-seg",
        "normal": "lllyasviel/sd-controlnet-normal",
    }

    if controlnet_type not in controlnet_models:
        raise ValueError(f"Tipo no soportado: {controlnet_type}")

    # Cargar ControlNet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_models[controlnet_type],
        torch_dtype=dtype,
    )

    # Cargar pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(device)

    # Usar scheduler mas rapido
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    return pipeline


def preprocess_canny(
    image: Union[str, Image.Image, np.ndarray],
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """
    Preprocesa imagen para ControlNet Canny.

    Args:
        image: Imagen de entrada
        low_threshold: Umbral bajo de Canny
        high_threshold: Umbral alto de Canny

    Returns:
        Imagen de bordes Canny
    """
    # Cargar imagen si es string
    if isinstance(image, str):
        image = load_image(image)

    # Convertir a numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Detectar bordes
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Convertir a PIL
    edges = Image.fromarray(edges)

    return edges


def generate_with_controlnet(
    pipeline: StableDiffusionControlNetPipeline,
    prompt: str,
    control_image: Image.Image,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Genera imagen con ControlNet.

    Args:
        pipeline: Pipeline de SD + ControlNet
        prompt: Descripcion del resultado
        control_image: Imagen de control (canny, depth, etc.)
        negative_prompt: Que evitar
        num_inference_steps: Pasos de denoising
        guidance_scale: CFG scale
        controlnet_conditioning_scale: Fuerza del control
        seed: Semilla

    Returns:
        Imagen generada
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )

    return output.images[0]


# Ejemplo de uso
def demo_controlnet():
    """Demo de ControlNet con Canny edges."""

    # Cargar pipeline
    pipeline = load_controlnet_pipeline("canny")

    # Cargar y preprocesar imagen
    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_input.png"
    )

    canny_image = preprocess_canny(input_image)

    # Generar
    prompt = "A beautiful Victorian mansion, highly detailed, professional photo"
    negative_prompt = "blurry, bad quality"

    output = generate_with_controlnet(
        pipeline=pipeline,
        prompt=prompt,
        control_image=canny_image,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.8,
        seed=42,
    )

    output.save("controlnet_output.png")
    print("Imagen guardada en controlnet_output.png")

    return output


if __name__ == "__main__":
    demo_controlnet()
```

---

## LoRA: Fine-tuning Eficiente

```
LoRA: LOW-RANK ADAPTATION
=========================

LoRA permite fine-tunear Stable Diffusion de forma eficiente,
entrenando solo una pequena fraccion de parametros.


IDEA CLAVE:
───────────

En lugar de actualizar todos los pesos W, descomponemos
el cambio en matrices de bajo rango:

    W' = W + ΔW = W + B @ A

Donde:
    - W: Pesos originales (frozen)
    - A: Matriz de rango bajo [d, r] donde r << d
    - B: Matriz de rango bajo [r, d]
    - r: Rango (tipico: 4-64)


COMPARACION DE PARAMETROS:
──────────────────────────

Original (fine-tune completo):
    - Parametros entrenables: ~1B
    - VRAM: 24GB+
    - Tiempo: Dias

LoRA (r=4):
    - Parametros entrenables: ~2M (0.2%)
    - VRAM: 8-12GB
    - Tiempo: Horas


VISUALIZACION:
──────────────

                  Original Weight W
                  ┌───────────────┐
                  │               │
    Input ──────▶ │   [d × d]     │ ──────▶ Output
                  │   (frozen)    │
                  │               │
                  └───────┬───────┘
                          │
                          │ +
                          │
              ┌───────────▼───────────┐
              │        ΔW = B @ A     │
              │                       │
              │  ┌─────┐    ┌─────┐   │
              │  │  B  │ @  │  A  │   │
              │  │[d×r]│    │[r×d]│   │
              │  └─────┘    └─────┘   │
              │     (trainable)       │
              └───────────────────────┘

    Total params LoRA: d×r + r×d = 2×d×r
    vs full: d×d

    Si d=1024, r=4:
    LoRA: 8,192 params
    Full: 1,048,576 params
    Reduccion: 128x


DONDE APLICAR LoRA EN SD:
─────────────────────────

Se aplica a las capas de atencion:
    - to_q, to_k, to_v (cross-attention)
    - to_out (proyeccion de salida)

Capas objetivo tipicas:
    - unet.down_blocks.*.attentions.*.transformer_blocks.*.attn1.*
    - unet.down_blocks.*.attentions.*.transformer_blocks.*.attn2.*
    - unet.up_blocks.*.attentions.*.transformer_blocks.*.attn1.*
    - unet.up_blocks.*.attentions.*.transformer_blocks.*.attn2.*
```

### Implementacion y Uso de LoRA

```python
"""
LoRA para fine-tuning de Stable Diffusion.
"""

from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """
    Capa Linear con LoRA adapter.

    W' = W + scale * B @ A
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        """
        Args:
            original_layer: Capa Linear original
            rank: Rango de la descomposicion
            alpha: Factor de escala
        """
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Congelar capa original
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Inicializacion
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        # Original forward
        result = self.original_layer(x)

        # LoRA forward: x @ A^T @ B^T
        lora_out = x @ self.lora_A.T @ self.lora_B.T

        return result + self.scale * lora_out


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: float = 1.0,
) -> Dict[str, LoRALinear]:
    """
    Inyecta LoRA adapters en un modelo.

    Args:
        model: Modelo a modificar
        target_modules: Nombres de modulos objetivo
        rank: Rango de LoRA
        alpha: Factor de escala

    Returns:
        Dict de modulos LoRA inyectados
    """
    lora_layers = {}

    for name, module in model.named_modules():
        # Verificar si es un modulo objetivo
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Crear LoRA wrapper
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)

                # Reemplazar en el modelo
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, lora_layer)

                lora_layers[name] = lora_layer

    return lora_layers


def get_lora_params(lora_layers: Dict[str, LoRALinear]) -> List[nn.Parameter]:
    """Obtiene parametros entrenables de LoRA."""
    params = []
    for layer in lora_layers.values():
        params.extend([layer.lora_A, layer.lora_B])
    return params


def save_lora_weights(
    lora_layers: Dict[str, LoRALinear],
    path: str,
):
    """Guarda solo los pesos de LoRA."""
    state_dict = {}

    for name, layer in lora_layers.items():
        state_dict[f"{name}.lora_A"] = layer.lora_A.data
        state_dict[f"{name}.lora_B"] = layer.lora_B.data
        state_dict[f"{name}.scale"] = layer.scale

    torch.save(state_dict, path)


def load_lora_weights(
    lora_layers: Dict[str, LoRALinear],
    path: str,
):
    """Carga pesos de LoRA."""
    state_dict = torch.load(path)

    for name, layer in lora_layers.items():
        layer.lora_A.data = state_dict[f"{name}.lora_A"]
        layer.lora_B.data = state_dict[f"{name}.lora_B"]


# =============================================================================
# USO CON DIFFUSERS (forma recomendada)
# =============================================================================

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


def create_lora_sd_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_weights_path: Optional[str] = None,
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """
    Crea pipeline de SD con soporte para LoRA.

    Args:
        model_id: ID del modelo base
        lora_weights_path: Path a pesos LoRA pre-entrenados
        device: Dispositivo

    Returns:
        Pipeline configurado
    """
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)

    if lora_weights_path:
        # Cargar LoRA weights
        pipeline.unet.load_attn_procs(lora_weights_path)

    return pipeline


def train_lora_sd(
    pipeline: StableDiffusionPipeline,
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    rank: int = 4,
    output_dir: str = "lora_weights",
):
    """
    Entrena LoRA para Stable Diffusion.

    Args:
        pipeline: Pipeline de SD
        train_dataloader: DataLoader con pares (imagen, caption)
        num_epochs: Epocas de entrenamiento
        learning_rate: Learning rate
        rank: Rango de LoRA
        output_dir: Directorio para guardar weights
    """
    from diffusers import DDPMScheduler
    import os

    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # Configurar LoRA con diffusers
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    # Inyectar LoRA (usando diffusers built-in)
    from diffusers.models.attention_processor import LoRAAttnProcessor

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    # Obtener parametros trainables
    lora_params = []
    for attn_processor in unet.attn_processors.values():
        if hasattr(attn_processor, "parameters"):
            lora_params.extend(attn_processor.parameters())

    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)

    # Training loop
    device = pipeline.device

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            images, captions = batch

            # Encode images
            images = images.to(device, dtype=torch.float16)
            latents = vae.encode(images * 2 - 1).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode text
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]

            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Guardar LoRA weights
    os.makedirs(output_dir, exist_ok=True)
    unet.save_attn_procs(output_dir)
    print(f"LoRA weights guardados en {output_dir}")


# Ejemplo de uso
def demo_lora():
    """Demo de carga y uso de LoRA."""

    # Cargar pipeline base
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Cargar LoRA weights (ejemplo: estilo anime)
    # pipeline.unet.load_attn_procs("path/to/lora/weights")

    # Generar con el estilo LoRA
    image = pipeline(
        "A girl with blue hair, anime style",
        num_inference_steps=30,
    ).images[0]

    image.save("lora_output.png")

    return image
```

---

## Resumen

```
STABLE DIFFUSION - COMPONENTES CLAVE
====================================

1. VAE (Variational Autoencoder)
   - Comprime imagenes 8x espacialmente
   - 512x512x3 → 64x64x4
   - Encoder: imagen → latent
   - Decoder: latent → imagen

2. CLIP Text Encoder
   - Convierte texto a embeddings
   - 77 tokens × 768 dimensiones
   - Usado para cross-attention

3. U-Net Condicionada
   - Opera en espacio latente
   - Cross-attention con texto
   - Predice ruido ε dado (z_t, t, text)

4. Scheduler (DDPM/DDIM/etc.)
   - Controla proceso de denoising
   - DDIM: mas rapido, determinista
   - Euler: balance velocidad/calidad

5. Classifier-Free Guidance
   - Controla fuerza del condicionamiento
   - Scale tipico: 7.5
   - Duplica compute (cond + uncond)

6. ControlNet
   - Control adicional (pose, edges, depth)
   - Copia del encoder de U-Net
   - Zero convolutions

7. LoRA
   - Fine-tuning eficiente
   - ~0.2% de parametros
   - Estilos, conceptos, personajes


FLUJO DE GENERACION:
───────────────────

prompt ──▶ CLIP ──▶ text_emb ──┐
                               │
           ┌───────────────────┘
           │
           ▼
z_T ──▶ U-Net ──▶ ε ──▶ scheduler ──▶ z_0 ──▶ VAE dec ──▶ imagen
  │       ▲
  │       │
  └─ t ───┘


RECURSOS RECOMENDADOS:
─────────────────────

- diffusers library: https://github.com/huggingface/diffusers
- Civitai (modelos community): https://civitai.com
- Stable Diffusion WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui
```

---

## Referencias

1. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models"
2. Ho, J. & Salimans, T. (2022). "Classifier-Free Diffusion Guidance"
3. Zhang, L., et al. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
4. Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
5. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
