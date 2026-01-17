# Diffusion Models: Fundamentos y Aplicaciones

## Introduccion

Los **Diffusion Models** (Modelos de Difusion) son una familia de modelos generativos que aprenden a generar datos mediante un proceso de **denoising** (eliminacion de ruido). La idea central es simple pero poderosa: destruir gradualmente la estructura de los datos agregando ruido gaussiano, y luego aprender a revertir este proceso.

```
INTUICION DE DIFFUSION MODELS
=============================

     Datos Reales               Ruido Puro
         x₀                         xₜ

    ┌─────────┐                ┌─────────┐
    │  🖼️    │                │ ░░░░░░░ │
    │ Imagen  │  ──────────▶   │  Ruido  │
    │ clara   │   Forward      │ Gaussiano│
    └─────────┘   Process      └─────────┘
         │        (destruir)        │
         │                          │
         │                          │
         │    ◀──────────────       │
         │      Reverse             │
         │      Process             │
         │     (reconstruir)        │
         ▼                          ▼

    El modelo aprende a          Comenzamos desde
    REVERTIR el proceso          ruido aleatorio
    de destruccion               N(0, I)


PROCESO PASO A PASO:
====================

Forward (q):  x₀ → x₁ → x₂ → ... → xₜ₋₁ → xₜ  (agregar ruido)
              │    │    │           │      │
              ▼    ▼    ▼           ▼      ▼
             [imagen cada vez mas ruidosa hasta ruido puro]

Reverse (p): xₜ → xₜ₋₁ → ... → x₂ → x₁ → x₀   (quitar ruido)
              │    │            │    │    │
              ▼    ▼            ▼    ▼    ▼
             [ruido que gradualmente se convierte en imagen]

La clave: El modelo neuronal aprende p(xₜ₋₁|xₜ)
```

## Matematicas del Forward Process

### Cadena de Markov con Ruido Gaussiano

El forward process es una **cadena de Markov** donde cada paso agrega ruido gaussiano:

```
FORWARD PROCESS (q)
===================

Definicion formal:

    q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜ·I)

Donde:
    - βₜ ∈ (0, 1) es el "variance schedule" en el paso t
    - √(1-βₜ) escala la senal (la reduce ligeramente)
    - βₜ controla cuanto ruido se agrega

Reparametrizacion:

    xₜ = √(1-βₜ)·xₜ₋₁ + √βₜ·εₜ    donde εₜ ~ N(0, I)

Visualizacion del schedule tipico:

    βₜ
    │                                          ●
    │                                      ●
    │                                  ●
    │                             ●
    │                        ●
    │                   ●
    │              ●
    │         ●
    │    ●
    │●
    └──────────────────────────────────────────▶ t
    0                                           T

    β₁ ≈ 10⁻⁴  (poco ruido al inicio)
    βₜ ≈ 0.02  (mas ruido al final)
```

### Closed-Form para Cualquier Paso t

Una propiedad crucial es que podemos calcular `xₜ` directamente desde `x₀` sin iterar:

```
PROPIEDAD FUNDAMENTAL
=====================

Definimos:
    αₜ = 1 - βₜ
    ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ = α₁ · α₂ · ... · αₜ

Entonces:
    q(xₜ|x₀) = N(xₜ; √ᾱₜ·x₀, (1-ᾱₜ)·I)

Reparametrizacion:
    xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε    donde ε ~ N(0, I)


DEMOSTRACION (por induccion):
─────────────────────────────

Base: t=1
    x₁ = √α₁·x₀ + √(1-α₁)·ε₁

    Varianza: (1-α₁) = 1 - ᾱ₁  ✓
    Media: √α₁·x₀ = √ᾱ₁·x₀    ✓

Paso inductivo: Asumiendo que vale para t-1
    xₜ₋₁ = √ᾱₜ₋₁·x₀ + √(1-ᾱₜ₋₁)·ε'

    xₜ = √αₜ·xₜ₋₁ + √βₜ·εₜ
       = √αₜ·(√ᾱₜ₋₁·x₀ + √(1-ᾱₜ₋₁)·ε') + √βₜ·εₜ
       = √(αₜ·ᾱₜ₋₁)·x₀ + √αₜ·√(1-ᾱₜ₋₁)·ε' + √βₜ·εₜ
       = √ᾱₜ·x₀ + √(αₜ(1-ᾱₜ₋₁) + βₜ)·ε

    Donde usamos que la suma de gaussianas:
    Var = αₜ(1-ᾱₜ₋₁) + βₜ = αₜ - αₜᾱₜ₋₁ + 1 - αₜ
        = 1 - ᾱₜ  ✓


INTERPRETACION VISUAL:
─────────────────────

t=0          t=T/4         t=T/2         t=3T/4        t=T
 │            │             │             │            │
 ▼            ▼             ▼             ▼            ▼
┌────┐      ┌────┐        ┌────┐        ┌────┐      ┌────┐
│🖼️  │      │🖼️ ░│        │░░🖼️░│       │░░░░│      │░░░░│
│    │      │  ░ │        │ ░░░ │        │░░░░│      │░░░░│
└────┘      └────┘        └────┘        └────┘      └────┘

√ᾱₜ ≈ 1    √ᾱₜ ≈ 0.9     √ᾱₜ ≈ 0.5    √ᾱₜ ≈ 0.1   √ᾱₜ ≈ 0
(senal)    (algo ruido)  (50/50)     (casi ruido) (ruido)
```

---

## Reverse Process y Score Matching

### El Objetivo: Aprender a Denoiser

```
REVERSE PROCESS (p)
===================

Queremos aprender:
    pθ(xₜ₋₁|xₜ) ≈ q(xₜ₋₁|xₜ, x₀)

La distribucion posterior real q(xₜ₋₁|xₜ, x₀) es GAUSSIANA:

    q(xₜ₋₁|xₜ, x₀) = N(xₜ₋₁; μ̃ₜ(xₜ, x₀), β̃ₜ·I)

Donde:
    μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁·βₜ)/(1-ᾱₜ)·x₀ + (√αₜ·(1-ᾱₜ₋₁))/(1-ᾱₜ)·xₜ

    β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ)·βₜ


PROBLEMA: No conocemos x₀ durante la generacion!

SOLUCION: Predecir x₀ desde xₜ usando una red neuronal

    x̂₀ = f_θ(xₜ, t)   // red que predice x₀ desde xₜ


FORMULACION EQUIVALENTE (Score Matching):
─────────────────────────────────────────

En lugar de predecir x₀, podemos predecir el RUIDO ε:

    xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε

    Despejando x₀:
    x₀ = (xₜ - √(1-ᾱₜ)·ε) / √ᾱₜ

    Si predecimos ε̂ = εθ(xₜ, t):
    x̂₀ = (xₜ - √(1-ᾱₜ)·ε̂) / √ᾱₜ


CONEXION CON SCORE FUNCTION:
────────────────────────────

El "score" es el gradiente del log de la densidad:

    sθ(xₜ, t) = ∇ₓₜ log p(xₜ)

Para una gaussiana q(xₜ|x₀):

    ∇ₓₜ log q(xₜ|x₀) = -ε / √(1-ᾱₜ)

Por tanto:
    εθ(xₜ, t) = -√(1-ᾱₜ) · sθ(xₜ, t)

Predecir ruido ε ≡ Predecir score function (hasta escala)
```

---

## DDPM: Denoising Diffusion Probabilistic Models

### Funcion de Perdida

```
DDPM TRAINING OBJECTIVE
=======================

Simplified Loss (Ho et al., 2020):

    L_simple = Eₜ,x₀,ε[ ||ε - εθ(xₜ, t)||² ]

Donde:
    - t ~ Uniform(1, T)
    - x₀ ~ p_data  (imagen real)
    - ε ~ N(0, I)  (ruido aleatorio)
    - xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε


ALGORITMO DE ENTRENAMIENTO:
───────────────────────────

1. repeat
2.    x₀ ~ p_data(x)           // Muestrea imagen real
3.    t ~ Uniform({1,...,T})   // Muestrea paso de tiempo
4.    ε ~ N(0, I)              // Muestrea ruido
5.    xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε  // Forward diffusion
6.    L = ||ε - εθ(xₜ, t)||²   // Calcula loss
7.    θ ← θ - η·∇θL            // Actualiza parametros
8. until convergencia


ALGORITMO DE MUESTREO:
──────────────────────

1. xₜ ~ N(0, I)                    // Empieza con ruido puro
2. for t = T, T-1, ..., 1:
3.    z ~ N(0, I) if t > 1, else z = 0
4.    ε̂ = εθ(xₜ, t)               // Predice ruido
5.    xₜ₋₁ = (1/√αₜ)·(xₜ - (βₜ/√(1-ᾱₜ))·ε̂) + σₜ·z
6. return x₀


DIAGRAMA DEL SAMPLING:
─────────────────────

t=T        t=T-1       t=T-2              t=1        t=0
 │          │           │                  │          │
 ▼          ▼           ▼                  ▼          ▼
┌──┐  ε̂   ┌──┐  ε̂    ┌──┐              ┌──┐  ε̂   ┌──┐
│░░│ ───▶ │░░│ ───▶  │░▓│  ... ───▶    │▓▓│ ───▶ │🖼️ │
│░░│  -   │░▓│  -    │▓▓│              │▓█│  -   │  │
└──┘      └──┘       └──┘              └──┘      └──┘
puro      algo       mas               casi      imagen
ruido     menos      claro             claro     final
```

### Variance Schedules

```
TIPOS DE NOISE SCHEDULES
========================

1. LINEAR SCHEDULE (original DDPM):
───────────────────────────────────
   βₜ = β₁ + (t-1)/(T-1) · (βₜ - β₁)

   β₁ = 10⁻⁴, βₜ = 0.02, T = 1000

   βₜ │                              ●
      │                          ●
      │                      ●
      │                  ●
      │              ●
      │          ●
      │      ●
      │  ●
      │●
      └──────────────────────────────▶ t

2. COSINE SCHEDULE (Nichol & Dhariwal):
───────────────────────────────────────
   ᾱₜ = f(t)/f(0), donde f(t) = cos((t/T + s)/(1+s) · π/2)²

   Proporciona transicion mas suave

   ᾱₜ │●
      │ ●
      │  ●
      │   ●●
      │     ●●
      │       ●●●
      │          ●●●●
      │              ●●●●●●
      │                    ●●●●●●●●●●●
      └─────────────────────────────────▶ t

3. SIGMOID SCHEDULE:
────────────────────
   βₜ = sigmoid(a + (b-a)·t/T) · (β_max - β_min) + β_min

   Parametrizable, flexible


COMPARACION EN CALIDAD DE SAMPLES:
──────────────────────────────────

Schedule   │ FID (↓ mejor) │ Problemas
───────────┼───────────────┼───────────────────────────
Linear     │    ~5.5       │ Ruido excesivo al inicio
Cosine     │    ~4.0       │ Ninguno significativo
Sigmoid    │    ~4.2       │ Requiere tuning de params

El cosine schedule es el mas usado actualmente.
```

---

## Arquitectura U-Net para Diffusion

### Estructura General

```
ARQUITECTURA U-NET PARA DIFFUSION
=================================

La U-Net es la arquitectura estandar para εθ(xₜ, t).
Procesa xₜ y t para predecir el ruido ε.


                    Time Embedding
                         │
                    ┌────┴────┐
                    │ Sinusoid│
                    │ + MLP   │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  t_emb  │───────────────────────────┐
                    └─────────┘                           │
                                                          │
INPUT xₜ                                             OUTPUT ε̂
   │                                                      ▲
   ▼                                                      │
┌──────────────────────────────────────────────────────────┐
│                      U-NET                               │
│                                                          │
│  ┌─────┐   ┌─────┐   ┌─────┐       ┌─────┐   ┌─────┐   │
│  │Conv │──▶│Down │──▶│Down │──...──│ Up  │──▶│ Up  │──▶│
│  │ 3x3 │   │Block│   │Block│       │Block│   │Block│   │
│  └─────┘   └──┬──┘   └──┬──┘       └──▲──┘   └──▲──┘   │
│               │         │             │         │      │
│               │         │    ┌───┐    │         │      │
│               │         └───▶│Mid│───▶┘         │      │
│               │              │   │              │      │
│               └──────────────│───│──────────────┘      │
│                     Skip     └───┘    Skip             │
│                   Connections       Connections        │
└──────────────────────────────────────────────────────────┘


DETALLE DE LOS BLOQUES:
──────────────────────

Down Block:                      Up Block:
┌─────────────────┐              ┌─────────────────┐
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │ GroupNorm │   │              │ │ Upsample  │   │
│ └─────┬─────┘   │              │ └─────┬─────┘   │
│       ▼         │              │       ▼         │
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │  SiLU     │   │              │ │ Concat    │◀──skip
│ └─────┬─────┘   │              │ └─────┬─────┘   │
│       ▼         │              │       ▼         │
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │ Conv 3x3  │   │              │ │ GroupNorm │   │
│ └─────┬─────┘   │              │ └─────┬─────┘   │
│       ▼         │              │       ▼         │
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │ + t_emb   │◀──t_emb          │ │  SiLU     │   │
│ └─────┬─────┘   │              │ └─────┬─────┘   │
│       ▼         │              │       ▼         │
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │ Attention │   │              │ │ Conv 3x3  │   │
│ └─────┬─────┘   │              │ └─────┬─────┘   │
│       ▼         │              │       ▼         │
│ ┌───────────┐   │              │ ┌───────────┐   │
│ │ Downsample│   │              │ │ + t_emb   │◀──t_emb
│ └───────────┘   │              │ └───────────┘   │
└─────────────────┘              └─────────────────┘


TIME EMBEDDING:
──────────────

Posicional sinusoidal (como en Transformers):

    PE(t, 2i)   = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))

Luego pasa por MLP:

    t_emb = MLP(PE(t)) = Linear(SiLU(Linear(PE(t))))
```

---

## Implementacion Completa en PyTorch

```python
"""
DDPM: Denoising Diffusion Probabilistic Models
Implementacion completa con PyTorch.

Author: AI-Path Course
"""

import math
from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass
class DDPMConfig:
    """Configuracion del modelo DDPM."""

    # Parametros de difusion
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_type: str = "cosine"  # "linear", "cosine", "sigmoid"

    # Arquitectura U-Net
    image_size: int = 64
    in_channels: int = 3
    model_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    dropout: float = 0.1

    # Entrenamiento
    learning_rate: float = 2e-4
    batch_size: int = 32
    num_epochs: int = 100


# =============================================================================
# NOISE SCHEDULES
# =============================================================================

def get_beta_schedule(
    schedule_type: str,
    num_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02
) -> Tensor:
    """
    Genera el schedule de varianzas beta.

    Args:
        schedule_type: Tipo de schedule ("linear", "cosine", "sigmoid")
        num_timesteps: Numero total de pasos T
        beta_start: Beta inicial
        beta_end: Beta final

    Returns:
        Tensor de shape (T,) con las betas
    """
    if schedule_type == "linear":
        # Schedule lineal original de DDPM
        betas = torch.linspace(beta_start, beta_end, num_timesteps)

    elif schedule_type == "cosine":
        # Schedule coseno (Improved DDPM - Nichol & Dhariwal)
        s = 0.008  # Offset para evitar singularidades
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)

        # f(t) = cos^2((t/T + s) / (1+s) * pi/2)
        f_t = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]

        # Calcular betas desde alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, min=0.0001, max=0.999)

    elif schedule_type == "sigmoid":
        # Schedule sigmoide
        t = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(t) * (beta_end - beta_start) + beta_start

    else:
        raise ValueError(f"Schedule desconocido: {schedule_type}")

    return betas


class DiffusionSchedule:
    """
    Precomputa todos los coeficientes necesarios para diffusion.

    Atributos precomputados:
        - betas: βₜ
        - alphas: αₜ = 1 - βₜ
        - alphas_cumprod: ᾱₜ = ∏αᵢ
        - sqrt_alphas_cumprod: √ᾱₜ
        - sqrt_one_minus_alphas_cumprod: √(1-ᾱₜ)
        - posterior_variance: β̃ₜ
    """

    def __init__(
        self,
        schedule_type: str = "cosine",
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.num_timesteps = num_timesteps
        self.device = device

        # Obtener betas
        betas = get_beta_schedule(schedule_type, num_timesteps, beta_start, beta_end)

        # Precomputar coeficientes
        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        # Para forward process: q(xₜ|x₀)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Para reverse process: q(xₜ₋₁|xₜ, x₀)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance: β̃ₜ = (1 - ᾱₜ₋₁)/(1 - ᾱₜ) * βₜ
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

        # Coeficientes para calcular mean posterior
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(device)

        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod)
        ).to(device)

    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward process: muestrea xₜ dado x₀.

        xₜ = √ᾱₜ * x₀ + √(1-ᾱₜ) * ε

        Args:
            x_0: Datos originales [B, C, H, W]
            t: Timesteps [B]
            noise: Ruido opcional (si None, se genera)

        Returns:
            x_t: Datos con ruido [B, C, H, W]
            noise: Ruido usado [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extraer coeficientes para cada muestra del batch
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def q_posterior_mean(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor
    ) -> Tensor:
        """
        Calcula la media del posterior q(xₜ₋₁|xₜ, x₀).

        μ̃ₜ = (√ᾱₜ₋₁ * βₜ)/(1-ᾱₜ) * x₀ + (√αₜ * (1-ᾱₜ₋₁))/(1-ᾱₜ) * xₜ
        """
        coef1 = self.posterior_mean_coef1[t][:, None, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None, None]

        return coef1 * x_0 + coef2 * x_t


# =============================================================================
# TIME EMBEDDING
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Embedding posicional sinusoidal para el timestep t.
    Similar al usado en Transformers.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timesteps [B]

        Returns:
            Embeddings [B, dim]
        """
        half_dim = self.dim // 2

        # Frecuencias: 1 / 10000^(2i/d)
        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=t.device) / half_dim
        )

        # Aplicar frecuencias: t * freq
        args = t[:, None].float() * freqs[None, :]

        # Concatenar sin y cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class TimeEmbedding(nn.Module):
    """
    Embedding de tiempo completo: sinusoidal + MLP.
    """

    def __init__(self, model_channels: int, time_embed_dim: int):
        super().__init__()

        self.sinusoidal = SinusoidalPositionalEmbedding(model_channels)
        self.mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timesteps [B]

        Returns:
            Time embeddings [B, time_embed_dim]
        """
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb


# =============================================================================
# BLOQUES DE LA U-NET
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Bloque residual con condicionamiento temporal.

    Architecture:
        x -> GroupNorm -> SiLU -> Conv -> (+t_emb) -> GroupNorm -> SiLU -> Dropout -> Conv -> (+x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Proyeccion del time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection (si cambian canales)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x: Features [B, C, H, W]
            t_emb: Time embedding [B, time_embed_dim]

        Returns:
            Output features [B, out_channels, H, W]
        """
        h = x

        # Primera conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        # Agregar time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        # Segunda conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block con GroupNorm.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Features [B, C, H, W]

        Returns:
            Output [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Reshape para attention: [B, H*W, C]
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)

        # Self-attention
        h, _ = self.attention(h, h, h)

        # Reshape back
        h = h.transpose(1, 2).view(B, C, H, W)

        return x + h


class Downsample(nn.Module):
    """Downsampling 2x con strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling 2x con interpolacion + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# =============================================================================
# U-NET COMPLETA
# =============================================================================

class UNet(nn.Module):
    """
    U-Net para prediccion de ruido en diffusion models.

    Args:
        config: Configuracion del modelo
    """

    def __init__(self, config: DDPMConfig):
        super().__init__()

        self.config = config
        ch = config.model_channels
        time_embed_dim = ch * 4

        # Time embedding
        self.time_embed = TimeEmbedding(ch, time_embed_dim)

        # Input convolution
        self.input_conv = nn.Conv2d(
            config.in_channels, ch, kernel_size=3, padding=1
        )

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_ch = ch
        current_res = config.image_size
        encoder_channels = [ch]

        for level, mult in enumerate(config.channel_mult):
            out_ch = ch * mult

            # ResBlocks
            for _ in range(config.num_res_blocks):
                block = ResidualBlock(in_ch, out_ch, time_embed_dim, config.dropout)
                self.down_blocks.append(block)

                # Attention si esta en la resolucion correcta
                if current_res in config.attention_resolutions:
                    self.down_blocks.append(AttentionBlock(out_ch))

                in_ch = out_ch
                encoder_channels.append(in_ch)

            # Downsample (excepto ultima capa)
            if level < len(config.channel_mult) - 1:
                self.down_samples.append(Downsample(in_ch))
                current_res //= 2
                encoder_channels.append(in_ch)

        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, time_embed_dim, config.dropout),
            AttentionBlock(in_ch),
            ResidualBlock(in_ch, in_ch, time_embed_dim, config.dropout),
        ])

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in enumerate(reversed(config.channel_mult)):
            out_ch = ch * mult

            for i in range(config.num_res_blocks + 1):
                # Concatenar con skip connection
                skip_ch = encoder_channels.pop()
                block = ResidualBlock(
                    in_ch + skip_ch, out_ch, time_embed_dim, config.dropout
                )
                self.up_blocks.append(block)

                if current_res in config.attention_resolutions:
                    self.up_blocks.append(AttentionBlock(out_ch))

                in_ch = out_ch

            # Upsample (excepto ultima capa)
            if level < len(config.channel_mult) - 1:
                self.up_samples.append(Upsample(in_ch))
                current_res *= 2

        # Output
        self.output_norm = nn.GroupNorm(32, in_ch)
        self.output_conv = nn.Conv2d(in_ch, config.in_channels, kernel_size=3, padding=1)

        # Inicializacion
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: Imagen con ruido [B, C, H, W]
            t: Timesteps [B]

        Returns:
            Ruido predicho [B, C, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Input
        h = self.input_conv(x)

        # Encoder con skip connections
        skips = [h]
        down_idx = 0

        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
            skips.append(h)

            # Check if we need to downsample
            if down_idx < len(self.down_samples):
                if len(skips) % (self.config.num_res_blocks + 1) == 0:
                    h = self.down_samples[down_idx](h)
                    skips.append(h)
                    down_idx += 1

        # Middle
        for block in self.middle_block:
            if isinstance(block, ResidualBlock):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder con skip connections
        up_idx = 0

        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            else:
                h = block(h)

            # Check if we need to upsample
            if up_idx < len(self.up_samples):
                h = self.up_samples[up_idx](h)
                up_idx += 1

        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h


# =============================================================================
# DDPM COMPLETO
# =============================================================================

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model completo.

    Combina:
        - Schedule de difusion
        - U-Net para prediccion de ruido
        - Metodos de entrenamiento y sampling
    """

    def __init__(self, config: DDPMConfig, device: str = "cuda"):
        super().__init__()

        self.config = config
        self.device = device

        # Schedule
        self.schedule = DiffusionSchedule(
            schedule_type=config.schedule_type,
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=device
        )

        # Modelo (U-Net)
        self.model = UNet(config).to(device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass: predice ruido."""
        return self.model(x, t)

    def training_loss(self, x_0: Tensor) -> Tensor:
        """
        Calcula la loss de entrenamiento.

        L = E[||ε - εθ(xₜ, t)||²]

        Args:
            x_0: Batch de imagenes limpias [B, C, H, W]

        Returns:
            Loss escalar
        """
        batch_size = x_0.shape[0]

        # Muestrear timesteps aleatorios
        t = torch.randint(
            0, self.config.num_timesteps,
            (batch_size,),
            device=self.device
        )

        # Forward process: agregar ruido
        x_t, noise = self.schedule.q_sample(x_0, t)

        # Predecir ruido
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        return_intermediates: bool = False
    ) -> Tensor:
        """
        Genera muestras usando DDPM sampling.

        Args:
            batch_size: Numero de muestras a generar
            return_intermediates: Si True, retorna pasos intermedios

        Returns:
            Muestras generadas [B, C, H, W]
        """
        shape = (
            batch_size,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size
        )

        # Empezar con ruido puro
        x = torch.randn(shape, device=self.device)

        intermediates = [x] if return_intermediates else None

        # Reverse process
        for t in tqdm(
            reversed(range(self.config.num_timesteps)),
            desc="Sampling",
            total=self.config.num_timesteps
        ):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predecir ruido
            noise_pred = self.model(x, t_batch)

            # Calcular x_{t-1}
            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alphas_cumprod[t]
            beta = self.schedule.betas[t]

            # Mean del reverse step
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
            )

            # Agregar ruido (excepto en t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.schedule.posterior_variance[t])
                x = mean + sigma * noise
            else:
                x = mean

            if return_intermediates and t % 100 == 0:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int = 1,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> Tensor:
        """
        DDIM sampling (mas rapido que DDPM).

        Args:
            batch_size: Numero de muestras
            num_steps: Pasos de muestreo (<<T)
            eta: Ruido estocastico (0=determinista)

        Returns:
            Muestras [B, C, H, W]
        """
        shape = (
            batch_size,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size
        )

        # Crear subsequencia de timesteps
        step_size = self.config.num_timesteps // num_steps
        timesteps = list(range(0, self.config.num_timesteps, step_size))

        # Empezar con ruido
        x = torch.randn(shape, device=self.device)

        for i in tqdm(
            reversed(range(len(timesteps))),
            desc="DDIM Sampling",
            total=len(timesteps)
        ):
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predecir ruido
            noise_pred = self.model(x, t_batch)

            # Coeficientes
            alpha_bar_t = self.schedule.alphas_cumprod[t]
            alpha_bar_prev = (
                self.schedule.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else torch.tensor(1.0)
            )

            # Predecir x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            # Direccion hacia x_t
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # DDIM step
            x = (
                torch.sqrt(alpha_bar_prev) * x_0_pred +
                torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
            )

            if eta > 0 and i > 0:
                x = x + sigma * torch.randn_like(x)

        return x


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_ddpm(
    model: DDPM,
    dataloader: torch.utils.data.DataLoader,
    config: DDPMConfig,
    save_path: str = "ddpm_model.pt"
) -> List[float]:
    """
    Entrena un modelo DDPM.

    Args:
        model: Modelo DDPM
        dataloader: DataLoader con imagenes
        config: Configuracion
        save_path: Donde guardar el modelo

    Returns:
        Lista de losses por epoca
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(dataloader)
    )

    losses = []

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in pbar:
            # Obtener imagenes (asumimos que batch es solo imagenes o (imagenes, labels))
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(model.device)

            # Normalizar a [-1, 1] si no lo esta
            if x.max() > 1:
                x = x / 255.0 * 2 - 1

            # Forward y loss
            optimizer.zero_grad()
            loss = model.training_loss(x)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Guardar checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, save_path)

    return losses


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Configuracion para MNIST/CIFAR (ejemplo reducido)
    config = DDPMConfig(
        num_timesteps=1000,
        schedule_type="cosine",
        image_size=32,
        in_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(8,),
        dropout=0.1,
        learning_rate=2e-4,
        batch_size=64,
        num_epochs=100
    )

    # Crear modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DDPM(config, device=device)

    print(f"Modelo creado en {device}")
    print(f"Parametros: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (4,), device=device)

    noise_pred = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {noise_pred.shape}")

    # Test loss
    loss = model.training_loss(x)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling (solo unos pasos para demo)
    print("\nGenerando muestra de prueba...")
    model.eval()

    # DDIM es mas rapido para test
    sample = model.ddim_sample(batch_size=1, num_steps=10)
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
```

---

## Uso con la Libreria Diffusers

```python
"""
Uso de Diffusion Models con la libreria diffusers de HuggingFace.
Mucho mas simple para experimentacion rapida.
"""

from typing import Optional, List
import torch
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UNet2DModel,
    DDPMPipeline,
)
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_ddpm_with_diffusers(
    image_size: int = 64,
    in_channels: int = 3,
    device: str = "cuda"
) -> tuple:
    """
    Crea un DDPM usando la libreria diffusers.

    Returns:
        (model, noise_scheduler)
    """
    # Configurar el scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",  # o "linear", "squaredcos_cap_v2"
        variance_type="fixed_small",
        prediction_type="epsilon",  # Predecir ruido
        clip_sample=True,
    )

    # Configurar el modelo U-Net
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    return model, noise_scheduler


def train_with_diffusers(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    num_epochs: int = 100,
    device: str = "cuda",
    use_ema: bool = True,
) -> List[float]:
    """
    Entrena un DDPM usando diffusers.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # EMA para mejor calidad de samples
    ema_model = None
    if use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.9999)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            # Normalizar a [-1, 1]
            images = images * 2 - 1

            batch_size = images.shape[0]

            # Muestrear ruido y timesteps
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()

            # Agregar ruido (forward process)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Predecir ruido
            noise_pred = model(noisy_images, timesteps).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema_model is not None:
                ema_model.step(model.parameters())

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return losses


@torch.no_grad()
def sample_with_diffusers(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    batch_size: int = 4,
    image_size: int = 64,
    in_channels: int = 3,
    device: str = "cuda",
    use_ddim: bool = False,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    Genera muestras usando diffusers.
    """
    model.eval()

    # Usar DDIM para sampling mas rapido
    if use_ddim:
        scheduler = DDIMScheduler.from_config(scheduler.config)
        scheduler.set_timesteps(ddim_steps)
    else:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    # Empezar con ruido
    images = torch.randn(
        batch_size, in_channels, image_size, image_size,
        device=device
    )

    # Reverse process
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        # Predecir ruido
        noise_pred = model(images, t).sample

        # DDPM/DDIM step
        images = scheduler.step(noise_pred, t, images).prev_sample

    # Denormalizar a [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    return images


def visualize_samples(samples: torch.Tensor, save_path: Optional[str] = None):
    """Visualiza muestras generadas."""
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, samples)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Sample {i+1}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# =============================================================================
# EJEMPLO COMPLETO CON CIFAR-10
# =============================================================================

def train_on_cifar10():
    """Ejemplo completo de entrenamiento en CIFAR-10."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Crear modelo
    model, scheduler = create_ddpm_with_diffusers(
        image_size=32,
        in_channels=3,
        device=device
    )

    print(f"Parametros del modelo: {sum(p.numel() for p in model.parameters()):,}")

    # Entrenar (reducido para demo)
    losses = train_with_diffusers(
        model=model,
        scheduler=scheduler,
        dataloader=dataloader,
        num_epochs=10,  # Aumentar para mejor calidad
        device=device
    )

    # Generar muestras
    samples = sample_with_diffusers(
        model=model,
        scheduler=scheduler,
        batch_size=8,
        image_size=32,
        in_channels=3,
        device=device,
        use_ddim=True,
        ddim_steps=50
    )

    visualize_samples(samples, "cifar10_samples.png")

    return model, scheduler


if __name__ == "__main__":
    train_on_cifar10()
```

---

## Score Matching y Score-Based Generative Models

```
SCORE MATCHING: PERSPECTIVA ALTERNATIVA
=======================================

En lugar de pensar en "agregar/quitar ruido",
podemos pensar en aprender el GRADIENTE de log p(x).

Score Function:
    s(x) = ∇ₓ log p(x)

Esta funcion apunta hacia regiones de alta densidad:

                    Distribucion p(x)
                         ╱╲
                        ╱  ╲
         s(x)←───      ╱    ╲      ───→s(x)
                      ╱      ╲
                     ╱        ╲
            ────────────────────────
                      x

Los scores apuntan hacia los modos de la distribucion.


SCORE MATCHING OBJECTIVE:
─────────────────────────

Queremos aprender sθ(x) ≈ ∇ₓ log p(x)

Loss original (impracticable, requiere ∇log p):
    L = E_p[ ||sθ(x) - ∇ₓ log p(x)||² ]

Denoising Score Matching (practicable):
    L_DSM = E_x~p E_ε~N[ ||sθ(x + σε) + ε/σ||² ]

Para ruido σ pequeno, sθ(x) ≈ ∇ log p(x)


CONEXION CON DIFFUSION:
───────────────────────

En diffusion, el score en el paso t es:

    sθ(xₜ, t) = ∇ₓₜ log p(xₜ)
              = -εθ(xₜ, t) / √(1 - ᾱₜ)

Por tanto:
    - Entrenar εθ para predecir ruido
    - Es EQUIVALENTE a score matching
    - Con diferentes niveles de ruido σₜ = √(1 - ᾱₜ)


LANGEVIN DYNAMICS SAMPLING:
───────────────────────────

Con el score, podemos muestrear via Langevin dynamics:

    x_{i+1} = xᵢ + (ε/2) · sθ(xᵢ) + √ε · z

Donde z ~ N(0, I) y ε es el step size.

Converge a muestras de p(x) cuando:
    - ε → 0
    - Numero de pasos → ∞

Diagrama del proceso:

x₀ (ruido)    x₁          x₂          ...    x_final
    │          │           │                    │
    │ +ε·s(x)  │ +ε·s(x)   │                    │
    │ +ruido   │ +ruido    │                    ▼
    └──────────┴───────────┴───────────────────(imagen)
```

---

## Comparacion: DDPM vs Score-Based vs Flow

```
COMPARACION DE PARADIGMAS GENERATIVOS
=====================================

                    DDPM/DDIM           Score-Based         Normalizing Flow
                    ─────────           ───────────         ────────────────
Representacion      Cadena Markov       Score function      Transformacion
                    discreta            continua            invertible

Forward             q(xₜ|xₜ₋₁)          Perturbacion        f: z → x
                    (agregar ruido)     continua            (determinista)

Reverse             pθ(xₜ₋₁|xₜ)         Langevin +          f⁻¹: x → z
                    (neural net)        score               (inversa exacta)

Training            ||ε - εθ||²         Score matching      -log p(x)
                                        + denoising         (cambio de variable)

Sampling            T pasos             Langevin            1 paso
                    (1000 tipico)       dynamics            (muy rapido)

Likelihood          Bound (ELBO)        Aproximada          Exacta
                                                            (pero costosa)

Calidad             Excelente           Excelente           Buena
                    (SOTA en imagenes)                      (mejorando)

Velocidad           Lento               Lento               Muy rapido
                    (DDIM: rapido)


DIAGRAMA UNIFICADO:
──────────────────

                        Datos x₀
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Diffusion│    │  Score  │    │  Flow   │
      │ (DDPM)  │    │Matching │    │(NF/FM)  │
      └────┬────┘    └────┬────┘    └────┬────┘
           │               │               │
           │ T pasos       │ SDE          │ 1 paso
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │  Ruido  │    │ Ruido   │    │   z ~   │
      │ N(0,I)  │    │ N(0,I)  │    │ N(0,I)  │
      └─────────┘    └─────────┘    └─────────┘

Todos empiezan de ruido gaussiano y llegan a datos.
La diferencia esta en COMO hacen la transformacion.
```

---

## Aplicaciones Practicas

```
APLICACIONES DE DIFFUSION MODELS
================================

1. GENERACION DE IMAGENES
─────────────────────────
   - Generacion incondicional (generar imagenes aleatorias)
   - Generacion condicional (con clase, texto, imagen de referencia)
   - Super-resolucion
   - Colorization
   - Outpainting / Inpainting

2. GENERACION DE AUDIO
──────────────────────
   - Text-to-speech (TTS)
   - Music generation
   - Audio super-resolution
   - Voice cloning

3. GENERACION DE VIDEO
──────────────────────
   - Text-to-video
   - Video interpolation
   - Video prediction

4. GENERACION 3D
────────────────
   - Point clouds
   - NeRFs
   - Meshes

5. CIENCIA
──────────
   - Protein structure prediction (AlphaFold-inspired)
   - Molecular generation
   - Weather prediction

6. RED TEAM / SEGURIDAD (contexto del curso)
────────────────────────────────────────────
   ⚠️  Deepfakes: Generacion de imagenes falsas
   ⚠️  Evasion de detectores: Generar variantes
   ⚠️  Data augmentation adversarial

   Defensas:
   - Watermarking de contenido generado
   - Detectores de imagenes sinteticas
   - Provenance tracking


METRICAS DE EVALUACION
======================

┌────────────────┬─────────────────────────────────────────┐
│ Metrica        │ Que mide                                │
├────────────────┼─────────────────────────────────────────┤
│ FID (↓)        │ Distancia entre distribuciones reales  │
│                │ y generadas (features de Inception)    │
├────────────────┼─────────────────────────────────────────┤
│ IS (↑)         │ Calidad + diversidad de imagenes       │
│                │ (Inception Score)                      │
├────────────────┼─────────────────────────────────────────┤
│ CLIP Score (↑) │ Alineacion imagen-texto                │
├────────────────┼─────────────────────────────────────────┤
│ LPIPS (↓)      │ Similitud perceptual                   │
├────────────────┼─────────────────────────────────────────┤
│ Precision      │ % de samples que parecen reales        │
├────────────────┼─────────────────────────────────────────┤
│ Recall         │ % de datos reales cubiertos            │
└────────────────┴─────────────────────────────────────────┘
```

---

## Resumen

```
CONCEPTOS CLAVE - DIFFUSION MODELS
==================================

1. PROCESO FORWARD
   - Destruir datos agregando ruido gaussiano
   - xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
   - T pasos (tipicamente 1000)

2. PROCESO REVERSE
   - Aprender a revertir la destruccion
   - Red neuronal predice el ruido εθ(xₜ, t)
   - Muestrear de ruido a datos

3. ARQUITECTURA U-NET
   - Encoder-decoder con skip connections
   - Condicionamiento temporal via embeddings
   - Attention en resoluciones bajas

4. ENTRENAMIENTO
   - Loss simple: ||ε - εθ(xₜ, t)||²
   - Muestrear t uniforme, agregar ruido, predecir

5. SAMPLING
   - DDPM: T pasos (lento, alta calidad)
   - DDIM: <<T pasos (rapido, determinista)

6. SCORE MATCHING
   - Perspectiva equivalente
   - sθ(x) ≈ ∇ log p(x)
   - Predecir ruido = predecir score

NEXT STEPS:
───────────
- Stable Diffusion (latent diffusion + texto)
- ControlNet (control fino)
- LoRA (fine-tuning eficiente)
```

---

## Referencias

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models"
2. Song, Y., et al. (2021). "Score-Based Generative Modeling through SDEs"
3. Nichol, A., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models"
4. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models" (DDIM)
5. Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models"
