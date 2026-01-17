# Normalizing Flows y Flow Matching

## Introduccion

Los **Normalizing Flows** son modelos generativos que transforman una distribucion simple (gaussiana) en una distribucion compleja mediante una secuencia de transformaciones **invertibles**. A diferencia de VAEs o GANs, los flows permiten calcular la **verosimilitud exacta** de los datos.

```
NORMALIZING FLOWS: IDEA CENTRAL
===============================

Objetivo: Transformar z ~ N(0, I) en x ~ p_data(x)

    z ~ N(0, I)                              x ~ p_data(x)
    ┌─────────────┐                          ┌─────────────┐
    │             │                          │             │
    │   Gaussiana │  ───────────────────────▶│   Compleja  │
    │    simple   │     f: z → x             │    (datos)  │
    │             │  ◀───────────────────────│             │
    │             │     f⁻¹: x → z           │             │
    └─────────────┘                          └─────────────┘


PROPIEDAD CLAVE: f es INVERTIBLE
────────────────────────────────

    f(z) = x        (generacion)
    f⁻¹(x) = z      (inference)

    Ambas direcciones son computables!


COMPARACION CON OTROS MODELOS:
──────────────────────────────

              │ Generacion │ Likelihood │ Invertible │
    ──────────┼────────────┼────────────┼────────────┤
    GAN       │     ✓      │     ✗      │     ✗      │
    VAE       │     ✓      │    bound   │     ✗      │
    Flow      │     ✓      │   exacta   │     ✓      │
    Diffusion │     ✓      │   approx   │     ~      │


FLUJO DE TRANSFORMACIONES:
──────────────────────────

    z₀ ───▶ f₁ ───▶ z₁ ───▶ f₂ ───▶ ... ───▶ fₖ ───▶ x

    z ~ N(0,I)                                p_data(x)

    La composicion de transformaciones invertibles
    es tambien invertible:

    f = fₖ ∘ fₖ₋₁ ∘ ... ∘ f₂ ∘ f₁
    f⁻¹ = f₁⁻¹ ∘ f₂⁻¹ ∘ ... ∘ fₖ⁻¹
```

---

## Matematicas: Cambio de Variable

### Formula de Cambio de Densidad

```
CAMBIO DE VARIABLE EN PROBABILIDAD
==================================

Si x = f(z) donde f es invertible, entonces:

    p(x) = p(z) · |det(∂z/∂x)|
         = p(z) · |det(J_{f⁻¹}(x))|
         = p(z) / |det(J_f(z))|

Donde J_f es la matriz Jacobiana:

         ┌                              ┐
         │ ∂f₁/∂z₁  ∂f₁/∂z₂  ...  ∂f₁/∂zₙ │
    J_f = │ ∂f₂/∂z₁  ∂f₂/∂z₂  ...  ∂f₂/∂zₙ │
         │   ...      ...    ...    ...   │
         │ ∂fₙ/∂z₁  ∂fₙ/∂z₂  ...  ∂fₙ/∂zₙ │
         └                              ┘


INTERPRETACION GEOMETRICA:
──────────────────────────

El determinante del Jacobiano mide como cambia el VOLUMEN:

    z-space                    x-space
    ┌───────┐                  ╭─────────╮
    │       │                 ╱           ╲
    │   1   │   ────f────▶   │     ?     │
    │       │               ╲           ╱
    └───────┘                  ╰─────────╯
    Volumen=1                  Volumen=|det(J)|

Si |det(J)| > 1: expansion
Si |det(J)| < 1: contraccion


LOG-LIKELIHOOD:
───────────────

Para una cadena de K transformaciones:

    log p(x) = log p(z₀) + Σₖ log |det(J_fₖ)|

Donde z₀ = f⁻¹(x) = f₁⁻¹(f₂⁻¹(...fₖ⁻¹(x)))

Para entrenar, maximizamos log p(x) sobre datos:

    max_θ  E_{x~data}[log p_θ(x)]


REQUISITOS PARA f:
──────────────────

1. INVERTIBLE: poder calcular f⁻¹
2. EFICIENTE: det(J) debe ser facil de calcular
3. EXPRESIVO: poder modelar distribuciones complejas

El desafio: Calcular det(J) para matrices D×D es O(D³)!
Solucion: Disenar f con Jacobiano TRIANGULAR → det = O(D)
```

---

## Arquitecturas de Flujos

### Coupling Layers (RealNVP)

```
AFFINE COUPLING LAYER
=====================

Idea: Dividir dimensiones en dos grupos y transformar
solo uno de ellos, condicionado en el otro.

    Input z = [z_a, z_b]  (mitades)

    Output x:
        x_a = z_a                          (sin cambio)
        x_b = z_b ⊙ exp(s(z_a)) + t(z_a)   (transformacion afin)

    Donde s() y t() son redes neuronales arbitrarias.


VISUALIZACION:
──────────────

    z = [z_a | z_b]
         │      │
         │      ▼
         │   ┌──────┐
         │   │ s(·) │ ◀──── s = scale (redes)
         │   │ t(·) │ ◀──── t = translation
         │   └──────┘
         │      │
         │      ▼
         ▼   z_b * exp(s) + t
    x = [x_a | x_b]


POR QUE ES INVERTIBLE?
──────────────────────

    Forward:
        x_a = z_a
        x_b = z_b * exp(s(z_a)) + t(z_a)

    Inverse:
        z_a = x_a
        z_b = (x_b - t(x_a)) / exp(s(x_a))
             = (x_b - t(x_a)) * exp(-s(x_a))

    No necesitamos invertir s() o t()!


JACOBIANO TRIANGULAR:
─────────────────────

    J = ┌                       ┐
        │   I_d        0        │
        │ ∂x_b/∂z_a  diag(exp(s))│
        └                       ┘

    det(J) = ∏ exp(sᵢ) = exp(Σ sᵢ)

    log |det(J)| = Σᵢ sᵢ(z_a)   ← O(D), muy eficiente!


ALTERNANDO PARTICIONES:
───────────────────────

Para que todas las dimensiones se transformen:

    Capa 1: [z_a | z_b] → [x_a | x_b]   (transforma b)
    Capa 2: [x_a | x_b] → [y_b | y_a]   (transforma a, invertido)
    Capa 3: [y_b | y_a] → [w_a | w_b]   (transforma b)
    ...

    Diferentes estrategias de particion:
    - Mitades: [1:D/2], [D/2+1:D]
    - Checkerboard (para imagenes)
    - Channel-wise (para imagenes)
```

### Implementacion RealNVP

```python
"""
RealNVP: Normalizing Flow con Affine Coupling.
Implementacion en PyTorch.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
import numpy as np


class CouplingNetwork(nn.Module):
    """
    Red neuronal para s(·) y t(·) en coupling layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        output_dim = output_dim or input_dim

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(h_dim),
            ])
            in_dim = h_dim

        # Output: s y t (doble dimension)
        layers.append(nn.Linear(in_dim, output_dim * 2))

        self.network = nn.Sequential(*layers)

        # Inicializar ultima capa a cero para estabilidad
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input [B, input_dim]

        Returns:
            s: Scale [B, output_dim]
            t: Translation [B, output_dim]
        """
        out = self.network(x)
        s, t = out.chunk(2, dim=-1)

        # Clamp s para estabilidad numerica
        s = torch.clamp(s, min=-5, max=5)

        return s, t


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer de RealNVP.

    x_a = z_a
    x_b = z_b * exp(s(z_a)) + t(z_a)
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        mask_type: str = "alternate",  # "alternate" o "half"
        mask_idx: int = 0,
    ):
        super().__init__()

        self.dim = dim

        # Crear mascara
        if mask_type == "alternate":
            mask = torch.zeros(dim)
            mask[mask_idx::2] = 1  # 0,1,0,1... o 1,0,1,0...
        else:  # half
            mask = torch.zeros(dim)
            if mask_idx == 0:
                mask[:dim//2] = 1
            else:
                mask[dim//2:] = 1

        self.register_buffer("mask", mask)

        # Red para s y t
        masked_dim = int(mask.sum().item())
        unmasked_dim = dim - masked_dim

        self.net = CouplingNetwork(masked_dim, hidden_dims, unmasked_dim)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward: z → x

        Returns:
            x: Output
            log_det: Log determinante del Jacobiano
        """
        z_a = z * self.mask
        z_b = z * (1 - self.mask)

        # Solo usar dimensiones enmascaradas como input
        z_a_input = z[:, self.mask.bool()]

        s, t = self.net(z_a_input)

        # Aplicar transformacion a dimensiones no enmascaradas
        # x_b = z_b * exp(s) + t
        exp_s = torch.exp(s)

        x_b = torch.zeros_like(z)
        x_b[:, (~self.mask.bool())] = z_b[:, (~self.mask.bool())] * exp_s + t

        x = z_a + x_b

        # Log determinante: sum(s)
        log_det = s.sum(dim=-1)

        return x, log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse: x → z
        """
        x_a = x * self.mask
        x_b = x * (1 - self.mask)

        x_a_input = x[:, self.mask.bool()]

        s, t = self.net(x_a_input)
        exp_s = torch.exp(s)

        # z_b = (x_b - t) / exp(s)
        z_b = torch.zeros_like(x)
        z_b[:, (~self.mask.bool())] = (x_b[:, (~self.mask.bool())] - t) / exp_s

        z = x_a + z_b

        log_det = -s.sum(dim=-1)

        return z, log_det


class ActNorm(nn.Module):
    """
    Activation Normalization (de Glow).
    Normaliza por canal con parametros aprendidos.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def initialize(self, x: Tensor):
        """Data-dependent initialization."""
        with torch.no_grad():
            mean = x.mean(dim=0)
            std = x.std(dim=0) + 1e-6

            self.bias.data = -mean
            self.scale.data = 1.0 / std

        self.initialized = True

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.initialized:
            self.initialize(z)

        x = (z + self.bias) * self.scale
        log_det = torch.log(torch.abs(self.scale)).sum() * torch.ones(z.shape[0], device=z.device)

        return x, log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = x / self.scale - self.bias
        log_det = -torch.log(torch.abs(self.scale)).sum() * torch.ones(x.shape[0], device=x.device)

        return z, log_det


class RealNVP(nn.Module):
    """
    RealNVP: Normalizing Flow completo.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 6,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers

        # Prior: N(0, I)
        self.prior = Normal(torch.tensor(0.0), torch.tensor(1.0))

        # Construir flujo
        layers = []

        for i in range(num_layers):
            # ActNorm
            layers.append(ActNorm(dim))

            # Coupling layer (alternando mascara)
            layers.append(
                AffineCouplingLayer(
                    dim=dim,
                    hidden_dims=hidden_dims,
                    mask_type="alternate",
                    mask_idx=i % 2,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward: z → x (generacion)

        Args:
            z: Samples del prior [B, dim]

        Returns:
            x: Samples generados [B, dim]
            log_det: Log det Jacobiano total
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)

        x = z
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det

        return x, log_det_total

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse: x → z (inference)
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Calcula log p(x).

        log p(x) = log p(z) + log |det(J)|
        """
        z, log_det = self.inverse(x)

        # Log prob del prior
        log_pz = self.prior.log_prob(z).sum(dim=-1)

        return log_pz + log_det

    def sample(self, num_samples: int, device: str = "cuda") -> Tensor:
        """
        Genera muestras.
        """
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.forward(z)
        return x

    def loss(self, x: Tensor) -> Tensor:
        """
        Negative log-likelihood loss.
        """
        return -self.log_prob(x).mean()


def train_realnvp(
    model: RealNVP,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
) -> List[float]:
    """
    Entrena RealNVP.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device)

            loss = model.loss(x)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: NLL = {avg_loss:.4f}")

    return losses


# =============================================================================
# EJEMPLO 2D
# =============================================================================

def demo_2d_flow():
    """Demo de RealNVP en datos 2D."""

    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Crear datos 2D (dos lunas)
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=1000, noise=0.05)
    X = torch.FloatTensor(X)

    # Normalizar
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    dataloader = torch.utils.data.DataLoader(X, batch_size=128, shuffle=True)

    # Modelo
    model = RealNVP(dim=2, num_layers=8, hidden_dims=(64, 64))

    # Entrenar
    losses = train_realnvp(model, dataloader, num_epochs=200, device=device)

    # Visualizar
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Datos originales
    axes[0].scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Datos Originales")

    # Espacio latente
    with torch.no_grad():
        z, _ = model.inverse(X.to(device))
        z = z.cpu().numpy()

    axes[1].scatter(z[:, 0], z[:, 1], s=5, alpha=0.5)
    axes[1].set_title("Espacio Latente (deberia ser gaussiano)")

    # Muestras generadas
    with torch.no_grad():
        samples = model.sample(1000, device=device).cpu().numpy()

    axes[2].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
    axes[2].set_title("Muestras Generadas")

    plt.savefig("realnvp_2d.png", dpi=150)
    plt.show()

    return model


if __name__ == "__main__":
    demo_2d_flow()
```

---

## Glow: Generative Flow with Invertible 1x1 Convolutions

```
GLOW: FLUJO PARA IMAGENES
=========================

Glow extiende RealNVP para imagenes con:
1. Invertible 1x1 convolutions
2. Arquitectura multi-escala
3. ActNorm (mejor que BatchNorm)


ARQUITECTURA MULTI-ESCALA:
──────────────────────────

    Input: x [C, H, W]
           │
           ▼
    ┌──────────────┐
    │  Squeeze 2x  │  [4C, H/2, W/2]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  K Flow Steps│  (ActNorm + Conv1x1 + Coupling)
    └──────┬───────┘
           │
           ├─────────────────▶ z₁ [2C, H/2, W/2]  (split)
           │
           ▼
    ┌──────────────┐
    │  Squeeze 2x  │  [8C, H/4, W/4]
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  K Flow Steps│
    └──────┬───────┘
           │
           ├─────────────────▶ z₂ [4C, H/4, W/4]
           │
           ▼
           ...
           │
           ▼
           z_L [final]


    z = concat(z₁, z₂, ..., z_L)  (latent total)


SQUEEZE OPERATION:
──────────────────

Reorganiza pixeles para aumentar canales:

    [C, H, W] → [4C, H/2, W/2]

    ┌───┬───┐       ┌───┐
    │ a │ b │       │ a │
    ├───┼───┤  →    │ b │  (4 pixeles → 4 canales)
    │ c │ d │       │ c │
    └───┴───┘       │ d │
                    └───┘


INVERTIBLE 1x1 CONVOLUTION:
───────────────────────────

Reemplaza la permutacion fija de RealNVP con una
permutacion APRENDIDA via convolucion 1x1.

    W: matriz CxC (pesos de la conv)

    Forward:  x_new = W @ x      (por canal)
    Inverse:  x = W⁻¹ @ x_new

    log det = H * W * log|det(W)|

Parametrizacion LU para eficiencia:
    W = P @ L @ U  (descomposicion LU)

    det(W) = det(L) * det(U) = ∏ diag(U)

    Esto hace det(W) O(C) en vez de O(C³)


FLOW STEP EN GLOW:
──────────────────

    Input
      │
      ▼
    ┌─────────────┐
    │   ActNorm   │  ← Normalizacion aprendida
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Conv 1x1   │  ← Permutacion aprendida
    │  (invertible)│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Affine     │  ← Coupling layer
    │  Coupling   │
    └──────┬──────┘
           │
           ▼
    Output
```

### Implementacion Glow

```python
"""
Glow: Generative Flow with Invertible 1x1 Convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List
import numpy as np


class InvertibleConv1x1(nn.Module):
    """
    Invertible 1x1 Convolution con parametrizacion LU.
    """

    def __init__(self, num_channels: int):
        super().__init__()

        self.num_channels = num_channels

        # Inicializar con matriz ortogonal
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]

        # Descomposicion LU
        P, L, U = torch.linalg.lu(W)

        # P es permutacion (fija)
        self.register_buffer("P", P)

        # L es lower triangular con 1s en diagonal
        # Guardamos la parte estrictamente lower
        self.L = nn.Parameter(torch.tril(L, diagonal=-1))
        self.register_buffer("L_mask", torch.tril(torch.ones_like(L), diagonal=-1))

        # U es upper triangular
        # Separamos diagonal (para det) y parte upper estricta
        self.U = nn.Parameter(torch.triu(U, diagonal=1))
        self.register_buffer("U_mask", torch.triu(torch.ones_like(U), diagonal=1))

        # Diagonal de U (determina el determinante)
        self.log_s = nn.Parameter(torch.log(torch.abs(torch.diag(U))))
        self.register_buffer("sign_s", torch.sign(torch.diag(U)))

    def _get_weight(self) -> Tensor:
        """Reconstruye la matriz W."""
        L = self.L * self.L_mask + torch.eye(self.num_channels, device=self.L.device)
        U = self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U
        return W

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward: x @ W

        Args:
            x: [B, C, H, W]

        Returns:
            y: [B, C, H, W]
            log_det: scalar (per sample)
        """
        B, C, H, W = x.shape

        weight = self._get_weight()

        # Aplicar conv 1x1
        y = F.conv2d(x, weight.view(C, C, 1, 1))

        # Log det
        log_det = H * W * self.log_s.sum()

        return y, log_det * torch.ones(B, device=x.device)

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse: y @ W^{-1}
        """
        B, C, H, W = y.shape

        weight = self._get_weight()
        weight_inv = torch.linalg.inv(weight)

        x = F.conv2d(y, weight_inv.view(C, C, 1, 1))

        log_det = -H * W * self.log_s.sum()

        return x, log_det * torch.ones(B, device=y.device)


class ActNorm2d(nn.Module):
    """ActNorm para imagenes."""

    def __init__(self, num_channels: int):
        super().__init__()

        self.num_channels = num_channels
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False

    @torch.no_grad()
    def initialize(self, x: Tensor):
        """Data-dependent initialization."""
        # x: [B, C, H, W]
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6

        self.bias.data = -mean
        self.scale.data = 1.0 / std
        self.initialized = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.initialized:
            self.initialize(x)

        B, C, H, W = x.shape

        y = (x + self.bias) * self.scale

        log_det = H * W * torch.log(torch.abs(self.scale)).sum()

        return y, log_det * torch.ones(B, device=x.device)

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, H, W = y.shape

        x = y / self.scale - self.bias

        log_det = -H * W * torch.log(torch.abs(self.scale)).sum()

        return x, log_det * torch.ones(B, device=y.device)


class AffineCouplingConv(nn.Module):
    """
    Affine Coupling para imagenes (CNN-based).
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
    ):
        super().__init__()

        # Split channels
        self.split_channels = num_channels // 2

        # NN para s y t
        self.net = nn.Sequential(
            nn.Conv2d(self.split_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.split_channels * 2, 3, padding=1),
        )

        # Inicializar ultima capa a cero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward coupling.
        """
        x_a, x_b = x.chunk(2, dim=1)

        # Compute s and t from x_a
        st = self.net(x_a)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)  # Bound scale

        # Transform x_b
        y_b = x_b * torch.exp(s) + t
        y = torch.cat([x_a, y_b], dim=1)

        # Log det
        log_det = s.sum(dim=[1, 2, 3])

        return y, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Inverse coupling.
        """
        y_a, y_b = y.chunk(2, dim=1)

        st = self.net(y_a)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)

        x_b = (y_b - t) * torch.exp(-s)
        x = torch.cat([y_a, x_b], dim=1)

        log_det = -s.sum(dim=[1, 2, 3])

        return x, log_det


class FlowStep(nn.Module):
    """Un paso del flow en Glow."""

    def __init__(self, num_channels: int, hidden_channels: int = 512):
        super().__init__()

        self.actnorm = ActNorm2d(num_channels)
        self.conv1x1 = InvertibleConv1x1(num_channels)
        self.coupling = AffineCouplingConv(num_channels, hidden_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det = 0

        x, ld = self.actnorm(x)
        log_det = log_det + ld

        x, ld = self.conv1x1(x)
        log_det = log_det + ld

        x, ld = self.coupling(x)
        log_det = log_det + ld

        return x, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        log_det = 0

        y, ld = self.coupling.inverse(y)
        log_det = log_det + ld

        y, ld = self.conv1x1.inverse(y)
        log_det = log_det + ld

        y, ld = self.actnorm.inverse(y)
        log_det = log_det + ld

        return y, log_det


def squeeze(x: Tensor) -> Tensor:
    """
    Squeeze: [B, C, H, W] -> [B, 4C, H/2, W/2]
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * 4, H // 2, W // 2)
    return x


def unsqueeze(x: Tensor) -> Tensor:
    """
    Unsqueeze: [B, 4C, H/2, W/2] -> [B, C, H, W]
    """
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // 4, H * 2, W * 2)
    return x


class Glow(nn.Module):
    """
    Glow: Multi-scale Normalizing Flow para imagenes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 512,
        num_levels: int = 3,
        num_steps: int = 32,
    ):
        super().__init__()

        self.num_levels = num_levels
        self.num_steps = num_steps

        # Build flow
        self.flows = nn.ModuleList()

        current_channels = in_channels * 4  # After first squeeze

        for level in range(num_levels):
            level_flows = nn.ModuleList()

            for _ in range(num_steps):
                level_flows.append(
                    FlowStep(current_channels, hidden_channels)
                )

            self.flows.append(level_flows)

            # After split (except last level)
            if level < num_levels - 1:
                current_channels = current_channels * 2  # squeeze doubles again

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Forward: x -> z (encode)

        Returns:
            z_list: Lista de latents por nivel
            log_det: Log det total
        """
        z_list = []
        log_det = torch.zeros(x.shape[0], device=x.device)

        # Squeeze inicial
        h = squeeze(x)

        for level in range(self.num_levels):
            # Flow steps
            for flow in self.flows[level]:
                h, ld = flow(h)
                log_det = log_det + ld

            # Split (except last level)
            if level < self.num_levels - 1:
                # Split canales por la mitad
                z, h = h.chunk(2, dim=1)
                z_list.append(z)

                # Squeeze para siguiente nivel
                h = squeeze(h)

        z_list.append(h)

        return z_list, log_det

    def inverse(self, z_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Inverse: z -> x (decode/generate)
        """
        log_det = torch.zeros(z_list[-1].shape[0], device=z_list[-1].device)

        h = z_list[-1]

        for level in range(self.num_levels - 1, -1, -1):
            # Unsplit (except first iteration)
            if level < self.num_levels - 1:
                h = unsqueeze(h)
                h = torch.cat([z_list[level], h], dim=1)

            # Reverse flow steps
            for flow in reversed(self.flows[level]):
                h, ld = flow.inverse(h)
                log_det = log_det + ld

        # Unsqueeze final
        x = unsqueeze(h)

        return x, log_det

    def log_prob(self, x: Tensor) -> Tensor:
        """Calcula log p(x)."""
        z_list, log_det = self.forward(x)

        # Log prob del prior (gaussiano)
        log_pz = 0
        for z in z_list:
            log_pz = log_pz - 0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=[1, 2, 3])

        return log_pz + log_det

    def sample(self, num_samples: int, temperature: float = 1.0, device: str = "cuda") -> Tensor:
        """Genera muestras."""
        # Necesitamos saber las shapes de cada z
        # Esto deberia calcularse basado en la arquitectura
        # Por simplicidad, asumimos imagenes 32x32x3

        z_list = []
        C = 3 * 4  # Despues de squeeze inicial

        # Esto es aproximado - las shapes reales dependen de la entrada
        shapes = [
            (num_samples, C, 16, 16),       # Nivel 0 split
            (num_samples, C * 2, 8, 8),     # Nivel 1 split
            (num_samples, C * 4, 4, 4),     # Nivel 2 (final)
        ]

        for shape in shapes:
            z = torch.randn(shape, device=device) * temperature
            z_list.append(z)

        x, _ = self.inverse(z_list)

        return x

    def loss(self, x: Tensor) -> Tensor:
        """NLL loss."""
        return -self.log_prob(x).mean()
```

---

## Flow Matching: Un Enfoque Moderno

```
FLOW MATCHING: ALTERNATIVA A SCORE MATCHING
===========================================

Flow Matching es un metodo reciente que entrena flows
de manera similar a diffusion models, pero mas simple.


IDEA CLAVE:
───────────

En lugar de disenar transformaciones invertibles especificas,
aprendemos un VECTOR FIELD que transporta el prior a los datos.


FORMULACION:
────────────

Definimos un flujo temporal:

    dx/dt = v_θ(x, t)     para t ∈ [0, 1]

Con condiciones de frontera:
    - x(0) ~ p₀(x) = N(0, I)     (prior)
    - x(1) ~ p₁(x) = p_data(x)   (datos)


El flujo determina una ODE:

    x₁ = x₀ + ∫₀¹ v_θ(x_t, t) dt


TRAINING OBJECTIVE:
───────────────────

Flow Matching loss (simplificada):

    L = E_{t, x₀, x₁}[ ||v_θ(x_t, t) - u_t(x_t)||² ]

Donde:
    - t ~ Uniform(0, 1)
    - x₀ ~ p₀ (prior)
    - x₁ ~ p_data (datos)
    - x_t = (1-t)x₀ + t·x₁   (interpolacion lineal)
    - u_t = x₁ - x₀          (velocidad "ground truth")


COMPARACION CON DIFFUSION:
──────────────────────────

Diffusion:                      Flow Matching:
──────────                      ─────────────

Forward: agregar ruido          Forward: interpolacion lineal
xₜ = √αₜ x₀ + √(1-αₜ) ε        xₜ = (1-t)x₀ + t·x₁

Target: predecir ruido          Target: predecir velocidad
εθ(xₜ, t) → ε                   vθ(xₜ, t) → x₁ - x₀

Sampling: iterativo             Sampling: resolver ODE
(DDPM, DDIM, etc.)              (Euler, RK4, etc.)


VENTAJAS DE FLOW MATCHING:
──────────────────────────

1. Paths rectos (mas simple de aprender)
2. Sin noise schedule que tunear
3. Likelihood exacta (via ODE)
4. Mas estable numericamente


DIAGRAMA:
─────────

    t=0         t=0.5         t=1
    prior       intermedio    datos

    ○             ◐            ●
    │             │            │
    │  ────────▶  │  ────────▶ │
    │   v_θ(x,t)  │   v_θ(x,t) │
    │             │            │

    El modelo aprende v_θ que
    "empuja" el prior hacia datos
```

### Implementacion Flow Matching

```python
"""
Flow Matching: Entrenamiento de flujos via regression.
Implementacion moderna y simple.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, Callable
from tqdm import tqdm


class VelocityNetwork(nn.Module):
    """
    Red neuronal para el campo de velocidad v_θ(x, t).
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.dim = dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Main network
        layers = []
        in_dim = dim + time_embed_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: Position [B, dim]
            t: Time [B, 1] or [B]

        Returns:
            v: Velocity [B, dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)


class FlowMatching(nn.Module):
    """
    Flow Matching para generacion.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        sigma_min: float = 0.001,  # Ruido minimo para estabilidad
    ):
        super().__init__()

        self.dim = dim
        self.sigma_min = sigma_min

        self.velocity = VelocityNetwork(dim, hidden_dims)

    def get_interpolation(
        self,
        x0: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calcula interpolacion y velocidad target.

        Interpolacion lineal optima (OT path):
            x_t = (1-t) * x0 + t * x1

        Velocidad target:
            u_t = x1 - x0

        Args:
            x0: Prior samples [B, dim]
            x1: Data samples [B, dim]
            t: Time [B, 1]

        Returns:
            x_t: Interpolated point [B, dim]
            u_t: Target velocity [B, dim]
        """
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0

        return x_t, u_t

    def loss(self, x1: Tensor) -> Tensor:
        """
        Flow Matching loss.

        L = E[ ||v_θ(x_t, t) - u_t||² ]

        Args:
            x1: Data batch [B, dim]

        Returns:
            Loss scalar
        """
        B = x1.shape[0]
        device = x1.device

        # Sample x0 from prior
        x0 = torch.randn_like(x1)

        # Sample t uniformly
        t = torch.rand(B, 1, device=device)

        # Get interpolation and target
        x_t, u_t = self.get_interpolation(x0, x1, t)

        # Predict velocity
        v_pred = self.velocity(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, u_t)

        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = "cuda",
        method: str = "euler",
    ) -> Tensor:
        """
        Genera muestras resolviendo la ODE.

        dx/dt = v_θ(x, t)
        x(0) ~ N(0, I)  ->  x(1) ~ p_data

        Args:
            num_samples: Numero de muestras
            num_steps: Pasos de integracion
            device: Dispositivo
            method: Metodo de integracion ("euler", "rk4")

        Returns:
            Samples [num_samples, dim]
        """
        # Initial samples from prior
        x = torch.randn(num_samples, self.dim, device=device)

        # Time steps
        dt = 1.0 / num_steps
        ts = torch.linspace(0, 1 - dt, num_steps, device=device)

        for t in tqdm(ts, desc="Sampling"):
            t_batch = torch.full((num_samples, 1), t, device=device)

            if method == "euler":
                # Euler method
                v = self.velocity(x, t_batch)
                x = x + v * dt

            elif method == "rk4":
                # Runge-Kutta 4
                k1 = self.velocity(x, t_batch)
                k2 = self.velocity(x + 0.5 * dt * k1, t_batch + 0.5 * dt)
                k3 = self.velocity(x + 0.5 * dt * k2, t_batch + 0.5 * dt)
                k4 = self.velocity(x + dt * k3, t_batch + dt)
                x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return x

    @torch.no_grad()
    def log_prob(
        self,
        x: Tensor,
        num_steps: int = 100,
    ) -> Tensor:
        """
        Calcula log p(x) via ODE inversa.

        Usa la formula de cambio de variable para ODEs:

        log p(x_1) = log p(x_0) - ∫₀¹ div(v_θ(x_t, t)) dt

        Args:
            x: Data points [B, dim]
            num_steps: Pasos de integracion

        Returns:
            Log probabilities [B]
        """
        B = x.shape[0]
        device = x.device

        # Integrate backwards: x_1 -> x_0
        dt = 1.0 / num_steps
        ts = torch.linspace(1 - dt, 0, num_steps, device=device)

        log_det = torch.zeros(B, device=device)

        for t in ts:
            t_batch = torch.full((B, 1), t, device=device)

            # Compute velocity
            x.requires_grad_(True)
            v = self.velocity(x, t_batch)

            # Compute divergence via autodiff
            div = 0
            for i in range(self.dim):
                div = div + torch.autograd.grad(
                    v[:, i].sum(), x, create_graph=False
                )[0][:, i]

            x = x.detach()

            # Update
            x = x - v * dt
            log_det = log_det + div * dt

        # Prior log prob
        log_p0 = -0.5 * (x ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        return log_p0 + log_det


def train_flow_matching(
    model: FlowMatching,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
) -> list:
    """Entrena Flow Matching."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device)

            loss = model.loss(x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    return losses


# =============================================================================
# CONDITIONAL FLOW MATCHING
# =============================================================================

class ConditionalFlowMatching(FlowMatching):
    """
    Conditional Flow Matching para generacion condicionada.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
    ):
        super().__init__(dim, hidden_dims)

        # Override velocity network to include condition
        self.velocity = ConditionalVelocityNetwork(dim, cond_dim, hidden_dims)

    def loss(self, x1: Tensor, cond: Tensor) -> Tensor:
        """
        Conditional Flow Matching loss.

        Args:
            x1: Data [B, dim]
            cond: Condition [B, cond_dim]
        """
        B = x1.shape[0]
        device = x1.device

        x0 = torch.randn_like(x1)
        t = torch.rand(B, 1, device=device)

        x_t, u_t = self.get_interpolation(x0, x1, t)

        v_pred = self.velocity(x_t, t, cond)

        return F.mse_loss(v_pred, u_t)

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        num_steps: int = 100,
        method: str = "euler",
    ) -> Tensor:
        """
        Conditional sampling.

        Args:
            cond: Conditions [B, cond_dim]
            num_steps: Integration steps
        """
        B = cond.shape[0]
        device = cond.device

        x = torch.randn(B, self.dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.velocity(x, t, cond)
            x = x + v * dt

        return x


class ConditionalVelocityNetwork(nn.Module):
    """Velocity network con condicionamiento."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        hidden_dims: Tuple[int, ...],
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        layers = []
        in_dim = dim + time_embed_dim + cond_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb, cond], dim=-1)

        return self.net(h)


# =============================================================================
# DEMO
# =============================================================================

import numpy as np

def demo_flow_matching():
    """Demo de Flow Matching en datos 2D."""

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_swiss_roll

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datos: Swiss roll 2D
    X, _ = make_swiss_roll(n_samples=2000, noise=0.1)
    X = X[:, [0, 2]]  # Solo 2D
    X = torch.FloatTensor(X)
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    dataloader = torch.utils.data.DataLoader(X, batch_size=256, shuffle=True)

    # Modelo
    model = FlowMatching(dim=2, hidden_dims=(256, 256, 256))

    # Entrenar
    losses = train_flow_matching(model, dataloader, num_epochs=200, device=device)

    # Visualizar
    model.eval()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Datos originales
    axes[0].scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Datos Originales")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)

    # Prior
    prior = torch.randn(2000, 2)
    axes[1].scatter(prior[:, 0], prior[:, 1], s=5, alpha=0.5, c='orange')
    axes[1].set_title("Prior N(0, I)")
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)

    # Muestras generadas
    samples = model.sample(2000, num_steps=100, device=device).cpu().numpy()
    axes[2].scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, c='green')
    axes[2].set_title("Muestras Generadas")
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig("flow_matching_demo.png", dpi=150)
    plt.show()

    return model


if __name__ == "__main__":
    demo_flow_matching()
```

---

## Aplicaciones en Density Estimation

```
DENSITY ESTIMATION CON FLOWS
============================

Normalizing Flows son especialmente utiles para:

1. ESTIMACION DE DENSIDAD EXACTA
────────────────────────────────

   Dado x, podemos calcular p(x) exactamente:

   log p(x) = log p(f⁻¹(x)) + log |det(J_{f⁻¹})|

   Aplicaciones:
   - Deteccion de anomalias (baja p(x) = anomalia)
   - Compresion sin perdida
   - Evaluacion de modelos


2. INFERENCE EN MODELOS BAYESIANOS
──────────────────────────────────

   Aproximar posteriors complejos:

   q(θ|data) = f_θ(z) donde z ~ N(0, I)

   Variational Inference con flows:
   - ELBO con likelihood exacta
   - Posteriors multimodales


3. GENERACION DE DATOS
──────────────────────

   z ~ N(0, I) → x = f(z)

   Con control de:
   - Temperatura (escalar z)
   - Interpolacion (interpolar en z-space)


METRICAS PARA FLOWS:
────────────────────

┌────────────────┬─────────────────────────────────────────┐
│ Metrica        │ Descripcion                             │
├────────────────┼─────────────────────────────────────────┤
│ BPD            │ Bits per dimension                      │
│ (bits/dim)     │ = -log₂ p(x) / dim                     │
│                │ Menor = mejor compresion               │
├────────────────┼─────────────────────────────────────────┤
│ NLL            │ Negative Log-Likelihood                 │
│                │ = -E[log p(x)]                         │
│                │ Menor = mejor fit                      │
├────────────────┼─────────────────────────────────────────┤
│ FID            │ Frechet Inception Distance             │
│                │ Para calidad visual (no likelihood)    │
└────────────────┴─────────────────────────────────────────┘


COMPARACION DE ARQUITECTURAS:
─────────────────────────────

┌─────────────┬───────────┬───────────┬───────────┬───────────┐
│             │ RealNVP   │ Glow      │ Flow      │ FFJORD    │
│             │           │           │ Matching  │           │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ Coupling    │ Affine    │ Affine +  │ ODE       │ Continuous│
│             │           │ Conv1x1   │           │ ODE       │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ Invertible  │ Por diseño│ Por diseño│ Solver    │ Solver    │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ log det     │ O(D)      │ O(D)      │ Traza     │ Hutchinson│
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ Expresividad│ Media     │ Alta      │ Alta      │ Muy alta  │
├─────────────┼───────────┼───────────┼───────────┼───────────┤
│ Velocidad   │ Muy rapido│ Rapido    │ Moderado  │ Lento     │
└─────────────┴───────────┴───────────┴───────────┴───────────┘
```

---

## Resumen

```
CONCEPTOS CLAVE - NORMALIZING FLOWS
===================================

1. FUNDAMENTOS
   - Transformaciones INVERTIBLES z ↔ x
   - Cambio de variable: p(x) = p(z) / |det(J)|
   - Likelihood EXACTA (no bound como VAE)

2. ARQUITECTURAS
   - RealNVP: Affine coupling layers
     x_b = z_b * exp(s(z_a)) + t(z_a)
   - Glow: + Conv1x1 invertible + multi-scale
   - Flow Matching: ODE dx/dt = v_θ(x,t)

3. JACOBIANO TRIANGULAR
   - Disenar f con J triangular
   - det(J) = producto de diagonal = O(D)
   - Clave para eficiencia

4. TRAINING
   - Maximizar log p(x) = log p(z) + log|det(J)|
   - Flow Matching: ||v_θ - u_t||² (regression)

5. SAMPLING
   - Flows clasicos: z → f(z) = x (un paso)
   - Flow Matching: resolver ODE (multiple pasos)

6. VENTAJAS
   - Likelihood exacta
   - Invertible (encode Y decode)
   - Sin modo collapse


EVOLUCION:
──────────

2015: NICE (Dinh et al.)
2017: RealNVP (Dinh et al.)
2018: Glow (Kingma & Dhariwal)
2018: FFJORD (Grathwohl et al.)
2023: Flow Matching (Lipman et al.)
2024: Rectified Flows, consistency models


CUANDO USAR FLOWS:
──────────────────

✓ Necesitas likelihood exacta
✓ Deteccion de anomalias
✓ Compresion
✓ Inference bayesiana

✗ Solo generacion (diffusion es mejor)
✗ Imagenes muy alta resolucion (costoso)
```

---

## Referencias

1. Dinh, L., et al. (2015). "NICE: Non-linear Independent Components Estimation"
2. Dinh, L., et al. (2017). "Density estimation using Real-NVP"
3. Kingma, D.P., & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1x1 Convolutions"
4. Grathwohl, W., et al. (2019). "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"
5. Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling"
6. Liu, X., et al. (2023). "Rectified Flow: A Marginal Preserving Approach to Optimal Transport"
