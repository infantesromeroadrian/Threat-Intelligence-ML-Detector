# Autoencoders

## Introduccion

Los **Autoencoders** son redes neuronales que aprenden a **comprimir** datos en una representacion de menor dimension (encoding) y luego **reconstruirlos** (decoding). Su objetivo es aprender representaciones latentes utiles de los datos de forma **no supervisada**.

```
ARQUITECTURA BASICA
===================

        Input                    Latent Space              Output
       (x)                          (z)                    (x̂)

    ┌───────┐                    ┌───────┐                ┌───────┐
    │ 784   │                    │   32  │                │ 784   │
    │       │  ┌──────────────┐  │       │  ┌──────────┐  │       │
    │ dims  │──│   ENCODER    │──│ dims  │──│ DECODER  │──│ dims  │
    │       │  │   (comprime) │  │       │  │(expande) │  │       │
    └───────┘  └──────────────┘  └───────┘  └──────────┘  └───────┘

              ←────────────────────────────────────────────→
                            Autoencoder

Objetivo: Minimizar ||x - x̂||² (error de reconstruccion)

La red aprende una representacion comprimida z que captura
las caracteristicas mas importantes de x.
```

## Tipos de Autoencoders

### Vision General

```
FAMILIA DE AUTOENCODERS
=======================

                        Autoencoder
                             │
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
    Vanilla             Regularizado          Generativo
       │                     │                     │
       │              ┌──────┴──────┐              │
       │              │             │              │
       │           Sparse      Denoising          VAE
       │         (L1 penalty) (ruido input)   (probabilistico)
       │
    ┌──┴──┐
    │     │
  Under  Over
complete complete
(z < x)  (z > x)
```

---

## Autoencoder Vanilla

### Arquitectura

```
AUTOENCODER UNDERCOMPLETE
=========================

Input Layer      Hidden Layers        Bottleneck       Hidden Layers     Output Layer
   (x)              (h)                  (z)               (h')             (x̂)

    ●                 ●                   ●                  ●                ●
    ●                 ●                                      ●                ●
    ●                 ●                   ●                  ●                ●
    ●       ───→      ●       ───→                 ───→      ●      ───→      ●
    ●                 ●                   ●                  ●                ●
    ●                 ●                                      ●                ●
    ●                 ●                   ●                  ●                ●

   784               256                  32                256              784

          └──────── Encoder ────────┘   │   └──────── Decoder ────────┘

Undercomplete: dim(z) < dim(x) → Fuerza compresion
Overcomplete:  dim(z) > dim(x) → Necesita regularizacion
```

### Implementacion PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple


class VanillaAutoencoder(nn.Module):
    """
    Autoencoder basico para compresion de datos.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (arquitectura simetrica)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output en [0, 1]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Comprime input a espacio latente."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruye desde espacio latente."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass completo."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolucional para imagenes.
    Mejor para datos con estructura espacial.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 7 x 7
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 4 x 4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder_conv = nn.Sequential(
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 7 x 7
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 14 x 14
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # 1 x 28 x 28
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3
) -> list[float]:
    """Entrena autoencoder vanilla."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # o BCELoss para imagenes normalizadas

    losses = []
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_x, _ in train_loader:  # Ignoramos labels
            optimizer.zero_grad()

            x_recon, z = model(batch_x)
            loss = criterion(x_recon, batch_x)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return losses
```

---

## Denoising Autoencoder (DAE)

### Concepto

```
DENOISING AUTOENCODER
=====================

La idea: Corromper el input con ruido y entrenar para reconstruir
         el original LIMPIO.

    Input Original     Input Corrupto      Reconstruccion
         (x)            (x̃ = x + noise)         (x̂)

        ┌───┐            ┌╌╌╌┐              ┌───┐
        │   │     →      ┆   ┆      →       │   │
        │ ● │   ruido    ┆●. ┆   encoder    │ ● │
        │   │            ┆.  ┆   decoder    │   │
        └───┘            └╌╌╌┘              └───┘

Loss = ||x - x̂||²  (comparamos con original limpio!)

Beneficios:
- Aprende features mas robustas
- Mejor generalizacion
- Representaciones mas significativas
```

### Tipos de Corrupcion

| Tipo | Descripcion | Uso |
|------|-------------|-----|
| Gaussian | Sumar ruido N(0, σ²) | Datos continuos |
| Dropout/Masking | Poner valores a 0 | Datos discretos |
| Salt & Pepper | Valores extremos aleatorios | Imagenes |
| Swap | Intercambiar features | Datos tabulares |

### Implementacion

```python
import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    """
    Autoencoder que aprende a limpiar datos corruptos.
    Util para deteccion de anomalias y feature learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        noise_type: str = 'gaussian',
        noise_factor: float = 0.3
    ):
        super().__init__()

        self.noise_type = noise_type
        self.noise_factor = noise_factor

        # Encoder
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Anade ruido al input."""
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise

        elif self.noise_type == 'masking':
            mask = torch.rand_like(x) > self.noise_factor
            return x * mask.float()

        elif self.noise_type == 'salt_pepper':
            noisy = x.clone()
            # Salt
            salt = torch.rand_like(x) < (self.noise_factor / 2)
            noisy[salt] = 1.0
            # Pepper
            pepper = torch.rand_like(x) < (self.noise_factor / 2)
            noisy[pepper] = 0.0
            return noisy

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        add_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            x_recon: reconstruccion
            z: representacion latente
            x_noisy: input con ruido (para debugging)
        """
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        z = self.encode(x_noisy)
        x_recon = self.decode(z)

        return x_recon, z, x_noisy


def train_denoising_ae(
    model: DenoisingAutoencoder,
    train_loader: DataLoader,
    epochs: int = 50
) -> None:
    """Entrena DAE comparando con input original."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            # Reconstruir desde input ruidoso
            x_recon, _, _ = model(batch_x, add_noise=True)

            # Loss contra original LIMPIO
            loss = criterion(x_recon, batch_x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
```

---

## Sparse Autoencoder

### Concepto

```
SPARSE AUTOENCODER
==================

Idea: Forzar que solo pocas neuronas se activen.

Hidden Layer sin sparsity:        Hidden Layer con sparsity:
┌─────────────────────────┐       ┌─────────────────────────┐
│ ● ● ● ● ● ● ● ● ● ●    │       │ ○ ● ○ ○ ● ○ ○ ● ○ ○    │
│ (muchas activaciones)   │       │ (pocas activaciones)    │
└─────────────────────────┘       └─────────────────────────┘

Regularizacion:
    L_total = L_reconstruction + λ · L_sparsity

L_sparsity: Penaliza activaciones promedio altas
            KL(ρ || ρ̂) donde ρ = target sparsity (~0.05)
                             ρ̂ = activacion promedio real

Beneficios:
- Features mas interpretables
- Permite overcomplete autoencoders
- Representaciones mas disentangled
```

### Implementacion

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Autoencoder con penalizacion de sparsity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        sparsity_target: float = 0.05,
        sparsity_weight: float = 1.0
    ):
        super().__init__()

        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

        # Encoder con activacion sigmoide para calcular sparsity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),  # Necesario para sparsity
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # Para almacenar activaciones
        self.activations: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Guardar activaciones intermedias
        self.activations = []

        # Forward manual para capturar activaciones
        h1 = torch.sigmoid(self.encoder[0](x))
        self.activations.append(h1)

        h2 = torch.sigmoid(self.encoder[2](h1))
        self.activations.append(h2)

        # Decoder
        x_recon = self.decoder(h2)

        return x_recon, h2

    def kl_divergence(self, rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
        """
        KL divergence entre distribucion target y real.
        KL(ρ || ρ̂) = ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))
        """
        rho = torch.tensor(rho)
        return rho * torch.log(rho / rho_hat) + \
               (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    def sparsity_loss(self) -> torch.Tensor:
        """Calcula perdida de sparsity sobre todas las capas."""
        total_loss = torch.tensor(0.0)

        for activation in self.activations:
            # Activacion promedio por neurona
            rho_hat = activation.mean(dim=0)
            # Evitar log(0)
            rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
            # KL divergence
            kl = self.kl_divergence(self.sparsity_target, rho_hat)
            total_loss = total_loss + kl.sum()

        return total_loss


def train_sparse_ae(
    model: SparseAutoencoder,
    train_loader: DataLoader,
    epochs: int = 50
) -> None:
    """Entrena sparse autoencoder."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_sparse = 0.0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            x_recon, _ = model(batch_x)

            # Reconstruction loss
            recon_loss = mse_loss(x_recon, batch_x)

            # Sparsity loss
            sparse_loss = model.sparsity_loss()

            # Total loss
            loss = recon_loss + model.sparsity_weight * sparse_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparse += sparse_loss.item()

        if epoch % 10 == 0:
            n = len(train_loader)
            print(f"Epoch {epoch}: Total={total_loss/n:.4f}, "
                  f"Recon={total_recon/n:.4f}, Sparse={total_sparse/n:.4f}")
```

---

## Variational Autoencoder (VAE)

### Concepto

```
VAE: AUTOENCODER PROBABILISTICO
===============================

Autoencoder Normal:           VAE:
x → z → x̂                    x → (μ, σ) → sample z → x̂
   deterministic                    probabilistic

                               ┌─────────────┐
                               │             │
    Input        Encoder       │  μ (mean)   │──┐
      x    ──────────────────→ │             │  │
                               │  σ (std)    │──┤
                               │             │  │
                               └─────────────┘  │
                                                ▼
                               ┌─────────────────────┐
                               │ z = μ + σ · ε       │
                               │     donde ε ~ N(0,1)│
                               │  (reparametrization)│
                               └─────────────────────┘
                                        │
                                        ▼
                               ┌─────────────┐
                               │   Decoder   │──→ x̂
                               └─────────────┘

Loss VAE = Reconstruction Loss + KL Divergence
         = E[log p(x|z)]      - KL(q(z|x) || p(z))
         ≈ -MSE(x, x̂)        + 0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### Reparametrization Trick

```
POR QUE ES NECESARIO
====================

Problema: Queremos hacer backprop a traves de sampling

    μ, σ → sample z ~ N(μ, σ²) → decoder → loss
              ↑
         NO DIFERENCIABLE!

Solucion: Reparametrization

    μ, σ → z = μ + σ * ε  donde ε ~ N(0,1) → decoder → loss
              ↑
         DIFERENCIABLE respecto a μ y σ

    ε es externo, no depende de parametros
    z tiene gradientes respecto a μ y σ
```

### Implementacion VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder.
    Aprende distribucion latente, permite generacion.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Layers para mu y log_var
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparametrization trick.
        z = mu + std * eps, where eps ~ N(0, 1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            x_recon: reconstruction
            mu: mean of latent distribution
            log_var: log variance of latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Genera nuevas muestras desde prior N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE Loss = Reconstruction + β * KL Divergence

    Args:
        x: input original
        x_recon: reconstruccion
        mu: media latente
        log_var: log varianza latente
        beta: peso del KL (beta-VAE)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (BCE o MSE)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL Divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    epochs: int = 50,
    beta: float = 1.0
) -> dict:
    """Entrena VAE."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {'total': [], 'recon': [], 'kl': []}
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            x_recon, mu, log_var = model(batch_x)
            loss, recon, kl = vae_loss(batch_x, x_recon, mu, log_var, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        n = len(train_loader.dataset)
        history['total'].append(total_loss / n)
        history['recon'].append(total_recon / n)
        history['kl'].append(total_kl / n)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total={total_loss/n:.2f}, "
                  f"Recon={total_recon/n:.2f}, KL={total_kl/n:.2f}")

    return history
```

### VAE Convolucional

```python
class ConvVAE(nn.Module):
    """VAE convolucional para imagenes."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 28x28 -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 14x14 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), # 7x7 -> 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 4x4 -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, output_padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
```

---

## Aplicaciones en Ciberseguridad

### 1. Deteccion de Anomalias con Autoencoder

```
PIPELINE DE DETECCION DE ANOMALIAS
==================================

Fase de Entrenamiento (solo datos NORMALES):

    Datos Normales    →    Autoencoder    →    Reconstruction Error
        (x)                                          bajo

Fase de Inferencia:

    Dato Normal       →    Autoencoder    →    Error bajo  →  OK
    Dato Anomalo      →    Autoencoder    →    Error ALTO  →  ALERTA!

                       ┌───────────────────────────────────────┐
                       │                                       │
    Reconstruction     │    Normales          Anomalos         │
    Error              │       ●●●              ●              │
                       │      ●●●●●            ● ●             │
                       │     ●●●●●●●                           │
                       │─────────────────── threshold ─────────│
                       │                                       │
                       └───────────────────────────────────────┘
                                        threshold

El modelo aprende "como es lo normal" y falla
al reconstruir comportamiento anomalo.
```

### Implementacion

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class NetworkAnomalyDetector:
    """
    Detector de anomalias de red usando Autoencoder.
    Entrena solo con trafico normal, detecta desviaciones.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32, 16],
        threshold_percentile: float = 95.0
    ):
        self.scaler = StandardScaler()
        self.threshold_percentile = threshold_percentile
        self.threshold = 0.0

        # Autoencoder
        self.model = self._build_model(input_dim, hidden_dims)

    def _build_model(
        self,
        input_dim: int,
        hidden_dims: list[int]
    ) -> nn.Module:
        """Construye autoencoder."""

        class AE(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int]):
                super().__init__()

                # Encoder
                encoder = []
                in_d = input_dim
                for h_d in hidden_dims:
                    encoder.extend([
                        nn.Linear(in_d, h_d),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    in_d = h_d
                self.encoder = nn.Sequential(*encoder)

                # Decoder (simetrico)
                decoder = []
                for h_d in reversed(hidden_dims[:-1]):
                    decoder.extend([
                        nn.Linear(in_d, h_d),
                        nn.ReLU()
                    ])
                    in_d = h_d
                decoder.append(nn.Linear(in_d, input_dim))
                self.decoder = nn.Sequential(*decoder)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.encoder(x)
                return self.decoder(z)

        return AE(input_dim, hidden_dims)

    def fit(
        self,
        X_normal: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64
    ) -> 'NetworkAnomalyDetector':
        """
        Entrena con datos normales.

        Args:
            X_normal: features de trafico normal [n_samples, n_features]
        """
        # Normalizar
        X_scaled = self.scaler.fit_transform(X_normal)
        X_tensor = torch.FloatTensor(X_scaled)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss(reduction='none')

        dataset = torch.utils.data.TensorDataset(X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.6f}")

        # Calcular threshold en datos de entrenamiento
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
            self.threshold = np.percentile(errors, self.threshold_percentile)

        print(f"Threshold establecido: {self.threshold:.6f}")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta anomalias.

        Args:
            X: features a evaluar

        Returns:
            predictions: 1 = anomalia, 0 = normal
            scores: error de reconstruccion
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()

        predictions = (errors > self.threshold).astype(int)

        return predictions, errors

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Retorna solo scores (para analisis)."""
        _, scores = self.predict(X)
        return scores


# Ejemplo de uso
def demo_network_anomaly():
    """Demo de deteccion de anomalias de red."""

    # Simular features de trafico de red
    # Features: bytes_in, bytes_out, packets, duration, ports, etc.
    np.random.seed(42)

    # Trafico normal
    n_normal = 10000
    X_normal = np.random.randn(n_normal, 20)  # 20 features

    # Trafico anomalo (valores diferentes)
    n_anomaly = 100
    X_anomaly = np.random.randn(n_anomaly, 20) * 3 + 2  # Diferente distribucion

    # Entrenar
    detector = NetworkAnomalyDetector(
        input_dim=20,
        hidden_dims=[64, 32, 16],
        threshold_percentile=99.0
    )
    detector.fit(X_normal, epochs=50)

    # Evaluar
    X_test = np.vstack([X_normal[:100], X_anomaly])
    y_true = np.array([0] * 100 + [1] * n_anomaly)

    predictions, scores = detector.predict(X_test)

    # Metricas
    from sklearn.metrics import classification_report
    print("\nResultados:")
    print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))


if __name__ == "__main__":
    demo_network_anomaly()
```

### 2. Deteccion de Malware con VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MalwareVAE(nn.Module):
    """
    VAE para deteccion de malware basado en features PE.
    Genera representaciones latentes de comportamiento.
    """

    def __init__(
        self,
        input_dim: int = 100,  # Features del PE
        hidden_dim: int = 64,
        latent_dim: int = 16
    ):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)

        # Clasificador sobre espacio latente
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # benign/malware
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        class_logits = self.classifier(z)
        return x_recon, mu, log_var, class_logits


def train_malware_vae(
    model: MalwareVAE,
    train_loader,
    epochs: int = 50,
    alpha: float = 0.5  # peso clasificacion
):
    """
    Entrena VAE semi-supervisado para malware.

    Loss = Reconstruction + KL + alpha * Classification
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            x_recon, mu, log_var, class_logits = model(batch_x)

            # Reconstruction
            recon_loss = F.binary_cross_entropy(x_recon, batch_x, reduction='sum')

            # KL
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Classification
            class_loss = ce_loss(class_logits, batch_y)

            # Total
            loss = recon_loss + kl_loss + alpha * class_loss * batch_x.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader.dataset):.4f}")
```

### 3. Feature Extraction para IDS

```
USO DE AUTOENCODERS PARA IDS
============================

Pipeline:

    Raw Network Data
          │
          ▼
    ┌─────────────────────┐
    │  Feature Extraction │
    │  (estadisticas)     │
    └─────────────────────┘
          │
          ▼
    ┌─────────────────────┐
    │    Autoencoder      │──→ Latent Features (comprimidas)
    │    Pre-entrenado    │
    └─────────────────────┘
          │
          ▼
    ┌─────────────────────┐
    │   Clasificador      │──→ Normal / Ataque
    │   (RF, SVM, NN)     │
    └─────────────────────┘

Ventajas:
1. Reduce dimensionalidad (100 features → 16 latentes)
2. Captura correlaciones no lineales
3. Robusto a ruido
4. Representacion generada es mas informativa
```

```python
class IDSFeatureExtractor:
    """
    Extractor de features usando autoencoder preentrenado.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32):
        self.autoencoder = VanillaAutoencoder(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            latent_dim=latent_dim
        )
        self.trained = False

    def fit(self, X_train: np.ndarray, epochs: int = 100):
        """Entrena autoencoder con datos normales."""
        X_tensor = torch.FloatTensor(X_train)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()

        self.autoencoder.train()
        for epoch in range(epochs):
            for batch_x, _ in loader:
                optimizer.zero_grad()
                x_recon, _ = self.autoencoder(batch_x)
                loss = criterion(x_recon, batch_x)
                loss.backward()
                optimizer.step()

        self.trained = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extrae features latentes."""
        if not self.trained:
            raise RuntimeError("Modelo no entrenado")

        self.autoencoder.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            z = self.autoencoder.encode(X_tensor)

        return z.numpy()

    def fit_transform(self, X: np.ndarray, epochs: int = 100) -> np.ndarray:
        """Entrena y transforma."""
        self.fit(X, epochs)
        return self.transform(X)
```

---

## Comparativa de Autoencoders

| Tipo | Loss | Regularizacion | Uso Principal |
|------|------|----------------|---------------|
| Vanilla | MSE/BCE | Ninguna (undercomplete) | Compresion basica |
| Denoising | MSE vs original limpio | Ruido en input | Features robustas |
| Sparse | MSE + KL sparsity | Activaciones dispersas | Features interpretables |
| VAE | ELBO (Recon + KL) | Prior N(0,I) | Generacion, sampling |
| β-VAE | ELBO con β > 1 | KL amplificado | Features disentangled |

---

## Hiperparametros Clave

| Parametro | Rango | Consideraciones |
|-----------|-------|-----------------|
| latent_dim | 8-256 | Menor = mas compresion, mayor = mas fidelidad |
| hidden_layers | 2-5 | Simetrico encoder/decoder |
| learning_rate | 1e-4 - 1e-3 | Bajo para VAE (evitar posterior collapse) |
| beta (VAE) | 0.1 - 4.0 | Mayor = features mas independientes |
| noise_factor (DAE) | 0.1 - 0.5 | Depende del nivel de ruido esperado |
| sparsity_target | 0.01 - 0.1 | Menor = mas sparse |

---

## Resumen

```
CUANDO USAR CADA AUTOENCODER
============================

┌─────────────────────────────────────────────────────────────┐
│ Tarea                          │ Autoencoder Recomendado   │
├─────────────────────────────────────────────────────────────┤
│ Compresion basica              │ Vanilla (undercomplete)   │
│ Deteccion anomalias            │ Vanilla o Denoising       │
│ Features robustas              │ Denoising                 │
│ Features interpretables        │ Sparse                    │
│ Generacion de muestras         │ VAE                       │
│ Features independientes        │ β-VAE                     │
│ Clasificacion semi-supervisada │ VAE con clasificador      │
└─────────────────────────────────────────────────────────────┘
```

### Puntos Clave

1. **Autoencoders aprenden compresion** de forma no supervisada
2. **Error de reconstruccion** = proxy para "normalidad"
3. **VAE** permite generacion al aprender distribucion latente
4. **Denoising** mejora robustez de representaciones
5. **En ciberseguridad**: deteccion de anomalias, extraccion de features
6. **Threshold** de anomalia debe calibrarse con datos de validacion
