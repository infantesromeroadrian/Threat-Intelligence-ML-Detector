# Variational Autoencoders Avanzados

## Introduccion

Los **Variational Autoencoders (VAEs)** son modelos generativos que aprenden representaciones latentes de datos de forma probabilistica. A diferencia de los autoencoders clasicos, los VAEs modelan la distribucion latente explicitamente, permitiendo generacion de nuevas muestras.

```
VAE vs AUTOENCODER CLASICO
==========================

Autoencoder Clasico:
────────────────────
    x ───▶ Encoder ───▶ z ───▶ Decoder ───▶ x̂

    - z es un vector DETERMINISTA
    - Solo reconstruccion, no generacion
    - Espacio latente puede ser discontinuo


VAE (Variational Autoencoder):
──────────────────────────────
                          ┌─────┐
    x ───▶ Encoder ───┬───│  μ  │──┐
                      │   └─────┘  │
                      │            ├───▶ z = μ + σ⊙ε ───▶ Decoder ───▶ x̂
                      │   ┌─────┐  │
                      └───│  σ  │──┘
                          └─────┘
                              ▲
                              │
                         ε ~ N(0,I)

    - z es una variable ALEATORIA
    - Puede generar nuevas muestras
    - Espacio latente continuo y estructurado


DISTRIBUCION LATENTE:
─────────────────────

                 Autoencoder              VAE
                 ──────────               ───

Espacio             ●                     ╭─────╮
latente           ●   ●                  │     │
                 ●  ●  ●                 │ N(μ,σ)│
                   ● ●                   │     │
                                          ╰─────╯

              Puntos discretos        Distribucion continua
              (sin estructura)        (muestreo posible)
```

---

## Matematicas del VAE: ELBO

### Objetivo: Maximizar la Verosimilitud

```
DERIVACION DEL ELBO
===================

Queremos maximizar log p(x), la log-verosimilitud de los datos.

Problema: p(x) = ∫ p(x|z)p(z) dz  es INTRATABLE

Solucion: Usar una distribucion aproximada q(z|x) ≈ p(z|x)


PASO 1: Identidad basica
────────────────────────

log p(x) = log p(x) · ∫ q(z|x) dz      (integrando 1)
         = ∫ q(z|x) log p(x) dz
         = ∫ q(z|x) log [p(x,z)/p(z|x)] dz

         = ∫ q(z|x) log [p(x,z) · q(z|x) / (p(z|x) · q(z|x))] dz

         = ∫ q(z|x) log [p(x,z)/q(z|x)] dz + ∫ q(z|x) log [q(z|x)/p(z|x)] dz

         = E_q[log p(x,z) - log q(z|x)] + KL(q(z|x) || p(z|x))

         = ELBO + KL(q||p)


PASO 2: Reordenando
───────────────────

log p(x) = ELBO + KL(q(z|x) || p(z|x))

Como KL ≥ 0:

    log p(x) ≥ ELBO

El ELBO es un LIMITE INFERIOR de log p(x).
Maximizar ELBO ≈ Maximizar log p(x)


PASO 3: Descomposicion del ELBO
───────────────────────────────

ELBO = E_q[log p(x,z) - log q(z|x)]

     = E_q[log p(x|z) + log p(z) - log q(z|x)]

     = E_q[log p(x|z)] - E_q[log q(z|x) - log p(z)]

     = E_q[log p(x|z)] - KL(q(z|x) || p(z))

       ─────────────     ──────────────────
       Reconstruccion    Regularizacion


INTERPRETACION:
───────────────

                    ELBO
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    Reconstruccion           Regularizacion
    E_q[log p(x|z)]         -KL(q(z|x)||p(z))
          │                       │
          │                       │
    "La imagen                "El latent z
     reconstruida              debe parecerse
     debe ser igual            a una gaussiana
     a la original"            N(0, I)"
```

### Calculo Explicito del ELBO

```
ELBO PARA GAUSSIANAS
====================

Asumimos:
    - Prior:     p(z) = N(0, I)
    - Encoder:   q(z|x) = N(μ(x), σ²(x)I)  donde μ, σ son redes neuronales
    - Decoder:   p(x|z) = N(f(z), σ_dec²I)  donde f es red neuronal


TERMINO DE RECONSTRUCCION:
──────────────────────────

E_q[log p(x|z)] = E_q[-||x - f(z)||² / (2σ_dec²) - const]

                ≈ -||x - f(μ + σ⊙ε)||² / (2σ_dec²)

                ≈ -||x - x̂||²  (MSE loss, asumiendo σ_dec=1)


TERMINO DE KL:
──────────────

KL(q(z|x) || p(z)) = KL(N(μ, σ²I) || N(0, I))

Para dos gaussianas univariadas:
    KL(N(μ, σ²) || N(0, 1)) = -½[1 + log(σ²) - μ² - σ²]

Para D dimensiones:
    KL = -½ Σᵢ [1 + log(σᵢ²) - μᵢ² - σᵢ²]


LOSS FINAL:
───────────

L(θ, φ) = -ELBO
        = ||x - x̂||² + β · KL(q||p)

Donde:
    - θ: parametros del decoder
    - φ: parametros del encoder
    - β: peso del termino KL (β=1 para VAE estandar)


DIAGRAMA DE LA LOSS:
────────────────────

    x ──────┐
            │
            ▼
    ┌───────────────┐      ┌─────┐
    │    Encoder    │──────│  μ  │───┐
    │     q_φ       │      └─────┘   │
    └───────────────┘      ┌─────┐   │
                           │log σ│───┤
                           └─────┘   │
                                     │
                              ┌──────┴──────┐
                              │ z = μ + σ⊙ε │
                              │   (sample)  │
                              └──────┬──────┘
                                     │
                                     ▼
                           ┌───────────────┐
    x̂ ◀───────────────────│    Decoder    │
                           │     p_θ       │
                           └───────────────┘


    Loss = MSE(x, x̂) + β · KL(N(μ,σ²) || N(0,1))
           ──────────   ────────────────────────
           Reconstruir     Regularizar latent
```

---

## Reparameterization Trick

```
REPARAMETERIZATION TRICK
========================

Problema: ¿Como backpropagar a traves de z ~ q(z|x)?

El sampling es una operacion ESTOCASTICA y no diferenciable.


SOLUCION: Reparametrizacion
───────────────────────────

En lugar de:
    z ~ N(μ, σ²)

Escribimos:
    ε ~ N(0, 1)
    z = μ + σ ⊙ ε

El gradiente ahora puede fluir:

    ∂L/∂μ = ∂L/∂z · ∂z/∂μ = ∂L/∂z · 1
    ∂L/∂σ = ∂L/∂z · ∂z/∂σ = ∂L/∂z · ε


VISUALIZACION:
──────────────

Sin reparametrizacion:              Con reparametrizacion:

     μ, σ²                               μ, σ
       │                                  │    ε ~ N(0,1)
       ▼                                  │    │
   ┌───────┐                              │    │
   │Sample │◀──── No gradiente!           │    ▼
   │z~N(μ,σ)│                            z = μ + σ⊙ε
   └───┬───┘                                  │
       │                                      │
       ▼                                      ▼
    Decoder                              Decoder
       │                                      │
       ▼                                      ▼
      Loss                                  Loss
       │                                      │
       ✗ No backprop                      ✓ Backprop OK!


CODIGO:
───────

# Sin reparametrizacion (NO funciona para training)
z = torch.normal(mu, sigma)  # No tiene gradiente

# Con reparametrizacion (SI funciona)
eps = torch.randn_like(mu)
z = mu + sigma * eps  # Gradientes fluyen a mu y sigma
```

---

## Implementacion Completa de VAE

```python
"""
VAE: Variational Autoencoder
Implementacion completa con PyTorch.
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class VAEConfig:
    """Configuracion del VAE."""

    input_dim: int = 784  # MNIST: 28*28
    hidden_dims: Tuple[int, ...] = (512, 256)
    latent_dim: int = 20
    beta: float = 1.0  # Peso del KL
    learning_rate: float = 1e-3


class VAEEncoder(nn.Module):
    """
    Encoder: x -> (μ, log σ²)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        latent_dim: int,
    ):
        super().__init__()

        # Build encoder MLP
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Output layers for μ and log σ²
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input [B, input_dim]

        Returns:
            mu: Media [B, latent_dim]
            logvar: Log varianza [B, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder: z -> x̂
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
    ):
        super().__init__()

        # Build decoder MLP (reverse of encoder)
        layers = []
        in_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(hidden_dims[0], output_dim))
        layers.append(nn.Sigmoid())  # Output en [0, 1]

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent [B, latent_dim]

        Returns:
            x_recon: Reconstruccion [B, output_dim]
        """
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder completo.
    """

    def __init__(self, config: VAEConfig):
        super().__init__()

        self.config = config
        self.latent_dim = config.latent_dim
        self.beta = config.beta

        self.encoder = VAEEncoder(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
        )

        self.decoder = VAEDecoder(
            config.latent_dim,
            config.hidden_dims,
            config.input_dim,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick: z = μ + σ ⊙ ε

        Args:
            mu: Media [B, latent_dim]
            logvar: Log varianza [B, latent_dim]

        Returns:
            z: Muestra del latent [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass completo.

        Args:
            x: Input [B, input_dim]

        Returns:
            x_recon: Reconstruccion [B, input_dim]
            mu: Media [B, latent_dim]
            logvar: Log varianza [B, latent_dim]
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss_function(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calcula la loss del VAE (negativo del ELBO).

        Loss = Recon_Loss + β * KL_Loss

        Args:
            x: Input original
            x_recon: Reconstruccion
            mu: Media del encoder
            logvar: Log varianza del encoder

        Returns:
            total_loss: Loss total
            recon_loss: Loss de reconstruccion
            kl_loss: Divergencia KL
        """
        # Reconstruction loss (BCE para imagenes binarias)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples: int, device: str = "cuda") -> Tensor:
        """
        Genera nuevas muestras desde el prior p(z) = N(0, I).

        Args:
            num_samples: Numero de muestras
            device: Dispositivo

        Returns:
            Muestras generadas [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Reconstruye inputs (encode + decode).
        """
        mu, _ = self.encoder(x)
        x_recon = self.decoder(mu)
        return x_recon

    @torch.no_grad()
    def interpolate(
        self,
        x1: Tensor,
        x2: Tensor,
        num_steps: int = 10,
    ) -> Tensor:
        """
        Interpola entre dos puntos en el espacio latente.

        Args:
            x1, x2: Imagenes a interpolar [1, input_dim]
            num_steps: Numero de pasos de interpolacion

        Returns:
            Interpolaciones [num_steps, input_dim]
        """
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)

        # Interpolacion lineal en espacio latente
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        z_interp = torch.stack([
            mu1 * (1 - alpha) + mu2 * alpha
            for alpha in alphas
        ])

        return self.decoder(z_interp.squeeze(1))


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    config: VAEConfig,
    num_epochs: int = 50,
    device: str = "cuda",
) -> List[float]:
    """
    Entrena el VAE.

    Args:
        model: Modelo VAE
        train_loader: DataLoader
        config: Configuracion
        num_epochs: Epocas
        device: Dispositivo

    Returns:
        Lista de losses por epoca
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Obtener datos
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.view(x.size(0), -1).to(device)

            # Forward
            x_recon, mu, logvar = model(x)

            # Loss
            loss, recon_loss, kl_loss = model.loss_function(x, x_recon, mu, logvar)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracking
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item()/len(x):.2f}",
                "recon": f"{recon_loss.item()/len(x):.2f}",
                "kl": f"{kl_loss.item()/len(x):.2f}",
            })

        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
              f"Recon={total_recon/len(train_loader.dataset):.4f}, "
              f"KL={total_kl/len(train_loader.dataset):.4f}")

    return losses


def visualize_latent_space(
    model: VAE,
    test_loader: DataLoader,
    device: str = "cuda",
):
    """Visualiza el espacio latente 2D (solo si latent_dim=2)."""

    if model.latent_dim != 2:
        print("Visualizacion solo disponible para latent_dim=2")
        return

    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.view(x.size(0), -1).to(device)
            mu, _ = model.encoder(x)
            zs.append(mu.cpu())
            labels.append(y)

    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("VAE Latent Space")
    plt.savefig("vae_latent_space.png", dpi=150)
    plt.show()


# =============================================================================
# EJEMPLO CON MNIST
# =============================================================================

def demo_vae_mnist():
    """Demo de VAE con MNIST."""
    from torchvision import datasets, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Modelo
    config = VAEConfig(
        input_dim=784,
        hidden_dims=(512, 256),
        latent_dim=20,
        beta=1.0,
        learning_rate=1e-3,
    )

    model = VAE(config)

    # Entrenar
    losses = train_vae(model, train_loader, config, num_epochs=20, device=device)

    # Generar muestras
    model.eval()
    samples = model.sample(64, device=device)
    samples = samples.view(64, 1, 28, 28).cpu()

    # Visualizar
    from torchvision.utils import make_grid, save_image
    grid = make_grid(samples, nrow=8, padding=2)
    save_image(grid, "vae_samples.png")

    print("Muestras guardadas en vae_samples.png")

    return model


if __name__ == "__main__":
    demo_vae_mnist()
```

---

## Beta-VAE: Control del Disentanglement

```
BETA-VAE: REPRESENTACIONES DISENTANGLED
=======================================

β-VAE es una variante que usa β > 1 en el termino KL
para forzar representaciones mas "disentangled".


QUE ES DISENTANGLEMENT?
───────────────────────

Una representacion es "disentangled" cuando cada dimension
del latent controla UN factor de variacion independiente.

Ejemplo con caras:

Entangled (VAE normal):              Disentangled (β-VAE):
──────────────────────               ─────────────────────

z₁: ???                              z₁: Pose (izq/der)
z₂: ???                              z₂: Sonrisa
z₃: ???                              z₃: Color de pelo
z₄: ???                              z₄: Edad

Cambiar z₁ afecta                    Cambiar z₁ SOLO
multiples atributos                  cambia la pose


EFECTO DE β:
────────────

Loss = Recon + β · KL

β = 1 (VAE estandar):
    - Balance reconstruccion/regularizacion
    - Latents pueden estar entangled

β > 1 (β-VAE):
    - Mas peso en KL
    - Fuerza latents hacia N(0,I)
    - Cada dimension mas independiente
    - Pero: peor reconstruccion


VISUALIZACION DEL TRADE-OFF:
────────────────────────────

    Calidad                                  Disentanglement
    Recon ↑                                        ↑
        │                                          │
        │    ●                                     │              ●
        │      ●                                   │            ●
        │        ●                                 │          ●
        │          ●                               │        ●
        │            ●                             │      ●
        │              ●                           │    ●
        │                ●                         │  ●
        └─────────────────────▶ β           └─────────────────────▶ β
              β optimo ≈ 4-10                     β alto = mejor


TRAVERSAL DE LATENTS:
─────────────────────

Para verificar disentanglement, variamos una dimension:

z₁ (pose):
    ┌───┬───┬───┬───┬───┐
    │ ◀ │ ◁ │ ○ │ ▷ │ ▶ │  <- Solo rota la cara
    └───┴───┴───┴───┴───┘
    -2   -1   0   +1  +2

z₂ (sonrisa):
    ┌───┬───┬───┬───┬───┐
    │ 😐 │ 🙂 │ 😊 │ 😄 │ 😁 │  <- Solo cambia sonrisa
    └───┴───┴───┴───┴───┘
    -2   -1   0   +1  +2
```

### Implementacion Beta-VAE

```python
"""
Beta-VAE para representaciones disentangled.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class BetaVAE(nn.Module):
    """
    Beta-VAE: VAE con beta > 1 para disentanglement.

    Loss = Recon + β · KL
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 10,
        beta: float = 4.0,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder CNN
        encoder_layers = []
        in_ch = input_channels

        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_ch = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Flatten size (asumiendo input 64x64)
        self.flatten_size = hidden_dims[-1] * 4 * 4

        # Latent projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        decoder_layers = []
        hidden_dims_rev = list(reversed(hidden_dims))

        for i in range(len(hidden_dims_rev) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    hidden_dims_rev[i], hidden_dims_rev[i+1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dims_rev[i+1]),
                nn.LeakyReLU(0.2),
            ])

        decoder_layers.extend([
            nn.ConvTranspose2d(
                hidden_dims_rev[-1], input_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input a distribucion latente."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent a imagen."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), -1, 4, 4)
        return self.decoder(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(
        self,
        x: Tensor,
        recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Beta-VAE loss.
        """
        # Reconstruction (MSE para imagenes continuas)
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total con beta
        total = recon_loss + self.beta * kl_loss

        return total, recon_loss, kl_loss

    @torch.no_grad()
    def traverse_latent(
        self,
        x: Tensor,
        dim: int,
        range_val: float = 3.0,
        num_steps: int = 11,
    ) -> Tensor:
        """
        Genera traversal de una dimension latente.

        Args:
            x: Imagen de referencia [1, C, H, W]
            dim: Dimension a variar
            range_val: Rango de variacion [-range, +range]
            num_steps: Numero de pasos

        Returns:
            Traversal images [num_steps, C, H, W]
        """
        mu, _ = self.encode(x)
        mu = mu.repeat(num_steps, 1)

        # Variar dimension especifica
        values = torch.linspace(-range_val, range_val, num_steps, device=x.device)
        mu[:, dim] = values

        return self.decode(mu)
```

---

## VQ-VAE: Vector Quantized VAE

```
VQ-VAE: VARIATIONAL AUTOENCODER CUANTIZADO
==========================================

VQ-VAE reemplaza el espacio latente continuo por uno DISCRETO,
usando un codebook de vectores aprendidos.


DIFERENCIA CON VAE:
───────────────────

VAE:
    x → Encoder → z ∈ ℝᵈ (continuo) → Decoder → x̂
                   │
                   └─ z ~ N(μ, σ²)

VQ-VAE:
    x → Encoder → z_e → Quantize → z_q ∈ Codebook → Decoder → x̂
                          │
                          └─ z_q = argmin ||z_e - e_k||
                                    k


CODEBOOK (Diccionario de embeddings):
─────────────────────────────────────

    Codebook E = {e₁, e₂, ..., e_K}

    Cada e_k ∈ ℝᵈ es un "embedding" aprendido.

    ┌─────────────────────────────────────────┐
    │  Codebook (K vectores)                  │
    │                                         │
    │  e₁: [0.2, -0.1, 0.5, ...]            │
    │  e₂: [0.8, 0.3, -0.2, ...]            │
    │  e₃: [-0.1, 0.7, 0.1, ...]            │
    │  ...                                   │
    │  e_K: [0.4, -0.5, 0.3, ...]           │
    └─────────────────────────────────────────┘


PROCESO DE CUANTIZACION:
────────────────────────

1. Encoder produce z_e (continuo)
2. Para cada vector en z_e, encontrar el embedding mas cercano:

       k* = argmin ||z_e - e_k||²
             k

3. Reemplazar z_e por e_{k*}:

       z_q = e_{k*}


Visualizacion 2D:

    ●  ●  ●  ●  ●  ●   <- Codebook embeddings (fijos durante forward)
    │
    │    ×             <- z_e (output del encoder)
    │    │
    │    └───▶ ●       <- z_q (embedding mas cercano)


LOSS FUNCTION:
──────────────

L = L_recon + L_codebook + β · L_commit

Donde:

1. L_recon = ||x - x̂||²
   (Reconstruccion normal)

2. L_codebook = ||sg[z_e] - e||²
   (Mover embeddings hacia encoder outputs)
   sg[] = stop gradient

3. L_commit = ||z_e - sg[e]||²
   (Mover encoder outputs hacia embeddings)
   "Commitment loss"


STRAIGHT-THROUGH ESTIMATOR:
───────────────────────────

Como la cuantizacion no es diferenciable, usamos:

    Forward:  z_q = quantize(z_e)
    Backward: ∂L/∂z_e = ∂L/∂z_q  (copiar gradiente)

En codigo:
    z_q = z_e + (quantize(z_e) - z_e).detach()

    Durante forward: z_q = quantize(z_e)
    Durante backward: gradientes fluyen a z_e directamente
```

### Implementacion VQ-VAE

```python
"""
VQ-VAE: Vector Quantized VAE
Implementacion con PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer.
    Mapea vectores continuos a embeddings discretos del codebook.
    """

    def __init__(
        self,
        num_embeddings: int,  # K: tamano del codebook
        embedding_dim: int,   # D: dimension de cada embedding
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: [K, D]
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

        # Inicializacion uniforme
        self.embeddings.weight.data.uniform_(
            -1.0 / num_embeddings,
            1.0 / num_embeddings
        )

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Cuantiza los inputs.

        Args:
            z: Encoder output [B, D, H, W]

        Returns:
            z_q: Cuantizado [B, D, H, W]
            loss: Codebook + commitment loss
            indices: Indices del codebook [B, H, W]
        """
        # Reshape: [B, D, H, W] -> [B*H*W, D]
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Calcular distancias al codebook: ||z - e||²
        # = ||z||² + ||e||² - 2*z·e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )

        # Encontrar embedding mas cercano
        indices = torch.argmin(distances, dim=1)

        # Obtener embeddings cuantizados
        z_q_flat = self.embeddings(indices)

        # Reshape back: [B*H*W, D] -> [B, D, H, W]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        indices = indices.view(B, H, W)

        # Losses
        # Codebook loss: mover embeddings hacia z_e
        codebook_loss = F.mse_loss(z_q, z.detach())

        # Commitment loss: mover z_e hacia embeddings
        commitment_loss = F.mse_loss(z, z_q.detach())

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """Obtiene embeddings dado indices."""
        return self.embeddings(indices)


class VQVAEEncoder(nn.Module):
    """Encoder para VQ-VAE."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        embedding_dim: int = 64,
    ):
        super().__init__()

        layers = []
        prev_ch = in_channels

        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_ch, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            prev_ch = h_dim

        # Projection to embedding dim
        layers.append(
            nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    """Decoder para VQ-VAE."""

    def __init__(
        self,
        out_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        embedding_dim: int = 64,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1),
            nn.ReLU(inplace=True),
        ]

        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i+1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.ReLU(inplace=True),
            ])

        layers.extend([
            nn.ConvTranspose2d(
                hidden_dims[-1], out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # Output en [-1, 1]
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_q: Tensor) -> Tensor:
        return self.decoder(z_q)


class VQVAE(nn.Module):
    """
    VQ-VAE completo.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        hidden_dims: Tuple[int, ...] = (32, 64, 128),
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.encoder = VQVAEEncoder(
            in_channels, hidden_dims, embedding_dim
        )

        self.quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )

        self.decoder = VQVAEDecoder(
            in_channels, tuple(reversed(hidden_dims)), embedding_dim
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            x_recon: Reconstruccion
            vq_loss: Loss de cuantizacion
            indices: Indices del codebook
        """
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, indices

    def loss_function(
        self,
        x: Tensor,
        x_recon: Tensor,
        vq_loss: Tensor,
    ) -> Tensor:
        """Calcula loss total."""
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss + vq_loss

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a indices discretos."""
        z_e = self.encoder(x)
        _, _, indices = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode desde indices del codebook."""
        B, H, W = indices.shape
        z_q = self.quantizer.get_codebook_entry(indices.view(-1))
        z_q = z_q.view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.decoder(z_q)


# =============================================================================
# VQ-VAE-2: VERSION JERARQUICA
# =============================================================================

class VQVAE2(nn.Module):
    """
    VQ-VAE-2: Version jerarquica con multiples niveles de cuantizacion.

    Estructura:
        - Bottom level: Alta resolucion, detalles finos
        - Top level: Baja resolucion, estructura global
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 128, 256),
        num_embeddings_top: int = 512,
        num_embeddings_bottom: int = 512,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # Bottom encoder (alta resolucion)
        self.encoder_bottom = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], hidden_dims[1], 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[1], embedding_dim, 3, 1, 1),
        )

        # Top encoder (baja resolucion)
        self.encoder_top = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dims[2], 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[2], embedding_dim, 3, 1, 1),
        )

        # Quantizers
        self.quantizer_top = VectorQuantizer(num_embeddings_top, embedding_dim)
        self.quantizer_bottom = VectorQuantizer(num_embeddings_bottom, embedding_dim)

        # Decoder top -> bottom
        self.decoder_top = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dims[2], 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[2], embedding_dim, 4, 2, 1),
        )

        # Decoder bottom
        self.decoder_bottom = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, hidden_dims[1], 3, 1, 1),  # Concat top + bottom
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dims[0], in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Encode bottom
        z_e_bottom = self.encoder_bottom(x)

        # Encode top
        z_e_top = self.encoder_top(z_e_bottom)

        # Quantize top
        z_q_top, loss_top, indices_top = self.quantizer_top(z_e_top)

        # Decode top
        z_top_decoded = self.decoder_top(z_q_top)

        # Quantize bottom (condicionado en top)
        z_q_bottom, loss_bottom, indices_bottom = self.quantizer_bottom(z_e_bottom)

        # Decode (concatenar top y bottom)
        z_combined = torch.cat([z_top_decoded, z_q_bottom], dim=1)
        x_recon = self.decoder_bottom(z_combined)

        total_vq_loss = loss_top + loss_bottom

        return x_recon, total_vq_loss, indices_top, indices_bottom
```

---

## Aplicaciones de VAEs

```
APLICACIONES DE VAEs
====================

1. GENERACION DE DATOS
──────────────────────

   z ~ N(0, I) ──▶ Decoder ──▶ Nueva muestra

   Aplicaciones:
   - Generacion de imagenes
   - Generacion de moleculas (drug discovery)
   - Sintesis de audio


2. COMPRESION APRENDIDA
───────────────────────

   Original              Comprimido           Reconstruido
   ┌───────┐            ┌───────┐            ┌───────┐
   │       │  Encode    │indices│   Decode   │       │
   │ 256KB │ ────────▶  │  2KB  │ ────────▶  │ ~256KB│
   │       │            │       │            │       │
   └───────┘            └───────┘            └───────┘

   VQ-VAE es especialmente util (indices discretos)


3. DETECCION DE ANOMALIAS
─────────────────────────

   VAE entrenado en datos normales.
   Anomalias tienen alta loss de reconstruccion.

   Normal:    x ──▶ VAE ──▶ x̂    ||x - x̂|| = bajo
   Anomalia:  x ──▶ VAE ──▶ x̂    ||x - x̂|| = ALTO

   Aplicaciones:
   - Deteccion de fraude
   - Monitoreo de sistemas
   - Control de calidad


4. INTERPOLACION EN ESPACIO LATENTE
───────────────────────────────────

   x₁         Interpolacion         x₂
   ┌─┐    ┌─┐  ┌─┐  ┌─┐  ┌─┐    ┌─┐
   │A│    │ │  │ │  │ │  │ │    │B│
   └─┘    └─┘  └─┘  └─┘  └─┘    └─┘
    │      │    │    │    │      │
    └──────┴────┴────┴────┴──────┘
                 ▲
           Transicion suave


5. EDICION SEMANTICA
────────────────────

   Encontrar direcciones en el espacio latente:

   z_original + α · direction_sonrisa = z_sonriente

   Direcciones aprendidas o encontradas:
   - Edad
   - Emocion
   - Iluminacion
   - Estilo


6. DATOS FALTANTES / INPAINTING
───────────────────────────────

   Imagen              Mascara            Completada
   ┌───────┐          ┌───────┐          ┌───────┐
   │  🖼️   │    +     │ ████  │    =     │  🖼️   │
   │  ???  │          │       │          │  OK!  │
   └───────┘          └───────┘          └───────┘

   VAE puede "adivinar" partes faltantes
```

### Ejemplo: Deteccion de Anomalias

```python
"""
Deteccion de anomalias con VAE.
"""

import torch
import numpy as np
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader


class VAEAnomalyDetector:
    """
    Detector de anomalias basado en VAE.

    Una muestra es anomala si su loss de reconstruccion
    excede un umbral aprendido de datos normales.
    """

    def __init__(
        self,
        vae: torch.nn.Module,
        device: str = "cuda",
    ):
        self.vae = vae.to(device)
        self.vae.eval()
        self.device = device

        self.threshold: float = 0.0
        self.mean_loss: float = 0.0
        self.std_loss: float = 1.0

    @torch.no_grad()
    def compute_reconstruction_loss(self, x: Tensor) -> Tensor:
        """Calcula loss de reconstruccion por muestra."""
        x = x.to(self.device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        x_flat = x.view(x.size(0), -1)
        x_recon, mu, logvar = self.vae(x_flat)

        # Loss por muestra (no promediado)
        loss = torch.sum((x_flat - x_recon) ** 2, dim=1)

        return loss

    def fit_threshold(
        self,
        normal_loader: DataLoader,
        percentile: float = 95.0,
    ):
        """
        Ajusta el umbral usando datos normales.

        Args:
            normal_loader: DataLoader con datos normales
            percentile: Percentil para el umbral
        """
        all_losses = []

        for batch in normal_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            losses = self.compute_reconstruction_loss(x)
            all_losses.append(losses.cpu())

        all_losses = torch.cat(all_losses).numpy()

        self.mean_loss = np.mean(all_losses)
        self.std_loss = np.std(all_losses)
        self.threshold = np.percentile(all_losses, percentile)

        print(f"Threshold ajustado: {self.threshold:.4f}")
        print(f"Mean loss: {self.mean_loss:.4f}, Std: {self.std_loss:.4f}")

    @torch.no_grad()
    def detect(
        self,
        x: Tensor,
        return_scores: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Detecta anomalias.

        Args:
            x: Datos a evaluar
            return_scores: Si retornar scores normalizados

        Returns:
            is_anomaly: Boolean tensor
            scores: Anomaly scores (si return_scores=True)
        """
        losses = self.compute_reconstruction_loss(x)

        # Normalizar scores
        scores = (losses - self.mean_loss) / self.std_loss

        # Detectar anomalias
        is_anomaly = losses > self.threshold

        if return_scores:
            return is_anomaly, scores.cpu()

        return is_anomaly


def demo_anomaly_detection():
    """Demo de deteccion de anomalias con MNIST."""
    from torchvision import datasets, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar modelo preentrenado (o entrenar uno)
    config = VAEConfig(input_dim=784, latent_dim=20)
    vae = VAE(config)
    # vae.load_state_dict(torch.load("vae_mnist.pt"))

    # Crear detector
    detector = VAEAnomalyDetector(vae, device)

    # Dataset: digitos 0-4 son "normales", 5-9 son "anomalias"
    transform = transforms.ToTensor()

    full_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Separar normales y anomalias
    normal_indices = [i for i, (_, y) in enumerate(full_dataset) if y < 5]
    anomaly_indices = [i for i, (_, y) in enumerate(full_dataset) if y >= 5]

    normal_subset = torch.utils.data.Subset(full_dataset, normal_indices[:1000])
    anomaly_subset = torch.utils.data.Subset(full_dataset, anomaly_indices[:1000])

    normal_loader = DataLoader(normal_subset, batch_size=64)

    # Ajustar umbral
    detector.fit_threshold(normal_loader, percentile=95)

    # Evaluar
    test_normal = next(iter(DataLoader(normal_subset, batch_size=100)))[0]
    test_anomaly = next(iter(DataLoader(anomaly_subset, batch_size=100)))[0]

    is_anom_normal, scores_normal = detector.detect(test_normal, return_scores=True)
    is_anom_anomaly, scores_anomaly = detector.detect(test_anomaly, return_scores=True)

    print(f"\nResultados:")
    print(f"Normales detectados como anomalia: {is_anom_normal.float().mean()*100:.1f}%")
    print(f"Anomalias detectadas: {is_anom_anomaly.float().mean()*100:.1f}%")

    return detector
```

---

## Comparacion de Variantes

```
COMPARACION: VAE vs β-VAE vs VQ-VAE vs VQ-VAE-2
===============================================

┌────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ Caracteristica │    VAE      │   β-VAE     │   VQ-VAE    │  VQ-VAE-2   │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Espacio latente│ Continuo    │ Continuo    │ Discreto    │ Discreto    │
│                │ N(μ, σ²)    │ N(μ, σ²)    │ Codebook    │ jerarquico  │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Regularizacion │ KL(q||p)    │ β·KL(q||p)  │ VQ loss     │ VQ loss     │
│                │ β=1         │ β>1         │ + commit    │ multi-nivel │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Calidad recon  │ Media       │ Baja-Media  │ Alta        │ Muy alta    │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Disentangle    │ Bajo        │ Alto        │ N/A         │ N/A         │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Generacion     │ z~N(0,I)    │ z~N(0,I)    │ Prior       │ Prior       │
│                │             │             │ autoregres. │ autoregres. │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Compresion     │ Flotantes   │ Flotantes   │ Indices     │ Indices     │
│                │             │             │ discretos   │ jerarquicos │
├────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ Uso tipico     │ Generativo  │ Disentangle │ Compresion  │ Alta calidad│
│                │ basico      │ representa. │ generacion  │ imagenes    │
└────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘


CUANDO USAR CADA UNO:
─────────────────────

VAE:
    ✓ Punto de partida simple
    ✓ Cuando necesitas interpolacion suave
    ✓ Deteccion de anomalias

β-VAE:
    ✓ Cuando necesitas representaciones interpretables
    ✓ Edicion semantica controlada
    ✓ Transferencia de atributos

VQ-VAE:
    ✓ Compresion de alta fidelidad
    ✓ Generacion con autoregressive prior (PixelCNN)
    ✓ Audio (mejor que VAE continuo)

VQ-VAE-2:
    ✓ Imagenes de alta resolucion
    ✓ Maxima calidad de reconstruccion
    ✓ Generacion state-of-the-art (antes de diffusion)
```

---

## Resumen

```
CONCEPTOS CLAVE - VAEs AVANZADOS
================================

1. FUNDAMENTOS VAE
   - ELBO = Reconstruccion - KL
   - Reparameterization trick para gradientes
   - Espacio latente continuo y estructurado

2. β-VAE
   - β > 1 fuerza disentanglement
   - Trade-off: disentangle vs reconstruccion
   - Util para representaciones interpretables

3. VQ-VAE
   - Espacio latente DISCRETO (codebook)
   - No KL, usa commitment + codebook loss
   - Straight-through estimator para gradientes
   - Mejor calidad de reconstruccion

4. VQ-VAE-2
   - Jerarquico: top (global) + bottom (detalles)
   - Excelente para imagenes alta resolucion
   - Base para modelos generativos pre-diffusion

5. APLICACIONES
   - Generacion de datos
   - Compresion aprendida
   - Deteccion de anomalias
   - Interpolacion/edicion semantica


FLUJO DE INFORMACION:
─────────────────────

VAE:     x → μ,σ → z=μ+σε → x̂
β-VAE:   x → μ,σ → z=μ+σε → x̂  (β>1 en KL)
VQ-VAE:  x → z_e → quantize(z_e) → x̂
VQ-VAE2: x → z_e → [z_top, z_bot] → x̂


EVOLUCION HISTORICA:
────────────────────

2013: VAE (Kingma & Welling)
2017: β-VAE (Higgins et al.)
2017: VQ-VAE (van den Oord et al.)
2019: VQ-VAE-2 (Razavi et al.)
2020: Diffusion Models superan VAEs en generacion
2022: VAE sigue siendo util para:
      - Componente de Stable Diffusion (latent space)
      - Deteccion de anomalias
      - Representaciones
```

---

## Referencias

1. Kingma, D.P. & Welling, M. (2013). "Auto-Encoding Variational Bayes"
2. Higgins, I., et al. (2017). "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
3. van den Oord, A., et al. (2017). "Neural Discrete Representation Learning" (VQ-VAE)
4. Razavi, A., et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2"
5. Burgess, C.P., et al. (2018). "Understanding disentangling in β-VAE"
