# GANs y Generación de Imágenes

## Introducción

Las Generative Adversarial Networks (GANs) son modelos generativos que aprenden a crear datos nuevos similares a los de entrenamiento. Consisten en dos redes que compiten: un Generador que crea imágenes falsas y un Discriminador que intenta distinguir reales de falsas.

```
Arquitectura GAN:

         Ruido z                         Imagen Real
            │                                 │
            ▼                                 │
    ┌───────────────┐                        │
    │   Generador   │                        │
    │      (G)      │                        │
    └───────┬───────┘                        │
            │                                 │
            ▼                                 ▼
    Imagen Generada            ┌───────────────────────┐
            │                  │    Discriminador      │
            └──────────────────►        (D)           │
                               └───────────┬───────────┘
                                           │
                                           ▼
                                    Real / Fake
                                    (0 - 1)

Juego min-max:
- G intenta engañar a D (maximizar errores de D)
- D intenta no ser engañado (minimizar errores)

min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]
```

## GAN Básica

### Implementación

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


class Generator(nn.Module):
    """
    Generador: Transforma ruido aleatorio en imágenes.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        img_shape: Tuple[int, int, int] = (1, 28, 28)
    ):
        super().__init__()

        self.img_shape = img_shape
        img_size = np.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, img_size),
            nn.Tanh()  # Output en [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """
    Discriminador: Clasifica imágenes como reales o falsas.
    """

    def __init__(self, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super().__init__()

        img_size = np.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilidad [0, 1]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class VanillaGAN:
    """
    GAN básica con entrenamiento adversarial.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        img_shape: Tuple[int, int, int] = (1, 28, 28),
        device: str = "cuda"
    ):
        self.latent_dim = latent_dim
        self.device = device

        self.generator = Generator(latent_dim, img_shape).to(device)
        self.discriminator = Discriminator(img_shape).to(device)

        self.criterion = nn.BCELoss()

        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

    def train_step(self, real_images: torch.Tensor) -> dict:
        """Un paso de entrenamiento."""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()

        # Loss con imágenes reales
        real_output = self.discriminator(real_images)
        d_loss_real = self.criterion(real_output, real_labels)

        # Loss con imágenes falsas
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        self.optimizer_G.zero_grad()

        # G quiere que D clasifique fakes como reales
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(fake_output, real_labels)

        g_loss.backward()
        self.optimizer_G.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_real": real_output.mean().item(),
            "d_fake": fake_output.mean().item()
        }

    @torch.no_grad()
    def generate(self, num_samples: int = 16) -> torch.Tensor:
        """Genera nuevas imágenes."""
        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        return fake_images.cpu()
```

## DCGAN (Deep Convolutional GAN)

### Arquitectura con Convoluciones

```
DCGAN: Usa convoluciones en lugar de fully connected.

Generator:
z (100)
    │
    ▼ Dense
(4×4×512)
    │
    ▼ ConvTranspose 4×4, s=2
(8×8×256)
    │
    ▼ ConvTranspose 4×4, s=2
(16×16×128)
    │
    ▼ ConvTranspose 4×4, s=2
(32×32×64)
    │
    ▼ ConvTranspose 4×4, s=2
(64×64×3)

Discriminator:
(64×64×3)
    │
    ▼ Conv 4×4, s=2
(32×32×64)
    │
    ▼ Conv 4×4, s=2
(16×16×128)
    │
    ▼ Conv 4×4, s=2
(8×8×256)
    │
    ▼ Conv 4×4, s=2
(4×4×512)
    │
    ▼ Dense
Real/Fake
```

```python
class DCGANGenerator(nn.Module):
    """
    DCGAN Generator usando convoluciones transpuestas.

    Mejoras sobre GAN básica:
    - BatchNorm (excepto output)
    - ReLU en G, LeakyReLU en D
    - Convoluciones en lugar de FC
    """

    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        channels: int = 3
    ):
        super().__init__()

        self.main = nn.Sequential(
            # Input: latent_dim → 4×4×(feature_maps*8)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # 4×4 → 8×8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16×16 → 32×32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim) → (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator."""

    def __init__(
        self,
        feature_maps: int = 64,
        channels: int = 3
    ):
        super().__init__()

        self.main = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32×32 → 16×16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16×16 → 8×8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8×8 → 4×4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4×4 → 1×1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.main(img).view(-1, 1)
```

## Conditional GAN (cGAN)

```python
class ConditionalGenerator(nn.Module):
    """
    Generator condicionado por etiqueta de clase.
    Permite controlar qué tipo de imagen generar.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embed_dim: int = 50,
        feature_maps: int = 64,
        channels: int = 3
    ):
        super().__init__()

        # Embedding para la condición
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        input_dim = latent_dim + embed_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed labels
        label_embed = self.label_embedding(labels)

        # Concatenar z y label embedding
        gen_input = torch.cat([z, label_embed], dim=1)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)

        return self.main(gen_input)


class ConditionalDiscriminator(nn.Module):
    """Discriminator condicionado."""

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 50,
        feature_maps: int = 64,
        channels: int = 3,
        img_size: int = 64
    ):
        super().__init__()

        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)

        self.main = nn.Sequential(
            nn.Conv2d(channels + 1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed label como canal adicional
        label_embed = self.label_embedding(labels)
        label_channel = label_embed.view(label_embed.size(0), 1, self.img_size, self.img_size)

        # Concatenar imagen y label channel
        d_input = torch.cat([img, label_channel], dim=1)

        return self.main(d_input).view(-1, 1)


class ConditionalGAN:
    """Entrenamiento de cGAN."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        device: str = "cuda"
    ):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

        self.generator = ConditionalGenerator(latent_dim, num_classes).to(device)
        self.discriminator = ConditionalDiscriminator(num_classes).to(device)

        self.criterion = nn.BCELoss()

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    @torch.no_grad()
    def generate_by_class(self, class_id: int, num_samples: int = 16) -> torch.Tensor:
        """Genera imágenes de una clase específica."""
        self.generator.eval()

        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)

        fake_images = self.generator(z, labels)
        return fake_images.cpu()
```

## StyleGAN

### Conceptos Clave

```
StyleGAN: Control de estilo en múltiples niveles.

                    Mapping Network
z (512) ──────────► [FC × 8] ──────────► w (512)

                          w
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
     ▼                    ▼                    ▼
  AdaIN              AdaIN                AdaIN
     │                    │                    │
┌────┴────┐          ┌────┴────┐          ┌────┴────┐
│ 4×4     │          │ 8×8     │          │ 64×64   │
│ Coarse  │    ───►  │ Middle  │    ───►  │ Fine    │
│ features│          │ features│          │ details │
└─────────┘          └─────────┘          └─────────┘

- Coarse (4×4 - 8×8): Pose, cara general
- Middle (16×16 - 32×32): Features faciales
- Fine (64×64+): Colores, texturas finas
```

```python
class MappingNetwork(nn.Module):
    """
    Mapping Network: Transforma z a espacio W más desenredado.
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        num_layers: int = 8
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Inyecta estilo en las features.
    """

    def __init__(self, channels: int, w_dim: int = 512):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Normalizar
        x = self.instance_norm(x)

        # Aplicar estilo
        style_scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        style_bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)

        return style_scale * x + style_bias


class StyleBlock(nn.Module):
    """Bloque de síntesis con inyección de estilo."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int = 512,
        upsample: bool = True
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else None
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)

        # Noise injection
        self.noise_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = self.upsample(x)

        x = self.conv(x)

        # Add noise
        noise = torch.randn_like(x) * self.noise_scale
        x = x + noise

        x = self.adain(x, w)
        x = self.activation(x)

        return x


class SimplifiedStyleGenerator(nn.Module):
    """
    Versión simplificada de StyleGAN generator.
    """

    def __init__(
        self,
        z_dim: int = 512,
        w_dim: int = 512,
        channels: int = 3
    ):
        super().__init__()

        self.mapping = MappingNetwork(z_dim, w_dim)

        # Constant input
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Style blocks
        self.blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim, upsample=False),  # 4×4
            StyleBlock(512, 512, w_dim),  # 8×8
            StyleBlock(512, 256, w_dim),  # 16×16
            StyleBlock(256, 128, w_dim),  # 32×32
            StyleBlock(128, 64, w_dim),   # 64×64
        ])

        self.to_rgb = nn.Conv2d(64, channels, 1)

    def forward(
        self,
        z: torch.Tensor,
        return_latents: bool = False
    ) -> torch.Tensor:
        batch_size = z.size(0)

        # Mapping
        w = self.mapping(z)

        # Synthesis
        x = self.constant.repeat(batch_size, 1, 1, 1)

        for block in self.blocks:
            x = block(x, w)

        img = self.to_rgb(x)
        img = torch.tanh(img)

        if return_latents:
            return img, w
        return img
```

## Pérdidas Avanzadas

### Wasserstein Loss (WGAN)

```python
class WGANLoss:
    """
    Wasserstein GAN loss.
    Más estable que BCE, sin mode collapse.

    D maximiza: E[D(real)] - E[D(fake)]
    G minimiza: -E[D(fake)]
    """

    @staticmethod
    def discriminator_loss(
        real_output: torch.Tensor,
        fake_output: torch.Tensor
    ) -> torch.Tensor:
        return -(real_output.mean() - fake_output.mean())

    @staticmethod
    def generator_loss(fake_output: torch.Tensor) -> torch.Tensor:
        return -fake_output.mean()


class WGANGPLoss:
    """
    WGAN with Gradient Penalty.
    Añade regularización del gradiente para Lipschitz constraint.
    """

    def __init__(self, lambda_gp: float = 10.0):
        self.lambda_gp = lambda_gp

    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        device: str
    ) -> torch.Tensor:
        """Calcula gradient penalty."""
        batch_size = real_images.size(0)

        # Interpolación aleatoria
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        # Discriminator output
        d_interpolated = discriminator(interpolated)

        # Calcular gradientes
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Penalizar desviación de norma 1
        gp = ((gradient_norm - 1) ** 2).mean()

        return self.lambda_gp * gp


class HingeLoss:
    """
    Hinge Loss: Usada en BigGAN, StyleGAN2.
    """

    @staticmethod
    def discriminator_loss(
        real_output: torch.Tensor,
        fake_output: torch.Tensor
    ) -> torch.Tensor:
        real_loss = torch.relu(1.0 - real_output).mean()
        fake_loss = torch.relu(1.0 + fake_output).mean()
        return real_loss + fake_loss

    @staticmethod
    def generator_loss(fake_output: torch.Tensor) -> torch.Tensor:
        return -fake_output.mean()
```

## Métricas de Evaluación

### FID y IS

```python
from scipy import linalg
from torchvision.models import inception_v3


class GANMetrics:
    """Métricas para evaluar calidad de GANs."""

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Inception para FID/IS
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remover última capa
        self.inception = self.inception.to(device)
        self.inception.eval()

    @torch.no_grad()
    def compute_inception_features(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Extrae features de Inception para imágenes."""
        # Resize a 299×299 para Inception
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear')

        # Normalizar
        images = (images + 1) / 2  # [-1, 1] → [0, 1]

        features = self.inception(images.to(self.device))
        return features.cpu()

    def compute_fid(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor
    ) -> float:
        """
        Fréchet Inception Distance (FID).
        Menor es mejor. FID=0 significa distribuciones idénticas.

        FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^0.5)
        """
        real_features = real_features.numpy()
        fake_features = fake_features.numpy()

        # Media y covarianza
        mu_r = real_features.mean(axis=0)
        mu_f = fake_features.mean(axis=0)
        sigma_r = np.cov(real_features, rowvar=False)
        sigma_f = np.cov(fake_features, rowvar=False)

        # FID
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)

        return float(fid)

    def compute_inception_score(
        self,
        fake_images: torch.Tensor,
        splits: int = 10
    ) -> Tuple[float, float]:
        """
        Inception Score (IS).
        Mayor es mejor. Mide calidad y diversidad.

        IS = exp(E[KL(p(y|x) || p(y))])
        """
        # Obtener predicciones de Inception
        self.inception.fc = nn.Linear(2048, 1000)  # Restaurar FC
        self.inception = self.inception.to(self.device)

        images = nn.functional.interpolate(fake_images, size=(299, 299), mode='bilinear')
        images = (images + 1) / 2

        preds = []
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(self.device)
            pred = nn.functional.softmax(self.inception(batch), dim=1)
            preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0).numpy()

        # Calcular IS por splits
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits)]
            py = np.mean(part, axis=0)
            scores = []
            for p in part:
                scores.append(np.sum(p * (np.log(p + 1e-10) - np.log(py + 1e-10))))
            split_scores.append(np.exp(np.mean(scores)))

        return float(np.mean(split_scores)), float(np.std(split_scores))
```

## Detección de Deepfakes

```python
class DeepfakeDetector(nn.Module):
    """
    Detector de deepfakes basado en CNN.
    Detecta imágenes generadas por GANs.
    """

    def __init__(self, backbone: str = "efficientnet"):
        super().__init__()

        from torchvision.models import efficientnet_b0

        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Real / Fake
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def detect(self, image: torch.Tensor) -> dict:
        """Detecta si imagen es real o generada."""
        self.eval()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        logits = self(image)
        probs = torch.softmax(logits, dim=1)[0]

        return {
            "is_fake": probs[1] > 0.5,
            "fake_probability": probs[1].item(),
            "real_probability": probs[0].item(),
            "confidence": max(probs[0].item(), probs[1].item())
        }


class FrequencyAnalysisDetector:
    """
    Detector de deepfakes basado en análisis de frecuencia.
    GANs dejan artefactos en el dominio de frecuencia.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_fft_features(image: np.ndarray) -> np.ndarray:
        """Extrae features del espectro de Fourier."""
        import cv2

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # FFT
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Features del espectro
        # Los deepfakes tienen patrones específicos en frecuencias altas
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Dividir en anillos de frecuencia
        features = []
        for radius in [10, 20, 40, 80, 160]:
            mask = np.zeros_like(magnitude)
            cv2.circle(mask, (center_w, center_h), radius, 1, -1)
            ring = magnitude * mask
            features.append(ring.sum())

        return np.array(features)

    def analyze(self, image: np.ndarray) -> dict:
        """Analiza imagen para detectar deepfake."""
        features = self.compute_fft_features(image)

        # Heurística simple: deepfakes tienen menos energía en altas frecuencias
        high_freq_ratio = features[-1] / (features[0] + 1e-10)

        # Umbral empírico
        is_suspicious = high_freq_ratio < 0.3

        return {
            "frequency_features": features.tolist(),
            "high_freq_ratio": high_freq_ratio,
            "is_suspicious": is_suspicious,
            "analysis": "Posible deepfake - patrón de frecuencia anómalo" if is_suspicious else "Probablemente real"
        }
```

## Aplicaciones en Ciberseguridad

### Generación de Datos de Entrenamiento

```python
class SecurityDataAugmentor:
    """
    Usa GANs para aumentar datasets de seguridad.
    Genera ejemplos sintéticos de amenazas.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        # GAN entrenada en screenshots de phishing, malware, etc.
        self.generator = None

    def augment_phishing_dataset(
        self,
        num_samples: int,
        class_id: int
    ) -> torch.Tensor:
        """Genera screenshots sintéticos de phishing."""
        if self.generator is None:
            raise ValueError("Generator no entrenado")

        return self.generator.generate_by_class(class_id, num_samples)


class AdversarialAttackGenerator:
    """
    Genera imágenes adversariales usando técnicas de GAN.
    Para testing de robustez de modelos.
    """

    def __init__(self, target_model: nn.Module, device: str = "cuda"):
        self.target_model = target_model.to(device)
        self.device = device

    def generate_adversarial(
        self,
        image: torch.Tensor,
        target_class: int,
        epsilon: float = 0.03,
        steps: int = 40
    ) -> torch.Tensor:
        """
        Genera imagen adversarial usando PGD.
        """
        image = image.to(self.device)
        adv_image = image.clone().requires_grad_(True)
        target = torch.tensor([target_class], device=self.device)

        for _ in range(steps):
            self.target_model.zero_grad()

            output = self.target_model(adv_image)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # Paso en dirección del gradiente
            with torch.no_grad():
                adv_image = adv_image - epsilon / steps * adv_image.grad.sign()
                # Clip para mantener en rango válido
                adv_image = torch.clamp(adv_image, image - epsilon, image + epsilon)
                adv_image = torch.clamp(adv_image, 0, 1)

            adv_image.requires_grad_(True)

        return adv_image.detach()


class SyntheticMalwareGenerator:
    """
    Genera visualizaciones de malware sintético.
    Para entrenamiento de detectores.
    """

    def __init__(self, latent_dim: int = 100, device: str = "cuda"):
        self.device = device
        self.latent_dim = latent_dim

        # cGAN para diferentes familias de malware
        self.generator = ConditionalGenerator(
            latent_dim=latent_dim,
            num_classes=5  # 5 familias de malware
        ).to(device)

    def generate_malware_visualization(
        self,
        family: str,
        num_samples: int = 10
    ) -> torch.Tensor:
        """Genera visualizaciones de binarios de malware."""
        family_ids = {
            "trojan": 0,
            "ransomware": 1,
            "worm": 2,
            "adware": 3,
            "rootkit": 4
        }

        class_id = family_ids.get(family, 0)

        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        labels = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)

        with torch.no_grad():
            fake_images = self.generator(z, labels)

        return fake_images.cpu()
```

## Resumen

| GAN | Características | Uso |
|-----|-----------------|-----|
| Vanilla GAN | Básica | Aprendizaje |
| DCGAN | Convolucional | Imágenes pequeñas |
| cGAN | Condicionada | Control de generación |
| WGAN-GP | Estable | Evitar mode collapse |
| StyleGAN | Alta calidad | Caras, arte |
| BigGAN | Gran escala | Diversidad |

### Checklist GANs

```
□ Learning rate bajo (0.0001-0.0002)
□ Adam con β₁=0.5
□ Batch normalization (no en D output)
□ LeakyReLU en D, ReLU en G
□ Monitorear balance D/G
□ FID para evaluar calidad
□ Guardar checkpoints frecuentes
□ Visualizar samples durante training
```

## Referencias

- Generative Adversarial Networks (Goodfellow et al., 2014)
- DCGAN: Unsupervised Representation Learning with Deep Convolutional GANs
- Conditional GANs (Mirza & Osindero)
- Wasserstein GAN
- A Style-Based Generator Architecture for GANs (StyleGAN)
- GANs Trained by Two-Timescale Update Rule Converge (FID)
