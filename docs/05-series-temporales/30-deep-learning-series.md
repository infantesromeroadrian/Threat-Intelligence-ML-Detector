# Deep Learning para Series Temporales

## Introduccion

El **Deep Learning** ofrece modelos potentes para series temporales que pueden capturar patrones complejos y no lineales. Estos modelos son especialmente utiles cuando hay muchos datos y los metodos clasicos no capturan bien las dependencias.

```
EVOLUCION DE MODELOS
====================

Clasicos:              Deep Learning:
─────────              ─────────────

ARIMA                  RNN/LSTM/GRU
  │                         │
  ├── Lineal                ├── No lineal
  ├── Univariado            ├── Multivariado nativo
  ├── Pocos params          ├── Muchos params
  └── Interpretable         └── Black box

ETS                    TCN
  │                         │
  ├── Componentes           ├── Dilatacion
  ├── Suavizado             ├── Convoluciones
  └── Estacionalidad        └── Memoria larga

                       Transformers
                            │
                            ├── Attention
                            ├── Paralelo
                            └── State of the art
```

---

## Arquitecturas LSTM/GRU para Forecasting

### Preparacion de Datos

```
VENTANAS DESLIZANTES (Sliding Windows)
======================================

Serie: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Window size = 3, Horizon = 1:

Input (X)        Target (y)
─────────        ──────────
[1, 2, 3]    →      [4]
[2, 3, 4]    →      [5]
[3, 4, 5]    →      [6]
[4, 5, 6]    →      [7]
...

Window size = 3, Horizon = 2:

Input (X)        Target (y)
─────────        ──────────
[1, 2, 3]    →      [4, 5]
[2, 3, 4]    →      [5, 6]
[3, 4, 5]    →      [6, 7]
...


Para series multivariadas:

Features: [temp, humidity, pressure]
Window size = 3

Input shape: (batch, window, features) = (N, 3, 3)
```

### Implementacion

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """
    Dataset para series temporales con ventanas deslizantes.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        horizon: int = 1,
        stride: int = 1
    ):
        """
        Args:
            data: Array [timesteps, features] o [timesteps]
            window_size: Tamano de ventana de entrada
            horizon: Pasos a predecir
            stride: Paso entre ventanas
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride

        # Calcular numero de samples
        self.n_samples = (len(data) - window_size - horizon + 1) // stride

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        X = self.data[start:end]
        y = self.data[end:end + self.horizon, 0]  # Solo primera feature como target

        return X, y


def create_dataloaders(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train/val/test.
    Split temporal (no random).
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end - window_size:val_end]  # Overlap para continuidad
    test_data = data[val_end - window_size:]

    train_dataset = TimeSeriesDataset(train_data, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data, window_size, horizon)
    test_dataset = TimeSeriesDataset(test_data, window_size, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
```

### Modelo LSTM para Forecasting

```python
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM para forecasting de series temporales.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            [batch, output_size]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Usar ultimo timestep
        last_output = lstm_out[:, -1, :]

        # Fully connected
        output = self.fc(last_output)

        return output


class Seq2SeqLSTM(nn.Module):
    """
    Encoder-Decoder LSTM para prediccion multi-step.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        forecast_horizon: int = 12,
        dropout: float = 0.2
    ):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size] - input sequence
            target: [batch, horizon] - target sequence (for teacher forcing)
            teacher_forcing_ratio: Probabilidad de usar target real

        Returns:
            [batch, horizon, output_size]
        """
        batch_size = x.size(0)

        # Encode
        _, (hidden, cell) = self.encoder(x)

        # Decoder input: ultimo valor de la secuencia
        decoder_input = x[:, -1:, :self.output_size]  # [batch, 1, output_size]

        outputs = []

        for t in range(self.forecast_horizon):
            # Decode step
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input, (hidden, cell)
            )

            # Prediccion
            prediction = self.fc(decoder_output)  # [batch, 1, output_size]
            outputs.append(prediction)

            # Decidir input para siguiente paso
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1].unsqueeze(-1)
            else:
                decoder_input = prediction

        return torch.cat(outputs, dim=1).squeeze(-1)  # [batch, horizon]


def train_lstm_forecaster(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10
) -> dict:
    """
    Entrena modelo LSTM con early stopping.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                val_loss += criterion(output, y).item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    model.load_state_dict(best_model_state)

    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss
    }
```

---

## Temporal Convolutional Networks (TCN)

### Concepto

```
TCN: Convoluciones Causales Dilatadas
=====================================

Ventajas sobre LSTM:
- Paralelizable (no secuencial)
- Memoria mas larga con menos parametros
- Gradientes mas estables

CONVOLUCION CAUSAL
──────────────────

Solo mira hacia el pasado:

Normal Conv:        Causal Conv:
    ╲│╱                 │
     ●                 ╲│
                        ●

Output_t depende solo de Input_{≤t}


CONVOLUCION DILATADA
────────────────────

Dilation = 1:       Dilation = 2:       Dilation = 4:
│ │ │ │ │           │   │   │           │       │
╲│╱ ╲│╱ ╲           ╲   │   ╱           ╲       ╱
 ●   ●   ●           ╲  │  ╱             ╲     ╱
                      ╲ │ ╱               ╲   ╱
                       ╲│╱                 ╲ ╱
                        ●                   ●

Receptive field crece exponencialmente:
Layer 1 (d=1): 3
Layer 2 (d=2): 7
Layer 3 (d=4): 15
Layer 4 (d=8): 31
...

Con kernel_size=3 y 10 layers: RF = 2^10 - 1 = 1023!
```

### Implementacion TCN

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """
    Convolucion causal: output_t solo depende de input_{<=t}
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, seq_len]
        out = self.conv(x)

        # Remover padding del futuro
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        return out


class TCNBlock(nn.Module):
    """
    Bloque residual TCN con dos convoluciones dilatadas.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network para forecasting.
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        num_channels: list[int] = [64, 64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(TCNBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))

        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            [batch, output_size]
        """
        # TCN espera [batch, channels, seq_len]
        x = x.permute(0, 2, 1)

        # Pasar por TCN
        out = self.network(x)

        # Usar ultimo timestep
        out = out[:, :, -1]

        return self.fc(out)

    @property
    def receptive_field(self) -> int:
        """Calcula campo receptivo."""
        num_levels = len(self.network)
        kernel_size = 3  # asumiendo kernel_size=3

        rf = 1
        for i in range(num_levels):
            dilation = 2 ** i
            rf += 2 * (kernel_size - 1) * dilation

        return rf


# Ejemplo de uso
def example_tcn():
    """Ejemplo de TCN para forecasting."""

    # Datos sinteticos
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    series = (
        np.sin(2 * np.pi * t / 50) +
        0.5 * np.sin(2 * np.pi * t / 20) +
        np.random.randn(n) * 0.1
    )

    # Preparar datos
    window_size = 50
    horizon = 1

    train_loader, val_loader, test_loader = create_dataloaders(
        series, window_size, horizon, batch_size=32
    )

    # Modelo
    model = TCN(
        input_size=1,
        output_size=horizon,
        num_channels=[32, 32, 32, 32],
        kernel_size=3,
        dropout=0.2
    )

    print(f"TCN Receptive Field: {model.receptive_field}")

    # Entrenar
    result = train_lstm_forecaster(model, train_loader, val_loader, epochs=50)

    print(f"Best validation loss: {result['best_val_loss']:.6f}")

    return result


if __name__ == "__main__":
    example_tcn()
```

---

## N-BEATS (Neural Basis Expansion Analysis)

### Concepto

```
N-BEATS: Bloques Apilados de Expansion
======================================

Arquitectura modular sin componentes recurrentes.

                    Input (lookback window)
                              │
                              ▼
                    ┌─────────────────┐
                    │    Stack 1      │
                    │   (Trend)       │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
      Backcast_1        Forecast_1            │
           │                 │                 │
           │                 ▼                 │
           │            ┌────────┐             │
           │            │ Stack 2│             │
           │            │(Season)│             │
           │            └───┬────┘             │
           │                │                  │
           │    ┌───────────┼──────────┐       │
           │    │           │          │       │
           │    ▼           ▼          │       │
           │ Backcast_2  Forecast_2    │       │
           │    │           │          │       │
           │    │           ▼          │       │
           │    │      ┌────────┐      │       │
           │    │      │  ...   │      │       │
           │    │      └────────┘      │       │
           │    │                      │       │
           ▼    ▼                      ▼       ▼
        Residual            Sum of Forecasts
        (should be 0)           = Final


Cada bloque:
1. FC layers para extraer features
2. Genera backcast (reconstruye input)
3. Genera forecast (predice futuro)
4. Residuo = Input - Backcast → siguiente bloque
```

### Implementacion N-BEATS Simplificada

```python
import torch
import torch.nn as nn


class NBEATSBlock(nn.Module):
    """
    Bloque basico de N-BEATS.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        backcast_size: int = None,
        forecast_size: int = None
    ):
        super().__init__()

        self.backcast_size = backcast_size or input_size
        self.forecast_size = forecast_size or input_size

        # FC stack
        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])

        self.fc_stack = nn.Sequential(*layers)

        # Theta para backcast y forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

        # Basis expansion (lineal simple para genericidad)
        self.backcast_basis = nn.Linear(theta_size, self.backcast_size)
        self.forecast_basis = nn.Linear(theta_size, self.forecast_size)

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_size]

        Returns:
            backcast: [batch, backcast_size]
            forecast: [batch, forecast_size]
        """
        # FC stack
        h = self.fc_stack(x)

        # Theta
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)

        return backcast, forecast


class NBEATSStack(nn.Module):
    """
    Stack de bloques N-BEATS.
    """

    def __init__(
        self,
        num_blocks: int,
        input_size: int,
        theta_size: int,
        hidden_size: int,
        forecast_size: int
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_size=input_size,
                theta_size=theta_size,
                hidden_size=hidden_size,
                backcast_size=input_size,
                forecast_size=forecast_size
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_size]

        Returns:
            residual: [batch, input_size]
            forecast: [batch, forecast_size]
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.blocks[0].forecast_size, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return residual, forecast


class NBEATS(nn.Module):
    """
    N-BEATS completo.
    """

    def __init__(
        self,
        input_size: int = 50,
        forecast_size: int = 10,
        num_stacks: int = 2,
        num_blocks: int = 3,
        hidden_size: int = 256,
        theta_size: int = 32
    ):
        super().__init__()

        self.input_size = input_size
        self.forecast_size = forecast_size

        self.stacks = nn.ModuleList([
            NBEATSStack(
                num_blocks=num_blocks,
                input_size=input_size,
                theta_size=theta_size,
                hidden_size=hidden_size,
                forecast_size=forecast_size
            )
            for _ in range(num_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features] o [batch, seq_len]

        Returns:
            forecast: [batch, forecast_size]
        """
        # Flatten input
        if x.dim() == 3:
            x = x[:, :, 0]  # Usar primera feature

        residual = x
        forecast = torch.zeros(x.size(0), self.forecast_size, device=x.device)

        for stack in self.stacks:
            stack_residual, stack_forecast = stack(residual)
            residual = stack_residual
            forecast = forecast + stack_forecast

        return forecast
```

---

## Transformers para Series Temporales

### Concepto

```
TRANSFORMERS PARA TIME SERIES
=============================

Adaptacion de Transformers para forecasting:

1. POSITIONAL ENCODING TEMPORAL
   - Sinusoidal (como NLP)
   - Aprendido
   - Time2Vec (encodings aprendidos de tiempo)

2. INPUT EMBEDDING
   - Proyeccion lineal de features
   - Convolucion 1D
   - Patching (dividir en segmentos)

3. ATTENTION MODIFICADA
   - Causal mask para decoder
   - Local attention (sparse)
   - LogSparse attention


Arquitectura tipica:
───────────────────

Input [batch, seq, features]
        │
        ▼
┌───────────────────┐
│ Input Embedding   │
│ + Positional Enc  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Transformer       │
│ Encoder           │
│ (Self-Attention)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Transformer       │
│ Decoder           │
│ (Cross-Attention) │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Output Projection │
└─────────┬─────────┘
          │
          ▼
Output [batch, horizon, 1]
```

### Implementacion Simplificada

```python
import torch
import torch.nn as nn
import math


class TimeSeriesTransformer(nn.Module):
    """
    Transformer simplificado para forecasting.
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

    def forward(
        self,
        src: torch.Tensor,
        tgt_len: int
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, src_len, input_size]
            tgt_len: Numero de pasos a predecir

        Returns:
            [batch, tgt_len]
        """
        batch_size = src.size(0)

        # Project and add positional encoding
        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Create target (zeros para autoregresivo simplificado)
        tgt = torch.zeros(batch_size, tgt_len, self.d_model, device=src.device)
        tgt = self.pos_encoder(tgt)

        # Causal mask para decoder
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src.device)

        # Transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Project to output
        output = self.output_projection(output).squeeze(-1)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding sinusoidal."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

---

## Aplicacion: Forecasting de Metricas de Seguridad

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List


class SecurityMetricsForecaster:
    """
    Sistema de forecasting para metricas de seguridad.
    Soporta multiples arquitecturas (LSTM, TCN, Transformer).
    """

    def __init__(
        self,
        model_type: str = 'lstm',
        window_size: int = 24,
        forecast_horizon: int = 12,
        hidden_size: int = 64
    ):
        self.model_type = model_type
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size

        self.model = None
        self.scaler = None

    def _create_model(self, input_size: int) -> nn.Module:
        """Crea modelo segun tipo especificado."""
        if self.model_type == 'lstm':
            return LSTMForecaster(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=2,
                output_size=self.forecast_horizon
            )
        elif self.model_type == 'tcn':
            return TCN(
                input_size=input_size,
                output_size=self.forecast_horizon,
                num_channels=[self.hidden_size] * 4
            )
        elif self.model_type == 'transformer':
            return TimeSeriesTransformer(
                input_size=input_size,
                d_model=self.hidden_size
            )
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}")

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: List[str] | None = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """
        Entrena el modelo.

        Args:
            data: DataFrame con metricas
            target_col: Columna objetivo
            feature_cols: Columnas de features (None = solo target)
            epochs: Numero de epocas
            batch_size: Tamano de batch

        Returns:
            Historial de entrenamiento
        """
        # Preparar features
        if feature_cols is None:
            features = data[[target_col]].values
        else:
            features = data[[target_col] + feature_cols].values

        # Normalizar
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Crear dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            features_scaled,
            self.window_size,
            self.forecast_horizon,
            batch_size
        )

        # Crear modelo
        input_size = features.shape[1]
        self.model = self._create_model(input_size)

        # Entrenar
        result = train_lstm_forecaster(
            self.model,
            train_loader,
            val_loader,
            epochs=epochs
        )

        return result['history']

    def predict(
        self,
        recent_data: np.ndarray
    ) -> np.ndarray:
        """
        Genera prediccion.

        Args:
            recent_data: Ultimos window_size valores

        Returns:
            Array con forecast_horizon predicciones
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        # Normalizar
        recent_scaled = self.scaler.transform(recent_data.reshape(-1, 1))

        # Preparar tensor
        x = torch.FloatTensor(recent_scaled).unsqueeze(0)

        # Predecir
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'transformer':
                pred = self.model(x, self.forecast_horizon)
            else:
                pred = self.model(x)

        # Desnormalizar
        pred_np = pred.numpy().reshape(-1, 1)
        pred_original = self.scaler.inverse_transform(
            np.hstack([pred_np, np.zeros((len(pred_np), self.scaler.n_features_in_ - 1))])
        )[:, 0]

        return pred_original

    def detect_anomaly(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        threshold_std: float = 2.5
    ) -> List[Dict]:
        """
        Detecta anomalias comparando actual vs predicho.
        """
        errors = actual - predicted
        std = np.std(errors)
        mean = np.mean(errors)

        anomalies = []
        for i, (a, p, e) in enumerate(zip(actual, predicted, errors)):
            z_score = (e - mean) / std if std > 0 else 0

            if abs(z_score) > threshold_std:
                anomalies.append({
                    'index': i,
                    'actual': a,
                    'predicted': p,
                    'error': e,
                    'z_score': z_score,
                    'type': 'spike' if z_score > 0 else 'drop'
                })

        return anomalies


# Ejemplo de uso
def demo_security_forecasting():
    """Demo de forecasting de metricas de seguridad."""

    np.random.seed(42)

    # Simular metricas (alertas por hora)
    n_hours = 24 * 30  # 30 dias
    hours = np.arange(n_hours)

    # Patron: diario + semanal + tendencia + ruido
    daily = 50 * np.sin(2 * np.pi * hours / 24)
    weekly = 20 * np.sin(2 * np.pi * hours / (24 * 7))
    trend = 0.1 * hours
    noise = np.random.randn(n_hours) * 10

    alerts = 200 + daily + weekly + trend + noise
    alerts = np.maximum(alerts, 0)

    # DataFrame
    dates = pd.date_range('2024-01-01', periods=n_hours, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'alerts': alerts,
        'hour': dates.hour,
        'dayofweek': dates.dayofweek
    })

    # Entrenar forecaster
    print("=== Security Metrics Forecaster ===")

    for model_type in ['lstm', 'tcn']:
        print(f"\nEntrenando {model_type.upper()}...")

        forecaster = SecurityMetricsForecaster(
            model_type=model_type,
            window_size=48,
            forecast_horizon=24,
            hidden_size=64
        )

        history = forecaster.fit(
            df,
            target_col='alerts',
            epochs=30,
            batch_size=32
        )

        # Prediccion
        recent = df['alerts'].values[-48:]
        prediction = forecaster.predict(recent)

        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Prediction (next 6h): {prediction[:6].round(1)}")


if __name__ == "__main__":
    demo_security_forecasting()
```

---

## Comparativa de Modelos

| Modelo | Ventajas | Desventajas | Cuando Usar |
|--------|----------|-------------|-------------|
| LSTM | Flexible, maneja dependencias | Lento, secuencial | Series moderadas |
| GRU | Mas rapido que LSTM | Menos capacidad | Recursos limitados |
| TCN | Paralelo, memoria larga | Menos flexible | Series largas |
| N-BEATS | Interpretable, robusto | Solo univariado | Forecasting puro |
| Transformer | SOTA, paralelo | Muchos datos | Series complejas |

---

## Hiperparametros Recomendados

| Parametro | LSTM/GRU | TCN | Transformer |
|-----------|----------|-----|-------------|
| hidden_size | 64-256 | - | d_model: 64-256 |
| num_layers | 2-3 | 4-8 levels | 2-4 |
| dropout | 0.2-0.4 | 0.1-0.3 | 0.1-0.2 |
| learning_rate | 1e-3 - 1e-4 | 1e-3 | 1e-4 |
| batch_size | 32-128 | 32-64 | 16-64 |
| window_size | 24-168 | Depende RF | 48-200 |

---

## Resumen

```
SELECCION DE MODELO DL
======================

1. ¿Cuantos datos tienes?
   < 1000 puntos: ARIMA/ETS mejor
   1000-10000: LSTM/GRU
   > 10000: TCN/Transformer

2. ¿Que tan largas son las dependencias?
   Cortas (<50): LSTM
   Medias (50-200): TCN
   Largas (>200): Transformer o TCN profundo

3. ¿Necesitas interpretabilidad?
   Si: N-BEATS, descomposicion
   No: Cualquiera

4. ¿Recursos computacionales?
   Limitados: GRU, TCN pequeno
   Abundantes: Transformer


CHECKLIST DL TIME SERIES
========================

[ ] Normalizar datos (StandardScaler)
[ ] Split temporal (no random!)
[ ] Window size >= periodo estacional
[ ] Early stopping con validation
[ ] Gradient clipping (1.0)
[ ] Learning rate scheduler
[ ] Evaluar con MASE, no solo MSE
```

### Puntos Clave

1. **Deep Learning** necesita **mas datos** que metodos clasicos
2. **Window size** debe capturar patrones relevantes
3. **Split temporal** es obligatorio (no shuffle)
4. **TCN** es excelente alternativa a LSTM
5. **Transformers** son SOTA pero requieren mas recursos
6. Siempre comparar con **baseline clasico** (ARIMA, ETS)
