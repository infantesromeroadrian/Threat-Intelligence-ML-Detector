# Redes Feedforward (MLP)

## 1. ¿Qué es un MLP?

### Definición

```
┌────────────────────────────────────────────────────────────────┐
│  MLP = Multi-Layer Perceptron                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Red neuronal "tradicional" donde la información fluye         │
│  en UNA SOLA DIRECCIÓN: de entrada a salida                    │
│                                                                │
│  También llamada:                                              │
│    • Feedforward Neural Network                                │
│    • Fully Connected Network (FCN)                             │
│    • Dense Network                                             │
│                                                                │
│  CARACTERÍSTICAS:                                              │
│    • Cada neurona de una capa se conecta a TODAS               │
│      las neuronas de la siguiente capa                         │
│    • No hay conexiones hacia atrás (no recurrente)             │
│    • No hay conexiones dentro de la misma capa                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Arquitectura

```
┌────────────────────────────────────────────────────────────────┐
│  ARQUITECTURA MLP                                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   CAPA           CAPA            CAPA           CAPA           │
│  ENTRADA       OCULTA 1        OCULTA 2       SALIDA           │
│                                                                │
│    x₁  ●─────────●──────────────●                              │
│          ╲     ╱ │ ╲          ╱ │ ╲                            │
│    x₂  ●──╲───╱──●──╲────────╱──●──╲                           │
│          ╲ ╳ ╱   │ ╲ ╳      ╱   │   ╲                          │
│    x₃  ●──╱─╲────●──╱─╲────╱────●────●────→ ŷ                  │
│          ╱   ╲   │ ╱   ╲  ╱     │   ╱                          │
│    x₄  ●─────────●──────╲╱──────●──╱                           │
│          ╱     ╲ │       ╳      │ ╱                            │
│    x₅  ●─────────●──────╱╲──────●                              │
│                                                                │
│   (5 neuronas) (4 neuronas) (3 neuronas) (1 neurona)           │
│                                                                │
│  Cada línea = peso (parámetro aprendible)                      │
│  Total parámetros = (5×4 + 4) + (4×3 + 3) + (3×1 + 1)         │
│                   = 24 + 15 + 4 = 43 parámetros                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Notación Matricial

```
┌────────────────────────────────────────────────────────────────┐
│  NOTACIÓN MATRICIAL (Eficiente para cómputo)                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Para un batch de m muestras:                                  │
│                                                                │
│  CAPA l:                                                       │
│    Z⁽ˡ⁾ = W⁽ˡ⁾ · A⁽ˡ⁻¹⁾ + b⁽ˡ⁾                                │
│    A⁽ˡ⁾ = f(Z⁽ˡ⁾)                                              │
│                                                                │
│  Donde:                                                        │
│    A⁽⁰⁾ = X                    (entrada)                       │
│    W⁽ˡ⁾ ∈ ℝ^(nˡ × nˡ⁻¹)       (pesos)                         │
│    b⁽ˡ⁾ ∈ ℝ^nˡ                (biases)                         │
│    Z⁽ˡ⁾ ∈ ℝ^(nˡ × m)          (pre-activación)                │
│    A⁽ˡ⁾ ∈ ℝ^(nˡ × m)          (activación)                    │
│    nˡ = número de neuronas en capa l                           │
│    m = tamaño del batch                                        │
│                                                                │
│  EJEMPLO:                                                      │
│    Capa con 64 neuronas, entrada de 128:                       │
│    W ∈ ℝ^(64 × 128) = 8192 pesos                               │
│    b ∈ ℝ^64 = 64 biases                                        │
│    Total: 8256 parámetros                                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 2. Diseño de Arquitectura

### Principios de Diseño

```
┌────────────────────────────────────────────────────────────────┐
│  PRINCIPIOS DE DISEÑO DE MLP                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. CAPA DE ENTRADA                                            │
│     ─────────────────                                          │
│     • Número de neuronas = número de features                  │
│     • No tiene pesos ni activación                             │
│                                                                │
│  2. CAPAS OCULTAS                                              │
│     ──────────────                                             │
│     • Empezar con 1-2 capas para problemas simples             │
│     • Más capas = más capacidad pero más difícil entrenar      │
│     • Neuronas: potencias de 2 (32, 64, 128, 256)              │
│                                                                │
│     Patrones comunes:                                          │
│       Pirámide:     128 → 64 → 32  (reduce gradualmente)       │
│       Constante:    64 → 64 → 64   (mismo tamaño)              │
│       Embudo:       256 → 128 → 64 (compresión)                │
│       Diamante:     64 → 128 → 64  (expandir y comprimir)      │
│                                                                │
│  3. CAPA DE SALIDA                                             │
│     ────────────────                                           │
│     • Binaria: 1 neurona + sigmoid                             │
│     • Multiclase: K neuronas + softmax                         │
│     • Regresión: 1 neurona + lineal (sin activación)           │
│     • Multi-output: N neuronas + activación apropiada          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Ejemplos de Arquitecturas

```
CLASIFICACIÓN BINARIA (ej: malware vs benigno):
───────────────────────────────────────────────
Input (100 features) → Dense(64, relu) → Dropout(0.3) →
Dense(32, relu) → Dropout(0.3) → Dense(1, sigmoid)


CLASIFICACIÓN MULTICLASE (ej: tipo de ataque):
──────────────────────────────────────────────
Input (50 features) → Dense(128, relu) → BatchNorm →
Dropout(0.4) → Dense(64, relu) → BatchNorm →
Dropout(0.4) → Dense(5, softmax)  # 5 clases


REGRESIÓN (ej: tiempo de respuesta a incidente):
────────────────────────────────────────────────
Input (20 features) → Dense(64, relu) → Dense(32, relu) →
Dense(1, linear)  # sin activación en salida


RED PROFUNDA (problema complejo):
─────────────────────────────────
Input → Dense(256, relu) → BatchNorm → Dropout(0.5) →
Dense(256, relu) → BatchNorm → Dropout(0.5) →
Dense(128, relu) → BatchNorm → Dropout(0.5) →
Dense(64, relu) → Dropout(0.3) → Dense(num_classes, softmax)
```

## 3. Inicialización de Pesos

### ¿Por qué importa la inicialización?

```
┌────────────────────────────────────────────────────────────────┐
│  IMPORTANCIA DE LA INICIALIZACIÓN                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  MALA inicialización:                                          │
│                                                                │
│  1. Pesos muy PEQUEÑOS:                                        │
│     Activaciones → 0                                           │
│     Gradientes → 0 (vanishing gradient)                        │
│     Red no aprende                                             │
│                                                                │
│  2. Pesos muy GRANDES:                                         │
│     Activaciones → ±∞ (saturación)                             │
│     Gradientes → 0 o ∞                                         │
│     Red inestable                                              │
│                                                                │
│  BUENA inicialización:                                         │
│     Activaciones con varianza ~1                               │
│     Gradientes fluyen correctamente                            │
│     Red aprende eficientemente                                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Métodos de Inicialización

```
┌────────────────────────────────────────────────────────────────┐
│  MÉTODOS DE INICIALIZACIÓN                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. XAVIER/GLOROT (para sigmoid, tanh)                         │
│     ────────────────────────────────                           │
│     W ~ N(0, σ²)  donde σ = √(2 / (n_in + n_out))             │
│                                                                │
│     O versión uniforme:                                        │
│     W ~ U(-a, a)  donde a = √(6 / (n_in + n_out))             │
│                                                                │
│  2. HE/KAIMING (para ReLU) ★ MÁS COMÚN                         │
│     ───────────────────────                                    │
│     W ~ N(0, σ²)  donde σ = √(2 / n_in)                       │
│                                                                │
│     ReLU "mata" la mitad de las activaciones                   │
│     He compensa multiplicando por √2                           │
│                                                                │
│  3. LECUN (para SELU)                                          │
│     ────────────────                                           │
│     W ~ N(0, σ²)  donde σ = √(1 / n_in)                       │
│                                                                │
│                                                                │
│  En frameworks modernos, la inicialización apropiada           │
│  se elige automáticamente según la activación                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código de Inicialización

```python
import torch.nn as nn

# PyTorch: inicialización automática según tipo de capa

# Manual (si necesitas control):
class CustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Inicialización He para ReLU
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

        # Inicialización Xavier para sigmoid
        nn.init.xavier_normal_(self.fc3.weight)

        # Biases a cero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
```

## 4. Batch Normalization

### Concepto

```
┌────────────────────────────────────────────────────────────────┐
│  BATCH NORMALIZATION                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Normaliza las activaciones en cada capa durante entrenamiento │
│                                                                │
│  Para cada feature en el batch:                                │
│                                                                │
│  1. Calcular media y varianza del batch:                       │
│     μ_B = (1/m) Σ xᵢ                                           │
│     σ²_B = (1/m) Σ (xᵢ - μ_B)²                                │
│                                                                │
│  2. Normalizar:                                                │
│     x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)                             │
│                                                                │
│  3. Escalar y desplazar (parámetros aprendibles):              │
│     yᵢ = γ · x̂ᵢ + β                                           │
│                                                                │
│  VENTAJAS:                                                     │
│    ✓ Permite learning rates más altos                          │
│    ✓ Reduce dependencia de inicialización                      │
│    ✓ Actúa como regularizador                                  │
│    ✓ Acelera convergencia                                      │
│                                                                │
│  POSICIÓN: Típicamente ANTES de la activación                  │
│    Dense → BatchNorm → ReLU → Dropout                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Código

```python
import torch.nn as nn

class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),    # BatchNorm para 1D
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
```

## 5. Implementación Completa

### PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================
# DEFINIR MODELO
# ============================================================
class MalwareClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_classes=2,
                 dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        # Capas ocultas
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Capa de salida
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================
# PREPARAR DATOS
# ============================================================
# Simular datos de características de malware
np.random.seed(42)
n_samples = 5000
n_features = 100

# Features aleatorias (en un caso real serían API calls, strings, etc.)
X = np.random.randn(n_samples, n_features)
# Etiquetas: 0=benigno, 1=malware
y = np.random.randint(0, 2, n_samples)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir a tensores
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================
# CREAR MODELO
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = MalwareClassifier(
    input_size=n_features,
    hidden_sizes=[128, 64, 32],
    num_classes=2,
    dropout_rate=0.3
).to(device)

print(f"Arquitectura del modelo:\n{model}")
print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# CONFIGURAR ENTRENAMIENTO
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ============================================================
# ENTRENAMIENTO
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total

# Training loop con early stopping
best_val_loss = float('inf')
patience = 10
patience_counter = 0
epochs = 100

print("\nEntrenando...")
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)

    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping en epoch {epoch+1}")
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('best_model.pth'))

# ============================================================
# EVALUACIÓN FINAL
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

print("\n" + "="*60)
print("RESULTADOS FINALES")
print("="*60)
print(classification_report(all_labels, all_preds,
      target_names=['Benigno', 'Malware']))
print("Matriz de confusión:")
print(confusion_matrix(all_labels, all_preds))
```

### Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Preparar datos
np.random.seed(42)
X = np.random.randn(5000, 100)
y = np.random.randint(0, 2, 5000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir modelo
model = Sequential([
    Dense(128, input_shape=(100,)),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    Dropout(0.3),

    Dense(32),
    BatchNormalization(),
    tf.keras.layers.ReLU(),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# Entrenar
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
```

## 6. Ejemplo Práctico: Detector de Intrusiones

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================
# SIMULAR DATOS DE TRÁFICO DE RED
# ============================================================
np.random.seed(42)
n_samples = 10000

# Features de conexión de red
data = {
    'duration': np.random.exponential(10, n_samples),
    'src_bytes': np.random.exponential(5000, n_samples),
    'dst_bytes': np.random.exponential(10000, n_samples),
    'num_failed_logins': np.random.poisson(0.5, n_samples),
    'num_access_files': np.random.poisson(2, n_samples),
    'count': np.random.poisson(10, n_samples),
    'srv_count': np.random.poisson(5, n_samples),
    'serror_rate': np.random.beta(1, 10, n_samples),
    'srv_serror_rate': np.random.beta(1, 10, n_samples),
    'same_srv_rate': np.random.beta(10, 1, n_samples),
}

df = pd.DataFrame(data)

# Crear etiquetas de ataque
conditions = [
    (df['num_failed_logins'] > 2) | (df['serror_rate'] > 0.5),  # DoS
    (df['count'] > 30) & (df['duration'] < 1),                   # Probe
    (df['num_access_files'] > 5) & (df['dst_bytes'] > 20000),   # R2L
    (df['num_failed_logins'] > 3) & (df['num_access_files'] > 3), # U2R
]
labels = ['DoS', 'Probe', 'R2L', 'U2R']

df['attack_type'] = 'Normal'
for cond, label in zip(conditions, labels):
    df.loc[cond & (df['attack_type'] == 'Normal'), 'attack_type'] = label

print("Distribución de clases:")
print(df['attack_type'].value_counts())

# Preparar datos
X = df.drop('attack_type', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['attack_type'])
num_classes = len(le.classes_)

print(f"\nClases: {le.classes_}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tensores
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                         batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                        batch_size=128, shuffle=False)

# ============================================================
# MODELO IDS
# ============================================================
class IntrusionDetector(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            # Capa 1
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Capa 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Capa 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Salida
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Crear modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IntrusionDetector(X_train.shape[1], num_classes).to(device)

# Pesos para clases desbalanceadas
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
class_weights = class_weights / class_weights.sum() * num_classes

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# ENTRENAMIENTO
# ============================================================
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Train
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()

    val_losses.append(val_loss / len(test_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, "
              f"Val Loss={val_losses[-1]:.4f}")

# ============================================================
# EVALUACIÓN
# ============================================================
model.eval()
all_preds = []
all_probs = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

print("\n" + "="*70)
print("DETECTOR DE INTRUSIONES - RESULTADOS")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_test, all_preds, target_names=le.classes_))

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, all_preds)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Curvas de Aprendizaje')
axes[0].legend()
axes[0].grid(True)

# Confusion matrix
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks(range(len(le.classes_)))
axes[1].set_yticks(range(len(le.classes_)))
axes[1].set_xticklabels(le.classes_, rotation=45)
axes[1].set_yticklabels(le.classes_)
axes[1].set_xlabel('Predicho')
axes[1].set_ylabel('Real')
axes[1].set_title('Matriz de Confusión')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
```

## 7. Ventajas y Desventajas

```
┌────────────────────────────────────────────────────────────────┐
│  VENTAJAS DE MLP                                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✓ Aproximador universal (puede aprender cualquier función)    │
│  ✓ Flexible: funciona para clasificación y regresión           │
│  ✓ Maneja relaciones no lineales complejas                     │
│  ✓ Escala bien con GPUs                                        │
│  ✓ Bien soportado por frameworks (PyTorch, TensorFlow)         │
│  ✓ Fácil de implementar y experimentar                         │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  DESVENTAJAS DE MLP                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ✗ No captura estructura espacial (usar CNN para imágenes)     │
│  ✗ No captura dependencias temporales (usar RNN para secuencias)
│  ✗ Muchos hiperparámetros que ajustar                          │
│  ✗ Puede requerir muchos datos                                 │
│  ✗ "Caja negra" - difícil de interpretar                       │
│  ✗ Propenso a overfitting                                      │
│  ✗ Tiempo de entrenamiento puede ser largo                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 8. Resumen

```
┌────────────────────────────────────────────────────────────────┐
│  MLP / FEEDFORWARD - RESUMEN                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ARQUITECTURA:                                                 │
│    • Capas densas (fully connected)                            │
│    • Información fluye en una dirección                        │
│    • Patrón: Input → Dense → Activation → Dropout → ... → Output
│                                                                │
│  COMPONENTES:                                                  │
│    • Dense layers: transformación lineal                       │
│    • BatchNorm: normalización de activaciones                  │
│    • Dropout: regularización                                   │
│    • Activaciones: ReLU (ocultas), sigmoid/softmax (salida)    │
│                                                                │
│  ENTRENAMIENTO:                                                │
│    • Optimizer: Adam (más común)                               │
│    • Loss: CrossEntropy (clasificación), MSE (regresión)       │
│    • Early stopping para evitar overfitting                    │
│                                                                │
│  BUENAS PRÁCTICAS:                                             │
│    • Escalar datos (StandardScaler)                            │
│    • BatchNorm después de Dense                                │
│    • Dropout después de activación                             │
│    • Arquitectura pirámide o constante                         │
│    • Monitorear train vs validation loss                       │
│                                                                │
│  EN CIBERSEGURIDAD:                                            │
│    • Clasificación de malware                                  │
│    • Detección de intrusiones                                  │
│    • Análisis de logs                                          │
│    • Detección de phishing                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

**Siguiente:** Redes Convolucionales (CNN)
