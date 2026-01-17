# Graph Convolutional Networks (GCN) y Graph Attention Networks (GAT)

## Indice

1. [Introduccion a GCN](#introduccion-a-gcn)
2. [Convolucion Espectral vs Espacial](#convolucion-espectral-vs-espacial)
3. [GCN: Formulacion Matematica](#gcn-formulacion-matematica)
4. [Implementacion de GCN](#implementacion-de-gcn)
5. [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
6. [Multi-Head Attention en Grafos](#multi-head-attention-en-grafos)
7. [Implementacion de GAT](#implementacion-de-gat)
8. [Comparativa GCN vs GAT](#comparativa-gcn-vs-gat)
9. [Implementacion con PyTorch Geometric](#implementacion-con-pytorch-geometric)
10. [Aplicaciones en Ciberseguridad](#aplicaciones-en-ciberseguridad)
11. [Ejercicios Practicos](#ejercicios-practicos)

---

## Introduccion a GCN

### De CNNs a GCNs

```
EVOLUCION: CNN -> GCN
=====================

CNN en Imagenes:
----------------
Imagen = Grid regular de pixeles

    +---+---+---+---+---+
    | P | P | P | P | P |
    +---+---+---+---+---+
    | P | P | P | P | P |    Kernel 3x3
    +---+---+---+---+---+    deslizante
    | P | P |[X]| P | P |    +---+---+---+
    +---+---+---+---+---+    | w | w | w |
    | P | P | P | P | P |    +---+---+---+
    +---+---+---+---+---+    | w |[c]| w |
    | P | P | P | P | P |    +---+---+---+
    +---+---+---+---+---+    | w | w | w |
                             +---+---+---+

- Vecindario FIJO de 8 vecinos
- Kernel de pesos COMPARTIDOS
- Invariante a traslacion


GCN en Grafos:
--------------
Grafo = Estructura irregular

          [A]                    [1]
         / | \                    |
        /  |  \                   |
      [B] [C] [D]               [2]---[3]---[4]
       |       |                     / \
       |       |                    /   \
      [E]     [F]                 [5]   [6]

- Vecindario VARIABLE
- Pesos ADAPTADOS a estructura
- Invariante a permutaciones


Idea clave: Generalizar convolucion a grafos
mediante agregacion de vecinos.
```

### Motivacion para GCN

```
EL PROBLEMA DE APRENDIZAJE EN GRAFOS
====================================

Dado:
- Grafo G = (V, E) con n nodos
- Features de nodos X in R^(n x d)
- Labels parciales Y (semi-supervisado)

Objetivo:
- Aprender representaciones h_v para cada nodo
- Que capturen estructura local Y global
- Para tareas downstream (clasificacion, link prediction)


Semi-supervisado: Solo algunos nodos tienen label

          [?]---[+]---[?]
           |     |     |
          [?]---[-]---[?]
           |     |     |
          [+]---[?]---[-]

    [+] = Clase positiva (conocida)
    [-] = Clase negativa (conocida)
    [?] = Clase desconocida (a predecir)

GCN propaga informacion de nodos etiquetados
a nodos sin etiquetar a traves de la estructura.
```

---

## Convolucion Espectral vs Espacial

### Convolucion Espectral

```
ENFOQUE ESPECTRAL
=================

Basado en la descomposicion espectral del Laplaciano:

1. Laplaciano del grafo:
   L = D - A

   Donde:
   - D = matriz diagonal de grados
   - A = matriz de adyacencia

2. Laplaciano normalizado:
   L_norm = I - D^(-1/2) A D^(-1/2)

3. Descomposicion espectral:
   L_norm = U Lambda U^T

   - U = matriz de autovectores
   - Lambda = diagonal de autovalores


CONVOLUCION ESPECTRAL:
----------------------
En el dominio de Fourier, convolucion = multiplicacion

    x * g = U g_theta U^T x

Donde g_theta son parametros aprendibles.


VISUALIZACION:
--------------

Espacio Original         Espacio Espectral        Convolucion
     (Grafo)              (Fourier)              (Filtrado)

    [x1]                   [f1]                   [f1*g]
    [x2]     --U^T-->      [f2]    --g_theta-->   [f2*g]
    [x3]                   [f3]                   [f3*g]
    [x4]                   [f4]                   [f4*g]

                              |
                              | U (inversa)
                              v

                           [y1]
                           [y2]  <-- Output filtrado
                           [y3]
                           [y4]


PROBLEMA:
- Calcular autovectores es O(n^3)
- No escala a grafos grandes
- Filtros no localizados
```

### Convolucion Espacial

```
ENFOQUE ESPACIAL (USADO EN GCN)
===============================

En lugar de transformar al dominio espectral,
operamos directamente en el grafo.

Idea: Agregar informacion de vecinos directamente

    h_v^(l+1) = sigma(W * AGG({h_u^(l) : u in N(v) U {v}}))


VISUALIZACION:
--------------

      Capa l                      Capa l+1

    [h_A] = [1,0]                [h_A'] = ?
       \                            ^
        \                           |
         \                     Agregar + Transformar
          \                         |
    [h_B] = [0,1] ----> AGG ---> W * [mean(h_A, h_B, h_C)]
          /                         |
         /                          v
        /                       [h_B'] = sigma(...)
       /
    [h_C] = [1,1]


Ejemplo numerico:
-----------------
Vecinos de B: {A, C}
h_A = [1, 0], h_C = [1, 1]
h_B = [0, 1]

Agregacion (mean con self-loop):
AGG = mean([1,0], [0,1], [1,1]) = [0.67, 0.67]

Transformacion (W es 2x2):
W = [[0.5, 0.3],
     [0.2, 0.6]]

h_B' = ReLU(W @ AGG)
     = ReLU([0.535, 0.536])
     = [0.535, 0.536]


VENTAJAS:
- Complejidad O(|E|) - lineal en aristas
- Filtros localizados (k capas = k-hop)
- Escala a grafos grandes
```

### Comparativa Visual

```
ESPECTRAL vs ESPACIAL
=====================

+------------------+-------------------+-------------------+
| Aspecto          | Espectral         | Espacial          |
+------------------+-------------------+-------------------+
| Complejidad      | O(n^3)            | O(|E|)            |
| Escalabilidad    | Baja              | Alta              |
| Localizacion     | Global            | k-hop local       |
| Interpretacion   | Frecuencias       | Vecindarios       |
| Grafos dinamicos | No soporta        | Soporta           |
| Induccion        | No (transductivo) | Si (inductivo*)   |
+------------------+-------------------+-------------------+

*Con GraphSAGE y variantes


ELECCION PRACTICA:
------------------
- Grafos pequenos (< 10K nodos): Ambos funcionan
- Grafos grandes (> 100K nodos): Solo espacial
- Produccion/real-time: Espacial
- Investigacion teorica: Espectral interesante
```

---

## GCN: Formulacion Matematica

### La Regla de Propagacion

```
FORMULACION GCN (Kipf & Welling, 2017)
======================================

Regla de propagacion por capa:

    H^(l+1) = sigma(D~^(-1/2) A~ D~^(-1/2) H^(l) W^(l))

Donde:
    - A~ = A + I_n (adyacencia con self-loops)
    - D~ = matriz diagonal de grados de A~
    - H^(l) = activaciones en capa l
    - W^(l) = pesos entrenables de capa l
    - sigma = funcion de activacion (ReLU)


DESGLOSE PASO A PASO:
---------------------

1. Self-loops: A~ = A + I

   Original:           Con self-loops:
   0 1 1 0             1 1 1 0
   1 0 1 0             1 1 1 0
   1 1 0 1             1 1 1 1
   0 0 1 0             0 0 1 1

2. Normalizacion simetrica: D~^(-1/2) A~ D~^(-1/2)

   Para cada arista (i,j):
   A_norm[i,j] = A~[i,j] / sqrt(d_i * d_j)

   Esto normaliza por el grado de ambos nodos.

3. Propagacion: A_norm @ H

   Cada nodo recibe media ponderada de vecinos.

4. Transformacion lineal: ... @ W

   Proyeccion a nuevo espacio de features.

5. No-linealidad: sigma(...)

   ReLU para expresividad.


INTUICION:
----------
- Self-loops: El nodo mantiene su propia informacion
- Normalizacion: Evita que nodos de alto grado dominen
- Media ponderada: Smooth sobre vecindario
```

### Derivacion desde Filtros Espectrales

```
DE CHEBYSHEV A GCN
==================

Filtros de Chebyshev (ChebNet):
g_theta * x = sum_{k=0}^{K-1} theta_k T_k(L~) x

Donde:
- T_k = polinomios de Chebyshev
- L~ = 2L/lambda_max - I (Laplaciano escalado)
- K = orden del filtro (k-hop)


Simplificacion GCN (K=1, lambda_max=2):

g_theta * x = theta_0 x + theta_1 (L - I) x
            = theta_0 x - theta_1 D^(-1/2) A D^(-1/2) x

Restringiendo theta = theta_0 = -theta_1:

g_theta * x = theta (I + D^(-1/2) A D^(-1/2)) x

Con renormalizacion (trick de self-loop):

g_theta * x = theta D~^(-1/2) A~ D~^(-1/2) x


RESUMEN:
--------
GCN = Filtro espectral de primer orden simplificado
    = Aproximacion lineal localizada
    = Muy eficiente y efectivo en practica
```

### Visualizacion de la Propagacion

```
PROPAGACION EN GCN
==================

Grafo:     A---B           Matriz A~:
           |\ /|           A B C D
           | X |           +-------
           |/ \|         A |1 1 1 0|
           C---D         B |1 1 1 1|
                         C |1 1 1 1|
                         D |0 1 1 1|

Grados D~:  A:3, B:4, C:4, D:3

Normalizacion D~^(-1/2) A~ D~^(-1/2):
-----------------------------------------

A_norm[A,B] = 1 / sqrt(3*4) = 0.289
A_norm[A,C] = 1 / sqrt(3*4) = 0.289
A_norm[B,C] = 1 / sqrt(4*4) = 0.250
A_norm[B,D] = 1 / sqrt(4*3) = 0.289
...

Resultado (aproximado):
       A     B     C     D
A  [0.333 0.289 0.289 0.000]
B  [0.289 0.250 0.250 0.289]
C  [0.289 0.250 0.250 0.289]
D  [0.000 0.289 0.289 0.333]


PROPAGACION DE FEATURES:
------------------------

H^(0):           A_norm @ H^(0):
A: [1, 0]        A: [0.33*[1,0] + 0.29*[0,1] + 0.29*[1,1]]
B: [0, 1]           = [0.62, 0.58]
C: [1, 1]        B: [0.29*[1,0] + 0.25*[0,1] + 0.25*[1,1] + 0.29*[0,0]]
D: [0, 0]           = [0.54, 0.50]
                 C: similar...
                 D: similar...

Los nodos "mezclan" features con sus vecinos!
```

---

## Implementacion de GCN

### Implementacion desde Cero

```python
"""
Implementacion de Graph Convolutional Network desde cero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numpy.typing import NDArray


class GCNLayer(nn.Module):
    """
    Capa de Graph Convolutional Network.

    Implementa: H' = sigma(D~^(-1/2) A~ D~^(-1/2) H W)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Pesos entrenables
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Inicializacion Xavier/Glorot."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        adj_normalized: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            adj_normalized: Matriz de adyacencia normalizada
                           D~^(-1/2) A~ D~^(-1/2)
        """
        # Transformacion lineal: H @ W
        support = torch.mm(x, self.weight)

        # Propagacion de mensaje: A_norm @ support
        output = torch.mm(adj_normalized, support)

        if self.bias is not None:
            output = output + self.bias

        return output


def normalize_adjacency(adj: Tensor, add_self_loops: bool = True) -> Tensor:
    """
    Normaliza matriz de adyacencia: D~^(-1/2) A~ D~^(-1/2)

    Args:
        adj: Matriz de adyacencia [n, n]
        add_self_loops: Agregar self-loops (A + I)
    """
    if add_self_loops:
        # A~ = A + I
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    # Calcular grados
    degrees = adj.sum(dim=1)

    # D^(-1/2)
    deg_inv_sqrt = torch.pow(degrees, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # D^(-1/2) como matriz diagonal
    deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)

    # D~^(-1/2) A~ D~^(-1/2)
    adj_normalized = deg_inv_sqrt_matrix @ adj @ deg_inv_sqrt_matrix

    return adj_normalized


class GCN(nn.Module):
    """
    Graph Convolutional Network completa.

    Arquitectura tipica:
    - 2-3 capas GCN
    - ReLU entre capas
    - Dropout para regularizacion
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Capas GCN
        self.layers = nn.ModuleList()

        # Primera capa: features -> hidden
        self.layers.append(GCNLayer(num_features, hidden_channels))

        # Capas intermedias: hidden -> hidden
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_channels, hidden_channels))

        # Ultima capa: hidden -> classes
        if num_layers > 1:
            self.layers.append(GCNLayer(hidden_channels, num_classes))
        else:
            # Una sola capa: features -> classes
            self.layers[-1] = GCNLayer(num_features, num_classes)

    def forward(self, x: Tensor, adj_normalized: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            adj_normalized: Adyacencia normalizada
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_normalized)

            # Aplicar ReLU y dropout excepto en ultima capa
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


# Ejemplo de uso
def demo_gcn() -> None:
    """Demuestra GCN en grafo simple."""

    # Grafo: Triangulo con un nodo extra
    #     0
    #    /|\
    #   1-+-2
    #     |
    #     3

    adj = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float)

    # Normalizar
    adj_norm = normalize_adjacency(adj)
    print("Matriz normalizada:")
    print(adj_norm.numpy().round(3))

    # Features (4 nodos, 3 features cada uno)
    x = torch.tensor([
        [1.0, 0.0, 0.0],  # Nodo 0: feature A
        [0.0, 1.0, 0.0],  # Nodo 1: feature B
        [0.0, 0.0, 1.0],  # Nodo 2: feature C
        [1.0, 1.0, 0.0],  # Nodo 3: features A+B
    ])

    # Crear modelo (2 clases)
    model = GCN(
        num_features=3,
        num_classes=2,
        hidden_channels=8,
        num_layers=2
    )

    # Forward pass
    output = model(x, adj_norm)
    print(f"\nOutput shape: {output.shape}")
    print(f"Predicciones (logits):\n{output.detach().numpy().round(3)}")

    # Probabilidades
    probs = F.softmax(output, dim=1)
    print(f"Probabilidades:\n{probs.detach().numpy().round(3)}")


if __name__ == "__main__":
    demo_gcn()
```

### Entrenamiento de GCN

```python
"""
Entrenamiento de GCN para clasificacion semi-supervisada.
"""
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import NamedTuple


class TrainingMetrics(NamedTuple):
    """Metricas de entrenamiento."""
    loss: float
    train_acc: float
    val_acc: float
    test_acc: float


def train_epoch(
    model: GCN,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor
) -> float:
    """
    Un epoch de entrenamiento.
    """
    model.train()
    optimizer.zero_grad()

    # Forward
    out = model(x, adj_norm)

    # Loss solo en nodos de entrenamiento
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    # Backward
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: GCN,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor
) -> TrainingMetrics:
    """
    Evalua modelo en train/val/test.
    """
    model.eval()
    out = model(x, adj_norm)
    pred = out.argmax(dim=1)

    # Calcular loss en validacion
    val_loss = F.cross_entropy(out[val_mask], y[val_mask]).item()

    # Accuracy por split
    train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
    val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
    test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

    return TrainingMetrics(
        loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc
    )


def train_gcn(
    model: GCN,
    x: torch.Tensor,
    adj_norm: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20
) -> tuple[GCN, list[TrainingMetrics]]:
    """
    Entrenamiento completo con early stopping.

    Args:
        model: Modelo GCN
        x: Features de nodos
        adj_norm: Adyacencia normalizada
        y: Labels
        train_mask: Mascara de nodos de entrenamiento
        val_mask: Mascara de validacion
        test_mask: Mascara de test
        epochs: Numero maximo de epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Epochs sin mejora para early stopping
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[TrainingMetrics] = []
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, optimizer, x, adj_norm, y, train_mask)

        # Evaluate
        metrics = evaluate(model, x, adj_norm, y, train_mask, val_mask, test_mask)
        history.append(metrics)

        # Early stopping check
        if metrics.val_acc > best_val_acc:
            best_val_acc = metrics.val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}: "
                  f"Loss={train_loss:.4f}, "
                  f"Train={metrics.train_acc:.4f}, "
                  f"Val={metrics.val_acc:.4f}, "
                  f"Test={metrics.test_acc:.4f}")

    # Restaurar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluacion final
    final_metrics = evaluate(model, x, adj_norm, y, train_mask, val_mask, test_mask)
    print(f"\nFinal: Test Accuracy = {final_metrics.test_acc:.4f}")

    return model, history
```

---

## Graph Attention Networks (GAT)

### Motivacion para Atencion

```
LIMITACIONES DE GCN
===================

GCN usa pesos FIJOS basados en grados:

    A_norm[i,j] = 1 / sqrt(d_i * d_j)

Problema: Todos los vecinos contribuyen igual
(ponderado solo por grado, no por relevancia)


Ejemplo:
--------

    [Spam]---[User]---[Normal]
               |
            [Normal]
               |
            [Normal]

Para clasificar [User]:
- GCN: Promedia todos los vecinos por igual
- Pero [Spam] deberia pesar mas si queremos detectar fraude!


SOLUCION: ATENCION
==================

Aprender pesos de atencion dinamicos:

    alpha_ij = Attention(h_i, h_j)

    h'_i = sigma(sum_{j in N(i)} alpha_ij * W * h_j)

Los pesos alpha_ij se aprenden end-to-end!


    [Spam]---alpha=0.6---[User]---alpha=0.2---[Normal]
                           |
                       alpha=0.1
                           |
                        [Normal]
                           |
                       alpha=0.1
                           |
                        [Normal]

Ahora [Spam] tiene mas influencia en la prediccion.
```

### Mecanismo de Atencion en GAT

```
ATENCION EN GAT (Velickovic et al., 2018)
=========================================

Paso 1: Proyeccion lineal
-------------------------
    z_i = W * h_i   para todo nodo i

    h_i in R^F  -->  z_i in R^F'


Paso 2: Calcular coeficientes de atencion
-----------------------------------------
Para cada par de vecinos (i, j):

    e_ij = LeakyReLU(a^T [z_i || z_j])

    Donde:
    - a in R^(2F') es un vector de atencion aprendible
    - || denota concatenacion
    - LeakyReLU con pendiente 0.2


VISUALIZACION:
--------------

    z_i = [0.1, 0.3]      z_j = [0.5, 0.2]
           |                     |
           +----------+----------+
                      |
                      v
              [z_i || z_j] = [0.1, 0.3, 0.5, 0.2]
                      |
                      v
              a^T @ [0.1, 0.3, 0.5, 0.2]  (a aprendido)
                      |
                      v
              LeakyReLU(resultado)
                      |
                      v
                   e_ij = score de atencion


Paso 3: Normalizar con softmax sobre vecinos
--------------------------------------------
    alpha_ij = exp(e_ij) / sum_{k in N(i)} exp(e_ik)

    Esto asegura que sum_j alpha_ij = 1


Paso 4: Agregar features ponderados
-----------------------------------
    h'_i = sigma(sum_{j in N(i)} alpha_ij * z_j)


DIAGRAMA COMPLETO:
------------------

         h_A          h_B          h_C
          |            |            |
          v            v            v
         W*           W*           W*
          |            |            |
          v            v            v
         z_A          z_B          z_C
          |    \    /  |  \    /   |
          |     \  /   |   \  /    |
          |      \/    |    \/     |
          |      /\    |    /\     |
          |     /  \   |   /  \    |
          v    v    v  v  v    v   v
        e_AB      e_BC     e_CA
          |        |        |
          v        v        v
      softmax  softmax  softmax
          |        |        |
          v        v        v
       alpha_AB  alpha_BC  alpha_CA
          |        |        |
          +--------+--------+
                   |
                   v
              Agregacion ponderada
                   |
                   v
                 h'_B
```

---

## Multi-Head Attention en Grafos

### Concepto de Multi-Head

```
MULTI-HEAD ATTENTION
====================

Problema: Una sola cabeza de atencion puede ser limitada

Solucion: K cabezas independientes, cada una aprende
diferentes patrones de atencion


HEAD 1:                HEAD 2:                HEAD K:

    [A]                    [A]                    [A]
   /   \                  /   \                  /   \
  0.7   0.3              0.2   0.8              0.5   0.5
 /       \              /       \              /       \
[B]     [C]            [B]     [C]            [B]     [C]

Atiende a B           Atiende a C           Atiende igual


AGREGACION DE CABEZAS:
----------------------

Opcion 1: Concatenar (capas intermedias)

    h'_i = ||_{k=1}^{K} sigma(sum_j alpha_ij^k * W^k * h_j)

    Dimension output: K * F'


Opcion 2: Promediar (capa final)

    h'_i = sigma(1/K * sum_{k=1}^{K} sum_j alpha_ij^k * W^k * h_j)

    Dimension output: F'


VISUALIZACION DE CONCATENACION:
-------------------------------

    Head 1 output:  [0.1, 0.2]
    Head 2 output:  [0.5, 0.3]     -->  h' = [0.1, 0.2, 0.5, 0.3, 0.7, 0.1]
    Head 3 output:  [0.7, 0.1]           Dimension: 3 * 2 = 6
```

### Intuicion Detras de Multi-Head

```
POR QUE MULTI-HEAD FUNCIONA
===========================

Cada cabeza puede capturar diferentes tipos de relaciones:

EJEMPLO: Red social con diferentes tipos de conexiones

         [Usuario]
        /    |    \
       /     |     \
   amigo   colega  familia
     /       |        \
[User1]  [User2]    [User3]


Head 1: Puede aprender a ponderar "amigos"
Head 2: Puede aprender a ponderar "colegas"
Head 3: Puede aprender a ponderar "familia"


EN CIBERSEGURIDAD:
------------------

         [IP sospechosa]
        /      |       \
       /       |        \
    ssh      http      dns
     /         |         \
[Server1]  [Server2]  [DNS]


Head 1: Atencion a conexiones SSH (alto riesgo)
Head 2: Atencion a patrones HTTP (medio riesgo)
Head 3: Atencion a consultas DNS (anomalias)

Cada cabeza "especializa" en un tipo de relacion.
```

---

## Implementacion de GAT

### Implementacion desde Cero

```python
"""
Implementacion de Graph Attention Network desde cero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GATLayer(nn.Module):
    """
    Capa de Graph Attention Network.

    Implementa atencion sobre vecinos con multi-head opcional.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_slope: float = 0.2,
        bias: bool = True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Transformacion lineal para cada cabeza
        self.W = nn.Parameter(
            torch.empty(num_heads, in_features, out_features)
        )

        # Vectores de atencion (uno por cabeza)
        # Dimension: [num_heads, 2 * out_features]
        self.attention = nn.Parameter(
            torch.empty(num_heads, 2 * out_features)
        )

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(num_heads * out_features))
        elif bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Inicializacion Xavier."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention.unsqueeze(-1))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Aristas [2, num_edges]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Paso 1: Transformacion lineal para cada cabeza
        # x: [N, F] -> [N, H, F']
        # W: [H, F, F']
        h = torch.einsum('nf,hfo->nho', x, self.W)  # [N, H, F']

        # Paso 2: Calcular scores de atencion
        # Para cada arista (i,j): e_ij = a^T [h_i || h_j]

        # Obtener features de nodos fuente y destino
        h_src = h[src]  # [E, H, F']
        h_dst = h[dst]  # [E, H, F']

        # Concatenar: [h_dst || h_src]
        edge_features = torch.cat([h_dst, h_src], dim=-1)  # [E, H, 2*F']

        # Producto punto con vector de atencion
        # attention: [H, 2*F']
        e = torch.einsum('ehf,hf->eh', edge_features, self.attention)  # [E, H]
        e = self.leaky_relu(e)

        # Paso 3: Normalizar con softmax sobre vecinos de cada nodo
        # Necesitamos softmax por nodo destino
        alpha = self._sparse_softmax(e, dst, num_nodes)  # [E, H]

        # Dropout en coeficientes de atencion
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Paso 4: Agregar features ponderados
        # h'_i = sum_j alpha_ij * h_j
        h_src = h[src]  # [E, H, F']
        weighted = alpha.unsqueeze(-1) * h_src  # [E, H, F']

        # Scatter add para agregar por nodo destino
        out = torch.zeros(num_nodes, self.num_heads, self.out_features,
                         device=x.device)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1)
        dst_expanded = dst_expanded.expand(-1, self.num_heads, self.out_features)
        out.scatter_add_(0, dst_expanded, weighted)

        # Paso 5: Concatenar o promediar cabezas
        if self.concat:
            # [N, H, F'] -> [N, H*F']
            out = out.view(num_nodes, -1)
        else:
            # [N, H, F'] -> [N, F']
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _sparse_softmax(
        self,
        scores: Tensor,
        index: Tensor,
        num_nodes: int
    ) -> Tensor:
        """
        Softmax disperso por grupos definidos por index.

        Args:
            scores: Scores [num_edges, num_heads]
            index: Indices de nodos destino [num_edges]
            num_nodes: Numero total de nodos
        """
        # Estabilidad numerica: restar max por grupo
        scores_max = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        scores_max.scatter_reduce_(
            0,
            index.unsqueeze(-1).expand_as(scores),
            scores,
            reduce='amax',
            include_self=False
        )
        scores_max = scores_max[index]
        scores = scores - scores_max

        # exp
        exp_scores = torch.exp(scores)

        # sum por grupo
        exp_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        exp_sum.scatter_add_(
            0,
            index.unsqueeze(-1).expand_as(exp_scores),
            exp_scores
        )
        exp_sum = exp_sum[index]

        # Normalizar
        return exp_scores / (exp_sum + 1e-8)


class GAT(nn.Module):
    """
    Graph Attention Network completa.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.6
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.layers = nn.ModuleList()

        # Primera capa: multi-head con concatenacion
        self.layers.append(GATLayer(
            in_features=num_features,
            out_features=hidden_channels,
            num_heads=num_heads,
            concat=True,
            dropout=dropout
        ))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(
                in_features=hidden_channels * num_heads,
                out_features=hidden_channels,
                num_heads=num_heads,
                concat=True,
                dropout=dropout
            ))

        # Ultima capa: multi-head con promedio
        if num_layers > 1:
            self.layers.append(GATLayer(
                in_features=hidden_channels * num_heads,
                out_features=num_classes,
                num_heads=1,  # O num_heads con concat=False
                concat=False,  # Promedio para output final
                dropout=dropout
            ))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass."""
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def demo_gat() -> None:
    """Demuestra GAT en grafo simple."""
    # Grafo: 4 nodos
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 2],
        [1, 0, 2, 1, 3, 2, 2, 0]
    ])

    # Features
    x = torch.randn(4, 8)

    # Modelo
    model = GAT(
        num_features=8,
        num_classes=3,
        hidden_channels=16,
        num_heads=4,
        num_layers=2
    )

    # Forward
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [4, 3]

    # Predicciones
    pred = output.argmax(dim=1)
    print(f"Predicciones: {pred}")


if __name__ == "__main__":
    demo_gat()
```

### Visualizacion de Atencion

```python
"""
Visualizacion de coeficientes de atencion en GAT.
"""
import torch
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional


def visualize_attention(
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    node_labels: Optional[list[str]] = None,
    figsize: tuple[int, int] = (10, 8)
) -> None:
    """
    Visualiza grafo con pesos de atencion en aristas.

    Args:
        edge_index: [2, E] indices de aristas
        attention_weights: [E] pesos de atencion
        node_labels: Etiquetas opcionales para nodos
    """
    # Crear grafo NetworkX
    G = nx.DiGraph()

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    weights = attention_weights.detach().numpy()

    # Agregar aristas con pesos
    for i, (s, d) in enumerate(zip(src, dst)):
        G.add_edge(s, d, weight=weights[i])

    # Layout
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color='lightblue')

    # Dibujar aristas con grosor proporcional a atencion
    edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color='gray',
        alpha=0.7,
        arrows=True,
        arrowsize=20
    )

    # Labels de nodos
    if node_labels:
        labels = {i: label for i, label in enumerate(node_labels)}
    else:
        labels = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=12)

    # Labels de aristas (pesos de atencion)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)

    ax.set_title("GAT Attention Weights", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def extract_attention_from_gat(
    model: GAT,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0
) -> torch.Tensor:
    """
    Extrae pesos de atencion de una capa GAT.

    Nota: Requiere modificar GAT para guardar atencion.
    Esta es una version simplificada.
    """
    # En una implementacion real, modificarias GATLayer
    # para retornar attention_weights durante forward

    # Aqui simulamos con pesos aleatorios para demo
    num_edges = edge_index.size(1)
    attention_weights = torch.softmax(torch.randn(num_edges), dim=0)

    return attention_weights
```

---

## Comparativa GCN vs GAT

### Resumen de Diferencias

```
GCN vs GAT: COMPARATIVA DETALLADA
=================================

+--------------------+---------------------------+---------------------------+
| Aspecto            | GCN                       | GAT                       |
+--------------------+---------------------------+---------------------------+
| Pesos de arista    | Fijos (basados en grado)  | Aprendidos (atencion)     |
| Complejidad        | O(|E| * F * F')           | O(|E| * F * F' * H)       |
| Parametros         | W (F x F')                | W + a (+ por cabeza)      |
| Multi-head         | No                        | Si (K cabezas)            |
| Interpretabilidad  | Baja                      | Alta (ver atencion)       |
| Overfitting risk   | Menor                     | Mayor (mas parametros)    |
| Inductive          | Limitado                  | Si (atencion generaliza)  |
+--------------------+---------------------------+---------------------------+


CUANDO USAR CADA UNO:
---------------------

GCN es mejor cuando:
- Grafo homogeneo (todos los nodos/aristas similares)
- Dataset pequeno (riesgo de overfitting)
- Computacion limitada
- No necesitas interpretabilidad de aristas

GAT es mejor cuando:
- Diferentes aristas tienen diferente importancia
- Quieres interpretar que vecinos importan
- Dataset grande (puede aprender atencion)
- Tipos heterogeneos de relaciones


EJEMPLO: Cuando GAT supera a GCN
--------------------------------

Red de citas academicas:
- Nodo A cita a B, C, D, E
- B es paper muy relevante
- C, D, E son citas genericas

GCN: Promedia B, C, D, E por igual
GAT: Aprende que B es mas importante -> alpha_AB >> alpha_AC

Resultado: GAT captura mejor la relevancia!
```

### Diagrama Comparativo

```
PROPAGACION: GCN vs GAT
=======================

                GCN                              GAT

         Features h_j                      Features h_j
             |                                  |
             v                                  v
        W @ h_j                           W @ h_j = z_j
             |                                  |
             v                             +----+----+
    1/sqrt(d_i * d_j)                      |         |
    (peso FIJO)                         z_i, z_j    z_k, z_j
             |                              |         |
             v                              v         v
         Suma                        Attention    Attention
             |                         e_ij        e_kj
             v                              |         |
        sigma(...)                          +----+----+
             |                                   |
             v                                   v
          h_i'                              Softmax
                                                |
                                                v
                                        alpha (APRENDIDO)
                                                |
                                                v
                                          Suma ponderada
                                                |
                                                v
                                            h_i'


GCN: Pesos deterministas         GAT: Pesos aprendidos
     Estructura fija                  Estructura adaptativa
```

---

## Implementacion con PyTorch Geometric

### GCN con PyG

```python
"""
Implementacion de GCN y GAT usando PyTorch Geometric.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import Data
from typing import Optional


class GCN_PyG(torch.nn.Module):
    """GCN usando PyTorch Geometric."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Primera capa
        self.convs.append(GCNConv(num_features, hidden_channels))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Ultima capa
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT_PyG(torch.nn.Module):
    """GAT usando PyTorch Geometric."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.6
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Primera capa: multi-head
        self.convs.append(GATConv(
            num_features,
            hidden_channels,
            heads=num_heads,
            dropout=dropout
        ))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_channels * num_heads,
                hidden_channels,
                heads=num_heads,
                dropout=dropout
            ))

        # Ultima capa: single head o promedio
        if num_layers > 1:
            self.convs.append(GATConv(
                hidden_channels * num_heads,
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout
            ))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def train_and_evaluate_pyg(
    model_class: type,
    dataset_name: str = "Cora",
    epochs: int = 200,
    lr: float = 0.01,
    **model_kwargs
) -> dict[str, float]:
    """
    Entrena y evalua modelo en dataset de PyG.
    """
    # Cargar dataset
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]

    # Crear modelo
    model = model_class(
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        **model_kwargs
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Training loop
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                best_test_acc = test_acc.item()

    return {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc
    }


def compare_gcn_gat() -> None:
    """Compara GCN vs GAT en Cora."""
    print("Comparando GCN vs GAT en Cora dataset...\n")

    # GCN
    print("Training GCN...")
    gcn_results = train_and_evaluate_pyg(
        GCN_PyG,
        dataset_name="Cora",
        hidden_channels=64,
        num_layers=2
    )
    print(f"GCN - Val: {gcn_results['best_val_acc']:.4f}, "
          f"Test: {gcn_results['best_test_acc']:.4f}")

    # GAT
    print("\nTraining GAT...")
    gat_results = train_and_evaluate_pyg(
        GAT_PyG,
        dataset_name="Cora",
        hidden_channels=8,
        num_heads=8,
        num_layers=2
    )
    print(f"GAT - Val: {gat_results['best_val_acc']:.4f}, "
          f"Test: {gat_results['best_test_acc']:.4f}")


if __name__ == "__main__":
    compare_gcn_gat()
```

### Graph-Level Classification

```python
"""
Clasificacion a nivel de grafo usando GCN/GAT.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class GraphClassifier(torch.nn.Module):
    """
    Clasificador de grafos completos.

    Arquitectura:
    1. Capas de convolucion para embeddings de nodos
    2. Pooling global para embedding de grafo
    3. MLP para clasificacion
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        pooling: str = "mean",
        conv_type: str = "GCN"
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.convs = torch.nn.ModuleList()

        ConvClass = GCNConv if conv_type == "GCN" else GATConv

        # Capas de convolucion
        self.convs.append(ConvClass(num_features, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(ConvClass(hidden_channels, hidden_channels))

        # MLP clasificador
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Aristas
            batch: Indica a que grafo pertenece cada nodo
        """
        # Convolucion
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Pooling global
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        else:
            # Combinar mean y max
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = x_mean + x_max

        # Clasificacion
        return self.classifier(x)


def train_graph_classifier() -> None:
    """Entrena clasificador de grafos en MUTAG."""
    # Dataset: Moleculas (mutagenic vs no mutagenic)
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

    # Split
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Modelo
    model = GraphClassifier(
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        hidden_channels=64,
        num_layers=3,
        pooling="mean",
        conv_type="GCN"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(100):
        model.train()
        total_loss = 0

        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data in test_loader:
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == data.y).sum().item()
                    total += data.y.size(0)

            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Test Acc={correct/total:.4f}")


if __name__ == "__main__":
    train_graph_classifier()
```

---

## Aplicaciones en Ciberseguridad

### Deteccion de Phishing con GAT

```python
"""
Detector de phishing usando GAT.

Modela:
- Nodos: URLs, dominios, IPs, certificados
- Aristas: redirecciones, DNS, hosted_on
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias


class NodeType(Enum):
    URL = auto()
    DOMAIN = auto()
    IP = auto()
    CERTIFICATE = auto()


class EdgeType(Enum):
    REDIRECTS_TO = auto()
    RESOLVES_TO = auto()
    HOSTED_ON = auto()
    HAS_CERT = auto()


@dataclass
class PhishingFeatures:
    """Features para deteccion de phishing."""
    # URL features
    url_length: float
    has_ip_in_url: bool
    num_dots: int
    has_suspicious_words: bool
    path_length: float

    # Domain features
    domain_age_days: int
    alexa_rank: int
    is_https: bool

    # SSL features
    cert_valid_days: int
    is_self_signed: bool


class PhishingGAT(nn.Module):
    """
    GAT para deteccion de phishing.
    """

    def __init__(
        self,
        url_features: int,
        domain_features: int,
        hidden_channels: int = 32,
        num_heads: int = 4
    ) -> None:
        super().__init__()

        # Embeddings por tipo de nodo
        self.url_embed = nn.Linear(url_features, hidden_channels)
        self.domain_embed = nn.Linear(domain_features, hidden_channels)
        self.ip_embed = nn.Linear(4, hidden_channels)  # IP como 4 octetos
        self.cert_embed = nn.Linear(3, hidden_channels)  # cert features

        # Capas GAT
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)  # Phishing / Legitimate
        )

    def forward(
        self,
        x_url: torch.Tensor,
        x_domain: torch.Tensor,
        x_ip: torch.Tensor,
        x_cert: torch.Tensor,
        edge_index: torch.Tensor,
        url_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_url: Features de URLs
            x_domain: Features de dominios
            x_ip: Features de IPs
            x_cert: Features de certificados
            edge_index: Conexiones
            url_indices: Indices de nodos URL a clasificar
        """
        # Embeber cada tipo de nodo
        h_url = F.relu(self.url_embed(x_url))
        h_domain = F.relu(self.domain_embed(x_domain))
        h_ip = F.relu(self.ip_embed(x_ip))
        h_cert = F.relu(self.cert_embed(x_cert))

        # Concatenar todos los nodos
        x = torch.cat([h_url, h_domain, h_ip, h_cert], dim=0)

        # GAT layers
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)

        # Clasificar solo nodos URL
        url_embeddings = x[url_indices]
        return self.classifier(url_embeddings)


def create_phishing_graph_example() -> Data:
    """
    Crea un grafo de ejemplo para deteccion de phishing.
    """
    # Estructura del grafo:
    #
    #  [URL1]---->[Domain1]---->[IP1]
    #    |           |
    #    v           v
    #  [URL2]---->[Domain2]---->[Cert1]
    #    |
    #    v
    #  [URL3]---->[Domain3]

    # Features de URLs (5 features cada una)
    x_url = torch.tensor([
        [50.0, 0.0, 3.0, 0.0, 20.0],   # URL1: Normal
        [120.0, 1.0, 8.0, 1.0, 60.0],  # URL2: Sospechosa
        [80.0, 0.0, 5.0, 1.0, 30.0],   # URL3: Sospechosa
    ], dtype=torch.float)

    # Features de dominios (3 features)
    x_domain = torch.tensor([
        [365.0, 1000.0, 1.0],   # Domain1: Viejo, rankeado, HTTPS
        [7.0, 0.0, 0.0],        # Domain2: Nuevo, sin rank, HTTP
        [30.0, 0.0, 1.0],       # Domain3: Reciente
    ], dtype=torch.float)

    # Features de IPs (4 octetos normalizados)
    x_ip = torch.tensor([
        [192/255, 168/255, 1/255, 1/255],  # IP privada
    ], dtype=torch.float)

    # Features de certificados
    x_cert = torch.tensor([
        [365.0, 0.0, 1.0],  # Cert valido
    ], dtype=torch.float)

    # Indices: URLs 0-2, Domains 3-5, IP 6, Cert 7
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 3, 4],  # src
        [3, 4, 5, 6, 7, 6, 4, 3]   # dst
    ], dtype=torch.long)

    # Labels: 0=legitimo, 1=phishing
    y = torch.tensor([0, 1, 1], dtype=torch.long)

    return Data(
        x_url=x_url,
        x_domain=x_domain,
        x_ip=x_ip,
        x_cert=x_cert,
        edge_index=edge_index,
        y=y
    )
```

### Deteccion de Amenazas en Red

```python
"""
Sistema de deteccion de amenazas en red usando GCN.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

TensorType: TypeAlias = torch.Tensor


@dataclass
class NetworkFlowFeatures:
    """Features de un flujo de red."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_sent: int
    bytes_received: int
    packets: int
    duration: float
    flags: int


class NetworkThreatDetector(nn.Module):
    """
    Detector de amenazas basado en grafo de trafico de red.

    Nodos: IPs
    Aristas: Conexiones con features de flujo
    Tarea: Clasificar IPs como maliciosas/benignas
    """

    def __init__(
        self,
        flow_features: int,
        ip_features: int = 8,
        hidden_channels: int = 32
    ) -> None:
        super().__init__()

        # Encoder de features de flujo (para aristas)
        self.edge_encoder = nn.Sequential(
            nn.Linear(flow_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Encoder de features de IP
        self.ip_encoder = nn.Linear(ip_features, hidden_channels)

        # Capas GCN
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 4)  # Normal, Scanner, C&C, Botnet
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        edge_attr: TensorType
    ) -> TensorType:
        """
        Forward pass.

        Args:
            x: Features de IPs [num_ips, ip_features]
            edge_index: Conexiones [2, num_connections]
            edge_attr: Features de flujos [num_connections, flow_features]
        """
        # Encode IP features
        x = self.ip_encoder(x)

        # Encode edge features (no usado directamente en GCN basico)
        # En una version mas avanzada, usariamos MPNN con edge features

        # GCN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)

        # Clasificar
        return self.classifier(x)


def extract_ip_features(ip_stats: dict) -> TensorType:
    """
    Extrae features de una IP para el grafo.

    Args:
        ip_stats: Estadisticas agregadas de la IP
    """
    features = torch.tensor([
        ip_stats.get('total_connections', 0) / 1000,  # Normalizado
        ip_stats.get('unique_ports', 0) / 65535,
        ip_stats.get('unique_destinations', 0) / 1000,
        ip_stats.get('bytes_sent', 0) / 1e9,
        ip_stats.get('bytes_received', 0) / 1e9,
        ip_stats.get('avg_connection_duration', 0) / 3600,
        ip_stats.get('failed_connections', 0) / 100,
        ip_stats.get('entropy_of_ports', 0) / 16,  # Max entropy
    ], dtype=torch.float)

    return features


def build_network_graph(flows: list[NetworkFlowFeatures]) -> dict:
    """
    Construye grafo de red desde flujos.
    """
    # Mapear IPs a indices
    ips = set()
    for flow in flows:
        ips.add(flow.src_ip)
        ips.add(flow.dst_ip)

    ip_to_idx = {ip: idx for idx, ip in enumerate(sorted(ips))}

    # Construir aristas y features
    edges_src = []
    edges_dst = []
    edge_features = []

    for flow in flows:
        src_idx = ip_to_idx[flow.src_ip]
        dst_idx = ip_to_idx[flow.dst_ip]

        edges_src.append(src_idx)
        edges_dst.append(dst_idx)

        # Normalizar features del flujo
        features = [
            flow.src_port / 65535,
            flow.dst_port / 65535,
            1.0 if flow.protocol == 'TCP' else 0.0,
            flow.bytes_sent / 1e6,
            flow.bytes_received / 1e6,
            flow.packets / 1000,
            flow.duration / 3600,
            flow.flags / 256
        ]
        edge_features.append(features)

    return {
        'ip_to_idx': ip_to_idx,
        'edge_index': torch.tensor([edges_src, edges_dst], dtype=torch.long),
        'edge_attr': torch.tensor(edge_features, dtype=torch.float)
    }
```

---

## Ejercicios Practicos

### Ejercicio 1: Implementar Normalizacion de Adyacencia

```python
"""
Ejercicio 1: Implementar diferentes normalizaciones de adyacencia.

Objetivo: Implementar y comparar:
1. Sin normalizacion: A
2. Row normalization: D^(-1) A
3. Symmetric normalization: D^(-1/2) A D^(-1/2)
4. Con self-loops: D~^(-1/2) A~ D~^(-1/2)
"""

def ejercicio_1_template() -> None:
    """
    TODO: Implementar las funciones de normalizacion y
    comparar su efecto en la propagacion de mensajes.

    Grafo de prueba:
        0---1---2
        |   |
        3---4

    Matriz A:
    [[0,1,0,1,0],
     [1,0,1,0,1],
     [0,1,0,0,0],
     [1,0,0,0,1],
     [0,1,0,1,0]]

    Implementar:
    1. normalize_row(A) -> D^(-1) A
    2. normalize_symmetric(A) -> D^(-1/2) A D^(-1/2)
    3. normalize_gcn(A) -> D~^(-1/2) A~ D~^(-1/2)

    Verificar que la suma de cada fila sea <= 1 (probabilidad).
    """
    import torch

    A = torch.tensor([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0]
    ], dtype=torch.float)

    # Tu implementacion aqui
    pass
```

### Ejercicio 2: Visualizar Atencion de GAT

```python
"""
Ejercicio 2: Extraer y visualizar pesos de atencion de GAT.

Objetivo: Modificar GATLayer para retornar attention weights
y visualizar que nodos reciben mas atencion.
"""

def ejercicio_2_template() -> None:
    """
    TODO:
    1. Modificar GATLayer.forward para retornar (output, attention_weights)
    2. Entrenar GAT en Cora
    3. Extraer attention weights de la primera capa
    4. Visualizar:
       - Histograma de pesos de atencion
       - Top-5 aristas con mayor atencion
       - Grafo coloreado por atencion recibida

    Preguntas a responder:
    - Hay nodos que reciben mucha mas atencion?
    - Correlaciona con el grado del nodo?
    - Diferentes cabezas atienden a diferentes patrones?
    """
    pass
```

### Ejercicio 3: Detector de Anomalias con GCN

```python
"""
Ejercicio 3: Implementar detector de anomalias en red.

Objetivo: Usar GCN para detectar nodos anomalos en un grafo
de conexiones de red (IPs escaneando, botnet, etc.)
"""

def ejercicio_3_template() -> None:
    """
    TODO:
    1. Generar grafo sintetico de red con:
       - 100 IPs normales (conectan a 2-5 destinos)
       - 5 IPs scanner (conectan a 50+ destinos)
       - 3 IPs botnet (patron de estrella hacia C&C)

    2. Extraer features por IP:
       - Grado de salida/entrada
       - Numero de puertos unicos
       - Clustering coefficient
       - Centralidad

    3. Entrenar GCN para clasificar:
       - 0: Normal
       - 1: Scanner
       - 2: Botnet

    4. Evaluar con precision/recall por clase

    Bonus: Usar GAT y analizar que conexiones reciben
    mas atencion para detectar anomalias.
    """
    pass
```

---

## Resumen

```
RESUMEN: GCN Y GAT
==================

GCN (Graph Convolutional Network):
----------------------------------
- Basado en aproximacion de filtro espectral
- Regla: H' = sigma(D~^(-1/2) A~ D~^(-1/2) H W)
- Pesos de arista FIJOS (basados en grado)
- Eficiente: O(|E|)
- Limitacion: Todos los vecinos contribuyen igual


GAT (Graph Attention Network):
------------------------------
- Mecanismo de atencion sobre vecinos
- Regla: h'_i = sigma(sum_j alpha_ij W h_j)
- Pesos de arista APRENDIDOS (alpha_ij)
- Multi-head attention para robustez
- Ventaja: Aprende importancia de cada vecino


CUANDO USAR CADA UNO:
---------------------
GCN: Grafos homogeneos, datasets pequenos, eficiencia
GAT: Aristas heterogeneas, interpretabilidad, datasets grandes


APLICACIONES EN CIBERSEGURIDAD:
-------------------------------
- Deteccion de phishing (grafo URL-Domain-IP)
- Analisis de trafico (grafo IP-conexiones)
- Deteccion de malware (call graphs)
- Identificacion de botnets (patron C&C)


SIGUIENTE TEMA:
---------------
GraphSAGE y metodos avanzados:
- Sampling de vecinos para escalabilidad
- Aprendizaje inductivo (generalizar a nuevos nodos)
- node2vec y DeepWalk
```

---

## Referencias

1. **Kipf & Welling (2017)** - "Semi-Supervised Classification with Graph Convolutional Networks" - Paper original de GCN
2. **Velickovic et al. (2018)** - "Graph Attention Networks" - Paper original de GAT
3. **Defferrard et al. (2016)** - "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" - ChebNet
4. **Hammond et al. (2011)** - "Wavelets on Graphs via Spectral Graph Theory" - Base teorica espectral
5. **PyTorch Geometric** - https://pytorch-geometric.readthedocs.io/
