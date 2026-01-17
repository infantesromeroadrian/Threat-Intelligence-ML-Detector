# GraphSAGE y Metodos Avanzados de Aprendizaje en Grafos

## Indice

1. [Limitaciones de GCN: El Problema de Escalabilidad](#limitaciones-de-gcn)
2. [GraphSAGE: Sampling de Vecinos](#graphsage-sampling-de-vecinos)
3. [Agregadores en GraphSAGE](#agregadores-en-graphsage)
4. [Aprendizaje Inductivo vs Transductivo](#aprendizaje-inductivo-vs-transductivo)
5. [Implementacion de GraphSAGE](#implementacion-de-graphsage)
6. [Node2Vec: Random Walks en Grafos](#node2vec-random-walks-en-grafos)
7. [DeepWalk y LINE](#deepwalk-y-line)
8. [Metodos de Negative Sampling](#metodos-de-negative-sampling)
9. [Mini-Batch Training para Grafos Grandes](#mini-batch-training-para-grafos-grandes)
10. [Aplicaciones en Ciberseguridad](#aplicaciones-en-ciberseguridad)
11. [Ejercicios Practicos](#ejercicios-practicos)

---

## Limitaciones de GCN

### El Problema del Full-Batch

```
PROBLEMA DE ESCALABILIDAD EN GCN
================================

GCN requiere procesar TODO el grafo en cada paso:

    H^(l+1) = sigma(A_norm @ H^(l) @ W)

Para un grafo con N nodos:
- Memoria: O(N * F) para H
- Multiplicacion: O(|E| * F) para A @ H
- Gradientes: Backprop a traves de todo el grafo


EJEMPLO PRACTICO:
-----------------

Dataset         | Nodos    | Aristas    | Memoria GCN
----------------+----------+------------+-------------
Cora            | 2,708    | 10,556     | ~10 MB
Pubmed          | 19,717   | 88,648     | ~100 MB
Reddit          | 233,000  | 114M       | ~10 GB
Twitter         | 41M      | 1.5B       | ~500 GB (!)
Facebook        | 2B       | 100B+      | Imposible


PROBLEMAS ADICIONALES:
----------------------

1. EXPANSION DE VECINDARIO (Neighborhood Explosion)

   Capa 1: Nodo v tiene 10 vecinos
   Capa 2: Cada vecino tiene 10 vecinos -> 100 nodos
   Capa 3: -> 1000 nodos
   Capa k: -> O(d^k) nodos

   Para calcular embedding de 1 nodo con k capas,
   necesitamos embeddings de d^k nodos!

   Visualizacion:
                       Capa 3
                     /  |  \
                    /   |   \
                 [   100 nodos   ]
                  \   /   \   /
                   \ /     \ /
                 [  10 nodos  ]    Capa 2
                    \   |   /
                     \  |  /
                       [v]         Capa 1


2. NO SOPORTA NODOS NUEVOS (Transductivo)

   GCN aprende embeddings FIJOS para cada nodo
   Si llega un nodo nuevo, hay que re-entrenar!


3. MEMORIA DE GPU LIMITADA

   GPUs tipicas: 8-24 GB
   Grafos reales: 100+ GB
   No cabe!
```

### La Solucion: Sampling

```
SOLUCION: MUESTREO DE VECINOS
=============================

En lugar de usar TODOS los vecinos,
muestrear un subconjunto fijo.

GCN Original:
-------------
h_v = AGG({h_u : u in N(v)})   <- Todos los vecinos

GraphSAGE:
----------
h_v = AGG({h_u : u in SAMPLE(N(v), k)})   <- k vecinos aleatorios


VENTAJAS:
---------
1. Complejidad fija por nodo: O(k) no O(degree)
2. Mini-batch training posible
3. Inductivo: Funciona con nodos nuevos


VISUALIZACION:
--------------

GCN (todos los vecinos):          GraphSAGE (k=2 vecinos):

       [N1]                              [N1]
       / |                                |
      /  |                                |
    [N2][N3]                            [N3]
     |   |                                |
     |   |                                |
    [N4][N5][N6]    ====>               [N5]
      \ | /                               |
       \|/                                |
       [v]                               [v]

Procesa 6 vecinos                 Procesa 2 vecinos
```

---

## GraphSAGE: Sampling de Vecinos

### Arquitectura de GraphSAGE

```
GRAPHSAGE: SAMPLING AND AGGREGATING
===================================

Idea clave: Aprender funcion agregadora, no embeddings fijos

Algoritmo:
----------
1. Para cada nodo v:
   a. Muestrear k vecinos de N(v)
   b. Agregar features de vecinos muestreados
   c. Concatenar con features propias
   d. Transformar con MLP

2. Repetir para cada capa

3. Loss: Supervisado o no supervisado


PSEUDOCODIGO:
-------------

function GRAPHSAGE(v, K_capas):
    h_v^0 = x_v  # Features iniciales

    for k = 1 to K:
        # Muestrear vecinos
        N_k = SAMPLE(N(v), num_samples[k])

        # Recursivamente obtener embeddings de vecinos
        h_N = {GRAPHSAGE(u, k-1) : u in N_k}

        # Agregar
        h_neighbors = AGGREGATE_k(h_N)

        # Combinar con self
        h_v^k = sigma(W^k @ CONCAT(h_v^(k-1), h_neighbors))

        # Normalizar (opcional)
        h_v^k = h_v^k / ||h_v^k||_2

    return h_v^K


DIAGRAMA DE FLUJO:
------------------

Capa 0 (features)     Capa 1              Capa 2 (final)

    [h_a^0]              |                    |
    [h_b^0]     sample   |                    |
    [h_c^0]   --------> AGG             AGG --|
    [h_d^0]              |       sample  |    |
                         v         |     |    v
    [h_v^0] ---------> CONCAT --> h_v^1 --> CONCAT --> h_v^2
                         |                    |
                        MLP                  MLP


NUMERO DE MUESTRAS POR CAPA (tipico):
-------------------------------------
- Capa 1: sample_size = 25 (vecinos directos)
- Capa 2: sample_size = 10 (vecinos de vecinos)

Total nodos procesados: 25 * 10 = 250 (vs miles con GCN)
```

### Matematica de GraphSAGE

```
FORMULACION MATEMATICA
======================

Para cada capa k:

1. AGREGACION de vecinos:
   h_N(v)^k = AGGREGATE_k({h_u^(k-1) : u in S_k(v)})

   Donde S_k(v) = SAMPLE(N(v), s_k)

2. CONCATENACION con self:
   h_v^k = sigma(W^k @ [h_v^(k-1) || h_N(v)^k])

3. NORMALIZACION L2 (opcional pero recomendada):
   h_v^k = h_v^k / ||h_v^k||_2


DIFERENCIA CON GCN:
-------------------

GCN:
    h_v^(k+1) = sigma(SUM_{u in N(v)} (1/c) * W * h_u^k)

    - Promedio ponderado de vecinos (incluyendo self)
    - Pesos implicitamente combinan self y neighbors

GraphSAGE:
    h_v^(k+1) = sigma(W * [h_v^k || AGG(neighbors)])

    - SEPARACION explicita de self y neighbors
    - Concatenacion mantiene informacion propia
    - Mas expresivo


INTUICION DE LA CONCATENACION:
------------------------------

GCN mezcla todo:
    [self + neighbors] --W--> output

GraphSAGE separa:
    [self] --|
             |--> CONCAT --> W --> output
 [neighbors]-|

El modelo puede aprender:
- Que informacion tomar de si mismo
- Que informacion tomar de vecinos
- Como combinarlas
```

---

## Agregadores en GraphSAGE

### Tipos de Agregadores

```
AGREGADORES EN GRAPHSAGE
========================

1. MEAN AGGREGATOR
------------------
El mas simple: promedio de embeddings de vecinos

    AGG_mean = (1/|S|) * SUM_{u in S} h_u

    Caracteristicas:
    - Simple y eficiente
    - Simetrico (invariante a orden)
    - Puede perder informacion (promedia todo)


2. POOLING AGGREGATOR (MAX)
---------------------------
Aplicar transformacion y tomar maximo elemento a elemento

    AGG_pool = MAX({sigma(W_pool * h_u + b) : u in S})

    Caracteristicas:
    - Captura features mas "salientes"
    - Mas expresivo que mean
    - Puede capturar outliers


3. LSTM AGGREGATOR
------------------
Usar LSTM sobre vecinos (requiere orden)

    AGG_lstm = LSTM([h_u1, h_u2, ..., h_uk])

    Nota: Los vecinos no tienen orden natural,
    se usa permutacion aleatoria.

    Caracteristicas:
    - Muy expresivo
    - Costoso computacionalmente
    - Sensible al orden (mitigado con aleatoriedad)


4. GCN AGGREGATOR (variante)
----------------------------
Similar a GCN pero con skip connection

    AGG_gcn = sigma(W * MEAN({h_v} U {h_u : u in S}))

    Incluye el nodo mismo en la agregacion.


COMPARATIVA:
------------

+------------+-------------+---------------+-------------+
| Agregador  | Complejidad | Expresividad  | Cuando usar |
+------------+-------------+---------------+-------------+
| Mean       | O(k*d)      | Baja          | Default     |
| MaxPool    | O(k*d*d')   | Media         | Features    |
| LSTM       | O(k*d*d')   | Alta          | Secuencias  |
| GCN-like   | O(k*d)      | Media         | Homogeneos  |
+------------+-------------+---------------+-------------+
```

### Visualizacion de Agregadores

```
COMPARATIVA VISUAL DE AGREGADORES
=================================

Vecinos muestreados con embeddings:
h_1 = [1.0, 0.2, 0.8]
h_2 = [0.3, 0.9, 0.1]
h_3 = [0.7, 0.5, 0.6]


MEAN:
-----
h_agg = [0.67, 0.53, 0.50]   <- Promedio simple

    h_1 --|
    h_2 --|---> MEAN --> [0.67, 0.53, 0.50]
    h_3 --|


MAX POOLING:
------------
Primero transformar, luego max:

    h_1 --> sigma(W*h_1) = [0.8, 1.2, 0.3]
    h_2 --> sigma(W*h_2) = [0.5, 0.7, 0.9]
    h_3 --> sigma(W*h_3) = [0.9, 0.4, 0.6]

    MAX element-wise:
    h_agg = [0.9, 1.2, 0.9]   <- Toma lo mas "saliente"


LSTM:
-----
Procesar secuencialmente (orden aleatorio):

    h_2 --> LSTM --> hidden_1
    h_1 --> LSTM --> hidden_2
    h_3 --> LSTM --> hidden_3 = h_agg

    El estado final resume la secuencia.


INTUICION EN CIBERSEGURIDAD:
----------------------------

Para detectar IP maliciosa basada en conexiones:

Vecinos (destinos):
- Web server (normal)
- DNS server (normal)
- C&C server (malicioso!)
- Mail server (normal)

MEAN: Promedia todo, el C&C se "diluye"
       Resultado: Parece normal

MAX:  Toma caracteristicas extremas
       Resultado: Detecta anomalia (C&C destacado)

En este caso, MAX pooling es mejor!
```

---

## Aprendizaje Inductivo vs Transductivo

### Definiciones

```
INDUCTIVO vs TRANSDUCTIVO
=========================

TRANSDUCTIVO (GCN tradicional):
-------------------------------
- Entrena viendo TODO el grafo
- No puede predecir nodos nuevos sin re-entrenar
- Embeddings fijos para cada nodo

    Entrenamiento:
    [Grafo completo] --> [Modelo] --> [Embeddings fijos]

    Inferencia nodo nuevo:
    [Nodo nuevo] --> ??? --> FALLA (no tiene embedding)


INDUCTIVO (GraphSAGE):
----------------------
- Aprende FUNCION agregadora
- Puede aplicarse a nodos nunca vistos
- Genera embeddings on-the-fly

    Entrenamiento:
    [Subgrafos] --> [Modelo: funcion AGG] --> [Aprender AGG]

    Inferencia nodo nuevo:
    [Nodo nuevo + vecinos] --> [AGG] --> [Embedding generado]


VISUALIZACION:
--------------

Tiempo 1: Entrenamiento

    [A]---[B]---[C]         GraphSAGE aprende:
     |     |     |          "Como agregar vecinos"
    [D]---[E]---[F]


Tiempo 2: Llega nodo nuevo [G]

    [A]---[B]---[C]
     |     |     |
    [D]---[E]---[F]---[G]   <- NUEVO!

    GCN: No puede procesar [G] (no tiene embedding)
    GraphSAGE: Agrega features de [F] --> Genera h_G


CASOS DE USO:
-------------

TRANSDUCTIVO (GCN):
- Grafos estaticos
- Todos los nodos conocidos
- Research/benchmarks

INDUCTIVO (GraphSAGE):
- Grafos dinamicos (nuevos usuarios, IPs, etc.)
- Produccion con datos en tiempo real
- Transferencia entre grafos similares
```

### Ejemplo Practico

```
EJEMPLO: DETECCION DE FRAUDE EN TIEMPO REAL
===========================================

Escenario: Sistema de pagos con usuarios nuevos cada dia

ENFOQUE TRANSDUCTIVO (No viable):
---------------------------------

Dia 1: Entrenar con usuarios {A, B, C, D}
       Modelo aprende embeddings fijos

Dia 2: Nuevo usuario E hace transaccion sospechosa
       PROBLEMA: E no tiene embedding!
       SOLUCION: Re-entrenar todo (costoso)


ENFOQUE INDUCTIVO (GraphSAGE):
------------------------------

Dia 1: Entrenar con usuarios {A, B, C, D}
       Modelo aprende: "Como combinar features de vecinos"

Dia 2: Nuevo usuario E
       - E conecta con A y C (transacciones)
       - Aplicar AGG(h_A, h_C) --> h_E
       - Clasificar E como fraudulento o no

       SIN re-entrenar!


FLUJO EN PRODUCCION:
--------------------

    [Nueva transaccion]
           |
           v
    [Identificar usuario]
           |
           v
    [Obtener vecinos del grafo]
           |
           v
    [Sample k vecinos]
           |
           v
    [Aplicar GraphSAGE] --> [Embedding del usuario]
           |
           v
    [Clasificador] --> [Fraude / No fraude]
           |
           v
    [Alerta si fraude]


Latencia tipica: < 100ms (inferencia inductiva)
```

---

## Implementacion de GraphSAGE

### Implementacion desde Cero

```python
"""
Implementacion de GraphSAGE desde cero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Literal
import random


class NeighborSampler:
    """
    Muestrea vecinos para cada nodo en el grafo.
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int
    ) -> None:
        """
        Args:
            edge_index: [2, num_edges] aristas del grafo
            num_nodes: Numero total de nodos
        """
        self.num_nodes = num_nodes

        # Construir lista de adyacencia para sampling eficiente
        self.adj_list: dict[int, list[int]] = {i: [] for i in range(num_nodes)}

        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            self.adj_list[d].append(s)  # Vecinos de entrada

    def sample(self, nodes: list[int], num_samples: int) -> list[list[int]]:
        """
        Muestrea vecinos para una lista de nodos.

        Args:
            nodes: Lista de nodos
            num_samples: Numero de vecinos a muestrear

        Returns:
            Lista de listas de vecinos muestreados
        """
        sampled_neighbors = []

        for node in nodes:
            neighbors = self.adj_list[node]

            if len(neighbors) == 0:
                # Nodo aislado: usar self-loop
                sampled = [node] * num_samples
            elif len(neighbors) < num_samples:
                # Menos vecinos que samples: repetir con reemplazo
                sampled = random.choices(neighbors, k=num_samples)
            else:
                # Suficientes vecinos: muestrear sin reemplazo
                sampled = random.sample(neighbors, num_samples)

            sampled_neighbors.append(sampled)

        return sampled_neighbors


class MeanAggregator(nn.Module):
    """Agregador de media simple."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, neighbor_embeddings: Tensor) -> Tensor:
        """
        Args:
            neighbor_embeddings: [batch_size, num_neighbors, in_features]
        Returns:
            [batch_size, out_features]
        """
        # Media sobre vecinos
        mean_neighbors = neighbor_embeddings.mean(dim=1)  # [batch, in_features]
        return self.linear(mean_neighbors)


class MaxPoolAggregator(nn.Module):
    """Agregador de max-pooling."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        self.linear = nn.Linear(out_features, out_features, bias=False)

    def forward(self, neighbor_embeddings: Tensor) -> Tensor:
        """
        Args:
            neighbor_embeddings: [batch_size, num_neighbors, in_features]
        """
        # Transformar cada vecino
        transformed = self.mlp(neighbor_embeddings)  # [batch, neighbors, out]

        # Max pooling sobre vecinos
        pooled = transformed.max(dim=1)[0]  # [batch, out_features]

        return self.linear(pooled)


class LSTMAggregator(nn.Module):
    """Agregador basado en LSTM."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=out_features,
            batch_first=True
        )

    def forward(self, neighbor_embeddings: Tensor) -> Tensor:
        """
        Args:
            neighbor_embeddings: [batch_size, num_neighbors, in_features]
        """
        # Permutar aleatoriamente el orden de vecinos
        batch_size, num_neighbors, _ = neighbor_embeddings.shape
        perm = torch.randperm(num_neighbors)
        shuffled = neighbor_embeddings[:, perm, :]

        # LSTM sobre vecinos
        _, (h_n, _) = self.lstm(shuffled)  # h_n: [1, batch, out]

        return h_n.squeeze(0)  # [batch, out_features]


class GraphSAGELayer(nn.Module):
    """
    Una capa de GraphSAGE.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator_type: Literal["mean", "maxpool", "lstm"] = "mean",
        normalize: bool = True
    ) -> None:
        super().__init__()

        self.normalize = normalize

        # Seleccionar agregador
        if aggregator_type == "mean":
            self.aggregator = MeanAggregator(in_features, out_features)
        elif aggregator_type == "maxpool":
            self.aggregator = MaxPoolAggregator(in_features, out_features)
        elif aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(in_features, out_features)
        else:
            raise ValueError(f"Agregador desconocido: {aggregator_type}")

        # Transformacion del nodo propio
        self.self_linear = nn.Linear(in_features, out_features, bias=False)

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self,
        x: Tensor,
        sampled_neighbors: list[list[int]],
        target_nodes: list[int]
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Embeddings de todos los nodos [num_nodes, in_features]
            sampled_neighbors: Lista de vecinos muestreados por nodo target
            target_nodes: Nodos para los que calcular nuevos embeddings
        """
        batch_size = len(target_nodes)
        num_neighbors = len(sampled_neighbors[0])
        in_features = x.size(1)

        # Obtener embeddings de vecinos
        # Shape: [batch_size, num_neighbors, in_features]
        neighbor_indices = torch.tensor(sampled_neighbors, dtype=torch.long)
        neighbor_embeddings = x[neighbor_indices.view(-1)].view(
            batch_size, num_neighbors, in_features
        )

        # Agregar vecinos
        aggregated = self.aggregator(neighbor_embeddings)  # [batch, out]

        # Transformar self
        self_features = x[target_nodes]  # [batch, in_features]
        self_transformed = self.self_linear(self_features)  # [batch, out]

        # Combinar (suma, no concatenacion en esta variante)
        output = aggregated + self_transformed + self.bias

        # Activacion
        output = F.relu(output)

        # Normalizar L2
        if self.normalize:
            output = F.normalize(output, p=2, dim=1)

        return output


class GraphSAGE(nn.Module):
    """
    GraphSAGE completo con multiples capas.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        sample_sizes: list[int] | None = None,
        aggregator_type: str = "mean",
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Sample sizes por capa (default: [25, 10])
        if sample_sizes is None:
            sample_sizes = [25, 10][:num_layers]
        self.sample_sizes = sample_sizes

        # Capas
        self.layers = nn.ModuleList()

        # Primera capa
        self.layers.append(GraphSAGELayer(
            in_features, hidden_features, aggregator_type
        ))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(
                hidden_features, hidden_features, aggregator_type
            ))

        # Ultima capa
        if num_layers > 1:
            self.layers.append(GraphSAGELayer(
                hidden_features, out_features, aggregator_type, normalize=False
            ))

        self.sampler: NeighborSampler | None = None

    def set_graph(self, edge_index: Tensor, num_nodes: int) -> None:
        """Configura el grafo para sampling."""
        self.sampler = NeighborSampler(edge_index, num_nodes)

    def forward(self, x: Tensor, target_nodes: list[int]) -> Tensor:
        """
        Forward pass para nodos especificos.

        Args:
            x: Features de todos los nodos [num_nodes, in_features]
            target_nodes: Nodos para los que generar embeddings
        """
        if self.sampler is None:
            raise RuntimeError("Llamar set_graph() antes de forward()")

        # Sampling de vecinos para cada capa
        # Hacemos sampling de "abajo hacia arriba" (desde nodos target)
        nodes_per_layer = [target_nodes]
        neighbors_per_layer = []

        current_nodes = target_nodes
        for layer_idx in range(self.num_layers):
            num_samples = self.sample_sizes[layer_idx]
            sampled = self.sampler.sample(current_nodes, num_samples)
            neighbors_per_layer.append(sampled)

            # Nodos unicos en esta capa (para siguiente iteracion)
            flat_neighbors = [n for sublist in sampled for n in sublist]
            current_nodes = list(set(flat_neighbors))
            nodes_per_layer.append(current_nodes)

        # Forward de "arriba hacia abajo"
        h = x
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            sampled = neighbors_per_layer[layer_idx]
            nodes = nodes_per_layer[layer_idx]

            h = layer(h, sampled, nodes)

            if layer_idx < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


def demo_graphsage() -> None:
    """Demuestra uso de GraphSAGE."""
    # Crear grafo simple
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 2, 1, 3],
        [1, 0, 2, 1, 3, 2, 4, 3, 2, 0, 3, 1]
    ])
    num_nodes = 5

    # Features
    x = torch.randn(num_nodes, 8)

    # Modelo
    model = GraphSAGE(
        in_features=8,
        hidden_features=16,
        out_features=4,
        num_layers=2,
        sample_sizes=[3, 2],
        aggregator_type="mean"
    )

    model.set_graph(edge_index, num_nodes)

    # Forward para nodos especificos
    target_nodes = [0, 2, 4]
    output = model(x, target_nodes)

    print(f"Target nodes: {target_nodes}")
    print(f"Output shape: {output.shape}")  # [3, 4]
    print(f"Output:\n{output}")


if __name__ == "__main__":
    demo_graphsage()
```

### GraphSAGE con PyTorch Geometric

```python
"""
GraphSAGE usando PyTorch Geometric.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Planetoid, Reddit


class GraphSAGE_PyG(torch.nn.Module):
    """GraphSAGE con PyTorch Geometric."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean"
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        # Primera capa
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # Ultima capa
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def train_with_neighbor_sampling(
    dataset_name: str = "Cora",
    batch_size: int = 256,
    num_neighbors: list[int] | None = None,
    epochs: int = 50
) -> dict[str, float]:
    """
    Entrena GraphSAGE con mini-batch neighbor sampling.
    """
    if num_neighbors is None:
        num_neighbors = [25, 10]  # Vecinos por capa

    # Cargar dataset
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    data = dataset[0]

    # Neighbor loader para mini-batch
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )

    # Modelo
    model = GraphSAGE_PyG(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=len(num_neighbors)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Solo nodos del batch (no vecinos muestreados)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            y = batch.y[:batch_size]

            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate (full-batch para simplicidad)
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()

            if val_acc > best_val_acc:
                best_val_acc = val_acc.item()
                best_test_acc = test_acc.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Val={val_acc:.4f}, Test={test_acc:.4f}")

    return {"best_val_acc": best_val_acc, "best_test_acc": best_test_acc}


if __name__ == "__main__":
    results = train_with_neighbor_sampling("Cora")
    print(f"\nFinal: Val={results['best_val_acc']:.4f}, Test={results['best_test_acc']:.4f}")
```

---

## Node2Vec: Random Walks en Grafos

### Concepto de Random Walks

```
NODE2VEC: APRENDIENDO EMBEDDINGS CON RANDOM WALKS
=================================================

Idea: Generar "oraciones" de nodos mediante caminatas
      aleatorias, luego aplicar Word2Vec

Analogia:
---------
NLP:    "the cat sat on the mat" --> Word2Vec --> Embeddings
Grafos: [A, B, C, D, E, F] (walk) --> Word2Vec --> Embeddings


RANDOM WALK BASICO:
-------------------

Dado nodo inicial v, caminar aleatoriamente:

    1. Empezar en v
    2. Mover a vecino aleatorio uniforme
    3. Repetir L pasos

    Ejemplo (L=5):
    [A] --> [B] --> [C] --> [B] --> [D] --> [E]


NODE2VEC: CAMINATAS SESGADAS
============================

Problema: Random walk uniforme no controla exploracion

Solucion: Parametros p y q para controlar sesgo:

- p (return parameter): Probabilidad de volver al nodo anterior
- q (in-out parameter): Controla BFS vs DFS


TRANSICION DESDE t A v, hacia x:
--------------------------------

    [t]---[v]---[x]
           |
          [y]

Si estamos en v (viniendo de t), probabilidad de ir a x:

    P(x|v,t) = (1/Z) * alpha_pq(t,x) * w_vx

Donde:
    alpha_pq(t,x) = 1/p  si d_tx = 0  (x = t, volver)
                  = 1    si d_tx = 1  (x vecino de t)
                  = 1/q  si d_tx = 2  (x no vecino de t)


INTUICION:
----------

p bajo, q bajo:  DFS-like (explora lejos)
p bajo, q alto:  BFS-like (explora local)
p alto, q bajo:  Mixto
p = q = 1:       Random walk uniforme


VISUALIZACION:
--------------

BFS-like (q=2, p=1):              DFS-like (q=0.5, p=1):
Explora vecindario local          Explora lejos

    [A]---[B]---[C]                  [A]---[B]---[C]
     |     |     |                    |     |     |
    [D]---[E]---[F]                  [D]---[E]---[F]
     |     |     |                    |     |     |
    [G]---[H]---[I]                  [G]---[H]---[I]

Walk desde E:                     Walk desde E:
E->B->E->D->E->F->E               E->B->C->F->I->H->G...

Captura estructura local          Captura estructura global
```

### Implementacion de Node2Vec

```python
"""
Implementacion de Node2Vec.
"""
import torch
import numpy as np
from collections import defaultdict
import random
from gensim.models import Word2Vec
from typing import TypeAlias

NodeId: TypeAlias = int
Walk: TypeAlias = list[NodeId]


class Node2Vec:
    """
    Node2Vec: Embeddings de nodos mediante random walks sesgados.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        embedding_dim: int = 64,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4
    ) -> None:
        """
        Args:
            edge_index: [2, num_edges]
            num_nodes: Numero de nodos
            embedding_dim: Dimension de embeddings
            walk_length: Longitud de cada walk
            num_walks: Numero de walks por nodo
            p: Return parameter
            q: In-out parameter
            workers: Workers para Word2Vec
        """
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers

        # Construir grafo
        self._build_graph(edge_index)

        # Precomputar probabilidades de transicion
        self._precompute_transition_probs()

    def _build_graph(self, edge_index: torch.Tensor) -> None:
        """Construye lista de adyacencia."""
        self.adj: dict[NodeId, list[NodeId]] = defaultdict(list)

        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            self.adj[s].append(d)
            # Asegurar que cada nodo tiene entrada
            if d not in self.adj:
                self.adj[d] = []

    def _precompute_transition_probs(self) -> None:
        """
        Precomputa probabilidades de transicion para cada arista.
        """
        self.alias_nodes: dict[NodeId, tuple] = {}
        self.alias_edges: dict[tuple[NodeId, NodeId], tuple] = {}

        # Para cada nodo
        for node in self.adj:
            neighbors = self.adj[node]
            if len(neighbors) == 0:
                continue

            # Probabilidades uniformes desde nodo
            probs = [1.0 / len(neighbors)] * len(neighbors)
            self.alias_nodes[node] = self._create_alias_table(probs)

        # Para cada arista (t, v)
        for source in self.adj:
            for dest in self.adj[source]:
                self._compute_edge_probs(source, dest)

    def _compute_edge_probs(self, t: NodeId, v: NodeId) -> None:
        """
        Computa probabilidades de transicion desde v, viniendo de t.
        """
        neighbors = self.adj[v]
        if len(neighbors) == 0:
            return

        t_neighbors = set(self.adj[t])
        probs = []

        for x in neighbors:
            if x == t:
                # Volver al nodo anterior
                probs.append(1.0 / self.p)
            elif x in t_neighbors:
                # Vecino comun con t (distancia 1)
                probs.append(1.0)
            else:
                # No vecino de t (distancia 2)
                probs.append(1.0 / self.q)

        # Normalizar
        prob_sum = sum(probs)
        probs = [p / prob_sum for p in probs]

        self.alias_edges[(t, v)] = self._create_alias_table(probs)

    def _create_alias_table(self, probs: list[float]) -> tuple[list[int], list[float]]:
        """
        Crea tabla de alias para sampling O(1).
        Implementacion simplificada - en produccion usar numpy.
        """
        # Simplificacion: retornar probabilidades directas
        return (list(range(len(probs))), probs)

    def _alias_sample(self, alias_table: tuple[list[int], list[float]]) -> int:
        """Muestrea de tabla de alias."""
        indices, probs = alias_table
        return random.choices(indices, weights=probs, k=1)[0]

    def _random_walk(self, start_node: NodeId) -> Walk:
        """
        Genera un random walk sesgado desde start_node.
        """
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = self.adj[cur]

            if len(neighbors) == 0:
                break

            if len(walk) == 1:
                # Primer paso: uniforme
                if cur in self.alias_nodes:
                    idx = self._alias_sample(self.alias_nodes[cur])
                    walk.append(neighbors[idx])
                else:
                    break
            else:
                # Pasos siguientes: sesgado
                prev = walk[-2]
                if (prev, cur) in self.alias_edges:
                    idx = self._alias_sample(self.alias_edges[(prev, cur)])
                    walk.append(neighbors[idx])
                else:
                    # Fallback a uniforme
                    walk.append(random.choice(neighbors))

        return walk

    def generate_walks(self) -> list[Walk]:
        """
        Genera todos los random walks.
        """
        walks = []
        nodes = list(self.adj.keys())

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node)
                if len(walk) > 1:
                    walks.append(walk)

        return walks

    def train(self) -> np.ndarray:
        """
        Entrena embeddings usando Word2Vec.

        Returns:
            Matriz de embeddings [num_nodes, embedding_dim]
        """
        # Generar walks
        walks = self.generate_walks()

        # Convertir a strings para Word2Vec
        walks_str = [[str(node) for node in walk] for walk in walks]

        # Entrenar Word2Vec
        model = Word2Vec(
            walks_str,
            vector_size=self.embedding_dim,
            window=10,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=5
        )

        # Extraer embeddings
        embeddings = np.zeros((self.num_nodes, self.embedding_dim))
        for node in range(self.num_nodes):
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]

        return embeddings

    def get_embeddings_tensor(self) -> torch.Tensor:
        """Retorna embeddings como tensor de PyTorch."""
        embeddings = self.train()
        return torch.from_numpy(embeddings).float()


def demo_node2vec() -> None:
    """Demuestra Node2Vec."""
    # Grafo simple
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 0, 2, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 2, 0, 0, 4]
    ])

    # Node2Vec con diferentes parametros
    print("Node2Vec BFS-like (p=1, q=2):")
    n2v_bfs = Node2Vec(
        edge_index, num_nodes=5,
        embedding_dim=16,
        walk_length=10,
        num_walks=5,
        p=1.0, q=2.0
    )
    embeddings_bfs = n2v_bfs.get_embeddings_tensor()
    print(f"Shape: {embeddings_bfs.shape}")

    print("\nNode2Vec DFS-like (p=1, q=0.5):")
    n2v_dfs = Node2Vec(
        edge_index, num_nodes=5,
        embedding_dim=16,
        walk_length=10,
        num_walks=5,
        p=1.0, q=0.5
    )
    embeddings_dfs = n2v_dfs.get_embeddings_tensor()
    print(f"Shape: {embeddings_dfs.shape}")

    # Comparar similitudes
    from torch.nn.functional import cosine_similarity

    print("\nSimilitud coseno entre nodos (BFS):")
    for i in range(5):
        for j in range(i+1, 5):
            sim = cosine_similarity(
                embeddings_bfs[i].unsqueeze(0),
                embeddings_bfs[j].unsqueeze(0)
            )
            print(f"  {i}-{j}: {sim.item():.3f}")


if __name__ == "__main__":
    demo_node2vec()
```

---

## DeepWalk y LINE

### DeepWalk

```
DEEPWALK: EL PRECURSOR DE NODE2VEC
==================================

DeepWalk = Random walks uniformes + Skip-gram

Diferencia con Node2Vec:
- DeepWalk: p=1, q=1 (uniforme)
- Node2Vec: p y q configurables


ALGORITMO:
----------

1. Para cada nodo v:
   a. Iniciar random walk desde v
   b. Caminar L pasos uniformemente
   c. Agregar walk al corpus

2. Entrenar Skip-gram sobre corpus de walks


SKIP-GRAM OBJETIVO:
-------------------

Maximizar:

    sum_{v in V} sum_{c in C(v)} log P(c | v)

Donde C(v) = contexto (vecinos en el walk)

    P(c | v) = exp(z_c^T * z_v) / sum_u exp(z_u^T * z_v)


EQUIVALENCIA CON FACTORIZACION:
-------------------------------

DeepWalk ~= Factorizar matriz:

    M = log(vol(G) * (1/T * sum_{r=1}^T (D^-1 A)^r) * D^-1) - log(b)

Donde:
- vol(G) = sum de grados
- T = longitud de walk
- D = matriz de grados
- b = numero de negative samples
```

### LINE: Large-scale Information Network Embedding

```
LINE: PRESERVANDO PROXIMIDAD DE PRIMER Y SEGUNDO ORDEN
======================================================

LINE optimiza dos tipos de proximidad:

1. PRIMER ORDEN (conexiones directas):
--------------------------------------
Si (i,j) estan conectados, deberian tener embeddings similares

    O_1 = - sum_{(i,j) in E} w_ij * log p_1(i,j)

    p_1(i,j) = sigmoid(z_i^T * z_j)


2. SEGUNDO ORDEN (vecinos compartidos):
---------------------------------------
Si i y j tienen vecinos similares, deberian ser similares

    O_2 = - sum_{(i,j) in E} w_ij * log p_2(j|i)

    p_2(j|i) = exp(z'_j^T * z_i) / sum_k exp(z'_k^T * z_i)

    Usa contexto embeddings z' separados.


VISUALIZACION:
--------------

Primer orden:                    Segundo orden:
(A conecta a B)                  (A y C tienen vecinos similares)

    [A]-----[B]                      [A]     [C]
                                       \     /
    z_A dot z_B alto                    \   /
                                       [X][Y]

                                  z_A similar a z_C
                                  (comparten vecinos X, Y)


VENTAJAS DE LINE:
-----------------
- Escala a grafos con millones de nodos
- Captura diferentes tipos de similitud
- Eficiente con negative sampling
```

### Implementacion de DeepWalk

```python
"""
Implementacion de DeepWalk.
"""
import torch
import numpy as np
from gensim.models import Word2Vec
import random


class DeepWalk:
    """
    DeepWalk: Random walks uniformes + Skip-gram.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        embedding_dim: int = 64,
        walk_length: int = 40,
        num_walks: int = 10,
        window_size: int = 5,
        workers: int = 4
    ) -> None:
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers

        # Construir grafo
        self.adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            self.adj[s].append(d)

    def _random_walk(self, start: int) -> list[int]:
        """Random walk uniforme."""
        walk = [start]

        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            neighbors = self.adj[cur]

            if len(neighbors) == 0:
                break

            walk.append(random.choice(neighbors))

        return walk

    def generate_walks(self) -> list[list[str]]:
        """Genera corpus de walks."""
        walks = []
        nodes = list(range(self.num_nodes))

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if len(self.adj[node]) > 0:
                    walk = self._random_walk(node)
                    walks.append([str(n) for n in walk])

        return walks

    def train(self) -> np.ndarray:
        """Entrena embeddings."""
        walks = self.generate_walks()

        model = Word2Vec(
            walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=10
        )

        embeddings = np.zeros((self.num_nodes, self.embedding_dim))
        for node in range(self.num_nodes):
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]

        return embeddings
```

---

## Metodos de Negative Sampling

### Negative Sampling para Grafos

```
NEGATIVE SAMPLING EN EMBEDDINGS DE GRAFOS
=========================================

Problema:
---------
Softmax sobre todos los nodos es O(|V|) - muy costoso

    P(j|i) = exp(z_j^T z_i) / sum_{k in V} exp(z_k^T z_i)


Solucion: Negative Sampling
---------------------------
Aproximar con muestras negativas

    log P(j|i) ≈ log sigma(z_j^T z_i) + sum_{k in neg_samples} log sigma(-z_k^T z_i)


INTUICION:
----------
En lugar de comparar con TODOS los nodos,
comparar con K nodos aleatorios (negativos)

    Positivo: (i, j) arista real -> z_i dot z_j ALTO
    Negativo: (i, k) aleatorio  -> z_i dot z_k BAJO


DISTRIBUCION DE MUESTREO:
-------------------------

Uniforme:
    P(k) = 1/|V|
    Simple pero no optima

Proporcional al grado:
    P(k) = degree(k) / sum_v degree(v)
    Nodos populares mas probables (realistic)

Suavizada (Word2Vec style):
    P(k) = degree(k)^0.75 / sum_v degree(v)^0.75
    Balance entre uniforme y grado


EJEMPLO:
--------

Arista positiva: (A, B)
Negativos muestreados: [C, D, E]

Loss:
    L = -log(sigma(z_A^T z_B))        # Positivo
        -log(sigma(-z_A^T z_C))       # Negativo 1
        -log(sigma(-z_A^T z_D))       # Negativo 2
        -log(sigma(-z_A^T z_E))       # Negativo 3

Optimizar:
    z_A dot z_B -> alto
    z_A dot z_C -> bajo
    z_A dot z_D -> bajo
    z_A dot z_E -> bajo
```

### Implementacion de Negative Sampling

```python
"""
Negative sampling para embeddings de grafos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class NegativeSampler:
    """
    Muestreador de ejemplos negativos.
    """

    def __init__(
        self,
        num_nodes: int,
        degrees: Tensor | None = None,
        power: float = 0.75
    ) -> None:
        """
        Args:
            num_nodes: Numero de nodos
            degrees: Grados de cada nodo (para sampling proporcional)
            power: Exponente para suavizar distribucion
        """
        self.num_nodes = num_nodes

        if degrees is not None:
            # Distribucion proporcional al grado^power
            probs = degrees.float() ** power
            self.probs = probs / probs.sum()
        else:
            # Uniforme
            self.probs = torch.ones(num_nodes) / num_nodes

    def sample(self, batch_size: int, num_negatives: int) -> Tensor:
        """
        Muestrea nodos negativos.

        Args:
            batch_size: Tamano del batch
            num_negatives: Negativos por ejemplo

        Returns:
            [batch_size, num_negatives]
        """
        return torch.multinomial(
            self.probs,
            batch_size * num_negatives,
            replacement=True
        ).view(batch_size, num_negatives)


class SkipGramGraphEmbedding(nn.Module):
    """
    Skip-gram con negative sampling para grafos.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 64,
        num_negatives: int = 5
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.num_negatives = num_negatives

        # Embeddings de nodos (target)
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

        # Embeddings de contexto
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # Inicializacion
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(
        self,
        pos_src: Tensor,
        pos_dst: Tensor,
        neg_dst: Tensor
    ) -> Tensor:
        """
        Calcula loss de skip-gram.

        Args:
            pos_src: Nodos fuente [batch_size]
            pos_dst: Nodos destino positivos [batch_size]
            neg_dst: Nodos destino negativos [batch_size, num_neg]
        """
        batch_size = pos_src.size(0)

        # Embeddings
        src_emb = self.embeddings(pos_src)  # [batch, dim]
        pos_ctx = self.context_embeddings(pos_dst)  # [batch, dim]
        neg_ctx = self.context_embeddings(neg_dst)  # [batch, num_neg, dim]

        # Score positivo: src dot pos_ctx
        pos_score = (src_emb * pos_ctx).sum(dim=1)  # [batch]

        # Score negativo: src dot neg_ctx
        neg_score = torch.bmm(
            neg_ctx,
            src_emb.unsqueeze(2)
        ).squeeze(2)  # [batch, num_neg]

        # Loss: -log(sigmoid(pos)) - sum log(sigmoid(-neg))
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss

    def get_embeddings(self) -> Tensor:
        """Retorna embeddings aprendidos."""
        return self.embeddings.weight.detach()


def train_skipgram_embeddings(
    edge_index: Tensor,
    num_nodes: int,
    embedding_dim: int = 64,
    num_negatives: int = 5,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 0.01
) -> Tensor:
    """
    Entrena embeddings con skip-gram y negative sampling.
    """
    # Calcular grados para sampling
    degrees = torch.zeros(num_nodes)
    for node in edge_index[0]:
        degrees[node] += 1

    # Sampler
    sampler = NegativeSampler(num_nodes, degrees)

    # Modelo
    model = SkipGramGraphEmbedding(num_nodes, embedding_dim, num_negatives)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Preparar aristas positivas
    num_edges = edge_index.size(1)

    for epoch in range(epochs):
        # Shuffle edges
        perm = torch.randperm(num_edges)
        edge_index_shuffled = edge_index[:, perm]

        total_loss = 0
        num_batches = 0

        for i in range(0, num_edges, batch_size):
            batch_edges = edge_index_shuffled[:, i:i+batch_size]
            pos_src = batch_edges[0]
            pos_dst = batch_edges[1]
            actual_batch = pos_src.size(0)

            # Sample negativos
            neg_dst = sampler.sample(actual_batch, num_negatives)

            # Forward
            loss = model(pos_src, pos_dst, neg_dst)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/num_batches:.4f}")

    return model.get_embeddings()
```

---

## Mini-Batch Training para Grafos Grandes

### Estrategias de Mini-Batch

```
MINI-BATCH TRAINING EN GRAFOS
=============================

Problema: No podemos cargar grafos grandes en memoria

Soluciones:
-----------

1. NEIGHBOR SAMPLING (GraphSAGE)
--------------------------------
Para cada nodo target, muestrear subconjunto de vecinos

    Batch de nodos: [A, B, C]
    Para cada nodo, sample k vecinos
    Solo cargar subgrafo necesario


2. SUBGRAPH SAMPLING (ClusterGCN)
---------------------------------
Particionar grafo en clusters
Cada batch = un cluster

    Grafo completo:
    [Cluster 1] -- [Cluster 2]
         |              |
    [Cluster 3] -- [Cluster 4]

    Batch 1: Solo Cluster 1
    Batch 2: Solo Cluster 2
    ...


3. LAYER-WISE SAMPLING (FastGCN)
--------------------------------
Muestrear nodos en cada capa, no vecinos

    Capa 2: Sample s2 nodos
    Capa 1: Sample s1 nodos
    Capa 0: Todos los nodos target


COMPARATIVA:
------------

+------------------+-------------+-------------+-------------+
| Metodo           | Complejidad | Varianza    | Escala      |
+------------------+-------------+-------------+-------------+
| Full batch       | O(|V| + |E|)| Baja        | No          |
| Neighbor sample  | O(B * k^L)  | Media       | Si          |
| Cluster GCN      | O(|V_c|)    | Media-alta  | Si          |
| Layer-wise       | O(B * s^L)  | Alta        | Si          |
+------------------+-------------+-------------+-------------+

B = batch size, k = neighbors, L = layers, s = samples
```

### Implementacion de Mini-Batch con PyG

```python
"""
Mini-batch training para grafos grandes.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.datasets import Reddit


class MiniBatchGraphSAGE(torch.nn.Module):
    """GraphSAGE para mini-batch training."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def train_with_neighbor_loader(
    data,
    model: torch.nn.Module,
    num_neighbors: list[int],
    batch_size: int = 1024,
    epochs: int = 10
) -> None:
    """
    Entrena con NeighborLoader (mini-batch neighbor sampling).
    """
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # batch.batch_size indica cuantos nodos son targets (no vecinos)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            y = batch.y[:batch_size]

            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += (out.argmax(dim=1) == y).sum().item()
            total_examples += batch_size

        acc = total_correct / total_examples
        loss = total_loss / total_examples

        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={acc:.4f}")


def train_with_cluster_loader(
    data,
    model: torch.nn.Module,
    num_parts: int = 100,
    epochs: int = 10
) -> None:
    """
    Entrena con ClusterLoader (ClusterGCN).
    """
    # Particionar grafo
    cluster_data = ClusterData(data, num_parts=num_parts)
    train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_examples = 0

        for batch in train_loader:
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            # Solo nodos con mascara de train
            train_mask = batch.train_mask
            if train_mask.sum() == 0:
                continue

            loss = F.cross_entropy(out[train_mask], batch.y[train_mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * train_mask.sum().item()
            total_examples += train_mask.sum().item()

        if total_examples > 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/total_examples:.4f}")


def demo_large_graph_training() -> None:
    """
    Demo de entrenamiento en grafo grande (Reddit).
    """
    print("Cargando Reddit dataset...")
    dataset = Reddit(root='/tmp/Reddit')
    data = dataset[0]

    print(f"Nodos: {data.num_nodes:,}")
    print(f"Aristas: {data.num_edges:,}")
    print(f"Features: {data.num_node_features}")
    print(f"Clases: {dataset.num_classes}")

    # Modelo
    model = MiniBatchGraphSAGE(
        in_channels=dataset.num_node_features,
        hidden_channels=256,
        out_channels=dataset.num_classes,
        num_layers=2
    )

    print("\nEntrenando con NeighborLoader...")
    train_with_neighbor_loader(
        data, model,
        num_neighbors=[25, 10],
        batch_size=1024,
        epochs=5
    )


if __name__ == "__main__":
    demo_large_graph_training()
```

---

## Aplicaciones en Ciberseguridad

### Deteccion de Botnets con GraphSAGE

```python
"""
Deteccion de botnets usando GraphSAGE inductivo.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np

TensorType: TypeAlias = torch.Tensor


@dataclass
class NetworkFlow:
    """Flujo de red para analisis."""
    src_ip: str
    dst_ip: str
    timestamp: float
    bytes_sent: int
    packets: int
    duration: float
    port: int


class BotnetDetectorGraphSAGE(nn.Module):
    """
    Detector de botnets usando GraphSAGE.

    Ventaja inductiva: Puede detectar botnets en IPs nunca vistas
    sin re-entrenar el modelo.
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()

        # Capas SAGE
        self.convs.append(SAGEConv(node_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)  # Normal, Bot, C&C
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType
    ) -> TensorType:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = torch.dropout(x, p=0.5, training=self.training)

        return self.classifier(x)


def extract_ip_features(
    ip: str,
    flows: list[NetworkFlow],
    time_window: float = 3600.0  # 1 hora
) -> np.ndarray:
    """
    Extrae features de comportamiento de una IP.
    """
    # Flujos donde esta IP es fuente
    outgoing = [f for f in flows if f.src_ip == ip]

    # Flujos donde esta IP es destino
    incoming = [f for f in flows if f.dst_ip == ip]

    # Features
    features = [
        # Volume
        len(outgoing),  # Conexiones salientes
        len(incoming),  # Conexiones entrantes
        sum(f.bytes_sent for f in outgoing),  # Bytes enviados
        sum(f.packets for f in outgoing),  # Paquetes enviados

        # Diversity
        len(set(f.dst_ip for f in outgoing)),  # IPs destino unicas
        len(set(f.port for f in outgoing)),  # Puertos unicos

        # Temporal
        np.std([f.timestamp for f in outgoing]) if len(outgoing) > 1 else 0,

        # Connection patterns
        np.mean([f.duration for f in outgoing]) if outgoing else 0,
    ]

    return np.array(features, dtype=np.float32)


def build_traffic_graph(
    flows: list[NetworkFlow],
    known_labels: dict[str, int] | None = None
) -> Data:
    """
    Construye grafo de trafico de red.

    Nodos: IPs
    Aristas: Conexiones
    """
    # Obtener IPs unicas
    ips = set()
    for f in flows:
        ips.add(f.src_ip)
        ips.add(f.dst_ip)

    ip_to_idx = {ip: idx for idx, ip in enumerate(sorted(ips))}
    num_nodes = len(ips)

    # Extraer features por IP
    features = []
    labels = []

    for ip in sorted(ips):
        feat = extract_ip_features(ip, flows)
        features.append(feat)

        if known_labels and ip in known_labels:
            labels.append(known_labels[ip])
        else:
            labels.append(-1)  # Unknown

    x = torch.tensor(np.stack(features), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Normalizar features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    # Construir aristas
    edges_src = []
    edges_dst = []

    for f in flows:
        src_idx = ip_to_idx[f.src_ip]
        dst_idx = ip_to_idx[f.dst_ip]
        edges_src.append(src_idx)
        edges_dst.append(dst_idx)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Mascaras
    train_mask = y != -1

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        ip_to_idx=ip_to_idx
    )


def detect_new_botnet_ip(
    model: BotnetDetectorGraphSAGE,
    existing_graph: Data,
    new_ip: str,
    new_flows: list[NetworkFlow]
) -> tuple[int, float]:
    """
    Detecta si una IP nueva es parte de una botnet.

    Ventaja de GraphSAGE inductivo:
    - No requiere re-entrenar
    - Usa estructura del grafo para propagar informacion
    """
    model.eval()

    # Agregar nueva IP al grafo
    ip_to_idx = existing_graph.ip_to_idx.copy()
    new_idx = len(ip_to_idx)
    ip_to_idx[new_ip] = new_idx

    # Features de nueva IP
    new_features = extract_ip_features(new_ip, new_flows)
    new_features = torch.tensor(new_features, dtype=torch.float).unsqueeze(0)

    # Normalizar con misma escala
    x_mean = existing_graph.x.mean(dim=0)
    x_std = existing_graph.x.std(dim=0)
    new_features = (new_features - x_mean) / (x_std + 1e-8)

    # Concatenar features
    x = torch.cat([existing_graph.x, new_features], dim=0)

    # Agregar aristas de nueva IP
    new_edges_src = []
    new_edges_dst = []

    for f in new_flows:
        if f.src_ip == new_ip and f.dst_ip in ip_to_idx:
            new_edges_src.append(new_idx)
            new_edges_dst.append(ip_to_idx[f.dst_ip])
        elif f.dst_ip == new_ip and f.src_ip in ip_to_idx:
            new_edges_src.append(ip_to_idx[f.src_ip])
            new_edges_dst.append(new_idx)

    new_edge_index = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)
    edge_index = torch.cat([existing_graph.edge_index, new_edge_index], dim=1)

    # Inferencia
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out[new_idx], dim=0)
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    return pred, confidence
```

---

## Ejercicios Practicos

### Ejercicio 1: Comparar Agregadores

```python
"""
Ejercicio 1: Comparar diferentes agregadores en GraphSAGE.

Objetivo: Implementar y comparar mean, maxpool, y LSTM agregadores
en el dataset Cora.
"""

def ejercicio_1_template() -> None:
    """
    TODO:
    1. Implementar MeanAggregator, MaxPoolAggregator, LSTMAggregator
    2. Entrenar GraphSAGE con cada agregador en Cora
    3. Comparar:
       - Accuracy en test
       - Tiempo de entrenamiento
       - Memoria utilizada
    4. Visualizar embeddings con t-SNE

    Preguntas a responder:
    - Cual agregador funciona mejor?
    - Hay trade-off entre expresividad y eficiencia?
    - Que agregador recomendarias para produccion?
    """
    pass
```

### Ejercicio 2: Node2Vec para Deteccion de Comunidades

```python
"""
Ejercicio 2: Usar Node2Vec para deteccion de comunidades.

Objetivo: Comparar diferentes configuraciones de p/q para
descubrir estructuras de comunidad.
"""

def ejercicio_2_template() -> None:
    """
    TODO:
    1. Generar grafo sintetico con 3 comunidades claras
       (usar stochastic block model)

    2. Entrenar Node2Vec con:
       - p=1, q=1 (uniforme)
       - p=0.5, q=2 (BFS-like)
       - p=2, q=0.5 (DFS-like)

    3. Para cada configuracion:
       - Visualizar embeddings con t-SNE
       - Aplicar K-means para clustering
       - Calcular NMI (Normalized Mutual Information) vs ground truth

    4. Cual configuracion detecta mejor las comunidades?
    """
    pass
```

### Ejercicio 3: GraphSAGE Inductivo para IPs Nuevas

```python
"""
Ejercicio 3: Sistema de deteccion de anomalias inductivo.

Objetivo: Implementar sistema que detecte IPs anomalas
en tiempo real, sin re-entrenar.
"""

def ejercicio_3_template() -> None:
    """
    TODO:
    1. Generar datos de trafico sintetico:
       - 1000 IPs normales
       - 50 IPs scanner (conectan a muchos destinos)
       - 10 IPs botnet (patron estrella)

    2. Entrenar GraphSAGE en 80% de IPs (inductivo)

    3. Simular llegada de IPs nuevas:
       - 10 IPs normales nuevas
       - 5 IPs scanner nuevas
       - 2 IPs botnet nuevas

    4. Evaluar:
       - Precision/Recall en IPs nuevas (nunca vistas)
       - Comparar con modelo transductivo (GCN)
       - Medir latencia de inferencia

    5. Visualizar:
       - Grafo con IPs coloreadas por prediccion
       - Confusion matrix para IPs nuevas
    """
    pass
```

---

## Resumen

```
RESUMEN: GRAPHSAGE Y METODOS AVANZADOS
======================================

GRAPHSAGE:
----------
- Sampling de vecinos para escalabilidad
- Funcion agregadora aprendida (no embeddings fijos)
- Inductivo: Funciona con nodos nuevos
- Agregadores: Mean, MaxPool, LSTM

NODE2VEC:
---------
- Random walks sesgados (parametros p, q)
- BFS-like (q alto): Captura estructura local
- DFS-like (q bajo): Captura estructura global
- Skip-gram para embeddings

MINI-BATCH TRAINING:
--------------------
- NeighborLoader: Sampling de vecinos
- ClusterGCN: Particion en clusters
- Escala a millones de nodos

APLICACIONES EN CIBERSEGURIDAD:
-------------------------------
- Deteccion de botnets (inductivo para IPs nuevas)
- Analisis de trafico en tiempo real
- Comunidades anomalas (Node2Vec)

RECOMENDACIONES:
----------------
- Grafos estaticos pequenos: GCN/GAT
- Grafos dinamicos: GraphSAGE (inductivo)
- Grafos muy grandes: Mini-batch + GraphSAGE
- Exploracion de estructura: Node2Vec


SIGUIENTE TEMA:
---------------
Aplicaciones especificas de GNNs en ciberseguridad:
- Fraud detection en transacciones
- Malware detection con call graphs
- Network intrusion detection
```

---

## Referencias

1. **Hamilton et al. (2017)** - "Inductive Representation Learning on Large Graphs" - Paper original de GraphSAGE
2. **Grover & Leskovec (2016)** - "node2vec: Scalable Feature Learning for Networks"
3. **Perozzi et al. (2014)** - "DeepWalk: Online Learning of Social Representations"
4. **Tang et al. (2015)** - "LINE: Large-scale Information Network Embedding"
5. **Chiang et al. (2019)** - "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"
6. **PyTorch Geometric NeighborLoader** - https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html
