# Introduccion a Grafos en Machine Learning

## Indice

1. [Por Que Grafos en ML](#por-que-grafos-en-ml)
2. [Fundamentos de Teoria de Grafos](#fundamentos-de-teoria-de-grafos)
3. [Representaciones de Grafos](#representaciones-de-grafos)
4. [Tipos de Grafos](#tipos-de-grafos)
5. [Tareas en Grafos](#tareas-en-grafos)
6. [Limitaciones de CNNs en Grafos](#limitaciones-de-cnns-en-grafos)
7. [Message Passing Paradigm](#message-passing-paradigm)
8. [Implementacion Practica con PyTorch Geometric](#implementacion-practica-con-pytorch-geometric)
9. [Aplicaciones en Ciberseguridad](#aplicaciones-en-ciberseguridad)
10. [Ejercicios Practicos](#ejercicios-practicos)

---

## Por Que Grafos en ML

### El Problema de los Datos Estructurados

Los datos del mundo real frecuentemente tienen estructura relacional que no puede capturarse
con formatos tabulares tradicionales o secuencias:

```
DATOS TRADICIONALES vs DATOS EN GRAFO
=====================================

Tabular (CSV):                    Grafo:
+----+-------+------+                    [Usuario A]
| ID | Nombre| Edad |                        /    \
+----+-------+------+                       /      \
| 1  | Alice | 25   |              [Usuario B]---[Usuario C]
| 2  | Bob   | 30   |                   |            |
| 3  | Carol | 28   |                   |            |
+----+-------+------+              [Usuario D]---[Usuario E]

Perdemos las RELACIONES!          Capturamos CONEXIONES!
```

### Dominios Donde los Grafos Son Esenciales

```
+------------------+------------------------+---------------------------+
| Dominio          | Nodos                  | Aristas                   |
+------------------+------------------------+---------------------------+
| Redes Sociales   | Usuarios               | Amistades, follows        |
| Biologia         | Proteinas/Genes        | Interacciones             |
| Ciberseguridad   | IPs, hosts, procesos   | Conexiones, llamadas      |
| Finanzas         | Cuentas, entidades     | Transacciones             |
| Quimica          | Atomos                 | Enlaces quimicos          |
| Conocimiento     | Entidades              | Relaciones semanticas     |
| Infraestructura  | Servidores             | Dependencias              |
+------------------+------------------------+---------------------------+
```

---

## Fundamentos de Teoria de Grafos

### Definicion Formal

Un grafo G se define como:

```
G = (V, E)

Donde:
- V = {v1, v2, ..., vn} : Conjunto de nodos (vertices)
- E = {e1, e2, ..., em} : Conjunto de aristas (edges)
- Cada arista e = (vi, vj) conecta dos nodos

EJEMPLO: Red de Computadoras
============================

        [Server1]
           /|\
          / | \
         /  |  \
   [PC1]   [PC2]   [PC3]
     \      |      /
      \     |     /
       \    |    /
        [Switch]
            |
        [Firewall]
            |
        [Internet]

V = {Server1, PC1, PC2, PC3, Switch, Firewall, Internet}
E = {(Server1,PC1), (Server1,PC2), (Server1,PC3),
     (PC1,Switch), (PC2,Switch), (PC3,Switch),
     (Switch,Firewall), (Firewall,Internet)}
```

### Terminologia Esencial

```python
"""
Conceptos fundamentales de grafos para ML.
"""
from dataclasses import dataclass
from typing import TypeAlias

NodeId: TypeAlias = int
EdgeTuple: TypeAlias = tuple[NodeId, NodeId]


@dataclass
class GraphConcepts:
    """Conceptos basicos de teoria de grafos."""

    # Grado de un nodo: numero de aristas conectadas
    # degree(v) = |{e in E : v in e}|

    # Vecinos de un nodo: nodos conectados directamente
    # N(v) = {u in V : (v,u) in E}

    # Camino: secuencia de nodos conectados por aristas
    # path = (v0, v1, ..., vk) donde (vi, vi+1) in E

    # Distancia: longitud del camino mas corto entre dos nodos
    # d(u,v) = min{|path| : path conecta u con v}

    # Componente conexa: subgrafo maximal conectado

    # Clustering coefficient: proporcion de vecinos que son vecinos entre si
    # C(v) = 2 * |{(u,w) in E : u,w in N(v)}| / (degree(v) * (degree(v)-1))


def calculate_degree(adjacency_list: dict[NodeId, list[NodeId]], node: NodeId) -> int:
    """Calcula el grado de un nodo."""
    return len(adjacency_list.get(node, []))


def get_neighbors(adjacency_list: dict[NodeId, list[NodeId]], node: NodeId) -> list[NodeId]:
    """Obtiene los vecinos de un nodo."""
    return adjacency_list.get(node, [])


def clustering_coefficient(
    adjacency_list: dict[NodeId, list[NodeId]],
    node: NodeId
) -> float:
    """Calcula el coeficiente de clustering local."""
    neighbors = set(adjacency_list.get(node, []))
    k = len(neighbors)

    if k < 2:
        return 0.0

    # Contar aristas entre vecinos
    edges_between_neighbors = 0
    for neighbor in neighbors:
        neighbor_connections = set(adjacency_list.get(neighbor, []))
        edges_between_neighbors += len(neighbors & neighbor_connections)

    # Dividir por 2 porque contamos cada arista dos veces
    edges_between_neighbors //= 2

    # Maximo posible de aristas entre k vecinos
    max_edges = k * (k - 1) // 2

    return edges_between_neighbors / max_edges if max_edges > 0 else 0.0
```

### Visualizacion de Propiedades

```
EJEMPLO: Clustering Coefficient
===============================

Nodo A con vecinos {B, C, D}:

Caso 1: C(A) = 0                  Caso 2: C(A) = 1
(vecinos no conectados)           (vecinos totalmente conectados)

       [A]                               [A]
      / | \                             / | \
     /  |  \                           /  |  \
   [B] [C] [D]                       [B]--[C]--[D]
                                       \  |  /
                                        \ | /
                                         [*]

Aristas entre vecinos: 0          Aristas entre vecinos: 3
Max posibles: 3                   Max posibles: 3
C(A) = 0/3 = 0.0                  C(A) = 3/3 = 1.0


Caso 3: C(A) = 0.33               Caso 4: C(A) = 0.67
(1 arista entre vecinos)          (2 aristas entre vecinos)

       [A]                               [A]
      / | \                             / | \
     /  |  \                           /  |  \
   [B]--[C] [D]                      [B]--[C]
                                       \     \
                                        \     \
                                         [D]---+

Aristas: 1, C(A) = 1/3            Aristas: 2, C(A) = 2/3
```

---

## Representaciones de Grafos

### 1. Matriz de Adyacencia (Adjacency Matrix)

```
MATRIZ DE ADYACENCIA
====================

Para un grafo G con n nodos, A es una matriz n x n donde:
A[i][j] = 1 si existe arista (i,j), 0 en caso contrario

Ejemplo:
               0   1   2   3
           +---+---+---+---+
         0 | 0 | 1 | 1 | 0 |     [0]---[1]
           +---+---+---+---+      |     |
         1 | 1 | 0 | 1 | 1 |      |     |
           +---+---+---+---+     [2]---[3]
         2 | 1 | 1 | 0 | 1 |
           +---+---+---+---+
         3 | 0 | 1 | 1 | 0 |
           +---+---+---+---+

Propiedades:
- Simetrica para grafos no dirigidos
- Diagonal = 0 (sin self-loops)
- Espacio: O(n^2) - problematico para grafos grandes
- Acceso arista: O(1)
- Iteracion vecinos: O(n)
```

```python
"""
Implementacion de matriz de adyacencia.
"""
import numpy as np
from numpy.typing import NDArray


class AdjacencyMatrix:
    """Representacion de grafo mediante matriz de adyacencia."""

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.matrix: NDArray[np.int8] = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.int8
        )

    def add_edge(self, src: int, dst: int, directed: bool = False) -> None:
        """Agrega una arista al grafo."""
        if not (0 <= src < self.num_nodes and 0 <= dst < self.num_nodes):
            raise ValueError(f"Nodos fuera de rango: {src}, {dst}")

        self.matrix[src, dst] = 1
        if not directed:
            self.matrix[dst, src] = 1

    def has_edge(self, src: int, dst: int) -> bool:
        """Verifica si existe una arista."""
        return bool(self.matrix[src, dst])

    def get_neighbors(self, node: int) -> list[int]:
        """Obtiene los vecinos de un nodo."""
        return list(np.where(self.matrix[node] == 1)[0])

    def get_degree(self, node: int) -> int:
        """Calcula el grado de un nodo."""
        return int(np.sum(self.matrix[node]))

    def to_normalized_laplacian(self) -> NDArray[np.float64]:
        """
        Calcula la Laplaciana normalizada: L = I - D^(-1/2) A D^(-1/2)
        Usada en GCN para propagacion de mensajes.
        """
        degrees = np.sum(self.matrix, axis=1)
        # Evitar division por cero
        degrees_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
        d_inv_sqrt = np.diag(degrees_inv_sqrt)

        # L = I - D^(-1/2) A D^(-1/2)
        identity = np.eye(self.num_nodes)
        normalized_adj = d_inv_sqrt @ self.matrix @ d_inv_sqrt

        return identity - normalized_adj

    def __repr__(self) -> str:
        return f"AdjacencyMatrix(nodes={self.num_nodes}, edges={np.sum(self.matrix)//2})"


# Ejemplo de uso
def demo_adjacency_matrix() -> None:
    """Demuestra el uso de matriz de adyacencia."""
    # Crear grafo de red simple
    graph = AdjacencyMatrix(num_nodes=5)

    # Agregar conexiones
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
    for src, dst in edges:
        graph.add_edge(src, dst)

    print("Matriz de Adyacencia:")
    print(graph.matrix)
    print(f"\nVecinos del nodo 1: {graph.get_neighbors(1)}")
    print(f"Grado del nodo 1: {graph.get_degree(1)}")
    print(f"\nLaplaciana normalizada:\n{graph.to_normalized_laplacian()}")
```

### 2. Lista de Aristas (Edge List)

```
LISTA DE ARISTAS
================

Representacion como lista de tuplas (src, dst, [weight])

Ejemplo:
edges = [
    (0, 1, 1.0),    # Nodo 0 -> Nodo 1, peso 1.0
    (0, 2, 0.5),    # Nodo 0 -> Nodo 2, peso 0.5
    (1, 2, 0.8),    # ...
    (1, 3, 1.2),
    (2, 3, 0.3),
    (3, 4, 0.9)
]

Propiedades:
- Espacio: O(m) donde m = numero de aristas
- Eficiente para grafos dispersos
- Ineficiente para buscar vecinos: O(m)
- Formato comun en datasets (COO format)
```

```python
"""
Implementacion de lista de aristas (Edge List).
"""
from dataclasses import dataclass


@dataclass
class Edge:
    """Representa una arista con peso opcional."""
    src: int
    dst: int
    weight: float = 1.0

    def reverse(self) -> "Edge":
        """Crea la arista inversa."""
        return Edge(src=self.dst, dst=self.src, weight=self.weight)


class EdgeList:
    """Representacion de grafo mediante lista de aristas."""

    def __init__(self) -> None:
        self.edges: list[Edge] = []
        self._node_set: set[int] = set()

    def add_edge(
        self,
        src: int,
        dst: int,
        weight: float = 1.0,
        directed: bool = False
    ) -> None:
        """Agrega una arista."""
        self.edges.append(Edge(src, dst, weight))
        self._node_set.add(src)
        self._node_set.add(dst)

        if not directed:
            self.edges.append(Edge(dst, src, weight))

    @property
    def num_nodes(self) -> int:
        """Numero de nodos unicos."""
        return len(self._node_set)

    @property
    def num_edges(self) -> int:
        """Numero de aristas."""
        return len(self.edges)

    def get_neighbors(self, node: int) -> list[int]:
        """Obtiene vecinos - O(m) operacion."""
        return [e.dst for e in self.edges if e.src == node]

    def to_coo_format(self) -> tuple[list[int], list[int], list[float]]:
        """
        Convierte a formato COO (Coordinate format).
        Retorna (row_indices, col_indices, values).
        """
        rows = [e.src for e in self.edges]
        cols = [e.dst for e in self.edges]
        values = [e.weight for e in self.edges]
        return rows, cols, values

    def to_adjacency_matrix(self) -> NDArray[np.float64]:
        """Convierte a matriz de adyacencia."""
        n = self.num_nodes
        matrix = np.zeros((n, n), dtype=np.float64)
        for edge in self.edges:
            matrix[edge.src, edge.dst] = edge.weight
        return matrix
```

### 3. Lista de Adyacencia (Adjacency List)

```
LISTA DE ADYACENCIA
===================

Diccionario donde cada nodo mapea a sus vecinos

Ejemplo:
adjacency = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [1, 2, 4],
    4: [3]
}

            [0]
           /   \
          /     \
        [1]-----[2]
          \     /
           \   /
            [3]
             |
            [4]

Propiedades:
- Espacio: O(n + m)
- Acceso vecinos: O(1)
- Verificar arista: O(degree(v))
- Mejor opcion para grafos dispersos
- Estructura mas usada en practica
```

```python
"""
Implementacion de lista de adyacencia.
"""
from collections import defaultdict


class AdjacencyList:
    """Representacion de grafo mediante lista de adyacencia."""

    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self._adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self._nodes: set[int] = set()

    def add_edge(self, src: int, dst: int, weight: float = 1.0) -> None:
        """Agrega una arista."""
        self._adj[src].append((dst, weight))
        self._nodes.add(src)
        self._nodes.add(dst)

        if not self.directed:
            self._adj[dst].append((src, weight))

    def add_node(self, node: int) -> None:
        """Agrega un nodo aislado."""
        self._nodes.add(node)
        if node not in self._adj:
            self._adj[node] = []

    def get_neighbors(self, node: int) -> list[int]:
        """Retorna lista de vecinos."""
        return [neighbor for neighbor, _ in self._adj.get(node, [])]

    def get_weighted_neighbors(self, node: int) -> list[tuple[int, float]]:
        """Retorna vecinos con pesos."""
        return self._adj.get(node, [])

    def degree(self, node: int) -> int:
        """Grado del nodo."""
        return len(self._adj.get(node, []))

    def has_edge(self, src: int, dst: int) -> bool:
        """Verifica existencia de arista."""
        return any(n == dst for n, _ in self._adj.get(src, []))

    @property
    def nodes(self) -> set[int]:
        """Conjunto de nodos."""
        return self._nodes

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        total = sum(len(neighbors) for neighbors in self._adj.values())
        return total if self.directed else total // 2

    def bfs(self, start: int) -> list[int]:
        """Busqueda en anchura desde un nodo."""
        from collections import deque

        visited = set()
        order = []
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            visited.add(node)
            order.append(node)

            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

        return order

    def dfs(self, start: int) -> list[int]:
        """Busqueda en profundidad desde un nodo."""
        visited = set()
        order = []

        def _dfs_recursive(node: int) -> None:
            if node in visited:
                return
            visited.add(node)
            order.append(node)
            for neighbor in self.get_neighbors(node):
                _dfs_recursive(neighbor)

        _dfs_recursive(start)
        return order
```

### Comparativa de Representaciones

```
COMPARATIVA DE REPRESENTACIONES
===============================

                    | Matriz Adj | Edge List | Adj List |
+-------------------+------------+-----------+----------+
| Espacio           | O(n^2)     | O(m)      | O(n+m)   |
| Agregar arista    | O(1)       | O(1)      | O(1)     |
| Eliminar arista   | O(1)       | O(m)      | O(deg)   |
| Verificar arista  | O(1)       | O(m)      | O(deg)   |
| Iterar vecinos    | O(n)       | O(m)      | O(deg)   |
| Iterar aristas    | O(n^2)     | O(m)      | O(n+m)   |
+-------------------+------------+-----------+----------+

Recomendaciones:
- Grafos DENSOS (m ~ n^2): Matriz de adyacencia
- Grafos DISPERSOS (m << n^2): Lista de adyacencia
- Solo iteracion de aristas: Edge list (COO format)
- PyTorch Geometric: Usa COO internamente
```

---

## Tipos de Grafos

### Clasificacion Visual

```
TIPOS DE GRAFOS
===============

1. DIRIGIDO vs NO DIRIGIDO
--------------------------
No Dirigido:              Dirigido:
    [A]---[B]                [A]-->[B]
     |     |                  |     |
     |     |                  v     v
    [C]---[D]                [C]<--[D]

2. CON PESO vs SIN PESO
-----------------------
Sin Peso:                 Con Peso:
    [A]---[B]                [A]-2.5-[B]
     |     |                  |       |
     |     |                 1.0     0.8
    [C]---[D]                [C]-3.2-[D]

3. BIPARTITO
------------
Dos conjuntos de nodos, aristas solo entre conjuntos

    [U1]    [U2]    [U3]     <- Usuarios
      \    /    \    /
       \  /      \  /
        \/        \/
    [I1]    [I2]    [I3]     <- Items

4. MULTIGRAFO
-------------
Multiples aristas entre mismos nodos

    [A] ====== [B]    <- Dos conexiones A-B
         \\
          \\
           [C]

5. HIPERGRAFO
-------------
Una arista puede conectar mas de 2 nodos

         [A]
        / | \
       /  |  \
      e1  e1  e1    <- e1 conecta {A,B,C}
       \  |  /
        \ | /
    [B]--e2--[C]    <- e2 conecta {B,C}

6. HETEROGENEO
--------------
Multiples tipos de nodos y aristas

    [Usuario]---compra--->[Producto]
        |                     |
     sigue               pertenece_a
        |                     |
        v                     v
    [Usuario]            [Categoria]
```

### Implementacion de Grafos Heterogeneos

```python
"""
Grafo heterogeneo para modelar sistemas complejos.
"""
from dataclasses import dataclass, field
from enum import Enum, auto


class NodeType(Enum):
    """Tipos de nodos en el sistema."""
    USER = auto()
    PRODUCT = auto()
    CATEGORY = auto()
    IP_ADDRESS = auto()
    PROCESS = auto()


class EdgeType(Enum):
    """Tipos de relaciones."""
    PURCHASES = auto()
    FOLLOWS = auto()
    BELONGS_TO = auto()
    CONNECTS_TO = auto()
    SPAWNS = auto()


@dataclass
class HeterogeneousNode:
    """Nodo con tipo y atributos."""
    id: int
    node_type: NodeType
    attributes: dict[str, float | str | int] = field(default_factory=dict)


@dataclass
class HeterogeneousEdge:
    """Arista con tipo y atributos."""
    src: int
    dst: int
    edge_type: EdgeType
    attributes: dict[str, float | str | int] = field(default_factory=dict)


class HeterogeneousGraph:
    """Grafo con multiples tipos de nodos y aristas."""

    def __init__(self) -> None:
        self.nodes: dict[int, HeterogeneousNode] = {}
        self.edges: list[HeterogeneousEdge] = []
        self._adj: dict[int, list[tuple[int, EdgeType]]] = defaultdict(list)

    def add_node(
        self,
        node_id: int,
        node_type: NodeType,
        attributes: dict[str, float | str | int] | None = None
    ) -> None:
        """Agrega un nodo tipado."""
        self.nodes[node_id] = HeterogeneousNode(
            id=node_id,
            node_type=node_type,
            attributes=attributes or {}
        )

    def add_edge(
        self,
        src: int,
        dst: int,
        edge_type: EdgeType,
        attributes: dict[str, float | str | int] | None = None
    ) -> None:
        """Agrega una arista tipada."""
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError("Los nodos deben existir antes de agregar aristas")

        edge = HeterogeneousEdge(
            src=src,
            dst=dst,
            edge_type=edge_type,
            attributes=attributes or {}
        )
        self.edges.append(edge)
        self._adj[src].append((dst, edge_type))

    def get_nodes_by_type(self, node_type: NodeType) -> list[HeterogeneousNode]:
        """Obtiene todos los nodos de un tipo."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[HeterogeneousEdge]:
        """Obtiene todas las aristas de un tipo."""
        return [e for e in self.edges if e.edge_type == edge_type]

    def get_metapath_neighbors(
        self,
        start_node: int,
        metapath: list[EdgeType]
    ) -> list[int]:
        """
        Encuentra vecinos siguiendo un metapath.
        Ejemplo: Usuario -> compra -> Producto -> pertenece_a -> Categoria
        """
        current_nodes = {start_node}

        for edge_type in metapath:
            next_nodes = set()
            for node in current_nodes:
                for neighbor, e_type in self._adj.get(node, []):
                    if e_type == edge_type:
                        next_nodes.add(neighbor)
            current_nodes = next_nodes

            if not current_nodes:
                return []

        return list(current_nodes)


# Ejemplo: Modelo de red para deteccion de fraude
def create_fraud_detection_graph() -> HeterogeneousGraph:
    """Crea un grafo para deteccion de fraude."""
    graph = HeterogeneousGraph()

    # Agregar usuarios
    graph.add_node(0, NodeType.USER, {"risk_score": 0.2, "country": "ES"})
    graph.add_node(1, NodeType.USER, {"risk_score": 0.8, "country": "XX"})
    graph.add_node(2, NodeType.USER, {"risk_score": 0.1, "country": "ES"})

    # Agregar productos
    graph.add_node(10, NodeType.PRODUCT, {"price": 999.99, "category": "electronics"})
    graph.add_node(11, NodeType.PRODUCT, {"price": 50.00, "category": "books"})

    # Agregar transacciones
    graph.add_edge(0, 10, EdgeType.PURCHASES, {"amount": 999.99, "timestamp": 1234567890})
    graph.add_edge(1, 10, EdgeType.PURCHASES, {"amount": 999.99, "timestamp": 1234567900})
    graph.add_edge(2, 11, EdgeType.PURCHASES, {"amount": 50.00, "timestamp": 1234567910})

    # Relaciones sociales (sospechosas si comparten patrones)
    graph.add_edge(0, 1, EdgeType.FOLLOWS)

    return graph
```

---

## Tareas en Grafos

### Niveles de Tareas

```
NIVELES DE TAREAS EN GRAFOS
===========================

1. NODE-LEVEL TASKS
-------------------
Predecir propiedades de nodos individuales

    Clasificacion de nodos:

    [Spam?]    [Legit]    [Spam?]
       |          |          |
       v          v          v
      [?]--------[+]--------[?]
       |          |          |
       |          |          |
      [+]--------[-]--------[+]

    Ejemplos:
    - Clasificar usuarios como fraudulentos/legitimos
    - Predecir funcion de proteinas
    - Detectar cuentas bot en redes sociales

2. EDGE-LEVEL TASKS
-------------------
Predecir existencia o propiedades de aristas

    Link Prediction:

      [A]---[B]   [C]
       |     |   / ?
       |     |  /
      [D]---[E]?

    Existe arista (B,C)? Existe arista (D,E)?

    Ejemplos:
    - Sistemas de recomendacion (usuario-item)
    - Prediccion de interacciones proteina-proteina
    - Prediccion de conexiones maliciosas

3. GRAPH-LEVEL TASKS
--------------------
Predecir propiedades del grafo completo

    Clasificacion de grafos:

    Molecula 1:       Molecula 2:
    (Toxica)          (No toxica)

    O=C               H-C-H
     |                 |
    C-N               H-C-H

    Ejemplos:
    - Clasificar moleculas como toxicas/no toxicas
    - Detectar malware basado en call graphs
    - Clasificar redes como botnets
```

### Implementacion de Tareas

```python
"""
Framework para diferentes tareas en grafos.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor


class GraphTask(ABC):
    """Clase base para tareas en grafos."""

    @abstractmethod
    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Calcula la perdida para la tarea."""
        pass

    @abstractmethod
    def evaluate(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Evalua metricas de la tarea."""
        pass


class NodeClassificationTask(GraphTask):
    """Tarea de clasificacion de nodos."""

    def __init__(self, num_classes: int, class_weights: Tensor | None = None) -> None:
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy loss para clasificacion."""
        return self.loss_fn(predictions, targets)

    def evaluate(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Metricas: accuracy, precision, recall, f1."""
        pred_classes = predictions.argmax(dim=1)
        correct = (pred_classes == targets).float()
        accuracy = correct.mean().item()

        # Calcular precision y recall por clase
        metrics = {"accuracy": accuracy}

        for c in range(self.num_classes):
            tp = ((pred_classes == c) & (targets == c)).sum().float()
            fp = ((pred_classes == c) & (targets != c)).sum().float()
            fn = ((pred_classes != c) & (targets == c)).sum().float()

            precision = (tp / (tp + fp + 1e-8)).item()
            recall = (tp / (tp + fn + 1e-8)).item()
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics[f"precision_class_{c}"] = precision
            metrics[f"recall_class_{c}"] = recall
            metrics[f"f1_class_{c}"] = f1

        return metrics


class LinkPredictionTask(GraphTask):
    """Tarea de prediccion de enlaces."""

    def __init__(self) -> None:
        self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Binary cross-entropy para prediccion de enlaces."""
        return self.loss_fn(predictions, targets.float())

    def compute_link_scores(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Calcula scores de enlaces usando producto punto.
        edge_index: [2, num_edges] con (src, dst) pairs
        """
        src_embeddings = node_embeddings[edge_index[0]]
        dst_embeddings = node_embeddings[edge_index[1]]

        # Producto punto como score de similitud
        scores = (src_embeddings * dst_embeddings).sum(dim=1)
        return scores

    def evaluate(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Metricas: AUC-ROC, precision, recall."""
        from sklearn.metrics import roc_auc_score, precision_score, recall_score

        pred_probs = torch.sigmoid(predictions).cpu().numpy()
        targets_np = targets.cpu().numpy()
        pred_binary = (pred_probs > 0.5).astype(int)

        return {
            "auc_roc": roc_auc_score(targets_np, pred_probs),
            "precision": precision_score(targets_np, pred_binary),
            "recall": recall_score(targets_np, pred_binary)
        }


class GraphClassificationTask(GraphTask):
    """Tarea de clasificacion de grafos completos."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy para clasificacion de grafos."""
        return self.loss_fn(predictions, targets)

    def global_pooling(
        self,
        node_embeddings: Tensor,
        batch_indices: Tensor,
        pooling_type: str = "mean"
    ) -> Tensor:
        """
        Agrega embeddings de nodos a nivel de grafo.

        Args:
            node_embeddings: [total_nodes, hidden_dim]
            batch_indices: [total_nodes] indica a que grafo pertenece cada nodo
            pooling_type: "mean", "max", or "sum"
        """
        num_graphs = batch_indices.max().item() + 1
        hidden_dim = node_embeddings.size(1)

        graph_embeddings = torch.zeros(num_graphs, hidden_dim)

        for g in range(num_graphs):
            mask = batch_indices == g
            graph_nodes = node_embeddings[mask]

            if pooling_type == "mean":
                graph_embeddings[g] = graph_nodes.mean(dim=0)
            elif pooling_type == "max":
                graph_embeddings[g] = graph_nodes.max(dim=0)[0]
            elif pooling_type == "sum":
                graph_embeddings[g] = graph_nodes.sum(dim=0)

        return graph_embeddings

    def evaluate(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Metricas para clasificacion de grafos."""
        pred_classes = predictions.argmax(dim=1)
        accuracy = (pred_classes == targets).float().mean().item()

        return {"accuracy": accuracy}
```

---

## Limitaciones de CNNs en Grafos

### Por Que las CNNs No Funcionan

```
POR QUE CNNs NO FUNCIONAN EN GRAFOS
===================================

1. ESTRUCTURA FIJA vs VARIABLE
------------------------------

CNN (Imagen):                     Grafo:
+---+---+---+---+                    [A]
| 1 | 2 | 3 | 4 |                   / | \
+---+---+---+---+                  /  |  \
| 5 | 6 | 7 | 8 |               [B] [C] [D]
+---+---+---+---+                 |   |
| 9 |10 |11 |12 |               [E] [F]
+---+---+---+---+

Grid regular 4x3                Variable y sin grid!
Siempre 12 "vecinos"           Grado variable por nodo

2. ORDEN DE VECINOS
-------------------

CNN: El vecino "arriba-izquierda" siempre esta en posicion fija

     Kernel 3x3:
     +---+---+---+
     | a | b | c |   <- posiciones fijas
     +---+---+---+
     | d | X | e |   <- X es el centro
     +---+---+---+
     | f | g | h |
     +---+---+---+

Grafo: No hay "arriba", "izquierda"...

        ?
        |
    ?--[X]--?    <- Los vecinos no tienen orden intrinseco
        |
        ?

3. NUMERO VARIABLE DE VECINOS
-----------------------------

CNN: Siempre 8 vecinos (kernel 3x3)

Grafo:
- Nodo A tiene 2 vecinos
- Nodo B tiene 5 vecinos
- Nodo C tiene 1 vecino

No podemos aplicar el mismo kernel!

4. INVARIANZA A PERMUTACIONES
-----------------------------

Un grafo es el MISMO independiente de como numeremos los nodos:

    [0]---[1]          [2]---[0]
     |     |    ===     |     |
    [2]---[3]          [1]---[3]

Mismo grafo, diferente numeracion!
CNNs NO son invariantes a permutaciones.
```

### Solucion: Message Passing

```
LA SOLUCION: MESSAGE PASSING NEURAL NETWORKS
=============================================

En lugar de convolucion fija, usamos:
1. Cada nodo RECOGE informacion de sus vecinos
2. AGREGA la informacion (invariante al orden)
3. ACTUALIZA su representacion

                    Paso 1: MENSAJE
                    ===============

    h_A           h_B           h_C
     |             |             |
     v             v             v
    [A]-----------[B]-----------[C]
     |             |             |
     +-------------+-------------+
                   |
                   v
              AGREGAR
              (sum/mean/max)
                   |
                   v
              h_B_new = f(h_B, AGG({h_A, h_C}))


El nodo B actualiza su embedding usando:
- Su propio estado h_B
- Informacion agregada de vecinos

VENTAJAS:
- Funciona con cualquier numero de vecinos
- Invariante a permutaciones (por la agregacion)
- Preserva estructura del grafo
```

---

## Message Passing Paradigm

### Framework General

```
MESSAGE PASSING NEURAL NETWORK (MPNN)
=====================================

Cada capa de una MPNN tiene 3 fases:

1. MESSAGE: Crear mensajes de vecinos
   m_j->i = MSG(h_i, h_j, e_ij)

2. AGGREGATE: Combinar mensajes
   m_i = AGG({m_j->i : j in N(i)})

3. UPDATE: Actualizar embedding
   h_i' = UPD(h_i, m_i)


DIAGRAMA DE FLUJO:
==================

    Capa k                         Capa k+1

    h_A^(k) ----MSG----> m_A->B
                              \
    h_B^(k) ----MSG----> m_B->C -> AGG -> h_C^(k+1)
                              /
    h_C^(k) (self)  ----------


Ejemplo numerico:
-----------------

    h_A = [1, 0]      h_B = [0, 1]
         \              /
          \            /
           \          /
            [C]-------
         h_C = [1, 1]

Mensajes (ejemplo: identidad):
m_A->C = [1, 0]
m_B->C = [0, 1]

Agregacion (mean):
m_C = mean([1,0], [0,1]) = [0.5, 0.5]

Update (concat + linear):
h_C' = Linear([h_C; m_C])
     = Linear([1, 1, 0.5, 0.5])
     = nuevo embedding
```

### Implementacion de Message Passing

```python
"""
Implementacion del framework de Message Passing.
"""
import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod


class MessagePassingLayer(ABC, nn.Module):
    """Capa base de message passing."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor | None = None
    ) -> Tensor:
        """
        Calcula mensajes de j hacia i.

        Args:
            x_i: Features del nodo destino [num_edges, in_features]
            x_j: Features del nodo origen [num_edges, in_features]
            edge_attr: Atributos de arista opcionales
        """
        pass

    @abstractmethod
    def aggregate(self, messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """
        Agrega mensajes por nodo destino.

        Args:
            messages: Mensajes [num_edges, msg_features]
            index: Indices de nodos destino [num_edges]
            num_nodes: Numero total de nodos
        """
        pass

    @abstractmethod
    def update(self, aggregated: Tensor, x: Tensor) -> Tensor:
        """
        Actualiza embeddings de nodos.

        Args:
            aggregated: Mensajes agregados [num_nodes, msg_features]
            x: Features actuales [num_nodes, in_features]
        """
        pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None
    ) -> Tensor:
        """
        Forward pass completo.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: [2, num_edges] con (src, dst)
            edge_attr: Opcional [num_edges, edge_features]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Obtener features de origen y destino
        x_j = x[src]  # Features de nodos origen
        x_i = x[dst]  # Features de nodos destino

        # 1. MESSAGE
        messages = self.message(x_i, x_j, edge_attr)

        # 2. AGGREGATE
        aggregated = self.aggregate(messages, dst, num_nodes)

        # 3. UPDATE
        return self.update(aggregated, x)


class SimpleMessagePassing(MessagePassingLayer):
    """Implementacion simple de message passing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggr: str = "mean"
    ) -> None:
        super().__init__(in_features, out_features)
        self.aggr = aggr

        # Transformaciones lineales
        self.lin_msg = nn.Linear(in_features, out_features)
        self.lin_update = nn.Linear(in_features + out_features, out_features)
        self.activation = nn.ReLU()

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor | None = None
    ) -> Tensor:
        """Mensaje simple: transformar features del vecino."""
        return self.lin_msg(x_j)

    def aggregate(self, messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Agregacion por suma, media o max."""
        out_features = messages.size(1)
        aggregated = torch.zeros(num_nodes, out_features, device=messages.device)

        if self.aggr == "sum":
            aggregated.scatter_add_(0, index.unsqueeze(1).expand_as(messages), messages)
        elif self.aggr == "mean":
            aggregated.scatter_add_(0, index.unsqueeze(1).expand_as(messages), messages)
            # Contar vecinos por nodo
            counts = torch.zeros(num_nodes, device=messages.device)
            counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float))
            counts = counts.clamp(min=1).unsqueeze(1)
            aggregated = aggregated / counts
        elif self.aggr == "max":
            aggregated.fill_(-float('inf'))
            aggregated.scatter_reduce_(
                0,
                index.unsqueeze(1).expand_as(messages),
                messages,
                reduce="amax"
            )
            aggregated[aggregated == -float('inf')] = 0

        return aggregated

    def update(self, aggregated: Tensor, x: Tensor) -> Tensor:
        """Combina embedding original con mensajes agregados."""
        combined = torch.cat([x, aggregated], dim=1)
        return self.activation(self.lin_update(combined))


class MultiLayerMPNN(nn.Module):
    """Red de message passing con multiples capas."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 3,
        aggr: str = "mean",
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        # Primera capa
        self.layers.append(SimpleMessagePassing(in_features, hidden_features, aggr))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.layers.append(SimpleMessagePassing(hidden_features, hidden_features, aggr))

        # Ultima capa
        if num_layers > 1:
            self.layers.append(SimpleMessagePassing(hidden_features, out_features, aggr))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None
    ) -> Tensor:
        """Forward pass por todas las capas."""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


# Ejemplo de uso
def demo_message_passing() -> None:
    """Demuestra el uso de message passing."""
    # Crear grafo simple
    #     0 -- 1
    #     |    |
    #     2 -- 3

    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3, 2, 3],  # src
        [1, 0, 2, 0, 3, 1, 3, 2]   # dst
    ])

    # Features de nodos (4 nodos, 8 features cada uno)
    x = torch.randn(4, 8)

    # Crear modelo
    model = MultiLayerMPNN(
        in_features=8,
        hidden_features=16,
        out_features=4,  # Para clasificacion en 4 clases
        num_layers=3
    )

    # Forward pass
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [4, 4]

    # Predicciones de clase
    predictions = output.argmax(dim=1)
    print(f"Predicciones: {predictions}")
```

### Visualizacion del Proceso

```
FLUJO DE INFORMACION EN MESSAGE PASSING
=======================================

Iteracion 1:
           Cada nodo "ve" a sus vecinos directos

           [A]---[B]---[C]

           A ve: {B}
           B ve: {A, C}
           C ve: {B}

Iteracion 2:
           Cada nodo "ve" a 2 saltos de distancia

           [A]---[B]---[C]

           A ve: {B, C} (C a traves de B)
           B ve: {A, C}
           C ve: {A, B} (A a traves de B)

Iteracion k:
           Cada nodo "ve" a k saltos de distancia

           Campo receptivo = k-hop neighborhood


OVER-SMOOTHING: EL PROBLEMA DE MUCHAS CAPAS
============================================

Con demasiadas capas, todos los nodos convergen
al mismo embedding (pierden identidad):

Capa 1:  [A]   [B]   [C]   [D]   <- Diferentes
          |     |     |     |
Capa 2:  [A']  [B']  [C']  [D']  <- Similares
          |     |     |     |
Capa 3:  [*]   [*]   [*]   [*]   <- Casi iguales!

Solucion: Usar 2-4 capas tipicamente, o skip connections.
```

---

## Implementacion Practica con PyTorch Geometric

### Instalacion y Setup

```python
"""
Configuracion de PyTorch Geometric.

Instalacion:
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html
"""
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.utils import to_networkx
import torch.nn.functional as F


def create_pyg_graph() -> Data:
    """
    Crea un grafo en formato PyTorch Geometric.
    """
    # Definir aristas en formato COO
    # Grafo:  0 -- 1 -- 2
    #         |    |
    #         3 -- 4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 1, 4, 3, 4],  # src
        [1, 0, 2, 1, 3, 0, 4, 1, 4, 3]   # dst
    ], dtype=torch.long)

    # Features de nodos (5 nodos, 3 features)
    x = torch.tensor([
        [1.0, 0.0, 0.0],  # Nodo 0
        [0.0, 1.0, 0.0],  # Nodo 1
        [0.0, 0.0, 1.0],  # Nodo 2
        [1.0, 1.0, 0.0],  # Nodo 3
        [0.0, 1.0, 1.0],  # Nodo 4
    ], dtype=torch.float)

    # Labels de nodos
    y = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)

    # Crear objeto Data
    data = Data(x=x, edge_index=edge_index, y=y)

    # Propiedades adicionales
    print(f"Numero de nodos: {data.num_nodes}")
    print(f"Numero de aristas: {data.num_edges}")
    print(f"Features por nodo: {data.num_node_features}")
    print(f"Tiene nodos aislados: {data.has_isolated_nodes()}")
    print(f"Tiene self-loops: {data.has_self_loops()}")
    print(f"Es dirigido: {data.is_directed()}")

    return data


def load_benchmark_datasets() -> tuple[Data, list[Data]]:
    """
    Carga datasets de benchmark.
    """
    # Cora: dataset de citas (node classification)
    # 2708 papers, 5429 links, 7 clases, 1433 features
    cora = Planetoid(root='/tmp/Cora', name='Cora')
    cora_data = cora[0]

    print("Dataset Cora:")
    print(f"  Nodos: {cora_data.num_nodes}")
    print(f"  Aristas: {cora_data.num_edges}")
    print(f"  Features: {cora_data.num_node_features}")
    print(f"  Clases: {cora.num_classes}")
    print(f"  Train/Val/Test: {cora_data.train_mask.sum()}/{cora_data.val_mask.sum()}/{cora_data.test_mask.sum()}")

    # MUTAG: dataset de moleculas (graph classification)
    # 188 moleculas, 2 clases (mutagenic or not)
    mutag = TUDataset(root='/tmp/MUTAG', name='MUTAG')

    print("\nDataset MUTAG:")
    print(f"  Numero de grafos: {len(mutag)}")
    print(f"  Clases: {mutag.num_classes}")
    print(f"  Features por nodo: {mutag.num_node_features}")

    return cora_data, list(mutag)


class SimpleGCN(torch.nn.Module):
    """
    Graph Convolutional Network simple.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # Primera capa
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Ultima capa
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def train_node_classification(
    model: torch.nn.Module,
    data: Data,
    epochs: int = 200,
    lr: float = 0.01
) -> list[float]:
    """
    Entrena modelo para clasificacion de nodos.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward
        out = model(data.x, data.edge_index)

        # Loss solo en nodos de entrenamiento
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # Backward
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            # Evaluar en validacion
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
            model.train()

    return losses


def evaluate_model(model: torch.nn.Module, data: Data) -> dict[str, float]:
    """
    Evalua modelo en test set.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Metricas
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = test_correct.float().mean().item()

        # Por clase
        results = {"test_accuracy": test_acc}

        for c in range(out.size(1)):
            class_mask = data.y[data.test_mask] == c
            if class_mask.sum() > 0:
                class_acc = test_correct[class_mask].float().mean().item()
                results[f"class_{c}_accuracy"] = class_acc

    return results


# Ejemplo completo
def main_example() -> None:
    """Ejemplo completo de entrenamiento."""
    # Cargar Cora
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # Crear modelo
    model = SimpleGCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=3
    )

    print(f"Modelo: {model}")
    print(f"Parametros: {sum(p.numel() for p in model.parameters())}")

    # Entrenar
    losses = train_node_classification(model, data, epochs=200)

    # Evaluar
    results = evaluate_model(model, data)
    print(f"\nResultados finales:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main_example()
```

---

## Aplicaciones en Ciberseguridad

### Casos de Uso Principales

```
APLICACIONES DE GRAFOS EN CIBERSEGURIDAD
========================================

1. DETECCION DE FRAUDE
----------------------
    [Usuario]---transaccion--->[Comercio]
         |                          |
    usa_tarjeta                 categoria
         |                          |
    [Tarjeta]<--asociada--[Direccion]

    Detectar patrones anomalos en el grafo
    de transacciones.

2. DETECCION DE INTRUSOS EN RED
-------------------------------
    [IP Ext]---conexion--->[Servidor]
                               |
                          proceso
                               |
    [IP Int]<--lateral--[Workstation]

    Identificar movimiento lateral y
    conexiones sospechosas.

3. ANALISIS DE MALWARE
----------------------
    [main()]---llama--->[decrypt()]
        |                    |
    llama               llama
        |                    |
    [read_file()]      [connect()]
                            |
                      [send_data()]

    Call graphs para clasificar
    comportamiento malicioso.

4. DETECCION DE BOTNETS
-----------------------
    [Bot1]---C&C--->[Server]
                        ^
    [Bot2]--------------+
                        ^
    [Bot3]--------------+

    Identificar estructuras de comando
    y control en redes sociales/IRC.
```

### Ejemplo: Deteccion de Cuentas Fraudulentas

```python
"""
Modelo GNN para deteccion de fraude en transacciones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from dataclasses import dataclass
from typing import TypeAlias

TensorType: TypeAlias = torch.Tensor


@dataclass
class TransactionFeatures:
    """Features de una transaccion."""
    amount: float
    hour: int
    day_of_week: int
    merchant_category: int
    is_online: bool
    distance_from_home: float
    velocity: float  # transacciones por hora


class FraudDetectionGNN(nn.Module):
    """
    GNN para detectar transacciones fraudulentas.

    El grafo modela:
    - Nodos: Usuarios, comercios, tarjetas, dispositivos
    - Aristas: Transacciones, asociaciones
    """

    def __init__(
        self,
        user_features: int,
        merchant_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        # Embeddings iniciales por tipo de nodo
        self.user_embed = nn.Linear(user_features, hidden_channels)
        self.merchant_embed = nn.Linear(merchant_features, hidden_channels)

        # Capas de GraphSAGE
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 2)  # Fraud / Not fraud
        )

    def forward(
        self,
        x_user: TensorType,
        x_merchant: TensorType,
        edge_index: TensorType,
        edge_label_index: TensorType
    ) -> TensorType:
        """
        Forward pass.

        Args:
            x_user: Features de usuarios [num_users, user_features]
            x_merchant: Features de comercios [num_merchants, merchant_features]
            edge_index: Conexiones en el grafo
            edge_label_index: Aristas a clasificar (transacciones)
        """
        # Embeber nodos
        h_user = F.relu(self.user_embed(x_user))
        h_merchant = F.relu(self.merchant_embed(x_merchant))

        # Concatenar todos los nodos
        x = torch.cat([h_user, h_merchant], dim=0)

        # Message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Obtener embeddings de pares (usuario, comercio) para cada transaccion
        src, dst = edge_label_index[0], edge_label_index[1]
        edge_embeddings = torch.cat([x[src], x[dst]], dim=1)

        # Clasificar
        return self.classifier(edge_embeddings)


def create_fraud_graph(
    transactions: list[dict],
    users: list[dict],
    merchants: list[dict]
) -> Data:
    """
    Crea grafo de transacciones para deteccion de fraude.
    """
    num_users = len(users)
    num_merchants = len(merchants)

    # Features de usuarios
    user_features = torch.tensor([
        [u['account_age'], u['avg_transaction'], u['num_transactions']]
        for u in users
    ], dtype=torch.float)

    # Features de comercios
    merchant_features = torch.tensor([
        [m['avg_amount'], m['fraud_rate'], m['category_code']]
        for m in merchants
    ], dtype=torch.float)

    # Construir aristas (transacciones)
    edges_src = []
    edges_dst = []
    edge_labels = []

    for tx in transactions:
        user_id = tx['user_id']
        merchant_id = tx['merchant_id'] + num_users  # Offset para merchants

        edges_src.extend([user_id, merchant_id])
        edges_dst.extend([merchant_id, user_id])
        edge_labels.append(tx['is_fraud'])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Crear Data object
    data = Data(
        x_user=user_features,
        x_merchant=merchant_features,
        edge_index=edge_index,
        y=torch.tensor(edge_labels, dtype=torch.long)
    )

    return data


def train_fraud_detector(
    model: FraudDetectionGNN,
    train_data: Data,
    epochs: int = 100
) -> list[float]:
    """
    Entrena detector de fraude.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Class weights para datos desbalanceados (fraude es raro)
    pos_weight = torch.tensor([1.0, 10.0])  # Penalizar mas los falsos negativos
    criterion = nn.CrossEntropyLoss(weight=pos_weight)

    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward
        out = model(
            train_data.x_user,
            train_data.x_merchant,
            train_data.edge_index,
            train_data.edge_index[:, ::2]  # Solo aristas user->merchant
        )

        loss = criterion(out, train_data.y)

        # Backward
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    return losses
```

---

## Ejercicios Practicos

### Ejercicio 1: Implementar BFS/DFS en Grafos

```python
"""
Ejercicio 1: Implementar algoritmos de recorrido.

Objetivo: Implementar BFS y DFS sobre una lista de adyacencia
y usarlos para detectar componentes conexas.
"""

def ejercicio_1_template() -> None:
    """
    TODO: Implementar las siguientes funciones:

    1. bfs(graph, start) -> list[int]
       Recorrido en anchura desde 'start'

    2. dfs(graph, start) -> list[int]
       Recorrido en profundidad desde 'start'

    3. find_connected_components(graph) -> list[list[int]]
       Encontrar todas las componentes conexas

    Grafo de prueba:

        [0]---[1]     [4]---[5]
         |     |             |
        [2]---[3]           [6]

    Componentes esperadas: [[0,1,2,3], [4,5,6]]
    """
    # Grafo como lista de adyacencia
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2],
        4: [5],
        5: [4, 6],
        6: [5]
    }

    # Tu implementacion aqui
    pass
```

### Ejercicio 2: Message Passing Manual

```python
"""
Ejercicio 2: Implementar message passing desde cero.

Objetivo: Implementar una capa de message passing sin usar
PyTorch Geometric, solo numpy/torch.
"""

def ejercicio_2_template() -> None:
    """
    TODO: Implementar clase MessagePassingManual con:

    1. message(x_i, x_j) -> mensajes de j a i
    2. aggregate(messages, edge_index) -> mensajes agregados por nodo
    3. update(x, aggregated) -> nuevos embeddings

    Probar en grafo simple:

        [0]---[1]
         |     |
        [2]---[3]

    Features iniciales: Matriz identidad 4x4
    Verificar que mensaje pasa correctamente.
    """
    import torch

    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 3, 2, 3],
        [1, 0, 2, 0, 3, 1, 3, 2]
    ])

    x = torch.eye(4)  # Identidad como features iniciales

    # Tu implementacion aqui
    pass
```

### Ejercicio 3: Deteccion de Anomalias en Red

```python
"""
Ejercicio 3: Construir grafo de red y detectar anomalias.

Objetivo: Dado un log de conexiones de red, construir el grafo
y detectar patrones anomalos (escaneo de puertos, DDoS, etc.)
"""

def ejercicio_3_template() -> None:
    """
    TODO:

    1. Parsear logs de conexion a grafo
    2. Calcular features de nodos:
       - Grado de entrada/salida
       - Numero de puertos unicos
       - Clustering coefficient
    3. Detectar anomalias:
       - Nodos con grado anormalmente alto
       - Patrones de estrella (DDoS, scan)
       - Comunidades aisladas (botnets)

    Log de ejemplo:
    timestamp,src_ip,dst_ip,dst_port,protocol
    1234567890,192.168.1.1,10.0.0.1,80,TCP
    1234567891,192.168.1.1,10.0.0.1,443,TCP
    1234567892,192.168.1.2,10.0.0.1,80,TCP
    ...
    """
    logs = [
        {"src": "192.168.1.1", "dst": "10.0.0.1", "port": 80},
        {"src": "192.168.1.1", "dst": "10.0.0.2", "port": 80},
        {"src": "192.168.1.1", "dst": "10.0.0.3", "port": 80},
        # ... mas conexiones
    ]

    # Tu implementacion aqui
    pass
```

---

## Resumen

```
RESUMEN: INTRODUCCION A GRAFOS EN ML
====================================

CONCEPTOS CLAVE:
----------------
1. Grafo G = (V, E): Nodos y aristas que capturan relaciones
2. Representaciones: Matriz de adyacencia, edge list, adjacency list
3. Tareas: Node-level, edge-level, graph-level
4. Message passing: Framework para propagar informacion en grafos

POR QUE GRAFOS:
---------------
- CNNs asumen estructura regular (grids)
- Grafos manejan estructura irregular
- Message passing es permutation-invariant

FORMULA CLAVE - MESSAGE PASSING:
--------------------------------
h_v^(k+1) = UPDATE(h_v^(k), AGG({h_u^(k) : u in N(v)}))

APLICACIONES EN CIBERSEGURIDAD:
-------------------------------
- Deteccion de fraude en transacciones
- Analisis de trafico de red
- Clasificacion de malware via call graphs
- Deteccion de botnets

SIGUIENTE TEMA:
---------------
GCN y GAT: Implementaciones concretas del paradigma
de message passing con convolucion y atencion.
```

---

## Referencias

1. **Kipf & Welling (2017)** - "Semi-Supervised Classification with Graph Convolutional Networks"
2. **Hamilton et al. (2017)** - "Inductive Representation Learning on Large Graphs" (GraphSAGE)
3. **Velickovic et al. (2018)** - "Graph Attention Networks"
4. **Gilmer et al. (2017)** - "Neural Message Passing for Quantum Chemistry"
5. **PyTorch Geometric Documentation** - https://pytorch-geometric.readthedocs.io/
