# Graph Neural Networks en Ciberseguridad

## Indice

1. [Introduccion: Por Que GNNs en Ciberseguridad](#introduccion-por-que-gnns-en-ciberseguridad)
2. [Deteccion de Fraude en Transacciones](#deteccion-de-fraude-en-transacciones)
3. [Network Intrusion Detection](#network-intrusion-detection)
4. [Malware Detection con Call Graphs](#malware-detection-con-call-graphs)
5. [Deteccion de Botnets y Analisis de Redes Sociales](#deteccion-de-botnets)
6. [Threat Intelligence con Knowledge Graphs](#threat-intelligence-con-knowledge-graphs)
7. [APT Detection con GNNs Temporales](#apt-detection-con-gnns-temporales)
8. [Implementacion Completa: Sistema de Deteccion](#implementacion-completa)
9. [Evaluacion y Metricas en Ciberseguridad](#evaluacion-y-metricas)
10. [Ejercicios Practicos](#ejercicios-practicos)

---

## Introduccion: Por Que GNNs en Ciberseguridad

### El Poder de las Relaciones

```
POR QUE GRAFOS EN CIBERSEGURIDAD
================================

Los ataques ciberneticos son inherentemente RELACIONALES:
- Un atacante compromete UN sistema para alcanzar OTROS
- El fraude involucra REDES de cuentas
- El malware se propaga a traves de CONEXIONES
- Los botnets forman ESTRUCTURAS de comando

Datos tradicionales (tabular) pierden estas relaciones:

ENFOQUE TABULAR:
----------------
| IP        | Bytes | Packets | Duration | Label |
|-----------|-------|---------|----------|-------|
| 10.0.0.1  | 1000  | 50      | 10.5     | ?     |
| 10.0.0.2  | 5000  | 200     | 45.0     | ?     |

Perdemos: Quien habla con quien, patrones de conexion,
         topologia de ataque, propagacion.


ENFOQUE DE GRAFO:
-----------------

         [Attacker]
             |
        C&C Server
             |
    +--------+--------+
    |        |        |
  [Bot1]   [Bot2]   [Bot3]
    |        |        |
  [Victim1][Victim2][Victim3]

Capturamos: Estructura de botnet, cadena de ataque,
           relaciones C&C, propagacion lateral.
```

### Ventajas de GNNs para Seguridad

```
VENTAJAS DE GNNS EN CIBERSEGURIDAD
==================================

1. DETECCION DE PATRONES ESTRUCTURALES
--------------------------------------
Los atacantes dejan "huellas" en la estructura:

    Escaneo de puertos:        Movimiento lateral:
    (patron estrella)          (patron cadena)

         [Scanner]                  [Entry]
        /  |  |  \                    |
       /   |  |   \                 [Host1]
      /    |  |    \                  |
    [T1] [T2][T3] [T4]              [Host2]
                                      |
                                   [Target]


2. PROPAGACION DE INFORMACION
-----------------------------
Si un vecino es malicioso, aumenta mi sospecha:

    [Malicious] --- [???] --- [Normal]
                      |
                   h_??? = AGG(h_malicious, h_normal)

    El nodo ??? hereda "sospecha" de su vecino malicioso.


3. MANEJO DE DATOS HETEROGENEOS
-------------------------------
    [Usuario]---usa--->[Dispositivo]
        |                   |
    ejecuta            conecta_a
        |                   |
    [App]---accede--->[Servidor]

    GNNs manejan naturalmente multiples tipos
    de entidades y relaciones.


4. APRENDIZAJE SEMI-SUPERVISADO
-------------------------------
Solo necesitamos etiquetar ALGUNOS nodos:

    [+]---[?]---[-]
     |     |     |
    [?]---[?]---[?]

    GNN propaga labels conocidos a desconocidos.
```

---

## Deteccion de Fraude en Transacciones

### El Problema del Fraude

```
FRAUDE EN SISTEMAS FINANCIEROS
==============================

Caracteristicas:
- Datos MASIVAMENTE desbalanceados (0.1% fraude)
- Fraudsters se adaptan (adversarial)
- Falsas alarmas son costosas
- Fraude organizado involucra multiples cuentas

TIPOS DE FRAUDE DETECTABLES CON GRAFOS:
---------------------------------------

1. FRAUDE EN ANILLO:
    Cuentas que intercambian dinero circularmente

         [A]---->[B]
          ^       |
          |       v
         [D]<----[C]

2. FRAUDE DE MULA:
    Cuenta receptora de multiples fuentes sospechosas

    [Fraud1]---+
               |
    [Fraud2]---+--->[$Mule$]---->[$$$]
               |
    [Fraud3]---+

3. FRAUDE DE IDENTIDAD SINTETICA:
    Multiples identidades falsas comparten atributos

    [FakeID1]---usa--->[Telefono]---usa---[FakeID2]
        |                                     |
       usa                                   usa
        |                                     |
    [Email]                              [Direccion]


POR QUE GNN FUNCIONA:
---------------------
- Fraude en anillo -> Ciclos en el grafo
- Fraude de mula -> Alto grado de entrada
- ID sintetica -> Nodos comparten vecinos
```

### Implementacion: Detector de Fraude

```python
"""
Sistema de deteccion de fraude basado en GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias
import numpy as np

TensorType: TypeAlias = torch.Tensor


class FraudType(Enum):
    """Tipos de fraude detectables."""
    NORMAL = 0
    RING_FRAUD = 1
    MULE_FRAUD = 2
    SYNTHETIC_ID = 3
    UNKNOWN = -1


@dataclass
class Transaction:
    """Una transaccion financiera."""
    id: str
    sender_id: str
    receiver_id: str
    amount: float
    timestamp: float
    merchant_category: int
    is_online: bool
    device_id: str | None = None


@dataclass
class Account:
    """Una cuenta de usuario."""
    id: str
    creation_date: float
    email_domain: str
    phone_prefix: str
    country: str
    risk_score: float = 0.0
    is_fraud: FraudType = FraudType.UNKNOWN


class FraudDetectionGNN(nn.Module):
    """
    GNN heterogeneo para deteccion de fraude.

    Tipos de nodos:
    - account: Cuentas de usuario
    - device: Dispositivos
    - merchant: Comercios
    - ip: Direcciones IP

    Tipos de aristas:
    - account -> (sends_to) -> account
    - account -> (uses) -> device
    - account -> (pays_at) -> merchant
    - account -> (connects_from) -> ip
    """

    def __init__(
        self,
        account_features: int,
        device_features: int,
        merchant_features: int,
        ip_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 4
    ) -> None:
        super().__init__()

        # Proyecciones iniciales por tipo de nodo
        self.account_lin = nn.Linear(account_features, hidden_channels)
        self.device_lin = nn.Linear(device_features, hidden_channels)
        self.merchant_lin = nn.Linear(merchant_features, hidden_channels)
        self.ip_lin = nn.Linear(ip_features, hidden_channels)

        # Capas heterogeneas
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv({
                ('account', 'sends_to', 'account'): SAGEConv(hidden_channels, hidden_channels),
                ('account', 'uses', 'device'): SAGEConv(hidden_channels, hidden_channels),
                ('device', 'used_by', 'account'): SAGEConv(hidden_channels, hidden_channels),
                ('account', 'pays_at', 'merchant'): SAGEConv(hidden_channels, hidden_channels),
                ('merchant', 'receives_from', 'account'): SAGEConv(hidden_channels, hidden_channels),
                ('account', 'connects_from', 'ip'): SAGEConv(hidden_channels, hidden_channels),
                ('ip', 'used_by', 'account'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)

        # Clasificador de cuentas
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x_dict: dict[str, TensorType], edge_index_dict: dict) -> TensorType:
        """
        Forward pass.

        Args:
            x_dict: Features por tipo de nodo
            edge_index_dict: Aristas por tipo de relacion
        """
        # Proyecciones iniciales
        x_dict = {
            'account': F.relu(self.account_lin(x_dict['account'])),
            'device': F.relu(self.device_lin(x_dict['device'])),
            'merchant': F.relu(self.merchant_lin(x_dict['merchant'])),
            'ip': F.relu(self.ip_lin(x_dict['ip'])),
        }

        # Message passing heterogeneo
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=0.3, training=self.training)
                     for key, x in x_dict.items()}

        # Clasificar solo cuentas
        return self.classifier(x_dict['account'])


class RingFraudDetector(nn.Module):
    """
    Detector especializado en fraude en anillo.
    Usa features de ciclos en el grafo.
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 32
    ) -> None:
        super().__init__()

        self.conv1 = SAGEConv(node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # MLP para combinar embedding + features de ciclo
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + 3, hidden_channels),  # +3 cycle features
            nn.ReLU(),
            nn.Linear(hidden_channels, 2)
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        cycle_features: TensorType
    ) -> TensorType:
        """
        Args:
            x: Node features
            edge_index: Aristas de transacciones
            cycle_features: Features de ciclos [num_nodes, 3]
                - num_cycles: Numero de ciclos que pasan por el nodo
                - min_cycle_length: Longitud del ciclo mas corto
                - cycle_amount: Suma de montos en ciclos
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(x, edge_index)

        # Concatenar con features de ciclo
        combined = torch.cat([h, cycle_features], dim=1)

        return self.classifier(combined)


def detect_cycles(edge_index: TensorType, max_length: int = 5) -> dict[int, list]:
    """
    Detecta ciclos en el grafo de transacciones.

    Returns:
        Diccionario: nodo -> lista de ciclos que lo contienen
    """
    from collections import defaultdict

    # Construir lista de adyacencia
    adj = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)

    # DFS para encontrar ciclos
    cycles_by_node: dict[int, list] = defaultdict(list)

    def dfs(start: int, current: int, path: list[int], visited: set[int]) -> None:
        if len(path) > max_length:
            return

        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 2:
                # Encontramos un ciclo
                cycle = path + [neighbor]
                for node in cycle[:-1]:
                    cycles_by_node[node].append(cycle)
            elif neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, path + [neighbor], visited)
                visited.remove(neighbor)

    nodes = set(src + dst)
    for node in nodes:
        dfs(node, node, [node], {node})

    return dict(cycles_by_node)


def extract_cycle_features(
    cycles_by_node: dict[int, list],
    transaction_amounts: dict[tuple[int, int], float],
    num_nodes: int
) -> TensorType:
    """
    Extrae features de ciclos para cada nodo.
    """
    features = torch.zeros(num_nodes, 3)

    for node, cycles in cycles_by_node.items():
        if cycles:
            # Numero de ciclos
            features[node, 0] = len(cycles)

            # Longitud minima de ciclo
            min_len = min(len(c) for c in cycles)
            features[node, 1] = min_len

            # Suma de montos en ciclos
            total_amount = 0
            for cycle in cycles:
                for i in range(len(cycle) - 1):
                    edge = (cycle[i], cycle[i+1])
                    total_amount += transaction_amounts.get(edge, 0)
            features[node, 2] = total_amount

    # Normalizar
    features = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-8)

    return features


def build_fraud_graph(
    transactions: list[Transaction],
    accounts: list[Account]
) -> HeteroData:
    """
    Construye grafo heterogeneo para deteccion de fraude.
    """
    # Mapear IDs a indices
    account_ids = {a.id: i for i, a in enumerate(accounts)}

    # Extraer IDs de dispositivos, merchants, IPs unicos
    devices = set()
    merchants = set()
    ips = set()

    for t in transactions:
        merchants.add(t.merchant_category)
        if t.device_id:
            devices.add(t.device_id)

    device_ids = {d: i for i, d in enumerate(devices)}
    merchant_ids = {m: i for i, m in enumerate(merchants)}
    # IPs se extraerian de datos adicionales

    # Features de cuentas
    account_features = []
    account_labels = []

    for acc in accounts:
        features = [
            acc.creation_date / 1e10,  # Normalizado
            hash(acc.email_domain) % 1000 / 1000,
            len(acc.phone_prefix),
            hash(acc.country) % 100 / 100,
            acc.risk_score
        ]
        account_features.append(features)
        account_labels.append(acc.is_fraud.value)

    x_account = torch.tensor(account_features, dtype=torch.float)
    y_account = torch.tensor(account_labels, dtype=torch.long)

    # Features de dispositivos (simplificado)
    x_device = torch.randn(len(devices), 4)  # En practica: OS, browser, etc.

    # Features de merchants
    x_merchant = torch.randn(len(merchants), 8)

    # Features de IPs (vacio por ahora)
    x_ip = torch.zeros(1, 6)

    # Aristas: account -> account (transacciones)
    sends_src = []
    sends_dst = []

    for t in transactions:
        if t.sender_id in account_ids and t.receiver_id in account_ids:
            sends_src.append(account_ids[t.sender_id])
            sends_dst.append(account_ids[t.receiver_id])

    # Aristas: account -> device
    uses_src = []
    uses_dst = []

    for t in transactions:
        if t.device_id and t.sender_id in account_ids and t.device_id in device_ids:
            uses_src.append(account_ids[t.sender_id])
            uses_dst.append(device_ids[t.device_id])

    # Aristas: account -> merchant
    pays_src = []
    pays_dst = []

    for t in transactions:
        if t.sender_id in account_ids:
            pays_src.append(account_ids[t.sender_id])
            pays_dst.append(merchant_ids[t.merchant_category])

    # Crear HeteroData
    data = HeteroData()

    data['account'].x = x_account
    data['account'].y = y_account
    data['device'].x = x_device
    data['merchant'].x = x_merchant
    data['ip'].x = x_ip

    data['account', 'sends_to', 'account'].edge_index = torch.tensor(
        [sends_src, sends_dst], dtype=torch.long
    )
    data['account', 'uses', 'device'].edge_index = torch.tensor(
        [uses_src, uses_dst], dtype=torch.long
    )
    data['device', 'used_by', 'account'].edge_index = torch.tensor(
        [uses_dst, uses_src], dtype=torch.long
    )
    data['account', 'pays_at', 'merchant'].edge_index = torch.tensor(
        [pays_src, pays_dst], dtype=torch.long
    )
    data['merchant', 'receives_from', 'account'].edge_index = torch.tensor(
        [pays_dst, pays_src], dtype=torch.long
    )

    return data
```

### Visualizacion de Patrones de Fraude

```
PATRONES DE FRAUDE EN GRAFO
===========================

1. FRAUDE EN ANILLO (Ring Fraud)
--------------------------------

    Flujo de dinero en ciclo para "limpiar" fondos:

        [$1000]
    [A] ---------> [B]
     ^              |
     |              | [$950]
     |              v
    [$900]        [C]
     |              |
     |              | [$920]
     +----[D]<-----+

    Caracteristicas GNN detecta:
    - Nodos con alto "cycle participation"
    - Montos similares circulando
    - Tiempos de transaccion correlacionados


2. FRAUDE DE MULA (Money Mule)
------------------------------

    Multiples fuentes fraudulentas -> Una cuenta mula

    [Fraud1] --$500-->+
                      |
    [Fraud2] --$300-->+--->[MULA]--$1200-->[Cash Out]
                      |
    [Fraud3] --$400-->+

    Caracteristicas GNN detecta:
    - Alto grado de entrada
    - Fuentes de alto riesgo
    - Agregacion y dispersion rapida


3. RED DE IDENTIDAD SINTETICA
-----------------------------

    IDs falsas comparten atributos "escondidos":

    [ID_1]---usa---[Tel: 555-1234]---usa---[ID_2]
       \                                    /
        \---usa---[Device_X]---usa--------+
                       |
                      usa
                       |
                    [ID_3]

    Caracteristicas GNN detecta:
    - Comunidades sospechosas de IDs
    - Comparticion anomala de atributos
    - Creacion temporal correlacionada
```

---

## Network Intrusion Detection

### Modelando Trafico de Red como Grafo

```
GRAFO DE TRAFICO DE RED
=======================

Nodos: Hosts (IPs, servidores, endpoints)
Aristas: Conexiones de red (flujos)
Features de nodos: Estadisticas de trafico
Features de aristas: Caracteristicas del flujo


ESTRUCTURA TIPICA:
------------------

                        [Internet]
                            |
                        [Firewall]
                       /    |    \
                      /     |     \
               [DMZ]   [Internal]  [DB]
                |          |         |
           [WebSrv]  [Workstations] [DBSrv]
                         / | \
                        /  |  \
                     [PC1][PC2][PC3]


ATAQUES Y SUS PATRONES EN GRAFO:
--------------------------------

1. PORT SCANNING:
   - Patron estrella: 1 nodo -> muchos destinos
   - Muchos puertos diferentes
   - Conexiones cortas/fallidas

        [Scanner]
       / | | | | \
      v  v v v v  v
    [T1][T2][T3][T4][T5]...


2. DDoS ATTACK:
   - Patron estrella inversa: muchos -> 1
   - Alto volumen de trafico
   - Patrones temporales sincronizados

    [Bot1] [Bot2] [Bot3] [Bot4]
       \     |     |     /
        \    |     |    /
         v   v     v   v
           [Victim]


3. LATERAL MOVEMENT:
   - Patron de cadena: A -> B -> C -> D
   - Secuencia temporal
   - Protocolos de admin (SSH, RDP, WMI)

    [Entry Point]
         |
         v
    [Server 1]
         |
         v
    [Server 2]
         |
         v
    [Target Data]


4. DATA EXFILTRATION:
   - Conexiones a IPs externas inusuales
   - Alto volumen de salida
   - Horarios anomalos

    [Internal]---LARGE DATA--->[External IP]
       (3am, unusual destination)
```

### Implementacion: IDS con GNN

```python
"""
Sistema de Deteccion de Intrusos basado en GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_add_pool
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from collections import defaultdict

TensorType: TypeAlias = torch.Tensor


class AttackType(Enum):
    """Tipos de ataque detectables."""
    NORMAL = 0
    PORT_SCAN = 1
    DDOS = 2
    LATERAL_MOVEMENT = 3
    EXFILTRATION = 4
    BRUTE_FORCE = 5


@dataclass
class NetworkFlow:
    """Flujo de red capturado."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    timestamp: float
    duration: float
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    flags: str


class NetworkIDSGNN(nn.Module):
    """
    GNN para deteccion de intrusiones de red.

    Clasifica cada host (nodo) como:
    - Normal
    - Atacante (scanner, DDoS source, etc.)
    - Victima
    - Comprometido (lateral movement)
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_channels: int = 64,
        num_classes: int = 6
    ) -> None:
        super().__init__()

        # Encoder de features de nodo
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Encoder de features de arista
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Capas GNN con atencion a aristas
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        # Clasificador de nodos
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        edge_attr: TensorType | None = None
    ) -> TensorType:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Aristas [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
        """
        # Encode nodes
        h = self.node_encoder(x)

        # GNN layers
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv3(h, edge_index)

        # Classify nodes
        return self.node_classifier(h)


class TemporalNetworkGNN(nn.Module):
    """
    GNN con componente temporal para detectar ataques
    que evolucionan en el tiempo (APT, lateral movement).
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_time_steps: int = 10,
        num_classes: int = 6
    ) -> None:
        super().__init__()

        self.num_time_steps = num_time_steps

        # GNN por snapshot temporal
        self.spatial_conv = SAGEConv(node_features, hidden_channels)

        # LSTM para evolucion temporal
        self.temporal_rnn = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True
        )

        # Clasificador
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x_sequence: list[TensorType],
        edge_index_sequence: list[TensorType]
    ) -> TensorType:
        """
        Forward pass sobre secuencia temporal de grafos.

        Args:
            x_sequence: Lista de features por timestep
            edge_index_sequence: Lista de aristas por timestep
        """
        # Procesar cada snapshot con GNN
        spatial_embeddings = []

        for x, edge_index in zip(x_sequence, edge_index_sequence):
            h = F.relu(self.spatial_conv(x, edge_index))
            spatial_embeddings.append(h)

        # Stack: [num_nodes, time_steps, hidden]
        # Asumiendo mismo numero de nodos por simplicidad
        h_temporal = torch.stack(spatial_embeddings, dim=1)

        # LSTM temporal
        rnn_out, _ = self.temporal_rnn(h_temporal)

        # Usar ultimo timestep
        h_final = rnn_out[:, -1, :]

        return self.classifier(h_final)


def extract_host_features(
    host_ip: str,
    flows: list[NetworkFlow],
    time_window: float = 60.0
) -> np.ndarray:
    """
    Extrae features de un host basado en su trafico.
    """
    # Separar flujos de entrada y salida
    outgoing = [f for f in flows if f.src_ip == host_ip]
    incoming = [f for f in flows if f.dst_ip == host_ip]

    # Features de volumen
    num_out_connections = len(outgoing)
    num_in_connections = len(incoming)
    bytes_out = sum(f.bytes_sent for f in outgoing)
    bytes_in = sum(f.bytes_received for f in incoming)
    packets_out = sum(f.packets_sent for f in outgoing)
    packets_in = sum(f.packets_received for f in incoming)

    # Features de diversidad
    unique_dst_ips = len(set(f.dst_ip for f in outgoing))
    unique_src_ips = len(set(f.src_ip for f in incoming))
    unique_dst_ports = len(set(f.dst_port for f in outgoing))
    unique_src_ports = len(set(f.src_port for f in incoming))

    # Features de protocolo
    tcp_ratio = sum(1 for f in outgoing if f.protocol == 'TCP') / max(len(outgoing), 1)
    udp_ratio = sum(1 for f in outgoing if f.protocol == 'UDP') / max(len(outgoing), 1)

    # Features de comportamiento
    avg_duration = np.mean([f.duration for f in outgoing]) if outgoing else 0
    avg_bytes_per_packet = bytes_out / max(packets_out, 1)

    # Features de patron (scan detection)
    if num_out_connections > 0:
        # Alto ratio destinos/conexiones indica scanning
        scan_ratio = unique_dst_ips / num_out_connections
    else:
        scan_ratio = 0

    # Features de anomalia temporal
    if len(outgoing) > 1:
        timestamps = sorted([f.timestamp for f in outgoing])
        inter_arrival = np.diff(timestamps)
        iat_std = np.std(inter_arrival) if len(inter_arrival) > 0 else 0
    else:
        iat_std = 0

    features = np.array([
        num_out_connections,
        num_in_connections,
        bytes_out / 1e6,  # MB
        bytes_in / 1e6,
        packets_out / 1000,
        packets_in / 1000,
        unique_dst_ips,
        unique_src_ips,
        unique_dst_ports,
        unique_src_ports,
        tcp_ratio,
        udp_ratio,
        avg_duration,
        avg_bytes_per_packet / 1000,
        scan_ratio,
        iat_std
    ], dtype=np.float32)

    return features


def build_network_graph(
    flows: list[NetworkFlow],
    labels: dict[str, AttackType] | None = None
) -> Data:
    """
    Construye grafo de red desde flujos.
    """
    # Obtener hosts unicos
    hosts = set()
    for f in flows:
        hosts.add(f.src_ip)
        hosts.add(f.dst_ip)

    host_to_idx = {h: i for i, h in enumerate(sorted(hosts))}
    num_nodes = len(hosts)

    # Extraer features por host
    node_features = []
    node_labels = []

    for host in sorted(hosts):
        feat = extract_host_features(host, flows)
        node_features.append(feat)

        if labels and host in labels:
            node_labels.append(labels[host].value)
        else:
            node_labels.append(-1)

    x = torch.tensor(np.stack(node_features), dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.long)

    # Normalizar features
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

    # Construir aristas (conexiones)
    edges_src = []
    edges_dst = []
    edge_features = []

    # Agrupar flujos por par de hosts
    flow_groups: dict[tuple, list] = defaultdict(list)
    for f in flows:
        key = (f.src_ip, f.dst_ip)
        flow_groups[key].append(f)

    for (src, dst), group_flows in flow_groups.items():
        src_idx = host_to_idx[src]
        dst_idx = host_to_idx[dst]

        edges_src.append(src_idx)
        edges_dst.append(dst_idx)

        # Features de la conexion agregada
        edge_feat = [
            len(group_flows),  # Numero de flujos
            sum(f.bytes_sent for f in group_flows) / 1e6,
            sum(f.packets_sent for f in group_flows) / 1000,
            np.mean([f.duration for f in group_flows]),
            len(set(f.dst_port for f in group_flows)),  # Puertos unicos
        ]
        edge_features.append(edge_feat)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Mascara de entrenamiento
    train_mask = y != -1

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        host_to_idx=host_to_idx
    )
```

---

## Malware Detection con Call Graphs

### Call Graphs para Analisis de Malware

```
CALL GRAPHS EN ANALISIS DE MALWARE
==================================

Un call graph representa las llamadas entre funciones:
- Nodos: Funciones del programa
- Aristas: Llamadas de funcion

EJEMPLO - PROGRAMA BENIGNO vs MALWARE:
--------------------------------------

Programa benigno (editor de texto):

    [main]
      |
      +-->[init_gui]
      |       |
      |       +-->[load_config]
      |
      +-->[event_loop]
              |
              +-->[handle_keypress]
              |
              +-->[save_file]


Malware (ransomware):

    [main]
      |
      +-->[check_sandbox]     <-- Evasion!
      |       |
      |       +-->[detect_vm]
      |
      +-->[enumerate_files]   <-- Reconocimiento
      |       |
      |       +-->[find_documents]
      |
      +-->[encrypt_files]     <-- Payload
      |       |
      |       +-->[generate_key]
      |       +-->[write_ransom_note]
      |
      +-->[connect_c2]        <-- C&C
              |
              +-->[send_key]
              +-->[receive_commands]


CARACTERISTICAS DETECTABLES CON GNN:
------------------------------------

1. PATRONES ESTRUCTURALES:
   - Funciones de evasion (check_sandbox, detect_vm)
   - Funciones de exfiltracion (connect_c2, send_data)
   - Funciones de cifrado (encrypt, generate_key)

2. TOPOLOGIA:
   - Malware: Estructura mas "modular" para evadir
   - Benigno: Estructura mas "organica"

3. API CALLS:
   - Malware: APIs de crypto, red, registry
   - Benigno: APIs de UI, I/O normales
```

### Implementacion: Clasificador de Malware

```python
"""
Clasificador de malware basado en call graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias
import numpy as np

TensorType: TypeAlias = torch.Tensor


class MalwareFamily(Enum):
    """Familias de malware."""
    BENIGN = 0
    RANSOMWARE = 1
    TROJAN = 2
    WORM = 3
    ROOTKIT = 4
    SPYWARE = 5
    ADWARE = 6


@dataclass
class Function:
    """Una funcion en el binario."""
    name: str
    address: int
    size: int
    num_basic_blocks: int
    num_instructions: int
    api_calls: list[str]
    strings: list[str]
    cyclomatic_complexity: int


class MalwareGNN(nn.Module):
    """
    GNN para clasificacion de malware basada en call graphs.

    Arquitectura:
    1. Encode features de funcion
    2. Message passing sobre call graph
    3. Pooling global para embedding del programa
    4. Clasificacion de familia de malware
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 7,
        pooling: str = "mean_max"
    ) -> None:
        super().__init__()

        self.pooling = pooling

        # Encoder de funcion
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Capas GNN (usamos GAT para atencion a llamadas importantes)
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))

        # Clasificador
        pool_dim = hidden_channels * 2 if pooling == "mean_max" else hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        batch: TensorType
    ) -> TensorType:
        """
        Forward pass para batch de grafos.

        Args:
            x: Node features [total_nodes, features]
            edge_index: Aristas [2, total_edges]
            batch: Asignacion nodo->grafo [total_nodes]
        """
        # Encode
        h = self.node_encoder(x)

        # Message passing
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=0.3, training=self.training)

        # Pooling global
        if self.pooling == "mean":
            graph_emb = global_mean_pool(h, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(h, batch)
        else:  # mean_max
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            graph_emb = torch.cat([h_mean, h_max], dim=1)

        # Clasificar
        return self.classifier(graph_emb)


class HierarchicalMalwareGNN(nn.Module):
    """
    GNN jerarquico para analisis de malware.

    Nivel 1: Funciones individuales
    Nivel 2: Modulos/Clusters de funciones
    Nivel 3: Programa completo
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_classes: int = 7
    ) -> None:
        super().__init__()

        # Nivel 1: Funciones
        self.func_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.ReLU()
        )
        self.func_conv = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)

        # Nivel 2: Clusters (pooling local)
        self.cluster_conv = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)

        # Nivel 3: Programa
        self.program_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        batch: TensorType,
        cluster_assignment: TensorType | None = None
    ) -> TensorType:
        """Forward pass jerarquico."""
        # Nivel 1: Procesar funciones
        h = self.func_encoder(x)
        h = F.elu(self.func_conv(h, edge_index))

        # Nivel 2: Si hay clusters, agregar
        if cluster_assignment is not None:
            # Pooling por cluster (simplificado)
            h_clustered = global_mean_pool(h, cluster_assignment)
            # En practica, construir grafo de clusters y aplicar GNN
        else:
            h_clustered = h

        # Nivel 3: Pooling global
        h_mean = global_mean_pool(h_clustered, batch)
        h_max = global_max_pool(h_clustered, batch)
        h_program = torch.cat([h_mean, h_max], dim=1)

        return self.program_classifier(h_program)


# Lista de APIs sospechosas por categoria
SUSPICIOUS_APIS = {
    'crypto': ['CryptEncrypt', 'CryptDecrypt', 'CryptGenKey', 'CryptHashData'],
    'network': ['WSAStartup', 'connect', 'send', 'recv', 'InternetOpen', 'HttpOpenRequest'],
    'process': ['CreateProcess', 'OpenProcess', 'WriteProcessMemory', 'VirtualAlloc'],
    'registry': ['RegSetValue', 'RegCreateKey', 'RegDeleteKey'],
    'file': ['CreateFile', 'WriteFile', 'DeleteFile', 'MoveFile'],
    'evasion': ['IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'GetTickCount', 'Sleep'],
    'injection': ['VirtualAllocEx', 'WriteProcessMemory', 'CreateRemoteThread', 'NtUnmapViewOfSection'],
}


def extract_function_features(func: Function) -> np.ndarray:
    """
    Extrae features de una funcion para el GNN.
    """
    # Features estructurales
    structural = [
        func.size / 10000,
        func.num_basic_blocks / 100,
        func.num_instructions / 1000,
        func.cyclomatic_complexity / 50,
    ]

    # Features de API calls por categoria
    api_features = []
    for category, apis in SUSPICIOUS_APIS.items():
        count = sum(1 for api in func.api_calls if api in apis)
        api_features.append(count / 10)  # Normalizado

    # Features de strings sospechosas
    suspicious_strings = ['http://', 'https://', '.exe', '.dll', 'password', 'encrypt']
    string_features = [
        sum(1 for s in func.strings if any(susp in s.lower() for susp in suspicious_strings)) / 10
    ]

    # Nombre de funcion (heuristicas)
    name_features = [
        1.0 if 'crypt' in func.name.lower() else 0.0,
        1.0 if 'connect' in func.name.lower() else 0.0,
        1.0 if 'inject' in func.name.lower() else 0.0,
    ]

    features = np.array(
        structural + api_features + string_features + name_features,
        dtype=np.float32
    )

    return features


def build_call_graph(
    functions: list[Function],
    calls: list[tuple[int, int]],  # (caller_idx, callee_idx)
    label: MalwareFamily = MalwareFamily.BENIGN
) -> Data:
    """
    Construye call graph de un programa.

    Args:
        functions: Lista de funciones
        calls: Lista de llamadas (caller, callee)
        label: Familia de malware (o benigno)
    """
    # Features de nodos
    node_features = []
    for func in functions:
        feat = extract_function_features(func)
        node_features.append(feat)

    x = torch.tensor(np.stack(node_features), dtype=torch.float)

    # Aristas
    if calls:
        edge_index = torch.tensor(
            [[c[0] for c in calls], [c[1] for c in calls]],
            dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Label del grafo
    y = torch.tensor([label.value], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


def train_malware_classifier(
    model: MalwareGNN,
    train_graphs: list[Data],
    val_graphs: list[Data],
    epochs: int = 100,
    batch_size: int = 32
) -> dict[str, list[float]]:
    """
    Entrena clasificador de malware.
    """
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Class weights para datos desbalanceados
    class_counts = torch.bincount(torch.tensor([g.y.item() for g in train_graphs]))
    class_weights = 1.0 / (class_counts.float() + 1)
    class_weights = class_weights / class_weights.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        val_acc = correct / total
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")

    return history
```

---

## Deteccion de Botnets

### Botnets como Grafos

```
ESTRUCTURA DE BOTNETS
=====================

Los botnets tienen estructuras caracteristicas:

1. CENTRALIZADO (Star topology):
-------------------------------
    Todos los bots conectan a un C&C central

              [C&C]
             / | | \
            /  | |  \
           /   | |   \
        [Bot][Bot][Bot][Bot]

    Deteccion: Nodo con alto grado de entrada


2. PEER-TO-PEER (Mesh):
-----------------------
    Bots se comunican entre si

        [Bot]---[Bot]
          |   X   |
        [Bot]---[Bot]

    Deteccion: Comunidad densa, patrones de comunicacion


3. JERARQUICO (Tree):
---------------------
    Multiples niveles de C&C

           [Master C&C]
           /          \
      [Sub-C&C]    [Sub-C&C]
       /    \       /    \
    [Bot] [Bot]  [Bot] [Bot]

    Deteccion: Estructura de arbol, nodos intermedios


4. FAST-FLUX:
-------------
    C&C cambia rapidamente de IP

    t=0: [Bot] --> [C&C @ IP1]
    t=1: [Bot] --> [C&C @ IP2]
    t=2: [Bot] --> [C&C @ IP3]

    Deteccion: Patrones temporales, DNS anomalo


FEATURES PARA DETECCION:
------------------------

Por nodo (Bot/C&C):
- Grado de entrada/salida
- Regularidad de conexiones
- Destinos geograficamente dispersos
- Trafico nocturno

Por comunidad:
- Densidad de conexiones
- Patron de comunicacion sincronizado
- Similitud de comportamiento
```

### Implementacion: Detector de Botnets

```python
"""
Sistema de deteccion de botnets con GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from collections import defaultdict

TensorType: TypeAlias = torch.Tensor


class BotnetRole(Enum):
    """Rol de un nodo en la red."""
    NORMAL = 0
    BOT = 1
    C2_SERVER = 2
    PROXY = 3


class BotnetDetectorGNN(nn.Module):
    """
    GNN para deteccion de botnets.

    Detecta:
    - Nodos que son bots
    - Servidores C&C
    - Proxies/relays
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_classes: int = 4
    ) -> None:
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # GNN layers
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        # Clasificador de nodos
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x: TensorType, edge_index: TensorType) -> TensorType:
        h = self.encoder(x)

        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv3(h, edge_index)

        return self.classifier(h)


class CommunityAwareBotnetGNN(nn.Module):
    """
    GNN que detecta comunidades sospechosas (posibles botnets).

    Usa pooling jerarquico para identificar clusters anomalos.
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64
    ) -> None:
        super().__init__()

        # Node-level GNN
        self.node_conv = SAGEConv(node_features, hidden_channels)

        # Community-level
        self.community_conv = SAGEConv(hidden_channels, hidden_channels)

        # Clasificador de comunidad (botnet vs normal)
        self.community_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 2)
        )

    def forward(
        self,
        x: TensorType,
        edge_index: TensorType,
        community_assignment: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Returns:
            node_pred: Prediccion por nodo
            community_pred: Prediccion por comunidad
        """
        # Node embeddings
        h = F.relu(self.node_conv(x, edge_index))

        # Agregar por comunidad
        num_communities = community_assignment.max().item() + 1
        community_emb = torch.zeros(num_communities, h.size(1))

        for c in range(num_communities):
            mask = community_assignment == c
            if mask.sum() > 0:
                community_emb[c] = h[mask].mean(dim=0)

        # Clasificar comunidades
        community_pred = self.community_classifier(community_emb)

        # Propagar prediccion de comunidad a nodos
        node_pred = community_pred[community_assignment]

        return node_pred, community_pred


def detect_botnet_communities(
    edge_index: TensorType,
    num_nodes: int,
    min_community_size: int = 5
) -> tuple[TensorType, list[set]]:
    """
    Detecta comunidades en el grafo usando Louvain o similar.

    Returns:
        community_assignment: [num_nodes] indice de comunidad
        communities: Lista de conjuntos de nodos
    """
    import networkx as nx
    from networkx.algorithms import community

    # Crear grafo NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    G.add_edges_from(zip(src, dst))

    # Detectar comunidades
    communities_gen = community.louvain_communities(G, seed=42)
    communities = [c for c in communities_gen if len(c) >= min_community_size]

    # Asignar nodos a comunidades
    assignment = torch.zeros(num_nodes, dtype=torch.long)
    for i, comm in enumerate(communities):
        for node in comm:
            assignment[node] = i

    return assignment, communities


def extract_botnet_features(
    ip: str,
    flows: list,  # NetworkFlow
    time_windows: list[float]
) -> np.ndarray:
    """
    Extrae features especificas para deteccion de botnets.
    """
    features = []

    # Features por ventana temporal
    for t_start, t_end in zip(time_windows[:-1], time_windows[1:]):
        window_flows = [f for f in flows
                       if f.src_ip == ip and t_start <= f.timestamp < t_end]

        # Volumen en ventana
        features.append(len(window_flows))

        # Destinos unicos
        features.append(len(set(f.dst_ip for f in window_flows)))

    # Features agregadas
    all_flows = [f for f in flows if f.src_ip == ip]

    # Regularidad (stddev de inter-arrival time)
    if len(all_flows) > 1:
        timestamps = sorted([f.timestamp for f in all_flows])
        iat = np.diff(timestamps)
        features.append(np.std(iat))
        features.append(np.mean(iat))
    else:
        features.extend([0, 0])

    # Ratio de puertos conocidos vs desconocidos
    known_ports = {80, 443, 53, 25, 110, 143}
    port_count = sum(1 for f in all_flows if f.dst_port in known_ports)
    features.append(port_count / max(len(all_flows), 1))

    # Conexiones nocturnas (00:00 - 06:00)
    night_flows = [f for f in all_flows
                   if 0 <= (f.timestamp % 86400) / 3600 < 6]
    features.append(len(night_flows) / max(len(all_flows), 1))

    # Bytes ratio (enviado vs recibido)
    bytes_out = sum(f.bytes_sent for f in all_flows)
    bytes_in = sum(f.bytes_received for f in all_flows)
    features.append(bytes_out / max(bytes_in + bytes_out, 1))

    return np.array(features, dtype=np.float32)
```

---

## Threat Intelligence con Knowledge Graphs

### Knowledge Graphs de Amenazas

```
KNOWLEDGE GRAPH DE AMENAZAS (TI)
================================

Modelar conocimiento de amenazas como grafo:

Entidades:
- Threat Actor (APT28, Lazarus, etc.)
- Malware (Emotet, TrickBot, etc.)
- Vulnerability (CVE-XXXX-XXXX)
- Attack Technique (MITRE ATT&CK)
- Indicator (IP, hash, domain)
- Target (sector, pais)

Relaciones:
- uses (Actor -> Malware)
- exploits (Malware -> Vulnerability)
- implements (Malware -> Technique)
- indicates (Indicator -> Malware)
- targets (Actor -> Target)


EJEMPLO DE SUBGRAFO:
--------------------

    [APT28]
       |
      uses
       |
       v
    [Sofacy]--implements-->[T1566: Phishing]
       |                        |
   exploits                  related
       |                        |
       v                        v
  [CVE-2017-0199]         [T1204: User Exec]
       |
   indicated_by
       |
       v
  [evil.domain.com]


APLICACIONES DE GNN:
--------------------

1. LINK PREDICTION:
   Predecir nuevas relaciones (que malware usara un actor)

2. ENTIDAD EMBEDDING:
   Embeddings de amenazas para similitud

3. ANOMALY DETECTION:
   Detectar indicadores que no encajan

4. ATTRIBUTION:
   Atribuir ataques a actores basado en patrones
```

### Implementacion: TI Knowledge Graph

```python
"""
Knowledge Graph de Threat Intelligence con GNNs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias

TensorType: TypeAlias = torch.Tensor


class EntityType(Enum):
    """Tipos de entidades en TI."""
    THREAT_ACTOR = 0
    MALWARE = 1
    VULNERABILITY = 2
    TECHNIQUE = 3  # MITRE ATT&CK
    INDICATOR = 4
    TARGET = 5


class RelationType(Enum):
    """Tipos de relaciones."""
    USES = 0            # Actor -> Malware
    EXPLOITS = 1        # Malware -> Vulnerability
    IMPLEMENTS = 2      # Malware -> Technique
    INDICATES = 3       # Indicator -> Malware
    TARGETS = 4         # Actor -> Target
    RELATED_TO = 5      # Cualquier entidad


@dataclass
class ThreatEntity:
    """Entidad en el knowledge graph."""
    id: str
    entity_type: EntityType
    name: str
    description: str
    attributes: dict


class ThreatIntelGNN(nn.Module):
    """
    GNN para knowledge graph de threat intelligence.

    Usa R-GCN para manejar multiples tipos de relaciones.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        hidden_channels: int = 64,
        num_layers: int = 2
    ) -> None:
        super().__init__()

        # Embeddings de entidades
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)

        # R-GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(embedding_dim, hidden_channels, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))

        # Para link prediction
        self.relation_embedding = nn.Embedding(num_relations, hidden_channels)

    def encode(
        self,
        entity_ids: TensorType,
        edge_index: TensorType,
        edge_type: TensorType
    ) -> TensorType:
        """
        Encode entities into embeddings.
        """
        x = self.entity_embedding(entity_ids)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        return x

    def decode(
        self,
        z: TensorType,
        edge_index: TensorType,
        edge_type: TensorType
    ) -> TensorType:
        """
        Decode para link prediction.
        Score = h_s^T * R * h_o
        """
        src, dst = edge_index[0], edge_index[1]

        h_src = z[src]
        h_dst = z[dst]
        r_emb = self.relation_embedding(edge_type)

        # DistMult-like scoring
        score = (h_src * r_emb * h_dst).sum(dim=1)

        return score

    def forward(
        self,
        entity_ids: TensorType,
        edge_index: TensorType,
        edge_type: TensorType
    ) -> TensorType:
        """Forward para link prediction."""
        z = self.encode(entity_ids, edge_index, edge_type)
        return self.decode(z, edge_index, edge_type)


class ThreatAttributionGNN(nn.Module):
    """
    GNN para atribucion de ataques a threat actors.

    Dado un conjunto de indicadores/tecnicas observadas,
    predice que actor es mas probable.
    """

    def __init__(
        self,
        indicator_features: int,
        technique_features: int,
        hidden_channels: int = 64,
        num_actors: int = 50
    ) -> None:
        super().__init__()

        # Encoders por tipo
        self.indicator_encoder = nn.Linear(indicator_features, hidden_channels)
        self.technique_encoder = nn.Linear(technique_features, hidden_channels)

        # GNN para propagar informacion
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations=6)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=6)

        # Clasificador de atribucion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_actors)
        )

    def forward(
        self,
        indicator_features: TensorType,
        technique_features: TensorType,
        edge_index: TensorType,
        edge_type: TensorType,
        observed_mask: TensorType
    ) -> TensorType:
        """
        Predice actor mas probable dado observaciones.

        Args:
            observed_mask: Mascara de indicadores/tecnicas observadas
        """
        # Encode
        h_ind = F.relu(self.indicator_encoder(indicator_features))
        h_tech = F.relu(self.technique_encoder(technique_features))

        # Concatenar todos los nodos
        h = torch.cat([h_ind, h_tech], dim=0)

        # GNN
        h = F.relu(self.conv1(h, edge_index, edge_type))
        h = self.conv2(h, edge_index, edge_type)

        # Agregar embeddings de nodos observados
        observed_emb = h[observed_mask].mean(dim=0)

        # Clasificar actor
        return self.classifier(observed_emb)


def link_prediction_loss(
    pos_scores: TensorType,
    neg_scores: TensorType
) -> TensorType:
    """
    Loss para link prediction.
    Margin ranking loss.
    """
    return F.margin_ranking_loss(
        pos_scores,
        neg_scores,
        target=torch.ones_like(pos_scores),
        margin=1.0
    )


def train_threat_intel_gnn(
    model: ThreatIntelGNN,
    data: HeteroData,
    epochs: int = 100
) -> None:
    """
    Entrena modelo de threat intelligence.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    entity_ids = torch.arange(data.num_nodes)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Positive edges (existentes)
        pos_edge_index = data.edge_index
        pos_edge_type = data.edge_type

        # Negative edges (muestreadas)
        num_neg = pos_edge_index.size(1)
        neg_src = torch.randint(0, data.num_nodes, (num_neg,))
        neg_dst = torch.randint(0, data.num_nodes, (num_neg,))
        neg_edge_index = torch.stack([neg_src, neg_dst])
        neg_edge_type = torch.randint(0, 6, (num_neg,))

        # Forward
        z = model.encode(entity_ids, pos_edge_index, pos_edge_type)
        pos_scores = model.decode(z, pos_edge_index, pos_edge_type)
        neg_scores = model.decode(z, neg_edge_index, neg_edge_type)

        # Loss
        loss = link_prediction_loss(pos_scores, neg_scores)

        # Backward
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## APT Detection con GNNs Temporales

### APTs y Patrones Temporales

```
APT (ADVANCED PERSISTENT THREAT) DETECTION
==========================================

Las APT son ataques sofisticados con fases temporales:

FASES TIPICAS:
--------------

Fase 1: RECONOCIMIENTO (dias/semanas)
    - Escaneo pasivo
    - OSINT
    - Phishing dirigido

Fase 2: INTRUSION INICIAL (dia 0)
    - Exploit de vulnerabilidad
    - Spear phishing exitoso
    - Supply chain

Fase 3: ESTABLECIMIENTO (dias 1-7)
    - Persistencia
    - Escalada de privilegios
    - Movimiento lateral inicial

Fase 4: EXPANSION (semanas)
    - Movimiento lateral extenso
    - Compromiso de mas sistemas
    - Instalacion de backdoors

Fase 5: EXFILTRACION (continuo)
    - Robo de datos
    - Comunicacion C2
    - Mantenimiento de acceso


TIMELINE EN GRAFO:
------------------

t=0                t=1                t=2
[Entry]           [Entry]           [Entry]
   |                 |                 |
   v                 v                 v
[Host1]          [Host1]--->[Host2] [Host1]--->[Host2]
                                           \       |
                                            \      v
                                             +->[Host3]
                                                   |
                                                   v
                                              [Exfil Target]

El grafo CRECE y EVOLUCIONA con el ataque.


POR QUE GNN TEMPORAL:
---------------------
1. Captura evolucion del ataque
2. Detecta patrones de expansion
3. Identifica "anomalias de crecimiento"
4. Correlaciona eventos separados en tiempo
```

### Implementacion: APT Detector Temporal

```python
"""
Detector de APT usando GNN temporal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from typing import TypeAlias
import numpy as np

TensorType: TypeAlias = torch.Tensor


class TemporalGraphSnapshot:
    """Un snapshot del grafo en un momento dado."""

    def __init__(
        self,
        timestamp: float,
        x: TensorType,
        edge_index: TensorType,
        edge_attr: TensorType | None = None
    ) -> None:
        self.timestamp = timestamp
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


class APTDetectorTGNN(nn.Module):
    """
    GNN temporal para deteccion de APT.

    Procesa secuencia de snapshots del grafo de red
    y detecta patrones de APT.
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 64,
        num_time_steps: int = 24,  # 24 horas por ejemplo
        num_classes: int = 2  # Normal / APT
    ) -> None:
        super().__init__()

        self.num_time_steps = num_time_steps

        # GNN espacial por snapshot
        self.spatial_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.ReLU()
        )
        self.spatial_conv = SAGEConv(hidden_channels, hidden_channels)

        # RNN temporal para evoluccion
        self.temporal_gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Attention sobre tiempo para focus en eventos criticos
        self.time_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels * 2,
            num_heads=4,
            batch_first=True
        )

        # Detector de anomalia
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )

        # Detector de fase de ataque
        self.phase_detector = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 5)  # 5 fases de APT
        )

    def forward(
        self,
        snapshots: list[TemporalGraphSnapshot],
        target_node: int
    ) -> tuple[TensorType, TensorType]:
        """
        Detecta APT para un nodo especifico.

        Args:
            snapshots: Lista de snapshots temporales
            target_node: Nodo a analizar

        Returns:
            anomaly_score: Probabilidad de APT
            phase_probs: Probabilidades de cada fase
        """
        # Procesar cada snapshot espacialmente
        temporal_embeddings = []

        for snap in snapshots:
            h = self.spatial_encoder(snap.x)
            h = F.relu(self.spatial_conv(h, snap.edge_index))

            # Embedding del nodo target en este tiempo
            node_emb = h[target_node]
            temporal_embeddings.append(node_emb)

        # Stack: [1, time_steps, hidden]
        h_temporal = torch.stack(temporal_embeddings, dim=0).unsqueeze(0)

        # GRU temporal
        gru_out, _ = self.temporal_gru(h_temporal)

        # Self-attention sobre tiempo
        attn_out, attn_weights = self.time_attention(
            gru_out, gru_out, gru_out
        )

        # Usar ultimo timestep con atencion
        h_final = attn_out[0, -1, :]

        # Detectar anomalia y fase
        anomaly_score = self.anomaly_detector(h_final)
        phase_probs = self.phase_detector(h_final)

        return anomaly_score, phase_probs


class EvolveGCN(nn.Module):
    """
    EvolveGCN: GCN con pesos que evolucionan en el tiempo.

    Los pesos del GCN son generados por un RNN,
    permitiendo adaptacion a cambios en el grafo.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # GRU para evolucionar pesos de GCN
        self.weight_gru = nn.GRUCell(
            input_size=in_channels * hidden_channels,
            hidden_size=in_channels * hidden_channels
        )

        # Pesos iniciales
        self.initial_weight = nn.Parameter(
            torch.randn(in_channels, hidden_channels)
        )

        # Capa de salida
        self.out_layer = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_sequence: list[TensorType],
        edge_index_sequence: list[TensorType]
    ) -> list[TensorType]:
        """
        Forward pass sobre secuencia temporal.
        """
        outputs = []
        weight = self.initial_weight.view(-1)

        for x, edge_index in zip(x_sequence, edge_index_sequence):
            # Reconstruir peso como matriz
            W = weight.view(self.in_channels, self.hidden_channels)

            # GCN manual con peso evolucionado
            # Normalizar adyacencia (simplificado)
            row, col = edge_index
            deg = torch.bincount(col, minlength=x.size(0)).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            # h = D^-0.5 A D^-0.5 X W
            h = x @ W
            h = deg_inv_sqrt.view(-1, 1) * h

            # Scatter add para agregacion
            out = torch.zeros_like(h)
            out.scatter_add_(0, col.unsqueeze(1).expand_as(h), h[row])
            h = deg_inv_sqrt.view(-1, 1) * out

            h = F.relu(h)
            outputs.append(self.out_layer(h))

            # Evolucionar peso
            weight = self.weight_gru(weight.unsqueeze(0), weight.unsqueeze(0))
            weight = weight.squeeze(0)

        return outputs


def detect_lateral_movement(
    snapshots: list[TemporalGraphSnapshot],
    entry_node: int,
    suspicious_threshold: float = 0.8
) -> list[int]:
    """
    Detecta posible movimiento lateral desde un punto de entrada.

    Returns:
        Lista de nodos potencialmente comprometidos
    """
    compromised = {entry_node}
    timeline = []

    for i, snap in enumerate(snapshots):
        src, dst = snap.edge_index[0].tolist(), snap.edge_index[1].tolist()

        # Buscar conexiones desde nodos comprometidos
        new_compromised = set()

        for s, d in zip(src, dst):
            if s in compromised and d not in compromised:
                # Potencial movimiento lateral
                new_compromised.add(d)

        if new_compromised:
            timeline.append({
                'timestamp': snap.timestamp,
                'new_nodes': new_compromised
            })
            compromised.update(new_compromised)

    return list(compromised), timeline
```

---

## Implementacion Completa

### Sistema Integrado de Deteccion

```python
"""
Sistema completo de deteccion de amenazas con GNN.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias, Protocol
import numpy as np
from datetime import datetime

TensorType: TypeAlias = torch.Tensor


class ThreatType(Enum):
    """Tipos de amenazas detectables."""
    NORMAL = 0
    FRAUD = 1
    INTRUSION = 2
    MALWARE = 3
    BOTNET = 4
    APT = 5


@dataclass
class Alert:
    """Alerta generada por el sistema."""
    timestamp: datetime
    threat_type: ThreatType
    severity: float  # 0-1
    affected_entities: list[str]
    description: str
    evidence: dict
    recommendations: list[str]


class ThreatDetector(Protocol):
    """Interface para detectores de amenazas."""

    def detect(self, data) -> list[Alert]:
        """Detecta amenazas en los datos."""
        ...


class UnifiedThreatDetectionSystem:
    """
    Sistema unificado de deteccion de amenazas.

    Combina multiples modelos GNN especializados.
    """

    def __init__(
        self,
        fraud_model: nn.Module,
        ids_model: nn.Module,
        malware_model: nn.Module,
        botnet_model: nn.Module,
        apt_model: nn.Module
    ) -> None:
        self.detectors = {
            ThreatType.FRAUD: fraud_model,
            ThreatType.INTRUSION: ids_model,
            ThreatType.MALWARE: malware_model,
            ThreatType.BOTNET: botnet_model,
            ThreatType.APT: apt_model
        }

        # Thresholds por tipo
        self.thresholds = {
            ThreatType.FRAUD: 0.7,
            ThreatType.INTRUSION: 0.6,
            ThreatType.MALWARE: 0.8,
            ThreatType.BOTNET: 0.75,
            ThreatType.APT: 0.5  # Bajo porque APT es muy critico
        }

    def analyze(
        self,
        network_graph,
        transaction_graph=None,
        call_graphs=None,
        temporal_snapshots=None
    ) -> list[Alert]:
        """
        Analiza multiples fuentes de datos y genera alertas.
        """
        alerts = []

        # Deteccion de intrusion (red)
        if network_graph is not None:
            ids_alerts = self._detect_intrusions(network_graph)
            alerts.extend(ids_alerts)

        # Deteccion de fraude (transacciones)
        if transaction_graph is not None:
            fraud_alerts = self._detect_fraud(transaction_graph)
            alerts.extend(fraud_alerts)

        # Deteccion de malware (call graphs)
        if call_graphs is not None:
            malware_alerts = self._detect_malware(call_graphs)
            alerts.extend(malware_alerts)

        # Deteccion de botnet (red)
        if network_graph is not None:
            botnet_alerts = self._detect_botnets(network_graph)
            alerts.extend(botnet_alerts)

        # Deteccion de APT (temporal)
        if temporal_snapshots is not None:
            apt_alerts = self._detect_apt(temporal_snapshots)
            alerts.extend(apt_alerts)

        # Correlacionar alertas
        alerts = self._correlate_alerts(alerts)

        # Ordenar por severidad
        alerts.sort(key=lambda a: a.severity, reverse=True)

        return alerts

    def _detect_intrusions(self, graph) -> list[Alert]:
        """Detecta intrusiones en grafo de red."""
        model = self.detectors[ThreatType.INTRUSION]
        model.eval()

        alerts = []

        with torch.no_grad():
            predictions = model(graph.x, graph.edge_index)
            probs = torch.softmax(predictions, dim=1)

            for node_idx, prob in enumerate(probs):
                max_prob, pred_class = prob.max(dim=0)

                if pred_class.item() != 0 and max_prob.item() > self.thresholds[ThreatType.INTRUSION]:
                    # Nodo anomalo detectado
                    host_ip = list(graph.host_to_idx.keys())[node_idx]

                    alert = Alert(
                        timestamp=datetime.now(),
                        threat_type=ThreatType.INTRUSION,
                        severity=max_prob.item(),
                        affected_entities=[host_ip],
                        description=f"Actividad de intrusion detectada en {host_ip}",
                        evidence={
                            'prediction_class': pred_class.item(),
                            'confidence': max_prob.item(),
                            'node_features': graph.x[node_idx].tolist()
                        },
                        recommendations=[
                            "Aislar host de la red",
                            "Revisar logs de conexion",
                            "Escanear en busca de malware"
                        ]
                    )
                    alerts.append(alert)

        return alerts

    def _detect_fraud(self, graph) -> list[Alert]:
        """Detecta fraude en grafo de transacciones."""
        model = self.detectors[ThreatType.FRAUD]
        model.eval()

        alerts = []

        with torch.no_grad():
            predictions = model(
                graph.x_dict,
                graph.edge_index_dict
            )
            probs = torch.softmax(predictions, dim=1)

            for acc_idx, prob in enumerate(probs):
                fraud_prob = prob[1:].sum()  # Suma de clases de fraude

                if fraud_prob > self.thresholds[ThreatType.FRAUD]:
                    alert = Alert(
                        timestamp=datetime.now(),
                        threat_type=ThreatType.FRAUD,
                        severity=fraud_prob.item(),
                        affected_entities=[f"Account_{acc_idx}"],
                        description=f"Posible fraude detectado en cuenta {acc_idx}",
                        evidence={
                            'fraud_probability': fraud_prob.item(),
                            'fraud_type_probs': prob.tolist()
                        },
                        recommendations=[
                            "Congelar cuenta temporalmente",
                            "Contactar al titular",
                            "Revisar transacciones recientes"
                        ]
                    )
                    alerts.append(alert)

        return alerts

    def _detect_malware(self, call_graphs: list) -> list[Alert]:
        """Detecta malware en call graphs."""
        model = self.detectors[ThreatType.MALWARE]
        model.eval()

        alerts = []

        for i, graph in enumerate(call_graphs):
            with torch.no_grad():
                batch = torch.zeros(graph.x.size(0), dtype=torch.long)
                prediction = model(graph.x, graph.edge_index, batch)
                probs = torch.softmax(prediction, dim=1)

                max_prob, pred_class = probs[0].max(dim=0)

                if pred_class.item() != 0:  # No benigno
                    family_names = ['Benign', 'Ransomware', 'Trojan', 'Worm', 'Rootkit', 'Spyware', 'Adware']

                    alert = Alert(
                        timestamp=datetime.now(),
                        threat_type=ThreatType.MALWARE,
                        severity=max_prob.item(),
                        affected_entities=[f"Binary_{i}"],
                        description=f"Malware detectado: {family_names[pred_class.item()]}",
                        evidence={
                            'malware_family': family_names[pred_class.item()],
                            'confidence': max_prob.item(),
                            'all_probs': {f: p for f, p in zip(family_names, probs[0].tolist())}
                        },
                        recommendations=[
                            "Poner en cuarentena inmediatamente",
                            "Escanear sistemas relacionados",
                            "Buscar IOCs en la red"
                        ]
                    )
                    alerts.append(alert)

        return alerts

    def _detect_botnets(self, graph) -> list[Alert]:
        """Detecta botnets en grafo de red."""
        # Implementacion similar a intrusion
        return []

    def _detect_apt(self, snapshots: list) -> list[Alert]:
        """Detecta APT en secuencia temporal."""
        # Implementacion usando APTDetectorTGNN
        return []

    def _correlate_alerts(self, alerts: list[Alert]) -> list[Alert]:
        """
        Correlaciona alertas relacionadas.
        Ejemplo: Intrusion + Malware en mismo host = APT potencial
        """
        # Agrupar por entidades afectadas
        entity_alerts: dict[str, list[Alert]] = {}

        for alert in alerts:
            for entity in alert.affected_entities:
                if entity not in entity_alerts:
                    entity_alerts[entity] = []
                entity_alerts[entity].append(alert)

        # Buscar correlaciones
        correlated = []

        for entity, entity_alert_list in entity_alerts.items():
            if len(entity_alert_list) > 1:
                # Multiples alertas para misma entidad
                threat_types = {a.threat_type for a in entity_alert_list}

                if ThreatType.INTRUSION in threat_types and ThreatType.MALWARE in threat_types:
                    # Posible APT
                    combined_severity = max(a.severity for a in entity_alert_list) * 1.2
                    combined_severity = min(combined_severity, 1.0)

                    apt_alert = Alert(
                        timestamp=datetime.now(),
                        threat_type=ThreatType.APT,
                        severity=combined_severity,
                        affected_entities=[entity],
                        description=f"POSIBLE APT: Correlacion de intrusion y malware en {entity}",
                        evidence={
                            'correlated_alerts': [
                                {'type': a.threat_type.name, 'severity': a.severity}
                                for a in entity_alert_list
                            ]
                        },
                        recommendations=[
                            "PRIORIDAD CRITICA: Iniciar respuesta a incidentes",
                            "Aislar segmento de red",
                            "Buscar movimiento lateral",
                            "Preservar evidencia forense"
                        ]
                    )
                    correlated.append(apt_alert)

        return alerts + correlated
```

---

## Evaluacion y Metricas

### Metricas Especificas de Ciberseguridad

```
METRICAS PARA EVALUACION EN CIBERSEGURIDAD
==========================================

1. METRICAS CLASICAS:
---------------------
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * P * R / (P + R)
- AUC-ROC

Problema: Datos muy desbalanceados (0.1% positivos)


2. METRICAS AJUSTADAS:
----------------------

PRECISION @ K:
- Ordena por score descendente
- Precision en top-K predicciones
- Util para alertas priorizadas

AUPRC (Area Under Precision-Recall Curve):
- Mejor que AUC-ROC para datos desbalanceados
- Enfoca en clase positiva (amenazas)

FALSE POSITIVE RATE (por dia):
- Numero de falsas alarmas diarias
- Critico para operaciones SOC


3. METRICAS DE NEGOCIO:
-----------------------

COSTO DE ERROR:
- False Negative de ransomware >> False Positive
- Ponderar metricas por impacto

TIEMPO DE DETECCION:
- Cuanto tarda en detectar amenaza
- Critico para APT

TASA DE ALERTA ACCIONABLE:
- % de alertas que requieren accion
- Evitar "alert fatigue"


4. EVALUACION TEMPORAL:
-----------------------

Para deteccion de APT:
- Detectar en fase 2 >> fase 5
- Metrica: "Time to detect" por fase

EARLY DETECTION SCORE:
- Recompensa deteccion temprana
- Penaliza deteccion tardia
```

### Implementacion de Evaluacion

```python
"""
Evaluacion de modelos de seguridad.
"""
import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from dataclasses import dataclass
from typing import TypeAlias

TensorType: TypeAlias = torch.Tensor


@dataclass
class SecurityMetrics:
    """Metricas de evaluacion para ciberseguridad."""
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    precision_at_k: dict[int, float]
    false_positive_rate: float
    detection_latency: float | None = None


def evaluate_security_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: list[int] | None = None
) -> SecurityMetrics:
    """
    Evalua modelo de seguridad con metricas relevantes.

    Args:
        y_true: Labels verdaderos (0=normal, 1=amenaza)
        y_scores: Scores de probabilidad de amenaza
        k_values: Valores de K para precision@K
    """
    if k_values is None:
        k_values = [10, 50, 100, 500]

    # Predicciones binarias (threshold 0.5)
    y_pred = (y_scores > 0.5).astype(int)

    # Metricas basicas
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_scores)
    else:
        auc_roc = 0.0

    # AUC-PR
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(rec_curve, prec_curve)

    # Precision @ K
    precision_at_k = {}
    sorted_indices = np.argsort(y_scores)[::-1]

    for k in k_values:
        if k <= len(y_true):
            top_k_labels = y_true[sorted_indices[:k]]
            precision_at_k[k] = np.mean(top_k_labels)

    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return SecurityMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        precision_at_k=precision_at_k,
        false_positive_rate=fpr
    )


def calculate_detection_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict[str, float] | None = None
) -> float:
    """
    Calcula costo total de errores de deteccion.

    Default costs:
    - False Negative (miss threat): $100,000
    - False Positive (false alarm): $100
    - True Positive (detect threat): -$50,000 (ahorro)
    - True Negative: $0
    """
    if cost_matrix is None:
        cost_matrix = {
            'fn': 100000,
            'fp': 100,
            'tp': -50000,
            'tn': 0
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (
        fn * cost_matrix['fn'] +
        fp * cost_matrix['fp'] +
        tp * cost_matrix['tp'] +
        tn * cost_matrix['tn']
    )

    return total_cost


def early_detection_score(
    detection_times: list[float],
    actual_times: list[float],
    max_time: float = 100.0
) -> float:
    """
    Score que recompensa deteccion temprana.

    Score = mean(1 - (detection_time / actual_attack_end))
    Mayor score = deteccion mas temprana
    """
    if not detection_times:
        return 0.0

    scores = []
    for det_time, actual_time in zip(detection_times, actual_times):
        if det_time <= actual_time:
            # Deteccion exitosa
            score = 1 - (det_time / max(actual_time, 1))
        else:
            # Deteccion tardia (penalizacion)
            score = -0.5

        scores.append(score)

    return np.mean(scores)


def print_security_report(metrics: SecurityMetrics) -> None:
    """Imprime reporte de metricas de seguridad."""
    print("=" * 50)
    print("REPORTE DE EVALUACION DE SEGURIDAD")
    print("=" * 50)
    print(f"\nMETRICAS BASICAS:")
    print(f"  Precision:  {metrics.precision:.4f}")
    print(f"  Recall:     {metrics.recall:.4f}")
    print(f"  F1-Score:   {metrics.f1:.4f}")

    print(f"\nMETRICAS DE RANKING:")
    print(f"  AUC-ROC:    {metrics.auc_roc:.4f}")
    print(f"  AUC-PR:     {metrics.auc_pr:.4f}")

    print(f"\nPRECISION @ K:")
    for k, p in sorted(metrics.precision_at_k.items()):
        print(f"  P@{k}:".ljust(12) + f"{p:.4f}")

    print(f"\nOPERACIONAL:")
    print(f"  FPR:        {metrics.false_positive_rate:.4f}")
    if metrics.detection_latency:
        print(f"  Latencia:   {metrics.detection_latency:.2f}s")

    print("=" * 50)
```

---

## Ejercicios Practicos

### Ejercicio 1: Detector de Fraude en Anillo

```python
"""
Ejercicio 1: Implementar detector de fraude en anillo.

Objetivo: Detectar ciclos sospechosos en grafo de transacciones.
"""

def ejercicio_1_template() -> None:
    """
    TODO:
    1. Generar grafo de transacciones sintetico:
       - 1000 cuentas normales
       - 10 anillos de fraude (5-10 cuentas cada uno)

    2. Extraer features:
       - Participacion en ciclos
       - Montos en ciclos
       - Temporalidad de ciclos

    3. Entrenar GNN para detectar:
       - Nodos en anillos
       - Nodos normales

    4. Evaluar:
       - Precision/Recall por tipo
       - Visualizar ciclos detectados
    """
    pass
```

### Ejercicio 2: IDS con GNN Temporal

```python
"""
Ejercicio 2: Sistema de deteccion de intrusos temporal.

Objetivo: Detectar ataques que evolucionan en el tiempo.
"""

def ejercicio_2_template() -> None:
    """
    TODO:
    1. Simular trafico de red durante 24 horas:
       - Hora 0-8: Normal
       - Hora 8-10: Port scanning
       - Hora 10-15: Lateral movement
       - Hora 15-24: Normal

    2. Crear snapshots cada hora

    3. Entrenar GNN temporal para detectar:
       - Momento del ataque
       - Tipo de ataque
       - Hosts comprometidos

    4. Evaluar:
       - Early detection score
       - Precision en identificar hosts comprometidos
    """
    pass
```

### Ejercicio 3: Clasificador de Malware

```python
"""
Ejercicio 3: Clasificador de familias de malware.

Objetivo: Clasificar malware basado en call graphs.
"""

def ejercicio_3_template() -> None:
    """
    TODO:
    1. Cargar dataset de call graphs (usar EMBER o sintetico)

    2. Implementar:
       - Extraccion de features de funciones
       - GNN para clasificacion de grafos
       - Multi-head attention para funciones importantes

    3. Entrenar clasificador:
       - 6 familias de malware + benigno
       - Data augmentation para clases minoritarias

    4. Evaluar:
       - Confusion matrix por familia
       - Visualizar funciones mas importantes (atencion)
       - Comparar con baseline (sin GNN)
    """
    pass
```

---

## Resumen

```
RESUMEN: GNNS EN CIBERSEGURIDAD
===============================

APLICACIONES:
-------------
1. Fraude: Grafos de transacciones, detectar anillos y mulas
2. IDS: Grafos de red, detectar patrones de ataque
3. Malware: Call graphs, clasificar familias
4. Botnets: Comunidades anomalas, estructuras C&C
5. APT: Evolucion temporal, movimiento lateral

VENTAJAS DE GNNS:
-----------------
- Capturan relaciones que datos tabulares pierden
- Propagan informacion de nodos etiquetados
- Detectan patrones estructurales
- Semi-supervisado (pocos labels necesarios)

ARQUITECTURAS RECOMENDADAS:
---------------------------
- Fraude: HeteroGNN + deteccion de ciclos
- IDS: SAGE + features de trafico
- Malware: GAT + pooling global
- APT: Temporal GNN (EvolveGCN, TGAT)

METRICAS IMPORTANTES:
---------------------
- AUC-PR (mejor que AUC-ROC para desbalance)
- Precision@K (para alertas priorizadas)
- Costo de errores (FN >> FP en seguridad)
- Early detection score (para APT)

CONSIDERACIONES DE PRODUCCION:
------------------------------
- Escalabilidad: Mini-batch, GraphSAGE inductivo
- Latencia: Inferencia rapida para tiempo real
- Explicabilidad: Visualizar atencion/importancia
- Actualizacion: Reentrenar con nuevas amenazas
```

---

## Referencias

1. **Rao et al. (2021)** - "Graph Neural Networks for Fraud Detection: A Survey"
2. **Zhou et al. (2020)** - "Graph Neural Networks: A Review of Methods and Applications"
3. **Liu et al. (2019)** - "Heterogeneous Graph Neural Networks for Malicious Account Detection"
4. **Wang et al. (2021)** - "Network Intrusion Detection Using Graph Neural Networks"
5. **MITRE ATT&CK** - https://attack.mitre.org/
6. **PyTorch Geometric** - https://pytorch-geometric.readthedocs.io/
7. **Deep Graph Library** - https://www.dgl.ai/
