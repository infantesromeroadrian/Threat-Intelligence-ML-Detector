# Multi-Agent Reinforcement Learning (MARL)

## 1. Introduccion a MARL

### De un Agente a Multiples Agentes

```
+------------------------------------------------------------------------+
|  SINGLE-AGENT vs MULTI-AGENT RL                                        |
+------------------------------------------------------------------------+
|                                                                        |
|  SINGLE-AGENT RL:                                                      |
|  ----------------                                                      |
|                                                                        |
|      +-------+                                                         |
|      | Agent |                                                         |
|      +---+---+                                                         |
|          |                                                             |
|          v action                                                      |
|      +-------+                                                         |
|      |  Env  | <-- Estado depende solo del agente                      |
|      +---+---+                                                         |
|          |                                                             |
|          v reward, next_state                                          |
|                                                                        |
|                                                                        |
|  MULTI-AGENT RL:                                                       |
|  ---------------                                                       |
|                                                                        |
|      +-------+     +-------+     +-------+                             |
|      |Agent 1|     |Agent 2|     |Agent N|                             |
|      +---+---+     +---+---+     +---+---+                             |
|          |             |             |                                 |
|          v a1          v a2          v an                              |
|      +----------------------------------------+                        |
|      |            Environment                 |                        |
|      | Estado depende de TODOS los agentes    |                        |
|      +----------------------------------------+                        |
|          |             |             |                                 |
|          v r1,s1'      v r2,s2'      v rn,sn'                          |
|                                                                        |
|  Cada agente:                                                          |
|    - Ve (posiblemente) diferente observacion                           |
|    - Recibe (posiblemente) diferente recompensa                        |
|    - Afecta el entorno para TODOS                                      |
|                                                                        |
+------------------------------------------------------------------------+
```

### Desafios Unicos de MARL

```
+------------------------------------------------------------------------+
|  DESAFIOS EN MULTI-AGENT RL                                            |
+------------------------------------------------------------------------+
|                                                                        |
|  1. NO-ESTACIONARIEDAD                                                 |
|     ------------------                                                 |
|     Desde la perspectiva de un agente, el entorno "cambia"             |
|     porque otros agentes estan aprendiendo                             |
|                                                                        |
|     Single-agent: P(s'|s,a) es fija                                    |
|     Multi-agent:  P(s'|s,a) cambia con las politicas de otros          |
|                                                                        |
|     +---+                                                              |
|     | A |---> El mundo se ve diferente                                 |
|     +---+     porque B esta aprendiendo                                |
|       |            |                                                   |
|       v            v                                                   |
|     Env <------+---+---+                                               |
|                | B | aprendiendo                                       |
|                +---+                                                   |
|                                                                        |
|                                                                        |
|  2. CREDIT ASSIGNMENT                                                  |
|     -----------------                                                  |
|     Quien es responsable del exito/fracaso del equipo?                 |
|                                                                        |
|     Equipo gana +100... quien contribuyo mas?                          |
|                                                                        |
|                                                                        |
|  3. EXPLOSION COMBINATORIA                                             |
|     ----------------------                                             |
|     Espacio de acciones conjuntas crece exponencialmente               |
|                                                                        |
|     N agentes, cada uno con K acciones                                 |
|     Acciones conjuntas: K^N                                            |
|                                                                        |
|     10 agentes, 5 acciones cada uno = 5^10 = ~10 millones              |
|                                                                        |
|                                                                        |
|  4. EQUILIBRIOS MULTIPLES                                              |
|     --------------------                                               |
|     Pueden existir multiples soluciones estables                       |
|     No hay "optimo" claro en juegos competitivos                       |
|                                                                        |
+------------------------------------------------------------------------+
```

## 2. Tipos de Entornos Multi-Agente

### Cooperativo vs Competitivo vs Mixto

```
+------------------------------------------------------------------------+
|  TAXONOMIA DE ENTORNOS MULTI-AGENTE                                    |
+------------------------------------------------------------------------+
|                                                                        |
|  COOPERATIVO (Fully Cooperative):                                      |
|  --------------------------------                                      |
|  - Todos comparten el mismo objetivo                                   |
|  - Recompensa comun: r1 = r2 = ... = rn                                |
|  - Ejemplo: equipo de robots moviendo objeto pesado                    |
|                                                                        |
|      +---+    +---+                                                    |
|      | A |    | B |   Objetivo comun: mover caja                       |
|      +---+    +---+                                                    |
|        |________|                                                      |
|           [===]                                                        |
|                                                                        |
|                                                                        |
|  COMPETITIVO (Fully Competitive / Zero-Sum):                           |
|  -------------------------------------------                           |
|  - Intereses opuestos: lo que gana uno, pierde otro                    |
|  - Suma de recompensas = 0: r1 + r2 = 0                                |
|  - Ejemplo: ajedrez, Go, juegos de pelea                               |
|                                                                        |
|      +---+        +---+                                                |
|      | A | <----> | B |   A gana = B pierde                            |
|      +---+  vs    +---+                                                |
|                                                                        |
|                                                                        |
|  MIXTO (Mixed / General-Sum):                                          |
|  ----------------------------                                          |
|  - Combinacion de cooperacion y competicion                            |
|  - Alianzas temporales, traicion posible                               |
|  - Ejemplo: economia, diplomacia, juegos sociales                      |
|                                                                        |
|      +---+                                                             |
|      | A |---cooperar---+                                              |
|      +---+              |                                              |
|        |                v                                              |
|        +---competir---+---+                                            |
|                       | B |                                            |
|                       +---+                                            |
|                                                                        |
+------------------------------------------------------------------------+
```

### Observabilidad

```
+------------------------------------------------------------------------+
|  OBSERVABILIDAD EN MARL                                                |
+------------------------------------------------------------------------+
|                                                                        |
|  FULL OBSERVABILITY:                                                   |
|  -------------------                                                   |
|  Todos ven el estado completo del juego                                |
|                                                                        |
|  Ejemplo: Ajedrez (tablero visible para ambos)                         |
|                                                                        |
|      Agent A                    Agent B                                |
|         |                          |                                   |
|         v                          v                                   |
|      [Observa todo]          [Observa todo]                            |
|           \                      /                                     |
|            \                    /                                      |
|             v                  v                                       |
|           [Estado completo del juego]                                  |
|                                                                        |
|                                                                        |
|  PARTIAL OBSERVABILITY:                                                |
|  ----------------------                                                |
|  Cada agente ve solo parte del estado                                  |
|                                                                        |
|  Ejemplo: Poker (cartas ocultas)                                       |
|                                                                        |
|      Agent A                    Agent B                                |
|         |                          |                                   |
|         v                          v                                   |
|      [Ve o1]                   [Ve o2]                                 |
|      (sus cartas)              (sus cartas)                            |
|           \                      /                                     |
|            \                    /                                      |
|             v                  v                                       |
|           [Estado real s: todas las cartas]                            |
|                                                                        |
|                                                                        |
|  COMMUNICATION:                                                        |
|  --------------                                                        |
|  Agentes pueden comunicarse (canales de mensaje)                       |
|                                                                        |
|      Agent A  --mensaje-->  Agent B                                    |
|         |                      |                                       |
|         v                      v                                       |
|      [Decide]               [Decide basado en mensaje]                 |
|                                                                        |
+------------------------------------------------------------------------+
```

## 3. Algoritmos para MARL

### Independent Learners

```
+------------------------------------------------------------------------+
|  INDEPENDENT LEARNERS (IL)                                             |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: Cada agente aprende como si estuviera solo                      |
|        Ignora que otros agentes estan aprendiendo                      |
|                                                                        |
|      +-------+          +-------+          +-------+                   |
|      | DQN 1 |          | DQN 2 |          | DQN N |                   |
|      +---+---+          +---+---+          +---+---+                   |
|          |                  |                  |                       |
|          v                  v                  v                       |
|      +-------------------------------------------+                     |
|      |             Environment                   |                     |
|      +-------------------------------------------+                     |
|                                                                        |
|  Cada agente:                                                          |
|    - Tiene su propia red Q(s, a; theta_i)                              |
|    - Entrena independientemente                                        |
|    - Ve el estado (o su observacion)                                   |
|    - No modela a otros agentes                                         |
|                                                                        |
|                                                                        |
|  VENTAJAS:                                                             |
|  ---------                                                             |
|  + Simple de implementar                                               |
|  + Escala bien (cada agente independiente)                             |
|  + Puede usar cualquier algoritmo single-agent                         |
|                                                                        |
|  DESVENTAJAS:                                                          |
|  ------------                                                          |
|  - No converge teoricamente (no-estacionariedad)                       |
|  - En practica funciona sorprendentemente bien                         |
|  - No hay coordinacion explicita                                       |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion Independent DQN

```python
"""
Independent DQN para Multi-Agent RL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import random


class DQNetwork(nn.Module):
    """Q-Network para un agente."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class IndependentDQNAgent:
    """
    Agente DQN independiente para MARL.
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 32
    ):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_network = DQNetwork(obs_dim, action_dim)
        self.target_network = DQNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer individual
        self.buffer = deque(maxlen=buffer_size)

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return q_values.argmax().item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Guarda transicion en buffer."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def update(self) -> float:
        """Actualiza Q-network."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.FloatTensor(np.array(obs))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_obs = torch.FloatTensor(np.array(next_obs))
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Actualiza target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class MultiAgentEnvironment:
    """
    Entorno multi-agente simple (Grid World cooperativo).

    Dos agentes deben encontrarse en el centro del grid.
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.n_agents = 2
        self.obs_dim = 4  # (x, y) para cada agente
        self.action_dim = 5  # up, down, left, right, stay

        self.reset()

    def reset(self) -> List[np.ndarray]:
        """Reinicia posiciones."""
        # Agente 0 en esquina superior izquierda
        self.positions = [
            np.array([0, 0]),
            np.array([self.grid_size - 1, self.grid_size - 1])
        ]
        return self._get_observations()

    def _get_observations(self) -> List[np.ndarray]:
        """Cada agente ve su posicion y la del otro."""
        obs = []
        for i in range(self.n_agents):
            other = 1 - i
            obs.append(np.concatenate([
                self.positions[i] / self.grid_size,
                self.positions[other] / self.grid_size
            ]))
        return obs

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        """
        Ejecuta acciones de todos los agentes.

        Returns:
            observations, rewards, done, info
        """
        # Movimientos: 0=up, 1=down, 2=left, 3=right, 4=stay
        moves = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),   # down
            np.array([0, -1]),  # left
            np.array([0, 1]),   # right
            np.array([0, 0])    # stay
        ]

        # Aplicar movimientos
        for i, action in enumerate(actions):
            new_pos = self.positions[i] + moves[action]
            # Clip al grid
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.positions[i] = new_pos

        # Calcular recompensas
        distance = np.linalg.norm(self.positions[0] - self.positions[1])

        # Recompensa compartida basada en distancia
        if distance == 0:  # Se encontraron!
            rewards = [10.0, 10.0]
            done = True
        else:
            rewards = [-0.1 - distance * 0.1] * 2  # Penalizar distancia
            done = False

        info = {"distance": distance}
        return self._get_observations(), rewards, done, info


def train_independent_dqn(
    n_episodes: int = 1000,
    max_steps: int = 50
) -> Tuple[List[IndependentDQNAgent], List[float]]:
    """Entrena agentes independientes."""
    env = MultiAgentEnvironment()

    agents = [
        IndependentDQNAgent(
            agent_id=i,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim
        )
        for i in range(env.n_agents)
    ]

    rewards_history = []

    for episode in range(n_episodes):
        observations = env.reset()
        episode_rewards = [0.0] * env.n_agents

        for step in range(max_steps):
            # Seleccionar acciones
            actions = [
                agents[i].select_action(observations[i])
                for i in range(env.n_agents)
            ]

            # Ejecutar
            next_observations, rewards, done, info = env.step(actions)

            # Guardar transiciones
            for i in range(env.n_agents):
                agents[i].store_transition(
                    observations[i], actions[i], rewards[i],
                    next_observations[i], done
                )
                episode_rewards[i] += rewards[i]

            # Update agentes
            for agent in agents:
                agent.update()

            observations = next_observations

            if done:
                break

        # Update targets y epsilon
        for agent in agents:
            if episode % 10 == 0:
                agent.update_target()
            agent.decay_epsilon()

        total_reward = sum(episode_rewards)
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agents[0].epsilon:.3f}")

    return agents, rewards_history
```

### Centralized Training with Decentralized Execution (CTDE)

```
+------------------------------------------------------------------------+
|  CTDE: CENTRALIZED TRAINING, DECENTRALIZED EXECUTION                   |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: Durante entrenamiento, acceso a informacion global              |
|        Durante ejecucion, cada agente actua solo con su observacion    |
|                                                                        |
|                                                                        |
|  ENTRENAMIENTO (Centralizado):                                         |
|  -----------------------------                                         |
|                                                                        |
|      +-------+    +-------+    +-------+                               |
|      |Agent 1|    |Agent 2|    |Agent N|                               |
|      +---+---+    +---+---+    +---+---+                               |
|          |            |            |                                   |
|          v            v            v                                   |
|      +----------------------------------------+                        |
|      |      CENTRAL CRITIC / MIXER            |                        |
|      |  (ve estado global, todas acciones)    |                        |
|      +----------------------------------------+                        |
|                                                                        |
|  Tiene acceso a:                                                       |
|    - Estado global s                                                   |
|    - Todas las observaciones o1, o2, ..., on                           |
|    - Todas las acciones a1, a2, ..., an                                |
|                                                                        |
|                                                                        |
|  EJECUCION (Descentralizada):                                          |
|  ----------------------------                                          |
|                                                                        |
|      +-------+    +-------+    +-------+                               |
|      |Agent 1|    |Agent 2|    |Agent N|                               |
|      +---+---+    +---+---+    +---+---+                               |
|          |            |            |                                   |
|          v            v            v                                   |
|        a1           a2           an                                    |
|                                                                        |
|  Cada agente solo usa su observacion local o_i                         |
|  No necesita comunicacion                                              |
|                                                                        |
|                                                                        |
|  VENTAJAS:                                                             |
|  ---------                                                             |
|  + Coordinacion durante entrenamiento                                  |
|  + No requiere comunicacion en ejecucion                               |
|  + Mas estable que IL                                                  |
|                                                                        |
+------------------------------------------------------------------------+
```

### QMIX: Value Decomposition

```
+------------------------------------------------------------------------+
|  QMIX: FACTORIZACION MONOTONICA DE Q                                   |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA: Como coordinar sin comunicacion en ejecucion?               |
|                                                                        |
|  SOLUCION: Descomponer Q_total en Q individuales                       |
|                                                                        |
|                                                                        |
|  CONSTRAINT CLAVE (IGM - Individual-Global-Max):                       |
|  ------------------------------------------------                      |
|                                                                        |
|  argmax_a Q_total(s, a) = (argmax_a1 Q1(o1,a1), ..., argmax_an Qn(on,an))
|                                                                        |
|  "Maximizar Q total = Maximizar cada Q individual"                     |
|                                                                        |
|  Esto se logra si Q_total es MONOTONICO en cada Qi:                    |
|                                                                        |
|  dQ_total / dQi >= 0   para todo i                                     |
|                                                                        |
|                                                                        |
|  ARQUITECTURA QMIX:                                                    |
|  ------------------                                                    |
|                                                                        |
|      o1          o2          on                                        |
|       |           |           |                                        |
|       v           v           v                                        |
|    +-----+     +-----+     +-----+                                     |
|    | Q1  |     | Q2  |     | Qn  |    Agent networks                   |
|    +--+--+     +--+--+     +--+--+                                     |
|       |           |           |                                        |
|       v           v           v                                        |
|    Q1(o1,a1)  Q2(o2,a2)  Qn(on,an)                                     |
|       |           |           |                                        |
|       +-----------+-----------+                                        |
|                   |                                                    |
|                   v                                                    |
|           +---------------+                                            |
|           | Mixing Network|<--- Estado global s                        |
|           | (monotonic)   |     (condiciona los pesos)                 |
|           +-------+-------+                                            |
|                   |                                                    |
|                   v                                                    |
|              Q_total(s, a)                                             |
|                                                                        |
|                                                                        |
|  Mixing Network:                                                       |
|    - Pesos generados por hypernetwork condicionada en s                |
|    - Pesos forzados a ser NO NEGATIVOS (abs o exp)                     |
|    - Garantiza monotonicidad                                           |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion QMIX

```python
"""
QMIX para Multi-Agent RL cooperativo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class AgentNetwork(nn.Module):
    """Red Q para un agente individual (con RNN para POMDP)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.hidden_dim = hidden_dim

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            hidden: (batch, hidden_dim)

        Returns:
            q_values: (batch, action_dim)
            new_hidden: (batch, hidden_dim)
        """
        x = torch.relu(self.fc1(obs))
        h = self.rnn(x, hidden)
        q = self.fc2(h)
        return q, h

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)


class HyperNetwork(nn.Module):
    """Hypernetwork que genera pesos para el mixer."""

    def __init__(self, state_dim: int, n_agents: int, embed_dim: int = 32):
        super().__init__()

        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Genera pesos para primera capa del mixer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )

        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Genera pesos para segunda capa
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, state: torch.Tensor, agent_qs: torch.Tensor) -> torch.Tensor:
        """
        Mezcla Q values de agentes.

        Args:
            state: (batch, state_dim)
            agent_qs: (batch, n_agents)

        Returns:
            q_total: (batch, 1)
        """
        batch_size = state.shape[0]

        # Primera capa
        w1 = torch.abs(self.hyper_w1(state))  # No negativo!
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # (batch, 1, n_agents) @ (batch, n_agents, embed) -> (batch, 1, embed)
        hidden = torch.relu(
            torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        )

        # Segunda capa
        w2 = torch.abs(self.hyper_w2(state))  # No negativo!
        w2 = w2.view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # (batch, 1, embed) @ (batch, embed, 1) -> (batch, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.squeeze(-1).squeeze(-1)


class QMIX:
    """
    QMIX para juegos cooperativos.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        buffer_size: int = 5000
    ):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Agent networks (compartidos o separados)
        self.agents = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim)
            for _ in range(n_agents)
        ])

        # Mixer
        self.mixer = HyperNetwork(state_dim, n_agents)

        # Targets
        self.target_agents = nn.ModuleList([
            AgentNetwork(obs_dim, action_dim)
            for _ in range(n_agents)
        ])
        self.target_mixer = HyperNetwork(state_dim, n_agents)

        # Copy weights
        self._update_targets(tau=1.0)

        # Optimizer
        params = list(self.agents.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        # Replay buffer
        self.buffer: List = []
        self.buffer_size = buffer_size

        # Hidden states
        self.hidden_states = None

    def init_hidden(self, batch_size: int = 1):
        """Inicializa hidden states para RNN."""
        self.hidden_states = [
            agent.init_hidden(batch_size)
            for agent in self.agents
        ]

    def select_actions(
        self,
        observations: List[np.ndarray],
        epsilon: float = 0.0
    ) -> List[int]:
        """Selecciona acciones para todos los agentes."""
        actions = []

        for i, (agent, obs) in enumerate(zip(self.agents, observations)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                q_values, self.hidden_states[i] = agent(
                    obs_tensor, self.hidden_states[i]
                )

            if np.random.random() < epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = q_values.argmax().item()

            actions.append(action)

        return actions

    def store_episode(self, episode_data: dict):
        """Guarda episodio completo."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(episode_data)

    def update(self) -> float:
        """Actualiza redes."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample batch de episodios
        batch_indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = [self.buffer[i] for i in batch_indices]

        # Procesar batch (simplificado - asumiendo episodios de misma longitud)
        # En implementacion real, necesitas padding y masking

        total_loss = 0.0

        for episode in batch:
            obs_list = episode['observations']  # List of (T, n_agents, obs_dim)
            actions = episode['actions']  # (T, n_agents)
            rewards = episode['rewards']  # (T,)
            states = episode['states']  # (T, state_dim)
            next_states = episode['next_states']
            dones = episode['dones']  # (T,)

            T = len(rewards)

            # Reset hidden states
            hidden = [agent.init_hidden(1) for agent in self.agents]
            target_hidden = [agent.init_hidden(1) for agent in self.target_agents]

            episode_loss = 0.0

            for t in range(T):
                # Current Q values
                agent_qs = []
                for i in range(self.n_agents):
                    obs = torch.FloatTensor(obs_list[t][i]).unsqueeze(0)
                    q, hidden[i] = self.agents[i](obs, hidden[i])
                    agent_qs.append(q[0, actions[t][i]])

                agent_qs = torch.stack(agent_qs)
                state = torch.FloatTensor(states[t]).unsqueeze(0)
                q_total = self.mixer(state, agent_qs.unsqueeze(0))

                # Target Q values
                with torch.no_grad():
                    target_agent_qs = []
                    for i in range(self.n_agents):
                        next_obs = torch.FloatTensor(obs_list[min(t+1, T-1)][i]).unsqueeze(0)
                        q, target_hidden[i] = self.target_agents[i](
                            next_obs, target_hidden[i]
                        )
                        target_agent_qs.append(q.max())

                    target_agent_qs = torch.stack(target_agent_qs)
                    next_state = torch.FloatTensor(next_states[t]).unsqueeze(0)
                    target_q_total = self.target_mixer(
                        next_state, target_agent_qs.unsqueeze(0)
                    )

                    target = rewards[t] + self.gamma * target_q_total * (1 - dones[t])

                episode_loss += (q_total - target) ** 2

            total_loss += episode_loss / T

        loss = total_loss / self.batch_size

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agents.parameters()) + list(self.mixer.parameters()),
            10.0
        )
        self.optimizer.step()

        return loss.item()

    def _update_targets(self, tau: float = 0.01):
        """Soft update de targets."""
        for agent, target in zip(self.agents, self.target_agents):
            for p, tp in zip(agent.parameters(), target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        for p, tp in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
```

## 4. Self-Play

### El Concepto

```
+------------------------------------------------------------------------+
|  SELF-PLAY: APRENDER JUGANDO CONTRA SI MISMO                           |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: En juegos competitivos, entrenar agente vs copias de si mismo   |
|                                                                        |
|      Version actual                                                    |
|      del agente                                                        |
|           |                                                            |
|           v                                                            |
|      +-------+          +-------+                                      |
|      |Agent v |   vs    |Agent v |  (o version anterior)               |
|      +-------+          +-------+                                      |
|           |                  |                                         |
|           v                  v                                         |
|      +---------------------------+                                     |
|      |         Game              |                                     |
|      +---------------------------+                                     |
|                   |                                                    |
|                   v                                                    |
|            Winner/Loser                                                |
|                   |                                                    |
|                   v                                                    |
|            Update Agent                                                |
|                                                                        |
|                                                                        |
|  VARIANTES:                                                            |
|  ----------                                                            |
|                                                                        |
|  1. NAIVE SELF-PLAY                                                    |
|     Siempre juega contra ultima version                                |
|     Problema: puede olvidar estrategias antiguas                       |
|                                                                        |
|  2. FICTITIOUS SELF-PLAY                                               |
|     Juega contra estrategia promedio historica                         |
|     Converge a Nash en algunos juegos                                  |
|                                                                        |
|  3. POPULATION-BASED SELF-PLAY                                         |
|     Mantiene poblacion de agentes                                      |
|     Selecciona oponentes de la poblacion                               |
|     Mas robusto, evita cycling                                         |
|                                                                        |
|  4. LEAGUE TRAINING (AlphaStar)                                        |
|     Poblacion con diferentes roles:                                    |
|     - Main agents: los que queremos mejorar                            |
|     - League exploiters: buscan debilidades                            |
|     - Main exploiters: explotan main agents                            |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion Self-Play

```python
"""
Self-Play para juegos competitivos.
"""
import torch
import numpy as np
from typing import List, Tuple
from collections import deque
import copy


class TicTacToeEnv:
    """
    Tic-Tac-Toe como ejemplo de juego competitivo.
    """

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 o -1

    def reset(self) -> np.ndarray:
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Estado desde perspectiva del jugador actual."""
        return (self.board * self.current_player).flatten().astype(np.float32)

    def get_valid_actions(self) -> List[int]:
        """Retorna posiciones vacias."""
        return list(np.where(self.board.flatten() == 0)[0])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta movimiento.

        Returns:
            state, reward (para jugador actual), done, info
        """
        row, col = action // 3, action % 3

        if self.board[row, col] != 0:
            # Movimiento invalido
            return self.get_state(), -10.0, True, {"winner": -self.current_player}

        self.board[row, col] = self.current_player

        # Check ganador
        winner = self._check_winner()

        if winner != 0:
            reward = 1.0 if winner == self.current_player else -1.0
            return self.get_state(), reward, True, {"winner": winner}

        # Check empate
        if len(self.get_valid_actions()) == 0:
            return self.get_state(), 0.0, True, {"winner": 0}

        # Cambiar jugador
        self.current_player *= -1

        return self.get_state(), 0.0, False, {"winner": None}

    def _check_winner(self) -> int:
        """Retorna 1, -1 o 0 (no ganador aun)."""
        # Filas
        for i in range(3):
            if abs(self.board[i].sum()) == 3:
                return int(np.sign(self.board[i].sum()))

        # Columnas
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return int(np.sign(self.board[:, j].sum()))

        # Diagonales
        if abs(self.board.trace()) == 3:
            return int(np.sign(self.board.trace()))
        if abs(np.fliplr(self.board).trace()) == 3:
            return int(np.sign(np.fliplr(self.board).trace()))

        return 0


class SelfPlayAgent:
    """
    Agente con self-play y population.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        population_size: int = 5
    ):
        self.action_dim = action_dim
        self.population_size = population_size

        # Red principal
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Poblacion de oponentes (checkpoints historicos)
        self.population: deque = deque(maxlen=population_size)
        self._save_to_population()

    def _save_to_population(self):
        """Guarda copia actual en poblacion."""
        checkpoint = copy.deepcopy(self.network.state_dict())
        self.population.append(checkpoint)

    def get_opponent(self) -> nn.Module:
        """Selecciona oponente de la poblacion."""
        opponent = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

        # Seleccionar checkpoint aleatorio (con sesgo hacia recientes)
        weights = np.array([1.0 * (i + 1) for i in range(len(self.population))])
        weights /= weights.sum()
        idx = np.random.choice(len(self.population), p=weights)

        opponent.load_state_dict(self.population[idx])
        return opponent

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: List[int],
        epsilon: float = 0.0
    ) -> int:
        """Selecciona accion."""
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)[0]

        # Mascara acciones invalidas
        mask = torch.full((self.action_dim,), float('-inf'))
        mask[valid_actions] = 0
        q_values = q_values + mask

        return q_values.argmax().item()

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float]
    ) -> float:
        """
        Update con REINFORCE simplificado.
        """
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Calcular returns (desde el final)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # Normalizar
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward
        logits = self.network(states)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Loss
        loss = -(selected_log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_self_play(n_episodes: int = 10000) -> SelfPlayAgent:
    """Entrena agente con self-play."""
    env = TicTacToeEnv()
    agent = SelfPlayAgent(state_dim=9, action_dim=9)

    win_rates = []
    recent_results = deque(maxlen=100)

    for episode in range(n_episodes):
        # Obtener oponente
        opponent = agent.get_opponent()

        state = env.reset()
        done = False

        # Determinar quien empieza
        agent_player = 1 if np.random.random() > 0.5 else -1

        agent_states, agent_actions, agent_rewards = [], [], []
        epsilon = max(0.1, 1.0 - episode / 5000)

        while not done:
            valid_actions = env.get_valid_actions()

            if env.current_player == agent_player:
                # Turno del agente
                action = agent.select_action(state, valid_actions, epsilon)
                agent_states.append(state.copy())
                agent_actions.append(action)
            else:
                # Turno del oponente
                with torch.no_grad():
                    opp_state = torch.FloatTensor(state * -1).unsqueeze(0)  # Desde su perspectiva
                    q_values = opponent(opp_state)[0]

                    mask = torch.full((9,), float('-inf'))
                    mask[valid_actions] = 0
                    q_values = q_values + mask

                    action = q_values.argmax().item()

            state, reward, done, info = env.step(action)

            if done and env.current_player == agent_player:
                # Recompensa final para el agente
                agent_rewards.append(reward)

        # Rellenar recompensas intermedias
        if len(agent_rewards) < len(agent_states):
            final_reward = reward if info["winner"] == agent_player else -reward
            agent_rewards = [0.0] * (len(agent_states) - 1) + [final_reward]

        # Update
        if agent_states:
            agent.update(agent_states, agent_actions, agent_rewards)

        # Track resultados
        if info["winner"] == agent_player:
            recent_results.append(1)
        elif info["winner"] == -agent_player:
            recent_results.append(-1)
        else:
            recent_results.append(0)

        # Guardar en poblacion periodicamente
        if (episode + 1) % 500 == 0:
            agent._save_to_population()

        # Log
        if (episode + 1) % 1000 == 0:
            wins = sum(1 for r in recent_results if r == 1)
            losses = sum(1 for r in recent_results if r == -1)
            draws = sum(1 for r in recent_results if r == 0)

            print(f"Episode {episode + 1} | "
                  f"Win: {wins}% | Loss: {losses}% | Draw: {draws}% | "
                  f"Epsilon: {epsilon:.2f}")

    return agent
```

## 5. Nash Equilibrium en Juegos

### Conceptos de Teoria de Juegos

```
+------------------------------------------------------------------------+
|  NASH EQUILIBRIUM                                                      |
+------------------------------------------------------------------------+
|                                                                        |
|  DEFINICION:                                                           |
|  -----------                                                           |
|  Un Nash Equilibrium es un perfil de estrategias donde NINGUN          |
|  jugador puede mejorar unilateralmente cambiando su estrategia.        |
|                                                                        |
|                                                                        |
|  EJEMPLO CLASICO - Dilema del Prisionero:                              |
|  ----------------------------------------                              |
|                                                                        |
|                      Jugador B                                         |
|                    Cooperar  Traicionar                                |
|                   +----------+-----------+                             |
|  Jugador A        |          |           |                             |
|  Cooperar         | (-1,-1)  |  (-3, 0)  |                             |
|                   +----------+-----------+                             |
|  Traicionar       |  (0,-3)  |  (-2,-2)  |                             |
|                   +----------+-----------+                             |
|                                                                        |
|  Nash Equilibrium: (Traicionar, Traicionar) -> (-2, -2)                |
|                                                                        |
|  Ninguno puede mejorar cambiando solo su estrategia:                   |
|    - Si A cambia a Cooperar: A obtiene -3 (peor)                       |
|    - Si B cambia a Cooperar: B obtiene -3 (peor)                       |
|                                                                        |
|                                                                        |
|  EN RL:                                                                |
|  ------                                                                |
|  - Single-agent: buscamos politica optima                              |
|  - Multi-agent: buscamos Nash Equilibrium (o aproximacion)             |
|                                                                        |
|  NASH EN JUEGOS DE SUMA CERO:                                          |
|  ----------------------------                                          |
|  minimax = maximin                                                     |
|                                                                        |
|  max_pi1 min_pi2 V(pi1, pi2) = min_pi2 max_pi1 V(pi1, pi2)             |
|                                                                        |
+------------------------------------------------------------------------+
```

### Convergencia a Nash

```
+------------------------------------------------------------------------+
|  CONVERGENCIA A NASH EN MARL                                           |
+------------------------------------------------------------------------+
|                                                                        |
|  PROBLEMA: En general, MARL no garantiza convergencia a Nash           |
|                                                                        |
|  RESULTADOS CONOCIDOS:                                                 |
|  ---------------------                                                 |
|                                                                        |
|  1. JUEGOS DE SUMA CERO, 2 JUGADORES                                   |
|     - Self-play converge a Nash (minimax)                              |
|     - Fictitious play converge                                         |
|     - CFR (Counterfactual Regret Minimization) converge                |
|                                                                        |
|  2. JUEGOS COOPERATIVOS                                                |
|     - Optimo global alcanzable                                         |
|     - QMIX, COMA, etc. funcionan bien                                  |
|                                                                        |
|  3. JUEGOS GENERALES (N jugadores, suma general)                       |
|     - No hay garantias teoricas fuertes                                |
|     - Ciclos posibles                                                  |
|     - En practica: poblacion diversa ayuda                             |
|                                                                        |
|                                                                        |
|  ESTRATEGIAS PARA APROXIMAR NASH:                                      |
|  ---------------------------------                                     |
|                                                                        |
|  1. Fictitious Play                                                    |
|     - Jugar mejor respuesta a estrategia promedio historica            |
|                                                                        |
|  2. Regret Minimization (CFR)                                          |
|     - Minimizar "arrepentimiento" de no haber jugado otras acciones    |
|                                                                        |
|  3. Policy Space Response Oracles (PSRO)                               |
|     - Construir poblacion de mejores respuestas                        |
|     - Meta-estrategia sobre la poblacion                               |
|                                                                        |
+------------------------------------------------------------------------+
```

## 6. Comunicacion entre Agentes

### Emergent Communication

```
+------------------------------------------------------------------------+
|  COMUNICACION EMERGENTE                                                |
+------------------------------------------------------------------------+
|                                                                        |
|  IDEA: Agentes desarrollan "lenguaje" propio para coordinarse          |
|                                                                        |
|                                                                        |
|  ARQUITECTURA CON COMUNICACION:                                        |
|  ------------------------------                                        |
|                                                                        |
|      Agent 1                              Agent 2                      |
|      +-----+                              +-----+                      |
|      |     |------- mensaje m1 --------->|     |                       |
|      |     |<------ mensaje m2 ----------|     |                       |
|      +-----+                              +-----+                      |
|         |                                    |                         |
|         v                                    v                         |
|        a1                                   a2                         |
|         |                                    |                         |
|         +-------------+  +-------------------+                         |
|                       |  |                                             |
|                       v  v                                             |
|                  Environment                                           |
|                                                                        |
|                                                                        |
|  TIPOS DE COMUNICACION:                                                |
|  ----------------------                                                |
|                                                                        |
|  1. DISCRETE MESSAGES                                                  |
|     - Vocabulario finito: {m1, m2, ..., mk}                            |
|     - Gumbel-Softmax para gradientes                                   |
|                                                                        |
|  2. CONTINUOUS MESSAGES                                                |
|     - Vector de numeros reales                                         |
|     - Mas flexible pero menos interpretable                            |
|                                                                        |
|  3. BROADCAST vs TARGETED                                              |
|     - Broadcast: todos reciben mismo mensaje                           |
|     - Targeted: mensajes especificos para cada agente                  |
|                                                                        |
+------------------------------------------------------------------------+
```

### Implementacion con Comunicacion

```python
"""
Multi-Agent RL con comunicacion (CommNet style).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class CommAgent(nn.Module):
    """
    Agente con capacidad de comunicacion.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 16,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.message_dim = message_dim

        # Encoder de observacion
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # Generador de mensaje
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

        # Procesador de mensajes recibidos
        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (combina obs encoding + mensajes procesados)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Critic
        self.critic = nn.Linear(hidden_dim * 2, 1)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(obs)

    def generate_message(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        return self.message_generator(encoded_obs)

    def forward(
        self,
        obs: torch.Tensor,
        received_messages: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            received_messages: (batch, n_other_agents, message_dim)

        Returns:
            action_logits, value, message_to_send
        """
        # Encode observacion
        encoded = self.encode_obs(obs)

        # Generar mensaje
        message = self.generate_message(encoded)

        # Procesar mensajes recibidos (mean pooling)
        if received_messages.shape[1] > 0:
            processed_messages = self.message_processor(
                received_messages.mean(dim=1)
            )
        else:
            processed_messages = torch.zeros_like(encoded)

        # Combinar
        combined = torch.cat([encoded, processed_messages], dim=-1)

        # Actor y Critic
        action_logits = self.actor(combined)
        value = self.critic(combined)

        return action_logits, value, message


class CommMARLSystem:
    """
    Sistema MARL con comunicacion.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        message_dim: int = 16,
        lr: float = 1e-3
    ):
        self.n_agents = n_agents
        self.message_dim = message_dim

        # Crear agentes
        self.agents = nn.ModuleList([
            CommAgent(obs_dim, action_dim, message_dim)
            for _ in range(n_agents)
        ])

        self.optimizer = optim.Adam(self.agents.parameters(), lr=lr)

    def communicate_and_act(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Ronda de comunicacion y seleccion de acciones.
        """
        obs_tensors = [
            torch.FloatTensor(obs).unsqueeze(0)
            for obs in observations
        ]

        # Fase 1: Generar mensajes
        messages = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_tensors)):
            encoded = agent.encode_obs(obs)
            message = agent.generate_message(encoded)
            messages.append(message)

        # Fase 2: Distribuir mensajes y actuar
        actions = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_tensors)):
            # Mensajes de otros agentes
            other_messages = torch.cat([
                messages[j] for j in range(self.n_agents) if j != i
            ], dim=0).unsqueeze(0)  # (1, n-1, message_dim)

            # Forward
            action_logits, _, _ = agent(obs, other_messages)

            if deterministic:
                action = action_logits.argmax().item()
            else:
                probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

            actions.append(action)

        message_arrays = [m.detach().numpy()[0] for m in messages]

        return actions, message_arrays


# Ejemplo de uso
class CooperativeNavigationEnv:
    """
    Entorno donde agentes deben llegar a objetivos coordinandose.
    """

    def __init__(self, n_agents: int = 3, grid_size: int = 10):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.obs_dim = 4 * n_agents  # Posicion de todos + objetivo
        self.action_dim = 5  # up, down, left, right, stay

        self.reset()

    def reset(self) -> List[np.ndarray]:
        # Posiciones aleatorias
        self.positions = [
            np.random.randint(0, self.grid_size, 2).astype(float)
            for _ in range(self.n_agents)
        ]
        self.targets = [
            np.random.randint(0, self.grid_size, 2).astype(float)
            for _ in range(self.n_agents)
        ]

        return self._get_observations()

    def _get_observations(self) -> List[np.ndarray]:
        obs = []
        for i in range(self.n_agents):
            # Observacion: mi posicion, mi objetivo, posiciones de otros
            agent_obs = [
                self.positions[i] / self.grid_size,
                self.targets[i] / self.grid_size
            ]
            for j in range(self.n_agents):
                if j != i:
                    agent_obs.append(self.positions[j] / self.grid_size)

            obs.append(np.concatenate(agent_obs))

        return obs

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        moves = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),   # down
            np.array([0, -1]),  # left
            np.array([0, 1]),   # right
            np.array([0, 0])    # stay
        ]

        # Aplicar movimientos
        for i, action in enumerate(actions):
            new_pos = self.positions[i] + moves[action]
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.positions[i] = new_pos

        # Calcular recompensas
        rewards = []
        for i in range(self.n_agents):
            dist = np.linalg.norm(self.positions[i] - self.targets[i])
            rewards.append(-dist * 0.1)

        # Penalizar colisiones
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.positions[i], self.positions[j]):
                    rewards[i] -= 1.0
                    rewards[j] -= 1.0

        # Check si todos llegaron
        all_reached = all(
            np.linalg.norm(self.positions[i] - self.targets[i]) < 0.5
            for i in range(self.n_agents)
        )

        if all_reached:
            rewards = [r + 10.0 for r in rewards]

        done = all_reached

        return self._get_observations(), rewards, done, {}
```

## 7. Aplicaciones en Ciberseguridad

```
+------------------------------------------------------------------------+
|  MARL EN CIBERSEGURIDAD                                                |
+------------------------------------------------------------------------+
|                                                                        |
|  1. RED TEAM vs BLUE TEAM                                              |
|     ----------------------                                             |
|     Juego competitivo suma-cero                                        |
|                                                                        |
|     Red Team (Atacante):         Blue Team (Defensor):                 |
|     - Explorar red               - Detectar intrusiones                |
|     - Encontrar vulns            - Parchear sistemas                   |
|     - Exfiltrar datos            - Responder incidentes                |
|                                                                        |
|     Self-play puede entrenar ambos simultaneamente!                    |
|                                                                        |
|                                                                        |
|  2. DEFENSA COORDINADA DE RED                                          |
|     --------------------------                                         |
|     Juego cooperativo                                                  |
|                                                                        |
|     Multiples IDS/Firewalls coordinados:                               |
|     - Cada uno ve trafico local                                        |
|     - Comparten alertas (comunicacion)                                 |
|     - Objetivo comun: minimizar ataques exitosos                       |
|                                                                        |
|                                                                        |
|  3. HONEYPOT COORDINATION                                              |
|     ----------------------                                             |
|     Multiples honeypots cooperando                                     |
|                                                                        |
|     - Diferentes servicios                                             |
|     - Atraer atacantes coordinadamente                                 |
|     - Recolectar inteligencia                                          |
|                                                                        |
|                                                                        |
|  4. ADVERSARIAL ML                                                     |
|     ---------------                                                    |
|     Atacante genera adversarial examples                               |
|     Defensor entrena modelos robustos                                  |
|                                                                        |
+------------------------------------------------------------------------+
```

### Ejemplo: Red vs Blue Team Simulation

```python
"""
Simulacion Red Team vs Blue Team como juego MARL.
"""
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class RedAction(Enum):
    SCAN = 0
    EXPLOIT = 1
    LATERAL = 2
    EXFIL = 3
    HIDE = 4


class BlueAction(Enum):
    MONITOR = 0
    PATCH = 1
    ISOLATE = 2
    HONEYPOT = 3
    RESET = 4


@dataclass
class NetworkState:
    """Estado de la red."""
    compromised_hosts: set
    detected_alerts: int
    exfiltrated_data: float
    patches_applied: set
    honeypots_active: set


class RedVsBlueEnv:
    """
    Entorno competitivo Red Team vs Blue Team.
    """

    def __init__(self, n_hosts: int = 10):
        self.n_hosts = n_hosts
        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reinicia juego."""
        self.state = NetworkState(
            compromised_hosts=set(),
            detected_alerts=0,
            exfiltrated_data=0.0,
            patches_applied=set(),
            honeypots_active=set()
        )

        self.step_count = 0
        self.max_steps = 100

        # Vulnerabilidades por host (algunas son honeypots)
        self.vulnerabilities = {
            i: np.random.random() > 0.5
            for i in range(self.n_hosts)
        }

        return self._get_red_obs(), self._get_blue_obs()

    def _get_red_obs(self) -> np.ndarray:
        """Observacion del Red Team (parcial)."""
        obs = np.zeros(self.n_hosts * 2 + 2)

        # Hosts comprometidos (conocidos)
        for i in self.state.compromised_hosts:
            obs[i] = 1.0

        # Hosts escaneados con vulns (parcialmente conocido)
        for i, has_vuln in self.vulnerabilities.items():
            if has_vuln and i not in self.state.patches_applied:
                obs[self.n_hosts + i] = 0.5  # Posible vuln

        obs[-2] = self.state.exfiltrated_data / 10.0
        obs[-1] = self.step_count / self.max_steps

        return obs

    def _get_blue_obs(self) -> np.ndarray:
        """Observacion del Blue Team."""
        obs = np.zeros(self.n_hosts * 3 + 2)

        # Alertas detectadas
        obs[0] = min(self.state.detected_alerts / 10.0, 1.0)

        # Estado de hosts
        for i in range(self.n_hosts):
            # Patches aplicados
            obs[self.n_hosts + i] = 1.0 if i in self.state.patches_applied else 0.0
            # Honeypots activos
            obs[2 * self.n_hosts + i] = 1.0 if i in self.state.honeypots_active else 0.0

        obs[-1] = self.step_count / self.max_steps

        return obs

    def step(
        self,
        red_action: int,
        blue_action: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, Dict]:
        """
        Ejecuta acciones de ambos equipos.

        Returns:
            (observations, rewards, done, info)
        """
        self.step_count += 1

        red_action = RedAction(red_action)
        blue_action = BlueAction(blue_action)

        red_reward = 0.0
        blue_reward = 0.0
        detected = False

        # Blue Team actua primero (defensor)
        if blue_action == BlueAction.PATCH:
            # Parchear host aleatorio vulnerable
            for i in range(self.n_hosts):
                if i not in self.state.patches_applied and self.vulnerabilities.get(i, False):
                    self.state.patches_applied.add(i)
                    blue_reward += 0.5
                    break

        elif blue_action == BlueAction.HONEYPOT:
            # Activar honeypot
            available = set(range(self.n_hosts)) - self.state.honeypots_active
            if available:
                hp = np.random.choice(list(available))
                self.state.honeypots_active.add(hp)

        elif blue_action == BlueAction.ISOLATE:
            # Aislar host comprometido conocido
            if self.state.compromised_hosts and self.state.detected_alerts > 0:
                isolated = np.random.choice(list(self.state.compromised_hosts))
                self.state.compromised_hosts.remove(isolated)
                blue_reward += 2.0

        elif blue_action == BlueAction.RESET:
            # Reset de red (costoso pero efectivo)
            self.state.compromised_hosts.clear()
            blue_reward -= 5.0  # Coste de downtime

        # Red Team actua
        if red_action == RedAction.SCAN:
            # Probabilidad de deteccion baja
            if np.random.random() < 0.1:
                detected = True
            red_reward -= 0.1

        elif red_action == RedAction.EXPLOIT:
            # Intentar explotar host vulnerable
            targets = [
                i for i in range(self.n_hosts)
                if i not in self.state.compromised_hosts
                and i not in self.state.patches_applied
                and self.vulnerabilities.get(i, False)
            ]

            if targets:
                target = np.random.choice(targets)

                # Check si es honeypot
                if target in self.state.honeypots_active:
                    detected = True
                    self.state.detected_alerts += 3
                    blue_reward += 5.0  # Capturado en honeypot!
                else:
                    self.state.compromised_hosts.add(target)
                    red_reward += 3.0

            if np.random.random() < 0.3:
                detected = True

        elif red_action == RedAction.LATERAL:
            # Movimiento lateral desde host comprometido
            if self.state.compromised_hosts:
                source = np.random.choice(list(self.state.compromised_hosts))
                neighbors = [
                    i for i in range(self.n_hosts)
                    if i not in self.state.compromised_hosts
                ]
                if neighbors and np.random.random() < 0.4:
                    target = np.random.choice(neighbors)
                    self.state.compromised_hosts.add(target)
                    red_reward += 2.0

            if np.random.random() < 0.2:
                detected = True

        elif red_action == RedAction.EXFIL:
            # Exfiltrar datos
            if self.state.compromised_hosts:
                amount = len(self.state.compromised_hosts) * 0.5
                self.state.exfiltrated_data += amount
                red_reward += amount * 2

                if np.random.random() < 0.5:
                    detected = True

        elif red_action == RedAction.HIDE:
            # Esconderse (reduce deteccion futura)
            red_reward -= 0.5  # Tiempo perdido

        # Procesar deteccion
        if detected:
            self.state.detected_alerts += 1
            blue_reward += 1.0
            red_reward -= 1.0

        # Recompensas basadas en estado final
        blue_reward -= len(self.state.compromised_hosts) * 0.1
        blue_reward -= self.state.exfiltrated_data * 0.2

        red_reward += len(self.state.compromised_hosts) * 0.1
        red_reward += self.state.exfiltrated_data * 0.1

        # Check fin
        done = (
            self.step_count >= self.max_steps or
            self.state.exfiltrated_data >= 10.0 or  # Red wins
            len(self.state.compromised_hosts) == 0 and self.step_count > 20  # Blue wins
        )

        info = {
            "compromised": len(self.state.compromised_hosts),
            "exfiltrated": self.state.exfiltrated_data,
            "alerts": self.state.detected_alerts
        }

        return (self._get_red_obs(), self._get_blue_obs()), (red_reward, blue_reward), done, info

    @property
    def red_obs_dim(self) -> int:
        return self.n_hosts * 2 + 2

    @property
    def blue_obs_dim(self) -> int:
        return self.n_hosts * 3 + 2

    @property
    def red_action_dim(self) -> int:
        return len(RedAction)

    @property
    def blue_action_dim(self) -> int:
        return len(BlueAction)
```

## 8. Resumen

```
+------------------------------------------------------------------------+
|  MULTI-AGENT RL - RESUMEN                                              |
+------------------------------------------------------------------------+
|                                                                        |
|  TIPOS DE ENTORNOS:                                                    |
|    - Cooperativo: todos comparten objetivo                             |
|    - Competitivo: suma cero, intereses opuestos                        |
|    - Mixto: cooperacion y competicion                                  |
|                                                                        |
|  DESAFIOS:                                                             |
|    - No-estacionariedad (otros aprenden)                               |
|    - Credit assignment (quien contribuyo?)                             |
|    - Explosion combinatoria                                            |
|    - Equilibrios multiples                                             |
|                                                                        |
|  ALGORITMOS:                                                           |
|    - Independent Learners: simple, sorprendentemente efectivo          |
|    - CTDE: entrenamiento centralizado, ejecucion descentralizada       |
|    - QMIX: descomposicion monotonica de Q                              |
|    - MADDPG: actor-critic multi-agente                                 |
|                                                                        |
|  SELF-PLAY:                                                            |
|    - Entrenar contra copias de si mismo                                |
|    - Population-based para robustez                                    |
|    - Converge a Nash en juegos 2-player zero-sum                       |
|                                                                        |
|  COMUNICACION:                                                         |
|    - Mensajes discretos o continuos                                    |
|    - Emergent communication                                            |
|    - Mejora coordinacion                                               |
|                                                                        |
|  APLICACIONES EN SEGURIDAD:                                            |
|    - Red vs Blue team simulation                                       |
|    - Defensa coordinada de red                                         |
|    - Honeypot coordination                                             |
|    - Adversarial ML                                                    |
|                                                                        |
+------------------------------------------------------------------------+
```

---

**Siguiente:** Aplicaciones de RL en Ciberseguridad
